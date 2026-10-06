"""CPU tests: the register/PC/memory API, stepping, traps, reset and tracing."""

import io
import unittest

from tests.helpers import SCRATCH, CPU, EmulatorTestCase, assemble, encode, make_cpu, run_asm

from riscv.exceptions import (
    Breakpoint,
    EmulatorError,
    EnvironmentCall,
    IllegalInstruction,
    InstructionAccessFault,
    InstructionAddressMisaligned,
    LoadAccessFault,
    LoadAddressMisaligned,
    RiscvException,
    StoreAccessFault,
    StoreAddressMisaligned,
)
from riscv.host import HostEnvironment

NOP = encode("ADDI", rd=0, rs1=0, imm=0)


class TestRegisterAPI(EmulatorTestCase):
    def setUp(self):
        self.cpu = CPU()

    def test_register_roundtrip(self):
        """cpu: get_reg/set_reg round-trip on every register"""
        for index in range(1, 32):
            self.cpu.set_reg(index, 0xDEAD0000 + index)
        for index in range(1, 32):
            self.assertReg(self.cpu, index, 0xDEAD0000 + index)

    def test_x0_is_immutable(self):
        """cpu: x0 reads zero and ignores writes"""
        self.assertReg(self.cpu, 0, 0)
        self.cpu.set_reg(0, 0xFFFFFFFF)
        self.assertReg(self.cpu, 0, 0)

    def test_x0_immutable_through_instructions(self):
        """x0 immutability: instructions cannot write x0 either"""
        cpu = run_asm("li x1, 5\naddi x0, x1, 10\nadd x0, x1, x1\nlui x0, 0x12345")
        self.assertReg(cpu, 0, 0)

    def test_writes_wrap_to_32_bits(self):
        """cpu: register writes wrap to 32 bits"""
        self.cpu.set_reg(1, 0x1_0000_0001)
        self.assertReg(self.cpu, 1, 1)
        self.cpu.set_reg(2, -1)
        self.assertReg(self.cpu, 2, 0xFFFFFFFF)

    def test_signed_accessor(self):
        """cpu: get_reg_signed returns a two's-complement value"""
        self.cpu.set_reg(1, 0xFFFFFFFF)
        self.assertEqual(self.cpu.get_reg_signed(1), -1)

    def test_named_access(self):
        """cpu: registers are reachable by ABI name"""
        self.cpu.write_register("sp", 0x8000)
        self.assertEqual(self.cpu.read_register("sp"), 0x8000)
        self.assertReg(self.cpu, 2, 0x8000)

    def test_bad_register_index_raises(self):
        """cpu: an out-of-range register index raises"""
        with self.assertRaises(EmulatorError):
            self.cpu.set_reg(32, 0)
        with self.assertRaises(EmulatorError):
            self.cpu.get_reg(-1)


class TestProgramCounter(EmulatorTestCase):
    def test_pc_advances_by_four(self):
        """cpu: the PC advances by 4 per instruction"""
        cpu = make_cpu([NOP, NOP, NOP])
        self.assertPC(cpu, 0)
        cpu.step()
        self.assertPC(cpu, 4)
        cpu.step()
        self.assertPC(cpu, 8)

    def test_set_pc(self):
        """cpu: set_pc redirects execution"""
        cpu = make_cpu([encode("ADDI", rd=1, rs1=0, imm=1), encode("ADDI", rd=2, rs1=0, imm=2)])
        cpu.set_pc(4)
        cpu.step()
        self.assertReg(cpu, 1, 0)
        self.assertReg(cpu, 2, 2)

    def test_pc_wraps_at_the_top_of_the_address_space(self):
        """cpu: PC arithmetic wraps modulo 2^32"""
        cpu = CPU(memory_size=1 << 32)
        cpu.load_program([encode("JAL", rd=0, imm=-4)], addr=0xFFFFFFFC)
        cpu.set_pc(0xFFFFFFFC)
        cpu.step()
        self.assertPC(cpu, 0xFFFFFFF8)

    def test_instret_counts_retired_instructions(self):
        """cpu: instret counts only retired instructions"""
        cpu = make_cpu([NOP, NOP, 0x00000000])
        cpu.step()
        cpu.step()
        self.assertEqual(cpu.instret, 2)
        with self.assertRaises(IllegalInstruction):
            cpu.step()
        self.assertEqual(cpu.instret, 2, "a trapping instruction does not retire")


class TestStepping(EmulatorTestCase):
    def setUp(self):
        self.program = assemble("li x1, 1\nli x2, 2\nli x3, 3\nli x4, 4")
        self.cpu = CPU()
        self.cpu.load_program(self.program.image)

    def test_step_executes_exactly_one_instruction(self):
        """cpu: step() executes exactly one instruction"""
        self.cpu.step()
        self.assertReg(self.cpu, 1, 1)
        self.assertReg(self.cpu, 2, 0)
        self.assertEqual(self.cpu.instret, 1)

    def test_step_returns_the_decoded_instruction(self):
        """cpu: step() returns the instruction it executed"""
        inst = self.cpu.step()
        self.assertEqual(inst.name, "ADDI")
        self.assertEqual(inst.rd, 1)

    def test_run_n_instructions(self):
        """cpu: run(n) executes exactly n instructions"""
        executed = self.cpu.run(2)
        self.assertEqual(executed, 2)
        self.assertReg(self.cpu, 2, 2)
        self.assertReg(self.cpu, 3, 0)
        self.assertPC(self.cpu, 8)

    def test_run_zero(self):
        """cpu: run(0) does nothing"""
        self.assertEqual(self.cpu.run(0), 0)
        self.assertEqual(self.cpu.instret, 0)

    def test_run_until(self):
        """cpu: run_until stops at a target address"""
        executed = self.cpu.run_until(12)
        self.assertPC(self.cpu, 12)
        self.assertEqual(executed, 3)

    def test_breakpoints(self):
        """cpu: run() stops at a breakpoint without executing it"""
        self.cpu.breakpoints.add(8)
        self.cpu.run(100)
        self.assertPC(self.cpu, 8)
        self.assertReg(self.cpu, 3, 0)
        self.cpu.breakpoints.clear()
        self.cpu.run(2)
        self.assertReg(self.cpu, 3, 3)

    def test_step_on_a_halted_cpu_raises(self):
        """cpu: stepping a halted CPU raises"""
        self.cpu.halt(0)
        with self.assertRaises(EmulatorError):
            self.cpu.step()

    def test_run_stops_when_halted(self):
        """cpu: run() returns as soon as the CPU halts"""
        cpu = CPU(ecall_handler=HostEnvironment(stdout=io.StringIO()))
        cpu.load_program(assemble("li a7, 93\nli a0, 0\necall\nli x5, 99").image)
        executed = cpu.run(100)
        self.assertTrue(cpu.halted)
        self.assertEqual(executed, 3)
        self.assertReg(cpu, 5, 0, "execution must stop at the ECALL")


class TestReset(EmulatorTestCase):
    def test_reset_restores_architectural_state(self):
        """cpu: reset() zeroes registers and returns the PC to the entry"""
        cpu = CPU()
        cpu.load_program(assemble("li x1, 1\nli x2, 2").image)
        cpu.run(2)
        cpu.reset()
        self.assertPC(cpu, 0)
        self.assertReg(cpu, 1, 0)
        self.assertReg(cpu, 2, 0)
        self.assertEqual(cpu.instret, 0)
        self.assertFalse(cpu.halted)

    def test_reset_preserves_memory(self):
        """cpu: reset() does not erase memory"""
        cpu = CPU()
        program = assemble("li x1, 7")
        cpu.load_program(program.image)
        cpu.write_u32(SCRATCH, 0xABCD)
        cpu.run(1)
        cpu.reset()
        self.assertEqual(cpu.read_u32(0), program.words[0])
        self.assertEqual(cpu.read_u32(SCRATCH), 0xABCD)

    def test_reset_is_rerunnable(self):
        """cpu: a program can be re-run after reset"""
        cpu = CPU()
        cpu.load_program(assemble("li x1, 10\nli x2, 20\nadd x3, x1, x2").image)
        cpu.run(3)
        first = cpu.get_reg(3)
        cpu.reset()
        cpu.run(3)
        self.assertEqual(cpu.get_reg(3), first)
        self.assertEqual(cpu.instret, 3)

    def test_reset_clears_halt(self):
        """cpu: reset() clears the halted flag"""
        cpu = CPU()
        cpu.halt(3)
        cpu.reset()
        self.assertFalse(cpu.halted)
        self.assertEqual(cpu.halt_code, 0)

    def test_reset_to_a_new_pc(self):
        """cpu: reset(pc) changes the reset vector"""
        cpu = CPU()
        cpu.reset(0x1000)
        self.assertPC(cpu, 0x1000)
        cpu.set_pc(0x2000)
        cpu.reset()
        self.assertPC(cpu, 0x1000)


class TestMemoryAPI(EmulatorTestCase):
    def setUp(self):
        self.cpu = CPU()

    def test_word_access(self):
        """cpu: host word read/write"""
        self.cpu.write_u32(SCRATCH, 0x12345678)
        self.assertEqual(self.cpu.read_u32(SCRATCH), 0x12345678)

    def test_byte_and_half_access(self):
        """cpu: host byte and halfword read/write"""
        self.cpu.write_u8(SCRATCH, 0xAB)
        self.cpu.write_u16(SCRATCH + 2, 0xCDEF)
        self.assertEqual(self.cpu.read_u8(SCRATCH), 0xAB)
        self.assertEqual(self.cpu.read_u16(SCRATCH + 2), 0xCDEF)

    def test_bulk_access(self):
        """cpu: host byte-string read/write"""
        self.cpu.write_bytes(SCRATCH, b"riscv")
        self.assertEqual(self.cpu.read_bytes(SCRATCH, 5), b"riscv")

    def test_load_program_from_words(self):
        """cpu: load_program accepts a list of words"""
        self.cpu.load_program([0x00A00093], addr=0x100)
        self.assertEqual(self.cpu.read_u32(0x100), 0x00A00093)
        self.assertPC(self.cpu, 0x100)

    def test_load_program_from_bytes(self):
        """cpu: load_program accepts raw bytes"""
        self.cpu.load_program(b"\x93\x00\xa0\x00", addr=0x200)
        self.assertEqual(self.cpu.read_u32(0x200), 0x00A00093)

    def test_load_program_without_moving_the_pc(self):
        """cpu: load_program(set_pc=False) leaves the PC alone"""
        self.cpu.set_pc(0x40)
        self.cpu.load_program([NOP], addr=0x300, set_pc=False)
        self.assertPC(self.cpu, 0x40)

    def test_guest_and_host_views_agree(self):
        """cpu: guest stores are visible through the host API"""
        cpu = run_asm(f"li x1, {SCRATCH}\nli x2, 0xCAFE\nsw x2, 0(x1)")
        self.assertEqual(cpu.read_u32(SCRATCH), 0xCAFE)


class TestTraps(EmulatorTestCase):
    def test_illegal_instruction(self):
        """trap: an illegal instruction raises with cause 2"""
        cpu = make_cpu([0x00000000])
        with self.assertRaises(IllegalInstruction) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 2)
        self.assertEqual(context.exception.tval, 0)
        self.assertPC(cpu, 0)

    def test_trap_leaves_the_pc_on_the_faulting_instruction(self):
        """trap: the PC stays on the faulting instruction (mepc semantics)"""
        cpu = make_cpu([NOP, 0x00000000, NOP])
        cpu.step()
        with self.assertRaises(IllegalInstruction):
            cpu.step()
        self.assertPC(cpu, 4)

    def test_load_address_misaligned(self):
        """trap: a misaligned LW raises cause 4"""
        cpu = make_cpu([encode("LW", rd=1, rs1=0, imm=2)])
        with self.assertRaises(LoadAddressMisaligned) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 4)
        self.assertEqual(context.exception.tval, 2)

    def test_store_address_misaligned(self):
        """trap: a misaligned SW raises cause 6"""
        cpu = make_cpu([encode("SW", rs1=0, rs2=0, imm=1)])
        with self.assertRaises(StoreAddressMisaligned) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 6)

    def test_halfword_alignment(self):
        """trap: halfword accesses must be 2-byte aligned"""
        cpu = make_cpu([encode("LH", rd=1, rs1=0, imm=1)])
        with self.assertRaises(LoadAddressMisaligned):
            cpu.step()
        cpu = make_cpu([encode("LH", rd=1, rs1=0, imm=2)])
        cpu.step()  # 2-byte aligned: fine

    def test_byte_accesses_are_always_aligned(self):
        """trap: byte accesses never raise an alignment trap"""
        for offset in range(4):
            cpu = make_cpu([encode("LB", rd=1, rs1=0, imm=offset)])
            cpu.step()
            self.assertEqual(cpu.instret, 1)

    def test_allow_misaligned_option(self):
        """trap: allow_misaligned suppresses alignment traps"""
        cpu = make_cpu(
            [encode("SW", rs1=0, rs2=0, imm=0x102), encode("LW", rd=1, rs1=0, imm=0x102)],
            allow_misaligned=True,
        )
        cpu.run(2)
        self.assertEqual(cpu.instret, 2)

    def test_instruction_address_misaligned(self):
        """trap: a jump to a misaligned address raises cause 0"""
        cpu = make_cpu([encode("JAL", rd=1, imm=2)])
        with self.assertRaises(InstructionAddressMisaligned) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 0)
        self.assertReg(cpu, 1, 0, "rd must not be written when the jump traps")

    def test_misaligned_branch_target(self):
        """trap: a taken branch to a misaligned address raises"""
        word = encode("BEQ", rs1=0, rs2=0, imm=4) | (1 << 8)  # force offset bit 1
        cpu = make_cpu([word])
        with self.assertRaises(InstructionAddressMisaligned):
            cpu.step()

    def test_untaken_branch_ignores_misalignment(self):
        """trap: an untaken branch never checks its target"""
        word = encode("BNE", rs1=0, rs2=0, imm=4) | (1 << 8)
        cpu = make_cpu([word, NOP])
        cpu.step()
        self.assertPC(cpu, 4)

    def test_load_access_fault(self):
        """trap: a load outside memory raises cause 5"""
        cpu = CPU(memory_size=4096)
        cpu.set_reg(1, 0x8000)
        cpu.load_program([encode("LW", rd=2, rs1=1, imm=0)])
        with self.assertRaises(LoadAccessFault) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 5)

    def test_store_access_fault(self):
        """trap: a store outside memory raises cause 7"""
        cpu = CPU(memory_size=4096)
        cpu.set_reg(1, 0x8000)
        cpu.load_program([encode("SW", rs1=1, rs2=0, imm=0)])
        with self.assertRaises(StoreAccessFault) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 7)

    def test_instruction_access_fault(self):
        """trap: fetching outside memory raises cause 1"""
        cpu = CPU(memory_size=4096)
        cpu.set_pc(0x8000)
        with self.assertRaises(InstructionAccessFault) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 1)

    def test_misaligned_pc_fetch(self):
        """trap: fetching from a misaligned PC raises cause 0"""
        cpu = make_cpu([NOP, NOP])
        cpu.set_pc(2)
        with self.assertRaises(InstructionAddressMisaligned):
            cpu.step()

    def test_ecall_traps_by_default(self):
        """trap: ECALL raises cause 11 with no handler installed"""
        cpu = make_cpu([encode("ECALL")])
        with self.assertRaises(EnvironmentCall) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 11)

    def test_ebreak_traps_by_default(self):
        """trap: EBREAK raises cause 3 with no handler installed"""
        cpu = make_cpu([encode("EBREAK")])
        with self.assertRaises(Breakpoint) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 3)

    def test_last_exception_is_recorded(self):
        """trap: the CPU records the last exception it took"""
        cpu = make_cpu([0x00000000])
        with self.assertRaises(RiscvException):
            cpu.step()
        self.assertIsInstance(cpu.last_exception, IllegalInstruction)

    def test_execution_can_resume_after_a_trap(self):
        """trap: the host can skip a faulting instruction and continue"""
        cpu = make_cpu([0x00000000, encode("ADDI", rd=1, rs1=0, imm=7)])
        with self.assertRaises(IllegalInstruction):
            cpu.step()
        cpu.set_pc(cpu.pc + 4)  # what a trap handler would do
        cpu.step()
        self.assertReg(cpu, 1, 7)


class TestEnvironmentHooks(EmulatorTestCase):
    def test_ecall_handler_is_invoked(self):
        """cpu: an ECALL handler replaces the architectural trap"""
        seen = []
        cpu = make_cpu([encode("ECALL"), NOP], ecall_handler=lambda c: seen.append(c.pc))
        cpu.run(2)
        self.assertEqual(seen, [0])
        self.assertEqual(cpu.instret, 2)

    def test_ebreak_handler_is_invoked(self):
        """cpu: an EBREAK handler replaces the architectural trap"""
        cpu = make_cpu([encode("EBREAK")], ebreak_handler=lambda c: c.halt(7))
        cpu.run(4)
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 7)

    def test_host_environment_write_and_exit(self):
        """cpu: the bundled host environment implements write and exit"""
        stream = io.StringIO()
        host = HostEnvironment(stdout=stream, capture=True)
        cpu = CPU(ecall_handler=host)
        program = assemble(
            """
                la   a1, msg
                li   a2, 5
                li   a0, 1
                li   a7, 64
                ecall
                li   a0, 3
                li   a7, 93
                ecall
            .data
            msg:
                .ascii "hello"
            """
        )
        program.load(cpu.memory)
        cpu.set_pc(program.entry)
        cpu.run(100)
        self.assertEqual(host.text(), "hello")
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 3)


class TestTracing(EmulatorTestCase):
    def setUp(self):
        self.cpu = CPU()
        self.cpu.load_program(assemble("addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2").image)
        self.stream = io.StringIO()

    def test_trace_output_format(self):
        """trace: output matches the format in the brief"""
        self.cpu.enable_trace(self.stream)
        self.cpu.run(3)
        self.assertEqual(
            self.stream.getvalue().splitlines(),
            [
                "PC=0x00000000  INST=0x00A00093  ADDI x1,x0,10",
                "PC=0x00000004  INST=0x01400113  ADDI x2,x0,20",
                "PC=0x00000008  INST=0x002081B3  ADD  x3,x1,x2",
            ],
        )

    def test_trace_can_be_disabled(self):
        """trace: disable_trace() stops the output"""
        self.cpu.enable_trace(self.stream)
        self.cpu.step()
        self.cpu.disable_trace()
        self.cpu.run(2)
        self.assertEqual(len(self.stream.getvalue().splitlines()), 1)

    def test_trace_with_register_writeback(self):
        """trace: register mode shows the value written"""
        self.cpu.enable_trace(self.stream, registers=True)
        self.cpu.run(3)
        lines = self.stream.getvalue().splitlines()
        self.assertIn("x1=0x0000000A", lines[0])
        self.assertIn("x3=0x0000001E", lines[2])

    def test_trace_hook(self):
        """trace: a hook can capture instructions programmatically"""
        seen = []
        self.cpu.trace = True
        self.cpu.trace_hook = lambda cpu, pc, word, inst: seen.append((pc, inst.name))
        self.cpu.run(3)
        self.assertEqual(seen, [(0, "ADDI"), (4, "ADDI"), (8, "ADD")])

    def test_trace_marks_a_trapping_instruction(self):
        """trace: a trapping instruction is still traced, with a marker"""
        cpu = make_cpu([encode("LW", rd=1, rs1=0, imm=1)])
        cpu.enable_trace(self.stream)
        with self.assertRaises(LoadAddressMisaligned):
            cpu.step()
        self.assertIn("load address misaligned", self.stream.getvalue())


class TestInspection(EmulatorTestCase):
    def test_state_snapshot(self):
        """cpu: state() reports the full architectural state"""
        cpu = CPU()
        cpu.load_program(assemble("li x1, 10\nli x2, 20\nadd x3, x1, x2").image)
        cpu.run(3)
        state = cpu.state()
        self.assertEqual(state["pc"], 12)
        self.assertEqual(state["registers"][3], 30)
        self.assertEqual(state["instret"], 3)
        self.assertFalse(state["halted"])

    def test_state_roundtrip(self):
        """cpu: restore_state(state()) reproduces the machine"""
        cpu = CPU()
        cpu.load_program(assemble("li x1, 42").image)
        cpu.run(1)
        state = cpu.state()
        clone = CPU()
        clone.restore_state(state)
        self.assertEqual(clone.get_reg(1), 42)
        self.assertEqual(clone.pc, cpu.pc)

    def test_dump_registers(self):
        """cpu: dump_registers renders every register"""
        cpu = CPU()
        cpu.set_reg(2, 0x8000)
        text = cpu.dump_registers()
        self.assertIn("x2/sp = 0x00008000", text)
        self.assertIn("x31/t6", text)

    def test_disassemble_range(self):
        """cpu: disassemble_range annotates memory with addresses"""
        cpu = CPU()
        cpu.load_program(assemble("addi x1, x0, 10\nadd x3, x1, x2").image)
        text = cpu.disassemble_range(0, 2)
        self.assertIn("0x00000000:  00A00093  ADDI x1,x0,10", text)
        self.assertIn("0x00000004:", text)

    def test_fetch_and_decode_are_separable(self):
        """cpu: fetch and decode are usable independently of execute"""
        cpu = make_cpu([0x00A00093])
        word = cpu.fetch()
        self.assertEqual(word, 0x00A00093)
        inst = cpu.decode(word)
        self.assertEqual(inst.name, "ADDI")
        self.assertEqual(cpu.instret, 0, "decoding must not retire an instruction")


if __name__ == "__main__":
    unittest.main()
