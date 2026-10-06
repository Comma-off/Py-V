"""A comprehensive internal validation suite.

This is what ``python riscv.py --self-test`` runs.  It is deliberately
independent of the ``tests/`` directory and of pytest: it imports nothing but
the emulator itself, so a copy of the ``riscv/`` package alone can still prove
itself correct.

Each check is a named assertion over observable architectural state.  The
suite reports one line per check group and a pass/fail summary, and the caller
turns that into a process exit code.
"""

import sys

from .assembler import assemble
from .bits import to_signed
from .cpu import CPU
from .decoder import Decoder
from .disasm import disassemble
from .encoder import encode
from .exceptions import (
    Breakpoint,
    EnvironmentCall,
    IllegalInstruction,
    InstructionAddressMisaligned,
    LoadAccessFault,
    LoadAddressMisaligned,
    RiscvException,
    StoreAddressMisaligned,
)
from .host import halt_on_ecall
from .instructions import RV32I, RV32I_INSTRUCTION_COUNT
from .memory import Memory


class SelfTest:
    """Collects named checks and reports pass/fail."""

    def __init__(self, verbose: bool = True, stream=None):
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stdout
        self.passed = 0
        self.failed = 0
        self.failures: list[tuple[str, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append((name, detail))
            if self.verbose:
                print(f"[FAIL] {name}: {detail}", file=self.stream)
        return bool(condition)

    def equal(self, name: str, actual, expected) -> bool:
        ok = actual == expected
        detail = ""
        if not ok:
            if isinstance(actual, int) and isinstance(expected, int):
                detail = f"expected 0x{expected:08X}, got 0x{actual:08X}"
            else:
                detail = f"expected {expected!r}, got {actual!r}"
        return self.check(name, ok, detail)

    def raises(self, name: str, exception, callable_, *args, **kwargs) -> bool:
        try:
            callable_(*args, **kwargs)
        except exception:
            return self.check(name, True)
        except Exception as exc:  # noqa: BLE001 - report the wrong exception
            return self.check(name, False, f"raised {type(exc).__name__}: {exc}")
        return self.check(name, False, f"expected {exception.__name__}, nothing raised")

    def group(self, title: str) -> None:
        if self.verbose:
            print(f"\n-- {title} " + "-" * max(0, 60 - len(title)), file=self.stream)

    def result(self, title: str) -> None:
        """Print a per-group [PASS]/[FAIL] line based on failures so far."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_cpu(words=(), **kwargs) -> CPU:
    """A CPU with ``words`` loaded at address 0 and the PC pointing at them."""
    cpu = CPU(**kwargs)
    if words:
        cpu.load_program(list(words), addr=0)
    return cpu


def run_source(source, steps=10000, **kwargs) -> CPU:
    """Assemble ``source``, run it to completion (or ``steps``), return the CPU.

    A program with no ECALL of its own gets an exit sequence appended, so that
    straight-line snippets stop cleanly instead of running into zeroed memory
    (which would -- correctly -- raise an illegal-instruction trap).
    """
    if "ecall" not in source.lower():
        source += "\n li a7, 93\n li a0, 0\n ecall\n"
    program = assemble(source)
    cpu = CPU(ecall_handler=halt_on_ecall, **kwargs)
    program.load(cpu.memory)
    cpu.set_pc(program.entry)
    cpu.run(steps)
    return cpu


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_registers(test: SelfTest) -> None:
    test.group("register file")
    cpu = CPU()
    cpu.set_reg(1, 0x12345678)
    test.equal("x1 round-trips", cpu.get_reg(1), 0x12345678)

    cpu.set_reg(0, 0xFFFFFFFF)
    test.equal("x0 ignores writes", cpu.get_reg(0), 0)

    cpu.set_reg(2, 0x1_0000_0000 + 7)
    test.equal("register writes truncate to 32 bits", cpu.get_reg(2), 7)

    cpu.set_reg(3, -1)
    test.equal("negative writes become two's complement", cpu.get_reg(3), 0xFFFFFFFF)
    test.equal("signed read of -1", cpu.get_reg_signed(3), -1)

    cpu.write_register("sp", 0x8000)
    test.equal("ABI name write (sp == x2)", cpu.get_reg(2), 0x8000)
    test.equal("ABI name read", cpu.read_register("x2"), 0x8000)

    for index in range(1, 32):
        cpu.set_reg(index, index)
    cpu.reset()
    test.equal("reset zeroes registers", sum(cpu.regs.snapshot()), 0)
    test.equal("reset restores pc", cpu.pc, cpu.reset_pc)
    test.equal("reset clears instret", cpu.instret, 0)


def check_memory(test: SelfTest) -> None:
    test.group("memory")
    mem = Memory(size=1 << 16)
    mem.write_u32(0x100, 0xDEADBEEF)
    test.equal("u32 round-trip", mem.read_u32(0x100), 0xDEADBEEF)
    test.equal("little-endian byte 0", mem.read_u8(0x100), 0xEF)
    test.equal("little-endian byte 3", mem.read_u8(0x103), 0xDE)
    test.equal("u16 low half", mem.read_u16(0x100), 0xBEEF)
    test.equal("unwritten memory reads zero", mem.read_u32(0x200), 0)

    mem.write_u32(0x0FFE, 0x11223344)
    test.equal("page-crossing write/read", mem.read_u32(0x0FFE), 0x11223344)
    test.equal("page-crossing byte order", mem.read_u8(0x1000), 0x22)

    mem.write_bytes(0x300, b"riscv")
    test.equal("byte-string round-trip", mem.read_bytes(0x300, 5), b"riscv")
    test.equal("signed byte read", mem.read_i8(0x300), ord("r"))

    mem.write_u8(0x400, 0xFF)
    test.equal("signed byte read of 0xFF", mem.read_i8(0x400), -1)
    test.equal("truncating store", (mem.write_u8(0x401, 0x1FF), mem.read_u8(0x401))[1], 0xFF)

    from .exceptions import MemoryError_

    test.raises("out-of-range read raises", MemoryError_, mem.read_u32, 1 << 16)
    test.raises("negative address raises", MemoryError_, mem.read_u8, -1)
    mem.clear()
    test.equal("clear() drops pages", mem.allocated_pages, 0)


def check_decoder(test: SelfTest) -> None:
    test.group("decoder")
    decoder = Decoder()

    inst = decoder.decode(0x00A00093)  # addi x1, x0, 10
    test.equal("ADDI decodes by name", inst.name, "ADDI")
    test.equal("ADDI rd", inst.rd, 1)
    test.equal("ADDI rs1", inst.rs1, 0)
    test.equal("ADDI imm", inst.imm_signed, 10)

    inst = decoder.decode(0xFFF00093)  # addi x1, x0, -1
    test.equal("negative I-immediate sign-extends", inst.imm, 0xFFFFFFFF)
    test.equal("negative I-immediate as signed", inst.imm_signed, -1)

    inst = decoder.decode(encode("SW", rs1=2, rs2=3, imm=-4))
    test.equal("S-immediate reassembles", inst.imm_signed, -4)

    inst = decoder.decode(encode("BEQ", rs1=1, rs2=2, imm=-8))
    test.equal("B-immediate reassembles", inst.imm_signed, -8)

    inst = decoder.decode(encode("LUI", rd=5, imm=0xFFFFF))
    test.equal("U-immediate is pre-shifted", inst.imm, 0xFFFFF000)

    inst = decoder.decode(encode("JAL", rd=1, imm=-2048))
    test.equal("J-immediate reassembles", inst.imm_signed, -2048)

    inst = decoder.decode(encode("SRAI", rd=1, rs1=2, imm=31))
    test.equal("SRAI decodes distinctly from SRLI", inst.name, "SRAI")
    test.equal("SRAI shamt", inst.shamt, 31)

    test.raises("all-zero word is illegal", IllegalInstruction, decoder.decode, 0x00000000)
    test.raises("all-ones word is illegal", IllegalInstruction, decoder.decode, 0xFFFFFFFF)
    test.raises(
        "SLLI with a stray funct7 bit is illegal",
        IllegalInstruction,
        decoder.decode,
        0x02001093,
    )
    test.raises(
        "unknown SYSTEM encoding is illegal (no CSRs yet)",
        IllegalInstruction,
        decoder.decode,
        0x30001073,
    )
    test.equal("ISA table holds 40 instructions", len(RV32I), RV32I_INSTRUCTION_COUNT)

    # Every instruction must encode, decode, and disassemble.
    round_trips = 0
    for definition in RV32I:
        word = definition.match
        decoded = decoder.decode(word)
        if decoded.name == definition.name and disassemble(word):
            round_trips += 1
    test.equal("every instruction decodes from its base encoding", round_trips, len(RV32I))


def check_arithmetic(test: SelfTest) -> None:
    test.group("arithmetic and logic")
    cases = [
        # (name, source, register, expected)
        ("ADD", "li x1, 10\nli x2, 20\nadd x3, x1, x2", 3, 30),
        ("ADD wraps", "li x1, 0x7FFFFFFF\nli x2, 1\nadd x3, x1, x2", 3, 0x80000000),
        ("ADD of two negatives", "li x1, -1\nli x2, -1\nadd x3, x1, x2", 3, 0xFFFFFFFE),
        ("SUB", "li x1, 10\nli x2, 20\nsub x3, x1, x2", 3, 0xFFFFFFF6),
        ("SUB underflows", "li x1, 0\nli x2, 1\nsub x3, x1, x2", 3, 0xFFFFFFFF),
        ("AND", "li x1, 0xFF00\nli x2, 0x0FF0\nand x3, x1, x2", 3, 0x0F00),
        ("OR", "li x1, 0xFF00\nli x2, 0x0FF0\nor x3, x1, x2", 3, 0xFFF0),
        ("XOR", "li x1, 0xFF00\nli x2, 0x0FF0\nxor x3, x1, x2", 3, 0xF0F0),
        ("SLL", "li x1, 1\nli x2, 31\nsll x3, x1, x2", 3, 0x80000000),
        ("SLL shamt wraps to 5 bits", "li x1, 1\nli x2, 33\nsll x3, x1, x2", 3, 2),
        ("SRL is logical", "li x1, -1\nli x2, 28\nsrl x3, x1, x2", 3, 0xF),
        ("SRA is arithmetic", "li x1, -1\nli x2, 28\nsra x3, x1, x2", 3, 0xFFFFFFFF),
        ("SRA of a positive", "li x1, 0x40000000\nli x2, 4\nsra x3, x1, x2", 3, 0x04000000),
        ("SLT signed", "li x1, -1\nli x2, 1\nslt x3, x1, x2", 3, 1),
        ("SLTU unsigned", "li x1, -1\nli x2, 1\nsltu x3, x1, x2", 3, 0),
        ("SLTI", "li x1, -5\nslti x3, x1, -4", 3, 1),
        ("SLTIU compares sign-extended imm", "li x1, 1\nsltiu x3, x1, -1", 3, 1),
        ("ADDI negative", "li x1, 5\naddi x3, x1, -10", 3, 0xFFFFFFFB),
        ("ANDI sign-extends its immediate", "li x1, -1\nandi x3, x1, -256", 3, 0xFFFFFF00),
        ("XORI -1 inverts", "li x1, 0x0F0F0F0F\nxori x3, x1, -1", 3, 0xF0F0F0F0),
        ("SLLI", "li x1, 3\nslli x3, x1, 4", 3, 48),
        ("SRLI", "li x1, -16\nsrli x3, x1, 4", 3, 0x0FFFFFFF),
        ("SRAI", "li x1, -16\nsrai x3, x1, 4", 3, 0xFFFFFFFF),
        ("LUI", "lui x3, 0x12345", 3, 0x12345000),
        ("LUI sets low bits to zero", "li x3, -1\nlui x3, 0", 3, 0),
    ]
    for name, source, reg, expected in cases:
        cpu = run_source(source)
        test.equal(name, cpu.get_reg(reg), expected)

    cpu = run_source("auipc x3, 0")
    test.equal("AUIPC at pc 0", cpu.get_reg(3), 0)
    cpu = run_source("nop\nnop\nauipc x3, 1")
    test.equal("AUIPC adds pc", cpu.get_reg(3), 0x1000 + 8)


def check_control_flow(test: SelfTest) -> None:
    test.group("branches and jumps")
    branch_cases = [
        ("BEQ taken", "li x1, 5\nli x2, 5\nbeq x1, x2, 8\nli x3, 1\nli x4, 2", 3, 0),
        ("BEQ not taken", "li x1, 5\nli x2, 6\nbeq x1, x2, 8\nli x3, 1", 3, 1),
        ("BNE taken", "li x1, 5\nli x2, 6\nbne x1, x2, 8\nli x3, 1", 3, 0),
        ("BLT signed taken", "li x1, -1\nli x2, 1\nblt x1, x2, 8\nli x3, 1", 3, 0),
        ("BLT signed not taken", "li x1, 1\nli x2, -1\nblt x1, x2, 8\nli x3, 1", 3, 1),
        ("BGE taken on equal", "li x1, 5\nli x2, 5\nbge x1, x2, 8\nli x3, 1", 3, 0),
        ("BLTU unsigned taken", "li x1, 1\nli x2, -1\nbltu x1, x2, 8\nli x3, 1", 3, 0),
        ("BGEU unsigned taken", "li x1, -1\nli x2, 1\nbgeu x1, x2, 8\nli x3, 1", 3, 0),
    ]
    for name, source, reg, expected in branch_cases:
        cpu = run_source(source, steps=10)
        test.equal(name, cpu.get_reg(reg), expected)

    cpu = run_source("jal x1, target\nli x3, 1\ntarget:\nli x4, 2", steps=4)
    test.equal("JAL links pc+4", cpu.get_reg(1), 4)
    test.equal("JAL skips the intervening instruction", cpu.get_reg(3), 0)
    test.equal("JAL reaches its target", cpu.get_reg(4), 2)

    cpu = run_source("li x1, 12\njalr x5, 0(x1)\nli x3, 1\nli x4, 2", steps=4)
    test.equal("JALR links pc+4", cpu.get_reg(5), 8)
    test.equal("JALR jumps to rs1+imm", cpu.get_reg(4), 2)
    test.equal("JALR skips the intervening instruction", cpu.get_reg(3), 0)

    cpu = run_source("li x1, 13\njalr x0, 0(x1)\nli x3, 1\nli x4, 2", steps=4)
    test.equal("JALR clears the low bit of its target", cpu.get_reg(4), 2)

    cpu = run_source("li sp, 0x1000\njal ra, f\nli x5, 7\nj done\nf:\nret\ndone:", steps=20)
    test.equal("call/return sequence resumes after the call", cpu.get_reg(5), 7)

    cpu = run_source("li x1, 3\nloop:\naddi x1, x1, -1\nbnez x1, loop", steps=20)
    test.equal("backward branch loop terminates", cpu.get_reg(1), 0)


def check_memory_instructions(test: SelfTest) -> None:
    test.group("loads and stores")
    cases = [
        ("SW/LW", "li x1, 0x400\nli x2, 0x12345678\nsw x2, 0(x1)\nlw x3, 0(x1)", 3, 0x12345678),
        ("SB/LB sign-extends", "li x1, 0x400\nli x2, 0xFF\nsb x2, 0(x1)\nlb x3, 0(x1)", 3, 0xFFFFFFFF),
        ("SB/LBU zero-extends", "li x1, 0x400\nli x2, 0xFF\nsb x2, 0(x1)\nlbu x3, 0(x1)", 3, 0xFF),
        ("SH/LH sign-extends", "li x1, 0x400\nli x2, 0x8000\nsh x2, 0(x1)\nlh x3, 0(x1)", 3, 0xFFFF8000),
        ("SH/LHU zero-extends", "li x1, 0x400\nli x2, 0x8000\nsh x2, 0(x1)\nlhu x3, 0(x1)", 3, 0x8000),
        ("SB truncates its source", "li x1, 0x400\nli x2, 0x1234\nsb x2, 0(x1)\nlbu x3, 0(x1)", 3, 0x34),
        ("negative store offset", "li x1, 0x400\nli x2, 42\nsw x2, -4(x1)\nlw x3, -4(x1)", 3, 42),
        ("little-endian byte order", "li x1, 0x400\nli x2, 0x11223344\nsw x2, 0(x1)\nlbu x3, 3(x1)", 3, 0x11),
    ]
    for name, source, reg, expected in cases:
        cpu = run_source(source)
        test.equal(name, cpu.get_reg(reg), expected)

    cpu = run_source("li x1, 0x400\nli x2, 0xAABBCCDD\nsw x2, 0(x1)")
    test.equal("store reaches memory", cpu.read_u32(0x400), 0xAABBCCDD)


def check_traps(test: SelfTest) -> None:
    test.group("traps")
    cpu = make_cpu([0x00000000])
    test.raises("illegal instruction traps", IllegalInstruction, cpu.step)
    test.equal("pc stays on the faulting instruction", cpu.pc, 0)
    test.equal("instret is not incremented by a trap", cpu.instret, 0)

    cpu = make_cpu([encode("ECALL")])
    test.raises("ECALL traps without a handler", EnvironmentCall, cpu.step)

    cpu = make_cpu([encode("EBREAK")])
    test.raises("EBREAK traps without a handler", Breakpoint, cpu.step)

    cpu = make_cpu([encode("LW", rd=1, rs1=0, imm=1)])
    test.raises("misaligned LW traps", LoadAddressMisaligned, cpu.step)

    cpu = make_cpu([encode("SW", rs1=0, rs2=0, imm=2)])
    test.raises("misaligned SW traps", StoreAddressMisaligned, cpu.step)

    cpu = make_cpu([encode("LH", rd=1, rs1=0, imm=1)])
    test.raises("misaligned LH traps", LoadAddressMisaligned, cpu.step)

    cpu = make_cpu([encode("LB", rd=1, rs1=0, imm=1)])
    cpu.step()
    test.equal("byte loads are never misaligned", cpu.instret, 1)

    cpu = make_cpu([encode("LW", rd=1, rs1=0, imm=1)], allow_misaligned=True)
    cpu.step()
    test.equal("allow_misaligned permits an unaligned LW", cpu.instret, 1)

    cpu = make_cpu([encode("JAL", rd=0, imm=2)])
    test.raises(
        "misaligned jump target traps",
        InstructionAddressMisaligned,
        cpu.step,
    )

    cpu = CPU(memory_size=4096)
    cpu.set_reg(1, 0x100000)
    cpu.load_program([encode("LW", rd=2, rs1=1, imm=0)])
    test.raises("load outside memory faults", LoadAccessFault, cpu.step)

    trap = None
    cpu = make_cpu([0x00000000])
    try:
        cpu.step()
    except RiscvException as exc:
        trap = exc
    test.equal("illegal-instruction cause code", trap.cause, 2)
    test.equal("trap carries the offending word in tval", trap.tval, 0)


def check_pc_and_stepping(test: SelfTest) -> None:
    test.group("program counter and stepping")
    program = assemble("addi x1, x0, 1\naddi x2, x0, 2\naddi x3, x0, 3")
    cpu = CPU()
    cpu.load_program(program.image)
    test.equal("pc starts at the load address", cpu.pc, 0)
    cpu.step()
    test.equal("pc advances by 4", cpu.pc, 4)
    test.equal("instret counts one", cpu.instret, 1)
    cpu.step()
    cpu.step()
    test.equal("pc after three instructions", cpu.pc, 12)
    test.equal("all three results are visible", (cpu.get_reg(1), cpu.get_reg(2), cpu.get_reg(3)), (1, 2, 3))

    executed = CPU().run(0)
    test.equal("run(0) executes nothing", executed, 0)

    cpu.reset()
    test.equal("reset returns the pc to the entry", cpu.pc, 0)
    test.equal("reset preserves memory", cpu.read_u32(0), program.words[0])
    executed = cpu.run(2)
    test.equal("run(n) executes exactly n instructions", executed, 2)
    test.equal("run(n) leaves the pc after n instructions", cpu.pc, 8)

    cpu.set_pc(0)
    test.equal("set_pc moves execution", cpu.pc, 0)

    cpu = CPU()
    cpu.load_program(assemble("li x1, 1\nli x2, 2\nli x3, 3\nli x4, 4").image)
    cpu.breakpoints.add(8)
    executed = cpu.run(100)
    test.equal("run stops at a breakpoint", cpu.pc, 8)
    test.equal("breakpoint stops before executing", cpu.get_reg(3), 0)


def check_state_and_trace(test: SelfTest) -> None:
    test.group("inspection, tracing and the host API")
    import io

    cpu = CPU()
    cpu.load_program(assemble("addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2").image)
    stream = io.StringIO()
    cpu.enable_trace(stream)
    cpu.run(3)
    lines = stream.getvalue().strip().splitlines()
    test.equal("trace emits one line per instruction", len(lines), 3)
    test.equal(
        "trace line format",
        lines[0],
        "PC=0x00000000  INST=0x00A00093  ADDI x1,x0,10",
    )
    test.equal("trace disassembles R-type", lines[2].split("  ", 2)[2], "ADD  x3,x1,x2")

    cpu.disable_trace()
    stream2 = io.StringIO()
    cpu.trace_stream = stream2
    cpu.reset()
    cpu.run(3)
    test.equal("disable_trace silences output", stream2.getvalue(), "")

    state = cpu.state()
    test.equal("state reports the pc", state["pc"], 12)
    test.equal("state reports x3", state["registers"][3], 30)
    test.equal("state reports instret", state["instret"], 3)

    cpu2 = CPU()
    cpu2.restore_state(state)
    test.equal("restore_state restores registers", cpu2.get_reg(3), 30)
    test.equal("restore_state restores the pc", cpu2.pc, 12)

    cpu3 = CPU()
    cpu3.write_u32(0x100, 0xCAFEBABE)
    test.equal("host word write/read", cpu3.read_u32(0x100), 0xCAFEBABE)
    cpu3.write_bytes(0x200, b"abc")
    test.equal("host byte write/read", cpu3.read_bytes(0x200, 3), b"abc")


def check_programs(test: SelfTest) -> None:
    test.group("end-to-end programs")
    cpu = run_source("addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2", steps=3)
    test.equal("the canonical three-instruction program", cpu.get_reg(3), 30)

    cpu = run_source(
        """
        li   t0, 0          # sum
        li   t1, 1          # i
    loop:
        li   t2, 11
        bge  t1, t2, done
        add  t0, t0, t1
        addi t1, t1, 1
        j    loop
    done:
        mv   s0, t0
        """
    )
    test.equal("sum 1..10 == 55", cpu.get_reg(8), 55)

    cpu = run_source(
        """
        li   sp, 0x8000
        li   a0, 6
        jal  ra, square
        mv   s0, a0
        li   a7, 93
        ecall
    square:
        mv   t0, a0
        li   a0, 0
        mv   t1, t0
    step:
        beqz t1, ret_
        add  a0, a0, t0
        addi t1, t1, -1
        j    step
    ret_:
        ret
        """
    )
    test.equal("function call computes 6*6", cpu.get_reg(8), 36)
    test.check("program halted via ECALL", cpu.halted)

    cpu = run_source(
        """
        li   t0, 0x1000
        li   t1, -1
        sw   t1, 0(t0)
        lw   s0, 0(t0)
        lbu  s1, 0(t0)
        lb   s2, 0(t0)
        """
    )
    test.equal("stored -1 reloads as -1", cpu.get_reg(8), 0xFFFFFFFF)
    test.equal("lbu of 0xFF", cpu.get_reg(9), 0xFF)
    test.equal("lb of 0xFF", cpu.get_reg(18), 0xFFFFFFFF)


def check_assembler_roundtrip(test: SelfTest) -> None:
    test.group("assembler / disassembler round-trip")
    program = assemble(
        """
        addi x1, x0, -1
        sw   x2, -4(x3)
        beq  x1, x2, -8
        jal  x1, 16
        lui  x5, 0x12345
        srai x6, x7, 31
        lbu  x8, 3(x9)
        fence
        ecall
        ebreak
        """
    )
    decoder = Decoder()
    mismatches = []
    for address, word, _line, _text in program.listing:
        text = disassemble(word, pc=address)
        try:
            rebuilt = assemble(text, text_base=address).words[0]
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{text}: {exc}")
            continue
        if rebuilt != word:
            mismatches.append(f"{text}: 0x{word:08X} -> 0x{rebuilt:08X}")
    test.check(
        "disassembly reassembles to the same words",
        not mismatches,
        "; ".join(mismatches),
    )

    for name in ("ADD", "ADDI", "LW", "SW", "BEQ", "JAL", "JALR", "LUI", "AUIPC", "SRAI"):
        definition = RV32I.by_name(name)
        word = definition.match
        test.equal(f"{name} decodes to itself", decoder.decode(word).name, name)


def check_elf(test: SelfTest) -> None:
    test.group("ELF32 loading and execution")
    from .elf import ElfError, load_elf, parse_elf, permission_text
    from .elfwriter import ElfWriter, program_to_elf
    from .memory import PERM_W, PERM_X, Memory

    PF_R, PF_W, PF_X = 0x4, 0x2, 0x1

    # -- round-trip through our own writer ------------------------------
    program = assemble("_start:\n li a0, 7\n li a7, 93\n ecall\n", text_base=0x10000)
    blob = program_to_elf(program)
    image = load_elf(blob)
    test.equal("ELF entry point", image.entry, 0x10000)
    test.equal("one loadable segment", len(image.segments), 1)
    test.equal("segment permissions", image.segments[0].permissions, "R-X")
    test.equal("segment lands at its vaddr", image.segments[0].vaddr, 0x10000)
    test.check("_start symbol survives", image.find_symbol("_start") is not None)

    cpu = CPU(memory_size=1 << 20, ecall_handler=halt_on_ecall)
    cpu.load_image(image)
    cpu.run(100)
    test.check("an ELF program runs", cpu.halted)
    test.equal("ELF program exit status", cpu.halt_code, 7)

    # -- data, BSS and permissions --------------------------------------
    writer = ElfWriter(entry=0x10000)
    writer.add_segment(0x10000, b"\x13\x00\x00\x00", PF_R | PF_X, align=0x1000)
    writer.add_segment(0x11000, b"\x11\x22\x33\x44", PF_R | PF_W, memsz=0x100, align=0x1000)
    image = load_elf(writer.build(with_symbols=False))
    memory = Memory(1 << 20)
    image.load_into(memory)
    test.equal("file bytes are placed", memory.read_u32(0x11000), 0x44332211)
    test.equal("BSS is zero-filled", memory.read_bytes(0x11004, 0xFC), bytes(0xFC))
    test.equal("two segments", len(image.segments), 2)
    test.equal("BSS size", image.segments[1].bss_size, 0xFC)

    cpu = CPU(memory_size=1 << 20)
    cpu.load_image(image, enforce_permissions=True)
    test.check("protection table installed", cpu.protection is not None)
    test.check("text is executable", cpu.protection.check(0x10000, 4, PERM_X))
    test.check("text is not writable", not cpu.protection.check(0x10000, 4, PERM_W))
    test.check("data is writable", cpu.protection.check(0x11000, 4, PERM_W))
    test.check("data is not executable", not cpu.protection.check(0x11000, 4, PERM_X))

    from .exceptions import StoreAccessFault

    cpu.set_reg(1, 0x10000)
    cpu.memory.write_u32(0x10000, encode("SW", rs1=1, rs2=0, imm=0))
    test.raises("writing to .text faults", StoreAccessFault, cpu.step)

    # -- malformed files are rejected cleanly ---------------------------
    bad_magic = b"\x7fELG" + blob[4:]
    test.raises("invalid magic is rejected", ElfError, load_elf, bad_magic)
    elf64 = bytearray(blob)
    elf64[4] = 2
    test.raises("ELF64 is rejected", ElfError, load_elf, bytes(elf64))
    big_endian = bytearray(blob)
    big_endian[5] = 2
    test.raises("big-endian is rejected", ElfError, load_elf, bytes(big_endian))
    wrong_machine = bytearray(blob)
    wrong_machine[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
    test.raises("wrong machine is rejected", ElfError, load_elf, bytes(wrong_machine))
    test.raises("truncated file is rejected", ElfError, load_elf, blob[:20])
    test.raises("empty input is rejected", ElfError, load_elf, b"")

    test.equal("permission_text renders flags", permission_text(PF_R | PF_X), "R-X")
    test.check("is_elf recognises the magic", parse_elf(blob).header.entry == 0x10000)


def check_rv32m(test: SelfTest) -> None:
    test.group("RV32M extension")
    from .isa import available_isas, get_isa
    from .rv32m import RV32IM

    test.equal("RV32I is still 40 instructions", len(RV32I), 40)
    test.equal("RV32IM adds eight", len(RV32IM) - len(RV32I), 8)
    test.check("RV32I does not contain MUL", "MUL" not in RV32I)
    test.check("rv32im is registered", "rv32im" in available_isas())
    test.check("get_isa returns the same object", get_isa("rv32im") is RV32IM)

    # Base encodings straight from the specification.
    for name, expected in (
        ("MUL", 0x02000033), ("MULH", 0x02001033), ("MULHSU", 0x02002033),
        ("MULHU", 0x02003033), ("DIV", 0x02004033), ("DIVU", 0x02005033),
        ("REM", 0x02006033), ("REMU", 0x02007033),
    ):
        test.equal(f"{name} encoding", encode(name, instruction_set=RV32IM), expected)

    def run(name, a, b):
        cpu = CPU(instruction_set=RV32IM)
        cpu.load_program([encode(name, rd=3, rs1=1, rs2=2, instruction_set=RV32IM)])
        cpu.set_reg(1, a & 0xFFFFFFFF)
        cpu.set_reg(2, b & 0xFFFFFFFF)
        cpu.step()
        return cpu.get_reg(3)

    test.equal("MUL 7*6", run("MUL", 7, 6), 42)
    test.equal("MUL keeps only the low word", run("MUL", 0x10000, 0x10000), 0)
    test.equal("MUL of negatives", run("MUL", -7, 6), 0xFFFFFFD6)
    test.equal("MULH high word", run("MULH", 0x10000, 0x10000), 1)
    test.equal("MULH sign-extends", run("MULH", -1, 1), 0xFFFFFFFF)
    test.equal("MULHU is unsigned", run("MULHU", -1, -1), 0xFFFFFFFE)
    test.equal("MULHSU is mixed", run("MULHSU", -1, -1), 0xFFFFFFFF)
    test.equal("DIV 42/6", run("DIV", 42, 6), 7)
    test.equal("DIV truncates toward zero", run("DIV", -7, 2), 0xFFFFFFFD)
    test.equal("REM sign follows the dividend", run("REM", -7, 2), 0xFFFFFFFF)
    test.equal("DIVU is unsigned", run("DIVU", -1, 2), 0x7FFFFFFF)
    test.equal("REMU is unsigned", run("REMU", -1, 16), 15)
    test.equal("DIV by zero is all ones", run("DIV", 42, 0), 0xFFFFFFFF)
    test.equal("REM by zero is the dividend", run("REM", 42, 0), 42)
    test.equal("DIVU by zero is all ones", run("DIVU", 42, 0), 0xFFFFFFFF)
    test.equal("REMU by zero is the dividend", run("REMU", 42, 0), 42)
    test.equal("DIV overflow wraps", run("DIV", -(1 << 31), -1), 0x80000000)
    test.equal("REM overflow is zero", run("REM", -(1 << 31), -1), 0)

    # The division identity, over the whole sign matrix.
    violations = []
    for a in (-2147483647, -99991, -7, -1, 0, 1, 7, 99991, 2147483647):
        for b in (-977, -2, -1, 1, 2, 977):
            quotient = to_signed(run("DIV", a, b))
            remainder = to_signed(run("REM", a, b))
            if (b * quotient + remainder) & 0xFFFFFFFF != a & 0xFFFFFFFF:
                violations.append(f"{a}/{b}")
    test.check("rs1 == rs2*DIV + REM everywhere", not violations, ", ".join(violations[:5]))

    # An M instruction must still be illegal on a base machine.
    word = encode("MUL", rd=3, rs1=1, rs2=2, instruction_set=RV32IM)
    cpu = CPU()
    cpu.load_program([word])
    test.raises("MUL is illegal on RV32I", IllegalInstruction, cpu.step)


def check_rv32a(test: SelfTest) -> None:
    test.group("RV32A extension")
    from .isa import available_isas, get_isa
    from .rv32a import RV32A_INSTRUCTION_COUNT, RV32IMA
    from .rv32m import RV32IM

    test.equal("RV32IMA adds eleven over RV32IM", len(RV32IMA) - len(RV32IM), RV32A_INSTRUCTION_COUNT)
    test.check("RV32IM does not contain LR.W", "LR.W" not in RV32IM)
    for name in ("rv32ia", "rv32ima", "rv32ia_zicsr", "rv32ima_zicsr"):
        test.check(f"{name} is registered", name in available_isas())
    test.check("get_isa returns the same object", get_isa("rv32ima") is RV32IMA)

    test.equal("LR.W encoding", encode("LR.W", instruction_set=RV32IMA), 0x1000202F)
    test.equal("SC.W encoding", encode("SC.W", instruction_set=RV32IMA), 0x1800202F)
    test.equal("AMOADD.W encoding", encode("AMOADD.W", instruction_set=RV32IMA), 0x0000202F)

    def run_amo(name, initial, operand):
        cpu = CPU(instruction_set=RV32IMA)
        cpu.write_u32(0x1000, initial & 0xFFFFFFFF)
        cpu.load_program([encode(name, rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)])
        cpu.set_reg(1, 0x1000)
        cpu.set_reg(2, operand & 0xFFFFFFFF)
        cpu.step()
        return cpu.get_reg(3), cpu.read_u32(0x1000)

    test.equal("AMOSWAP.W", run_amo("AMOSWAP.W", 11, 22), (11, 22))
    test.equal("AMOADD.W", run_amo("AMOADD.W", 10, 32), (10, 42))
    test.equal("AMOADD.W wraps", run_amo("AMOADD.W", 0xFFFFFFFF, 2), (0xFFFFFFFF, 1))
    test.equal(
        "AMOMINU.W is unsigned (0xFFFFFFFF is largest)",
        run_amo("AMOMINU.W", 0xFFFFFFFF, 3),
        (0xFFFFFFFF, 3),
    )

    # LR/SC: a reservation is consumed by its SC.W, and any trap clears it.
    cpu = CPU(instruction_set=RV32IMA)
    cpu.write_u32(0x1000, 7)
    cpu.load_program(
        [
            encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA),
            encode("SC.W", rd=3, rs1=2, rs2=4, instruction_set=RV32IMA),
            encode("SC.W", rd=3, rs1=2, rs2=5, instruction_set=RV32IMA),
        ]
    )
    cpu.set_reg(2, 0x1000)
    cpu.set_reg(4, 99)
    cpu.set_reg(5, 100)
    cpu.step()
    test.equal("LR.W sets the reservation", cpu.reservation, 0x1000)
    cpu.step()
    test.equal("first SC.W succeeds", cpu.get_reg(3), 0)
    test.equal("first SC.W stores", cpu.read_u32(0x1000), 99)
    cpu.step()
    test.equal("second SC.W fails: the reservation was consumed", cpu.get_reg(3), 1)
    test.equal("failed SC.W does not store", cpu.read_u32(0x1000), 99)

    cpu.reset()
    test.check("reset clears the reservation", cpu.reservation is None)

    # An A instruction must still be illegal on a machine without it.
    word = encode("AMOADD.W", rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)
    cpu = CPU(instruction_set=RV32IM)
    cpu.load_program([word])
    test.raises("AMOADD.W is illegal on RV32IM", IllegalInstruction, cpu.step)


def check_rv32c(test: SelfTest) -> None:
    test.group("RV32C extension")
    from .isa import available_isas, get_isa
    from .rv32c import RV32IC, expand

    test.equal("RV32IC adds no new mnemonics", len(RV32IC), len(RV32I))
    for name in ("rv32ic", "rv32imc", "rv32iac", "rv32imac",
                 "rv32ic_zicsr", "rv32imc_zicsr", "rv32iac_zicsr", "rv32imac_zicsr"):
        test.check(f"{name} is registered", name in available_isas())
    test.check("rv32imac is marked compressed", get_isa("rv32imac").compressed)
    test.check("rv32im is not", not get_isa("rv32im").compressed)

    # A handful of the 192-case toolchain table (see tests.test_rv32c for all of them).
    for half, expected, ctext in (
        (0x0451, 0x01440413, "c.addi s0,20"),
        (0x6315, 0x00005337, "c.lui t1,5"),
        (0x8c09, 0x40A40433, "c.sub s0,a0"),
        (0x4302, 0x00012303, "c.lwsp t1,0(sp)"),
        (0x9002, 0x00100073, "c.ebreak"),
    ):
        word, _name = expand(half)
        test.equal(ctext, word, expected)

    cpu = CPU(instruction_set=RV32IC)
    cpu.load_program([0x0451], addr=0)  # c.addi s0, 20
    cpu.set_reg(8, 100)
    cpu.step()
    test.equal("c.addi s0,20 behaves like addi s0,s0,20", cpu.get_reg(8), 120)
    test.equal("a compressed instruction advances pc by 2", cpu.pc, 2)

    cpu = CPU(instruction_set=RV32IC)
    cpu.load_program([0x0000], addr=0)  # C.ADDI4SPN nzuimm=0: reserved
    test.raises("a reserved compressed encoding traps", IllegalInstruction, cpu.step)

    # A real GCC -march=rv32imac binary, if the toolchain is present, is
    # covered end-to-end by tests.test_real_binaries; this self-test stays
    # toolchain-free, so it stops at the hand-picked table above.


def check_csr_and_traps(test: SelfTest) -> None:
    test.group("CSRs and architectural traps")
    from .csr import (
        MCAUSE, MEPC, MSCRATCH, MSTATUS, MSTATUS_MIE, MTVAL, MTVEC,
        MVENDORID, PRIV_MACHINE, CSRFile,
    )
    from .zicsr import RV32IM_ZICSR

    ISA = RV32IM_ZICSR

    test.equal("RV32I is untouched", len(RV32I), 40)
    test.equal("RV32IM_Zicsr adds seven", len(ISA) - 48, 7)
    for name, expected in (
        ("CSRRW", 0x00001073), ("CSRRS", 0x00002073), ("CSRRC", 0x00003073),
        ("CSRRWI", 0x00005073), ("CSRRSI", 0x00006073), ("CSRRCI", 0x00007073),
        ("MRET", 0x30200073),
    ):
        test.equal(f"{name} encoding", encode(name, instruction_set=ISA), expected)

    # -- CSR field rules -------------------------------------------------
    csr = CSRFile()
    csr.write(MSTATUS, 0xFFFFFFFF)
    # MIE (0x8) | MPIE (0x80) | MPP (0x1800); every other bit is WPRI.
    test.equal("mstatus keeps only implemented fields", csr.get(MSTATUS), 0x1888)
    test.equal("mstatus.MPP is WARL to machine mode", csr.mpp, PRIV_MACHINE)
    csr.write(MEPC, 0x1006)
    test.equal("mepc is 4-byte aligned", csr.get(MEPC), 0x1004)
    csr.write(MTVEC, 0x3002)
    test.equal("reserved mtvec mode falls back to Direct", csr.mtvec_mode, 0)
    test.equal("...keeping the BASE", csr.mtvec_base, 0x3000)
    csr.write(MTVEC, 0x4001)
    test.equal("mtvec vectored mode is accepted", csr.mtvec_mode, 1)
    test.raises("unimplemented CSR raises", IllegalInstruction, csr.read, 0x7C0)
    test.raises("read-only CSR write raises", IllegalInstruction, csr.write, MVENDORID, 1)

    # -- CSR instructions ------------------------------------------------
    def run_words(words, **kwargs):
        cpu = CPU(instruction_set=ISA, **kwargs)
        cpu.load_program(list(words))
        cpu.run(len(words))
        return cpu

    cpu = run_words([
        encode("CSRRWI", rd=1, rs1=31, imm=MSCRATCH, instruction_set=ISA),
        encode("CSRRS", rd=2, rs1=0, imm=MSCRATCH, instruction_set=ISA),
    ])
    test.equal("CSRRWI writes a zero-extended immediate", cpu.csr.get(MSCRATCH), 31)
    test.equal("CSRRS with rs1=x0 reads", cpu.get_reg(2), 31)

    cpu = CPU(instruction_set=ISA)
    cpu.csr.set(MSCRATCH, 0x99)
    cpu.load_program([encode("CSRRS", rd=0, rs1=0, imm=MSCRATCH, instruction_set=ISA)])
    cpu.step()
    test.equal("a CSR read into x0 leaves x0 alone", cpu.get_reg(0), 0)
    test.equal("...and does not disturb the CSR", cpu.csr.get(MSCRATCH), 0x99)

    cpu = CPU()  # no Zicsr
    cpu.load_program([encode("CSRRW", rd=1, rs1=0, imm=MSTATUS, instruction_set=ISA)])
    test.raises("CSR access is illegal without Zicsr", IllegalInstruction, cpu.step)

    # -- architectural trap entry ---------------------------------------
    def trapping(words, mtvec=0x800):
        cpu = CPU(instruction_set=ISA)
        cpu.load_program(list(words))
        cpu.csr.set(MTVEC, mtvec)
        cpu.step()
        return cpu

    cpu = trapping([0x00000000])
    test.equal("illegal instruction cause", cpu.csr.get(MCAUSE), 2)
    test.equal("mepc is the faulting pc", cpu.csr.get(MEPC), 0)
    test.equal("pc is the trap vector", cpu.pc, 0x800)
    test.equal("the faulting instruction does not retire", cpu.instret, 0)
    test.equal("the trap is counted", cpu.traps_taken, 1)

    cpu = trapping([encode("LW", rd=1, rs1=0, imm=2)])
    test.equal("misaligned load cause", cpu.csr.get(MCAUSE), 4)
    test.equal("misaligned load mtval", cpu.csr.get(MTVAL), 2)

    cpu = trapping([encode("SW", rs1=0, rs2=0, imm=1)])
    test.equal("misaligned store cause", cpu.csr.get(MCAUSE), 6)

    cpu = trapping([encode("ECALL")])
    test.equal("ECALL cause", cpu.csr.get(MCAUSE), 11)

    cpu = trapping([encode("EBREAK")])
    test.equal("EBREAK cause", cpu.csr.get(MCAUSE), 3)

    cpu = trapping([encode("JAL", rd=1, imm=2)])
    test.equal("misaligned jump cause", cpu.csr.get(MCAUSE), 0)

    # -- mstatus transitions --------------------------------------------
    cpu = CPU(instruction_set=ISA)
    cpu.load_program([0x00000000])
    cpu.csr.set(MTVEC, 0x800)
    cpu.csr.write(MSTATUS, MSTATUS_MIE)
    cpu.step()
    test.check("MIE is cleared on trap entry", not cpu.csr.mie_enabled)
    test.check("MPIE saves the previous MIE", cpu.csr.mpie)
    cpu.load_program([encode("MRET", instruction_set=ISA)], addr=0x800, set_pc=False)
    cpu.step()
    test.check("MRET restores MIE", cpu.csr.mie_enabled)
    test.check("MRET sets MPIE", cpu.csr.mpie)
    test.equal("MRET returns to mepc", cpu.pc, 0)

    # -- with no handler installed, the host still sees the fault --------
    cpu = CPU(instruction_set=ISA)
    cpu.load_program([0x00000000])
    test.check("no mtvec means no architectural trap", not cpu.traps_architecturally)
    test.raises("...so the host sees the exception", IllegalInstruction, cpu.step)

    # -- the whole flow, executed by the guest ---------------------------
    from .host import halt_on_ecall

    program = assemble("""
        _start:
                la   t0, handler
                csrw mtvec, t0
                li   s0, 0
                .word 0
                li   s1, 0xAB
                li   a7, 93
                li   a0, 0
                ecall
        handler:
                addi s0, s0, 1
                csrr t1, mcause
                csrr t2, mepc
                addi t2, t2, 4
                csrw mepc, t2
                mret
    """, instruction_set=ISA)
    cpu = CPU(instruction_set=ISA, ecall_handler=halt_on_ecall)
    program.load(cpu.memory)
    cpu.set_pc(program.entry)
    cpu.run(500)
    test.check("a guest trap handler runs to completion", cpu.halted)
    test.equal("the guest handler was entered once", cpu.get_reg(8), 1)
    test.equal("MRET resumed the guest", cpu.get_reg(9), 0xAB)


def check_timer(test: SelfTest) -> None:
    test.group("machine timer and interrupts")
    from .csr import (
        IRQ_MACHINE_TIMER, MCAUSE, MEPC, MIE, MIP, MIP_MTIP, MSTATUS, MSTATUS_MIE,
        MTVAL, MTVEC, MTVEC_MODE_VECTORED,
    )
    from .host import halt_on_ecall
    from .timer import (
        CLINT_BASE, MTIMECMP_OFFSET, MTIMECMP_RESET, MTIME_OFFSET, MachineTimer,
    )
    from .zicsr import RV32IM_ZICSR

    ISA = RV32IM_ZICSR
    MTIMECMP_ADDR = CLINT_BASE + MTIMECMP_OFFSET
    NOP = encode("ADDI", rd=0, rs1=0, imm=0)

    # -- the 64-bit registers -------------------------------------------
    timer = MachineTimer()
    test.equal("mtimecmp resets to all ones", timer.mtimecmp, MTIMECMP_RESET)
    test.check("a reset timer is not firing", not timer.mtip)
    timer.mtime = 0x1122_3344_5566_7788
    test.equal("mtime low word", timer.read(MTIME_OFFSET, 4), 0x55667788)
    test.equal("mtime high word", timer.read(MTIME_OFFSET + 4, 4), 0x11223344)
    timer.mtimecmp = 0
    timer.write(MTIMECMP_OFFSET, 4, 0xDEADBEEF)
    timer.write(MTIMECMP_OFFSET + 4, 4, 0x12345678)
    test.equal("two 32-bit writes build a 64-bit value", timer.mtimecmp, 0x12345678DEADBEEF)
    timer.write(MTIMECMP_OFFSET, 4, 0)
    test.equal("a low write leaves the high half alone", timer.mtimecmp, 0x1234567800000000)

    # -- the comparison --------------------------------------------------
    timer.mtimecmp = 100
    timer.set_time(99)
    test.check("99 < 100 does not fire", not timer.mtip)
    timer.advance(1)
    test.check("100 >= 100 fires (the compare is >=)", timer.mtip)
    timer.set_time(1 << 32)
    timer.mtimecmp = 0xFFFFFFFF
    test.check("the compare is 64-bit, not 32-bit", timer.mtip)
    timer.set_time(0xFFFFFFFF)
    timer.mtimecmp = 1 << 32
    test.check("...in both directions", not timer.mtip)
    timer.set_time((1 << 64) - 1)
    timer.advance(1)
    test.equal("mtime wraps at 2**64", timer.mtime, 0)

    # -- determinism -----------------------------------------------------
    cpu = CPU(instruction_set=ISA, timer=MachineTimer())
    cpu.load_program([NOP] * 5)
    cpu.run(5)
    test.equal("mtime advances one tick per step", cpu.timer.mtime, 5)
    frozen = CPU(instruction_set=ISA, timer=MachineTimer(ticks_per_step=0))
    frozen.load_program([NOP] * 5)
    frozen.run(5)
    test.equal("ticks_per_step=0 freezes time", frozen.timer.mtime, 0)
    frozen.advance_time(77)
    test.equal("advance_time moves it explicitly", frozen.timer.mtime, 77)

    # -- MTIP through mip -------------------------------------------------
    cpu = CPU(instruction_set=ISA, timer=MachineTimer())
    cpu.timer.mtimecmp = 3
    test.equal("mip is clear before the deadline", cpu.csr.get(MIP), 0)
    cpu.advance_time(3)
    test.equal("mip reports MTIP live", cpu.csr.get(MIP), MIP_MTIP)
    cpu.csr.write(MIP, 0)
    test.equal("mip ignores guest writes", cpu.csr.get(MIP), MIP_MTIP)
    cpu.timer.mtimecmp = 1 << 40
    test.equal("moving the comparator clears MTIP", cpu.csr.get(MIP), 0)

    # -- gating -----------------------------------------------------------
    def armed(mode=0, mtvec=0x800):
        machine = CPU(instruction_set=ISA, timer=MachineTimer())
        machine.load_program([NOP] * 8)
        machine.csr.set(MTVEC, mtvec | mode)
        machine.timer.mtimecmp = 0
        return machine

    cpu = armed()
    cpu.csr.write(MSTATUS, MSTATUS_MIE)  # MIE without MTIE
    cpu.run(4)
    test.equal("MTIE gates delivery", cpu.interrupts_taken, 0)
    cpu = armed()
    cpu.csr.write(MIE, MIP_MTIP)  # MTIE without MIE
    cpu.run(4)
    test.equal("mstatus.MIE gates delivery", cpu.interrupts_taken, 0)

    # -- delivery ---------------------------------------------------------
    cpu = armed()
    cpu.csr.write(MSTATUS, MSTATUS_MIE)
    cpu.csr.write(MIE, MIP_MTIP)
    cpu.csr.set(MTVAL, 0xDEADBEEF)
    cpu.step()
    test.equal("an enabled, pending timer interrupt is delivered", cpu.interrupts_taken, 1)
    test.equal("mcause has the interrupt bit and cause 7", cpu.csr.get(MCAUSE), 0x80000007)
    test.equal("mcause cause field", cpu.csr.get(MCAUSE) & 0x7FFFFFFF, IRQ_MACHINE_TIMER)
    test.equal("mepc is the interrupted pc", cpu.csr.get(MEPC), 0)
    test.equal("mtval is zero for an interrupt", cpu.csr.get(MTVAL), 0)
    test.equal("control transfers to mtvec", cpu.pc, 0x800)
    test.check("MIE is cleared in the handler", not cpu.csr.mie_enabled)
    test.check("MPIE saved the old MIE", cpu.csr.mpie)
    test.equal("the interrupted instruction did not retire", cpu.instret, 0)

    # -- vectoring --------------------------------------------------------
    cpu = armed(mode=MTVEC_MODE_VECTORED)
    cpu.csr.write(MSTATUS, MSTATUS_MIE)
    cpu.csr.write(MIE, MIP_MTIP)
    cpu.step()
    test.equal("vectored mode offsets the timer by 4*7", cpu.pc, 0x800 + 28)
    cpu = CPU(instruction_set=ISA, timer=MachineTimer())
    cpu.load_program([0x00000000])
    cpu.csr.set(MTVEC, 0x800 | MTVEC_MODE_VECTORED)
    cpu.step()
    test.equal("vectored mode does not offset exceptions", cpu.pc, 0x800)

    # -- the whole loop, run by the guest ---------------------------------
    program = assemble(f"""
        _start:
                la   t0, handler
                csrw mtvec, t0
                li   s0, 0
                li   t1, {MTIMECMP_ADDR}
                li   t2, 30
                sw   t2, 0(t1)
                sw   x0, 4(t1)
                li   t3, 0x80
                csrs mie, t3
                li   t3, 8
                csrs mstatus, t3
        loop:   addi s1, s1, 1
                li   t4, 400
                blt  s1, t4, loop
                li   a7, 93
                li   a0, 0
                ecall
        handler:
                addi s0, s0, 1
                csrr s2, mcause
                csrr s3, mepc
                li   t5, {MTIMECMP_ADDR}
                li   t6, -1
                sw   t6, 0(t5)
                sw   t6, 4(t5)
                mret
    """, instruction_set=ISA)
    cpu = CPU(instruction_set=ISA, timer=MachineTimer(), ecall_handler=halt_on_ecall)
    program.load(cpu.memory)
    cpu.set_pc(program.entry)
    cpu.run(5000)
    test.check("a guest timer handler runs to completion", cpu.halted)
    test.equal("it was entered exactly once", cpu.get_reg(8), 1)
    test.equal("the guest read mcause itself", cpu.get_reg(18), 0x80000007)
    test.check("...and a plausible mepc", cpu.get_reg(19) != 0)
    test.equal("the loop finished after MRET resumed it", cpu.get_reg(9), 400)
    test.equal("resolving MTIP stopped it retriggering", cpu.interrupts_taken, 1)


def run_self_test(verbose: bool = True, stream=None) -> SelfTest:
    """Run every internal check; return the :class:`SelfTest` collector."""
    test = SelfTest(verbose=verbose, stream=stream)
    groups = [
        ("register file", check_registers),
        ("memory", check_memory),
        ("decoder", check_decoder),
        ("arithmetic and logic", check_arithmetic),
        ("branches and jumps", check_control_flow),
        ("loads and stores", check_memory_instructions),
        ("traps", check_traps),
        ("pc and stepping", check_pc_and_stepping),
        ("state, tracing, host API", check_state_and_trace),
        ("end-to-end programs", check_programs),
        ("assembler round-trip", check_assembler_roundtrip),
        ("ELF loading and execution", check_elf),
        ("RV32M extension", check_rv32m),
        ("RV32A extension", check_rv32a),
        ("RV32C extension", check_rv32c),
        ("CSRs and traps", check_csr_and_traps),
        ("machine timer", check_timer),
    ]
    out = stream if stream is not None else sys.stdout
    for title, function in groups:
        before = test.failed
        function(test)
        status = "PASS" if test.failed == before else "FAIL"
        if verbose:
            print(f"[{status}] {title}", file=out)
    if verbose:
        print(file=out)
        print(f"Checks: {test.passed + test.failed}", file=out)
        print(f"Passed: {test.passed}", file=out)
        print(f"Failed: {test.failed}", file=out)
    return test
