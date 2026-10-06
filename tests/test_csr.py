"""CSR instructions, machine CSR state, architectural trap entry and MRET."""

import unittest

from tests.helpers import CPU, EmulatorTestCase, assemble

from riscv.csr import (
    MCAUSE,
    MEPC,
    MIE,
    MIP,
    MISA,
    MSCRATCH,
    MSTATUS,
    MSTATUS_MIE,
    MSTATUS_MPIE,
    MSTATUS_MPP,
    MSTATUS_MPP_SHIFT,
    MTVAL,
    MTVEC,
    MTVEC_MODE_DIRECT,
    MTVEC_MODE_VECTORED,
    MVENDORID,
    PRIV_MACHINE,
    CSRFile,
    csr_names,
)
from riscv.encoder import encode
from riscv.exceptions import IllegalInstruction
from riscv.zicsr import (
    PRIVILEGED_INSTRUCTIONS,
    RV32I_ZICSR,
    RV32IM_ZICSR,
    ZICSR_INSTRUCTIONS,
)

ISA = RV32IM_ZICSR


def csr_cpu(source=None, words=None, **kwargs):
    """A CPU with Zicsr, loaded with either assembly or raw words."""
    kwargs.setdefault("instruction_set", ISA)
    cpu = CPU(**kwargs)
    if source is not None:
        cpu.load_program(assemble(source, instruction_set=ISA).words)
    elif words is not None:
        cpu.load_program(list(words))
    return cpu


def run_source(source, **kwargs):
    """Assemble, load and execute exactly as many steps as there are words.

    Counting steps by hand is a trap: ``li`` with a large constant expands to
    two instructions, so a hand-written budget silently truncates the program.
    """
    program = assemble(source, instruction_set=ISA)
    cpu = csr_cpu(words=program.words, **kwargs)
    cpu.run(len(program.words))
    return cpu


# ---------------------------------------------------------------------------
# Encoding / decoding / disassembly
# ---------------------------------------------------------------------------
class TestEncoding(unittest.TestCase):
    #: Base encodings from the specification, plus real GNU `as` output.
    REFERENCE = {
        "CSRRW": 0x00001073, "CSRRS": 0x00002073, "CSRRC": 0x00003073,
        "CSRRWI": 0x00005073, "CSRRSI": 0x00006073, "CSRRCI": 0x00007073,
        "MRET": 0x30200073,
    }
    #: Full instructions as emitted by riscv32-linux-gnu-as.
    TOOLCHAIN = [
        ("csrrw t0, mstatus, t1", 0x300312F3),
        ("csrrs t0, mtvec, x0", 0x305022F3),
        ("csrrc a0, mie, a1", 0x3045B573),
        ("csrrwi t0, mepc, 5", 0x3412D2F3),
        ("csrrsi a0, mstatus, 8", 0x30046573),
        ("csrrci a0, mstatus, 31", 0x300FF573),
        ("csrr t2, mcause", 0x342023F3),
        ("csrw mtvec, t0", 0x30529073),
        ("csrs mstatus, t0", 0x3002A073),
        ("csrc mstatus, t0", 0x3002B073),
        ("csrwi mscratch, 1", 0x3400D073),
        ("csrr a0, mhartid", 0xF1402573),
        ("csrr a0, misa", 0x30102573),
        ("mret", 0x30200073),
    ]

    def test_base_encodings(self):
        """csr: base encodings match the specification"""
        for name, expected in self.REFERENCE.items():
            self.assertEqual(encode(name, instruction_set=ISA), expected, name)

    def test_toolchain_encodings(self):
        """csr: 14 encodings match riscv32-linux-gnu-as output"""
        for text, expected in self.TOOLCHAIN:
            word = assemble(text, instruction_set=ISA).words[0]
            self.assertEqual(word, expected, f"{text}: got 0x{word:08X}")

    def test_csr_address_is_unsigned(self):
        """csr: a CSR address above 0x7FF is not sign-extended"""
        # mvendorid is 0xF11; a sign-extending I-type decode would give
        # 0xFFFFFF11 and look for a CSR that does not exist.
        from riscv.decoder import Decoder

        inst = Decoder(ISA).decode(0xF1402573)
        self.assertEqual(inst.name, "CSRRS")
        self.assertEqual(inst.csr, 0xF14)
        # inst.imm is the raw decoded immediate.  An I-type decode would
        # sign-extend it to 0xFFFFFF14; the CSR format must not.
        self.assertEqual(inst.imm, 0xF14, "the CSR address must not be sign-extended")
        self.assertEqual(inst.imm_signed, 0xF14, "and it is unsigned, so this matches")

    def test_disassembly(self):
        """csr: instructions disassemble with symbolic CSR names"""
        from riscv.disasm import disassemble

        cases = {
            0x300312F3: "CSRRW t0,mstatus,t1",
            0x3412D2F3: "CSRRWI t0,mepc,5",
            0x342023F3: "CSRRS t2,mcause,zero",
            0x30200073: "MRET",
        }
        for word, expected in cases.items():
            self.assertEqual(disassemble(word, instruction_set=ISA, abi=True), expected)

    def test_unknown_csr_disassembles_as_hex(self):
        """csr: an unimplemented CSR number prints as hex"""
        from riscv.disasm import disassemble

        word = encode("CSRRS", rd=1, rs1=0, imm=0x7C0, instruction_set=ISA)
        self.assertIn("0x7C0", disassemble(word, instruction_set=ISA))

    def test_round_trip(self):
        """csr: disassembly reassembles to the same word"""
        from riscv.disasm import disassemble

        for text, word in self.TOOLCHAIN:
            rebuilt = assemble(
                disassemble(word, instruction_set=ISA), instruction_set=ISA
            ).words[0]
            self.assertEqual(rebuilt, word, text)

    def test_instruction_counts(self):
        """csr: Zicsr adds six CSR instructions plus MRET"""
        from riscv.instructions import RV32I
        from riscv.rv32m import RV32IM

        self.assertEqual(len(RV32I), 40, "the base set must not change")
        self.assertEqual(len(RV32IM), 48, "RV32IM must not change")
        self.assertEqual(len(RV32I_ZICSR), 47)
        self.assertEqual(len(RV32IM_ZICSR), 55)
        for name in ZICSR_INSTRUCTIONS + PRIVILEGED_INSTRUCTIONS:
            self.assertIn(name, RV32IM_ZICSR)
            self.assertNotIn(name, RV32I, "the base set must stay pure")

    def test_illegal_without_zicsr(self):
        """csr: CSR instructions trap on a machine without Zicsr"""
        word = encode("CSRRW", rd=1, rs1=2, imm=MSTATUS, instruction_set=ISA)
        cpu = CPU()  # plain RV32I
        cpu.load_program([word])
        with self.assertRaises(IllegalInstruction):
            cpu.step()

    def test_mret_illegal_without_zicsr(self):
        """csr: MRET traps on a machine without the privileged instructions"""
        cpu = CPU()
        cpu.load_program([0x30200073])
        with self.assertRaises(IllegalInstruction):
            cpu.step()


# ---------------------------------------------------------------------------
# CSR instruction semantics
# ---------------------------------------------------------------------------
class TestCsrInstructions(EmulatorTestCase):
    def test_csrrw(self):
        """CSRRW"""
        cpu = run_source("li t0, 0x1234\ncsrrw t1, mscratch, t0\ncsrrw t2, mscratch, x0")
        self.assertReg(cpu, 6, 0, "first read returns the old value (0)")
        self.assertReg(cpu, 7, 0x1234, "second read returns what was written")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0, "the second write stored x0")

    def test_csrrw_writes_even_when_rs1_is_x0(self):
        """CSRRW writes unconditionally, even from x0"""
        cpu = run_source("li t0, 0xFF\ncsrrw x0, mscratch, t0")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0xFF)
        cpu.load_program([encode("CSRRW", rd=0, rs1=0, imm=MSCRATCH, instruction_set=ISA)])
        cpu.step()
        self.assertEqual(cpu.csr.get(MSCRATCH), 0, "writing zero is a real write")

    def test_csrrs(self):
        """CSRRS"""
        cpu = run_source("li t0, 0x0F\ncsrrs t1, mscratch, t0\nli t0, 0xF0\ncsrrs t2, mscratch, t0")
        self.assertReg(cpu, 6, 0)
        self.assertReg(cpu, 7, 0x0F, "reads the value before this set")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0xFF, "bits accumulate")

    def test_csrrc(self):
        """CSRRC"""
        cpu = run_source(
            "li t0, 0xFF\ncsrrw x0, mscratch, t0\nli t0, 0x0F\ncsrrc t1, mscratch, t0"
        )
        self.assertReg(cpu, 6, 0xFF)
        self.assertEqual(cpu.csr.get(MSCRATCH), 0xF0)

    def test_csrrwi(self):
        """CSRRWI"""
        cpu = run_source("csrrwi t0, mscratch, 31\ncsrrwi t1, mscratch, 0")
        self.assertReg(cpu, 5, 0)
        self.assertReg(cpu, 6, 31, "the immediate is zero-extended, not sign-extended")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0)

    def test_csrrsi_and_csrrci(self):
        """CSRRSI / CSRRCI"""
        cpu = run_source("csrrsi x0, mscratch, 31\ncsrrci t0, mscratch, 15")
        self.assertReg(cpu, 5, 31)
        self.assertEqual(cpu.csr.get(MSCRATCH), 16)

    def test_rd_x0_does_not_read(self):
        """csr: CSRRW with rd=x0 must not read the CSR"""
        cpu = csr_cpu()
        reads = []
        original = cpu.csr.read

        def watched(number, *args, **kwargs):
            reads.append(number)
            return original(number, *args, **kwargs)

        cpu.csr.read = watched
        cpu.load_program([encode("CSRRW", rd=0, rs1=1, imm=MSCRATCH, instruction_set=ISA)])
        cpu.set_reg(1, 0x55)
        cpu.step()
        self.assertEqual(reads, [], "rd=x0 must suppress the read (and its side effects)")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0x55, "but the write still happens")

    def test_rs1_x0_does_not_write(self):
        """csr: CSRRS with rs1=x0 must not write the CSR"""
        cpu = csr_cpu()
        writes = []
        original = cpu.csr.write

        def watched(number, value, *args, **kwargs):
            writes.append(number)
            return original(number, value, *args, **kwargs)

        cpu.csr.write = watched
        cpu.csr.set(MSCRATCH, 0x99)
        cpu.load_program([encode("CSRRS", rd=1, rs1=0, imm=MSCRATCH, instruction_set=ISA)])
        cpu.step()
        self.assertEqual(writes, [], "rs1=x0 must suppress the write entirely")
        self.assertReg(cpu, 1, 0x99)

    def test_zero_immediate_does_not_write(self):
        """csr: CSRRSI/CSRRCI with a zero immediate must not write"""
        cpu = csr_cpu()
        cpu.csr.set(MSCRATCH, 0x77)
        for name in ("CSRRSI", "CSRRCI"):
            cpu.load_program([encode(name, rd=1, rs1=0, imm=MSCRATCH, instruction_set=ISA)])
            cpu.set_pc(0)
            cpu.step()
            self.assertReg(cpu, 1, 0x77, name)
            self.assertEqual(cpu.csr.get(MSCRATCH), 0x77, f"{name} must not write")

    def test_read_only_csr_via_csrrs_x0_is_legal(self):
        """csr: reading a read-only CSR with csrr is legal"""
        # csrr rd, mvendorid is csrrs rd, mvendorid, x0 -- no write, so no trap.
        cpu = run_source("csrr t0, mvendorid\ncsrr t1, misa\ncsrr t2, mhartid")
        self.assertReg(cpu, 5, 0)
        self.assertNotEqual(cpu.get_reg(6), 0, "misa must describe the machine")

    def test_writing_a_read_only_csr_traps(self):
        """csr: writing a read-only CSR raises an illegal instruction"""
        cpu = csr_cpu()
        cpu.load_program([encode("CSRRW", rd=0, rs1=1, imm=MVENDORID, instruction_set=ISA)])
        with self.assertRaises(IllegalInstruction) as context:
            cpu.step()
        self.assertIn("read-only", str(context.exception))

    def test_unimplemented_csr_traps(self):
        """csr: accessing an unimplemented CSR raises an illegal instruction"""
        for number in (0x000, 0x7C0, 0xBFF, 0x100, 0x304 + 1):
            if number in csr_names().values():
                continue
            cpu = csr_cpu()
            cpu.load_program([encode("CSRRS", rd=1, rs1=0, imm=number, instruction_set=ISA)])
            with self.assertRaises(IllegalInstruction) as context:
                cpu.step()
            self.assertIn("not implemented", str(context.exception))

    def test_csr_access_does_not_write_x0(self):
        """csr: a CSR read into x0 leaves x0 alone"""
        cpu = csr_cpu()
        cpu.csr.set(MSCRATCH, 0xFFFFFFFF)
        cpu.load_program([encode("CSRRS", rd=0, rs1=0, imm=MSCRATCH, instruction_set=ISA)])
        cpu.step()
        self.assertReg(cpu, 0, 0)

    def test_values_are_32_bit(self):
        """csr: CSR values wrap to 32 bits"""
        cpu = run_source("li t0, -1\ncsrrw x0, mscratch, t0\ncsrr t1, mscratch")
        self.assertEqual(cpu.csr.get(MSCRATCH), 0xFFFFFFFF)
        self.assertReg(cpu, 6, 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# CSR state and field rules
# ---------------------------------------------------------------------------
class TestCsrState(EmulatorTestCase):
    def setUp(self):
        self.csr = CSRFile()

    def test_reset_values(self):
        """csr state: reset values"""
        self.assertEqual(self.csr.get(MTVEC), 0)
        self.assertEqual(self.csr.get(MEPC), 0)
        self.assertEqual(self.csr.get(MCAUSE), 0)
        self.assertEqual(self.csr.get(MTVAL), 0)
        self.assertEqual(self.csr.get(MIE), 0)
        self.assertEqual(self.csr.get(MIP), 0)
        self.assertEqual(self.csr.mpp, PRIV_MACHINE)
        self.assertFalse(self.csr.mie_enabled)

    def test_mstatus_wpri_bits_are_ignored(self):
        """csr state: mstatus keeps only the implemented fields"""
        self.csr.write(MSTATUS, 0xFFFFFFFF)
        value = self.csr.get(MSTATUS)
        self.assertTrue(value & MSTATUS_MIE)
        self.assertTrue(value & MSTATUS_MPIE)
        self.assertEqual(value & ~(MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP), 0,
                         "reserved bits must read as zero")

    def test_mstatus_mpp_is_warl(self):
        """csr state: mstatus.MPP only ever reads as machine mode"""
        for attempt in (0, 1, 2, 3):
            self.csr.write(MSTATUS, attempt << MSTATUS_MPP_SHIFT)
            self.assertEqual(self.csr.mpp, PRIV_MACHINE,
                             "only machine mode is implemented, so MPP is WARL to 3")

    def test_mtvec_modes(self):
        """csr state: mtvec BASE and MODE"""
        self.csr.write(MTVEC, 0x1000)
        self.assertEqual(self.csr.mtvec_base, 0x1000)
        self.assertEqual(self.csr.mtvec_mode, MTVEC_MODE_DIRECT)
        self.csr.write(MTVEC, 0x2001)
        self.assertEqual(self.csr.mtvec_base, 0x2000)
        self.assertEqual(self.csr.mtvec_mode, MTVEC_MODE_VECTORED)

    def test_mtvec_reserved_mode_is_rejected(self):
        """csr state: a reserved mtvec MODE falls back to Direct"""
        for reserved in (2, 3):
            self.csr.write(MTVEC, 0x3000 | reserved)
            self.assertEqual(self.csr.mtvec_mode, MTVEC_MODE_DIRECT,
                             "modes 2 and 3 are reserved")
            self.assertEqual(self.csr.mtvec_base, 0x3000, "the BASE is kept")

    def test_mepc_is_always_aligned(self):
        """csr state: mepc's low two bits are always zero (IALIGN=32)"""
        self.csr.write(MEPC, 0x1006)
        self.assertEqual(self.csr.get(MEPC), 0x1004)
        self.csr.write(MEPC, 0xFFFFFFFF)
        self.assertEqual(self.csr.get(MEPC), 0xFFFFFFFC)

    def test_mie_holds_only_the_defined_bits(self):
        """csr state: mie keeps only MSIE/MTIE/MEIE"""
        self.csr.write(MIE, 0xFFFFFFFF)
        self.assertEqual(self.csr.get(MIE), 0x888)

    def test_mip_is_read_only_here(self):
        """csr state: mip ignores writes (there are no interrupt sources yet)"""
        self.csr.write(MIP, 0xFFFFFFFF)
        self.assertEqual(self.csr.get(MIP), 0)

    def test_mscratch_and_mtval_are_plain_registers(self):
        """csr state: mscratch and mtval hold any 32-bit value"""
        for number in (MSCRATCH, MTVAL, MCAUSE):
            self.csr.write(number, 0xDEADBEEF)
            self.assertEqual(self.csr.get(number), 0xDEADBEEF)

    def test_misa_describes_the_machine(self):
        """csr state: misa reports MXL=32 and the extension letters"""
        from riscv.rv32m import RV32IM

        value = CSRFile(RV32IM).get(MISA)
        self.assertEqual(value >> 30, 1, "MXL=1 means XLEN 32")
        self.assertTrue(value & (1 << (ord("i") - ord("a"))), "I must be set")
        self.assertTrue(value & (1 << (ord("m") - ord("a"))), "M must be set")
        self.assertFalse(CSRFile().get(MISA) & (1 << (ord("m") - ord("a"))))

    def test_misa_ignores_writes(self):
        """csr state: misa is read-only in this implementation"""
        before = self.csr.get(MISA)
        self.csr.write(MISA, 0)
        self.assertEqual(self.csr.get(MISA), before)

    def test_unknown_csr_raises(self):
        """csr state: reading or writing an unimplemented CSR raises"""
        with self.assertRaises(IllegalInstruction):
            self.csr.read(0x7C0)
        with self.assertRaises(IllegalInstruction):
            self.csr.write(0x7C0, 1)

    def test_reset(self):
        """csr state: reset() restores every register"""
        self.csr.write(MTVEC, 0x8000)
        self.csr.write(MSCRATCH, 0x1234)
        self.csr.reset()
        self.assertEqual(self.csr.get(MTVEC), 0)
        self.assertEqual(self.csr.get(MSCRATCH), 0)
        self.assertFalse(self.csr.trap_vector_installed)

    def test_cpu_reset_clears_csrs(self):
        """csr state: cpu.reset() resets the CSR file too"""
        cpu = csr_cpu("csrrwi x0, mtvec, 16")
        cpu.step()
        self.assertEqual(cpu.csr.get(MTVEC), 16)
        cpu.reset()
        self.assertEqual(cpu.csr.get(MTVEC), 0)

    def test_state_snapshot_round_trip(self):
        """csr state: CSRs survive state()/restore_state()"""
        cpu = csr_cpu()
        cpu.csr.set(MTVEC, 0x4000)
        cpu.csr.set(MSCRATCH, 0xABCD)
        clone = csr_cpu()
        clone.restore_state(cpu.state())
        self.assertEqual(clone.csr.get(MTVEC), 0x4000)
        self.assertEqual(clone.csr.get(MSCRATCH), 0xABCD)


# ---------------------------------------------------------------------------
# Architectural trap entry
# ---------------------------------------------------------------------------
class TestTrapEntry(EmulatorTestCase):
    HANDLER = 0x800

    def machine(self, words, mtvec=HANDLER, mode=0):
        cpu = csr_cpu(words=words)
        cpu.csr.set(MTVEC, mtvec | mode)
        return cpu

    def assertTrap(self, cpu, cause, tval, faulting_pc):
        self.assertEqual(cpu.csr.get(MCAUSE), cause, "mcause")
        self.assertEqual(cpu.csr.get(MTVAL), tval, "mtval")
        self.assertEqual(cpu.csr.get(MEPC), faulting_pc, "mepc")
        self.assertPC(cpu, self.HANDLER, "pc must be the trap vector")

    def test_illegal_instruction(self):
        """trap: illegal instruction -> cause 2"""
        cpu = self.machine([0x00000000])
        cpu.step()
        self.assertTrap(cpu, 2, 0x00000000, 0)

    def test_illegal_instruction_tval_is_the_word(self):
        """trap: mtval holds the offending instruction word"""
        cpu = self.machine([0x00000000, 0xFFFFFFFF])
        cpu.set_pc(4)
        cpu.step()
        self.assertEqual(cpu.csr.get(MTVAL), 0xFFFFFFFF)

    def test_instruction_address_misaligned(self):
        """trap: misaligned jump target -> cause 0"""
        cpu = self.machine([encode("JAL", rd=1, imm=2)])
        cpu.step()
        self.assertTrap(cpu, 0, 2, 0)

    def test_instruction_access_fault(self):
        """trap: fetching outside memory -> cause 1"""
        cpu = csr_cpu(words=[0x13], memory_size=4096)
        cpu.csr.set(MTVEC, self.HANDLER)
        cpu.set_pc(0x8000)
        cpu.step()
        self.assertEqual(cpu.csr.get(MCAUSE), 1)
        self.assertEqual(cpu.csr.get(MEPC), 0x8000)
        self.assertPC(cpu, self.HANDLER)

    def test_load_address_misaligned(self):
        """trap: misaligned load -> cause 4"""
        cpu = self.machine([encode("LW", rd=1, rs1=0, imm=2)])
        cpu.step()
        self.assertTrap(cpu, 4, 2, 0)

    def test_store_address_misaligned(self):
        """trap: misaligned store -> cause 6"""
        cpu = self.machine([encode("SW", rs1=0, rs2=0, imm=1)])
        cpu.step()
        self.assertTrap(cpu, 6, 1, 0)

    def test_load_access_fault(self):
        """trap: load outside memory -> cause 5"""
        cpu = csr_cpu(words=[encode("LW", rd=2, rs1=1, imm=0)], memory_size=4096)
        cpu.csr.set(MTVEC, self.HANDLER)
        cpu.set_reg(1, 0x8000)
        cpu.step()
        self.assertEqual(cpu.csr.get(MCAUSE), 5)
        self.assertEqual(cpu.csr.get(MTVAL), 0x8000)

    def test_store_access_fault(self):
        """trap: store outside memory -> cause 7"""
        cpu = csr_cpu(words=[encode("SW", rs1=1, rs2=0, imm=0)], memory_size=4096)
        cpu.csr.set(MTVEC, self.HANDLER)
        cpu.set_reg(1, 0x8000)
        cpu.step()
        self.assertEqual(cpu.csr.get(MCAUSE), 7)

    def test_ecall(self):
        """trap: ECALL -> cause 11"""
        cpu = self.machine([encode("ECALL")])
        cpu.step()
        self.assertTrap(cpu, 11, 0, 0)

    def test_ebreak(self):
        """trap: EBREAK -> cause 3"""
        cpu = self.machine([encode("EBREAK")])
        cpu.step()
        self.assertEqual(cpu.csr.get(MCAUSE), 3)
        self.assertPC(cpu, self.HANDLER)

    def test_faulting_instruction_does_not_retire(self):
        """trap: the faulting instruction is not retired"""
        # An illegal instruction faults during decode...
        cpu = self.machine([0x00000000])
        cpu.step()
        self.assertEqual(cpu.instret, 0, "instret must not count a trapping instruction")
        self.assertEqual(cpu.traps_taken, 1)

    def test_execution_time_fault_does_not_retire(self):
        """trap: a fault raised while executing does not retire either"""
        # ...but a misaligned load decodes fine and faults during execution,
        # which is a separate path through step().
        nop = encode("ADDI", rd=0, rs1=0, imm=0)
        for words, cause in (
            ([nop, encode("LW", rd=1, rs1=0, imm=2)], 4),
            ([nop, encode("SW", rs1=0, rs2=0, imm=1)], 6),
            ([nop, encode("ECALL")], 11),
        ):
            cpu = self.machine(words)
            cpu.run(2)
            self.assertEqual(cpu.csr.get(MCAUSE), cause)
            self.assertEqual(cpu.instret, 1, "only the NOP retired")
            self.assertEqual(cpu.traps_taken, 1)

    def test_mepc_points_at_the_faulting_instruction(self):
        """trap: mepc is the faulting PC, not the next one"""
        nop = encode("ADDI", rd=0, rs1=0, imm=0)
        cpu = self.machine([nop, nop, 0x00000000])
        cpu.run(3)
        self.assertEqual(cpu.csr.get(MEPC), 8)

    def test_mstatus_transitions_on_entry(self):
        """trap: mstatus.MPIE <- MIE, MIE <- 0, MPP <- privilege"""
        cpu = self.machine([0x00000000])
        cpu.csr.write(MSTATUS, MSTATUS_MIE)
        self.assertTrue(cpu.csr.mie_enabled)
        cpu.step()
        self.assertFalse(cpu.csr.mie_enabled, "interrupts disabled inside the handler")
        self.assertTrue(cpu.csr.mpie, "the previous MIE is saved in MPIE")
        self.assertEqual(cpu.csr.mpp, PRIV_MACHINE)

    def test_mpie_records_a_disabled_mie(self):
        """trap: MPIE reflects MIE even when it was clear"""
        cpu = self.machine([0x00000000])
        cpu.csr.write(MSTATUS, 0)
        cpu.step()
        self.assertFalse(cpu.csr.mpie)
        self.assertFalse(cpu.csr.mie_enabled)

    def test_direct_mode_sends_every_cause_to_base(self):
        """trap: in Direct mode all exceptions vector to BASE"""
        for words, in (([0x00000000],), ([encode("ECALL")],)):
            cpu = self.machine(words, mode=MTVEC_MODE_DIRECT)
            cpu.step()
            self.assertPC(cpu, self.HANDLER)

    def test_vectored_mode_still_sends_exceptions_to_base(self):
        """trap: in Vectored mode synchronous exceptions still go to BASE"""
        # Only interrupts are vectored; getting this backwards is a classic bug.
        cpu = self.machine([encode("ECALL")], mode=MTVEC_MODE_VECTORED)
        cpu.step()
        self.assertPC(cpu, self.HANDLER, "an exception must not be offset by its cause")

    def test_vectored_mode_offsets_interrupts(self):
        """trap: in Vectored mode an interrupt vectors to BASE + 4*cause"""
        cpu = self.machine([0x13], mode=MTVEC_MODE_VECTORED)
        self.assertEqual(cpu.csr.trap_target(7, interrupt=True), self.HANDLER + 28)
        self.assertEqual(cpu.csr.trap_target(7, interrupt=False), self.HANDLER)

    def test_interrupt_bit_in_mcause(self):
        """trap: mcause bit 31 distinguishes interrupts from exceptions"""
        cpu = csr_cpu()
        cpu.csr.enter_trap(0x100, cause=7, tval=0, interrupt=True)
        self.assertEqual(cpu.csr.get(MCAUSE), 0x80000007)
        cpu.csr.enter_trap(0x100, cause=7, tval=0, interrupt=False)
        self.assertEqual(cpu.csr.get(MCAUSE), 7)


# ---------------------------------------------------------------------------
# Trap handling policy
# ---------------------------------------------------------------------------
class TestTrapPolicy(EmulatorTestCase):
    def test_without_mtvec_the_host_sees_the_exception(self):
        """trap policy: with no handler installed, the trap reaches the host"""
        cpu = csr_cpu(words=[0x00000000])
        self.assertFalse(cpu.traps_architecturally)
        with self.assertRaises(IllegalInstruction):
            cpu.step()

    def test_writing_mtvec_enables_architectural_traps(self):
        """trap policy: writing mtvec switches to architectural trap entry"""
        cpu = csr_cpu("csrrwi x0, mtvec, 16")
        self.assertFalse(cpu.traps_architecturally)
        cpu.step()
        self.assertTrue(cpu.traps_architecturally)

    def test_explicit_host_mode(self):
        """trap policy: trap_handling='host' always raises"""
        cpu = csr_cpu(words=[0x00000000], trap_handling="host")
        cpu.csr.set(MTVEC, 0x800)
        with self.assertRaises(IllegalInstruction):
            cpu.step()

    def test_explicit_architectural_mode(self):
        """trap policy: trap_handling='architectural' always vectors"""
        cpu = csr_cpu(words=[0x00000000], trap_handling="architectural")
        cpu.step()
        self.assertPC(cpu, 0, "mtvec is still zero, so the vector is address 0")
        self.assertEqual(cpu.csr.get(MCAUSE), 2)

    def test_host_ecall_handler_takes_precedence(self):
        """trap policy: an installed ECALL handler intercepts before the trap"""
        seen = []
        cpu = csr_cpu(words=[encode("ECALL")], ecall_handler=lambda c: seen.append(c.pc))
        cpu.csr.set(MTVEC, 0x800)
        cpu.step()
        self.assertEqual(seen, [0], "the host handler runs")
        self.assertEqual(cpu.traps_taken, 0, "and no architectural trap is taken")

    def test_last_exception_is_recorded_either_way(self):
        """trap policy: the CPU records the exception even when it vectors"""
        cpu = csr_cpu(words=[0x00000000])
        cpu.csr.set(MTVEC, 0x800)
        cpu.step()
        self.assertIsInstance(cpu.last_exception, IllegalInstruction)

    def test_step_returns_none_on_a_trap(self):
        """trap policy: step() returns None when it takes a trap"""
        cpu = csr_cpu(words=[0x00000000])
        cpu.csr.set(MTVEC, 0x800)
        self.assertIsNone(cpu.step())


# ---------------------------------------------------------------------------
# MRET
# ---------------------------------------------------------------------------
class TestMret(EmulatorTestCase):
    def test_mret_returns_to_mepc(self):
        """MRET returns to mepc"""
        cpu = csr_cpu(words=[encode("MRET", instruction_set=ISA)])
        cpu.csr.set(MEPC, 0x1234)
        cpu.step()
        self.assertPC(cpu, 0x1234)

    def test_mret_restores_interrupt_state(self):
        """MRET: MIE <- MPIE, MPIE <- 1"""
        cpu = csr_cpu(words=[encode("MRET", instruction_set=ISA)])
        cpu.csr.write(MSTATUS, MSTATUS_MPIE)  # MPIE set, MIE clear
        self.assertFalse(cpu.csr.mie_enabled)
        cpu.step()
        self.assertTrue(cpu.csr.mie_enabled, "MIE is restored from MPIE")
        self.assertTrue(cpu.csr.mpie, "MPIE is set to 1")

    def test_mret_with_mpie_clear(self):
        """MRET: a clear MPIE leaves interrupts disabled"""
        cpu = csr_cpu(words=[encode("MRET", instruction_set=ISA)])
        cpu.csr.write(MSTATUS, MSTATUS_MIE)  # MIE set, MPIE clear
        cpu.step()
        self.assertFalse(cpu.csr.mie_enabled)
        self.assertTrue(cpu.csr.mpie)

    def test_mret_restores_privilege(self):
        """MRET: privilege is restored from MPP"""
        cpu = csr_cpu(words=[encode("MRET", instruction_set=ISA)])
        cpu.step()
        self.assertEqual(cpu.csr.privilege, PRIV_MACHINE)
        self.assertEqual(cpu.csr.mpp, PRIV_MACHINE, "MPP is reset after MRET")

    def test_trap_then_mret_is_a_round_trip(self):
        """MRET: mstatus returns to its pre-trap state"""
        cpu = csr_cpu(words=[0x00000000])
        cpu.csr.set(MTVEC, 0x800)
        cpu.csr.write(MSTATUS, MSTATUS_MIE)
        cpu.step()                       # take the trap
        self.assertFalse(cpu.csr.mie_enabled, "interrupts off inside the handler")
        cpu.load_program([encode("MRET", instruction_set=ISA)], addr=0x800, set_pc=False)
        cpu.step()                       # and return from it
        self.assertTrue(cpu.csr.mie_enabled, "MIE comes back from MPIE")
        self.assertEqual(cpu.csr.mpp, PRIV_MACHINE)
        # MPIE is *not* restored to its old value: the specification says MRET
        # sets it to 1, so mstatus is deliberately not bit-identical afterwards.
        self.assertTrue(cpu.csr.mpie, "MRET sets MPIE to 1")
        self.assertPC(cpu, 0, "execution resumes at the faulting instruction")


# ---------------------------------------------------------------------------
# The complete flow, executed by the guest
# ---------------------------------------------------------------------------
class TestGuestTrapHandler(EmulatorTestCase):
    """A trap handler written in RISC-V assembly, run on the emulated CPU.

    No Python callback takes part in the trap: the emulator enters the handler
    architecturally, the guest's own code reads the CSRs, adjusts ``mepc`` and
    executes ``MRET``.  The host ECALL handler is installed only so the program
    can exit -- an installed handler intercepts ECALL before it becomes a trap,
    so the guest handler still only ever sees real faults.

    Register map: s0=x8 entries, s1=x9 mcause, s2=x18 mepc, s3=x19 mtval,
    s4=x20 mstatus, s5=x21 progress, s11=x27 completion marker.
    """

    SOURCE = """
        _start:
                la      t0, handler
                csrw    mtvec, t0
                li      s0, 0                   # handler entry count
                li      s5, 0                   # progress marker

                .word   0                       # illegal instruction (cause 2)
                addi    s5, s5, 1

                li      t0, 0x1001
                lw      t1, 0(t0)               # misaligned load (cause 4)
                addi    s5, s5, 1

                li      s11, 0xD0               # reached the end
                li      a7, 93
                li      a0, 0
                ecall                           # host syscall: stop here

        handler:
                addi    s0, s0, 1
                csrr    s1, mcause
                csrr    s2, mepc
                csrr    s3, mtval
                csrr    s4, mstatus
                addi    s2, s2, 4               # step over the faulting insn
                csrw    mepc, s2
                mret
    """

    def run_guest(self, source=None, steps=500):
        from riscv.host import halt_on_ecall

        cpu = csr_cpu(source if source is not None else self.SOURCE,
                      ecall_handler=halt_on_ecall)
        cpu.run(steps)
        return cpu

    def test_handler_runs_on_the_emulated_cpu(self):
        """guest trap handler: the handler executes as guest code"""
        cpu = self.run_guest()
        self.assertTrue(cpu.halted, "the program must reach its exit ECALL")
        self.assertReg(cpu, 8, 2, "the handler ran twice")
        self.assertEqual(cpu.traps_taken, 2)

    def test_execution_resumes_after_each_fault(self):
        """guest trap handler: MRET resumes the interrupted code"""
        cpu = self.run_guest()
        self.assertReg(cpu, 21, 2, "both post-fault instructions ran")
        self.assertReg(cpu, 27, 0xD0, "the program reached its end")

    def test_handler_observes_the_trap_csrs(self):
        """guest trap handler: the guest reads mcause/mepc/mtval itself"""
        cpu = self.run_guest()
        self.assertReg(cpu, 9, 4, "last cause seen was the misaligned load")
        self.assertReg(cpu, 19, 0x1001, "mtval was the misaligned address")
        # mepc was read then advanced by 4, so s2 is the resume address.
        self.assertEqual(cpu.get_reg(18) % 4, 0, "mepc is always aligned")

    def test_handler_sees_machine_mode_in_mstatus(self):
        """guest trap handler: mstatus.MPP reads as machine mode in the handler"""
        cpu = self.run_guest()
        self.assertEqual((cpu.get_reg(20) & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT,
                         PRIV_MACHINE)
        self.assertFalse(cpu.get_reg(20) & MSTATUS_MIE,
                         "interrupts are disabled while the handler runs")

    def test_the_faulting_instruction_would_repeat_without_the_fixup(self):
        """guest trap handler: mepc points at the fault, so the handler must advance it"""
        source = self.SOURCE.replace("addi    s2, s2, 4", "addi    s2, s2, 0")
        cpu = self.run_guest(source, steps=300)
        self.assertFalse(cpu.halted, "the program can never reach its exit")
        self.assertGreater(cpu.traps_taken, 10,
                           "not advancing mepc re-executes the fault forever")
        self.assertReg(cpu, 21, 0, "so the program makes no progress at all")

    def test_mtvec_can_be_moved_at_run_time(self):
        """guest trap handler: a second handler can replace the first"""
        source = """
        _start:
                la      t0, first
                csrw    mtvec, t0
                .word   0                       # -> first handler
                la      t0, second
                csrw    mtvec, t0
                .word   0                       # -> second handler
                li      a7, 93
                li      a0, 0
                ecall
        first:
                li      s0, 0x11
                csrr    t1, mepc
                addi    t1, t1, 4
                csrw    mepc, t1
                mret
        second:
                li      s1, 0x22
                csrr    t1, mepc
                addi    t1, t1, 4
                csrw    mepc, t1
                mret
        """
        cpu = self.run_guest(source)
        self.assertTrue(cpu.halted)
        self.assertReg(cpu, 8, 0x11, "the first handler ran")
        self.assertReg(cpu, 9, 0x22, "and then the second one did")


if __name__ == "__main__":
    unittest.main()
