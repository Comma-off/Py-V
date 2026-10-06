"""One or more tests for every RV32I instruction.

Each test executes the instruction and checks the resulting architectural state
(registers, PC, memory), not merely that nothing crashed.  Test names are the
mnemonics, so the runner prints ``[PASS] ADD``, ``[PASS] ADDI`` and so on.
"""

import unittest

from tests.helpers import SCRATCH, CPU, EmulatorTestCase, encode, make_cpu, run_asm, run_words

from riscv.exceptions import Breakpoint, EnvironmentCall


class InstructionTestCase(EmulatorTestCase):
    """Helpers for executing a single instruction against a known state."""

    def execute(self, name, *, regs=None, memory=None, pc=0, **fields) -> CPU:
        """Run one instruction with the given starting state; return the CPU."""
        cpu = CPU()
        cpu.load_program([encode(name, **fields)], addr=pc)
        cpu.set_pc(pc)
        for index, value in (regs or {}).items():
            cpu.set_reg(index, value)
        for address, value in (memory or {}).items():
            cpu.write_u32(address, value)
        cpu.step()
        return cpu


# ---------------------------------------------------------------------------
# Register-register ALU
# ---------------------------------------------------------------------------
class TestRegisterArithmetic(InstructionTestCase):
    def test_add(self):
        """ADD"""
        cpu = self.execute("ADD", rd=3, rs1=1, rs2=2, regs={1: 10, 2: 20})
        self.assertReg(cpu, 3, 30)
        self.assertPC(cpu, 4)

    def test_add_wraps(self):
        """ADD wraps at 32 bits"""
        cpu = self.execute("ADD", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 1})
        self.assertReg(cpu, 3, 0)
        cpu = self.execute("ADD", rd=3, rs1=1, rs2=2, regs={1: 0x7FFFFFFF, 2: 1})
        self.assertReg(cpu, 3, 0x80000000, "signed overflow wraps silently")

    def test_sub(self):
        """SUB"""
        cpu = self.execute("SUB", rd=3, rs1=1, rs2=2, regs={1: 20, 2: 10})
        self.assertReg(cpu, 3, 10)
        cpu = self.execute("SUB", rd=3, rs1=1, rs2=2, regs={1: 10, 2: 20})
        self.assertReg(cpu, 3, 0xFFFFFFF6, "10 - 20 == -10")
        cpu = self.execute("SUB", rd=3, rs1=1, rs2=2, regs={1: 0, 2: 1})
        self.assertReg(cpu, 3, 0xFFFFFFFF, "underflow wraps")

    def test_sll(self):
        """SLL"""
        cpu = self.execute("SLL", rd=3, rs1=1, rs2=2, regs={1: 1, 2: 31})
        self.assertReg(cpu, 3, 0x80000000)
        cpu = self.execute("SLL", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 4})
        self.assertReg(cpu, 3, 0xFFFFFFF0, "bits shifted out are discarded")

    def test_sll_uses_five_bits_of_shift(self):
        """SLL uses only the low 5 bits of rs2"""
        cpu = self.execute("SLL", rd=3, rs1=1, rs2=2, regs={1: 1, 2: 32})
        self.assertReg(cpu, 3, 1, "shift of 32 behaves as a shift of 0")
        cpu = self.execute("SLL", rd=3, rs1=1, rs2=2, regs={1: 1, 2: 0xFFFFFFFF})
        self.assertReg(cpu, 3, 0x80000000, "shift of -1 behaves as a shift of 31")

    def test_slt(self):
        """SLT"""
        cpu = self.execute("SLT", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 1})
        self.assertReg(cpu, 3, 1, "-1 < 1 signed")
        cpu = self.execute("SLT", rd=3, rs1=1, rs2=2, regs={1: 1, 2: 0xFFFFFFFF})
        self.assertReg(cpu, 3, 0)
        cpu = self.execute("SLT", rd=3, rs1=1, rs2=2, regs={1: 5, 2: 5})
        self.assertReg(cpu, 3, 0, "equal values are not less-than")

    def test_sltu(self):
        """SLTU"""
        cpu = self.execute("SLTU", rd=3, rs1=1, rs2=2, regs={1: 1, 2: 0xFFFFFFFF})
        self.assertReg(cpu, 3, 1, "1 < 0xFFFFFFFF unsigned")
        cpu = self.execute("SLTU", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 1})
        self.assertReg(cpu, 3, 0, "0xFFFFFFFF is not less than 1 unsigned")

    def test_xor(self):
        """XOR"""
        cpu = self.execute("XOR", rd=3, rs1=1, rs2=2, regs={1: 0xFF00FF00, 2: 0x0FF00FF0})
        self.assertReg(cpu, 3, 0xF0F0F0F0)

    def test_srl(self):
        """SRL"""
        cpu = self.execute("SRL", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 28})
        self.assertReg(cpu, 3, 0x0000000F, "SRL is a logical shift: zeros come in")
        cpu = self.execute("SRL", rd=3, rs1=1, rs2=2, regs={1: 0x80000000, 2: 31})
        self.assertReg(cpu, 3, 1)

    def test_sra(self):
        """SRA"""
        cpu = self.execute("SRA", rd=3, rs1=1, rs2=2, regs={1: 0xFFFFFFFF, 2: 28})
        self.assertReg(cpu, 3, 0xFFFFFFFF, "SRA is arithmetic: the sign is replicated")
        cpu = self.execute("SRA", rd=3, rs1=1, rs2=2, regs={1: 0x80000000, 2: 31})
        self.assertReg(cpu, 3, 0xFFFFFFFF)
        cpu = self.execute("SRA", rd=3, rs1=1, rs2=2, regs={1: 0x40000000, 2: 4})
        self.assertReg(cpu, 3, 0x04000000, "a positive value shifts in zeros")

    def test_or(self):
        """OR"""
        cpu = self.execute("OR", rd=3, rs1=1, rs2=2, regs={1: 0xFF00, 2: 0x00FF})
        self.assertReg(cpu, 3, 0xFFFF)

    def test_and(self):
        """AND"""
        cpu = self.execute("AND", rd=3, rs1=1, rs2=2, regs={1: 0xFF0F, 2: 0x0FF0})
        self.assertReg(cpu, 3, 0x0F00)

    def test_same_register_as_source_and_destination(self):
        """R-type: rd may alias rs1/rs2"""
        cpu = self.execute("ADD", rd=1, rs1=1, rs2=1, regs={1: 21})
        self.assertReg(cpu, 1, 42)
        cpu = self.execute("SUB", rd=1, rs1=1, rs2=1, regs={1: 12345})
        self.assertReg(cpu, 1, 0)


# ---------------------------------------------------------------------------
# Register-immediate ALU
# ---------------------------------------------------------------------------
class TestImmediateArithmetic(InstructionTestCase):
    def test_addi(self):
        """ADDI"""
        cpu = self.execute("ADDI", rd=1, rs1=0, imm=10)
        self.assertReg(cpu, 1, 10)
        self.assertPC(cpu, 4)

    def test_addi_negative(self):
        """ADDI with a negative immediate"""
        cpu = self.execute("ADDI", rd=1, rs1=2, imm=-10, regs={2: 5})
        self.assertReg(cpu, 1, 0xFFFFFFFB, "5 + (-10) == -5")

    def test_addi_boundaries(self):
        """ADDI at the immediate range boundaries"""
        cpu = self.execute("ADDI", rd=1, rs1=0, imm=2047)
        self.assertReg(cpu, 1, 2047)
        cpu = self.execute("ADDI", rd=1, rs1=0, imm=-2048)
        self.assertReg(cpu, 1, 0xFFFFF800)

    def test_addi_wraps(self):
        """ADDI wraps at 32 bits"""
        cpu = self.execute("ADDI", rd=1, rs1=2, imm=1, regs={2: 0xFFFFFFFF})
        self.assertReg(cpu, 1, 0)

    def test_slti(self):
        """SLTI"""
        cpu = self.execute("SLTI", rd=1, rs1=2, imm=-4, regs={2: 0xFFFFFFFB})  # -5
        self.assertReg(cpu, 1, 1, "-5 < -4 signed")
        cpu = self.execute("SLTI", rd=1, rs1=2, imm=-5, regs={2: 0xFFFFFFFB})
        self.assertReg(cpu, 1, 0)

    def test_sltiu(self):
        """SLTIU"""
        cpu = self.execute("SLTIU", rd=1, rs1=2, imm=1, regs={2: 0})
        self.assertReg(cpu, 1, 1, "sltiu rd, rs1, 1 is the seqz idiom")
        cpu = self.execute("SLTIU", rd=1, rs1=2, imm=1, regs={2: 5})
        self.assertReg(cpu, 1, 0)
        cpu = self.execute("SLTIU", rd=1, rs1=2, imm=-1, regs={2: 5})
        self.assertReg(cpu, 1, 1, "the immediate sign-extends, then compares unsigned")

    def test_xori(self):
        """XORI"""
        cpu = self.execute("XORI", rd=1, rs1=2, imm=-1, regs={2: 0x0F0F0F0F})
        self.assertReg(cpu, 1, 0xF0F0F0F0, "xori rd, rs, -1 is the not idiom")

    def test_ori(self):
        """ORI"""
        cpu = self.execute("ORI", rd=1, rs1=2, imm=0x0F0, regs={2: 0xF00})
        self.assertReg(cpu, 1, 0xFF0)

    def test_andi(self):
        """ANDI"""
        cpu = self.execute("ANDI", rd=1, rs1=2, imm=0x0FF, regs={2: 0xABCD})
        self.assertReg(cpu, 1, 0xCD)

    def test_andi_sign_extends(self):
        """ANDI sign-extends its immediate"""
        cpu = self.execute("ANDI", rd=1, rs1=2, imm=-256, regs={2: 0xFFFFFFFF})
        self.assertReg(cpu, 1, 0xFFFFFF00, "-256 becomes 0xFFFFFF00, not 0x00000F00")

    def test_slli(self):
        """SLLI"""
        cpu = self.execute("SLLI", rd=1, rs1=2, imm=4, regs={2: 3})
        self.assertReg(cpu, 1, 48)
        cpu = self.execute("SLLI", rd=1, rs1=2, imm=31, regs={2: 1})
        self.assertReg(cpu, 1, 0x80000000)
        cpu = self.execute("SLLI", rd=1, rs1=2, imm=0, regs={2: 0xABCD})
        self.assertReg(cpu, 1, 0xABCD, "a zero shift is a move")

    def test_srli(self):
        """SRLI"""
        cpu = self.execute("SRLI", rd=1, rs1=2, imm=4, regs={2: 0xFFFFFFF0})
        self.assertReg(cpu, 1, 0x0FFFFFFF, "logical: zeros shift in")

    def test_srai(self):
        """SRAI"""
        cpu = self.execute("SRAI", rd=1, rs1=2, imm=4, regs={2: 0xFFFFFFF0})
        self.assertReg(cpu, 1, 0xFFFFFFFF, "arithmetic: the sign shifts in")
        cpu = self.execute("SRAI", rd=1, rs1=2, imm=31, regs={2: 0x7FFFFFFF})
        self.assertReg(cpu, 1, 0)


# ---------------------------------------------------------------------------
# Upper-immediate
# ---------------------------------------------------------------------------
class TestUpperImmediate(InstructionTestCase):
    def test_lui(self):
        """LUI"""
        cpu = self.execute("LUI", rd=1, imm=0x12345)
        self.assertReg(cpu, 1, 0x12345000)

    def test_lui_overwrites_low_bits(self):
        """LUI clears the low 12 bits"""
        cpu = self.execute("LUI", rd=1, imm=0, regs={1: 0xFFFFFFFF})
        self.assertReg(cpu, 1, 0)

    def test_lui_top_bit(self):
        """LUI with the sign bit set"""
        cpu = self.execute("LUI", rd=1, imm=0xFFFFF)
        self.assertReg(cpu, 1, 0xFFFFF000)

    def test_auipc(self):
        """AUIPC"""
        cpu = self.execute("AUIPC", rd=1, imm=0, pc=0x1000)
        self.assertReg(cpu, 1, 0x1000, "AUIPC adds to the address of the instruction")
        cpu = self.execute("AUIPC", rd=1, imm=1, pc=0x1000)
        self.assertReg(cpu, 1, 0x2000)

    def test_auipc_negative_offset(self):
        """AUIPC with a negative upper immediate"""
        cpu = self.execute("AUIPC", rd=1, imm=0xFFFFF, pc=0x2000)
        self.assertReg(cpu, 1, 0x1000, "0x2000 + 0xFFFFF000 wraps to 0x1000")


# ---------------------------------------------------------------------------
# Jumps
# ---------------------------------------------------------------------------
class TestJumps(InstructionTestCase):
    def test_jal(self):
        """JAL"""
        cpu = self.execute("JAL", rd=1, imm=16, pc=0x100)
        self.assertPC(cpu, 0x110)
        self.assertReg(cpu, 1, 0x104, "rd = pc + 4")

    def test_jal_backwards(self):
        """JAL with a negative offset"""
        cpu = self.execute("JAL", rd=1, imm=-16, pc=0x100)
        self.assertPC(cpu, 0xF0)

    def test_jal_to_x0_is_a_plain_jump(self):
        """JAL with rd=x0 discards the link"""
        cpu = self.execute("JAL", rd=0, imm=8, pc=0x100)
        self.assertPC(cpu, 0x108)
        self.assertReg(cpu, 0, 0)

    def test_jalr(self):
        """JALR"""
        cpu = self.execute("JALR", rd=1, rs1=2, imm=4, pc=0x100, regs={2: 0x200})
        self.assertPC(cpu, 0x204)
        self.assertReg(cpu, 1, 0x104)

    def test_jalr_clears_the_low_bit(self):
        """JALR clears bit 0 of its target"""
        cpu = self.execute("JALR", rd=1, rs1=2, imm=0, pc=0x100, regs={2: 0x205})
        self.assertPC(cpu, 0x204, "the low bit is cleared after adding the immediate")

    def test_jalr_negative_offset(self):
        """JALR with a negative immediate"""
        cpu = self.execute("JALR", rd=1, rs1=2, imm=-4, pc=0x100, regs={2: 0x200})
        self.assertPC(cpu, 0x1FC)

    def test_jalr_link_before_jump(self):
        """JALR may use the same register for rd and rs1"""
        cpu = self.execute("JALR", rd=1, rs1=1, imm=0, pc=0x100, regs={1: 0x200})
        self.assertPC(cpu, 0x200, "the target is computed before rd is written")
        self.assertReg(cpu, 1, 0x104)


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------
class TestBranches(InstructionTestCase):
    def assertBranch(self, name, rs1_value, rs2_value, taken, message=""):
        cpu = self.execute(name, rs1=1, rs2=2, imm=16, pc=0x100, regs={1: rs1_value, 2: rs2_value})
        expected = 0x110 if taken else 0x104
        self.assertPC(cpu, expected)

    def test_beq(self):
        """BEQ"""
        self.assertBranch("BEQ", 5, 5, taken=True)
        self.assertBranch("BEQ", 5, 6, taken=False)
        self.assertBranch("BEQ", 0xFFFFFFFF, 0xFFFFFFFF, taken=True)

    def test_bne(self):
        """BNE"""
        self.assertBranch("BNE", 5, 6, taken=True)
        self.assertBranch("BNE", 5, 5, taken=False)

    def test_blt(self):
        """BLT"""
        self.assertBranch("BLT", 0xFFFFFFFF, 1, taken=True)   # -1 < 1
        self.assertBranch("BLT", 1, 0xFFFFFFFF, taken=False)
        self.assertBranch("BLT", 5, 5, taken=False)
        self.assertBranch("BLT", 0x80000000, 0x7FFFFFFF, taken=True)  # INT_MIN < INT_MAX

    def test_bge(self):
        """BGE"""
        self.assertBranch("BGE", 5, 5, taken=True)
        self.assertBranch("BGE", 1, 0xFFFFFFFF, taken=True)
        self.assertBranch("BGE", 0xFFFFFFFF, 1, taken=False)
        self.assertBranch("BGE", 0x7FFFFFFF, 0x80000000, taken=True)

    def test_bltu(self):
        """BLTU"""
        self.assertBranch("BLTU", 1, 0xFFFFFFFF, taken=True)
        self.assertBranch("BLTU", 0xFFFFFFFF, 1, taken=False)
        self.assertBranch("BLTU", 0, 1, taken=True)
        self.assertBranch("BLTU", 5, 5, taken=False)

    def test_bgeu(self):
        """BGEU"""
        self.assertBranch("BGEU", 0xFFFFFFFF, 1, taken=True)
        self.assertBranch("BGEU", 1, 0xFFFFFFFF, taken=False)
        self.assertBranch("BGEU", 5, 5, taken=True)

    def test_branch_does_not_write_registers(self):
        """branches write no registers"""
        cpu = self.execute("BEQ", rs1=1, rs2=2, imm=16, regs={1: 5, 2: 5})
        self.assertEqual([value for value in cpu.regs][3:], [0] * 29)

    def test_signed_and_unsigned_branches_differ(self):
        """BLT and BLTU disagree on negative values"""
        self.assertBranch("BLT", 0xFFFFFFFF, 1, taken=True)
        self.assertBranch("BLTU", 0xFFFFFFFF, 1, taken=False)


# ---------------------------------------------------------------------------
# Loads and stores
# ---------------------------------------------------------------------------
class TestLoadsAndStores(InstructionTestCase):
    def test_lw_sw(self):
        """LW/SW"""
        cpu = run_asm(
            f"li x1, {SCRATCH}\nli x2, 0x12345678\nsw x2, 0(x1)\nlw x3, 0(x1)"
        )
        self.assertReg(cpu, 3, 0x12345678)
        self.assertWord(cpu, SCRATCH, 0x12345678)

    def test_lw(self):
        """LW"""
        cpu = self.execute("LW", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0xAABBCCDD})
        self.assertReg(cpu, 1, 0xAABBCCDD)

    def test_lw_with_offset(self):
        """LW with positive and negative offsets"""
        cpu = self.execute("LW", rd=1, rs1=2, imm=8, regs={2: SCRATCH}, memory={SCRATCH + 8: 0x1111})
        self.assertReg(cpu, 1, 0x1111)
        cpu = self.execute("LW", rd=1, rs1=2, imm=-8, regs={2: SCRATCH}, memory={SCRATCH - 8: 0x2222})
        self.assertReg(cpu, 1, 0x2222)

    def test_lb(self):
        """LB"""
        cpu = self.execute("LB", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x0000007F})
        self.assertReg(cpu, 1, 0x7F)

    def test_lb_sign_extends(self):
        """LB sign-extends"""
        cpu = self.execute("LB", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x000000FF})
        self.assertReg(cpu, 1, 0xFFFFFFFF)
        cpu = self.execute("LB", rd=1, rs1=2, imm=1, regs={2: SCRATCH}, memory={SCRATCH: 0x0000FF00})
        self.assertReg(cpu, 1, 0xFFFFFFFF)

    def test_lbu(self):
        """LBU"""
        cpu = self.execute("LBU", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x000000FF})
        self.assertReg(cpu, 1, 0x000000FF, "LBU zero-extends")

    def test_lh(self):
        """LH"""
        cpu = self.execute("LH", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x00007FFF})
        self.assertReg(cpu, 1, 0x7FFF)

    def test_lh_sign_extends(self):
        """LH sign-extends"""
        cpu = self.execute("LH", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x00008000})
        self.assertReg(cpu, 1, 0xFFFF8000)

    def test_lhu(self):
        """LHU"""
        cpu = self.execute("LHU", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x00008000})
        self.assertReg(cpu, 1, 0x00008000, "LHU zero-extends")

    def test_sb(self):
        """SB"""
        cpu = self.execute("SB", rs1=2, rs2=3, imm=0, regs={2: SCRATCH, 3: 0x12345678})
        self.assertEqual(cpu.read_u8(SCRATCH), 0x78, "SB stores only the low byte")
        self.assertEqual(cpu.read_u8(SCRATCH + 1), 0, "neighbouring bytes are untouched")

    def test_sh(self):
        """SH"""
        cpu = self.execute("SH", rs1=2, rs2=3, imm=0, regs={2: SCRATCH, 3: 0x12345678})
        self.assertEqual(cpu.read_u16(SCRATCH), 0x5678)
        self.assertEqual(cpu.read_u16(SCRATCH + 2), 0)

    def test_sw(self):
        """SW"""
        cpu = self.execute("SW", rs1=2, rs2=3, imm=0, regs={2: SCRATCH, 3: 0x12345678})
        self.assertWord(cpu, SCRATCH, 0x12345678)

    def test_store_offsets(self):
        """stores honour positive and negative offsets"""
        cpu = self.execute("SW", rs1=2, rs2=3, imm=16, regs={2: SCRATCH, 3: 0xAAAA})
        self.assertWord(cpu, SCRATCH + 16, 0xAAAA)
        cpu = self.execute("SW", rs1=2, rs2=3, imm=-16, regs={2: SCRATCH, 3: 0xBBBB})
        self.assertWord(cpu, SCRATCH - 16, 0xBBBB)

    def test_little_endian_layout(self):
        """stores use little-endian byte order"""
        cpu = self.execute("SW", rs1=2, rs2=3, imm=0, regs={2: SCRATCH, 3: 0x11223344})
        self.assertEqual(cpu.read_bytes(SCRATCH, 4), bytes([0x44, 0x33, 0x22, 0x11]))

    def test_load_store_all_widths_roundtrip(self):
        """byte/half/word stores and loads round-trip"""
        cpu = run_asm(
            f"""
            li x1, {SCRATCH}
            li x2, 0xFF
            sb x2, 0(x1)
            li x2, 0xFFFF
            sh x2, 2(x1)
            li x2, 0xFFFFFFFF
            sw x2, 4(x1)
            lbu x3, 0(x1)
            lhu x4, 2(x1)
            lw  x5, 4(x1)
            """
        )
        self.assertReg(cpu, 3, 0xFF)
        self.assertReg(cpu, 4, 0xFFFF)
        self.assertReg(cpu, 5, 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# MISC-MEM and SYSTEM
# ---------------------------------------------------------------------------
class TestSystemInstructions(InstructionTestCase):
    def test_fence(self):
        """FENCE"""
        cpu = make_cpu([encode("FENCE", imm=0xFF)])
        cpu.step()
        self.assertPC(cpu, 4)
        self.assertEqual(cpu.instret, 1)
        self.assertEqual(cpu.fences, 1, "FENCE is a no-op but is counted")

    def test_ecall(self):
        """ECALL"""
        cpu = make_cpu([encode("ECALL")])
        with self.assertRaises(EnvironmentCall):
            cpu.step()
        self.assertPC(cpu, 0, "the PC stays on the ECALL")

    def test_ecall_with_handler(self):
        """ECALL with a host handler"""
        calls = []
        cpu = make_cpu([encode("ECALL")], ecall_handler=lambda c: calls.append(c.get_reg(17)))
        cpu.set_reg(17, 93)
        cpu.step()
        self.assertEqual(calls, [93])
        self.assertPC(cpu, 4, "execution continues after a handled ECALL")

    def test_ebreak(self):
        """EBREAK"""
        cpu = make_cpu([encode("EBREAK")])
        with self.assertRaises(Breakpoint):
            cpu.step()


# ---------------------------------------------------------------------------
# Cross-cutting architectural rules
# ---------------------------------------------------------------------------
class TestArchitecturalRules(InstructionTestCase):
    def test_x0_immutability(self):
        """x0 immutability"""
        for name, fields in [
            ("ADDI", dict(rd=0, rs1=1, imm=100)),
            ("ADD", dict(rd=0, rs1=1, rs2=1)),
            ("LUI", dict(rd=0, imm=0xFFFFF)),
            ("AUIPC", dict(rd=0, imm=1)),
            ("JAL", dict(rd=0, imm=4)),
            ("SLLI", dict(rd=0, rs1=1, imm=4)),
        ]:
            cpu = self.execute(name, regs={1: 0xFFFFFFFF}, **fields)
            self.assertReg(cpu, 0, 0, f"{name} must not write x0")

    def test_x0_reads_as_zero_as_a_source(self):
        """x0 reads as zero when used as a source"""
        cpu = self.execute("ADD", rd=1, rs1=0, rs2=0)
        self.assertReg(cpu, 1, 0)
        cpu = self.execute("ADDI", rd=1, rs1=0, imm=42)
        self.assertReg(cpu, 1, 42)

    def test_sign_extension(self):
        """Sign extension"""
        # Immediates
        cpu = self.execute("ADDI", rd=1, rs1=0, imm=-1)
        self.assertReg(cpu, 1, 0xFFFFFFFF, "I-immediates sign-extend")
        # Loads
        cpu = self.execute("LB", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x80})
        self.assertReg(cpu, 1, 0xFFFFFF80, "LB sign-extends")
        cpu = self.execute("LH", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x8000})
        self.assertReg(cpu, 1, 0xFFFF8000, "LH sign-extends")
        # ...but not the unsigned forms
        cpu = self.execute("LBU", rd=1, rs1=2, imm=0, regs={2: SCRATCH}, memory={SCRATCH: 0x80})
        self.assertReg(cpu, 1, 0x80, "LBU does not sign-extend")

    def test_32_bit_wrapping(self):
        """32-bit overflow wraps silently"""
        cases = [
            ("ADD", dict(rd=3, rs1=1, rs2=2), {1: 0x7FFFFFFF, 2: 0x7FFFFFFF}, 0xFFFFFFFE),
            ("ADD", dict(rd=3, rs1=1, rs2=2), {1: 0x80000000, 2: 0x80000000}, 0x00000000),
            ("SUB", dict(rd=3, rs1=1, rs2=2), {1: 0x80000000, 2: 1}, 0x7FFFFFFF),
            ("SLL", dict(rd=3, rs1=1, rs2=2), {1: 0xFFFFFFFF, 2: 31}, 0x80000000),
        ]
        for name, fields, regs, expected in cases:
            cpu = self.execute(name, regs=regs, **fields)
            self.assertReg(cpu, 3, expected, name)

    def test_every_instruction_advances_or_redirects_the_pc(self):
        """every instruction leaves the PC well defined"""
        from riscv.instructions import RV32I

        skip = {"ECALL", "EBREAK"}  # these trap without a handler
        for definition in RV32I:
            if definition.name in skip:
                continue
            cpu = make_cpu([definition.match], allow_misaligned=True)
            cpu.step()
            self.assertEqual(cpu.pc % 4, 0, f"{definition.name} left a misaligned PC")
            self.assertEqual(cpu.instret, 1, definition.name)


if __name__ == "__main__":
    unittest.main()
