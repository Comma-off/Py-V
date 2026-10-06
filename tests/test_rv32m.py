"""RV32M tests: multiply, divide, remainder, and the ISA-selection machinery.

The division cases get the most attention because they are where a hand-written
RV32M implementation goes wrong: RISC-V truncates toward zero while Python's
``//`` floors, and divide-by-zero/overflow are defined results rather than
traps.
"""

import random
import unittest

from tests.helpers import CPU, EmulatorTestCase, assemble

from riscv.decoder import Decoder
from riscv.encoder import encode
from riscv.exceptions import AssemblerError, IllegalInstruction
from riscv.instructions import RV32I
from riscv.isa import available_isas, get_isa
from riscv.rv32m import (
    INT32_MIN,
    RV32IM,
    RV32M_INSTRUCTION_COUNT,
    RV32M_INSTRUCTIONS,
    divide_signed,
    divide_unsigned,
    remainder_signed,
    remainder_unsigned,
)

MASK32 = 0xFFFFFFFF
UINT32_MAX = MASK32


def signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


class RV32MTestCase(EmulatorTestCase):
    """Executes a single RV32M instruction against known operands."""

    def execute(self, name, rs1_value, rs2_value):
        cpu = CPU(instruction_set=RV32IM)
        cpu.load_program([encode(name, rd=3, rs1=1, rs2=2, instruction_set=RV32IM)])
        cpu.set_reg(1, rs1_value & MASK32)
        cpu.set_reg(2, rs2_value & MASK32)
        cpu.step()
        return cpu.get_reg(3)

    def assertOp(self, name, a, b, expected, note=""):
        got = self.execute(name, a, b)
        if got != (expected & MASK32):
            self.fail(
                f"{name}(0x{a & MASK32:08X}, 0x{b & MASK32:08X}) = 0x{got:08X}, "
                f"expected 0x{expected & MASK32:08X}" + (f" ({note})" if note else "")
            )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
class TestEncodings(unittest.TestCase):
    #: Base encodings straight from the RISC-V spec (opcode OP, funct7 0000001).
    REFERENCE = {
        "MUL": 0x02000033, "MULH": 0x02001033, "MULHSU": 0x02002033, "MULHU": 0x02003033,
        "DIV": 0x02004033, "DIVU": 0x02005033, "REM": 0x02006033, "REMU": 0x02007033,
    }

    def test_reference_encodings(self):
        """rv32m: base encodings match the specification"""
        for name, expected in self.REFERENCE.items():
            word = encode(name, rd=0, rs1=0, rs2=0, instruction_set=RV32IM)
            self.assertEqual(word, expected, f"{name}: got 0x{word:08X}")

    def test_operand_placement(self):
        """rv32m: register fields land in the right places"""
        # mul a0, a1, a2 as emitted by a real assembler.
        self.assertEqual(encode("MUL", rd=10, rs1=11, rs2=12, instruction_set=RV32IM), 0x02C58533)
        # div ra, sp, gp
        self.assertEqual(encode("DIV", rd=1, rs1=2, rs2=3, instruction_set=RV32IM), 0x023140B3)
        # remu a6, a7, t6
        self.assertEqual(encode("REMU", rd=16, rs1=17, rs2=31, instruction_set=RV32IM), 0x03F8F833)

    def test_all_eight_are_present(self):
        """rv32m: the extension adds exactly eight instructions"""
        self.assertEqual(len(RV32IM) - len(RV32I), RV32M_INSTRUCTION_COUNT)
        for name in RV32M_INSTRUCTIONS:
            self.assertIn(name, RV32IM)

    def test_decode_round_trip(self):
        """rv32m: every instruction encodes and decodes with random operands"""
        decoder = Decoder(RV32IM)
        rng = random.Random(31337)
        for name in RV32M_INSTRUCTIONS:
            for _ in range(20):
                rd, rs1, rs2 = (rng.randrange(32) for _ in range(3))
                word = encode(name, rd=rd, rs1=rs1, rs2=rs2, instruction_set=RV32IM)
                decoded = decoder.decode(word)
                self.assertEqual(decoded.name, name)
                self.assertEqual((decoded.rd, decoded.rs1, decoded.rs2), (rd, rs1, rs2))

    def test_disassembly(self):
        """rv32m: instructions disassemble in R-type form"""
        from riscv.disasm import disassemble

        word = encode("MUL", rd=10, rs1=11, rs2=12, instruction_set=RV32IM)
        self.assertEqual(disassemble(word, instruction_set=RV32IM), "MUL  x10,x11,x12")
        self.assertEqual(disassemble(word, instruction_set=RV32IM, abi=True), "MUL  a0,a1,a2")
        word = encode("MULHSU", rd=1, rs1=2, rs2=3, instruction_set=RV32IM)
        self.assertEqual(disassemble(word, instruction_set=RV32IM), "MULHSU x1,x2,x3")

    def test_disassembly_round_trip(self):
        """rv32m: disassembly reassembles to the same word"""
        from riscv.disasm import disassemble

        for name in RV32M_INSTRUCTIONS:
            word = encode(name, rd=5, rs1=6, rs2=7, instruction_set=RV32IM)
            text = disassemble(word, instruction_set=RV32IM)
            rebuilt = assemble(text, instruction_set=RV32IM).words[0]
            self.assertEqual(rebuilt, word, text)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------
class TestMultiply(RV32MTestCase):
    def test_mul(self):
        """MUL"""
        self.assertOp("MUL", 7, 6, 42)
        self.assertOp("MUL", 0, 12345, 0)
        self.assertOp("MUL", 1, 0xDEADBEEF, 0xDEADBEEF)
        self.assertOp("MUL", -7, 6, -42)
        self.assertOp("MUL", -7, -6, 42)

    def test_mul_keeps_only_the_low_word(self):
        """MUL discards the high half"""
        self.assertOp("MUL", 0x10000, 0x10000, 0x00000000, "2^32 truncates to zero")
        self.assertOp("MUL", -1, -1, 1)
        self.assertOp("MUL", 0xFFFFFFFF, 2, 0xFFFFFFFE)
        self.assertOp("MUL", INT32_MIN, INT32_MIN, 0)

    def test_mul_is_signedness_agnostic(self):
        """MUL gives the same low word whether operands are read signed or not"""
        for a, b in ((-3, 5), (0xFFFFFFFD, 5), (-1, -1), (0xFFFFFFFF, 0xFFFFFFFF)):
            self.assertOp("MUL", a, b, (a * b) & MASK32)

    def test_mulh(self):
        """MULH"""
        self.assertOp("MULH", 0x10000, 0x10000, 1, "2^32 >> 32 == 1")
        self.assertOp("MULH", -1, 1, 0xFFFFFFFF, "sign-extends into the high word")
        self.assertOp("MULH", -1, -1, 0, "1 has no high bits")
        self.assertOp("MULH", INT32_MIN, INT32_MIN, 0x40000000)
        self.assertOp("MULH", 1, 1, 0)
        self.assertOp("MULH", 0x7FFFFFFF, 0x7FFFFFFF, 0x3FFFFFFF)

    def test_mulhu(self):
        """MULHU"""
        self.assertOp("MULHU", 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE)
        self.assertOp("MULHU", 0x10000, 0x10000, 1)
        self.assertOp("MULHU", 0xFFFFFFFF, 1, 0, "0xFFFFFFFF unsigned times 1 fits")
        self.assertOp("MULHU", INT32_MIN, 2, 1)

    def test_mulhsu(self):
        """MULHSU"""
        self.assertOp("MULHSU", -1, 0xFFFFFFFF, 0xFFFFFFFF, "signed x unsigned")
        self.assertOp("MULHSU", -1, 1, 0xFFFFFFFF)
        self.assertOp("MULHSU", 1, 0xFFFFFFFF, 0, "positive x large fits below 2^32")
        self.assertOp("MULHSU", INT32_MIN, 2, 0xFFFFFFFF)

    def test_the_three_high_multiplies_differ(self):
        """MULH/MULHU/MULHSU disagree on the same bit pattern, as they must"""
        a = b = 0xFFFFFFFF  # -1 signed, 4294967295 unsigned
        self.assertOp("MULH", a, b, 0x00000000)
        self.assertOp("MULHU", a, b, 0xFFFFFFFE)
        self.assertOp("MULHSU", a, b, 0xFFFFFFFF)

    def test_high_low_pair_reconstructs_the_full_product(self):
        """MUL and MULH together give the exact 64-bit signed product"""
        rng = random.Random(4242)
        for _ in range(300):
            a = rng.randrange(-(1 << 31), 1 << 31)
            b = rng.randrange(-(1 << 31), 1 << 31)
            low = self.execute("MUL", a, b)
            high = self.execute("MULH", a, b)
            combined = (high << 32) | low
            if combined & (1 << 63):
                combined -= 1 << 64
            self.assertEqual(combined, a * b, f"{a} * {b}")

    def test_unsigned_high_low_pair(self):
        """MUL and MULHU together give the exact 64-bit unsigned product"""
        rng = random.Random(99)
        for _ in range(300):
            a = rng.randrange(0, 1 << 32)
            b = rng.randrange(0, 1 << 32)
            combined = (self.execute("MULHU", a, b) << 32) | self.execute("MUL", a, b)
            self.assertEqual(combined, a * b, f"{a} * {b}")


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------
class TestDivide(RV32MTestCase):
    def test_div(self):
        """DIV"""
        self.assertOp("DIV", 42, 6, 7)
        self.assertOp("DIV", 0, 5, 0)
        self.assertOp("DIV", 5, 5, 1)
        self.assertOp("DIV", 4, 5, 0)

    def test_div_truncates_toward_zero(self):
        """DIV truncates toward zero, not toward negative infinity"""
        # This is the case Python's // gets wrong: -7 // 2 == -4.
        self.assertOp("DIV", -7, 2, -3, "Python's // would give -4")
        self.assertOp("DIV", 7, -2, -3, "Python's // would give -4")
        self.assertOp("DIV", -7, -2, 3)
        self.assertOp("DIV", -1, 2, 0, "Python's // would give -1")
        self.assertOp("DIV", 1, -2, 0)

    def test_divu(self):
        """DIVU"""
        self.assertOp("DIVU", 42, 6, 7)
        self.assertOp("DIVU", 0xFFFFFFFF, 2, 0x7FFFFFFF, "unsigned: not -1/2")
        self.assertOp("DIVU", 0xFFFFFFFF, 0xFFFFFFFF, 1)
        self.assertOp("DIVU", 1, 0xFFFFFFFF, 0)

    def test_rem(self):
        """REM"""
        self.assertOp("REM", 42, 5, 2)
        self.assertOp("REM", 42, 6, 0)
        self.assertOp("REM", 3, 5, 3)

    def test_rem_sign_follows_the_dividend(self):
        """REM takes the sign of the dividend"""
        # Python's % gets this wrong: -7 % 2 == 1.
        self.assertOp("REM", -7, 2, -1, "Python's % would give 1")
        self.assertOp("REM", 7, -2, 1, "Python's % would give -1")
        self.assertOp("REM", -7, -2, -1)
        self.assertOp("REM", -1, 2, -1)

    def test_remu(self):
        """REMU"""
        self.assertOp("REMU", 42, 5, 2)
        self.assertOp("REMU", 0xFFFFFFFF, 2, 1)
        self.assertOp("REMU", 0xFFFFFFFF, 0x10, 0xF)

    def test_division_by_zero_does_not_trap(self):
        """division by zero yields defined values, not a trap"""
        for dividend in (0, 1, -1, 42, INT32_MIN, 0x7FFFFFFF):
            self.assertOp("DIV", dividend, 0, -1, "DIV by zero is all ones")
            self.assertOp("REM", dividend, 0, dividend, "REM by zero is the dividend")
            self.assertOp("DIVU", dividend, 0, UINT32_MAX, "DIVU by zero is all ones")
            self.assertOp("REMU", dividend, 0, dividend, "REMU by zero is the dividend")

    def test_signed_overflow(self):
        """INT32_MIN / -1 wraps instead of trapping"""
        self.assertOp("DIV", INT32_MIN, -1, INT32_MIN, "the one signed overflow")
        self.assertOp("REM", INT32_MIN, -1, 0)
        # Unsigned has no overflow case: 0x80000000 / 0xFFFFFFFF is just 0.
        self.assertOp("DIVU", INT32_MIN, -1, 0)
        self.assertOp("REMU", INT32_MIN, -1, 0x80000000)

    def test_division_identity(self):
        """rs1 == rs2 * DIV(rs1, rs2) + REM(rs1, rs2) for random operands"""
        rng = random.Random(2024)
        checked = 0
        for _ in range(500):
            a = rng.randrange(-(1 << 31), 1 << 31)
            b = rng.randrange(-(1 << 31), 1 << 31)
            if b == 0 or (a == INT32_MIN and b == -1):
                continue  # the two cases the spec exempts
            quotient = signed(self.execute("DIV", a, b))
            remainder = signed(self.execute("REM", a, b))
            self.assertEqual((b * quotient + remainder) & MASK32, a & MASK32, f"{a} / {b}")
            checked += 1
        self.assertGreater(checked, 400)

    def test_unsigned_division_identity(self):
        """rs1 == rs2 * DIVU + REMU for random unsigned operands"""
        rng = random.Random(777)
        for _ in range(500):
            a = rng.randrange(0, 1 << 32)
            b = rng.randrange(1, 1 << 32)
            quotient = self.execute("DIVU", a, b)
            remainder = self.execute("REMU", a, b)
            self.assertEqual((b * quotient + remainder) & MASK32, a)
            self.assertLess(remainder, b, "the remainder must be smaller than the divisor")


class TestHelperFunctions(unittest.TestCase):
    """The helpers are exported, so check them directly as well."""

    def test_divide_signed(self):
        """rv32m helpers: divide_signed truncates toward zero"""
        self.assertEqual(divide_signed(-7, 2), -3)
        self.assertEqual(divide_signed(7, -2), -3)
        self.assertEqual(divide_signed(-7, -2), 3)
        self.assertEqual(divide_signed(5, 0), -1)
        self.assertEqual(divide_signed(INT32_MIN, -1), INT32_MIN)

    def test_remainder_signed(self):
        """rv32m helpers: remainder_signed follows the dividend's sign"""
        self.assertEqual(remainder_signed(-7, 2), -1)
        self.assertEqual(remainder_signed(7, -2), 1)
        self.assertEqual(remainder_signed(5, 0), 5)
        self.assertEqual(remainder_signed(INT32_MIN, -1), 0)

    def test_unsigned_helpers(self):
        """rv32m helpers: unsigned division and remainder"""
        self.assertEqual(divide_unsigned(7, 2), 3)
        self.assertEqual(divide_unsigned(7, 0), UINT32_MAX)
        self.assertEqual(remainder_unsigned(7, 2), 1)
        self.assertEqual(remainder_unsigned(7, 0), 7)


# ---------------------------------------------------------------------------
# ISA selection
# ---------------------------------------------------------------------------
class TestIsaSelection(EmulatorTestCase):
    def test_rv32i_stays_pure(self):
        """isa: RV32I does not gain the M instructions"""
        self.assertEqual(len(RV32I), 40)
        for name in RV32M_INSTRUCTIONS:
            self.assertNotIn(name, RV32I)

    def test_m_instructions_are_illegal_on_rv32i(self):
        """isa: an M instruction traps on a base RV32I machine"""
        word = encode("MUL", rd=3, rs1=1, rs2=2, instruction_set=RV32IM)
        cpu = CPU()  # default: RV32I
        cpu.load_program([word])
        with self.assertRaises(IllegalInstruction) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 2)
        self.assertEqual(context.exception.tval, word)

    def test_m_instructions_run_on_rv32im(self):
        """isa: the same word executes on an RV32IM machine"""
        word = encode("MUL", rd=3, rs1=1, rs2=2, instruction_set=RV32IM)
        cpu = CPU(instruction_set=RV32IM)
        cpu.load_program([word])
        cpu.set_reg(1, 6)
        cpu.set_reg(2, 7)
        cpu.step()
        self.assertReg(cpu, 3, 42)

    def test_registry(self):
        """isa: instruction sets are reachable by name"""
        self.assertIn("rv32i", available_isas())
        self.assertIn("rv32im", available_isas())
        self.assertIs(get_isa("rv32i"), RV32I)
        self.assertIs(get_isa("rv32im"), RV32IM)
        self.assertIs(get_isa("RV32IM"), RV32IM, "lookup is case-insensitive")

    def test_unknown_isa_name(self):
        """isa: an unknown name raises with the available ones listed"""
        with self.assertRaises(KeyError) as context:
            get_isa("rv32q")
        self.assertIn("rv32im", context.exception.args[0])

    def test_assembler_respects_the_isa(self):
        """isa: the assembler rejects M mnemonics on RV32I"""
        self.assertEqual(len(assemble("mul x1, x2, x3", instruction_set=RV32IM).words), 1)
        with self.assertRaises(AssemblerError) as context:
            assemble("mul x1, x2, x3")
        self.assertIn("unknown instruction", str(context.exception))

    def test_base_instructions_are_unchanged_on_rv32im(self):
        """isa: RV32I instructions behave identically on RV32IM"""
        program = assemble("li x1, 10\nli x2, 20\nadd x3, x1, x2\nsub x4, x2, x1")
        base, extended = CPU(), CPU(instruction_set=RV32IM)
        for cpu in (base, extended):
            cpu.load_program(program.words)
            cpu.run(len(program.words))
        # Registers, PC and retire count must match exactly.  The full state()
        # deliberately does *not*, because misa reports the machine's actual
        # extensions -- see below.
        self.assertEqual(base.regs.snapshot(), extended.regs.snapshot())
        self.assertEqual(base.pc, extended.pc)
        self.assertEqual(base.instret, extended.instret)

    def test_misa_reports_the_extensions(self):
        """isa: misa distinguishes an RV32I machine from an RV32IM one"""
        m_bit = 1 << (ord("m") - ord("a"))
        self.assertFalse(CPU().csr.get("misa") & m_bit, "RV32I must not claim M")
        self.assertTrue(
            CPU(instruction_set=RV32IM).csr.get("misa") & m_bit, "RV32IM must claim M"
        )


class TestArchStringMapping(unittest.TestCase):
    def test_arch_strings(self):
        """isa: ELF architecture strings map to instruction sets"""
        from riscv.elf import isa_name_from_arch_string as resolve

        self.assertEqual(resolve("rv32i2p1"), "rv32i")
        self.assertEqual(resolve("rv32i2p1_m2p0"), "rv32im")
        self.assertEqual(resolve("rv32im"), "rv32im")
        self.assertEqual(resolve("rv32i2p1_m2p0_zmmul1p0"), "rv32im")
        self.assertEqual(resolve("rv32i_zmmul1p0"), "rv32im", "Zmmul needs only multiply")
        self.assertEqual(resolve("rv32i2p1_zicsr2p0"), "rv32i_zicsr")
        self.assertEqual(resolve("rv32i2p1_m2p0_zicsr2p0"), "rv32im_zicsr")

    def test_unsupported_arch_strings_are_reported_honestly(self):
        """isa: an unsupported architecture maps to None rather than a guess"""
        from riscv.elf import isa_name_from_arch_string as resolve

        for arch in ("rv32imafdc", "rv32i2p1_f2p0", "rv64i2p1_m2p0", "rv32e2p0",
                     "", "nonsense"):
            self.assertIsNone(resolve(arch), arch)


if __name__ == "__main__":
    unittest.main()
