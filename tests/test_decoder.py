"""Unit tests for instruction decoding and immediate reconstruction."""

import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.decoder import (
    Decoder,
    decode_b_imm,
    decode_i_imm,
    decode_j_imm,
    decode_s_imm,
    decode_u_imm,
    extract_fields,
)
from riscv.encoder import encode
from riscv.exceptions import IllegalInstruction
from riscv.instructions import RV32I, RV32I_INSTRUCTION_COUNT


class TestFieldExtraction(unittest.TestCase):
    def test_fields_of_a_known_word(self):
        """decode: fixed fields of ADDI x1, x0, 10 (0x00A00093)"""
        fields = extract_fields(0x00A00093)
        self.assertEqual(fields["opcode"], 0x13)
        self.assertEqual(fields["rd"], 1)
        self.assertEqual(fields["funct3"], 0)
        self.assertEqual(fields["rs1"], 0)

    def test_fields_of_an_r_type(self):
        """decode: fixed fields of ADD x3, x1, x2 (0x002081B3)"""
        fields = extract_fields(0x002081B3)
        self.assertEqual(fields["opcode"], 0x33)
        self.assertEqual(fields["rd"], 3)
        self.assertEqual(fields["rs1"], 1)
        self.assertEqual(fields["rs2"], 2)
        self.assertEqual(fields["funct7"], 0)


class TestImmediateDecoding(unittest.TestCase):
    def test_i_immediate(self):
        """immediates: I-type, positive and negative"""
        self.assertEqual(decode_i_imm(0x00A00093), 10)          # addi x1, x0, 10
        self.assertEqual(decode_i_imm(0xFFF00093), 0xFFFFFFFF)  # addi x1, x0, -1
        self.assertEqual(decode_i_imm(0x80000093), 0xFFFFF800)  # addi x1, x0, -2048
        self.assertEqual(decode_i_imm(0x7FF00093), 0x000007FF)  # addi x1, x0, 2047

    def test_s_immediate(self):
        """immediates: S-type is split across two fields"""
        self.assertEqual(decode_s_imm(encode("SW", rs1=2, rs2=3, imm=0)), 0)
        self.assertEqual(decode_s_imm(encode("SW", rs1=2, rs2=3, imm=2047)), 2047)
        self.assertEqual(decode_s_imm(encode("SW", rs1=2, rs2=3, imm=-1)), 0xFFFFFFFF)
        self.assertEqual(decode_s_imm(encode("SW", rs1=2, rs2=3, imm=-2048)), 0xFFFFF800)
        # 0x00112623 is "sw ra, 12(sp)" from a real toolchain.
        self.assertEqual(decode_s_imm(0x00112623), 12)

    def test_b_immediate(self):
        """immediates: B-type is scrambled and always even"""
        for offset in (0, 2, 4, -2, -4, 4094, -4096, 2048, -2048):
            word = encode("BEQ", rs1=1, rs2=2, imm=offset)
            self.assertEqual(decode_b_imm(word), offset & 0xFFFFFFFF, f"offset {offset}")
        # 0x00B50463 is "beq a0, a1, 8" from a real toolchain.
        self.assertEqual(decode_b_imm(0x00B50463), 8)

    def test_u_immediate(self):
        """immediates: U-type keeps its 20 bits in place"""
        self.assertEqual(decode_u_imm(0x12345537), 0x12345000)  # lui a0, 0x12345
        self.assertEqual(decode_u_imm(0xFFFFF037), 0xFFFFF000)
        self.assertEqual(decode_u_imm(0x00000037), 0)

    def test_j_immediate(self):
        """immediates: J-type is scrambled and always even"""
        for offset in (0, 2, -2, 4, -4, 1048574, -1048576, 2048, -2048):
            word = encode("JAL", rd=1, imm=offset)
            self.assertEqual(decode_j_imm(word), offset & 0xFFFFFFFF, f"offset {offset}")
        # 0xFFDFF06F is "jal x0, -4" from a real toolchain.
        self.assertEqual(decode_j_imm(0xFFDFF06F), 0xFFFFFFFC)

    def test_immediate_signed_view(self):
        """immediates: imm_signed presents the two's-complement value"""
        decoder = Decoder()
        self.assertEqual(decoder.decode(0xFFF00093).imm_signed, -1)
        self.assertEqual(decoder.decode(0x00A00093).imm_signed, 10)


class TestInstructionIdentification(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder()

    def test_every_rv32i_instruction_decodes(self):
        """decode: all 40 RV32I instructions are recognised"""
        self.assertEqual(len(RV32I), RV32I_INSTRUCTION_COUNT)
        for definition in RV32I:
            decoded = self.decoder.decode(definition.match)
            self.assertEqual(decoded.name, definition.name)
            self.assertEqual(decoded.fmt, definition.fmt)

    def test_funct7_distinguishes_add_and_sub(self):
        """decode: ADD and SUB differ only in funct7"""
        self.assertEqual(self.decoder.decode(0x00B50533).name, "ADD")
        self.assertEqual(self.decoder.decode(0x40B50533).name, "SUB")

    def test_funct7_distinguishes_srl_and_sra(self):
        """decode: SRL and SRA differ only in funct7"""
        self.assertEqual(self.decoder.decode(encode("SRL", rd=1, rs1=2, rs2=3)).name, "SRL")
        self.assertEqual(self.decoder.decode(encode("SRA", rd=1, rs1=2, rs2=3)).name, "SRA")
        self.assertEqual(self.decoder.decode(encode("SRLI", rd=1, rs1=2, imm=4)).name, "SRLI")
        self.assertEqual(self.decoder.decode(encode("SRAI", rd=1, rs1=2, imm=4)).name, "SRAI")

    def test_ecall_and_ebreak(self):
        """decode: ECALL and EBREAK are distinguished by imm[0]"""
        self.assertEqual(self.decoder.decode(0x00000073).name, "ECALL")
        self.assertEqual(self.decoder.decode(0x00100073).name, "EBREAK")

    def test_shamt_field(self):
        """decode: shift-immediate shamt comes from the low 5 bits"""
        inst = self.decoder.decode(encode("SLLI", rd=1, rs1=2, imm=31))
        self.assertEqual(inst.shamt, 31)
        self.assertEqual(self.decoder.decode(encode("SRAI", rd=1, rs1=2, imm=0)).shamt, 0)


class TestIllegalInstructions(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder()

    def assertIllegal(self, word, why=""):
        with self.assertRaises(IllegalInstruction, msg=f"0x{word:08X} {why}"):
            self.decoder.decode(word)

    def test_zero_word_is_illegal(self):
        """decode: the all-zeros word is an illegal instruction"""
        self.assertIllegal(0x00000000)

    def test_all_ones_word_is_illegal(self):
        """decode: the all-ones word is an illegal instruction"""
        self.assertIllegal(0xFFFFFFFF)

    def test_unknown_opcodes_are_illegal(self):
        """decode: unassigned opcodes are illegal"""
        for opcode in (0x00, 0x02, 0x0B, 0x1F, 0x2B, 0x3B, 0x7F):
            self.assertIllegal(opcode, "unassigned opcode")

    def test_bad_funct3_is_illegal(self):
        """decode: reserved funct3 values are illegal"""
        self.assertIllegal(0x00002063, "branch funct3=010")
        self.assertIllegal(0x00003063, "branch funct3=011")
        self.assertIllegal(0x00006003, "load funct3=110")
        self.assertIllegal(0x00003023, "store funct3=011")
        self.assertIllegal(0x00001067, "jalr funct3=001")

    def test_bad_funct7_is_illegal(self):
        """decode: reserved funct7 values are illegal"""
        self.assertIllegal(0x20000033, "add with funct7=0x10")
        self.assertIllegal(0x40001033, "sll with funct7=0x20")
        self.assertIllegal(0x02001093, "slli with a stray funct7 bit")
        self.assertIllegal(0x20005013, "srli with funct7=0x10")

    def test_system_encodings_other_than_ecall_ebreak(self):
        """decode: CSR instructions are illegal (not implemented yet)"""
        self.assertIllegal(0x30001073, "csrrw")
        self.assertIllegal(0x00200073, "uret")
        self.assertIllegal(0x30200073, "mret")


class TestDecodeCache(unittest.TestCase):
    def test_cache_returns_identical_objects(self):
        """decode: repeated decodes are memoised"""
        decoder = Decoder()
        first = decoder.decode(0x00A00093)
        second = decoder.decode(0x00A00093)
        self.assertIs(first, second)

    def test_cache_can_be_cleared(self):
        """decode: clear_cache() drops memoised instructions"""
        decoder = Decoder()
        first = decoder.decode(0x00A00093)
        decoder.clear_cache()
        self.assertIsNot(decoder.decode(0x00A00093), first)

    def test_cache_is_bounded(self):
        """decode: the cache does not grow without bound"""
        decoder = Decoder(cache_size=8)
        for value in range(64):
            decoder.decode(encode("ADDI", rd=1, rs1=0, imm=value))
        self.assertLessEqual(len(decoder._cache), 8)

    def test_decoding_is_position_independent(self):
        """decode: the same word decodes identically at any pc"""
        decoder = Decoder()
        one = decoder.decode(0x00A00093, pc=0)
        decoder.clear_cache()
        two = decoder.decode(0x00A00093, pc=0x1000)
        self.assertEqual(one.imm, two.imm)
        self.assertEqual(one.name, two.name)


if __name__ == "__main__":
    unittest.main()
