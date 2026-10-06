"""Encoder tests: reference encodings and exhaustive decoder round-trips."""

import random
import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.decoder import Decoder
from riscv.encoder import encode
from riscv.exceptions import AssemblerError
from riscv.instructions import RV32I
from riscv.isa import B_TYPE, I_TYPE, J_TYPE, N_TYPE, R_TYPE, S_TYPE, U_TYPE

#: Encodings produced by a standard RISC-V toolchain.  If the emulator agrees
#: with these, its bit layout is right -- not merely self-consistent.
REFERENCE_ENCODINGS = [
    ("ADDI", dict(rd=0, rs1=0, imm=0), 0x00000013),        # nop
    ("ADDI", dict(rd=1, rs1=0, imm=10), 0x00A00093),
    ("ADDI", dict(rd=2, rs1=0, imm=20), 0x01400113),
    ("ADDI", dict(rd=2, rs1=2, imm=-16), 0xFF010113),      # addi sp, sp, -16
    ("ADD", dict(rd=3, rs1=1, rs2=2), 0x002081B3),
    ("ADD", dict(rd=10, rs1=10, rs2=11), 0x00B50533),
    ("SUB", dict(rd=10, rs1=10, rs2=11), 0x40B50533),
    ("XOR", dict(rd=10, rs1=10, rs2=11), 0x00B54533),
    ("OR", dict(rd=10, rs1=10, rs2=11), 0x00B56533),
    ("AND", dict(rd=10, rs1=10, rs2=11), 0x00B57533),
    ("SLTU", dict(rd=10, rs1=10, rs2=11), 0x00B53533),
    ("SLLI", dict(rd=10, rs1=10, imm=31), 0x01F51513),
    ("SRLI", dict(rd=10, rs1=10, imm=31), 0x01F55513),
    ("SRAI", dict(rd=10, rs1=10, imm=31), 0x41F55513),
    ("LW", dict(rd=1, rs1=2, imm=12), 0x00C12083),         # lw ra, 12(sp)
    ("SW", dict(rs1=2, rs2=1, imm=12), 0x00112623),        # sw ra, 12(sp)
    ("LB", dict(rd=10, rs1=11, imm=-1), 0xFFF58503),
    ("LBU", dict(rd=10, rs1=11, imm=3), 0x0035C503),
    ("LHU", dict(rd=10, rs1=11, imm=2), 0x0025D503),
    ("SB", dict(rs1=11, rs2=10, imm=1), 0x00A580A3),
    ("SH", dict(rs1=11, rs2=10, imm=2), 0x00A59123),
    ("BEQ", dict(rs1=10, rs2=11, imm=8), 0x00B50463),
    ("BNE", dict(rs1=10, rs2=11, imm=-8), 0xFEB51CE3),
    ("JAL", dict(rd=1, imm=0), 0x000000EF),
    ("JAL", dict(rd=0, imm=-4), 0xFFDFF06F),
    ("JALR", dict(rd=1, rs1=10, imm=0), 0x000500E7),
    ("JALR", dict(rd=0, rs1=1, imm=0), 0x00008067),        # ret
    ("LUI", dict(rd=10, imm=0x12345), 0x12345537),
    ("AUIPC", dict(rd=10, imm=0), 0x00000517),
    ("ANDI", dict(rd=10, rs1=11, imm=-1), 0xFFF5F513),
    ("ECALL", dict(), 0x00000073),
    ("EBREAK", dict(), 0x00100073),
]


class TestReferenceEncodings(unittest.TestCase):
    def test_reference_encodings(self):
        """encode: 32 encodings match standard toolchain output"""
        for name, kwargs, expected in REFERENCE_ENCODINGS:
            word = encode(name, **kwargs)
            self.assertEqual(
                word,
                expected,
                f"{name} {kwargs}: got 0x{word:08X}, expected 0x{expected:08X}",
            )

    def test_encoder_accepts_register_names(self):
        """encode: register arguments accept ABI names"""
        self.assertEqual(encode("ADDI", rd="sp", rs1="sp", imm=-16), 0xFF010113)
        self.assertEqual(encode("ADD", rd="a0", rs1="a0", rs2="a1"), 0x00B50533)


class TestRoundTrips(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder()

    def test_roundtrip_every_instruction(self):
        """encode/decode: every instruction round-trips with random operands"""
        rng = random.Random(20240501)
        for definition in RV32I:
            for _ in range(20):
                kwargs = {}
                fmt = definition.fmt
                if fmt in (R_TYPE, I_TYPE, U_TYPE, J_TYPE):
                    kwargs["rd"] = rng.randrange(32)
                if fmt in (R_TYPE, I_TYPE, S_TYPE, B_TYPE):
                    kwargs["rs1"] = rng.randrange(32)
                if fmt in (R_TYPE, S_TYPE, B_TYPE):
                    kwargs["rs2"] = rng.randrange(32)
                if fmt in (I_TYPE, S_TYPE):
                    if definition.name in ("SLLI", "SRLI", "SRAI"):
                        kwargs["imm"] = rng.randrange(32)
                    else:
                        kwargs["imm"] = rng.randrange(-2048, 2048)
                elif fmt == B_TYPE:
                    kwargs["imm"] = rng.randrange(-2048, 2048) * 2
                elif fmt == U_TYPE:
                    kwargs["imm"] = rng.randrange(0, 1 << 20)
                elif fmt == J_TYPE:
                    kwargs["imm"] = rng.randrange(-(1 << 19), 1 << 19) * 2

                word = encode(definition.name, **kwargs)
                decoded = self.decoder.decode(word)
                self.assertEqual(decoded.name, definition.name, f"0x{word:08X}")
                for field in ("rd", "rs1", "rs2"):
                    if field in kwargs:
                        self.assertEqual(getattr(decoded, field), kwargs[field], definition.name)
                if "imm" in kwargs:
                    expected = kwargs["imm"]
                    if definition.name in ("SLLI", "SRLI", "SRAI"):
                        self.assertEqual(decoded.shamt, expected)
                    elif fmt == U_TYPE:
                        self.assertEqual(decoded.imm, (expected << 12) & 0xFFFFFFFF)
                    else:
                        self.assertEqual(decoded.imm_signed, expected, definition.name)

    def test_boundary_immediates(self):
        """encode/decode: immediates at their range boundaries survive"""
        for imm in (-2048, -1, 0, 1, 2047):
            self.assertEqual(self.decoder.decode(encode("ADDI", rd=1, rs1=0, imm=imm)).imm_signed, imm)
            self.assertEqual(self.decoder.decode(encode("SW", rs1=1, rs2=2, imm=imm)).imm_signed, imm)
        for imm in (-4096, -2, 0, 2, 4094):
            self.assertEqual(self.decoder.decode(encode("BEQ", rs1=1, rs2=2, imm=imm)).imm_signed, imm)
        for imm in (-(1 << 20), -2, 0, 2, (1 << 20) - 2):
            self.assertEqual(self.decoder.decode(encode("JAL", rd=1, imm=imm)).imm_signed, imm)


class TestEncoderValidation(unittest.TestCase):
    def test_out_of_range_immediate_raises(self):
        """encode: an out-of-range immediate is rejected"""
        with self.assertRaises(AssemblerError):
            encode("ADDI", rd=1, rs1=0, imm=2048)
        with self.assertRaises(AssemblerError):
            encode("ADDI", rd=1, rs1=0, imm=-2049)
        with self.assertRaises(AssemblerError):
            encode("SLLI", rd=1, rs1=0, imm=32)
        with self.assertRaises(AssemblerError):
            encode("BEQ", rs1=0, rs2=0, imm=4096)

    def test_odd_branch_offset_raises(self):
        """encode: an odd branch or jump offset is rejected"""
        with self.assertRaises(AssemblerError):
            encode("BEQ", rs1=0, rs2=0, imm=3)
        with self.assertRaises(AssemblerError):
            encode("JAL", rd=0, imm=1)

    def test_bad_register_raises(self):
        """encode: an invalid register is rejected"""
        with self.assertRaises(Exception):
            encode("ADD", rd=32, rs1=0, rs2=0)

    def test_unknown_instruction_raises(self):
        """encode: an unknown mnemonic is rejected"""
        with self.assertRaises(KeyError):
            encode("MUL", rd=1, rs1=2, rs2=3)


class TestExtensibility(unittest.TestCase):
    def test_a_new_instruction_can_be_added(self):
        """isa: an extension instruction decodes and executes without core changes"""
        from riscv import CPU
        from riscv.isa import R_TYPE as R

        extended = RV32I.copy("RV32IM-subset")

        @extended.define("MUL", match=0x02000033, mask=0xFE00707F, fmt=R, extension="RV32M")
        def exec_mul(cpu, inst):
            product = cpu.get_reg(inst.rs1) * cpu.get_reg(inst.rs2)
            cpu.set_reg(inst.rd, product & 0xFFFFFFFF)

        word = encode("MUL", rd=3, rs1=1, rs2=2, instruction_set=extended)
        self.assertEqual(word, 0x022081B3, "MUL encoding must match the RV32M spec")

        cpu = CPU(instruction_set=extended)
        cpu.set_reg(1, 6)
        cpu.set_reg(2, 7)
        cpu.load_program([word])
        cpu.step()
        self.assertEqual(cpu.get_reg(3), 42)

        # ...and the base set is untouched.
        self.assertNotIn("MUL", RV32I)


if __name__ == "__main__":
    unittest.main()
