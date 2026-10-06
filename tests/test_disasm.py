"""Disassembler tests, including the trace-line format from the brief."""

import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.disasm import disassemble, safe_disassemble, trace_line
from riscv.encoder import encode
from riscv.exceptions import IllegalInstruction
from riscv.instructions import RV32I


class TestDisassemblyText(unittest.TestCase):
    def test_brief_example_instructions(self):
        """disasm: the three instructions from the project brief"""
        self.assertEqual(disassemble(0x00A00093), "ADDI x1,x0,10")
        self.assertEqual(disassemble(0x01400113), "ADDI x2,x0,20")
        self.assertEqual(disassemble(0x002081B3), "ADD  x3,x1,x2")

    def test_operand_forms(self):
        """disasm: each format uses its conventional operand syntax"""
        self.assertEqual(disassemble(encode("LW", rd=1, rs1=2, imm=12)), "LW   x1,12(x2)")
        self.assertEqual(disassemble(encode("SW", rs1=2, rs2=1, imm=-4)), "SW   x1,-4(x2)")
        self.assertEqual(disassemble(encode("BEQ", rs1=1, rs2=2, imm=-8)), "BEQ  x1,x2,-8")
        self.assertEqual(disassemble(encode("JAL", rd=1, imm=16)), "JAL  x1,16")
        self.assertEqual(disassemble(encode("JALR", rd=1, rs1=5, imm=0)), "JALR x1,0(x5)")
        self.assertEqual(disassemble(encode("LUI", rd=5, imm=0x12345)), "LUI  x5,0x12345")
        self.assertEqual(disassemble(encode("AUIPC", rd=5, imm=1)), "AUIPC x5,0x1")
        self.assertEqual(disassemble(encode("SRAI", rd=1, rs1=2, imm=31)), "SRAI x1,x2,31")
        self.assertEqual(disassemble(encode("ECALL")), "ECALL")
        self.assertEqual(disassemble(encode("EBREAK")), "EBREAK")

    def test_negative_immediates_print_signed(self):
        """disasm: negative immediates print as signed decimals"""
        self.assertEqual(disassemble(0xFFF00093), "ADDI x1,x0,-1")
        self.assertEqual(disassemble(encode("ADDI", rd=2, rs1=2, imm=-16)), "ADDI x2,x2,-16")

    def test_abi_names(self):
        """disasm: ABI register names are available"""
        self.assertEqual(disassemble(0xFF010113, abi=True), "ADDI sp,sp,-16")
        self.assertEqual(disassemble(0x00008067, abi=True), "JALR zero,0(ra)")

    def test_fence_fields(self):
        """disasm: FENCE prints its predecessor/successor sets"""
        self.assertEqual(disassemble(0x0FF0000F), "FENCE iorw,iorw")

    def test_every_instruction_disassembles(self):
        """disasm: every RV32I instruction produces text"""
        for definition in RV32I:
            text = disassemble(definition.match)
            self.assertTrue(text.startswith(definition.name), text)


class TestTraceLine(unittest.TestCase):
    def test_trace_line_format(self):
        """disasm: trace lines match the format in the brief"""
        self.assertEqual(
            trace_line(0x00000000, 0x00A00093),
            "PC=0x00000000  INST=0x00A00093  ADDI x1,x0,10",
        )
        self.assertEqual(
            trace_line(0x00000004, 0x01400113),
            "PC=0x00000004  INST=0x01400113  ADDI x2,x0,20",
        )
        self.assertEqual(
            trace_line(0x00000008, 0x002081B3),
            "PC=0x00000008  INST=0x002081B3  ADD  x3,x1,x2",
        )


class TestUndecodableWords(unittest.TestCase):
    def test_disassemble_raises_on_illegal(self):
        """disasm: an undecodable word raises"""
        with self.assertRaises(IllegalInstruction):
            disassemble(0x00000000)

    def test_safe_disassemble_falls_back(self):
        """disasm: safe_disassemble renders undecodable words as .word"""
        self.assertEqual(safe_disassemble(0x00000000), ".word 0x00000000")
        self.assertEqual(safe_disassemble(0xFFFFFFFF), ".word 0xFFFFFFFF")
        self.assertEqual(safe_disassemble(0x0000007F), ".word 0x0000007F")

    def test_deadbeef_is_a_valid_jal(self):
        """disasm: 0xDEADBEEF is a legal JAL, not garbage (opcode 0x6F)"""
        # A reminder that "looks like garbage" and "is an illegal instruction"
        # are different questions -- only the opcode field decides.
        self.assertEqual(disassemble(0xDEADBEEF), "JAL  x29,-150038")


class TestDisassemblyRoundTrip(unittest.TestCase):
    def test_disassembly_reassembles(self):
        """disasm: disassembly feeds back through the assembler unchanged"""
        from riscv.assembler import assemble

        words = [
            encode("ADDI", rd=1, rs1=0, imm=-1),
            encode("ADD", rd=3, rs1=1, rs2=2),
            encode("SUB", rd=3, rs1=1, rs2=2),
            encode("LW", rd=1, rs1=2, imm=-4),
            encode("SW", rs1=2, rs2=1, imm=-4),
            encode("SB", rs1=2, rs2=1, imm=7),
            encode("BEQ", rs1=1, rs2=2, imm=-8),
            encode("BGEU", rs1=1, rs2=2, imm=64),
            encode("JAL", rd=1, imm=-16),
            encode("JALR", rd=1, rs1=2, imm=4),
            encode("LUI", rd=5, imm=0xABCDE),
            encode("AUIPC", rd=5, imm=0x1),
            encode("SLLI", rd=1, rs1=2, imm=1),
            encode("SRAI", rd=1, rs1=2, imm=31),
            encode("SLTIU", rd=1, rs1=2, imm=-1),
            encode("FENCE", imm=0xFF),
            encode("ECALL"),
            encode("EBREAK"),
        ]
        for word in words:
            text = disassemble(word)
            rebuilt = assemble(text).words[0]
            self.assertEqual(
                rebuilt,
                word,
                f"{text}: 0x{word:08X} disassembled then reassembled to 0x{rebuilt:08X}",
            )


if __name__ == "__main__":
    unittest.main()
