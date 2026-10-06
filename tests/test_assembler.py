"""Assembler tests: syntax, labels, pseudo-instructions, directives, errors."""

import os
import unittest

from tests.helpers import EXAMPLES_DIR, EmulatorTestCase  # noqa: F401  (path setup)

from riscv.assembler import assemble, assemble_file
from riscv.encoder import encode
from riscv.exceptions import AssemblerError


class TestBasicSyntax(unittest.TestCase):
    def test_three_instruction_program(self):
        """assembler: the brief's example assembles to the expected words"""
        program = assemble("addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2")
        self.assertEqual(program.words, [0x00A00093, 0x01400113, 0x002081B3])

    def test_comments_and_blank_lines(self):
        """assembler: '#' and '//' comments and blank lines are ignored"""
        program = assemble(
            """
            # a comment
            addi x1, x0, 1   # trailing comment
                             // another style

            addi x2, x0, 2
            """
        )
        self.assertEqual(len(program.words), 2)

    def test_abi_register_names(self):
        """assembler: ABI register names are accepted"""
        self.assertEqual(assemble("addi sp, sp, -16").words[0], 0xFF010113)
        self.assertEqual(assemble("add a0, a0, a1").words[0], 0x00B50533)
        self.assertEqual(assemble("addi fp, s0, 0").words[0], encode("ADDI", rd=8, rs1=8, imm=0))

    def test_number_formats(self):
        """assembler: decimal, hex, binary, octal and char literals"""
        self.assertEqual(assemble("addi x1, x0, 10").words[0], encode("ADDI", rd=1, rs1=0, imm=10))
        self.assertEqual(assemble("addi x1, x0, 0xA").words[0], encode("ADDI", rd=1, rs1=0, imm=10))
        self.assertEqual(assemble("addi x1, x0, 0b1010").words[0], encode("ADDI", rd=1, rs1=0, imm=10))
        self.assertEqual(assemble("addi x1, x0, 0o12").words[0], encode("ADDI", rd=1, rs1=0, imm=10))
        self.assertEqual(assemble("addi x1, x0, 'A'").words[0], encode("ADDI", rd=1, rs1=0, imm=65))

    def test_expressions(self):
        """assembler: arithmetic expressions in immediates"""
        self.assertEqual(assemble("addi x1, x0, 2+3*4").words[0], encode("ADDI", rd=1, rs1=0, imm=14))
        self.assertEqual(assemble("addi x1, x0, (2+3)*4").words[0], encode("ADDI", rd=1, rs1=0, imm=20))
        self.assertEqual(assemble("addi x1, x0, 1<<10").words[0], encode("ADDI", rd=1, rs1=0, imm=1024))
        self.assertEqual(assemble("addi x1, x0, -(1+1)").words[0], encode("ADDI", rd=1, rs1=0, imm=-2))

    def test_memory_operand_forms(self):
        """assembler: both offset(base) and three-operand forms work"""
        self.assertEqual(assemble("lw x1, 12(x2)").words[0], encode("LW", rd=1, rs1=2, imm=12))
        self.assertEqual(assemble("lw x1, x2, 12").words[0], encode("LW", rd=1, rs1=2, imm=12))
        self.assertEqual(assemble("sw x1, 12(x2)").words[0], encode("SW", rs1=2, rs2=1, imm=12))
        self.assertEqual(assemble("lw x1, (x2)").words[0], encode("LW", rd=1, rs1=2, imm=0))


class TestLabels(unittest.TestCase):
    def test_forward_branch(self):
        """assembler: a forward branch resolves to a positive offset"""
        program = assemble("beq x1, x2, skip\naddi x3, x0, 1\nskip:\naddi x4, x0, 2")
        self.assertEqual(program.words[0], encode("BEQ", rs1=1, rs2=2, imm=8))

    def test_backward_branch(self):
        """assembler: a backward branch resolves to a negative offset"""
        program = assemble("loop:\naddi x1, x1, -1\nbnez x1, loop")
        self.assertEqual(program.words[1], encode("BNE", rs1=1, rs2=0, imm=-4))

    def test_numeric_branch_target_is_relative(self):
        """assembler: a numeric branch target is a PC-relative offset"""
        self.assertEqual(assemble("beq x1, x2, 8").words[0], encode("BEQ", rs1=1, rs2=2, imm=8))
        self.assertEqual(assemble("beq x1, x2, -8").words[0], encode("BEQ", rs1=1, rs2=2, imm=-8))

    def test_jal_to_label(self):
        """assembler: JAL resolves a label to a PC-relative offset"""
        program = assemble("jal x1, target\nnop\ntarget:\nnop")
        self.assertEqual(program.words[0], encode("JAL", rd=1, imm=8))

    def test_labels_respect_the_text_base(self):
        """assembler: labels are absolute addresses based on text_base"""
        program = assemble("start:\nnop\nend:\nnop", text_base=0x8000)
        self.assertEqual(program.symbols["start"], 0x8000)
        self.assertEqual(program.symbols["end"], 0x8004)

    def test_label_on_the_same_line(self):
        """assembler: a label may share a line with an instruction"""
        program = assemble("here: addi x1, x0, 1\nbeq x0, x0, here")
        self.assertEqual(program.words[1], encode("BEQ", rs1=0, rs2=0, imm=-4))

    def test_duplicate_label_raises(self):
        """assembler: a duplicate label is an error"""
        with self.assertRaises(AssemblerError):
            assemble("a:\nnop\na:\nnop")

    def test_undefined_label_raises(self):
        """assembler: an undefined label is an error"""
        with self.assertRaises(AssemblerError):
            assemble("jal x1, nowhere")


class TestPseudoInstructions(unittest.TestCase):
    def test_simple_expansions(self):
        """assembler: one-to-one pseudo-instructions expand correctly"""
        cases = {
            "nop": encode("ADDI", rd=0, rs1=0, imm=0),
            "mv x1, x2": encode("ADDI", rd=1, rs1=2, imm=0),
            "not x1, x2": encode("XORI", rd=1, rs1=2, imm=-1),
            "neg x1, x2": encode("SUB", rd=1, rs1=0, rs2=2),
            "seqz x1, x2": encode("SLTIU", rd=1, rs1=2, imm=1),
            "snez x1, x2": encode("SLTU", rd=1, rs1=0, rs2=2),
            "ret": encode("JALR", rd=0, rs1=1, imm=0),
            "jr x5": encode("JALR", rd=0, rs1=5, imm=0),
            "j 8": encode("JAL", rd=0, imm=8),
            "beqz x1, 8": encode("BEQ", rs1=1, rs2=0, imm=8),
            "bnez x1, 8": encode("BNE", rs1=1, rs2=0, imm=8),
            "bgez x1, 8": encode("BGE", rs1=1, rs2=0, imm=8),
            "bgt x1, x2, 8": encode("BLT", rs1=2, rs2=1, imm=8),
            "ble x1, x2, 8": encode("BGE", rs1=2, rs2=1, imm=8),
        }
        for source, expected in cases.items():
            self.assertEqual(assemble(source).words[0], expected, source)

    def test_li_small_uses_one_instruction(self):
        """assembler: li with a small constant becomes a single ADDI"""
        program = assemble("li x1, 100")
        self.assertEqual(len(program.words), 1)
        self.assertEqual(program.words[0], encode("ADDI", rd=1, rs1=0, imm=100))

    def test_li_boundaries(self):
        """assembler: li switches to LUI+ADDI outside the 12-bit range"""
        self.assertEqual(len(assemble("li x1, 2047").words), 1)
        self.assertEqual(len(assemble("li x1, -2048").words), 1)
        self.assertEqual(len(assemble("li x1, 2048").words), 2)
        self.assertEqual(len(assemble("li x1, -2049").words), 2)

    def test_li_large_values(self):
        """assembler: li builds any 32-bit constant"""
        from tests.helpers import run_words

        for value in (0x12345678, 0xDEADBEEF, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF, 0x00000800, -1):
            words = assemble(f"li x1, {value}").words
            cpu = run_words(words)
            self.assertEqual(cpu.get_reg(1), value & 0xFFFFFFFF, f"li x1, {value}")

    def test_la_is_pc_relative(self):
        """assembler: la computes a label's address with AUIPC+ADDI"""
        from tests.helpers import run_words

        program = assemble("la x1, target\nnop\ntarget:\nnop", text_base=0)
        cpu = run_words(program.words[:2])
        self.assertEqual(cpu.get_reg(1), program.symbols["target"])

    def test_call_expands_to_auipc_jalr(self):
        """assembler: call expands to AUIPC+JALR and links ra"""
        program = assemble("call f\nnop\nf:\nret")
        self.assertEqual(len(program.words), 4)
        from riscv.decoder import Decoder

        decoder = Decoder()
        self.assertEqual(decoder.decode(program.words[0]).name, "AUIPC")
        self.assertEqual(decoder.decode(program.words[1]).name, "JALR")

    def test_wrong_operand_count_raises(self):
        """assembler: a pseudo-instruction with the wrong operand count raises"""
        with self.assertRaises(AssemblerError):
            assemble("mv x1")
        with self.assertRaises(AssemblerError):
            assemble("nop x1")


class TestDirectives(unittest.TestCase):
    def test_word_half_byte(self):
        """assembler: .word/.half/.byte emit little-endian data"""
        program = assemble(".data\n.word 0x12345678\n.half 0xABCD\n.byte 0xEF, 1")
        data = program.segments[0][1]
        self.assertEqual(data[:4], bytes([0x78, 0x56, 0x34, 0x12]))
        self.assertEqual(data[4:6], bytes([0xCD, 0xAB]))
        self.assertEqual(data[6:8], bytes([0xEF, 0x01]))

    def test_string_directives(self):
        """assembler: .string adds a NUL, .ascii does not"""
        program = assemble('.data\n.string "hi"\n.ascii "hi"')
        self.assertEqual(program.segments[0][1], b"hi\x00hi")

    def test_string_escapes(self):
        """assembler: string escapes are decoded"""
        program = assemble(r'.data' + '\n' + r'.ascii "a\nb\t\\\x41"')
        self.assertEqual(program.segments[0][1], b"a\nb\t\\A")

    def test_space_and_align(self):
        """assembler: .space reserves zeroes and .align pads"""
        program = assemble(".data\n.byte 1\n.align 2\n.word 0xFF\n.space 4")
        data = program.segments[0][1]
        self.assertEqual(len(data), 12)
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1:4], b"\x00\x00\x00")
        self.assertEqual(data[4:8], bytes([0xFF, 0, 0, 0]))

    def test_equ_constants(self):
        """assembler: .equ defines a reusable constant"""
        program = assemble(".equ SIZE, 40\naddi x1, x0, SIZE")
        self.assertEqual(program.words[0], encode("ADDI", rd=1, rs1=0, imm=40))

    def test_data_follows_text(self):
        """assembler: the data section is placed after the text section"""
        program = assemble("nop\nnop\n.data\n.word 1", text_base=0x1000)
        self.assertEqual(program.text_base, 0x1000)
        self.assertEqual(program.data_base, 0x1008)
        self.assertEqual(program.segments[1][0], 0x1008)

    def test_org(self):
        """assembler: .org pads the text section to an address"""
        program = assemble("nop\n.org 0x10\nnop")
        self.assertEqual(len(program.image), 0x14)
        self.assertEqual(program.words[4], encode("ADDI", rd=0, rs1=0, imm=0))

    def test_unknown_directive_raises(self):
        """assembler: an unknown directive is an error"""
        with self.assertRaises(AssemblerError):
            assemble(".frobnicate 1")


class TestErrorReporting(unittest.TestCase):
    def test_errors_carry_a_line_number(self):
        """assembler: errors report the offending line"""
        with self.assertRaises(AssemblerError) as context:
            assemble("nop\nnop\nbogus x1, x2, x3\n")
        self.assertIn("line 3", str(context.exception))

    def test_unknown_instruction_raises(self):
        """assembler: an unknown mnemonic is an error"""
        with self.assertRaises(AssemblerError):
            assemble("mul x1, x2, x3")

    def test_bad_register_raises(self):
        """assembler: an invalid register name is an error"""
        with self.assertRaises(AssemblerError):
            assemble("addi x1, x99, 0")

    def test_out_of_range_immediate_raises(self):
        """assembler: an out-of-range immediate is an error"""
        with self.assertRaises(AssemblerError):
            assemble("addi x1, x0, 5000")

    def test_out_of_range_branch_raises(self):
        """assembler: a branch that is too far away is an error"""
        with self.assertRaises(AssemblerError):
            assemble("beq x1, x2, 8192")


class TestExampleFiles(unittest.TestCase):
    def test_hex_dumps_match_their_sources(self):
        """assembler: the generated examples/*.hex match their .asm sources"""
        for name in sorted(os.listdir(EXAMPLES_DIR)):
            if not name.endswith(".hex"):
                continue
            source = os.path.join(EXAMPLES_DIR, name[:-4] + ".asm")
            self.assertTrue(os.path.exists(source), f"{name} has no .asm source")
            expected = assemble_file(source).words
            words = []
            with open(os.path.join(EXAMPLES_DIR, name), encoding="utf-8") as handle:
                for line in handle:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        words.append(int(line, 16))
            self.assertEqual(
                words,
                expected,
                f"{name} is stale -- rerun python examples/build_examples.py",
            )

    def test_every_example_assembles(self):
        """assembler: every file in examples/ assembles cleanly"""
        names = sorted(name for name in os.listdir(EXAMPLES_DIR) if name.endswith(".asm"))
        self.assertGreaterEqual(len(names), 6, "expected at least six example programs")
        for name in names:
            program = assemble_file(os.path.join(EXAMPLES_DIR, name))
            self.assertGreater(len(program.words), 0, name)
            self.assertEqual(len(program.image) % 4, 0, name)


if __name__ == "__main__":
    unittest.main()
