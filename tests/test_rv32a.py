"""RV32A tests: atomics, the AMO operand format, and LR/SC reservations.

The LR/SC tests get the most attention because the reservation is the one bit
of real architectural state this extension adds, and the rule that any trap
invalidates it is easy to get wrong (or to forget to test).
"""

import random
import unittest

from tests.helpers import CPU, EmulatorTestCase, SCRATCH, assemble

from riscv.decoder import Decoder
from riscv.encoder import encode
from riscv.exceptions import IllegalInstruction
from riscv.instructions import RV32I
from riscv.isa import available_isas, get_isa
from riscv.rv32a import (
    RV32A_INSTRUCTION_COUNT,
    RV32A_INSTRUCTIONS,
    RV32IA,
    RV32IMA,
)
from riscv.rv32m import RV32IM

MASK32 = 0xFFFFFFFF


def signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
class TestEncodings(unittest.TestCase):
    #: Base encodings straight from the RISC-V spec (opcode AMO, funct3 010,
    #: aq=rl=0, rs2=0 where it doesn't apply).
    REFERENCE = {
        "LR.W": 0x1000202F, "SC.W": 0x1800202F, "AMOSWAP.W": 0x0800202F,
        "AMOADD.W": 0x0000202F, "AMOXOR.W": 0x2000202F, "AMOOR.W": 0x4000202F,
        "AMOAND.W": 0x6000202F, "AMOMIN.W": 0x8000202F, "AMOMAX.W": 0xA000202F,
        "AMOMINU.W": 0xC000202F, "AMOMAXU.W": 0xE000202F,
    }

    def test_reference_encodings(self):
        """rv32a: base encodings match the specification"""
        for name, expected in self.REFERENCE.items():
            word = encode(name, rd=0, rs1=0, rs2=0, instruction_set=RV32IMA)
            self.assertEqual(word, expected, f"{name}: got 0x{word:08X}")

    def test_operand_placement(self):
        """rv32a: register fields land in the right places"""
        # amoadd.w a0, a1, (a2)
        self.assertEqual(
            encode("AMOADD.W", rd=10, rs1=12, rs2=11, instruction_set=RV32IMA), 0x00B6252F
        )
        # lr.w t0, (sp)
        self.assertEqual(encode("LR.W", rd=5, rs1=2, instruction_set=RV32IMA), 0x100122AF)

    def test_aq_rl_bits_are_ignored(self):
        """rv32a: any aq/rl combination still decodes as the same instruction"""
        decoder = Decoder(RV32IMA)
        base = encode("AMOADD.W", rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)
        for aq, rl in ((0, 0), (0, 1), (1, 0), (1, 1)):
            word = base | (aq << 26) | (rl << 25)
            decoded = decoder.decode(word)
            self.assertEqual(decoded.name, "AMOADD.W")
            self.assertEqual((decoded.rd, decoded.rs1, decoded.rs2), (3, 1, 2))

    def test_all_eleven_are_present(self):
        """rv32a: the extension adds exactly eleven instructions"""
        self.assertEqual(len(RV32IA) - len(RV32I), RV32A_INSTRUCTION_COUNT)
        self.assertEqual(len(RV32IMA) - len(RV32IM), RV32A_INSTRUCTION_COUNT)
        for name in RV32A_INSTRUCTIONS:
            self.assertIn(name, RV32IA)
            self.assertIn(name, RV32IMA)

    def test_registered_isa_names(self):
        """rv32a: all four combinations are registered"""
        for name in ("rv32ia", "rv32ima", "rv32ia_zicsr", "rv32ima_zicsr"):
            self.assertIn(name, available_isas())
        self.assertIs(get_isa("rv32ima"), RV32IMA)

    def test_decode_round_trip(self):
        """rv32a: AMOs encode and decode with random operands"""
        decoder = Decoder(RV32IMA)
        rng = random.Random(2024)
        for name in RV32A_INSTRUCTIONS:
            for _ in range(20):
                rd, rs1 = (rng.randrange(32) for _ in range(2))
                rs2 = 0 if name == "LR.W" else rng.randrange(32)
                word = encode(name, rd=rd, rs1=rs1, rs2=rs2, instruction_set=RV32IMA)
                decoded = decoder.decode(word)
                self.assertEqual(decoded.name, name)
                self.assertEqual(decoded.rd, rd)
                self.assertEqual(decoded.rs1, rs1)

    def test_disassembly(self):
        """rv32a: LR prints with no rs2, AMOs print rd,rs2,(rs1)"""
        from riscv.disasm import disassemble

        word = encode("LR.W", rd=5, rs1=2, instruction_set=RV32IMA)
        self.assertEqual(disassemble(word, instruction_set=RV32IMA), "LR.W x5,(x2)")
        word = encode("AMOADD.W", rd=10, rs1=11, rs2=12, instruction_set=RV32IMA)
        self.assertEqual(disassemble(word, instruction_set=RV32IMA), "AMOADD.W x10,x12,(x11)")

    def test_disassembly_round_trip(self):
        """rv32a: disassembly reassembles to the same word"""
        from riscv.disasm import disassemble

        for name in RV32A_INSTRUCTIONS:
            rs2 = 0 if name == "LR.W" else 7
            word = encode(name, rd=5, rs1=6, rs2=rs2, instruction_set=RV32IMA)
            text = disassemble(word, instruction_set=RV32IMA)
            rebuilt = assemble(text, instruction_set=RV32IMA).words[0]
            self.assertEqual(rebuilt, word, text)

    def test_illegal_on_a_machine_without_a(self):
        """rv32a: an A instruction is illegal on RV32IM"""
        word = encode("AMOADD.W", rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)
        cpu = CPU(instruction_set=RV32IM)
        cpu.load_program([word])
        self.assertRaises(IllegalInstruction, cpu.step)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class RV32ATestCase(EmulatorTestCase):
    def setUp(self):
        self.cpu = CPU(instruction_set=RV32IMA)
        self.cpu.write_u32(SCRATCH, 0)

    def amo(self, name, initial, operand):
        """rd=x3 <- old mem[SCRATCH]; mem[SCRATCH] <- combine(old, x2=operand)."""
        self.cpu.write_u32(SCRATCH, initial & MASK32)
        self.cpu.load_program(
            [encode(name, rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)]
        )
        self.cpu.set_reg(1, SCRATCH)
        self.cpu.set_reg(2, operand & MASK32)
        self.cpu.step()
        return self.cpu.get_reg(3), self.cpu.read_u32(SCRATCH)


class TestAtomicMemoryOps(RV32ATestCase):
    def test_amoswap(self):
        """AMOSWAP.W"""
        old, new_mem = self.amo("AMOSWAP.W", 11, 22)
        self.assertEqual((old, new_mem), (11, 22))

    def test_amoadd(self):
        """AMOADD.W"""
        old, new_mem = self.amo("AMOADD.W", 10, 32)
        self.assertEqual((old, new_mem), (10, 42))

    def test_amoadd_wraps(self):
        """AMOADD.W wraps modulo 2**32"""
        old, new_mem = self.amo("AMOADD.W", 0xFFFFFFFF, 2)
        self.assertEqual((old, new_mem), (0xFFFFFFFF, 1))

    def test_amoxor(self):
        """AMOXOR.W"""
        old, new_mem = self.amo("AMOXOR.W", 0xFF00FF00, 0x0F0F0F0F)
        self.assertEqual((old, new_mem), (0xFF00FF00, 0xF00FF00F))

    def test_amoand(self):
        """AMOAND.W"""
        old, new_mem = self.amo("AMOAND.W", 0xFF00FF00, 0x0FF00FF0)
        self.assertEqual((old, new_mem), (0xFF00FF00, 0x0F000F00))

    def test_amoor(self):
        """AMOOR.W"""
        old, new_mem = self.amo("AMOOR.W", 0xFF00FF00, 0x00FF00FF)
        self.assertEqual((old, new_mem), (0xFF00FF00, 0xFFFFFFFF))

    def test_amomin_signed(self):
        """AMOMIN.W compares signed"""
        old, new_mem = self.amo("AMOMIN.W", signed(-5) & MASK32, 3)
        self.assertEqual((signed(old), signed(new_mem)), (-5, -5))
        old, new_mem = self.amo("AMOMIN.W", 3, signed(-5) & MASK32)
        self.assertEqual((old, signed(new_mem)), (3, -5))

    def test_amomax_signed(self):
        """AMOMAX.W compares signed"""
        old, new_mem = self.amo("AMOMAX.W", signed(-5) & MASK32, 3)
        self.assertEqual((signed(old), new_mem), (-5, 3))

    def test_amominu_unsigned(self):
        """AMOMINU.W compares unsigned: -1 is the largest value, not the smallest"""
        old, new_mem = self.amo("AMOMINU.W", 0xFFFFFFFF, 3)
        self.assertEqual((old, new_mem), (0xFFFFFFFF, 3))

    def test_amomaxu_unsigned(self):
        """AMOMAXU.W compares unsigned"""
        old, new_mem = self.amo("AMOMAXU.W", 3, 0xFFFFFFFF)
        self.assertEqual((old, new_mem), (3, 0xFFFFFFFF))

    def test_rd_can_equal_rs1(self):
        """An AMO whose rd aliases rs1 still uses the pre-write address"""
        cpu = CPU(instruction_set=RV32IMA)
        cpu.write_u32(SCRATCH, 99)
        cpu.load_program([encode("AMOADD.W", rd=1, rs1=1, rs2=2, instruction_set=RV32IMA)])
        cpu.set_reg(1, SCRATCH)
        cpu.set_reg(2, 1)
        cpu.step()
        self.assertReg(cpu, 1, 99)  # rd got the *old* memory value
        self.assertWord(cpu, SCRATCH, 100)


class TestLoadReservedStoreConditional(EmulatorTestCase):
    def setUp(self):
        self.cpu = CPU(instruction_set=RV32IMA)
        self.cpu.write_u32(SCRATCH, 7)

    def test_lr_reads_and_reserves(self):
        """LR.W loads the word and records the reservation"""
        self.cpu.load_program([encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA)])
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.step()
        self.assertReg(self.cpu, 1, 7)
        self.assertEqual(self.cpu.reservation, SCRATCH)

    def test_sc_succeeds_right_after_lr(self):
        """SC.W after a matching LR.W stores and returns 0"""
        self.cpu.load_program(
            [
                encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA),
                encode("SC.W", rd=3, rs1=2, rs2=4, instruction_set=RV32IMA),
            ]
        )
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.set_reg(4, 99)
        self.cpu.step()
        self.cpu.step()
        self.assertReg(self.cpu, 3, 0)
        self.assertWord(self.cpu, SCRATCH, 99)
        self.assertIsNone(self.cpu.reservation)

    def test_sc_without_lr_fails(self):
        """SC.W with no reservation at all fails and does not store"""
        self.cpu.load_program([encode("SC.W", rd=3, rs1=2, rs2=4, instruction_set=RV32IMA)])
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.set_reg(4, 99)
        self.cpu.step()
        self.assertReg(self.cpu, 3, 1)
        self.assertWord(self.cpu, SCRATCH, 7)  # unchanged

    def test_sc_to_a_different_address_fails(self):
        """SC.W to an address other than the reservation fails"""
        other = SCRATCH + 0x100
        self.cpu.write_u32(other, 0)
        self.cpu.load_program(
            [
                encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA),
                encode("SC.W", rd=3, rs1=5, rs2=4, instruction_set=RV32IMA),
            ]
        )
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.set_reg(5, other)
        self.cpu.set_reg(4, 99)
        self.cpu.step()
        self.cpu.step()
        self.assertReg(self.cpu, 3, 1)
        self.assertWord(self.cpu, other, 0)

    def test_second_sc_fails(self):
        """A reservation is consumed by its SC.W, successful or not"""
        self.cpu.load_program(
            [
                encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA),
                encode("SC.W", rd=3, rs1=2, rs2=4, instruction_set=RV32IMA),
                encode("SC.W", rd=3, rs1=2, rs2=5, instruction_set=RV32IMA),
            ]
        )
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.set_reg(4, 99)
        self.cpu.set_reg(5, 100)
        self.cpu.step()
        self.cpu.step()
        self.cpu.step()
        self.assertReg(self.cpu, 3, 1)  # the second SC.W fails
        self.assertWord(self.cpu, SCRATCH, 99)  # left at what the first SC.W wrote

    def test_trap_between_lr_and_sc_invalidates_the_reservation(self):
        """A trap taken between LR.W and SC.W clears the reservation"""
        cpu = CPU(instruction_set=RV32IMA, trap_handling="host")
        cpu.write_u32(SCRATCH, 7)
        cpu.load_program([encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA)])
        cpu.set_reg(2, SCRATCH)
        cpu.step()
        self.assertEqual(cpu.reservation, SCRATCH)
        from riscv.exceptions import IllegalInstruction

        cpu.last_exception = None
        try:
            raise IllegalInstruction(tval=0, pc=cpu.pc)
        except IllegalInstruction as exc:
            cpu.take_trap(exc)
        self.assertIsNone(cpu.reservation)

    def test_reset_invalidates_the_reservation(self):
        """Resetting the CPU clears any outstanding reservation"""
        self.cpu.load_program([encode("LR.W", rd=1, rs1=2, instruction_set=RV32IMA)])
        self.cpu.set_reg(2, SCRATCH)
        self.cpu.step()
        self.assertEqual(self.cpu.reservation, SCRATCH)
        self.cpu.reset()
        self.assertIsNone(self.cpu.reservation)

    def test_compare_and_swap_loop(self):
        """A textbook CAS: amo instructions assembled as a retry loop"""
        source = """
            .text
            la    a0, counter
        retry:
            lr.w    t0, (a0)
            addi    t1, t0, 1
            sc.w    t2, t1, (a0)
            bnez    t2, retry
            li      a7, 93
            li      a0, 0
            ecall
            .data
        counter:
            .word 0
        """
        from riscv.host import halt_on_ecall

        program = assemble(source, instruction_set=RV32IMA)
        cpu = CPU(instruction_set=RV32IMA, ecall_handler=halt_on_ecall)
        program.load(cpu.memory)
        cpu.set_pc(program.entry)
        cpu.run(1000)
        self.assertEqual(cpu.read_u32(program.symbols["counter"]), 1)


class TestArchStringMapping(unittest.TestCase):
    def test_arch_strings(self):
        """isa: ELF architecture strings with 'a' map to the A-extension sets"""
        from riscv.elf import isa_name_from_arch_string as resolve

        self.assertEqual(resolve("rv32i2p1_a2p1"), "rv32ia")
        self.assertEqual(resolve("rv32i2p1_m2p0_a2p1"), "rv32ima")
        self.assertEqual(resolve("rv32i2p1_a2p1_zicsr2p0"), "rv32ia_zicsr")
        self.assertEqual(resolve("rv32i2p1_m2p0_a2p1_zicsr2p0"), "rv32ima_zicsr")


if __name__ == "__main__":
    unittest.main()
