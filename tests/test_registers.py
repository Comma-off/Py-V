"""Unit tests for the register file, including x0 immutability."""

import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.exceptions import EmulatorError
from riscv.registers import ABI_NAMES, NUM_REGISTERS, RegisterFile, register_number


class TestRegisterFile(unittest.TestCase):
    def setUp(self):
        self.regs = RegisterFile()

    def test_initial_state_is_zero(self):
        """registers: all 32 registers start at zero"""
        self.assertEqual(len(self.regs), NUM_REGISTERS)
        self.assertEqual(self.regs.snapshot(), [0] * 32)

    def test_read_write_roundtrip(self):
        """registers: every register x1-x31 round-trips"""
        for index in range(1, 32):
            self.regs.write(index, 0x1000 + index)
        for index in range(1, 32):
            self.assertEqual(self.regs.read(index), 0x1000 + index)

    def test_x0_reads_zero(self):
        """x0 immutability: x0 always reads as zero"""
        self.assertEqual(self.regs.read(0), 0)

    def test_x0_ignores_writes(self):
        """x0 immutability: writes to x0 are discarded"""
        for value in (1, -1, 0xFFFFFFFF, 0x80000000):
            self.regs.write(0, value)
            self.assertEqual(self.regs.read(0), 0)

    def test_writes_truncate_to_32_bits(self):
        """registers: writes are truncated to 32 bits"""
        self.regs.write(1, 0x1_2345_6789)
        self.assertEqual(self.regs.read(1), 0x23456789)

    def test_negative_writes_become_twos_complement(self):
        """registers: negative writes are stored as two's complement"""
        self.regs.write(1, -1)
        self.assertEqual(self.regs.read(1), 0xFFFFFFFF)
        self.assertEqual(self.regs.read_signed(1), -1)
        self.regs.write(2, -2147483648)
        self.assertEqual(self.regs.read(2), 0x80000000)
        self.assertEqual(self.regs.read_signed(2), -2147483648)

    def test_out_of_range_index_raises(self):
        """registers: an out-of-range index raises"""
        with self.assertRaises(EmulatorError):
            self.regs.read(32)
        with self.assertRaises(EmulatorError):
            self.regs.write(-1, 0)

    def test_reset_clears_everything(self):
        """registers: reset() zeroes the whole file"""
        for index in range(1, 32):
            self.regs.write(index, 0xFFFFFFFF)
        self.regs.reset()
        self.assertEqual(self.regs.snapshot(), [0] * 32)

    def test_snapshot_and_restore(self):
        """registers: snapshot/restore preserves state and keeps x0 zero"""
        self.regs.write(5, 0xAAAA)
        state = self.regs.snapshot()
        self.regs.write(5, 0)
        self.regs.restore(state)
        self.assertEqual(self.regs.read(5), 0xAAAA)
        broken = [0xFF] * 32
        self.regs.restore(broken)
        self.assertEqual(self.regs.read(0), 0, "restore must keep x0 hardwired")

    def test_subscript_access(self):
        """registers: subscript access by number and by ABI name"""
        self.regs[3] = 7
        self.assertEqual(self.regs[3], 7)
        self.regs["sp"] = 0x8000
        self.assertEqual(self.regs[2], 0x8000)
        self.assertEqual(self.regs["x2"], 0x8000)
        self.regs["zero"] = 5
        self.assertEqual(self.regs["zero"], 0)


class TestRegisterNames(unittest.TestCase):
    def test_abi_names_are_complete(self):
        """registers: there are 32 ABI names"""
        self.assertEqual(len(ABI_NAMES), 32)
        self.assertEqual(ABI_NAMES[0], "zero")
        self.assertEqual(ABI_NAMES[1], "ra")
        self.assertEqual(ABI_NAMES[2], "sp")
        self.assertEqual(ABI_NAMES[10], "a0")
        self.assertEqual(ABI_NAMES[31], "t6")

    def test_register_number_resolution(self):
        """registers: names, numbers and aliases all resolve"""
        self.assertEqual(register_number(5), 5)
        self.assertEqual(register_number("x5"), 5)
        self.assertEqual(register_number("t0"), 5)
        self.assertEqual(register_number("a0"), 10)
        self.assertEqual(register_number("fp"), 8)
        self.assertEqual(register_number("s0"), 8)
        self.assertEqual(register_number("RA"), 1)

    def test_unknown_register_raises(self):
        """registers: an unknown register name raises"""
        with self.assertRaises(EmulatorError):
            register_number("x32")
        with self.assertRaises(EmulatorError):
            register_number("banana")


if __name__ == "__main__":
    unittest.main()
