"""Unit tests for the 32-bit helpers everything else is built on."""

import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.bits import bit, bits, mask32, sign_extend, to_signed, to_unsigned


class TestMasking(unittest.TestCase):
    def test_mask32_truncates(self):
        """mask32 truncates to 32 bits"""
        self.assertEqual(mask32(0x1_0000_0000), 0)
        self.assertEqual(mask32(0x1_FFFF_FFFF), 0xFFFFFFFF)
        self.assertEqual(mask32(-1), 0xFFFFFFFF)
        self.assertEqual(mask32(-2), 0xFFFFFFFE)

    def test_to_signed(self):
        """to_signed reinterprets the top bit as a sign"""
        self.assertEqual(to_signed(0), 0)
        self.assertEqual(to_signed(0x7FFFFFFF), 2147483647)
        self.assertEqual(to_signed(0x80000000), -2147483648)
        self.assertEqual(to_signed(0xFFFFFFFF), -1)

    def test_to_signed_narrow(self):
        """to_signed honours a narrower width"""
        self.assertEqual(to_signed(0xFFF, 12), -1)
        self.assertEqual(to_signed(0x7FF, 12), 2047)
        self.assertEqual(to_signed(0x800, 12), -2048)

    def test_to_unsigned(self):
        """to_unsigned is the inverse of to_signed"""
        self.assertEqual(to_unsigned(-1), 0xFFFFFFFF)
        self.assertEqual(to_unsigned(-1, 12), 0xFFF)
        for value in (0, 1, -1, 2147483647, -2147483648):
            self.assertEqual(to_signed(to_unsigned(value)), value)


class TestSignExtension(unittest.TestCase):
    def test_sign_extend_12_bit(self):
        """sign_extend widens a negative 12-bit field"""
        self.assertEqual(sign_extend(0xFFF, 12), 0xFFFFFFFF)
        self.assertEqual(sign_extend(0x800, 12), 0xFFFFF800)
        self.assertEqual(sign_extend(0x7FF, 12), 0x000007FF)
        self.assertEqual(sign_extend(0x000, 12), 0)

    def test_sign_extend_other_widths(self):
        """sign_extend works at 8, 13, 16, 20 and 21 bits"""
        self.assertEqual(sign_extend(0x80, 8), 0xFFFFFF80)
        self.assertEqual(sign_extend(0x7F, 8), 0x7F)
        self.assertEqual(sign_extend(0x1000, 13), 0xFFFFF000)
        self.assertEqual(sign_extend(0x8000, 16), 0xFFFF8000)
        self.assertEqual(sign_extend(0x100000, 21), 0xFFF00000)

    def test_sign_extend_ignores_high_bits(self):
        """sign_extend ignores bits above the field width"""
        # Only the low 12 bits matter, and here their sign bit (bit 11) is clear.
        self.assertEqual(sign_extend(0xFFFFF001, 12), 0x00000001)
        # ...while here it is set, so the result is negative.
        self.assertEqual(sign_extend(0x00000801, 12), 0xFFFFF801)
        self.assertEqual(sign_extend(0x12345, 4), sign_extend(0x5, 4))


class TestFieldExtraction(unittest.TestCase):
    def test_bit(self):
        """bit() extracts a single bit"""
        self.assertEqual(bit(0b1010, 1), 1)
        self.assertEqual(bit(0b1010, 0), 0)
        self.assertEqual(bit(0x80000000, 31), 1)

    def test_bits(self):
        """bits() extracts an inclusive field"""
        self.assertEqual(bits(0xDEADBEEF, 7, 0), 0xEF)
        self.assertEqual(bits(0xDEADBEEF, 31, 24), 0xDE)
        self.assertEqual(bits(0x00A00093, 31, 20), 0x00A)
        self.assertEqual(bits(0x00A00093, 6, 0), 0x13)


if __name__ == "__main__":
    unittest.main()
