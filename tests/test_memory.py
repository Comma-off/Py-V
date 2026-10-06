"""Unit tests for the sparse little-endian memory."""

import unittest

from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.exceptions import MemoryError_
from riscv.memory import Memory


class TestScalarAccess(unittest.TestCase):
    def setUp(self):
        self.mem = Memory(size=1 << 16)

    def test_word_roundtrip(self):
        """memory: 32-bit store/load round-trip"""
        self.mem.write_u32(0x100, 0xDEADBEEF)
        self.assertEqual(self.mem.read_u32(0x100), 0xDEADBEEF)

    def test_little_endian_byte_order(self):
        """memory: words are stored little-endian"""
        self.mem.write_u32(0x100, 0x11223344)
        self.assertEqual(self.mem.read_u8(0x100), 0x44)
        self.assertEqual(self.mem.read_u8(0x101), 0x33)
        self.assertEqual(self.mem.read_u8(0x102), 0x22)
        self.assertEqual(self.mem.read_u8(0x103), 0x11)

    def test_halfword_access(self):
        """memory: 16-bit access reads the correct halves"""
        self.mem.write_u32(0x200, 0xAABBCCDD)
        self.assertEqual(self.mem.read_u16(0x200), 0xCCDD)
        self.assertEqual(self.mem.read_u16(0x202), 0xAABB)
        self.mem.write_u16(0x200, 0x1234)
        self.assertEqual(self.mem.read_u32(0x200), 0xAABB1234)

    def test_writes_truncate(self):
        """memory: stores truncate to their width"""
        self.mem.write_u8(0x300, 0x1FF)
        self.assertEqual(self.mem.read_u8(0x300), 0xFF)
        self.mem.write_u16(0x302, 0x1FFFF)
        self.assertEqual(self.mem.read_u16(0x302), 0xFFFF)
        self.mem.write_u32(0x304, 0x1_FFFF_FFFF)
        self.assertEqual(self.mem.read_u32(0x304), 0xFFFFFFFF)

    def test_unwritten_memory_reads_zero(self):
        """memory: unwritten addresses read as zero"""
        self.assertEqual(self.mem.read_u32(0x1234), 0)
        self.assertEqual(self.mem.read_u8(0xABC), 0)
        self.assertEqual(self.mem.read_bytes(0x900, 8), b"\x00" * 8)

    def test_signed_helpers(self):
        """memory: signed read helpers sign-extend"""
        self.mem.write_u8(0x400, 0xFF)
        self.mem.write_u16(0x402, 0x8000)
        self.mem.write_u32(0x404, 0xFFFFFFFF)
        self.assertEqual(self.mem.read_i8(0x400), -1)
        self.assertEqual(self.mem.read_i16(0x402), -32768)
        self.assertEqual(self.mem.read_i32(0x404), -1)


class TestPaging(unittest.TestCase):
    def test_page_crossing_access(self):
        """memory: accesses that straddle a page boundary work"""
        mem = Memory(size=1 << 16, page_bits=12)
        mem.write_u32(0x0FFE, 0x11223344)
        self.assertEqual(mem.read_u32(0x0FFE), 0x11223344)
        self.assertEqual(mem.read_u8(0x0FFE), 0x44)
        self.assertEqual(mem.read_u8(0x1000), 0x22)
        self.assertEqual(mem.allocated_pages, 2)

    def test_lazy_allocation(self):
        """memory: pages are allocated only when written"""
        mem = Memory(size=1 << 20)
        self.assertEqual(mem.allocated_pages, 0)
        mem.read_u32(0x8000)
        self.assertEqual(mem.allocated_pages, 0, "reads must not allocate")
        mem.write_u8(0x8000, 1)
        self.assertEqual(mem.allocated_pages, 1)

    def test_large_address_space_is_cheap(self):
        """memory: a 4 GiB space costs nothing until it is written"""
        mem = Memory(size=1 << 32)
        mem.write_u32(0xFFFFFFF0, 0x12345678)
        self.assertEqual(mem.read_u32(0xFFFFFFF0), 0x12345678)
        self.assertEqual(mem.allocated_pages, 1)

    def test_clear(self):
        """memory: clear() releases every page"""
        mem = Memory(size=1 << 16)
        mem.write_u32(0, 1)
        mem.write_u32(0x4000, 1)
        mem.clear()
        self.assertEqual(mem.allocated_pages, 0)
        self.assertEqual(mem.read_u32(0), 0)


class TestBounds(unittest.TestCase):
    def setUp(self):
        self.mem = Memory(size=0x1000)

    def test_read_past_the_end_raises(self):
        """memory: reads past the end of the address space raise"""
        with self.assertRaises(MemoryError_):
            self.mem.read_u8(0x1000)
        with self.assertRaises(MemoryError_):
            self.mem.read_u32(0x0FFD)

    def test_write_past_the_end_raises(self):
        """memory: writes past the end of the address space raise"""
        with self.assertRaises(MemoryError_):
            self.mem.write_u8(0x1000, 1)
        with self.assertRaises(MemoryError_):
            self.mem.write_bytes(0x0FFE, b"abcd")

    def test_negative_address_raises(self):
        """memory: negative addresses raise"""
        with self.assertRaises(MemoryError_):
            self.mem.read_u8(-1)

    def test_last_valid_byte_is_accessible(self):
        """memory: the final byte of the space is usable"""
        self.mem.write_u8(0x0FFF, 0xAB)
        self.assertEqual(self.mem.read_u8(0x0FFF), 0xAB)
        self.mem.write_u32(0x0FFC, 0xCAFEBABE)
        self.assertEqual(self.mem.read_u32(0x0FFC), 0xCAFEBABE)


class TestBulkAccess(unittest.TestCase):
    def test_bytes_roundtrip(self):
        """memory: byte-string write/read round-trip"""
        mem = Memory(size=1 << 16)
        mem.write_bytes(0x100, b"Hello, RISC-V!")
        self.assertEqual(mem.read_bytes(0x100, 14), b"Hello, RISC-V!")

    def test_load_words(self):
        """memory: load_words writes little-endian words"""
        mem = Memory(size=1 << 16)
        mem.load_words(0x40, [0x00A00093, 0x01400113])
        self.assertEqual(mem.read_u32(0x40), 0x00A00093)
        self.assertEqual(mem.read_u32(0x44), 0x01400113)
        self.assertEqual(mem.read_bytes(0x40, 4), bytes([0x93, 0x00, 0xA0, 0x00]))

    def test_hexdump(self):
        """memory: hexdump renders hex and ASCII"""
        mem = Memory(size=1 << 16)
        mem.write_bytes(0, b"AB\x00\xff")
        dump = mem.hexdump(0, 4)
        self.assertIn("41 42 00 FF", dump)
        self.assertIn("|AB..|", dump)

    def test_zero_length_access(self):
        """memory: zero-length reads and writes are no-ops"""
        mem = Memory(size=0x100)
        self.assertEqual(mem.read_bytes(0, 0), b"")
        mem.write_bytes(0, b"")
        self.assertEqual(mem.allocated_pages, 0)


if __name__ == "__main__":
    unittest.main()
