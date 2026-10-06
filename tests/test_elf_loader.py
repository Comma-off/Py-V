"""Loader tests: placing an image into memory, BSS zero-fill and permissions."""

import unittest

from tests.elf_fixtures import (
    PF_R,
    PF_W,
    PF_X,
    TEXT_BASE,
    make_elf,
    simple_exit_elf,
    two_segment_elf,
)
from tests.helpers import CPU, EmulatorTestCase, encode

from riscv.elf import load_elf
from riscv.elfwriter import ElfWriter
from riscv.exceptions import InstructionAccessFault, StoreAccessFault
from riscv.memory import PERM_R, PERM_RW, PERM_W, PERM_X, Memory, MemoryProtection

NOP = encode("ADDI", rd=0, rs1=0, imm=0)


class TestSegmentPlacement(EmulatorTestCase):
    def test_file_bytes_land_at_p_vaddr(self):
        """loader: file-backed bytes are copied to p_vaddr"""
        blob = make_elf([{"data": bytes([1, 2, 3, 4, 5, 6, 7, 8]), "vaddr": 0x20000, "align": 1}])
        memory = Memory(1 << 20)
        load_elf(blob).load_into(memory)
        self.assertEqual(memory.read_bytes(0x20000, 8), bytes([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_multiple_segments_are_placed_independently(self):
        """loader: each segment lands at its own address"""
        memory = Memory(1 << 20)
        image = load_elf(two_segment_elf(text_words=(NOP, NOP), data_bytes=b"\xEF\xBE\xAD\xDE"))
        image.load_into(memory)
        self.assertEqual(memory.read_u32(TEXT_BASE), NOP)
        self.assertEqual(memory.read_u32(TEXT_BASE + 0x1000), 0xDEADBEEF)

    def test_nothing_outside_the_segments_is_touched(self):
        """loader: memory outside every segment is left alone"""
        memory = Memory(1 << 20)
        memory.write_u32(0x30000, 0xA5A5A5A5)
        load_elf(two_segment_elf()).load_into(memory)
        self.assertEqual(memory.read_u32(0x30000), 0xA5A5A5A5)

    def test_loading_is_idempotent(self):
        """loader: loading the same image twice gives the same memory"""
        image = load_elf(two_segment_elf())
        first, second = Memory(1 << 20), Memory(1 << 20)
        image.load_into(first)
        image.load_into(second)
        image.load_into(second)
        self.assertEqual(
            first.read_bytes(TEXT_BASE, 0x2000), second.read_bytes(TEXT_BASE, 0x2000)
        )

    def test_segment_data_is_not_copied_by_the_parser(self):
        """loader: segment data is a view onto the file, not a copy"""
        image = load_elf(two_segment_elf())
        self.assertIsInstance(image.segments[0].data, memoryview)

    def test_image_geometry(self):
        """loader: the image reports its address range"""
        image = load_elf(two_segment_elf(data_bytes=b"\x00" * 16, bss=0x100))
        self.assertEqual(image.min_address, TEXT_BASE)
        self.assertEqual(image.max_address, TEXT_BASE + 0x1000 + 16 + 0x100)
        self.assertEqual(image.segment_at(TEXT_BASE).permissions, "R-X")
        self.assertIsNone(image.segment_at(0x50000))


class TestBssZeroFill(EmulatorTestCase):
    def test_bss_reads_as_zero(self):
        """loader: the p_memsz tail beyond p_filesz is zero"""
        memory = Memory(1 << 20)
        image = load_elf(two_segment_elf(data_bytes=b"\x11\x22\x33\x44", bss=64))
        image.load_into(memory)
        base = TEXT_BASE + 0x1000
        self.assertEqual(memory.read_u32(base), 0x44332211, "file bytes survive")
        self.assertEqual(memory.read_bytes(base + 4, 64), bytes(64), "bss is zero")

    def test_bss_overwrites_stale_memory(self):
        """loader: BSS is zeroed even if that memory was dirty"""
        memory = Memory(1 << 20)
        base = TEXT_BASE + 0x1000
        memory.write_bytes(base, b"\xFF" * 128)
        image = load_elf(two_segment_elf(data_bytes=b"\x11\x22\x33\x44", bss=64))
        image.load_into(memory)
        self.assertEqual(memory.read_bytes(base + 4, 64), bytes(64))
        self.assertEqual(
            memory.read_u8(base + 68), 0xFF, "only the declared BSS range is cleared"
        )

    def test_large_bss_stays_sparse(self):
        """loader: a huge BSS does not allocate pages it does not need"""
        writer = ElfWriter(entry=TEXT_BASE)
        writer.add_segment(TEXT_BASE, NOP.to_bytes(4, "little"), PF_R | PF_X, align=0x1000)
        writer.add_segment(
            0x100000, b"\x01\x02\x03\x04", PF_R | PF_W, memsz=64 << 20, align=0x1000
        )
        memory = Memory(1 << 30)
        load_elf(writer.build(with_symbols=False)).load_into(memory)
        self.assertLess(
            memory.allocated_pages, 16, "a 64 MiB BSS must not densify the address space"
        )
        self.assertEqual(memory.read_u32(0x100000), 0x04030201)
        self.assertEqual(memory.read_u32(0x100000 + (32 << 20)), 0, "still reads as zero")

    def test_zero_filesz_segment(self):
        """loader: a segment with no file bytes at all is pure BSS"""
        blob = make_elf(
            [
                {"data": b"\x13\x00\x00\x00", "vaddr": TEXT_BASE, "align": 1},
                {"offset": 0, "filesz": 0, "memsz": 0x400, "vaddr": 0x30000,
                 "flags": PF_R | PF_W, "align": 1},
            ]
        )
        memory = Memory(1 << 20)
        load_elf(blob).load_into(memory)
        self.assertEqual(memory.read_bytes(0x30000, 0x400), bytes(0x400))


class TestCpuLoadImage(EmulatorTestCase):
    def test_load_image_sets_the_entry_point(self):
        """loader: cpu.load_image() places segments and sets the PC"""
        image = load_elf(simple_exit_elf(0))
        cpu = CPU(memory_size=1 << 20)
        cpu.load_image(image)
        self.assertPC(cpu, image.entry)
        self.assertEqual(cpu.read_u32(image.entry), cpu.fetch())

    def test_pc_can_be_assigned_directly(self):
        """loader: cpu.pc = image.entry works as documented"""
        image = load_elf(simple_exit_elf(0))
        cpu = CPU(memory_size=1 << 20)
        cpu.load_image(image, set_pc=False)
        cpu.pc = image.entry
        self.assertEqual(cpu.step().name, "ADDI")

    def test_load_image_does_not_require_elf(self):
        """loader: load_image accepts any image-shaped object"""

        class FakeSegment:
            vaddr, data, memsz, flags, index = 0x1000, b"\x13\x00\x00\x00", 8, PF_R | PF_X, 0

        class FakeImage:
            segments = [FakeSegment()]
            entry = 0x1000

        cpu = CPU(memory_size=1 << 20)
        cpu.load_image(FakeImage())
        self.assertPC(cpu, 0x1000)
        self.assertEqual(cpu.read_u32(0x1000), NOP)
        self.assertEqual(cpu.read_u32(0x1004), 0, "the memsz tail is zeroed")

    def test_reset_returns_to_the_elf_entry(self):
        """loader: reset() returns to the ELF entry point"""
        image = load_elf(simple_exit_elf(0))
        cpu = CPU(memory_size=1 << 20)
        cpu.load_image(image)
        cpu.step()
        cpu.reset()
        self.assertPC(cpu, image.entry)

    def test_setup_stack(self):
        """loader: setup_stack points sp at a 16-byte aligned top"""
        cpu = CPU(memory_size=1 << 20)
        stack_pointer = cpu.setup_stack(0x8000A)
        self.assertEqual(stack_pointer, 0x80000)
        self.assertEqual(cpu.get_reg(2), 0x80000)


class TestPermissionEnforcement(EmulatorTestCase):
    def build(self):
        """A CPU with a text (R-X) and data (RW-) segment, permissions enforced."""
        image = load_elf(two_segment_elf(text_words=(NOP, NOP, NOP), data_bytes=b"\x00" * 16))
        cpu = CPU(memory_size=1 << 20)
        cpu.load_image(image, enforce_permissions=True)
        return cpu, image

    def test_permissions_are_off_by_default(self):
        """protection: a CPU has no protection table unless asked"""
        cpu = CPU()
        self.assertIsNone(cpu.protection)
        cpu.load_image(load_elf(two_segment_elf()))
        self.assertIsNone(cpu.protection, "load_image must not enable it silently")

    def test_writing_to_text_faults(self):
        """protection: storing into a read-only segment raises a store fault"""
        cpu, _ = self.build()
        cpu.set_reg(1, TEXT_BASE)
        cpu.write_u32(TEXT_BASE, encode("SW", rs1=1, rs2=0, imm=0))
        with self.assertRaises(StoreAccessFault) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 7)

    def test_reading_from_text_is_allowed(self):
        """protection: a readable segment can still be loaded from"""
        cpu, _ = self.build()
        cpu.set_reg(1, TEXT_BASE)
        cpu.write_u32(TEXT_BASE, encode("LW", rd=2, rs1=1, imm=0))
        cpu.step()
        self.assertEqual(cpu.instret, 1)

    def test_executing_data_faults(self):
        """protection: executing a non-executable segment raises a fetch fault"""
        cpu, _ = self.build()
        data_base = TEXT_BASE + 0x1000
        cpu.write_u32(data_base, NOP)
        cpu.set_pc(data_base)
        with self.assertRaises(InstructionAccessFault) as context:
            cpu.step()
        self.assertEqual(context.exception.cause, 1)

    def test_writing_to_data_is_allowed(self):
        """protection: a writable segment accepts stores"""
        cpu, _ = self.build()
        cpu.set_reg(1, TEXT_BASE + 0x1000)
        cpu.set_reg(2, 0x1234)
        cpu.write_u32(TEXT_BASE, encode("SW", rs1=1, rs2=2, imm=0))
        cpu.step()
        self.assertWord(cpu, TEXT_BASE + 0x1000, 0x1234)

    def test_executing_unmapped_memory_faults(self):
        """protection: jumping into unmapped memory faults at the fetch"""
        cpu, _ = self.build()
        cpu.set_pc(0x70000)
        with self.assertRaises(InstructionAccessFault):
            cpu.step()

    def test_stack_region_is_writable(self):
        """protection: setup_stack registers a readable/writable stack"""
        cpu, _ = self.build()
        top = cpu.setup_stack(0x80000, 0x1000)
        cpu.set_reg(1, top - 16)
        cpu.set_reg(2, 0xABCD)
        cpu.write_u32(TEXT_BASE, encode("SW", rs1=1, rs2=2, imm=0))
        cpu.step()
        self.assertWord(cpu, top - 16, 0xABCD)

    def test_host_api_bypasses_protection(self):
        """protection: the host API is not subject to guest permissions"""
        cpu, _ = self.build()
        cpu.write_u32(TEXT_BASE, 0xDEADBEEF)  # would fault for the guest
        self.assertEqual(cpu.read_u32(TEXT_BASE), 0xDEADBEEF)


class TestMemoryProtectionUnit(EmulatorTestCase):
    def test_default_allows_read_write_not_execute(self):
        """protection: unmapped addresses are RW but never executable"""
        protection = MemoryProtection()
        self.assertTrue(protection.check(0x1000, 4, PERM_R))
        self.assertTrue(protection.check(0x1000, 4, PERM_W))
        self.assertFalse(protection.check(0x1000, 4, PERM_X))

    def test_region_permissions(self):
        """protection: regions grant exactly their own bits"""
        protection = MemoryProtection()
        protection.add_region(0x1000, 0x1000, PERM_R | PERM_X, "text")
        self.assertTrue(protection.check(0x1000, 4, PERM_X))
        self.assertFalse(protection.check(0x1000, 4, PERM_W))
        self.assertEqual(protection.permissions_at(0x1000), PERM_R | PERM_X)

    def test_access_spanning_two_regions(self):
        """protection: an access spanning regions needs the right in both"""
        protection = MemoryProtection()
        protection.add_region(0x1000, 0x10, PERM_R | PERM_X, "text")
        protection.add_region(0x1010, 0x10, PERM_RW, "data")
        self.assertFalse(protection.check(0x100E, 4, PERM_X), "second half is not executable")
        self.assertTrue(protection.check(0x100E, 4, PERM_R), "both halves are readable")

    def test_access_spanning_a_denied_gap(self):
        """protection: a denied region inside a span is not skipped"""
        protection = MemoryProtection(default_flags=PERM_R)
        protection.add_region(0x800, 0x100, 0, "no access")
        protection.permissions_at(0x800)  # prime the internal cache
        self.assertFalse(protection.check(0x500, 0x1000, PERM_R))

    def test_zero_length_access(self):
        """protection: a zero-length access is trivially permitted"""
        self.assertTrue(MemoryProtection().check(0x1000, 0, PERM_X))

    def test_overlap_resolution_is_deterministic(self):
        """protection: on overlap the highest start address wins"""
        protection = MemoryProtection()
        protection.add_region(0x1000, 0x1000, PERM_R, "outer")
        protection.add_region(0x1400, 0x100, PERM_RW, "inner")
        self.assertEqual(protection.permissions_at(0x1400), PERM_RW)
        self.assertEqual(protection.permissions_at(0x1300), PERM_R)


if __name__ == "__main__":
    unittest.main()
