"""ELF32 parser tests: valid headers, and every way a file can be malformed."""

import struct
import unittest

from tests.elf_fixtures import (
    EM_X86_64,
    PF_R,
    PF_W,
    PF_X,
    PT_LOAD,
    PT_NOTE,
    TEXT_BASE,
    make_elf,
    make_ident,
    simple_exit_elf,
    two_segment_elf,
)
from tests.helpers import EmulatorTestCase  # noqa: F401  (path setup)

from riscv.elf import (
    EHDR_SIZE,
    ELFCLASS64,
    ELFDATA2MSB,
    ELF_MAGIC,
    EM_RISCV,
    ET_EXEC,
    ET_REL,
    PHDR_SIZE,
    ElfError,
    is_elf,
    load_elf,
    parse_elf,
    permission_text,
    segment_type_name,
)

NOP = 0x00000013


class TestHeaderParsing(unittest.TestCase):
    def setUp(self):
        self.blob = make_elf([{"data": b"\x13\x00\x00\x00"}], entry=0x10000)

    def test_parses_a_valid_header(self):
        """elf: a well-formed ELF32 RISC-V header parses"""
        elf = parse_elf(self.blob)
        self.assertEqual(elf.header.elf_class, 1)
        self.assertEqual(elf.header.machine, EM_RISCV)
        self.assertEqual(elf.header.type, ET_EXEC)
        self.assertEqual(elf.header.entry, 0x10000)

    def test_header_names(self):
        """elf: header fields render as readable names"""
        header = parse_elf(self.blob).header
        self.assertEqual(header.class_name, "ELF32")
        self.assertEqual(header.machine_name, "RISC-V")
        self.assertEqual(header.type_name, "EXEC")
        self.assertEqual(header.endian_name, "little-endian")

    def test_entry_point_is_read_from_e_entry(self):
        """elf: the entry point comes from e_entry"""
        for entry in (0, 0x10000, 0x80000000, 0xFFFFFFFC):
            blob = make_elf([{"data": b"\x13\x00\x00\x00", "vaddr": 0}], entry=entry)
            self.assertEqual(parse_elf(blob).header.entry, entry)

    def test_program_headers_are_parsed(self):
        """elf: program headers are parsed with all eight fields"""
        blob = make_elf(
            [{"data": b"\x13\x00\x00\x00", "vaddr": 0x20000, "flags": PF_R | PF_W,
              "memsz": 0x100, "align": 4}]
        )
        header = parse_elf(blob).program_headers[0]
        self.assertEqual(header.type, PT_LOAD)
        self.assertEqual(header.vaddr, 0x20000)
        self.assertEqual(header.filesz, 4)
        self.assertEqual(header.memsz, 0x100)
        self.assertEqual(header.flags, PF_R | PF_W)
        self.assertEqual(header.align, 4)
        self.assertEqual(header.bss_size, 0xFC)

    def test_is_elf(self):
        """elf: is_elf() recognises the magic without parsing"""
        self.assertTrue(is_elf(self.blob))
        self.assertFalse(is_elf(b"not an elf at all"))
        self.assertFalse(is_elf(b""))

    def test_riscv_flags_decode(self):
        """elf: RISC-V e_flags decode to readable names"""
        self.assertIn("soft-float", parse_elf(make_elf([{"data": b"\x00" * 4}])).header.flag_names())
        blob = make_elf([{"data": b"\x00" * 4}], flags=0x1 | 0x4)
        names = parse_elf(blob).header.flag_names()
        self.assertIn("RVC", names)
        self.assertIn("double-float", names)


class TestRejectsMalformed(unittest.TestCase):
    def assertRejects(self, blob, fragment, **kwargs):
        with self.assertRaises(ElfError) as context:
            parse_elf(blob, **kwargs).build_image()
        message = str(context.exception).lower()
        self.assertIn(fragment.lower(), message, f"message was: {context.exception}")

    def test_invalid_magic(self):
        """elf reject: invalid magic"""
        blob = bytearray(make_elf([{"data": b"\x13\x00\x00\x00"}]))
        blob[0:4] = b"\x7fELG"
        self.assertRejects(bytes(blob), "invalid ELF magic")
        self.assertRejects(b"MZ\x90\x00" + bytes(200), "invalid ELF magic")

    def test_unsupported_elf_class(self):
        """elf reject: ELF64"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], ident=make_ident(elf_class=ELFCLASS64))
        self.assertRejects(blob, "unsupported ELF class: ELF64")

    def test_unsupported_endianness(self):
        """elf reject: big-endian encoding"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], ident=make_ident(encoding=ELFDATA2MSB))
        self.assertRejects(blob, "big-endian")

    def test_unsupported_ident_version(self):
        """elf reject: unknown EI_VERSION"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], ident=make_ident(version=7))
        self.assertRejects(blob, "unsupported ELF version")

    def test_wrong_machine(self):
        """elf reject: wrong machine"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], machine=EM_X86_64)
        self.assertRejects(blob, "unsupported machine: x86-64")
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], machine=40)
        self.assertRejects(blob, "unsupported machine: ARM")

    def test_truncated_identification(self):
        """elf reject: file shorter than e_ident"""
        self.assertRejects(ELF_MAGIC + b"\x01", "truncated ELF file")

    def test_truncated_header(self):
        """elf reject: truncated ELF header"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}])
        self.assertRejects(blob[:EHDR_SIZE - 1], "truncated ELF header")

    def test_truncated_program_header_table(self):
        """elf reject: truncated program header table"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}, {"data": b"\x13\x00\x00\x00"}])
        self.assertRejects(blob[:EHDR_SIZE + PHDR_SIZE + 4], "truncated program header table")

    def test_program_header_table_past_end_of_file(self):
        """elf reject: e_phoff points past the end of the file"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], phoff=0x100000)
        self.assertRejects(blob, "truncated program header table")

    def test_bad_phentsize(self):
        """elf reject: wrong e_phentsize"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], phentsize=PHDR_SIZE + 8)
        self.assertRejects(blob, "unsupported program header size")

    def test_phnum_without_phoff(self):
        """elf reject: e_phnum set but e_phoff is zero"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], phoff=0)
        self.assertRejects(blob, "e_phoff is 0")

    def test_segment_extends_beyond_file(self):
        """elf reject: segment extends beyond the ELF file"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00", "filesz": 0x1000}])
        self.assertRejects(blob, "extends beyond the ELF file")

    def test_segment_offset_past_end_of_file(self):
        """elf reject: p_offset past the end of the file"""
        blob = make_elf([{"offset": 0x9000, "filesz": 4, "memsz": 4}])
        self.assertRejects(blob, "extends beyond the ELF file")

    def test_filesz_greater_than_memsz(self):
        """elf reject: p_filesz > p_memsz"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00" * 4, "memsz": 4}])
        self.assertRejects(blob, "exceeds p_memsz")

    def test_segment_past_the_address_space(self):
        """elf reject: segment past the end of the 32-bit address space"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00", "vaddr": 0xFFFFFFF0, "memsz": 0x100}])
        self.assertRejects(blob, "32-bit address space")

    def test_misaligned_segment(self):
        """elf reject: p_vaddr and p_offset not congruent modulo p_align"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00", "vaddr": 0x10001, "align": 0x1000}])
        self.assertRejects(blob, "congruent")

    def test_no_loadable_segments(self):
        """elf reject: nothing to load"""
        blob = make_elf([{"type": PT_NOTE, "data": b"note"}])
        self.assertRejects(blob, "no PT_LOAD segments")
        self.assertRejects(make_elf([]), "no PT_LOAD segments")

    def test_relocatable_object_is_rejected_as_executable(self):
        """elf reject: a relocatable object is not executable"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], elf_type=ET_REL)
        with self.assertRaises(ElfError) as context:
            load_elf(blob)
        self.assertIn("not an executable", str(context.exception))
        self.assertIn("link it first", str(context.exception))

    def test_relocatable_object_can_still_be_inspected(self):
        """elf: a non-executable file can still be parsed for inspection"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}], elf_type=ET_REL)
        elf = parse_elf(blob, require_executable=False)
        self.assertEqual(elf.header.type_name, "REL")

    def test_missing_file(self):
        """elf reject: unreadable path gives a clean error"""
        with self.assertRaises(ElfError) as context:
            parse_elf("/nonexistent/definitely-not-here.elf")
        self.assertIn("cannot read", str(context.exception))


class TestSegmentSelection(unittest.TestCase):
    def test_only_pt_load_is_loaded(self):
        """elf: non-loadable program headers are ignored"""
        blob = make_elf(
            [
                {"type": PT_NOTE, "data": b"note-data"},
                {"type": PT_LOAD, "data": b"\x13\x00\x00\x00", "vaddr": TEXT_BASE},
                {"type": 0x70000003, "data": b"attrs"},  # PT_RISCV_ATTRIBUTES
            ]
        )
        elf = parse_elf(blob)
        self.assertEqual(len(elf.program_headers), 3)
        image = elf.build_image()
        self.assertEqual(len(image.segments), 1)
        self.assertEqual(image.segments[0].vaddr, TEXT_BASE)

    def test_multiple_loadable_segments(self):
        """elf: multiple PT_LOAD segments are all kept, sorted by address"""
        image = load_elf(two_segment_elf())
        self.assertEqual(len(image.segments), 2)
        self.assertLess(image.segments[0].vaddr, image.segments[1].vaddr)
        self.assertEqual(image.segments[0].permissions, "R-X")
        self.assertEqual(image.segments[1].permissions, "RW-")

    def test_segments_are_sorted_even_if_the_table_is_not(self):
        """elf: segments are sorted by virtual address"""
        blob = make_elf(
            [
                {"data": b"\xAA\xAA\xAA\xAA", "vaddr": 0x20000, "align": 1},
                {"data": b"\xBB\xBB\xBB\xBB", "vaddr": 0x10000, "align": 1},
            ]
        )
        image = load_elf(blob)
        self.assertEqual([segment.vaddr for segment in image.segments], [0x10000, 0x20000])

    def test_permission_helpers(self):
        """elf: permission bits render as RWX text"""
        self.assertEqual(permission_text(PF_R | PF_X), "R-X")
        self.assertEqual(permission_text(PF_R | PF_W), "RW-")
        self.assertEqual(permission_text(PF_R | PF_W | PF_X), "RWX")
        self.assertEqual(permission_text(0), "---")

    def test_segment_type_names(self):
        """elf: program header types have readable names"""
        self.assertEqual(segment_type_name(PT_LOAD), "LOAD")
        self.assertEqual(segment_type_name(PT_NOTE), "NOTE")
        self.assertEqual(segment_type_name(0x70000003), "RISCV_ATTRIBUTES")
        self.assertEqual(segment_type_name(0x12345678), "0x12345678")


class TestSymbols(unittest.TestCase):
    def test_symbols_are_optional_metadata(self):
        """elf: symbols are parsed when present"""
        image = load_elf(simple_exit_elf(3))
        self.assertIsNotNone(image.find_symbol("_start"))
        self.assertEqual(image.find_symbol("_start").value, TEXT_BASE)

    def test_execution_does_not_need_symbols(self):
        """elf: a file with no section headers still loads"""
        blob = make_elf([{"data": b"\x13\x00\x00\x00"}])
        image = load_elf(blob)
        self.assertEqual(image.symbols, [])
        self.assertEqual(len(image.segments), 1)

    def test_damaged_section_table_does_not_break_loading(self):
        """elf: a corrupt section table does not stop the program loading"""
        blob = bytearray(make_elf([{"data": b"\x13\x00\x00\x00"}]))
        struct.pack_into("<I", blob, 32, 0xFFFF0000)  # e_shoff into nowhere
        struct.pack_into("<H", blob, 46, 40)          # e_shentsize
        struct.pack_into("<H", blob, 48, 4)           # e_shnum
        image = load_elf(bytes(blob))
        self.assertEqual(len(image.segments), 1, "segments must still load")
        self.assertEqual(image.symbols, [], "symbols degrade to empty, not an error")

    def test_symbol_lookup_by_address(self):
        """elf: symbols can be looked up by address"""
        image = load_elf(simple_exit_elf(0))
        symbol = image.symbol_for_address(TEXT_BASE)
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.value, TEXT_BASE)
        self.assertIsNone(image.find_symbol("no_such_symbol"))


if __name__ == "__main__":
    unittest.main()
