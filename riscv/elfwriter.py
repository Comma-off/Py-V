"""Write ELF32 little-endian RISC-V executables.

Two jobs:

* let ``riscv.py asm --format elf`` turn a ``.asm`` source into a real ELF, so
  the bundled ELF examples can be built with no toolchain installed;
* give the test suite a way to construct valid *and* deliberately malformed ELF
  files in memory, instead of depending on checked-in binary fixtures.

The output is ordinary ELF32 -- ``readelf`` and ``objdump`` read it -- not a
private format.  It is intentionally minimal: an ELF header, a program header
table, the segments themselves, and optionally a symbol table.  No relocations,
no dynamic linking, no section-header niceties beyond what makes ``objdump``
happy.
"""

import struct

from .elf import (
    EHDR_SIZE,
    ELFCLASS32,
    ELFDATA2LSB,
    ELF_MAGIC,
    EM_RISCV,
    ET_EXEC,
    EV_CURRENT,
    PF_R,
    PF_W,
    PF_X,
    PHDR_SIZE,
    PT_LOAD,
    SHDR_SIZE,
    SHF_ALLOC,
    SHF_EXECINSTR,
    SHF_WRITE,
    SHT_PROGBITS,
    SHT_RISCV_ATTRIBUTES,
    SHT_STRTAB,
    SHT_SYMTAB,
    SYM_SIZE,
    TAG_RISCV_ARCH,
    STT_FUNC,
    STT_NOTYPE,
)

DEFAULT_ALIGN = 0x1000


def build_attributes_section(arch: str = "rv32i2p1") -> bytes:
    """Build a ``.riscv.attributes`` blob recording ``Tag_RISCV_arch``.

    Same format a real toolchain emits, so ``readelf -A`` reads it back and the
    emulator's own architecture auto-detection works on files it produced.
    """
    def uleb128(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    # Tag_File (1) sub-sub-section: tag, uint32 size, then the attributes.
    body = uleb128(TAG_RISCV_ARCH) + arch.encode("ascii") + b"\x00"
    tag_file = b"\x01" + struct.pack("<I", 5 + len(body)) + body
    # Vendor sub-section: uint32 length, vendor name, then the sub-sub-sections.
    vendor = b"riscv\x00"
    subsection = struct.pack("<I", 4 + len(vendor) + len(tag_file)) + vendor + tag_file
    return b"\x41" + subsection  # format version 'A'


class OutputSegment:
    """One ``PT_LOAD`` segment to emit."""

    __slots__ = ("vaddr", "data", "memsz", "flags", "align", "name")

    def __init__(self, vaddr, data, flags, memsz=None, align=DEFAULT_ALIGN, name=".text"):
        self.vaddr = vaddr
        self.data = bytes(data)
        self.memsz = len(self.data) if memsz is None else memsz
        self.flags = flags
        self.align = align
        self.name = name
        if self.memsz < len(self.data):
            raise ValueError("memsz cannot be smaller than the segment's file data")


class ElfWriter:
    """Assemble segments into an ELF32 RISC-V executable image."""

    def __init__(self, entry=0, machine=EM_RISCV, elf_type=ET_EXEC, flags=0, arch=None):
        self.entry = entry
        self.machine = machine
        self.elf_type = elf_type
        self.flags = flags
        #: Architecture string written to ``.riscv.attributes``; ``None`` omits
        #: the section entirely (as a minimal hand-built object would).
        self.arch = arch
        self.segments: list[OutputSegment] = []
        self.symbols: list[tuple[str, int, int, int]] = []  # (name, value, size, type)

    def add_segment(self, vaddr, data, flags, memsz=None, align=DEFAULT_ALIGN, name=".text"):
        self.segments.append(OutputSegment(vaddr, data, flags, memsz, align, name))
        return self

    def add_symbol(self, name, value, size=0, symbol_type=STT_NOTYPE):
        self.symbols.append((name, value, size, symbol_type))
        return self

    # ------------------------------------------------------------------
    def build(self, *, with_symbols: bool = True) -> bytes:
        """Return the complete ELF file as bytes."""
        segments = sorted(self.segments, key=lambda segment: segment.vaddr)
        phnum = len(segments)
        header_bytes = EHDR_SIZE + phnum * PHDR_SIZE

        # Lay the segments out in the file.  Each segment's file offset must be
        # congruent to its vaddr modulo p_align, which is what a real linker
        # arranges and what our own loader validates.
        offset = header_bytes
        placements = []
        for segment in segments:
            align = segment.align if segment.align > 1 else 1
            target = segment.vaddr % align
            if offset % align != target:
                offset += (target - offset) % align
            placements.append(offset)
            offset += len(segment.data)

        symtab_blob, strtab_blob, shdr_blob, shoff, shnum, shstrndx = b"", b"", b"", 0, 0, 0
        if with_symbols and self.symbols:
            symtab_blob, strtab_blob = self._build_symbol_table(segments)
            attributes_blob = build_attributes_section(self.arch) if self.arch else b""
            symtab_offset = offset
            strtab_offset = symtab_offset + len(symtab_blob)
            attributes_offset = strtab_offset + len(strtab_blob)
            shstrtab_offset = attributes_offset + len(attributes_blob)
            shstrtab_blob, name_offsets = _build_shstrtab(
                [".symtab", ".strtab", ".shstrtab", ".riscv.attributes"]
                + [segment.name for segment in segments]
            )
            shoff = shstrtab_offset + len(shstrtab_blob)
            sections = [_null_section()]
            for index, segment in enumerate(segments):
                sections.append(
                    _section(
                        name_offsets[segment.name],
                        SHT_PROGBITS,
                        _section_flags(segment.flags),
                        segment.vaddr,
                        placements[index],
                        len(segment.data),
                    )
                )
            symtab_index = len(sections)
            strtab_index = symtab_index + 1
            sections.append(
                _section(
                    name_offsets[".symtab"], SHT_SYMTAB, 0, 0, symtab_offset,
                    len(symtab_blob), link=strtab_index, info=1,
                    addralign=4, entsize=SYM_SIZE,
                )
            )
            sections.append(
                _section(name_offsets[".strtab"], SHT_STRTAB, 0, 0, strtab_offset,
                         len(strtab_blob), addralign=1)
            )
            if attributes_blob:
                sections.append(
                    _section(
                        name_offsets[".riscv.attributes"], SHT_RISCV_ATTRIBUTES, 0, 0,
                        attributes_offset, len(attributes_blob), addralign=1,
                    )
                )
            shstrndx = len(sections)
            sections.append(
                _section(name_offsets[".shstrtab"], SHT_STRTAB, 0, 0, shstrtab_offset,
                         len(shstrtab_blob), addralign=1)
            )
            shnum = len(sections)
            shdr_blob = b"".join(sections)
            trailer = (
                symtab_blob + strtab_blob + attributes_blob + shstrtab_blob + shdr_blob
            )
        else:
            trailer = b""

        ident = bytearray(16)
        ident[0:4] = ELF_MAGIC
        ident[4] = ELFCLASS32
        ident[5] = ELFDATA2LSB
        ident[6] = EV_CURRENT
        header = bytes(ident) + struct.pack(
            "<HHIIIIIHHHHHH",
            self.elf_type, self.machine, EV_CURRENT, self.entry,
            EHDR_SIZE if phnum else 0,      # e_phoff
            shoff,                          # e_shoff
            self.flags, EHDR_SIZE,
            PHDR_SIZE if phnum else 0, phnum,
            SHDR_SIZE if shnum else 0, shnum, shstrndx,
        )

        phdrs = b"".join(
            struct.pack(
                "<IIIIIIII",
                PT_LOAD, placements[index], segment.vaddr, segment.vaddr,
                len(segment.data), segment.memsz, segment.flags, segment.align,
            )
            for index, segment in enumerate(segments)
        )

        blob = bytearray(header + phdrs)
        for index, segment in enumerate(segments):
            if len(blob) < placements[index]:
                blob.extend(bytes(placements[index] - len(blob)))
            blob[placements[index]:placements[index] + len(segment.data)] = segment.data
        blob.extend(trailer)
        return bytes(blob)

    def _build_symbol_table(self, segments):
        names = bytearray(b"\x00")
        entries = [struct.pack("<IIIBBH", 0, 0, 0, 0, 0, 0)]  # mandatory null symbol
        for name, value, size, symbol_type in self.symbols:
            name_offset = len(names)
            names.extend(name.encode("utf-8") + b"\x00")
            section_index = 1
            for index, segment in enumerate(segments):
                if segment.vaddr <= value < segment.vaddr + segment.memsz:
                    section_index = index + 1
                    break
            info = (1 << 4) | symbol_type  # STB_GLOBAL
            entries.append(struct.pack("<IIIBBH", name_offset, value, size, info, 0, section_index))
        return b"".join(entries), bytes(names)

    def write(self, path, **kwargs) -> int:
        blob = self.build(**kwargs)
        with open(path, "wb") as handle:
            handle.write(blob)
        return len(blob)


def _section_flags(segment_flags: int) -> int:
    flags = SHF_ALLOC
    if segment_flags & PF_W:
        flags |= SHF_WRITE
    if segment_flags & PF_X:
        flags |= SHF_EXECINSTR
    return flags


def _null_section() -> bytes:
    return struct.pack("<IIIIIIIIII", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def _section(name, sh_type, flags, addr, offset, size, link=0, info=0, addralign=4, entsize=0):
    return struct.pack(
        "<IIIIIIIIII", name, sh_type, flags, addr, offset, size, link, info, addralign, entsize
    )


def _build_shstrtab(names):
    blob = bytearray(b"\x00")
    offsets = {}
    for name in names:
        if name in offsets:
            continue
        offsets[name] = len(blob)
        blob.extend(name.encode("utf-8") + b"\x00")
    return bytes(blob), offsets


# ---------------------------------------------------------------------------
# Convenience: assemble a source file straight into an ELF
# ---------------------------------------------------------------------------
def arch_string_for(instruction_set) -> str:
    """The ELF architecture string describing an instruction set."""
    if instruction_set is None:
        return "rv32i2p1"
    extensions = {definition.extension for definition in instruction_set}
    arch = "rv32i2p1"
    if "RV32M" in extensions:
        arch += "_m2p0"
    if "Zicsr" in extensions:
        arch += "_zicsr2p0"
    return arch


def program_to_elf(program, *, entry=None, text_flags=PF_R | PF_X, data_flags=PF_R | PF_W,
                   with_symbols=True, instruction_set=None, arch=None) -> bytes:
    """Turn an assembled :class:`riscv.assembler.Program` into an ELF image.

    The text section becomes an ``R-X`` segment and the data section an ``RW-``
    segment, so the resulting file exercises the same permission handling as a
    linker-produced binary.
    """
    writer = ElfWriter(
        entry=program.entry if entry is None else entry,
        arch=arch if arch is not None else arch_string_for(instruction_set),
    )
    bss_size = getattr(program, "bss_size", 0)
    bss_base = getattr(program, "bss_base", 0)
    data_present = any(address != program.text_base for address, _ in program.segments)

    for address, blob in program.segments:
        is_text = address == program.text_base
        memsz = len(blob)
        if not is_text and bss_size:
            # Fold .bss into the writable segment's memsz, which is exactly how
            # a linker emits it: file bytes for .data, a zero-filled tail beyond.
            memsz = (bss_base + bss_size) - address
        writer.add_segment(
            address,
            blob,
            text_flags if is_text else data_flags,
            memsz=memsz,
            align=DEFAULT_ALIGN,
            name=".text" if is_text else ".data",
        )
    if bss_size and not data_present:
        # No .data at all: .bss becomes its own writable, file-less segment.
        writer.add_segment(bss_base, b"", data_flags, memsz=bss_size,
                           align=DEFAULT_ALIGN, name=".bss")
    if with_symbols:
        text_blob = program.segments[0][1] if program.segments else b""
        text_end = program.text_base + len(text_blob)
        for name, value in sorted(program.symbols.items(), key=lambda item: item[1]):
            in_text = program.text_base <= value < text_end
            writer.add_symbol(name, value, 0, STT_FUNC if in_text else STT_NOTYPE)
    return writer.build(with_symbols=with_symbols)
