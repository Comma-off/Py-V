"""ELF32 little-endian RISC-V reader and program loader.

This module is deliberately independent of the CPU: it turns a file (or a bytes
object) into a :class:`LoadedImage` -- a plain description of "these byte ranges
belong at these addresses with these permissions, and execution starts here".
Nothing here imports :mod:`riscv.cpu`, and :mod:`riscv.cpu` does not import this
module, so the executable format and the ISA stay independent::

    ELF file -> ElfFile (parsed headers)
             -> LoadedImage (what to place in memory)
             -> Memory / CPU

Execution is driven by **program headers** (``PT_LOAD``), not section headers.
Section headers are parsed only for optional extras -- symbol names and the
region table used by ``disasm`` -- and a file with no section headers at all
still loads and runs.

Reference: the ELF-32 specification (Tool Interface Standard, chapters 1-2) plus
the RISC-V ELF psABI for ``EM_RISCV``/``e_flags``.
"""

import struct

from .exceptions import EmulatorError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ELF_MAGIC = b"\x7fELF"

EI_CLASS, EI_DATA, EI_VERSION, EI_OSABI, EI_ABIVERSION = 4, 5, 6, 7, 8
EI_NIDENT = 16

ELFCLASS32, ELFCLASS64 = 1, 2
ELFDATA2LSB, ELFDATA2MSB = 1, 2
EV_CURRENT = 1

ET_NONE, ET_REL, ET_EXEC, ET_DYN, ET_CORE = 0, 1, 2, 3, 4
EM_RISCV = 243

#: Sizes mandated by the ELF32 specification.
EHDR_SIZE = 52
PHDR_SIZE = 32
SHDR_SIZE = 40
SYM_SIZE = 16

# Program header types
PT_NULL, PT_LOAD, PT_DYNAMIC, PT_INTERP, PT_NOTE, PT_SHLIB, PT_PHDR, PT_TLS = range(8)
PT_GNU_EH_FRAME = 0x6474E550
PT_GNU_STACK = 0x6474E551
PT_GNU_RELRO = 0x6474E552
PT_RISCV_ATTRIBUTES = 0x70000003

# Segment permission flags
PF_X, PF_W, PF_R = 0x1, 0x2, 0x4

# Section header types
SHT_NULL, SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB = 0, 1, 2, 3
SHT_NOBITS = 8
SHT_RISCV_ATTRIBUTES = 0x70000003

#: Tags inside a ``.riscv.attributes`` section (RISC-V psABI).  Odd-numbered
#: tags carry a NUL-terminated string, even-numbered ones a ULEB128 -- the
#: convention RISC-V inherited from the ARM attributes format.
TAG_RISCV_STACK_ALIGN = 4
TAG_RISCV_ARCH = 5
TAG_RISCV_UNALIGNED_ACCESS = 6
TAG_RISCV_PRIV_SPEC = 8
SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x2, 0x4

# Symbol types (low nibble of st_info)
STT_NOTYPE, STT_OBJECT, STT_FUNC, STT_SECTION, STT_FILE = 0, 1, 2, 3, 4

#: RISC-V ``e_flags`` bits (psABI).
EF_RISCV_RVC = 0x0001
EF_RISCV_FLOAT_ABI_SINGLE = 0x0002
EF_RISCV_FLOAT_ABI_DOUBLE = 0x0004
EF_RISCV_FLOAT_ABI_QUAD = 0x0006
EF_RISCV_FLOAT_ABI_MASK = 0x0006
EF_RISCV_RVE = 0x0008
EF_RISCV_TSO = 0x0010

_ET_NAMES = {ET_NONE: "NONE", ET_REL: "REL", ET_EXEC: "EXEC", ET_DYN: "DYN", ET_CORE: "CORE"}
_PT_NAMES = {
    PT_NULL: "NULL", PT_LOAD: "LOAD", PT_DYNAMIC: "DYNAMIC", PT_INTERP: "INTERP",
    PT_NOTE: "NOTE", PT_SHLIB: "SHLIB", PT_PHDR: "PHDR", PT_TLS: "TLS",
    PT_GNU_EH_FRAME: "GNU_EH_FRAME", PT_GNU_STACK: "GNU_STACK",
    PT_GNU_RELRO: "GNU_RELRO", PT_RISCV_ATTRIBUTES: "RISCV_ATTRIBUTES",
}
#: Machine numbers worth naming in an error message.
_EM_NAMES = {
    0: "none", 3: "Intel 80386", 8: "MIPS", 20: "PowerPC", 40: "ARM",
    62: "x86-64", 183: "AArch64", 243: "RISC-V",
}


class ElfError(EmulatorError):
    """A malformed, unsupported, or truncated ELF file.

    Carries a message suitable for printing straight to the user -- the CLI
    prints ``error: <message>`` and never a traceback.
    """


def _machine_name(value: int) -> str:
    return _EM_NAMES.get(value, f"unknown (0x{value:X})")


def permission_text(flags: int) -> str:
    """Render segment flags as ``RWX``-style text, e.g. ``R-X``."""
    return (
        ("R" if flags & PF_R else "-")
        + ("W" if flags & PF_W else "-")
        + ("X" if flags & PF_X else "-")
    )


def segment_type_name(value: int) -> str:
    return _PT_NAMES.get(value, f"0x{value:08X}")


# ---------------------------------------------------------------------------
# Header records
# ---------------------------------------------------------------------------
class ElfHeader:
    """The ELF file header (``Elf32_Ehdr``)."""

    __slots__ = (
        "ident", "type", "machine", "version", "entry", "phoff", "shoff",
        "flags", "ehsize", "phentsize", "phnum", "shentsize", "shnum", "shstrndx",
    )

    _FORMAT = "<HHIIIIIHHHHHH"  # everything after e_ident[16]

    def __init__(self, data: bytes):
        if len(data) < EHDR_SIZE:
            raise ElfError(
                f"truncated ELF header: need {EHDR_SIZE} bytes, file has {len(data)}"
            )
        self.ident = bytes(data[:EI_NIDENT])
        (
            self.type, self.machine, self.version, self.entry, self.phoff,
            self.shoff, self.flags, self.ehsize, self.phentsize, self.phnum,
            self.shentsize, self.shnum, self.shstrndx,
        ) = struct.unpack_from(self._FORMAT, data, EI_NIDENT)

    @property
    def elf_class(self) -> int:
        return self.ident[EI_CLASS]

    @property
    def data_encoding(self) -> int:
        return self.ident[EI_DATA]

    @property
    def type_name(self) -> str:
        return _ET_NAMES.get(self.type, f"0x{self.type:X}")

    @property
    def machine_name(self) -> str:
        return _machine_name(self.machine)

    @property
    def class_name(self) -> str:
        return {ELFCLASS32: "ELF32", ELFCLASS64: "ELF64"}.get(
            self.elf_class, f"unknown ({self.elf_class})"
        )

    @property
    def endian_name(self) -> str:
        return {ELFDATA2LSB: "little-endian", ELFDATA2MSB: "big-endian"}.get(
            self.data_encoding, f"unknown ({self.data_encoding})"
        )

    def flag_names(self) -> list[str]:
        """Human-readable RISC-V ``e_flags`` decoding."""
        names = []
        if self.flags & EF_RISCV_RVC:
            names.append("RVC")
        float_abi = self.flags & EF_RISCV_FLOAT_ABI_MASK
        names.append(
            {
                0: "soft-float",
                EF_RISCV_FLOAT_ABI_SINGLE: "single-float",
                EF_RISCV_FLOAT_ABI_DOUBLE: "double-float",
                EF_RISCV_FLOAT_ABI_QUAD: "quad-float",
            }[float_abi]
        )
        if self.flags & EF_RISCV_RVE:
            names.append("RVE")
        if self.flags & EF_RISCV_TSO:
            names.append("TSO")
        return names

    def __repr__(self) -> str:
        return f"<ElfHeader {self.class_name} {self.type_name} entry=0x{self.entry:08X}>"


class ProgramHeader:
    """One program header (``Elf32_Phdr``) -- a loader instruction."""

    __slots__ = ("index", "type", "offset", "vaddr", "paddr", "filesz", "memsz", "flags", "align")

    _FORMAT = "<IIIIIIII"

    def __init__(self, index: int, data: bytes):
        (
            self.type, self.offset, self.vaddr, self.paddr,
            self.filesz, self.memsz, self.flags, self.align,
        ) = struct.unpack_from(self._FORMAT, data)
        self.index = index

    @property
    def is_load(self) -> bool:
        return self.type == PT_LOAD

    @property
    def type_name(self) -> str:
        return segment_type_name(self.type)

    @property
    def permissions(self) -> str:
        return permission_text(self.flags)

    @property
    def vaddr_end(self) -> int:
        """One past the last virtual address the segment occupies in memory."""
        return self.vaddr + self.memsz

    @property
    def bss_size(self) -> int:
        """Bytes that are zero-filled rather than copied from the file."""
        return self.memsz - self.filesz

    def __repr__(self) -> str:
        return (
            f"<ProgramHeader #{self.index} {self.type_name} vaddr=0x{self.vaddr:08X} "
            f"filesz=0x{self.filesz:X} memsz=0x{self.memsz:X} {self.permissions}>"
        )


class SectionHeader:
    """One section header (``Elf32_Shdr``) -- link-time metadata, not a loader instruction."""

    __slots__ = (
        "index", "name_offset", "name", "type", "flags", "addr",
        "offset", "size", "link", "info", "addralign", "entsize",
    )

    _FORMAT = "<IIIIIIIIII"

    def __init__(self, index: int, data: bytes):
        (
            self.name_offset, self.type, self.flags, self.addr, self.offset,
            self.size, self.link, self.info, self.addralign, self.entsize,
        ) = struct.unpack_from(self._FORMAT, data)
        self.index = index
        self.name = ""

    @property
    def is_executable(self) -> bool:
        return bool(self.flags & SHF_EXECINSTR)

    @property
    def occupies_file(self) -> bool:
        return self.type != SHT_NOBITS

    def __repr__(self) -> str:
        return f"<SectionHeader #{self.index} {self.name!r} addr=0x{self.addr:08X} size=0x{self.size:X}>"


class Symbol:
    """One symbol table entry (``Elf32_Sym``)."""

    __slots__ = ("name", "value", "size", "info", "other", "shndx")

    def __init__(self, name, value, size, info, other, shndx):
        self.name = name
        self.value = value
        self.size = size
        self.info = info
        self.other = other
        self.shndx = shndx

    @property
    def type(self) -> int:
        return self.info & 0xF

    @property
    def binding(self) -> int:
        return self.info >> 4

    @property
    def is_function(self) -> bool:
        return self.type == STT_FUNC

    def contains(self, address: int) -> bool:
        return self.value <= address < self.value + max(self.size, 1)

    def __repr__(self) -> str:
        return f"<Symbol {self.name!r} value=0x{self.value:08X} size={self.size}>"


# ---------------------------------------------------------------------------
# The loaded image
# ---------------------------------------------------------------------------
class Segment:
    """A ``PT_LOAD`` segment resolved into "put these bytes here".

    ``data`` is a :class:`memoryview` onto the ELF file buffer, so building an
    image never copies segment contents.
    """

    __slots__ = ("vaddr", "data", "memsz", "flags", "align", "index")

    def __init__(self, vaddr, data, memsz, flags, align=0, index=0):
        self.vaddr = vaddr
        self.data = data
        self.memsz = memsz
        self.flags = flags
        self.align = align
        self.index = index

    @property
    def filesz(self) -> int:
        return len(self.data)

    @property
    def end(self) -> int:
        return self.vaddr + self.memsz

    @property
    def bss_size(self) -> int:
        return self.memsz - self.filesz

    @property
    def permissions(self) -> str:
        return permission_text(self.flags)

    @property
    def readable(self) -> bool:
        return bool(self.flags & PF_R)

    @property
    def writable(self) -> bool:
        return bool(self.flags & PF_W)

    @property
    def executable(self) -> bool:
        return bool(self.flags & PF_X)

    def __repr__(self) -> str:
        return (
            f"<Segment vaddr=0x{self.vaddr:08X} filesz=0x{self.filesz:X} "
            f"memsz=0x{self.memsz:X} {self.permissions}>"
        )


class LoadedImage:
    """Everything needed to place a program in memory and start it.

    This is the hand-off between the executable format and the machine.  It is
    deliberately dumb: a list of segments, an entry point, and optional symbol
    metadata that execution never depends on.
    """

    def __init__(self, segments, entry, *, header=None, symbols=(), sections=(),
                 path=None, arch_string="", isa_name=None):
        self.segments = sorted(segments, key=lambda segment: segment.vaddr)
        self.entry = entry & 0xFFFFFFFF
        self.header = header
        self.symbols = list(symbols)
        self.sections = list(sections)
        self.path = path
        #: What the toolchain said it built this for, e.g. ``"rv32i2p1_m2p0"``.
        #: Purely informational -- execution never depends on it.
        self.arch_string = arch_string
        #: The registered instruction-set name implied by ``arch_string``, or
        #: ``None`` when the file is silent or names something unsupported.
        self.isa_name = isa_name

    # -- geometry -------------------------------------------------------
    @property
    def min_address(self) -> int:
        return min((segment.vaddr for segment in self.segments), default=0)

    @property
    def max_address(self) -> int:
        """One past the highest address any segment occupies."""
        return max((segment.end for segment in self.segments), default=0)

    @property
    def total_memsz(self) -> int:
        return sum(segment.memsz for segment in self.segments)

    def executable_segments(self):
        return [segment for segment in self.segments if segment.executable]

    def regions(self):
        """``[(start, length, flags), ...]`` for every loadable segment."""
        return [(segment.vaddr, segment.memsz, segment.flags) for segment in self.segments]

    def segment_at(self, address: int):
        for segment in self.segments:
            if segment.vaddr <= address < segment.end:
                return segment
        return None

    # -- placement ------------------------------------------------------
    def load_into(self, memory, zero_bss: bool = True) -> None:
        """Write every segment into ``memory``.

        File-backed bytes are copied verbatim; the ``p_memsz - p_filesz`` tail
        (``.bss``) is zeroed.  Zeroing goes through :meth:`Memory.zero_fill`,
        which skips pages that were never allocated -- they already read as
        zero -- so a 100 MiB ``.bss`` costs nothing.
        """
        for segment in self.segments:
            if segment.filesz:
                memory.write_bytes(segment.vaddr, segment.data)
            if zero_bss and segment.bss_size:
                memory.zero_fill(segment.vaddr + segment.filesz, segment.bss_size)

    # -- symbols (optional; never required to execute) -------------------
    def find_symbol(self, name: str):
        for symbol in self.symbols:
            if symbol.name == name:
                return symbol
        return None

    def symbol_for_address(self, address: int):
        """The best-matching symbol covering ``address``, or ``None``."""
        best = None
        for symbol in self.symbols:
            if symbol.value <= address and (symbol.size == 0 or address < symbol.value + symbol.size):
                if best is None or symbol.value > best.value:
                    best = symbol
        return best

    def function_symbols(self):
        return sorted(
            (symbol for symbol in self.symbols if symbol.is_function and symbol.name),
            key=lambda symbol: symbol.value,
        )

    def __repr__(self) -> str:
        return (
            f"<LoadedImage entry=0x{self.entry:08X} segments={len(self.segments)} "
            f"symbols={len(self.symbols)}>"
        )


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------
class ElfFile:
    """A parsed ELF32 little-endian RISC-V file."""

    def __init__(self, data, path=None, *, require_riscv=True, require_executable=False):
        self.data = memoryview(bytes(data)) if not isinstance(data, memoryview) else data
        self.path = path
        self._validate_identification(require_riscv)
        self.header = ElfHeader(bytes(self.data[:EHDR_SIZE]))
        self._validate_header(require_riscv, require_executable)
        self.program_headers = self._parse_program_headers()
        self._sections = None
        self._symbols = None
        self._attributes = None

    # -- validation -----------------------------------------------------
    def _validate_identification(self, require_riscv: bool) -> None:
        data = self.data
        if len(data) < EI_NIDENT:
            raise ElfError(
                f"truncated ELF file: {len(data)} byte(s), need at least {EI_NIDENT}"
            )
        if bytes(data[:4]) != ELF_MAGIC:
            raise ElfError(
                "invalid ELF magic: expected 7F 45 4C 46, got "
                + " ".join(f"{byte:02X}" for byte in bytes(data[:4]))
            )
        elf_class = data[EI_CLASS]
        if elf_class != ELFCLASS32:
            name = {ELFCLASS64: "ELF64"}.get(elf_class, f"unknown ({elf_class})")
            raise ElfError(f"unsupported ELF class: {name} (this emulator is RV32, needs ELF32)")
        encoding = data[EI_DATA]
        if encoding != ELFDATA2LSB:
            name = {ELFDATA2MSB: "big-endian"}.get(encoding, f"unknown ({encoding})")
            raise ElfError(f"unsupported ELF data encoding: {name} (RISC-V here is little-endian)")
        if data[EI_VERSION] != EV_CURRENT:
            raise ElfError(f"unsupported ELF version: {data[EI_VERSION]}")

    def _validate_header(self, require_riscv: bool, require_executable: bool) -> None:
        header = self.header
        if require_riscv and header.machine != EM_RISCV:
            raise ElfError(f"unsupported machine: {header.machine_name} (expected RISC-V)")
        if header.version != EV_CURRENT:
            raise ElfError(f"unsupported ELF object version: {header.version}")
        if require_executable and header.type not in (ET_EXEC, ET_DYN):
            raise ElfError(
                f"not an executable ELF: type is {header.type_name}"
                + (" (relocatable object -- link it first)" if header.type == ET_REL else "")
            )
        if header.phnum and header.phentsize != PHDR_SIZE:
            raise ElfError(
                f"unsupported program header size: {header.phentsize} (expected {PHDR_SIZE})"
            )
        if header.shnum and header.shentsize != SHDR_SIZE:
            raise ElfError(
                f"unsupported section header size: {header.shentsize} (expected {SHDR_SIZE})"
            )

    def _parse_program_headers(self):
        header = self.header
        if header.phnum == 0:
            return []
        if header.phoff == 0:
            raise ElfError("program header count is non-zero but e_phoff is 0")
        table_size = header.phnum * PHDR_SIZE
        end = header.phoff + table_size
        if end > len(self.data):
            raise ElfError(
                f"truncated program header table: needs bytes {header.phoff}..{end} "
                f"but the file is {len(self.data)} byte(s)"
            )
        headers = []
        for index in range(header.phnum):
            start = header.phoff + index * PHDR_SIZE
            headers.append(ProgramHeader(index, bytes(self.data[start:start + PHDR_SIZE])))
        return headers

    # -- sections and symbols (lazy; optional) --------------------------
    @property
    def sections(self):
        """Section headers, parsed on first use.  Never required for execution."""
        if self._sections is None:
            self._sections = self._parse_sections()
        return self._sections

    def _parse_sections(self):
        header = self.header
        if header.shnum == 0 or header.shoff == 0:
            return []
        end = header.shoff + header.shnum * SHDR_SIZE
        if end > len(self.data):
            raise ElfError(
                f"truncated section header table: needs bytes {header.shoff}..{end} "
                f"but the file is {len(self.data)} byte(s)"
            )
        sections = []
        for index in range(header.shnum):
            start = header.shoff + index * SHDR_SIZE
            sections.append(SectionHeader(index, bytes(self.data[start:start + SHDR_SIZE])))
        # Resolve names from the section-header string table.
        if header.shstrndx < len(sections):
            strtab = sections[header.shstrndx]
            if strtab.occupies_file and strtab.offset + strtab.size <= len(self.data):
                blob = bytes(self.data[strtab.offset:strtab.offset + strtab.size])
                for section in sections:
                    section.name = _read_string(blob, section.name_offset)
        return sections

    @property
    def symbols(self):
        """Symbols from ``.symtab``, parsed on first use.  Optional metadata."""
        if self._symbols is None:
            self._symbols = self._parse_symbols()
        return self._symbols

    def _parse_symbols(self):
        try:
            sections = self.sections
        except ElfError:
            return []  # a damaged section table must not stop the program running
        symbols = []
        for section in sections:
            if section.type != SHT_SYMTAB or section.entsize != SYM_SIZE:
                continue
            if section.offset + section.size > len(self.data):
                continue
            strtab = b""
            if section.link < len(sections):
                link = sections[section.link]
                if link.type == SHT_STRTAB and link.offset + link.size <= len(self.data):
                    strtab = bytes(self.data[link.offset:link.offset + link.size])
            count = section.size // SYM_SIZE
            for index in range(count):
                start = section.offset + index * SYM_SIZE
                name_offset, value, size, info, other, shndx = struct.unpack_from(
                    "<IIIBBH", self.data, start
                )
                name = _read_string(strtab, name_offset)
                if name:
                    symbols.append(Symbol(name, value, size, info, other, shndx))
        return symbols

    @property
    def attributes(self) -> dict:
        """Decoded ``.riscv.attributes``, or ``{}``.  Optional metadata."""
        if self._attributes is None:
            self._attributes = self._parse_attributes()
        return self._attributes

    def _parse_attributes(self) -> dict:
        # The attributes live in a section, but a program header of type
        # PT_RISCV_ATTRIBUTES points at the same bytes -- use whichever the file
        # provides, so a stripped binary still reports its architecture.
        spans = []
        try:
            for section in self.sections:
                if section.type == SHT_RISCV_ATTRIBUTES or section.name == ".riscv.attributes":
                    spans.append((section.offset, section.size))
        except ElfError:
            pass
        for header in self.program_headers:
            if header.type == PT_RISCV_ATTRIBUTES:
                spans.append((header.offset, header.filesz))
        for offset, size in spans:
            if size and offset + size <= len(self.data):
                decoded = parse_riscv_attributes(bytes(self.data[offset:offset + size]))
                if decoded:
                    return decoded
        return {}

    @property
    def arch_string(self) -> str:
        """The toolchain's architecture string, e.g. ``"rv32i2p1_m2p0"``."""
        value = self.attributes.get(TAG_RISCV_ARCH, "")
        return value if isinstance(value, str) else ""

    @property
    def isa_name(self) -> str | None:
        """The registered instruction-set name this file needs, if it says."""
        return isa_name_from_arch_string(self.arch_string)

    # -- image construction ---------------------------------------------
    def loadable_segments(self):
        return [header for header in self.program_headers if header.is_load]

    def build_image(self, *, load_symbols: bool = True) -> LoadedImage:
        """Resolve ``PT_LOAD`` headers into a :class:`LoadedImage`.

        Validates each segment against the file's actual size and against the
        32-bit address space before any of it reaches memory.
        """
        segments = []
        for header in self.program_headers:
            if not header.is_load:
                continue  # PT_NOTE, PT_RISCV_ATTRIBUTES, PT_GNU_STACK, ... are not loaded
            self._validate_segment(header)
            data = self.data[header.offset:header.offset + header.filesz]
            segments.append(
                Segment(
                    vaddr=header.vaddr,
                    data=data,
                    memsz=header.memsz,
                    flags=header.flags,
                    align=header.align,
                    index=header.index,
                )
            )
        if not segments:
            raise ElfError("no PT_LOAD segments: nothing to execute")

        symbols = self.symbols if load_symbols else []
        sections = []
        if load_symbols:
            try:
                sections = self.sections
            except ElfError:
                sections = []
        return LoadedImage(
            segments,
            self.header.entry,
            header=self.header,
            symbols=symbols,
            sections=sections,
            path=self.path,
            arch_string=self.arch_string,
            isa_name=self.isa_name,
        )

    def _validate_segment(self, header: ProgramHeader) -> None:
        where = f"segment {header.index}"
        if header.filesz > header.memsz:
            raise ElfError(
                f"{where}: p_filesz (0x{header.filesz:X}) exceeds p_memsz (0x{header.memsz:X})"
            )
        if header.offset + header.filesz > len(self.data):
            raise ElfError(
                f"{where}: extends beyond the ELF file "
                f"(offset 0x{header.offset:X} + 0x{header.filesz:X} > 0x{len(self.data):X})"
            )
        if header.vaddr + header.memsz > 0x1_0000_0000:
            raise ElfError(
                f"{where}: extends past the end of the 32-bit address space "
                f"(vaddr 0x{header.vaddr:08X} + memsz 0x{header.memsz:X})"
            )
        if header.align > 1 and (header.vaddr - header.offset) % header.align:
            raise ElfError(
                f"{where}: p_vaddr (0x{header.vaddr:08X}) and p_offset (0x{header.offset:X}) "
                f"are not congruent modulo p_align (0x{header.align:X})"
            )

    def describe(self) -> str:
        """Multi-line summary used by ``riscv.py elf``."""
        header = self.header
        lines = [
            f"{header.class_name} {header.machine_name} ({header.endian_name})",
            f"Type: {header.type_name}",
            f"Entry: 0x{header.entry:08X}",
            f"Flags: 0x{header.flags:08X} [{', '.join(header.flag_names())}]",
            f"Arch:  {self.arch_string or '(not recorded)'}"
            + (f"  -> {self.isa_name}" if self.isa_name else ""),
            f"Program headers: {header.phnum} at offset {header.phoff}",
            f"Section headers: {header.shnum} at offset {header.shoff}",
            "",
        ]
        if not self.program_headers:
            lines.append("(no program headers)")
        for program_header in self.program_headers:
            marker = "  " if program_header.is_load else "# "
            lines.append(
                f"{marker}{program_header.type_name:<16} "
                f"offset=0x{program_header.offset:06X}  "
                f"vaddr=0x{program_header.vaddr:08X}  "
                f"filesz=0x{program_header.filesz:06X}  "
                f"memsz=0x{program_header.memsz:06X}  "
                f"{program_header.permissions}  "
                f"align=0x{program_header.align:X}"
            )
        return "\n".join(lines)


def _read_uleb128(blob: bytes, offset: int):
    """Read an unsigned LEB128; returns ``(value, next_offset)``."""
    value = 0
    shift = 0
    while offset < len(blob):
        byte = blob[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("over-long LEB128")
    raise ValueError("truncated LEB128")


def parse_riscv_attributes(blob: bytes) -> dict:
    """Decode a ``.riscv.attributes`` section into ``{tag: value}``.

    The section records what the toolchain built the object for -- most usefully
    ``Tag_RISCV_arch`` (5), a string such as ``"rv32i2p1_m2p0"``.  It is pure
    metadata: a malformed or absent section returns ``{}`` rather than raising,
    because nothing about *executing* the program depends on it.
    """
    attributes: dict[int, object] = {}
    try:
        if not blob or blob[0] != 0x41:  # format version 'A'
            return {}
        offset = 1
        while offset + 4 <= len(blob):
            (length,) = struct.unpack_from("<I", blob, offset)
            if length < 4 or offset + length > len(blob):
                break
            end = offset + length
            cursor = offset + 4
            vendor_end = blob.find(b"\x00", cursor, end)
            if vendor_end < 0:
                break
            vendor = blob[cursor:vendor_end].decode("ascii", "replace")
            cursor = vendor_end + 1
            if vendor != "riscv":
                offset = end  # some other vendor's sub-section
                continue
            while cursor + 5 <= end:
                tag = blob[cursor]
                (size,) = struct.unpack_from("<I", blob, cursor + 1)
                if size < 5 or cursor + size > end:
                    break
                sub_end = cursor + size
                inner = cursor + 5
                if tag != 1:  # only Tag_File applies to the whole object
                    cursor = sub_end
                    continue
                while inner < sub_end:
                    attribute_tag, inner = _read_uleb128(blob, inner)
                    if attribute_tag % 2 == 1:  # odd tag -> NUL-terminated string
                        text_end = blob.find(b"\x00", inner, sub_end)
                        if text_end < 0:
                            inner = sub_end
                            break
                        attributes[attribute_tag] = blob[inner:text_end].decode(
                            "utf-8", "replace"
                        )
                        inner = text_end + 1
                    else:                        # even tag -> ULEB128
                        attributes[attribute_tag], inner = _read_uleb128(blob, inner)
                cursor = sub_end
            offset = end
    except (ValueError, struct.error):
        return attributes  # keep whatever was decoded before the damage
    return attributes


def isa_name_from_arch_string(arch: str) -> str | None:
    """Map an ELF architecture string onto a registered instruction-set name.

    ``"rv32i2p1_m2p0"`` -> ``"rv32im"``.  Version suffixes (``2p1``) are dropped.
    Extensions this emulator does not implement yield ``None``, so a caller can
    say so honestly instead of running the binary on the wrong machine.

    ``Zmmul`` -- the multiply-only subset of M that recent GCC records
    alongside ``m`` -- is accepted, because everything it asks for is present.
    Likewise a current toolchain spells ``a`` and ``c`` as the sub-extensions
    they were split into (``Zaamo`` + ``Zalrsc`` for ``a``, ``Zca`` for ``c``,
    since there is no F/D here to put the floating load/store forms ``Zcf``/
    ``Zcd`` would add) -- accepted the same way, as long as *both* halves of
    ``a`` are present; ``Zaamo`` alone is a smaller extension (no LR/SC) this
    emulator does not implement on its own.
    """
    import re

    if not arch:
        return None
    text = arch.lower().strip()
    if not text.startswith("rv32"):
        return None  # rv64/rv128: a different XLEN entirely

    #: Extensions this emulator can satisfy.
    KNOWN = {"i", "m", "a", "c", "zmmul", "zicsr", "zaamo", "zalrsc", "zca"}

    letters = []
    for token in text[4:].split("_"):
        if not token:
            continue
        if token[0] in "zsx":  # multi-letter extension, one per underscore group
            letters.append(re.sub(r"\d+p\d+$", "", token))
            continue
        while token:  # a run of single-letter extensions, e.g. "i2p1m2p0"
            match = re.match(r"^([a-z])(\d+p\d+)?", token)
            if not match:
                return None
            letters.append(match.group(1))
            token = token[match.end():]

    if "i" not in letters:
        return None  # RV32E and friends are not implemented
    if any(letter not in KNOWN for letter in letters):
        return None
    if ("zaamo" in letters) != ("zalrsc" in letters):
        return None  # one half of A alone is a smaller extension we don't have
    name = "rv32i"
    if "m" in letters or "zmmul" in letters:
        name += "m"
    if "a" in letters or ("zaamo" in letters and "zalrsc" in letters):
        name += "a"
    if "c" in letters or "zca" in letters:
        name += "c"
    if "zicsr" in letters:
        name += "_zicsr"
    return name


def _read_string(blob: bytes, offset: int) -> str:
    """Read a NUL-terminated string out of a string table."""
    if offset >= len(blob):
        return ""
    end = blob.find(b"\x00", offset)
    if end < 0:
        end = len(blob)
    return blob[offset:end].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def parse_elf(source, *, require_riscv=True, require_executable=False) -> ElfFile:
    """Parse ``source`` (a path or a bytes-like object) into an :class:`ElfFile`."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return ElfFile(source, path=None, require_riscv=require_riscv,
                       require_executable=require_executable)
    try:
        with open(source, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ElfError(f"cannot read {source}: {exc.strerror}") from None
    return ElfFile(data, path=str(source), require_riscv=require_riscv,
                   require_executable=require_executable)


def load_elf(source, *, require_riscv=True, require_executable=True,
             load_symbols=True) -> LoadedImage:
    """Parse an ELF32 RISC-V executable and return its :class:`LoadedImage`.

    ::

        image = load_elf("program.elf")
        cpu.load_image(image)          # places segments, sets the PC
    """
    return parse_elf(
        source, require_riscv=require_riscv, require_executable=require_executable
    ).build_image(load_symbols=load_symbols)


def is_elf(source) -> bool:
    """True if ``source`` starts with the ELF magic (no other validation)."""
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return bytes(source[:4]) == ELF_MAGIC
        with open(source, "rb") as handle:
            return handle.read(4) == ELF_MAGIC
    except OSError:
        return False
