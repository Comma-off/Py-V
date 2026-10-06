"""Synthetic ELF construction for tests.

Tests build ELF files in memory rather than depending on checked-in binaries,
so the ELF suite runs anywhere and every malformed case is expressible.  Valid
files go through :class:`riscv.elfwriter.ElfWriter` (the same writer the CLI
uses); malformed ones are built here field by field, because a correct writer
cannot produce them.
"""

import struct

from tests.helpers import assemble  # noqa: F401  (path setup)

from riscv.elf import (
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
    PT_NOTE,
)
from riscv.elfwriter import ElfWriter

#: Where synthetic programs are placed, matching the ELF examples.
TEXT_BASE = 0x00010000

EM_X86_64 = 62


def make_ident(elf_class=ELFCLASS32, encoding=ELFDATA2LSB, version=EV_CURRENT, magic=ELF_MAGIC):
    ident = bytearray(16)
    ident[0:4] = magic
    ident[4] = elf_class
    ident[5] = encoding
    ident[6] = version
    return bytes(ident)


def make_elf(
    segments=(),
    *,
    entry=TEXT_BASE,
    elf_type=ET_EXEC,
    machine=EM_RISCV,
    ident=None,
    version=EV_CURRENT,
    phoff=None,
    phnum=None,
    phentsize=PHDR_SIZE,
    shoff=0,
    shnum=0,
    shentsize=0,
    shstrndx=0,
    flags=0,
    truncate=None,
    body_offset=None,
):
    """Build an ELF file byte for byte, with every field overridable.

    ``segments`` is a sequence of dicts with ``type``, ``offset``, ``vaddr``,
    ``filesz``, ``memsz``, ``flags``, ``align`` and optional ``data``.  Anything
    omitted gets a sane default, so a malformed case only has to state the one
    field that is wrong.
    """
    segments = [dict(segment) for segment in segments]
    count = len(segments) if phnum is None else phnum
    header_end = EHDR_SIZE + len(segments) * PHDR_SIZE
    cursor = header_end if body_offset is None else body_offset

    for segment in segments:
        data = segment.get("data", b"")
        segment.setdefault("type", PT_LOAD)
        segment.setdefault("vaddr", TEXT_BASE)
        segment.setdefault("flags", PF_R | PF_X)
        segment.setdefault("align", 1)
        if "offset" not in segment:
            segment["offset"] = cursor
            cursor += len(data)
        segment.setdefault("filesz", len(data))
        segment.setdefault("memsz", segment["filesz"])

    header = (make_ident() if ident is None else ident) + struct.pack(
        "<HHIIIIIHHHHHH",
        elf_type, machine, version, entry,
        EHDR_SIZE if phoff is None else phoff,
        shoff, flags, EHDR_SIZE,
        phentsize, count,
        shentsize, shnum, shstrndx,
    )

    phdrs = b"".join(
        struct.pack(
            "<IIIIIIII",
            segment["type"], segment["offset"], segment["vaddr"], segment["vaddr"],
            segment["filesz"], segment["memsz"], segment["flags"], segment["align"],
        )
        for segment in segments
    )

    blob = bytearray(header + phdrs)
    for segment in segments:
        data = segment.get("data", b"")
        if not data:
            continue
        offset = segment["offset"]
        if len(blob) < offset:
            blob.extend(bytes(offset - len(blob)))
        blob[offset:offset + len(data)] = data
    if truncate is not None:
        del blob[truncate:]
    return bytes(blob)


def assemble_elf(source, text_base=TEXT_BASE, entry=None, with_symbols=True):
    """Assemble ``source`` and wrap it in a real ELF, as the CLI's ``asm`` does."""
    from riscv.elfwriter import program_to_elf

    return program_to_elf(
        assemble(source, text_base=text_base), entry=entry, with_symbols=with_symbols
    )


def simple_exit_elf(code=0, text_base=TEXT_BASE):
    """An ELF that exits immediately with ``code`` through the host ABI."""
    return assemble_elf(f"_start:\n li a0, {code}\n li a7, 93\n ecall\n", text_base=text_base)


def two_segment_elf(text_words=(0x00000013,), data_bytes=b"\xEF\xBE\xAD\xDE",
                    text_base=TEXT_BASE, data_base=TEXT_BASE + 0x1000, bss=0):
    """A hand-built ELF with a separate R-X text and RW- data segment."""
    writer = ElfWriter(entry=text_base)
    blob = b"".join(word.to_bytes(4, "little") for word in text_words)
    writer.add_segment(text_base, blob, PF_R | PF_X, align=0x1000, name=".text")
    writer.add_segment(
        data_base, data_bytes, PF_R | PF_W,
        memsz=len(data_bytes) + bss, align=0x1000, name=".data",
    )
    return writer.build(with_symbols=False)


__all__ = [
    "EM_X86_64",
    "PF_R",
    "PF_W",
    "PF_X",
    "PT_LOAD",
    "PT_NOTE",
    "TEXT_BASE",
    "assemble_elf",
    "make_elf",
    "make_ident",
    "simple_exit_elf",
    "two_segment_elf",
]
