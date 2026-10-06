"""Shared test scaffolding.

Tests are written against the *public* emulator API only -- the same API the
brief specifies for embedders -- so they double as executable documentation of
that API.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riscv import CPU, assemble, encode  # noqa: E402
from riscv.host import halt_on_ecall  # noqa: E402

EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)

#: Address used by tests that need a scratch area; comfortably inside the
#: default 1 MiB memory and clear of any loaded program.
SCRATCH = 0x1000


def make_cpu(words=(), *, base=0, **kwargs) -> CPU:
    """A CPU with ``words`` (32-bit ints) loaded at ``base`` and the PC there."""
    cpu = CPU(**kwargs)
    cpu.load_program(list(words), addr=base)
    return cpu


def run_words(words, steps=None, *, base=0, **kwargs) -> CPU:
    """Load ``words``, execute exactly ``steps`` (default: all of them)."""
    cpu = make_cpu(words, base=base, **kwargs)
    cpu.run(len(words) if steps is None else steps)
    return cpu


def assemble_words(source, text_base=0):
    """Assemble ``source`` and return its instruction words."""
    return assemble(source, text_base=text_base).words


def run_asm(source, steps=None, *, halt_on_exit=True, **kwargs) -> CPU:
    """Assemble and run a snippet.

    Snippets with no ECALL of their own get an exit sequence appended so they
    stop cleanly at the end instead of executing zeroed memory.
    """
    if halt_on_exit and "ecall" not in source.lower():
        source = source + "\n li a7, 93\n li a0, 0\n ecall\n"
    program = assemble(source)
    kwargs.setdefault("ecall_handler", halt_on_ecall)
    cpu = CPU(**kwargs)
    program.load(cpu.memory)
    cpu.set_pc(program.entry)
    cpu.run(10000 if steps is None else steps)
    return cpu


def run_example(name, steps=200000, capture=False):
    """Assemble and run ``examples/<name>.asm``; return ``(cpu, host)``."""
    from riscv import HostEnvironment, assemble_file

    program = assemble_file(os.path.join(EXAMPLES_DIR, f"{name}.asm"))
    host = HostEnvironment(capture=capture, stdout=_NullStream() if capture else None)
    cpu = CPU(ecall_handler=host)
    program.load(cpu.memory)
    cpu.set_pc(program.entry)
    cpu.run(steps)
    return cpu, host


def build_elf_example(name, out_dir=None):
    """Build ``examples/elf/<name>.S`` with the bundled assembler.

    Loaded by explicit file path rather than by putting ``examples/elf`` on
    ``sys.path``, so a module named ``build`` elsewhere cannot shadow it.
    """
    import importlib.util

    global _ELF_BUILDER
    if _ELF_BUILDER is None:
        path = os.path.join(os.path.dirname(EXAMPLES_DIR), "examples", "elf", "build.py")
        spec = importlib.util.spec_from_file_location("riscv_elf_example_builder", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ELF_BUILDER = module
    if out_dir is None:
        return _ELF_BUILDER.build(name)
    return _ELF_BUILDER.build(name, out_dir=out_dir)


_ELF_BUILDER = None

#: The ELF example programs, and the directory holding their sources.
ELF_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "elf"
)
ELF_EXAMPLE_NAMES = ("hello", "arithmetic", "branches", "memory", "functions", "trap")


class _NullStream:
    def write(self, text):  # pragma: no cover - trivial
        return len(text)

    def flush(self):  # pragma: no cover - trivial
        pass


class EmulatorTestCase(unittest.TestCase):
    """Adds assertions that report register values in hex."""

    def assertReg(self, cpu, index, expected, message=""):
        actual = cpu.get_reg(index)
        expected &= 0xFFFFFFFF
        if actual != expected:
            self.fail(
                f"x{index} = 0x{actual:08X}, expected 0x{expected:08X}"
                + (f" ({message})" if message else "")
            )

    def assertRegs(self, cpu, expected: dict):
        for index, value in expected.items():
            self.assertReg(cpu, index, value)

    def assertPC(self, cpu, expected, message=""):
        if cpu.pc != (expected & 0xFFFFFFFF):
            self.fail(
                f"pc = 0x{cpu.pc:08X}, expected 0x{expected & 0xFFFFFFFF:08X}"
                + (f" ({message})" if message else "")
            )

    def assertWord(self, cpu, addr, expected, message=""):
        actual = cpu.read_u32(addr)
        if actual != (expected & 0xFFFFFFFF):
            self.fail(
                f"memory[0x{addr:08X}] = 0x{actual:08X}, expected 0x{expected & 0xFFFFFFFF:08X}"
                + (f" ({message})" if message else "")
            )


__all__ = [
    "CPU",
    "ELF_EXAMPLES_DIR",
    "ELF_EXAMPLE_NAMES",
    "EXAMPLES_DIR",
    "EmulatorTestCase",
    "SCRATCH",
    "assemble",
    "assemble_words",
    "build_elf_example",
    "encode",
    "make_cpu",
    "run_asm",
    "run_example",
    "run_words",
]
