#!/usr/bin/env python3
"""Build the ELF examples with the emulator's own assembler -- no toolchain.

    python examples/elf/build.py            # writes build/<name>.elf
    python riscv.py exec examples/elf/build/hello.elf

The same ``.S`` sources also build with a real RISC-V toolchain (see the
Makefile next to this script); both routes produce ELF32 little-endian RISC-V
executables that the emulator loads identically.  This script exists so the
examples -- and the tests that run them -- work on a machine with no
cross-compiler installed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from riscv.assembler import assemble_file  # noqa: E402
from riscv.elfwriter import program_to_elf  # noqa: E402
from riscv.isa import get_isa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")

#: Where the linker script (and the Makefile) place the text section.
TEXT_BASE = 0x00010000

PROGRAMS = (
    "hello", "arithmetic", "branches", "memory", "functions", "trap",
    "multiply", "trap_handler", "timer_interrupt",
)

#: Programs needing an extension beyond the base ISA.  The chosen set is both
#: what assembles the source and what the ELF records in .riscv.attributes, so
#: `riscv.py exec` auto-selects the right machine for the binary.
PROGRAM_ISA = {
    "multiply": "rv32im",
    "trap_handler": "rv32i_zicsr",
    "timer_interrupt": "rv32i_zicsr",
}
DEFAULT_ISA = "rv32i"


def build(name: str, out_dir: str = BUILD_DIR, text_base: int = TEXT_BASE) -> str:
    """Assemble ``<name>.S`` and write ``<out_dir>/<name>.elf``."""
    source = os.path.join(HERE, f"{name}.S")
    instruction_set = get_isa(PROGRAM_ISA.get(name, DEFAULT_ISA))
    program = assemble_file(source, text_base=text_base, instruction_set=instruction_set)
    blob = program_to_elf(program, instruction_set=instruction_set)
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, f"{name}.elf")
    with open(output, "wb") as handle:
        handle.write(blob)
    return output


def main() -> int:
    names = sys.argv[1:] or list(PROGRAMS)
    for name in names:
        path = build(name)
        size = os.path.getsize(path)
        print(f"built {os.path.relpath(path)}  ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
