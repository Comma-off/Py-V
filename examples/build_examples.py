#!/usr/bin/env python3
"""Regenerate the ``.hex`` machine-code dumps next to each ``.asm`` example.

The emulator assembles ``.asm`` files itself, so these dumps are not required to
run anything -- they exist so the example programs are readable (and runnable)
as plain machine code, with no assembler in the loop at all::

    python examples/build_examples.py          # regenerate every .hex
    python riscv.py run examples/loops.hex     # run one directly

Each line is one little-endian instruction word, annotated with its address and
the source line it came from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riscv import assemble_file  # noqa: E402
from riscv.disasm import safe_disassemble  # noqa: E402

EXAMPLES = os.path.dirname(os.path.abspath(__file__))


def build(name: str) -> str:
    source_path = os.path.join(EXAMPLES, f"{name}.asm")
    program = assemble_file(source_path)
    lines = [
        f"# {name}.hex -- generated from {name}.asm by build_examples.py",
        f"# one 32-bit little-endian instruction word per line; text base 0x{program.text_base:08X}",
    ]
    if len(program.segments) > 1:
        lines.append(
            f"# note: this program also has a data section at 0x{program.data_base:08X} "
            "which is NOT included below; run the .asm to get it"
        )
    lines.append("")
    for address, word, _line_no, text in program.listing:
        lines.append(f"{word:08X}  # 0x{address:08X}  {safe_disassemble(word, pc=address):<22} {text.strip()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    names = sorted(
        entry[:-4] for entry in os.listdir(EXAMPLES) if entry.endswith(".asm")
    )
    for name in names:
        output_path = os.path.join(EXAMPLES, f"{name}.hex")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(build(name))
        print(f"wrote {os.path.relpath(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
