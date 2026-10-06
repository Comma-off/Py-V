#!/usr/bin/env python3
"""Command-line entry point for the RV32I emulator.

    python riscv.py run examples/loops.asm --trace
    python riscv.py --self-test

The implementation lives in the ``riscv`` package next to this file; this script
is a thin wrapper so the documented commands work straight from a checkout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from riscv.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
