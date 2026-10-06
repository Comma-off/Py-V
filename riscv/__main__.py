"""Allow ``python -m riscv ...`` as an alias for ``python riscv.py ...``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
