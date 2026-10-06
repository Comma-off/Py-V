"""The architectural register file.

Kept separate from the CPU because it has exactly one interesting rule -- ``x0``
is hardwired to zero -- and that rule deserves to be enforced in one place and
tested on its own.
"""

from .bits import MASK32, to_signed
from .exceptions import EmulatorError

NUM_REGISTERS = 32

#: Standard ABI names, indexed by register number.
ABI_NAMES = (
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
)

#: Every accepted spelling of a register -> its number ("x5", "t0", "fp", "5").
REGISTER_ALIASES: dict[str, int] = {}
for _index, _abi in enumerate(ABI_NAMES):
    REGISTER_ALIASES[f"x{_index}"] = _index
    REGISTER_ALIASES[_abi] = _index
REGISTER_ALIASES["fp"] = 8  # s0 doubles as the frame pointer
del _index, _abi


def register_number(name) -> int:
    """Resolve ``5``, ``"x5"``, ``"t0"`` or ``"5"`` to a register number."""
    if isinstance(name, int):
        if not 0 <= name < NUM_REGISTERS:
            raise EmulatorError(f"register number {name} out of range 0..31")
        return name
    key = str(name).strip().lower()
    if key in REGISTER_ALIASES:
        return REGISTER_ALIASES[key]
    if key.isdigit() and int(key) < NUM_REGISTERS:
        return int(key)
    raise EmulatorError(f"unknown register {name!r}")


class RegisterFile:
    """32 unsigned 32-bit registers, with ``x0`` hardwired to zero.

    Writes to ``x0`` are silently discarded (they are architecturally legal and
    are the canonical way to encode ``nop``/``j``/``ret``); reads always yield 0.
    """

    __slots__ = ("_regs",)

    def __init__(self):
        self._regs = [0] * NUM_REGISTERS

    def reset(self) -> None:
        """Clear every register to zero."""
        regs = self._regs
        for index in range(NUM_REGISTERS):
            regs[index] = 0

    def read(self, index: int) -> int:
        """Read a register as an unsigned 32-bit value."""
        if not 0 <= index < NUM_REGISTERS:
            raise EmulatorError(f"register number {index} out of range 0..31")
        return self._regs[index]

    def write(self, index: int, value: int) -> None:
        """Write a register, truncating to 32 bits.  Writes to x0 are dropped."""
        if not 0 <= index < NUM_REGISTERS:
            raise EmulatorError(f"register number {index} out of range 0..31")
        if index:
            self._regs[index] = value & MASK32

    def read_signed(self, index: int) -> int:
        return to_signed(self.read(index))

    # Convenience: cpu.regs[3] / cpu.regs["sp"] = 0x1000
    def __getitem__(self, key) -> int:
        return self.read(register_number(key))

    def __setitem__(self, key, value) -> None:
        self.write(register_number(key), value)

    def __len__(self) -> int:
        return NUM_REGISTERS

    def __iter__(self):
        return iter(self._regs)

    def snapshot(self) -> list[int]:
        """A plain copy of all 32 values (x0 first)."""
        return list(self._regs)

    def restore(self, values) -> None:
        """Load all 32 values from a snapshot; x0 is forced to zero."""
        values = list(values)
        if len(values) != NUM_REGISTERS:
            raise EmulatorError(f"expected {NUM_REGISTERS} values, got {len(values)}")
        self._regs = [0] + [value & MASK32 for value in values[1:]]

    def format(self, columns: int = 4, abi: bool = True) -> str:
        """Human-readable dump of the whole file."""
        lines = []
        for row in range(0, NUM_REGISTERS, columns):
            parts = []
            for index in range(row, min(row + columns, NUM_REGISTERS)):
                label = f"x{index}" if not abi else f"x{index}/{ABI_NAMES[index]}"
                parts.append(f"{label:>8} = 0x{self._regs[index]:08X}")
            lines.append("  ".join(parts))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<RegisterFile x1=0x{self._regs[1]:08X} ... x31=0x{self._regs[31]:08X}>"
