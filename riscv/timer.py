"""The machine timer: 64-bit ``mtime`` / ``mtimecmp``, and MTIP.

RISC-V puts the machine timer in the *memory* map, not in CSRs -- there is no
``mtimecmp`` CSR -- so this module implements the two registers as a small
memory-mapped device using the standard CLINT layout that SiFive defined and
QEMU's ``virt`` machine uses::

    base + 0x0000   msip      (software interrupt; not implemented here)
    base + 0x4000   mtimecmp  64-bit, little-endian
    base + 0xBFF8   mtime     64-bit, little-endian

Both registers are genuinely 64 bits wide.  On RV32 a guest reaches them as two
32-bit halves, and this module never truncates the underlying value -- a common
shortcut that breaks the moment ``mtime`` crosses 2**32.

MTIP
----
``mip.MTIP`` is not a register the guest can write.  It is a *wire*: the timer
asserts it exactly while ``mtime >= mtimecmp``, and the only way to clear it is
to move ``mtimecmp`` past ``mtime``.  :attr:`MachineTimer.mtip` is that wire,
and the CSR file reads it through an interrupt source (see
:meth:`riscv.csr.CSRFile.add_interrupt_source`), so ``mip`` reports the live
value rather than a stale copy.

Determinism
-----------
``mtime`` has nothing to do with wall-clock time.  It advances by
``ticks_per_step`` for every step the CPU takes, so a given program always sees
the timer fire at exactly the same instruction, on every machine and every run.
:meth:`advance` and :meth:`set_time` let a test move time explicitly.
"""

MASK64 = (1 << 64) - 1
MASK32 = 0xFFFFFFFF

#: Standard CLINT layout.
CLINT_BASE = 0x02000000
CLINT_SIZE = 0x10000
MSIP_OFFSET = 0x0000
MTIMECMP_OFFSET = 0x4000
MTIME_OFFSET = 0xBFF8

#: ``mtimecmp`` resets to all ones, which is what hardware does: with the
#: comparator at its maximum the timer cannot fire until the guest sets it.
MTIMECMP_RESET = MASK64


class MachineTimer:
    """A 64-bit machine timer with a comparator, mapped CLINT-style.

    Parameters
    ----------
    ticks_per_step:
        How much ``mtime`` advances per CPU step.  ``1`` (the default) makes one
        tick equal one step, which is the simplest thing to reason about in a
        test.  ``0`` freezes time so a test can drive it entirely by hand.
    base:
        Where the CLINT region is mapped.
    """

    __slots__ = ("_mtime", "_mtimecmp", "ticks_per_step", "base", "ticks")

    def __init__(self, ticks_per_step: int = 1, base: int = CLINT_BASE):
        self._mtime = 0
        self._mtimecmp = MTIMECMP_RESET
        self.ticks_per_step = ticks_per_step
        self.base = base
        #: Number of times :meth:`tick` has been called -- the machine's own
        #: step count, useful for asserting determinism.
        self.ticks = 0

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    @property
    def mtime(self) -> int:
        """The 64-bit time counter."""
        return self._mtime

    @mtime.setter
    def mtime(self, value: int) -> None:
        self._mtime = value & MASK64

    @property
    def mtimecmp(self) -> int:
        """The 64-bit comparator."""
        return self._mtimecmp

    @mtimecmp.setter
    def mtimecmp(self, value: int) -> None:
        self._mtimecmp = value & MASK64

    def tick(self, amount: int | None = None) -> int:
        """Advance ``mtime`` by one step's worth of time; returns the new value."""
        self.ticks += 1
        self._mtime = (
            self._mtime + (self.ticks_per_step if amount is None else amount)
        ) & MASK64
        return self._mtime

    def advance(self, amount: int) -> int:
        """Move ``mtime`` forward explicitly, without counting a step.

        This is how a test jumps to just before or just after a deadline
        without executing thousands of instructions.
        """
        self._mtime = (self._mtime + amount) & MASK64
        return self._mtime

    def set_time(self, value: int) -> None:
        self._mtime = value & MASK64

    def reset(self) -> None:
        """Power-on state: time zero, comparator at its maximum, MTIP clear."""
        self._mtime = 0
        self._mtimecmp = MTIMECMP_RESET
        self.ticks = 0

    # ------------------------------------------------------------------
    # The MTIP wire
    # ------------------------------------------------------------------
    @property
    def mtip(self) -> bool:
        """True exactly while ``mtime >= mtimecmp``.

        The comparison is over the full 64 bits.  Comparing only the low words
        would fire early (or not at all) once either value passes 2**32.
        """
        return self._mtime >= self._mtimecmp

    def time_until_fire(self) -> int | None:
        """Ticks remaining before MTIP asserts, or ``None`` if it already has."""
        if self.mtip:
            return None
        return self._mtimecmp - self._mtime

    # ------------------------------------------------------------------
    # Memory-mapped access (CLINT layout)
    # ------------------------------------------------------------------
    def read(self, offset: int, width: int) -> int:
        """Device read at ``offset`` within the CLINT region."""
        return self._extract(self._register_value(offset), offset, width)

    def write(self, offset: int, width: int, value: int) -> None:
        """Device write at ``offset`` within the CLINT region.

        Writes narrower than 64 bits update only the addressed half, which is
        what an RV32 guest does: ``sw`` the low word, then ``sw`` the high word.
        """
        if MTIMECMP_OFFSET <= offset < MTIMECMP_OFFSET + 8:
            self._mtimecmp = self._merge(
                self._mtimecmp, offset - MTIMECMP_OFFSET, width, value
            )
        elif MTIME_OFFSET <= offset < MTIME_OFFSET + 8:
            self._mtime = self._merge(self._mtime, offset - MTIME_OFFSET, width, value)
        # msip and unmapped offsets: software interrupts are not implemented,
        # so writes are dropped rather than silently pretending to work.

    def _register_value(self, offset: int) -> int:
        if MTIMECMP_OFFSET <= offset < MTIMECMP_OFFSET + 8:
            return self._mtimecmp
        if MTIME_OFFSET <= offset < MTIME_OFFSET + 8:
            return self._mtime
        return 0

    @staticmethod
    def _extract(register: int, offset: int, width: int) -> int:
        """Read ``width`` bytes out of a 64-bit register at a byte offset."""
        byte_in_register = offset & 0x7
        shift = byte_in_register * 8
        mask = (1 << (width * 8)) - 1
        return (register >> shift) & mask

    @staticmethod
    def _merge(register: int, byte_offset: int, width: int, value: int) -> int:
        """Write ``width`` bytes into a 64-bit register at a byte offset."""
        shift = byte_offset * 8
        mask = ((1 << (width * 8)) - 1) << shift
        return ((register & ~mask) | ((value << shift) & mask)) & MASK64

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "mtime": self._mtime,
            "mtimecmp": self._mtimecmp,
            "mtip": self.mtip,
            "ticks": self.ticks,
        }

    def format(self) -> str:
        remaining = self.time_until_fire()
        return (
            f"  mtime    = {self._mtime} (0x{self._mtime:016X})\n"
            f"  mtimecmp = {self._mtimecmp} (0x{self._mtimecmp:016X})\n"
            f"  MTIP     = {int(self.mtip)}"
            + ("" if remaining is None else f"   ({remaining} ticks to go)")
        )

    def __repr__(self) -> str:
        return (
            f"<MachineTimer mtime={self._mtime} mtimecmp={self._mtimecmp} "
            f"mtip={int(self.mtip)}>"
        )
