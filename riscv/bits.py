"""Low-level bit manipulation helpers.

Everything in the emulator represents architectural register/memory values as
*unsigned* Python ints in the range ``[0, 2**32)``.  Signed interpretation is
applied explicitly at the point of use.  Keeping that convention in exactly one
place is what makes 32-bit wrap-around behaviour easy to reason about (and easy
to port to a language such as Luau, which has no native 32-bit integer type
either).
"""

XLEN = 32
MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000


def mask32(value: int) -> int:
    """Truncate ``value`` to 32 bits, returning an unsigned int."""
    return value & MASK32


def to_signed(value: int, bits: int = XLEN) -> int:
    """Reinterpret the low ``bits`` of ``value`` as a two's-complement signed int."""
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def to_unsigned(value: int, bits: int = XLEN) -> int:
    """Reinterpret ``value`` as an unsigned ``bits``-wide integer."""
    return value & ((1 << bits) - 1)


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a ``bits``-wide field to a 32-bit *unsigned* representation.

    ``sign_extend(0xFFF, 12) == 0xFFFFFFFF`` (i.e. -1).
    """
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value |= ~((1 << bits) - 1)
    return value & MASK32


def bit(value: int, index: int) -> int:
    """Extract a single bit."""
    return (value >> index) & 1


def bits(value: int, high: int, low: int) -> int:
    """Extract the inclusive bit field ``value[high:low]``."""
    return (value >> low) & ((1 << (high - low + 1)) - 1)
