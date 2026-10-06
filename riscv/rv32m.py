"""The RV32M standard extension: integer multiply and divide.

Eight instructions, all R-type in the ``OP`` opcode with ``funct7 = 0000001``::

    MUL     rd = (rs1 * rs2)[31:0]
    MULH    rd = (signed(rs1)   * signed(rs2))[63:32]
    MULHSU  rd = (signed(rs1)   * unsigned(rs2))[63:32]
    MULHU   rd = (unsigned(rs1) * unsigned(rs2))[63:32]
    DIV     rd = signed(rs1)   / signed(rs2)      truncated toward zero
    DIVU    rd = unsigned(rs1) / unsigned(rs2)
    REM     rd = signed(rs1)   % signed(rs2)      sign follows the dividend
    REMU    rd = unsigned(rs1) % unsigned(rs2)

This module *extends* rather than mutates: :data:`~riscv.instructions.RV32I`
stays a faithful pure-integer reference, and :data:`RV32IM` is a copy with these
eight added.  Nothing in the CPU, decoder, encoder, assembler or disassembler
changes to support them -- they are all driven off the instruction table.

Two things make this easy to get wrong, and both are handled explicitly below.

**Python's ``//`` and ``%`` floor; RISC-V truncates toward zero.**
``-7 // 2`` is ``-4`` in Python but RISC-V's ``DIV`` must give ``-3``; ``-7 % 2``
is ``1`` in Python but ``REM`` must give ``-1``, because the remainder takes the
sign of the dividend.  Using the Python operators directly is the classic bug in
a hand-written RV32M implementation, so :func:`divide_signed` and
:func:`remainder_signed` compute on magnitudes and reapply the sign.

**Divide-by-zero and overflow do not trap.**  RV32M has no division exception;
the spec fixes the result instead (spec table 7.1):

===================  ========  =======  ========  =====  =========  ====
Condition            Dividend  Divisor  DIVU      REMU   DIV        REM
===================  ========  =======  ========  =====  =========  ====
Division by zero     x         0        2**32-1   x      -1         x
Overflow (signed)    -2**31    -1       --        --     -2**31     0
===================  ========  =======  ========  =====  =========  ====

Outside those cases the identity ``rs1 == rs2 * DIV(rs1, rs2) + REM(rs1, rs2)``
always holds, and :mod:`tests.test_rv32m` checks it over a large random sample.
"""

from .bits import MASK32, to_signed
from .instructions import RV32I
from .isa import R_TYPE, register_isa

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1

#: RV32I plus the M extension.  Build further extensions from this the same way.
RV32IM = RV32I.copy("RV32IM")


# ---------------------------------------------------------------------------
# Arithmetic helpers (exported so tests can exercise them directly)
# ---------------------------------------------------------------------------
def divide_signed(dividend: int, divisor: int) -> int:
    """Signed division truncated toward zero, with RV32M's special cases.

    ``dividend`` and ``divisor`` are signed 32-bit values; the result is signed.
    """
    if divisor == 0:
        return -1  # all ones, per the spec -- not a trap
    if dividend == INT32_MIN and divisor == -1:
        return INT32_MIN  # the one signed overflow; wraps rather than trapping
    quotient = abs(dividend) // abs(divisor)
    return -quotient if (dividend < 0) != (divisor < 0) else quotient


def remainder_signed(dividend: int, divisor: int) -> int:
    """Signed remainder whose sign follows the dividend, with the special cases."""
    if divisor == 0:
        return dividend
    if dividend == INT32_MIN and divisor == -1:
        return 0
    remainder = abs(dividend) % abs(divisor)
    return -remainder if dividend < 0 else remainder


def divide_unsigned(dividend: int, divisor: int) -> int:
    """Unsigned division; division by zero yields all ones."""
    if divisor == 0:
        return UINT32_MAX
    return dividend // divisor


def remainder_unsigned(dividend: int, divisor: int) -> int:
    """Unsigned remainder; division by zero yields the dividend."""
    if divisor == 0:
        return dividend
    return dividend % divisor


def high_word(product: int) -> int:
    """Bits 63:32 of a 64-bit two's-complement product.

    Python's ``>>`` on a negative int is an arithmetic shift, which is exactly
    the two's-complement high word once masked to 32 bits.
    """
    return (product >> 32) & MASK32


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------
@RV32IM.define("MUL", match=0x02000033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_mul(cpu, inst):
    """Low 32 bits of the product.

    Signedness is irrelevant here: the low half of a two's-complement product is
    the same whether the operands are read as signed or unsigned.
    """
    product = cpu.get_reg(inst.rs1) * cpu.get_reg(inst.rs2)
    cpu.set_reg(inst.rd, product & MASK32)


@RV32IM.define("MULH", match=0x02001033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_mulh(cpu, inst):
    """High 32 bits of a signed x signed product."""
    product = to_signed(cpu.get_reg(inst.rs1)) * to_signed(cpu.get_reg(inst.rs2))
    cpu.set_reg(inst.rd, high_word(product))


@RV32IM.define("MULHSU", match=0x02002033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_mulhsu(cpu, inst):
    """High 32 bits of a signed x *unsigned* product.

    The asymmetric one, used when building wide signed multiplication out of
    32-bit pieces: rs1 is signed, rs2 is not.
    """
    product = to_signed(cpu.get_reg(inst.rs1)) * cpu.get_reg(inst.rs2)
    cpu.set_reg(inst.rd, high_word(product))


@RV32IM.define("MULHU", match=0x02003033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_mulhu(cpu, inst):
    """High 32 bits of an unsigned x unsigned product."""
    product = cpu.get_reg(inst.rs1) * cpu.get_reg(inst.rs2)
    cpu.set_reg(inst.rd, high_word(product))


# ---------------------------------------------------------------------------
# Division and remainder
# ---------------------------------------------------------------------------
@RV32IM.define("DIV", match=0x02004033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_div(cpu, inst):
    """Signed division, truncated toward zero."""
    quotient = divide_signed(
        to_signed(cpu.get_reg(inst.rs1)), to_signed(cpu.get_reg(inst.rs2))
    )
    cpu.set_reg(inst.rd, quotient & MASK32)


@RV32IM.define("DIVU", match=0x02005033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_divu(cpu, inst):
    """Unsigned division."""
    quotient = divide_unsigned(cpu.get_reg(inst.rs1), cpu.get_reg(inst.rs2))
    cpu.set_reg(inst.rd, quotient & MASK32)


@RV32IM.define("REM", match=0x02006033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_rem(cpu, inst):
    """Signed remainder; the sign follows the dividend."""
    remainder = remainder_signed(
        to_signed(cpu.get_reg(inst.rs1)), to_signed(cpu.get_reg(inst.rs2))
    )
    cpu.set_reg(inst.rd, remainder & MASK32)


@RV32IM.define("REMU", match=0x02007033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_remu(cpu, inst):
    """Unsigned remainder."""
    remainder = remainder_unsigned(cpu.get_reg(inst.rs1), cpu.get_reg(inst.rs2))
    cpu.set_reg(inst.rd, remainder & MASK32)


#: The eight instructions this module adds, in encoding order.
RV32M_INSTRUCTIONS = ("MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU")
RV32M_INSTRUCTION_COUNT = 8

register_isa("rv32im", RV32IM)
