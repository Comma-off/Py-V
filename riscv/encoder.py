"""Instruction encoding -- the exact inverse of :mod:`riscv.decoder`.

The encoder is table-driven off the same :class:`~riscv.isa.InstructionSet` the
decoder uses, so it cannot drift out of sync with the decoder and it picks up
any instruction added by an extension for free::

    >>> hex(encode("ADDI", rd=1, rs1=0, imm=10))
    '0xa00093'

Tests use this to build instruction words directly, without going through the
assembler, so a decoder bug cannot hide behind a matching assembler bug.

Immediate conventions
---------------------
* I/S: signed 12-bit value, ``-2048..2047``.
* Shift immediates (SLLI/SRLI/SRAI): ``shamt`` in ``0..31``.
* B/J: **byte offset relative to the instruction**, even, and within range.
* U (LUI/AUIPC): the 20-bit field as written in assembly -- ``LUI rd, 0x12345``
  loads ``0x12345000``.
"""

from .exceptions import AssemblerError
from .isa import (
    AMO_TYPE,
    B_TYPE,
    CSR_TYPE,
    FENCE_TYPE,
    I_TYPE,
    J_TYPE,
    N_TYPE,
    R_TYPE,
    S_TYPE,
    U_TYPE,
)
from .registers import register_number


def _check_register(value, what):
    number = register_number(value)
    return number


def _check_range(value, low, high, what):
    if not low <= value <= high:
        raise AssemblerError(f"{what} {value} out of range {low}..{high}")
    return value


def encode_r(opcode, rd, funct3, rs1, rs2, funct7) -> int:
    return (
        ((funct7 & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    ) & 0xFFFFFFFF


def encode_i(opcode, rd, funct3, rs1, imm) -> int:
    return (
        ((imm & 0xFFF) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    ) & 0xFFFFFFFF


def encode_s(opcode, funct3, rs1, rs2, imm) -> int:
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((imm & 0x1F) << 7)
        | (opcode & 0x7F)
    ) & 0xFFFFFFFF


def encode_b(opcode, funct3, rs1, rs2, imm) -> int:
    imm &= 0x1FFF
    return (
        (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7)
        | (opcode & 0x7F)
    ) & 0xFFFFFFFF


def encode_u(opcode, rd, imm20) -> int:
    return (((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | (opcode & 0x7F)) & 0xFFFFFFFF


def encode_j(opcode, rd, imm) -> int:
    imm &= 0x1FFFFF
    return (
        (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    ) & 0xFFFFFFFF


def is_shift_immediate(definition) -> bool:
    """True when an I-type's funct7 is fixed, i.e. its immediate is a shamt."""
    return definition.fmt == I_TYPE and ((definition.mask >> 25) & 0x7F) == 0x7F


def encode(name, rd=0, rs1=0, rs2=0, imm=0, instruction_set=None) -> int:
    """Encode one instruction by name using the ISA table.

    Register arguments accept numbers or names (``"sp"``, ``"x2"``).
    """
    if instruction_set is None:
        from .instructions import RV32I

        instruction_set = RV32I
    definition = instruction_set.by_name(name)
    word = definition.match
    fmt = definition.fmt
    rd = _check_register(rd, "rd")
    # In the CSRRxI forms rs1 holds a 5-bit unsigned immediate, not a register
    # number -- the field is the same width, so the range check is the same.
    rs1 = _check_register(rs1, "rs1")
    rs2 = _check_register(rs2, "rs2")

    if fmt == R_TYPE or fmt == AMO_TYPE:
        # AMO's aq/rl bits (word[26:25]) are left zero: this is a single-hart,
        # strictly-in-order core, so they carry no meaning here, and the
        # decoder's mask ignores them on the way back in.
        word |= (rd << 7) | (rs1 << 15) | (rs2 << 20)
    elif fmt == I_TYPE:
        if is_shift_immediate(definition):
            _check_range(imm, 0, 31, f"{definition.name} shift amount")
            word |= (rd << 7) | (rs1 << 15) | ((imm & 0x1F) << 20)
        else:
            _check_range(imm, -2048, 2047, f"{definition.name} immediate")
            word |= (rd << 7) | (rs1 << 15) | ((imm & 0xFFF) << 20)
    elif fmt == S_TYPE:
        _check_range(imm, -2048, 2047, f"{definition.name} immediate")
        imm &= 0xFFF
        word |= ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | ((imm & 0x1F) << 7)
    elif fmt == B_TYPE:
        _check_range(imm, -4096, 4094, f"{definition.name} branch offset")
        if imm & 1:
            raise AssemblerError(f"{definition.name} branch offset {imm} is not even")
        imm &= 0x1FFF
        word |= (
            (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3F) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (((imm >> 1) & 0xF) << 8)
            | (((imm >> 11) & 1) << 7)
        )
    elif fmt == U_TYPE:
        _check_range(imm, -0x80000, 0xFFFFF, f"{definition.name} immediate")
        word |= (rd << 7) | ((imm & 0xFFFFF) << 12)
    elif fmt == J_TYPE:
        _check_range(imm, -(1 << 20), (1 << 20) - 2, f"{definition.name} jump offset")
        if imm & 1:
            raise AssemblerError(f"{definition.name} jump offset {imm} is not even")
        imm &= 0x1FFFFF
        word |= (
            (((imm >> 20) & 1) << 31)
            | (((imm >> 1) & 0x3FF) << 21)
            | (((imm >> 11) & 1) << 20)
            | (((imm >> 12) & 0xFF) << 12)
            | (rd << 7)
        )
    elif fmt == CSR_TYPE:
        # `imm` is the CSR address; `rs1` is either a register or, for the
        # immediate forms, a 5-bit unsigned value -- both live in the same field.
        _check_range(imm, 0, 0xFFF, f"{definition.name} CSR address")
        word |= (rd << 7) | (rs1 << 15) | ((imm & 0xFFF) << 20)
    elif fmt == FENCE_TYPE:
        word |= (rd << 7) | (rs1 << 15) | ((imm & 0xFFF) << 20)
    elif fmt == N_TYPE:
        if rd or rs1 or rs2 or imm:
            raise AssemblerError(f"{definition.name} takes no operands")
    else:  # pragma: no cover - guarded by InstructionDef's format validation
        raise AssemblerError(f"cannot encode format {fmt}")
    return word & 0xFFFFFFFF
