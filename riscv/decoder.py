"""Instruction decoding: a 32-bit word in, a :class:`Instruction` record out.

Decoding is a *pure function* of the instruction word -- it never touches CPU or
memory state -- which makes it independently testable and lets the CPU memoise
results in a small cache.

Immediate encodings (RV32I, spec table 2.3).  The scrambling looks arbitrary but
is designed so that every immediate's sign bit is always word bit 31 and so that
field bits keep the same position across formats::

    I-type   imm[11:0]  = word[31:20]                                sign-extend 12
    S-type   imm[11:0]  = word[31:25] | word[11:7]                   sign-extend 12
    B-type   imm[12:1]  = word[31] word[7] word[30:25] word[11:8]    sign-extend 13, bit0=0
    U-type   imm[31:12] = word[31:12]                                already 32-bit
    J-type   imm[20:1]  = word[31] word[19:12] word[20] word[30:21]  sign-extend 21, bit0=0
"""

from .bits import bits, sign_extend
from .exceptions import IllegalInstruction
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


class Instruction:
    """A decoded instruction.

    All operand fields are always populated (raw field extraction is cheap and
    unconditional); ``imm`` is the format-appropriate immediate as an *unsigned
    32-bit* value, already sign-extended.  Fields that are meaningless for a
    given format are still present but should be ignored -- ``fmt`` says which
    ones matter.
    """

    __slots__ = ("raw", "definition", "opcode", "rd", "funct3", "rs1", "rs2", "funct7", "imm")

    def __init__(self, raw, definition, opcode, rd, funct3, rs1, rs2, funct7, imm):
        self.raw = raw
        self.definition = definition
        self.opcode = opcode
        self.rd = rd
        self.funct3 = funct3
        self.rs1 = rs1
        self.rs2 = rs2
        self.funct7 = funct7
        self.imm = imm

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def fmt(self) -> str:
        return self.definition.fmt

    @property
    def imm_signed(self) -> int:
        """The immediate interpreted as a signed 32-bit integer."""
        return self.imm - 0x100000000 if self.imm & 0x80000000 else self.imm

    @property
    def csr(self) -> int:
        """CSR address for a CSR-type instruction (unsigned 12 bits)."""
        return self.imm & 0xFFF

    @property
    def uimm(self) -> int:
        """The 5-bit unsigned immediate the ``CSRRxI`` forms keep in ``rs1``."""
        return self.rs1

    @property
    def shamt(self) -> int:
        """Shift amount for SLLI/SRLI/SRAI (the low 5 bits of the I-immediate)."""
        return self.imm & 0x1F

    def __repr__(self) -> str:
        return f"<Instruction {self.name} raw=0x{self.raw:08X} rd=x{self.rd} rs1=x{self.rs1} rs2=x{self.rs2} imm=0x{self.imm:08X}>"

    def __eq__(self, other):
        return (
            isinstance(other, Instruction)
            and self.raw == other.raw
            and self.definition is other.definition
        )

    def __hash__(self):
        return hash((self.raw, self.definition.name))


# ---------------------------------------------------------------------------
# Immediate extraction (exported individually so tests can hit them directly)
# ---------------------------------------------------------------------------
def decode_i_imm(word: int) -> int:
    return sign_extend(bits(word, 31, 20), 12)


def decode_s_imm(word: int) -> int:
    return sign_extend((bits(word, 31, 25) << 5) | bits(word, 11, 7), 12)


def decode_b_imm(word: int) -> int:
    value = (
        (bits(word, 31, 31) << 12)
        | (bits(word, 7, 7) << 11)
        | (bits(word, 30, 25) << 5)
        | (bits(word, 11, 8) << 1)
    )
    return sign_extend(value, 13)


def decode_csr_imm(word: int) -> int:
    """The CSR address: word[31:20], unsigned (never sign-extended)."""
    return bits(word, 31, 20)


def decode_u_imm(word: int) -> int:
    return word & 0xFFFFF000


def decode_j_imm(word: int) -> int:
    value = (
        (bits(word, 31, 31) << 20)
        | (bits(word, 19, 12) << 12)
        | (bits(word, 20, 20) << 11)
        | (bits(word, 30, 21) << 1)
    )
    return sign_extend(value, 21)


_IMM_DECODERS = {
    R_TYPE: lambda word: 0,
    I_TYPE: decode_i_imm,
    S_TYPE: decode_s_imm,
    B_TYPE: decode_b_imm,
    U_TYPE: decode_u_imm,
    J_TYPE: decode_j_imm,
    N_TYPE: lambda word: 0,
    FENCE_TYPE: decode_i_imm,
    CSR_TYPE: decode_csr_imm,
    AMO_TYPE: lambda word: 0,
}


def extract_fields(word: int) -> dict:
    """Return every fixed-position field of an instruction word.

    Field positions are identical across all RV32I formats, which is exactly why
    they can be pulled out before knowing which instruction this is.
    """
    return {
        "opcode": word & 0x7F,
        "rd": bits(word, 11, 7),
        "funct3": bits(word, 14, 12),
        "rs1": bits(word, 19, 15),
        "rs2": bits(word, 24, 20),
        "funct7": bits(word, 31, 25),
    }


class Decoder:
    """Decodes words against an :class:`~riscv.isa.InstructionSet`."""

    def __init__(self, instruction_set=None, cache_size: int = 4096):
        if instruction_set is None:
            from .instructions import RV32I

            instruction_set = RV32I
        self.instruction_set = instruction_set
        self.cache_size = cache_size
        self._cache: dict[int, Instruction] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def decode(self, word: int, pc: int | None = None) -> Instruction:
        """Decode ``word``; raise :class:`IllegalInstruction` if unrecognised.

        ``pc`` only decorates the exception -- decoding itself is position
        independent.
        """
        word &= 0xFFFFFFFF
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        definition = self.instruction_set.lookup(word)
        if definition is None:
            raise IllegalInstruction(tval=word, pc=pc)

        inst = Instruction(
            raw=word,
            definition=definition,
            opcode=word & 0x7F,
            rd=bits(word, 11, 7),
            funct3=bits(word, 14, 12),
            rs1=bits(word, 19, 15),
            rs2=bits(word, 24, 20),
            funct7=bits(word, 31, 25),
            imm=_IMM_DECODERS[definition.fmt](word),
        )
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[word] = inst
        return inst

    def __repr__(self) -> str:
        return f"<Decoder set={self.instruction_set.name} cached={len(self._cache)}>"
