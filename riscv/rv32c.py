"""The RV32C standard (compressed) extension.

C adds no new *instructions* -- every one of its 16-bit encodings is defined by
the specification as shorthand for an existing RV32I instruction (``C.ADDI``
is ``ADDI``, ``C.MV`` is ``ADD``, and so on). So rather than giving each one its
own :class:`~riscv.isa.InstructionDef` and execute function, :func:`expand`
*decodes* a compressed halfword straight into the equivalent 32-bit RV32I word
and hands that to the ordinary :class:`~riscv.decoder.Decoder` -- the ``ADDI``
executor runs completely unaware that it was reached via a 2-byte encoding.
That also means ``len(RV32IC) == len(RV32I)``: compressed adds an alternate
*encoding*, not new mnemonics, which is exactly what the architecture says it
is.

The one thing :func:`expand` has to hand back on the side is the canonical
``C.*`` name, purely so tracing and disassembly can show what was actually in
memory (``C.ADDI``, not ``ADDI``) without that leaking into the cached,
shared :class:`~riscv.decoder.Instruction` object the expanded word decodes
to -- the same 32-bit word can equally be reached by a real 4-byte
instruction elsewhere in the program, and must still show *that* correctly.
See :meth:`riscv.cpu.CPU.fetch_instruction`, which is where the two paths
(compressed and not) rejoin.

Immediate scrambles
--------------------
Several C formats scatter an immediate's bits across the word in a
non-contiguous order (chosen, per the spec's rationale, to reuse as much of
the non-compressed decode logic in hardware as possible). Rather than writing
each as a hand-rolled chain of shifts -- easy to get subtly wrong and hard to
check by eye -- each is a declarative list of ``(destination_bit,
source_bit)`` pairs taken straight from the spec's own ``imm[...]`` notation,
assembled by :func:`_scramble`. Every one of them is verified in
:mod:`tests.test_rv32c` against a real GNU assembler.

Reserved vs. HINT encodings
----------------------------
The spec distinguishes encodings that are architecturally *reserved* (must
trap as illegal) from ones that are merely *HINTs* (legal, defined to have no
effect beyond what the base instruction already does, reserved for future use
as a microarchitectural hint). This module only special-cases the former --
``C.ADDI4SPN``/``C.LUI``/``C.ADDI16SP`` with an all-zero immediate, an
out-of-range RV32 shift amount, the RV64-only ``C.SUBW``/``C.ADDW`` slot, and
``rd=x0`` on ``C.JR``. Every HINT (``C.NOP``, ``C.LI x0,...``, ``C.MV
x0,...``, a zero-immediate ``C.ADDI``, ...) is left to expand and execute
completely normally: writing to ``x0`` is already a no-op, so "doing nothing
extra" is exactly what running it as a real instruction gives you for free.
"""

from .bits import bits, to_signed
from .encoder import encode
from .exceptions import IllegalInstruction
from .instructions import RV32I
from .isa import register_isa
from .rv32a import RV32IA, RV32IA_ZICSR, RV32IMA, RV32IMA_ZICSR
from .rv32m import RV32IM
from .zicsr import RV32I_ZICSR, RV32IM_ZICSR

#: Every base/extension combination this module adds compressed encodings to.
#: ``RV32IC`` is a copy purely so ``len()`` and ``"RV32C" in {...extensions}``
#: stay meaningful per set; :func:`expand` itself needs none of this, since it
#: only ever emits plain RV32I words.
RV32IC = RV32I.copy("RV32IC")
RV32IMC = RV32IM.copy("RV32IMC")
RV32IAC = RV32IA.copy("RV32IAC")
RV32IMAC = RV32IMA.copy("RV32IMAC")
RV32IC_ZICSR = RV32I_ZICSR.copy("RV32IC_Zicsr")
RV32IMC_ZICSR = RV32IM_ZICSR.copy("RV32IMC_Zicsr")
RV32IAC_ZICSR = RV32IA_ZICSR.copy("RV32IAC_Zicsr")
RV32IMAC_ZICSR = RV32IMA_ZICSR.copy("RV32IMAC_Zicsr")

for _target in (
    RV32IC, RV32IMC, RV32IAC, RV32IMAC,
    RV32IC_ZICSR, RV32IMC_ZICSR, RV32IAC_ZICSR, RV32IMAC_ZICSR,
):
    _target.compressed = True

register_isa("rv32ic", RV32IC)
register_isa("rv32imc", RV32IMC)
register_isa("rv32iac", RV32IAC)
register_isa("rv32imac", RV32IMAC)
register_isa("rv32ic_zicsr", RV32IC_ZICSR)
register_isa("rv32imc_zicsr", RV32IMC_ZICSR)
register_isa("rv32iac_zicsr", RV32IAC_ZICSR)
register_isa("rv32imac_zicsr", RV32IMAC_ZICSR)


def _bit(word: int, n: int) -> int:
    return (word >> n) & 1


def _creg(field: int) -> int:
    """x8-x15 only: the three compressed register fields' usual range."""
    return 8 + field


def _scramble(half: int, pairs) -> int:
    """Reassemble an immediate from ``(destination_bit, source_bit)`` pairs."""
    value = 0
    for dest, src in pairs:
        value |= _bit(half, src) << dest
    return value


#: imm[11|4|9:8|10|6|7|3:1|5] -- shared by C.J and C.JAL.
_CJ_BITS = ((11, 12), (4, 11), (9, 10), (8, 9), (10, 8), (6, 7), (7, 6), (3, 5), (2, 4), (1, 3), (5, 2))
#: imm[8|4:3|7:6|2:1|5] -- shared by C.BEQZ and C.BNEZ.
_CB_BITS = ((8, 12), (4, 11), (3, 10), (7, 6), (6, 5), (2, 4), (1, 3), (5, 2))
#: uimm[5:4|9:6|2|3] -- C.ADDI4SPN.
_CIW_BITS = ((5, 12), (4, 11), (9, 10), (8, 9), (7, 8), (6, 7), (2, 6), (3, 5))
#: uimm[5:3|2|6] -- C.LW / C.SW.
_CLS_BITS = ((5, 12), (4, 11), (3, 10), (2, 6), (6, 5))
#: imm[9|4|6|8:7|5] -- C.ADDI16SP.
_ADDI16SP_BITS = ((9, 12), (4, 6), (6, 5), (8, 4), (7, 3), (5, 2))
#: uimm[5|4:2|7:6] -- C.LWSP.
_LWSP_BITS = ((5, 12), (4, 6), (3, 5), (2, 4), (7, 3), (6, 2))
#: uimm[5:2|7:6] -- C.SWSP.
_SWSP_BITS = ((5, 12), (4, 11), (3, 10), (2, 9), (7, 8), (6, 7))


def _illegal(half, pc):
    raise IllegalInstruction(tval=half, pc=pc)


def expand(half: int, pc: int | None = None):
    """Decode one 16-bit compressed word into ``(word32, name)``.

    ``word32`` is an ordinary RV32I-encoded instruction word, ready for
    :meth:`riscv.decoder.Decoder.decode`; ``name`` is the canonical ``C.*``
    mnemonic for display. Raises :class:`~riscv.exceptions.IllegalInstruction`
    for encodings the specification reserves (quadrant 3, i.e. ``half & 0x3
    == 0b11``, is a 4-byte instruction and is never passed in here).
    """
    half &= 0xFFFF
    op = half & 0x3
    funct3 = bits(half, 15, 13)

    if op == 0b00:
        rs1_ = _creg(bits(half, 9, 7))
        rdrs2_ = _creg(bits(half, 4, 2))
        if funct3 == 0b000:  # C.ADDI4SPN
            imm = _scramble(half, _CIW_BITS)
            if imm == 0:
                _illegal(half, pc)  # all-zero is reserved, not a HINT
            return encode("ADDI", rd=rdrs2_, rs1=2, imm=imm, instruction_set=RV32I), "C.ADDI4SPN"
        if funct3 == 0b010:  # C.LW
            imm = _scramble(half, _CLS_BITS)
            return encode("LW", rd=rdrs2_, rs1=rs1_, imm=imm, instruction_set=RV32I), "C.LW"
        if funct3 == 0b110:  # C.SW
            imm = _scramble(half, _CLS_BITS)
            return encode("SW", rs1=rs1_, rs2=rdrs2_, imm=imm, instruction_set=RV32I), "C.SW"
        _illegal(half, pc)  # 001/011/101/111: F/D-only loads/stores, not implemented; 100: reserved

    if op == 0b01:
        rd = bits(half, 11, 7)
        if funct3 == 0b000:  # C.ADDI (rd==0, imm==0 is the canonical C.NOP)
            imm = to_signed((_bit(half, 12) << 5) | bits(half, 6, 2), 6)
            return encode("ADDI", rd=rd, rs1=rd, imm=imm, instruction_set=RV32I), (
                "C.NOP" if rd == 0 and imm == 0 else "C.ADDI"
            )
        if funct3 == 0b001:  # C.JAL -- RV32-only slot (C.ADDIW on RV64)
            imm = to_signed(_scramble(half, _CJ_BITS), 12)
            return encode("JAL", rd=1, imm=imm, instruction_set=RV32I), "C.JAL"
        if funct3 == 0b010:  # C.LI
            imm = to_signed((_bit(half, 12) << 5) | bits(half, 6, 2), 6)
            return encode("ADDI", rd=rd, rs1=0, imm=imm, instruction_set=RV32I), "C.LI"
        if funct3 == 0b011:
            if rd == 2:  # C.ADDI16SP
                imm = to_signed(_scramble(half, _ADDI16SP_BITS), 10)
                if imm == 0:
                    _illegal(half, pc)
                return encode("ADDI", rd=2, rs1=2, imm=imm, instruction_set=RV32I), "C.ADDI16SP"
            nzimm = to_signed((_bit(half, 12) << 5) | bits(half, 6, 2), 6)  # C.LUI
            if nzimm == 0 or rd == 0:
                _illegal(half, pc)
            return encode("LUI", rd=rd, imm=nzimm, instruction_set=RV32I), "C.LUI"
        if funct3 == 0b100:
            rd_ = _creg(bits(half, 9, 7))
            group = bits(half, 11, 10)
            if group == 0b00:  # C.SRLI
                if _bit(half, 12):
                    _illegal(half, pc)  # shamt[5]=1 needs XLEN=64
                return encode(
                    "SRLI", rd=rd_, rs1=rd_, imm=bits(half, 6, 2), instruction_set=RV32I
                ), "C.SRLI"
            if group == 0b01:  # C.SRAI
                if _bit(half, 12):
                    _illegal(half, pc)
                return encode(
                    "SRAI", rd=rd_, rs1=rd_, imm=bits(half, 6, 2), instruction_set=RV32I
                ), "C.SRAI"
            if group == 0b10:  # C.ANDI
                imm = to_signed((_bit(half, 12) << 5) | bits(half, 6, 2), 6)
                return encode("ANDI", rd=rd_, rs1=rd_, imm=imm, instruction_set=RV32I), "C.ANDI"
            # group == 0b11: C.SUB/C.XOR/C.OR/C.AND, or the RV64-only C.SUBW/C.ADDW slot
            if _bit(half, 12):
                _illegal(half, pc)
            rs2_ = _creg(bits(half, 4, 2))
            name, mnemonic = {
                0b00: ("SUB", "C.SUB"), 0b01: ("XOR", "C.XOR"),
                0b10: ("OR", "C.OR"), 0b11: ("AND", "C.AND"),
            }[bits(half, 6, 5)]
            return encode(name, rd=rd_, rs1=rd_, rs2=rs2_, instruction_set=RV32I), mnemonic
        if funct3 == 0b101:  # C.J
            imm = to_signed(_scramble(half, _CJ_BITS), 12)
            return encode("JAL", rd=0, imm=imm, instruction_set=RV32I), "C.J"
        if funct3 in (0b110, 0b111):  # C.BEQZ / C.BNEZ
            rs1_ = _creg(bits(half, 9, 7))
            imm = to_signed(_scramble(half, _CB_BITS), 9)
            name = "BEQ" if funct3 == 0b110 else "BNE"
            return encode(name, rs1=rs1_, rs2=0, imm=imm, instruction_set=RV32I), (
                "C.BEQZ" if funct3 == 0b110 else "C.BNEZ"
            )

    if op == 0b10:
        rd = bits(half, 11, 7)
        if funct3 == 0b000:  # C.SLLI
            if _bit(half, 12):
                _illegal(half, pc)
            return encode(
                "SLLI", rd=rd, rs1=rd, imm=bits(half, 6, 2), instruction_set=RV32I
            ), "C.SLLI"
        if funct3 == 0b010:  # C.LWSP
            if rd == 0:
                _illegal(half, pc)  # rd=x0 is reserved (unlike most other HINTs)
            imm = _scramble(half, _LWSP_BITS)
            return encode("LW", rd=rd, rs1=2, imm=imm, instruction_set=RV32I), "C.LWSP"
        if funct3 == 0b100:
            funct4 = bits(half, 15, 12)
            rs2 = bits(half, 6, 2)
            if funct4 == 0b1000:
                if rs2 == 0:
                    if rd == 0:
                        _illegal(half, pc)  # "C.JR x0" is not a thing
                    return encode("JALR", rd=0, rs1=rd, imm=0, instruction_set=RV32I), "C.JR"
                return encode("ADD", rd=rd, rs1=0, rs2=rs2, instruction_set=RV32I), "C.MV"
            if rs2 == 0:
                if rd == 0:
                    return encode("EBREAK", instruction_set=RV32I), "C.EBREAK"
                return encode("JALR", rd=1, rs1=rd, imm=0, instruction_set=RV32I), "C.JALR"
            return encode("ADD", rd=rd, rs1=rd, rs2=rs2, instruction_set=RV32I), "C.ADD"
        if funct3 == 0b110:  # C.SWSP
            rs2 = bits(half, 6, 2)
            imm = _scramble(half, _SWSP_BITS)
            return encode("SW", rs1=2, rs2=rs2, imm=imm, instruction_set=RV32I), "C.SWSP"
        _illegal(half, pc)  # 001/011/101/111: F/D-only stack loads/stores

    _illegal(half, pc)
