"""Disassembly, used by trace mode, the ``dis`` CLI command and error messages.

Output format matches the trace example in the project brief::

    PC=0x00000000  INST=0x00A00093  ADDI x1,x0,10

The mnemonic is left-justified to 4 characters, operands are comma-separated
with no spaces.  Branch and jump offsets print as signed decimals relative to
the instruction (which is exactly what :mod:`riscv.assembler` accepts back), so
disassembly round-trips through the assembler.
"""

from .csr import csr_numbers
from .isa import (
    AMO_TYPE,
    B_TYPE,
    CSR_TYPE,
    FENCE_TYPE,
    I_TYPE,
    J_TYPE,
    R_TYPE,
    S_TYPE,
    U_TYPE,
)
from .registers import ABI_NAMES

#: Instructions whose I-type immediate is a shift amount rather than a value.
SHIFT_IMMEDIATES = frozenset({"SLLI", "SRLI", "SRAI"})
#: I-type instructions printed in ``rd,imm(rs1)`` form.
MEMORY_FORM = frozenset({"LB", "LH", "LW", "LBU", "LHU", "JALR"})

_FENCE_LETTERS = ("i", "o", "r", "w")

#: CSR number -> name, so disassembly reads like objdump's (``mstatus``, not
#: ``0x300``).  The assembler accepts the same names, so this round-trips.
_CSR_NAMES = csr_numbers()

#: CSR instructions whose ``rs1`` field is a 5-bit immediate, not a register.
CSR_IMMEDIATE_FORMS = frozenset({"CSRRWI", "CSRRSI", "CSRRCI"})


def csr_name(number: int) -> str:
    """Symbolic CSR name if we implement it, else its hex address."""
    return _CSR_NAMES.get(number, f"0x{number:03X}")


def register_name(index: int, abi: bool = False) -> str:
    return ABI_NAMES[index] if abi else f"x{index}"


def _fence_bits(value: int) -> str:
    letters = "".join(
        letter for shift, letter in zip((3, 2, 1, 0), _FENCE_LETTERS) if value & (1 << shift)
    )
    return letters or "0"


def operands(inst, abi: bool = False) -> str:
    """Render just the operand list of a decoded instruction."""
    reg = lambda index: register_name(index, abi)  # noqa: E731
    fmt = inst.fmt
    name = inst.name

    if fmt == R_TYPE:
        return f"{reg(inst.rd)},{reg(inst.rs1)},{reg(inst.rs2)}"
    if fmt == I_TYPE:
        if name in SHIFT_IMMEDIATES:
            return f"{reg(inst.rd)},{reg(inst.rs1)},{inst.shamt}"
        if name in MEMORY_FORM:
            return f"{reg(inst.rd)},{inst.imm_signed}({reg(inst.rs1)})"
        return f"{reg(inst.rd)},{reg(inst.rs1)},{inst.imm_signed}"
    if fmt == S_TYPE:
        return f"{reg(inst.rs2)},{inst.imm_signed}({reg(inst.rs1)})"
    if fmt == B_TYPE:
        return f"{reg(inst.rs1)},{reg(inst.rs2)},{inst.imm_signed}"
    if fmt == U_TYPE:
        return f"{reg(inst.rd)},0x{inst.imm >> 12:X}"
    if fmt == J_TYPE:
        return f"{reg(inst.rd)},{inst.imm_signed}"
    if fmt == CSR_TYPE:
        source = inst.uimm if name in CSR_IMMEDIATE_FORMS else reg(inst.rs1)
        return f"{reg(inst.rd)},{csr_name(inst.csr)},{source}"
    if fmt == FENCE_TYPE:
        pred = _fence_bits((inst.imm >> 4) & 0xF)
        succ = _fence_bits(inst.imm & 0xF)
        return f"{pred},{succ}"
    if fmt == AMO_TYPE:
        if name.startswith("LR."):
            return f"{reg(inst.rd)},({reg(inst.rs1)})"
        return f"{reg(inst.rd)},{reg(inst.rs2)},({reg(inst.rs1)})"
    return ""  # N_TYPE: ECALL / EBREAK


def disassemble(word, pc=None, instruction=None, instruction_set=None, abi=False) -> str:
    """Disassemble an instruction word (or an already-decoded instruction).

    Raises :class:`~riscv.exceptions.IllegalInstruction` for words that do not
    decode; callers that want a placeholder should use :func:`safe_disassemble`.
    """
    if instruction is None:
        from .decoder import Decoder

        instruction = Decoder(instruction_set).decode(word, pc)
    text = operands(instruction, abi=abi)
    if not text:
        return instruction.name
    return f"{instruction.name:<4} {text}"


def safe_disassemble(word, pc=None, instruction_set=None, abi=False) -> str:
    """Like :func:`disassemble`, but renders undecodable words as ``.word``."""
    from .exceptions import IllegalInstruction

    try:
        return disassemble(word, pc=pc, instruction_set=instruction_set, abi=abi)
    except IllegalInstruction:
        return f".word 0x{word & 0xFFFFFFFF:08X}"


def disassemble_compressed(name: str, instruction, abi: bool = False) -> str:
    """Render a compressed instruction: the real ``C.*`` mnemonic, with the
    operands of the full-width instruction it was expanded to and already
    decoded as (that is what actually executes, and what populated its
    ``rd``/``rs1``/``rs2``/``imm`` fields)."""
    text = operands(instruction, abi=abi)
    return f"{name:<4} {text}" if text else name


def disassemble_bytes(data, base=0, instruction_set=None, abi=False):
    """Disassemble a byte string, honouring C's mixed 2-/4-byte widths.

    Yields ``(address, size, raw, text)`` per instruction. Undecodable words
    render as ``.word``/``.half`` rather than raising, like
    :func:`safe_disassemble`; a single trailing byte with no instruction left
    to decode is simply not yielded. When ``instruction_set`` has no C, this
    is exactly a 4-byte-at-a-time walk -- the shape every caller used before.
    """
    from .decoder import Decoder
    from .exceptions import IllegalInstruction

    compressed = bool(getattr(instruction_set, "compressed", False))
    decoder = Decoder(instruction_set) if compressed else None
    offset = 0
    end = len(data)
    while offset < end:
        address = base + offset
        if compressed and end - offset >= 2:
            half = int.from_bytes(data[offset:offset + 2], "little")
            if (half & 0x3) != 0x3:
                try:
                    from .rv32c import expand

                    word, name = expand(half, pc=address)
                    text = disassemble_compressed(name, decoder.decode(word, address), abi=abi)
                except IllegalInstruction:
                    text = f".half 0x{half:04X}"
                yield address, 2, half, text
                offset += 2
                continue
            if end - offset < 4:
                yield address, 2, half, f".half 0x{half:04X}"
                offset += 2
                continue
        if end - offset < 4:
            break
        word = int.from_bytes(data[offset:offset + 4], "little")
        yield address, 4, word, safe_disassemble(word, pc=address, instruction_set=instruction_set, abi=abi)
        offset += 4


def trace_line(pc: int, word: int, instruction=None, abi: bool = False) -> str:
    """Format one trace record: ``PC=... INST=... MNEMONIC ops``."""
    if instruction is not None:
        text = disassemble(word, pc=pc, instruction=instruction, abi=abi)
    else:
        text = safe_disassemble(word, pc=pc, abi=abi)
    return f"PC=0x{pc:08X}  INST=0x{word:08X}  {text}"
