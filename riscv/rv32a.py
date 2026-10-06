"""The RV32A standard extension: load-reserved/store-conditional and atomic
memory operations.

Eleven instructions, all in a dedicated ``AMO`` opcode (``0101111``), funct3
``010`` (word width)::

    LR.W      rd, (rs1)            rd = mem[rs1]; reserve rs1
    SC.W      rd, rs2, (rs1)       if reservation on rs1 still holds:
                                        mem[rs1] = rs2; rd = 0
                                   else:
                                        rd = 1
    AMOSWAP.W rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = rs2
    AMOADD.W  rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = rd + rs2
    AMOXOR.W  rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = rd ^ rs2
    AMOAND.W  rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = rd & rs2
    AMOOR.W   rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = rd | rs2
    AMOMIN.W  rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = min(signed(rd), signed(rs2))
    AMOMAX.W  rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = max(signed(rd), signed(rs2))
    AMOMINU.W rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = min(rd, rs2)       (unsigned)
    AMOMAXU.W rd, rs2, (rs1)       rd = mem[rs1]; mem[rs1] = max(rd, rs2)       (unsigned)

Every AMO reads the old word into ``rd``, then writes a new word back -- the
"atomic" part is that nothing else can happen in between.  On a single-hart,
strictly-sequential emulator that is true for free: there is no other hart
that could interleave, so every AMO is implemented as a plain load followed by
a plain store with no special locking.  The ``aq``/``rl`` ordering bits
(``funct7``'s bits 26 and 25) are consequently irrelevant and decoded as
"don't care" -- any value in those two bits still matches the instruction.

**LR/SC's reservation** is the one piece of real state this extension adds.
``LR.W`` records the address it read; ``SC.W`` checks that address, written at
:attr:`~riscv.cpu.CPU.reservation`, and succeeds (writes the store, returns 0
in ``rd``) only if it is still the live reservation -- otherwise it fails
(skips the store, returns 1). The reservation is invalidated by any trap or
interrupt (:meth:`~riscv.cpu.CPU.take_trap`, :meth:`~riscv.cpu.CPU.take_interrupt`)
and by :meth:`~riscv.cpu.CPU.reset`, matching the architectural rule that a
context switch between the ``LR`` and its ``SC`` must make the pair fail. A
real multi-hart implementation would also invalidate it when another hart
writes the reserved line; there is only one hart here, so that case cannot
arise.

Like :mod:`riscv.rv32m`, this extends rather than mutates: the base sets stay
exactly as they were. It builds on top of :mod:`riscv.zicsr` as well as the
bare base sets, so every combination a toolchain might ask for -- A alone, A
with M, either with Zicsr -- is registered and selectable by name.
"""

from .bits import MASK32, to_signed
from .instructions import RV32I
from .isa import AMO_TYPE, register_isa
from .rv32m import RV32IM
from .zicsr import RV32I_ZICSR, RV32IM_ZICSR

#: RV32I/RV32IM plus the A extension, and the same two again with Zicsr.
RV32IA = RV32I.copy("RV32IA")
RV32IMA = RV32IM.copy("RV32IMA")
RV32IA_ZICSR = RV32I_ZICSR.copy("RV32IA_Zicsr")
RV32IMA_ZICSR = RV32IM_ZICSR.copy("RV32IMA_Zicsr")

#: Every instruction set this module adds the extension to.
_TARGETS = (RV32IA, RV32IMA, RV32IA_ZICSR, RV32IMA_ZICSR)

OPCODE_AMO = 0x2F
FUNCT3_W = 0x2

#: funct5 values (word[31:27]), per the specification.
_FUNCT5 = {
    "AMOADD.W": 0b00000,
    "AMOSWAP.W": 0b00001,
    "LR.W": 0b00010,
    "SC.W": 0b00011,
    "AMOXOR.W": 0b00100,
    "AMOOR.W": 0b01000,
    "AMOAND.W": 0b01100,
    "AMOMIN.W": 0b10000,
    "AMOMAX.W": 0b10100,
    "AMOMINU.W": 0b11000,
    "AMOMAXU.W": 0b11100,
}

#: Base word shared by every AMO encoding: opcode and funct3, nothing else.
_BASE = OPCODE_AMO | (FUNCT3_W << 12)
#: funct3 + opcode fixed, funct5 fixed; aq/rl (bits 26:25) are don't-cares.
_MASK = 0xF800707F
#: LR.W additionally pins rs2 to zero -- the field is reserved, not operand.
_LR_MASK = _MASK | 0x01F00000


def _define(name, execute):
    """Register one AMO instruction, with LR.W's extra rs2-must-be-zero mask."""
    match = _BASE | (_FUNCT5[name] << 27)
    mask = _LR_MASK if name == "LR.W" else _MASK
    for target in _TARGETS:
        target.add_from(name, match, mask, AMO_TYPE, execute, extension="RV32A")
    return execute


# ---------------------------------------------------------------------------
# Load-reserved / store-conditional
# ---------------------------------------------------------------------------
def _exec_lr_w(cpu, inst):
    """rd = mem[rs1]; remember rs1 as the live reservation."""
    addr = cpu.get_reg(inst.rs1)
    cpu.set_reg(inst.rd, cpu.load(addr, 4, signed=True))
    cpu.reservation = addr


_define("LR.W", _exec_lr_w)


def _exec_sc_w(cpu, inst):
    """Store rs2 to mem[rs1] iff the LR.W reservation on rs1 is still live.

    ``rd`` gets 0 on success, 1 on failure -- the store never happens on
    failure, which is what lets this implement a lock-free compare-and-swap
    loop without ever corrupting memory on a lost race.
    """
    addr = cpu.get_reg(inst.rs1)
    if cpu.reservation == addr:
        cpu.store(addr, 4, cpu.get_reg(inst.rs2))
        cpu.set_reg(inst.rd, 0)
    else:
        cpu.set_reg(inst.rd, 1)
    cpu.reservation = None  # consumed either way


_define("SC.W", _exec_sc_w)


# ---------------------------------------------------------------------------
# Atomic memory operations
# ---------------------------------------------------------------------------
def _amo(cpu, inst, combine):
    """Shared body: read-modify-write mem[rs1], old value goes to rd."""
    addr = cpu.get_reg(inst.rs1)
    old = cpu.load(addr, 4, signed=False)
    new = combine(old, cpu.get_reg(inst.rs2)) & MASK32
    cpu.store(addr, 4, new)
    cpu.set_reg(inst.rd, old)


_define("AMOSWAP.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: new))
_define("AMOADD.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old + new))
_define("AMOXOR.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old ^ new))
_define("AMOAND.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old & new))
_define("AMOOR.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old | new))
_define(
    "AMOMIN.W",
    lambda cpu, inst: _amo(
        cpu, inst, lambda old, new: old if to_signed(old) <= to_signed(new) else new
    ),
)
_define(
    "AMOMAX.W",
    lambda cpu, inst: _amo(
        cpu, inst, lambda old, new: old if to_signed(old) >= to_signed(new) else new
    ),
)
_define("AMOMINU.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old if old <= new else new))
_define("AMOMAXU.W", lambda cpu, inst: _amo(cpu, inst, lambda old, new: old if old >= new else new))


#: The eleven instructions this module adds, in encoding order.
RV32A_INSTRUCTIONS = (
    "LR.W", "SC.W", "AMOSWAP.W", "AMOADD.W", "AMOXOR.W", "AMOOR.W", "AMOAND.W",
    "AMOMIN.W", "AMOMAX.W", "AMOMINU.W", "AMOMAXU.W",
)
RV32A_INSTRUCTION_COUNT = 11

register_isa("rv32ia", RV32IA)
register_isa("rv32ima", RV32IMA)
register_isa("rv32ia_zicsr", RV32IA_ZICSR)
register_isa("rv32ima_zicsr", RV32IMA_ZICSR)
