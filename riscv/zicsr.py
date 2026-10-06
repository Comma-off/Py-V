"""The Zicsr extension (CSR access) and machine-mode trap return (``MRET``).

Seven instructions, all in the ``SYSTEM`` opcode::

    CSRRW  rd, csr, rs1     swap:  rd = csr; csr = rs1
    CSRRS  rd, csr, rs1     set:   rd = csr; csr |= rs1
    CSRRC  rd, csr, rs1     clear: rd = csr; csr &= ~rs1
    CSRRWI rd, csr, uimm    the same three, with a 5-bit zero-extended
    CSRRSI rd, csr, uimm    immediate in place of rs1
    CSRRCI rd, csr, uimm
    MRET                    return from a machine-mode trap

Like :mod:`riscv.rv32m`, this extends rather than mutates: the base sets stay
exactly as they were, and the CSR instructions remain illegal on a machine that
was not built with them.

Read/write side effects
-----------------------
The specification is precise about when a CSR is *not* touched, because reads
and writes can have side effects:

* ``CSRRW``/``CSRRWI`` with ``rd == x0`` must **not read** the CSR.
* ``CSRRS``/``CSRRC``/``CSRRSI``/``CSRRCI`` with a zero source must **not
  write** the CSR -- and that also means they must not raise the
  illegal-instruction exception that writing a read-only CSR would.

That last point is what makes ``csrr rd, mvendorid`` (encoded as
``csrrs rd, mvendorid, x0``) legal even though ``mvendorid`` is read-only.
"""

from .bits import MASK32
from .csr import PRIV_MACHINE
from .instructions import RV32I
from .isa import CSR_TYPE, N_TYPE, register_isa
from .rv32m import RV32IM

#: RV32I plus Zicsr, and RV32IM plus Zicsr.  Built as copies so the base sets
#: keep their exact instruction counts.
RV32I_ZICSR = RV32I.copy("RV32I_Zicsr")
RV32IM_ZICSR = RV32IM.copy("RV32IM_Zicsr")

#: The sets this module extends; every definition below is added to all of them.
_TARGETS = (RV32I_ZICSR, RV32IM_ZICSR)


def _define(name, match, mask, fmt, extension):
    """Register one instruction into every Zicsr-bearing instruction set."""

    def register(func):
        for target in _TARGETS:
            target.add_from(name, match, mask, fmt, func, extension)
        return func

    return register


def _csr_operation(cpu, inst, source, write):
    """Shared body of all six CSR instructions.

    ``source`` is the value to combine with (a register or a zero-extended
    immediate); ``write`` says whether this encoding writes the CSR at all.
    """
    number = inst.csr
    # A CSRRW/CSRRWI with rd == x0 must not read the CSR: a read can have side
    # effects, and the architecture guarantees it does not happen here.
    if inst.rd != 0 or not _is_swap(inst):
        old = cpu.csr.read(number, pc=cpu.pc, word=inst.raw)
    else:
        old = 0

    if write:
        if _is_swap(inst):
            new = source
        elif _is_clear(inst):
            new = old & ~source
        else:
            new = old | source
        cpu.csr.write(number, new & MASK32, pc=cpu.pc, word=inst.raw)

    if inst.rd != 0:
        cpu.set_reg(inst.rd, old)


def _is_swap(inst) -> bool:
    return inst.name in ("CSRRW", "CSRRWI")


def _is_clear(inst) -> bool:
    return inst.name in ("CSRRC", "CSRRCI")


# ---------------------------------------------------------------------------
# Register forms
# ---------------------------------------------------------------------------
@_define("CSRRW", match=0x00001073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrw(cpu, inst):
    """Atomically swap rs1 into the CSR; the old value goes to rd.

    Always writes -- there is no "rs1 == x0 means no write" rule for CSRRW,
    because writing zero is a meaningful operation.
    """
    _csr_operation(cpu, inst, cpu.get_reg(inst.rs1), write=True)


@_define("CSRRS", match=0x00002073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrs(cpu, inst):
    """Set the bits of rs1 in the CSR.  ``rs1 == x0`` reads without writing."""
    _csr_operation(cpu, inst, cpu.get_reg(inst.rs1), write=inst.rs1 != 0)


@_define("CSRRC", match=0x00003073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrc(cpu, inst):
    """Clear the bits of rs1 in the CSR.  ``rs1 == x0`` reads without writing."""
    _csr_operation(cpu, inst, cpu.get_reg(inst.rs1), write=inst.rs1 != 0)


# ---------------------------------------------------------------------------
# Immediate forms: the rs1 field is a 5-bit zero-extended immediate
# ---------------------------------------------------------------------------
@_define("CSRRWI", match=0x00005073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrwi(cpu, inst):
    """Write a 5-bit immediate into the CSR; the old value goes to rd."""
    _csr_operation(cpu, inst, inst.uimm, write=True)


@_define("CSRRSI", match=0x00006073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrsi(cpu, inst):
    """Set the bits of a 5-bit immediate.  A zero immediate reads without writing."""
    _csr_operation(cpu, inst, inst.uimm, write=inst.uimm != 0)


@_define("CSRRCI", match=0x00007073, mask=0x0000707F, fmt=CSR_TYPE, extension="Zicsr")
def exec_csrrci(cpu, inst):
    """Clear the bits of a 5-bit immediate.  A zero immediate reads without writing."""
    _csr_operation(cpu, inst, inst.uimm, write=inst.uimm != 0)


# ---------------------------------------------------------------------------
# Trap return
# ---------------------------------------------------------------------------
@_define("MRET", match=0x30200073, mask=0xFFFFFFFF, fmt=N_TYPE, extension="Priv")
def exec_mret(cpu, inst):
    """Return from a machine-mode trap.

    This is a real state transition, not ``pc = mepc``: privilege is restored
    from ``mstatus.MPP``, ``mstatus.MIE`` is restored from ``MPIE``, ``MPIE``
    is set, and only then does control return to ``mepc``.  See
    :meth:`riscv.csr.CSRFile.machine_return`.
    """
    if cpu.csr.privilege < PRIV_MACHINE:
        from .exceptions import IllegalInstruction

        raise IllegalInstruction(
            tval=inst.raw, pc=cpu.pc, message="MRET requires machine mode"
        )
    cpu.next_pc = cpu.csr.machine_return()


#: What this module adds, in encoding order.
ZICSR_INSTRUCTIONS = ("CSRRW", "CSRRS", "CSRRC", "CSRRWI", "CSRRSI", "CSRRCI")
PRIVILEGED_INSTRUCTIONS = ("MRET",)
ZICSR_INSTRUCTION_COUNT = len(ZICSR_INSTRUCTIONS) + len(PRIVILEGED_INSTRUCTIONS)

register_isa("rv32i_zicsr", RV32I_ZICSR)
register_isa("rv32im_zicsr", RV32IM_ZICSR)
