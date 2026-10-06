"""Trap and error types.

Two distinct families live here:

* :class:`RiscvException` and its subclasses model *architectural* traps.  Each
  one carries the standard RISC-V exception ``cause`` code and ``tval`` value
  from the privileged spec, so that when CSRs/privilege modes are added later a
  trap handler can be dropped in without changing any instruction semantics.
  Until then, a trap simply propagates out of :meth:`riscv.cpu.CPU.step`.

* :class:`EmulatorError` and its subclasses model *host* errors -- misuse of the
  emulator API itself (bad register index, malformed program image, ...).  These
  are bugs in the embedding program, not in the guest program.
"""


# --------------------------------------------------------------------------
# Architectural traps
# --------------------------------------------------------------------------
class RiscvException(Exception):
    """Base class for architectural traps taken by the guest program.

    Attributes
    ----------
    cause:
        The value that would be written to ``mcause`` (interrupt bit clear).
    tval:
        The value that would be written to ``mtval``.
    pc:
        Address of the instruction that raised the trap (``mepc``).
    """

    cause = -1
    mnemonic = "unknown"

    def __init__(self, tval: int = 0, pc: int | None = None, message: str | None = None):
        self.tval = tval & 0xFFFFFFFF
        self.pc = None if pc is None else pc & 0xFFFFFFFF
        pc_text = "unknown" if self.pc is None else f"0x{self.pc:08X}"
        detail = message or f"tval=0x{self.tval:08X}"
        super().__init__(f"{self.mnemonic} (cause={self.cause}) at pc={pc_text}: {detail}")


class InstructionAddressMisaligned(RiscvException):
    cause = 0
    mnemonic = "instruction address misaligned"


class InstructionAccessFault(RiscvException):
    cause = 1
    mnemonic = "instruction access fault"


class IllegalInstruction(RiscvException):
    cause = 2
    mnemonic = "illegal instruction"


class Breakpoint(RiscvException):
    cause = 3
    mnemonic = "breakpoint"


class LoadAddressMisaligned(RiscvException):
    cause = 4
    mnemonic = "load address misaligned"


class LoadAccessFault(RiscvException):
    cause = 5
    mnemonic = "load access fault"


class StoreAddressMisaligned(RiscvException):
    cause = 6
    mnemonic = "store/AMO address misaligned"


class StoreAccessFault(RiscvException):
    cause = 7
    mnemonic = "store/AMO access fault"


class EnvironmentCall(RiscvException):
    """ECALL from machine mode (cause 11).

    Only machine mode exists today, so the U-mode (8) and S-mode (9) causes are
    not used.  A host may intercept ECALL before it becomes a trap by installing
    a ``ecall_handler`` on the CPU.
    """

    cause = 11
    mnemonic = "environment call from M-mode"


#: Maps a standard cause number to its exception class.  Useful for tests and
#: for a future trap handler that needs to re-materialise a trap.
CAUSE_TO_EXCEPTION = {
    cls.cause: cls
    for cls in (
        InstructionAddressMisaligned,
        InstructionAccessFault,
        IllegalInstruction,
        Breakpoint,
        LoadAddressMisaligned,
        LoadAccessFault,
        StoreAddressMisaligned,
        StoreAccessFault,
        EnvironmentCall,
    )
}


# --------------------------------------------------------------------------
# Host-side errors
# --------------------------------------------------------------------------
class EmulatorError(Exception):
    """Base class for misuse of the emulator API (a host bug, not a guest trap)."""


class MemoryError_(EmulatorError):
    """Raised for out-of-range physical accesses made through the host API.

    Guest accesses are translated into access-fault traps instead; this is only
    raised when host code pokes memory directly with a bad address.
    """


class AssemblerError(EmulatorError):
    """Raised when the built-in assembler cannot assemble a source file."""

    def __init__(self, message: str, line_no: int | None = None, line: str | None = None):
        self.line_no = line_no
        self.line = line
        if line_no is not None:
            message = f"line {line_no}: {message}"
            if line is not None:
                message = f"{message}\n    {line.strip()}"
        super().__init__(message)


class HaltRequest(EmulatorError):
    """Internal signal used by host ECALL handlers to stop :meth:`CPU.run`.

    Never raised out of the public API -- :meth:`riscv.cpu.CPU.step` catches it
    and sets :attr:`riscv.cpu.CPU.halted` instead.
    """

    def __init__(self, code: int = 0):
        self.code = code
        super().__init__(f"halt requested with code {code}")
