"""Control and Status Registers, and the machine-mode trap state they hold.

All CSR knowledge lives here: the register numbers, what each one stores, which
bits are writable, and the two state transitions that matter -- taking a trap
and returning from one.  The CPU calls :meth:`CSRFile.enter_trap` and
:meth:`CSRFile.machine_return` and otherwise knows nothing about CSR layout, so
adding a register (or a whole privilege mode later) is a change in one file.

Design
------
A CSR is a :class:`CSRDefinition`: a number, a name, a reset value, a write mask
and an optional ``sanitize`` hook.  Reads and writes go through the definition,
which is what implements the specification's field rules:

* **WPRI** (reserved, writes preserve/ignore) -- bits outside ``write_mask`` keep
  their current value, so writing ``0xFFFFFFFF`` to ``mstatus`` sets only the
  fields this machine implements.
* **WARL** (write any, read legal) -- ``sanitize`` maps an illegal field value
  onto a legal one; ``mtvec``'s reserved MODE values and ``mstatus.MPP`` use it.
* **Read-only** -- CSR address bits ``[11:10] == 0b11`` marks a read-only
  register per the specification, and writing one raises an illegal-instruction
  exception.  ``misa`` is additionally read-only here by choice (it is WARL, and
  an implementation may ignore writes to it).

Access to an unimplemented CSR raises :class:`~riscv.exceptions.IllegalInstruction`,
which is what the specification requires -- not a silent zero.
"""

from .bits import MASK32
from .exceptions import IllegalInstruction

# ---------------------------------------------------------------------------
# CSR numbers (machine level)
# ---------------------------------------------------------------------------
MSTATUS = 0x300
MISA = 0x301
MIE = 0x304
MTVEC = 0x305
MSCRATCH = 0x340
MEPC = 0x341
MCAUSE = 0x342
MTVAL = 0x343
MIP = 0x344
MVENDORID = 0xF11
MARCHID = 0xF12
MIMPID = 0xF13
MHARTID = 0xF14

# ---------------------------------------------------------------------------
# mstatus fields (only those this machine implements)
# ---------------------------------------------------------------------------
MSTATUS_MIE = 1 << 3        # machine interrupt enable
MSTATUS_MPIE = 1 << 7       # previous MIE, saved on trap entry
MSTATUS_MPP_SHIFT = 11
MSTATUS_MPP = 0b11 << MSTATUS_MPP_SHIFT  # previous privilege mode
#: Everything the machine implements; every other bit is WPRI and reads zero.
MSTATUS_WRITABLE = MSTATUS_MIE | MSTATUS_MPIE | MSTATUS_MPP

# Interrupt-enable / pending bits (declared so mie/mip have the right shape;
# no interrupt sources exist yet, so nothing ever sets them).
MIP_MSIP = 1 << 3
MIP_MTIP = 1 << 7
MIP_MEIP = 1 << 11
MI_MASK = MIP_MSIP | MIP_MTIP | MIP_MEIP

#: Interrupt cause numbers (mcause with bit 31 set).  Machine-level only.
IRQ_MACHINE_SOFTWARE = 3
IRQ_MACHINE_TIMER = 7
IRQ_MACHINE_EXTERNAL = 11

#: Delivery priority when several are pending at once, highest first
#: (privileged specification: MEI, MSI, MTI).
INTERRUPT_PRIORITY = (IRQ_MACHINE_EXTERNAL, IRQ_MACHINE_SOFTWARE, IRQ_MACHINE_TIMER)


# ---------------------------------------------------------------------------
# mtvec modes
# ---------------------------------------------------------------------------
MTVEC_MODE_DIRECT = 0
MTVEC_MODE_VECTORED = 1
MTVEC_MODE_MASK = 0b11
MTVEC_BASE_MASK = MASK32 & ~0b11

# ---------------------------------------------------------------------------
# Privilege levels.  Only machine mode is implemented; the constants exist so
# that MPP and the CPU's privilege field mean something today and can grow.
# ---------------------------------------------------------------------------
PRIV_USER = 0
PRIV_SUPERVISOR = 1
PRIV_MACHINE = 3

#: misa: MXL=1 (XLEN 32) in bits 31:30, then a bitmap of extension letters.
MISA_MXL_32 = 1 << 30


def misa_value(instruction_set=None) -> int:
    """Build a ``misa`` value describing an instruction set."""
    value = MISA_MXL_32 | (1 << (ord("i") - ord("a")))
    if instruction_set is not None:
        extensions = {definition.extension for definition in instruction_set}
        if "RV32M" in extensions:
            value |= 1 << (ord("m") - ord("a"))
        if "RV32A" in extensions:
            value |= 1 << (ord("a") - ord("a"))
        if getattr(instruction_set, "compressed", False):
            value |= 1 << (ord("c") - ord("a"))
    return value


class CSRDefinition:
    """One control and status register."""

    __slots__ = (
        "number", "name", "reset", "write_mask", "sanitize", "read_hook", "description",
    )

    def __init__(self, number, name, reset=0, write_mask=MASK32, sanitize=None,
                 read_hook=None, description=""):
        self.number = number
        self.name = name
        self.reset = reset & MASK32
        self.write_mask = write_mask & MASK32
        self.sanitize = sanitize
        #: Optional ``callable(stored_value) -> value`` for a register whose
        #: value is driven by hardware rather than by writes -- ``mip`` reads
        #: the live interrupt wires through this.
        self.read_hook = read_hook
        self.description = description

    @property
    def read_only(self) -> bool:
        """True when the CSR's address marks it read-only (bits 11:10 == 0b11)."""
        return ((self.number >> 10) & 0b11) == 0b11

    @property
    def required_privilege(self) -> int:
        """Lowest privilege level allowed to access this CSR (bits 9:8)."""
        return (self.number >> 8) & 0b11

    def __repr__(self) -> str:
        return f"<CSRDefinition {self.name} 0x{self.number:03X}>"


def _sanitize_mtvec(value: int, previous: int) -> int:
    """``mtvec`` MODE is WARL: keep Direct and Vectored, reject the rest.

    Modes 2 and 3 are reserved.  Rather than silently storing a mode the machine
    cannot implement, a write naming one keeps the BASE and forces Direct -- the
    behaviour is documented rather than arbitrary.
    """
    mode = value & MTVEC_MODE_MASK
    if mode not in (MTVEC_MODE_DIRECT, MTVEC_MODE_VECTORED):
        mode = MTVEC_MODE_DIRECT
    return (value & MTVEC_BASE_MASK) | mode


def _sanitize_mstatus(value: int, previous: int) -> int:
    """``mstatus.MPP`` is WARL, and machine mode is the only legal value here."""
    return (value & ~MSTATUS_MPP) | (PRIV_MACHINE << MSTATUS_MPP_SHIFT)


class CSRFile:
    """The machine's control and status registers, plus its trap state.

    ``privilege`` is tracked alongside, because trap entry and ``MRET`` change
    it; with only machine mode implemented it never leaves
    :data:`PRIV_MACHINE`, but the transitions are real rather than assumed.
    """

    def __init__(self, instruction_set=None):
        self.definitions: dict[int, CSRDefinition] = {}
        self.values: dict[int, int] = {}
        self.privilege = PRIV_MACHINE
        #: Set once ``mtvec`` has been written, so the CPU can tell a machine
        #: with a trap handler installed from one that has never configured one.
        self.trap_vector_installed = False
        #: Hardware interrupt wires: each is ``callable() -> pending bit mask``.
        #: The machine timer registers one; an external controller would too.
        self.interrupt_sources: list = []
        self._install_machine_csrs(instruction_set)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def define(self, definition: CSRDefinition) -> CSRDefinition:
        """Register a CSR.  This is the extension point for new registers."""
        self.definitions[definition.number] = definition
        self.values[definition.number] = definition.reset
        return definition

    def _install_machine_csrs(self, instruction_set) -> None:
        self.define(CSRDefinition(
            MSTATUS, "mstatus",
            reset=PRIV_MACHINE << MSTATUS_MPP_SHIFT,
            write_mask=MSTATUS_WRITABLE, sanitize=_sanitize_mstatus,
            description="machine status: MIE, MPIE, MPP",
        ))
        self.define(CSRDefinition(
            MISA, "misa", reset=misa_value(instruction_set), write_mask=0,
            description="ISA and extensions (read-only here)",
        ))
        self.define(CSRDefinition(
            MIE, "mie", write_mask=MI_MASK,
            description="machine interrupt enable",
        ))
        self.define(CSRDefinition(
            MTVEC, "mtvec", sanitize=_sanitize_mtvec,
            description="machine trap vector base and mode",
        ))
        self.define(CSRDefinition(
            MSCRATCH, "mscratch", description="scratch register for trap handlers",
        ))
        self.define(CSRDefinition(
            MEPC, "mepc", write_mask=MASK32 & ~0b11,
            description="machine exception program counter (IALIGN=32, so bits 1:0 are zero)",
        ))
        self.define(CSRDefinition(MCAUSE, "mcause", description="machine trap cause"))
        self.define(CSRDefinition(MTVAL, "mtval", description="machine trap value"))
        self.define(CSRDefinition(
            MIP, "mip", write_mask=0, read_hook=self._read_mip,
            description="machine interrupt pending (driven by hardware, not by writes)",
        ))
        for number, name, value in (
            (MVENDORID, "mvendorid", 0),   # 0 = not a commercial implementation
            (MARCHID, "marchid", 0),
            (MIMPID, "mimpid", 0),
            (MHARTID, "mhartid", 0),       # single hart, always 0
        ):
            self.define(CSRDefinition(number, name, reset=value, write_mask=0,
                                      description="read-only machine identification"))

    def add_interrupt_source(self, source) -> None:
        """Register a hardware interrupt wire.

        ``source`` is called on every ``mip`` read and on every interrupt
        evaluation, and returns the ``mip`` bits it is currently asserting.
        Reading the wire live is what makes MTIP track ``mtime >= mtimecmp``
        without anyone having to remember to update a copy.
        """
        self.interrupt_sources.append(source)

    def _read_mip(self, stored: int) -> int:
        """``mip`` is the OR of its stored bits and every hardware wire."""
        for source in self.interrupt_sources:
            stored |= source()
        return stored & MI_MASK

    def pending_interrupts(self) -> int:
        """Live ``mip``: which machine interrupts are asserted right now."""
        return self._read_mip(self.values[MIP])

    def deliverable_interrupts(self) -> int:
        """Interrupts that are pending *and* enabled *and* globally permitted.

        Returns zero unless ``mstatus.MIE`` allows machine interrupts, which is
        what keeps a handler from being re-entered by the same wire.
        """
        if not self.mie_enabled:
            return 0
        return self.pending_interrupts() & self.values[MIE] & MI_MASK

    def reset(self) -> None:
        """Return every CSR to its reset value and re-enter machine mode.

        Interrupt sources are *not* dropped: they are wires from hardware that
        outlives a CPU reset, and the devices behind them reset themselves.
        """
        for number, definition in self.definitions.items():
            self.values[number] = definition.reset
        self.privilege = PRIV_MACHINE
        self.trap_vector_installed = False

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def exists(self, number: int) -> bool:
        return number in self.definitions

    def definition(self, number: int) -> CSRDefinition:
        return self.definitions[number]

    def by_name(self, name: str) -> CSRDefinition:
        for definition in self.definitions.values():
            if definition.name == name:
                return definition
        raise KeyError(f"unknown CSR {name!r}")

    def _check(self, number: int, pc=None, word=0) -> CSRDefinition:
        definition = self.definitions.get(number)
        if definition is None:
            raise IllegalInstruction(
                tval=word, pc=pc,
                message=f"CSR 0x{number:03X} is not implemented",
            )
        if definition.required_privilege > self.privilege:
            raise IllegalInstruction(
                tval=word, pc=pc,
                message=f"CSR {definition.name} needs a higher privilege level",
            )
        return definition

    def read(self, number: int, pc=None, word=0) -> int:
        """Read a CSR; raises IllegalInstruction if it is not implemented."""
        definition = self._check(number, pc, word)
        value = self.values[number]
        if definition.read_hook is not None:
            value = definition.read_hook(value)
        return value & MASK32

    def write(self, number: int, value: int, pc=None, word=0) -> None:
        """Write a CSR, honouring its read-only status, WPRI mask and WARL rules."""
        definition = self._check(number, pc, word)
        if definition.read_only:
            raise IllegalInstruction(
                tval=word, pc=pc,
                message=f"CSR {definition.name} is read-only",
            )
        previous = self.values[number]
        merged = (previous & ~definition.write_mask) | (value & definition.write_mask)
        if definition.sanitize is not None:
            merged = definition.sanitize(merged, previous)
        self.values[number] = merged & MASK32
        if number == MTVEC:
            self.trap_vector_installed = True

    # -- host-side access (never traps; for tests, the CLI and debuggers) --
    def get(self, number_or_name) -> int:
        """Read a CSR from the host side, by number or name."""
        number = self._resolve(number_or_name)
        value = self.values.get(number, 0)
        definition = self.definitions.get(number)
        if definition is not None and definition.read_hook is not None:
            value = definition.read_hook(value)
        return value & MASK32

    def set(self, number_or_name, value: int) -> None:
        """Write a CSR from the host side, bypassing read-only and WARL rules."""
        number = self._resolve(number_or_name)
        if number not in self.definitions:
            raise KeyError(f"unknown CSR 0x{number:03X}")
        self.values[number] = value & MASK32
        if number == MTVEC:
            self.trap_vector_installed = True

    def _resolve(self, number_or_name) -> int:
        if isinstance(number_or_name, int):
            return number_or_name
        return self.by_name(str(number_or_name)).number

    # ------------------------------------------------------------------
    # mstatus fields
    # ------------------------------------------------------------------
    @property
    def mie_enabled(self) -> bool:
        """``mstatus.MIE`` -- are machine interrupts enabled?"""
        return bool(self.values[MSTATUS] & MSTATUS_MIE)

    @property
    def mpie(self) -> bool:
        return bool(self.values[MSTATUS] & MSTATUS_MPIE)

    @property
    def mpp(self) -> int:
        return (self.values[MSTATUS] & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT

    # ------------------------------------------------------------------
    # mtvec
    # ------------------------------------------------------------------
    @property
    def mtvec_base(self) -> int:
        return self.values[MTVEC] & MTVEC_BASE_MASK

    @property
    def mtvec_mode(self) -> int:
        return self.values[MTVEC] & MTVEC_MODE_MASK

    def trap_target(self, cause: int, interrupt: bool) -> int:
        """Where a trap vectors to, per ``mtvec``'s BASE and MODE.

        Direct mode sends everything to BASE.  Vectored mode sends *interrupts*
        to ``BASE + 4 * cause`` and synchronous exceptions still to BASE --
        which is what the specification says, and a detail that is easy to get
        backwards.
        """
        base = self.mtvec_base
        if self.mtvec_mode == MTVEC_MODE_VECTORED and interrupt:
            return (base + 4 * cause) & MASK32
        return base

    # ------------------------------------------------------------------
    # Trap entry and return -- the two transitions that matter
    # ------------------------------------------------------------------
    def enter_trap(self, pc: int, cause: int, tval: int = 0, interrupt: bool = False) -> int:
        """Take a trap into machine mode; returns the PC to continue at.

        Per the privileged specification:

        * ``mepc``   <- the address of the interrupted instruction
        * ``mcause`` <- the cause, with bit 31 set for an interrupt
        * ``mtval``  <- the fault-specific value
        * ``mstatus.MPIE`` <- ``mstatus.MIE``; ``mstatus.MIE`` <- 0
        * ``mstatus.MPP``  <- the privilege level that was interrupted
        * privilege  <- machine
        """
        self.values[MEPC] = pc & MASK32 & ~0b11
        self.values[MCAUSE] = ((1 << 31) | cause) & MASK32 if interrupt else cause & MASK32
        self.values[MTVAL] = tval & MASK32

        status = self.values[MSTATUS]
        status = (status & ~MSTATUS_MPIE) | (MSTATUS_MPIE if status & MSTATUS_MIE else 0)
        status &= ~MSTATUS_MIE                       # interrupts off inside the handler
        status = (status & ~MSTATUS_MPP) | (self.privilege << MSTATUS_MPP_SHIFT)
        self.values[MSTATUS] = status & MASK32

        self.privilege = PRIV_MACHINE
        return self.trap_target(cause, interrupt)

    def machine_return(self) -> int:
        """Execute ``MRET``'s state change; returns the PC to continue at.

        * privilege <- ``mstatus.MPP``
        * ``mstatus.MIE``  <- ``mstatus.MPIE``
        * ``mstatus.MPIE`` <- 1
        * ``mstatus.MPP``  <- the least-privileged supported mode (machine here)
        """
        status = self.values[MSTATUS]
        target_privilege = (status & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT

        status = (status & ~MSTATUS_MIE) | (MSTATUS_MIE if status & MSTATUS_MPIE else 0)
        status |= MSTATUS_MPIE                        # MPIE is set to 1
        status = (status & ~MSTATUS_MPP) | (PRIV_MACHINE << MSTATUS_MPP_SHIFT)
        self.values[MSTATUS] = status & MASK32

        self.privilege = target_privilege
        return self.values[MEPC] & MASK32

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return dict(self.values)

    def restore(self, values: dict) -> None:
        for number, value in values.items():
            if number in self.definitions:
                self.values[number] = value & MASK32

    def format(self, only=None) -> str:
        """Human-readable dump, in CSR-number order."""
        lines = []
        for number in sorted(self.definitions):
            definition = self.definitions[number]
            if only and definition.name not in only:
                continue
            lines.append(
                f"  {definition.name:<10} 0x{number:03X} = 0x{self.values[number]:08X}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<CSRFile {len(self.definitions)} registers, privilege={self.privilege}>"


#: Every CSR name the assembler and disassembler should recognise, in number
#: order.  Built from a throwaway file so the list cannot drift from the code.
def csr_names() -> dict:
    """``{name: number}`` for every implemented CSR."""
    reference = CSRFile()
    return {
        definition.name: number
        for number, definition in sorted(reference.definitions.items())
    }


def csr_numbers() -> dict:
    """``{number: name}`` for every implemented CSR."""
    return {number: name for name, number in csr_names().items()}
