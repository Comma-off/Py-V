"""Instruction-set description machinery.

An :class:`InstructionSet` is nothing more than a list of :class:`InstructionDef`
records, each of which says "if ``word & mask == match``, this is instruction
*name*, it has *format*, and *execute* implements it".

This table-driven shape is what makes the emulator extensible.  Adding RV32M is
a matter of building a new set and registering seven more entries -- no changes
to the decoder, the CPU, or any existing instruction::

    from riscv.isa import InstructionSet, R_TYPE
    from riscv.instructions import RV32I

    RV32IM = RV32I.copy(name="RV32IM")

    @RV32IM.define("MUL", match=0x02000033, mask=0xFE00707F, fmt=R_TYPE)
    def exec_mul(cpu, inst):
        cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) * cpu.get_reg(inst.rs2))

    cpu = CPU(instruction_set=RV32IM)
"""

# Instruction formats.  Plain strings keep this trivially portable and make
# repr()/debug output readable.
R_TYPE = "R"
I_TYPE = "I"
S_TYPE = "S"
B_TYPE = "B"
U_TYPE = "U"
J_TYPE = "J"
#: Fully-specified encodings with no operand fields at all (ECALL, EBREAK, MRET).
N_TYPE = "N"
#: FENCE: encoded like an I-type but its "immediate" is a set of ordering bits.
FENCE_TYPE = "FENCE"
#: CSR instructions: encoded like an I-type, but the immediate is an *unsigned*
#: 12-bit CSR address rather than a signed value, and ``rs1`` doubles as a
#: 5-bit unsigned immediate in the ``*I`` forms.  Sign-extending a CSR address
#: would turn 0xF11 (mvendorid) into 0xFFFFFF11, so it gets its own format.
CSR_TYPE = "CSR"
#: Atomic memory operations (the A extension): encoded with the same field
#: layout as R-type (opcode, rd, funct3, rs1, rs2 all in the usual places),
#: but funct7 splits into a 5-bit operation plus ``aq``/``rl`` ordering bits,
#: and the assembler/disassembler use the ``rd,rs2,(rs1)`` operand order
#: (``rd,(rs1)`` for LR, which has no rs2).  Kept distinct from R_TYPE so the
#: tools can render it correctly; decoding needs nothing extra since the
#: fields already line up.
AMO_TYPE = "AMO"

FORMATS = (
    R_TYPE, I_TYPE, S_TYPE, B_TYPE, U_TYPE, J_TYPE, N_TYPE, FENCE_TYPE, CSR_TYPE,
    AMO_TYPE,
)

# Opcode values (word[6:0]) used by RV32I.
OPCODE_LOAD = 0x03
OPCODE_MISC_MEM = 0x0F
OPCODE_OP_IMM = 0x13
OPCODE_AUIPC = 0x17
OPCODE_STORE = 0x23
OPCODE_OP = 0x33
OPCODE_LUI = 0x37
OPCODE_BRANCH = 0x63
OPCODE_JALR = 0x67
OPCODE_JAL = 0x6F
OPCODE_SYSTEM = 0x73


#: Named instruction sets, so a CLI flag or an ELF's architecture string can
#: select one by name.  Modules register into this as they are imported:
#: :mod:`riscv.instructions` registers ``rv32i``, :mod:`riscv.rv32m` registers
#: ``rv32im``.  A future A/F/D extension registers itself the same way, with no
#: changes needed here or in the CPU.
_ISA_REGISTRY: dict[str, "InstructionSet"] = {}


def register_isa(name, instruction_set):
    """Register ``instruction_set`` under a lowercase ``name``; returns the set."""
    _ISA_REGISTRY[name.lower()] = instruction_set
    return instruction_set


#: Modules that register instruction sets, imported on demand.  A new extension
#: adds its module here (or registers itself when the user imports it).
_ISA_MODULES = (".instructions", ".rv32m", ".zicsr", ".rv32a", ".rv32c")
_registration_done = False


def _ensure_registered():
    """Import the modules that populate the registry, once."""
    global _registration_done
    if _registration_done:
        return
    _registration_done = True  # set first: the imports below re-enter via register_isa
    import importlib

    for module in _ISA_MODULES:
        importlib.import_module(module, __package__)


def get_isa(name):
    """Look up a registered instruction set by name.

    Importing :mod:`riscv.rv32m` is what makes ``rv32im`` available; this does
    that on demand, so callers need not know which module supplies which name.
    """
    _ensure_registered()
    key = str(name).lower()
    if key not in _ISA_REGISTRY:
        raise KeyError(
            f"unknown instruction set {name!r}; available: {', '.join(available_isas())}"
        )
    return _ISA_REGISTRY[key]


def available_isas():
    """Every registered instruction set name, sorted."""
    _ensure_registered()
    return sorted(_ISA_REGISTRY)


class InstructionDef:
    """A single decodable instruction."""

    __slots__ = ("name", "match", "mask", "fmt", "execute", "extension")

    def __init__(self, name, match, mask, fmt, execute, extension="RV32I"):
        if fmt not in FORMATS:
            raise ValueError(f"unknown instruction format {fmt!r}")
        if match & ~mask & 0xFFFFFFFF:
            raise ValueError(
                f"{name}: match 0x{match:08X} has bits set outside mask 0x{mask:08X}"
            )
        self.name = name
        self.match = match & 0xFFFFFFFF
        self.mask = mask & 0xFFFFFFFF
        self.fmt = fmt
        self.execute = execute
        self.extension = extension

    @property
    def opcode(self) -> int:
        return self.match & 0x7F

    @property
    def opcode_is_fixed(self) -> bool:
        return (self.mask & 0x7F) == 0x7F

    def matches(self, word: int) -> bool:
        return (word & self.mask) == self.match

    def __repr__(self) -> str:
        return f"<InstructionDef {self.name} {self.fmt} match=0x{self.match:08X} mask=0x{self.mask:08X}>"


class InstructionSet:
    """An ordered collection of :class:`InstructionDef` records."""

    def __init__(self, name: str = "custom", defs=()):
        self.name = name
        self._defs: list[InstructionDef] = []
        self._by_name: dict[str, InstructionDef] = {}
        self._dispatch: dict[int, tuple] | None = None
        #: Whether this set includes the C (compressed) extension. Pure
        #: metadata -- :mod:`riscv.rv32c` adds no :class:`InstructionDef`
        #: entries at all (a C instruction is an alternate *encoding* of an
        #: existing one, not a new mnemonic) -- but the CPU needs to know
        #: whether to fetch 16 bits at a time and relax IALIGN to 16.
        self.compressed = False
        for definition in defs:
            self.add(definition)

    # -- construction ---------------------------------------------------
    def add(self, definition: InstructionDef) -> InstructionDef:
        """Register a fully-built :class:`InstructionDef`."""
        if definition.name in self._by_name:
            raise ValueError(f"instruction {definition.name!r} is already defined")
        for existing in self._defs:
            if existing.opcode_is_fixed and definition.opcode_is_fixed:
                if existing.opcode != definition.opcode:
                    continue
            overlap = existing.mask & definition.mask
            if (existing.match & overlap) == (definition.match & overlap):
                raise ValueError(
                    f"encoding of {definition.name} overlaps {existing.name}"
                )
        self._defs.append(definition)
        self._by_name[definition.name] = definition
        self._dispatch = None
        return definition

    def add_from(self, name, match, mask, fmt, execute, extension="RV32I"):
        """Register an instruction from its parts; returns the definition.

        The decorator :meth:`define` is sugar over this.  Extensions that add
        the same instruction to several sets at once use this directly.
        """
        return self.add(InstructionDef(name, match, mask, fmt, execute, extension))

    def define(self, name, match, mask, fmt, extension="RV32I"):
        """Decorator form: registers the decorated function as the executor."""

        def register(func):
            self.add(InstructionDef(name, match, mask, fmt, func, extension))
            return func

        return register

    def copy(self, name: str | None = None) -> "InstructionSet":
        """Return an independent copy, ready to be extended."""
        clone = InstructionSet(name or f"{self.name}-copy")
        for definition in self._defs:
            clone.add(
                InstructionDef(
                    definition.name,
                    definition.match,
                    definition.mask,
                    definition.fmt,
                    definition.execute,
                    definition.extension,
                )
            )
        return clone

    # -- lookup ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._defs)

    def __iter__(self):
        return iter(self._defs)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> list[str]:
        return [definition.name for definition in self._defs]

    def by_name(self, name: str) -> InstructionDef:
        try:
            return self._by_name[name.upper()]
        except KeyError:
            raise KeyError(f"{name!r} is not part of instruction set {self.name}") from None

    def _build_dispatch(self) -> dict[int, tuple]:
        """Group definitions by opcode; most-specific mask first."""
        buckets: dict[int, list[InstructionDef]] = {}
        for definition in self._defs:
            key = definition.opcode if definition.opcode_is_fixed else -1
            buckets.setdefault(key, []).append(definition)
        wildcards = buckets.get(-1, [])
        dispatch = {}
        for key, entries in buckets.items():
            if key == -1:
                continue
            combined = entries + wildcards
            combined.sort(key=lambda d: bin(d.mask).count("1"), reverse=True)
            dispatch[key] = tuple(combined)
        dispatch[-1] = tuple(wildcards)
        self._dispatch = dispatch
        return dispatch

    def lookup(self, word: int) -> InstructionDef | None:
        """Return the definition matching ``word``, or ``None``."""
        dispatch = self._dispatch or self._build_dispatch()
        for definition in dispatch.get(word & 0x7F, dispatch[-1]):
            if (word & definition.mask) == definition.match:
                return definition
        return None

    def __repr__(self) -> str:
        return f"<InstructionSet {self.name}: {len(self._defs)} instructions>"
