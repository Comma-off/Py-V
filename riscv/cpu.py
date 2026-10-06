"""The RV32I hart: fetch, decode, execute.

The CPU owns architectural state (registers + PC) and the policies that
instructions need but should not each re-implement: memory access with the
correct trap causes, alignment rules, tracing, and the host escape hatches for
ECALL/EBREAK.

Execution model
---------------
``step()`` performs exactly one instruction::

    1. check PC alignment            -> instruction-address-misaligned
    2. fetch 4 bytes at PC           -> instruction-access-fault
    3. decode                        -> illegal-instruction
    4. next_pc = pc + 4
    5. execute (may overwrite next_pc, may trap)
    6. pc = next_pc, instret += 1

If a trap escapes step 1-5, ``pc`` is left pointing at the faulting instruction
(the value a real machine would put in ``mepc``) and ``instret`` is *not*
incremented, so a caller can inspect the exact machine state at the fault.  When
trap handling, CSRs and privilege modes are added later, they slot in as a
``_take_trap`` method here without touching any instruction implementation.
"""

import sys

from .bits import MASK32, to_signed
from .csr import INTERRUPT_PRIORITY, MIP_MTIP, CSRFile
from .decoder import Decoder
from .exceptions import (
    Breakpoint,
    EmulatorError,
    EnvironmentCall,
    HaltRequest,
    IllegalInstruction,
    InstructionAccessFault,
    InstructionAddressMisaligned,
    LoadAccessFault,
    LoadAddressMisaligned,
    MemoryError_,
    RiscvException,
    StoreAccessFault,
    StoreAddressMisaligned,
)
from .memory import DEFAULT_SIZE, PERM_R, PERM_RW, PERM_W, PERM_X, Memory, MemoryProtection
from .registers import RegisterFile, register_number
from .timer import MachineTimer

DEFAULT_RESET_PC = 0x00000000


class CPU:
    """A single RV32I hart.

    Parameters
    ----------
    memory:
        A :class:`~riscv.memory.Memory`; one is created if omitted.
    memory_size:
        Size of the auto-created memory.
    reset_pc:
        PC value installed by :meth:`reset`.
    trace / trace_stream / trace_registers:
        Instruction tracing; see :meth:`enable_trace`.
    allow_misaligned:
        When ``False`` (the default, and what real RV32I hardware without the
        misaligned-access extension does) a misaligned load/store raises the
        corresponding address-misaligned trap.  Set ``True`` to emulate a core
        that handles misaligned accesses in hardware.
    ecall_handler / ebreak_handler:
        ``callable(cpu) -> None`` invoked instead of raising the architectural
        trap.  This is the seam a host OS/monitor plugs into.
    instruction_set:
        Override to run an extended ISA (see :mod:`riscv.isa`).
    protection:
        Optional :class:`~riscv.memory.MemoryProtection` enforcing R/W/X on
        every fetch, load and store.  ``None`` disables all checking.
    trap_handling:
        What a synchronous exception does.  ``"architectural"`` takes a real
        RISC-V trap: ``mepc``/``mcause``/``mtval``/``mstatus`` are updated and
        control transfers to ``mtvec``.  ``"host"`` propagates the Python
        exception out of :meth:`step`, which is how the emulator has always
        behaved and what a debugger wants.  ``"auto"`` (the default) picks
        architectural once the guest has written ``mtvec`` -- i.e. once it has
        actually installed a handler -- and host otherwise, so a program with
        no trap handler still surfaces its faults to you instead of vanishing
        into address zero.
    timer:
        A :class:`~riscv.timer.MachineTimer`, or ``True`` for a default one.
        When present it is mapped into memory CLINT-style, wired to ``mip.MTIP``
        and advanced one tick per :meth:`step`.  ``None`` (the default) is a
        machine with no timer: no device is mapped and no time passes, so a CPU
        costs exactly what it always did.
    """

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        memory_size: int = DEFAULT_SIZE,
        reset_pc: int = DEFAULT_RESET_PC,
        trace: bool = False,
        trace_stream=None,
        trace_registers: bool = False,
        allow_misaligned: bool = False,
        ecall_handler=None,
        ebreak_handler=None,
        instruction_set=None,
        protection=None,
        trap_handling="auto",
        timer=None,
    ):
        self.memory = memory if memory is not None else Memory(memory_size)
        self.decoder = Decoder(instruction_set)
        #: Whether the configured ISA includes the C extension. Determines
        #: whether fetch reads one 16-bit halfword at a time (expanding a
        #: compressed one via :mod:`riscv.rv32c`) or always a 4-byte word, and
        #: whether branch/jump targets must only be 2-byte (IALIGN=16) rather
        #: than 4-byte aligned. Fixed for the CPU's lifetime, set once here so
        #: a machine with no C costs exactly what it always did.
        self.compressed = bool(getattr(self.decoder.instruction_set, "compressed", False))
        self.regs = RegisterFile()
        #: Control and status registers, and the machine-mode trap state.
        self.csr = CSRFile(self.decoder.instruction_set)
        self.trap_handling = trap_handling
        self.timer = None
        self.reset_pc = reset_pc & MASK32
        self.allow_misaligned = allow_misaligned
        self.ecall_handler = ecall_handler
        self.ebreak_handler = ebreak_handler
        #: Optional :class:`~riscv.memory.MemoryProtection`.  ``None`` (the
        #: default) means every access is permitted, which is how the CPU has
        #: always behaved; an ELF loaded with ``enforce_permissions=True``
        #: installs one built from the segments' R/W/X flags.
        self.protection = protection

        self.trace = trace
        self.trace_stream = trace_stream if trace_stream is not None else sys.stdout
        self.trace_registers = trace_registers
        #: Optional ``callable(cpu, pc, word, inst)`` invoked instead of writing
        #: a trace line; handy for capturing a trace in tests.
        self.trace_hook = None

        #: Addresses at which :meth:`run` stops before executing.
        self.breakpoints: set[int] = set()

        self.pc = self.reset_pc
        self.next_pc = self.reset_pc
        self.instret = 0
        self.fences = 0
        self.halted = False
        self.halt_code = 0
        self.last_exception: RiscvException | None = None
        self.last_instruction = None
        #: Address of the outstanding LR.W reservation (the A extension), or
        #: ``None``.  A single-hart core needs no reservation *set*, just one
        #: address: any trap, interrupt or reset invalidates it, and SC.W
        #: consumes it whether it succeeds or not.
        self.reservation: int | None = None
        #: Traps taken architecturally (not those propagated to the host).
        self.traps_taken = 0
        #: How many of those were interrupts rather than exceptions.
        self.interrupts_taken = 0

        # Attached last: it maps a device into memory and registers an
        # interrupt source, so every other attribute must already exist.
        if timer is not None and timer is not False:
            self.attach_timer(MachineTimer() if timer is True else timer)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, pc: int | None = None) -> None:
        """Return the hart to its power-on state.

        Registers are zeroed, the PC returns to ``reset_pc`` (or ``pc``), and
        all counters/flags are cleared.  Memory contents are deliberately *not*
        touched -- a reset does not erase RAM, and tests routinely reset the CPU
        while keeping a loaded program in place.  Use ``cpu.memory.clear()`` for
        that.
        """
        if pc is not None:
            self.reset_pc = pc & MASK32
        self.regs.reset()
        self.csr.reset()
        if self.timer is not None:
            self.timer.reset()
        self.traps_taken = 0
        self.interrupts_taken = 0
        self.pc = self.reset_pc
        self.next_pc = self.reset_pc
        self.instret = 0
        self.fences = 0
        self.halted = False
        self.halt_code = 0
        self.last_exception = None
        self.last_instruction = None
        self.reservation = None
        self.decoder.clear_cache()

    def halt(self, code: int = 0) -> None:
        """Stop execution; :meth:`run` returns and :meth:`step` refuses to run."""
        self.halted = True
        self.halt_code = code

    # ------------------------------------------------------------------
    # Register access
    # ------------------------------------------------------------------
    def get_reg(self, index: int) -> int:
        """Read register ``index`` as unsigned 32-bit."""
        return self.regs.read(index)

    def set_reg(self, index: int, value: int) -> None:
        """Write register ``index`` (truncated to 32 bits; x0 ignores writes)."""
        self.regs.write(index, value)

    def get_reg_signed(self, index: int) -> int:
        return to_signed(self.regs.read(index))

    def read_register(self, name) -> int:
        """Read by number or name: ``cpu.read_register("sp")``."""
        return self.regs.read(register_number(name))

    def write_register(self, name, value: int) -> None:
        """Write by number or name: ``cpu.write_register("a0", 42)``."""
        self.regs.write(register_number(name), value)

    # ------------------------------------------------------------------
    # PC access
    # ------------------------------------------------------------------
    def get_pc(self) -> int:
        return self.pc

    def set_pc(self, value: int) -> None:
        """Set the PC.  Alignment is validated at fetch time, not here."""
        self.pc = value & MASK32
        self.next_pc = self.pc

    # ------------------------------------------------------------------
    # Memory access (host-facing convenience wrappers)
    # ------------------------------------------------------------------
    def read_u8(self, addr: int) -> int:
        return self.memory.read_u8(addr)

    def read_u16(self, addr: int) -> int:
        return self.memory.read_u16(addr)

    def read_u32(self, addr: int) -> int:
        return self.memory.read_u32(addr)

    def write_u8(self, addr: int, value: int) -> None:
        self.memory.write_u8(addr, value)

    def write_u16(self, addr: int, value: int) -> None:
        self.memory.write_u16(addr, value)

    def write_u32(self, addr: int, value: int) -> None:
        self.memory.write_u32(addr, value)

    def read_bytes(self, addr: int, length: int) -> bytes:
        return self.memory.read_bytes(addr, length)

    def write_bytes(self, addr: int, data) -> None:
        self.memory.write_bytes(addr, data)

    def load_program(self, data, addr: int = 0, set_pc: bool = True) -> None:
        """Copy a program image into memory and (by default) point the PC at it.

        ``data`` may be ``bytes`` or an iterable of 32-bit instruction words.
        """
        if isinstance(data, (bytes, bytearray, memoryview)):
            self.memory.write_bytes(addr, data)
        else:
            self.memory.load_words(addr, data)
        if set_pc:
            self.reset_pc = addr & MASK32
            self.set_pc(addr)

    def load_image(self, image, *, set_pc: bool = True, enforce_permissions: bool = False,
                   zero_bss: bool = True) -> None:
        """Place a loaded program image into memory.

        ``image`` is anything with ``segments`` (each having ``vaddr``, ``data``,
        ``memsz`` and ``flags``) and an ``entry`` -- in practice a
        :class:`riscv.elf.LoadedImage`.  The CPU deliberately knows nothing
        about ELF: the executable format is the loader's business, and this
        method only consumes the neutral image description::

            image = load_elf("program.elf")
            cpu.load_image(image)

        With ``enforce_permissions`` the segments' R/W/X flags are installed as
        a :class:`~riscv.memory.MemoryProtection`, so writing to ``.text`` or
        executing ``.data`` raises the corresponding access fault.
        """
        loader = getattr(image, "load_into", None)
        if loader is not None:
            loader(self.memory, zero_bss=zero_bss)
        else:  # a bare sequence of segments
            for segment in image.segments:
                if len(segment.data):
                    self.memory.write_bytes(segment.vaddr, segment.data)
                if zero_bss and segment.memsz > len(segment.data):
                    self.memory.zero_fill(
                        segment.vaddr + len(segment.data), segment.memsz - len(segment.data)
                    )
        if enforce_permissions:
            protection = MemoryProtection()
            for segment in image.segments:
                protection.add_region(
                    segment.vaddr, segment.memsz, segment.flags, f"segment{segment.index}"
                )
            if self.timer is not None:
                # Memory-mapped devices are part of the machine, not the image:
                # the ELF says nothing about them, so grant them explicitly.
                from .timer import CLINT_SIZE

                protection.add_region(
                    self.timer.base, CLINT_SIZE, PERM_RW, "clint"
                )
            self.protection = protection
        if set_pc:
            self.reset_pc = image.entry & MASK32
            self.set_pc(image.entry)

    def setup_stack(self, top: int, size: int = 0x10000, *, set_sp: bool = True) -> int:
        """Reserve a stack and point ``sp`` at it; returns the stack pointer.

        ``top`` is the (exclusive) high end of the stack, rounded down to the
        16-byte alignment the RISC-V calling convention requires.  When a
        protection table is installed, ``[top-size, top)`` is registered as
        readable and writable so the stack does not fall foul of the loader's
        segment permissions.
        """
        stack_pointer = top & ~0xF
        if self.protection is not None:
            self.protection.add_region(stack_pointer - size, size, PERM_RW, "stack")
        if set_sp:
            self.set_reg(2, stack_pointer)
        return stack_pointer

    # ------------------------------------------------------------------
    # Memory access used by instructions (correct trap causes live here)
    # ------------------------------------------------------------------
    def load(self, addr: int, width: int, signed: bool) -> int:
        """Load ``width`` bytes for the guest, honouring the alignment policy."""
        if not self.allow_misaligned and addr % width:
            raise LoadAddressMisaligned(tval=addr, pc=self.pc)
        if self.protection is not None and not self.protection.check(addr, width, PERM_R):
            raise LoadAccessFault(
                tval=addr, pc=self.pc, message=f"no read permission at 0x{addr:08X}"
            )
        try:
            if width == 1:
                value = self.memory.read_u8(addr)
            elif width == 2:
                value = self.memory.read_u16(addr)
            else:
                value = self.memory.read_u32(addr)
        except MemoryError_ as exc:
            raise LoadAccessFault(tval=addr, pc=self.pc, message=str(exc)) from None
        if signed and width < 4:
            sign_bit = 1 << (width * 8 - 1)
            if value & sign_bit:
                value -= sign_bit << 1
        return value & MASK32

    def store(self, addr: int, width: int, value: int) -> None:
        """Store ``width`` bytes for the guest, honouring the alignment policy."""
        if not self.allow_misaligned and addr % width:
            raise StoreAddressMisaligned(tval=addr, pc=self.pc)
        if self.protection is not None and not self.protection.check(addr, width, PERM_W):
            raise StoreAccessFault(
                tval=addr, pc=self.pc, message=f"no write permission at 0x{addr:08X}"
            )
        try:
            if width == 1:
                self.memory.write_u8(addr, value)
            elif width == 2:
                self.memory.write_u16(addr, value)
            else:
                self.memory.write_u32(addr, value)
        except MemoryError_ as exc:
            raise StoreAccessFault(tval=addr, pc=self.pc, message=str(exc)) from None

    # ------------------------------------------------------------------
    # Machine timer
    # ------------------------------------------------------------------
    def attach_timer(self, timer) -> None:
        """Wire a machine timer to this hart.

        Maps it into memory using the CLINT layout so the guest can reach
        ``mtime``/``mtimecmp`` with ordinary loads and stores, and registers its
        MTIP output as an interrupt source so ``mip`` reports the live wire.
        """
        from .timer import CLINT_SIZE

        self.timer = timer
        self.memory.map_device(timer.base, CLINT_SIZE, timer, "clint")
        self.csr.add_interrupt_source(lambda: MIP_MTIP if timer.mtip else 0)
        if self.protection is not None:
            self.protection.add_region(timer.base, CLINT_SIZE, PERM_RW, "clint")

    def advance_time(self, ticks: int) -> int:
        """Move emulated time forward without executing anything.

        Deterministic by construction -- there is no wall clock anywhere in
        the emulator.
        """
        if self.timer is None:
            raise EmulatorError("this CPU has no timer attached")
        return self.timer.advance(ticks)

    # ------------------------------------------------------------------
    # Architectural trap entry
    # ------------------------------------------------------------------
    @property
    def traps_architecturally(self) -> bool:
        """Whether a synchronous exception vectors to ``mtvec``.

        Under ``trap_handling="auto"`` this becomes true as soon as the guest
        writes ``mtvec``: a machine that has installed a handler should use it,
        and one that has not should report the fault to the host instead of
        jumping to address zero.
        """
        if self.trap_handling == "architectural":
            return True
        if self.trap_handling == "host":
            return False
        return self.csr.trap_vector_installed

    def take_trap(self, exception) -> int:
        """Enter the machine-mode trap handler for ``exception``.

        Updates ``mepc``, ``mcause``, ``mtval`` and ``mstatus`` and sets the PC
        to the vector; returns the new PC.  The faulting instruction does not
        retire, so ``instret`` is untouched -- ``mepc`` points *at* it, which is
        what makes a handler able to inspect, fix up and re-execute it.
        """
        faulting_pc = self.pc if exception.pc is None else exception.pc
        target = self.csr.enter_trap(
            faulting_pc, exception.cause, exception.tval, interrupt=False
        )
        self.pc = target
        self.next_pc = target
        self.traps_taken += 1
        self.reservation = None
        return target

    def pending_interrupt(self) -> int | None:
        """The cause of a deliverable interrupt, or ``None``.

        An interrupt is deliverable only when it is pending in ``mip``, enabled
        in ``mie``, and permitted by ``mstatus.MIE``.  When several qualify the
        privileged specification's order applies: external, then software, then
        timer.
        """
        deliverable = self.csr.deliverable_interrupts()
        if not deliverable:
            return None
        for cause in INTERRUPT_PRIORITY:
            if deliverable & (1 << cause):
                return cause
        return None

    def take_interrupt(self, cause: int) -> int:
        """Enter the handler for an asynchronous interrupt.

        Goes through exactly the same :meth:`riscv.csr.CSRFile.enter_trap` as a
        synchronous exception -- the only differences are that ``mcause`` gets
        its interrupt bit, ``mtval`` is zero (an interrupt has no faulting
        address), and ``mepc`` is the instruction that has *not* run yet.
        """
        target = self.csr.enter_trap(self.pc, cause, tval=0, interrupt=True)
        self.pc = target
        self.next_pc = target
        self.traps_taken += 1
        self.interrupts_taken += 1
        self.reservation = None
        return target

    # ------------------------------------------------------------------
    # Environment hooks (ECALL / EBREAK)
    # ------------------------------------------------------------------
    def environment_call(self) -> None:
        if self.ecall_handler is None:
            raise EnvironmentCall(tval=0, pc=self.pc)
        self.ecall_handler(self)

    def breakpoint(self) -> None:
        if self.ebreak_handler is None:
            raise Breakpoint(tval=self.pc, pc=self.pc)
        self.ebreak_handler(self)

    # ------------------------------------------------------------------
    # Tracing
    # ------------------------------------------------------------------
    def enable_trace(self, stream=None, registers: bool = False) -> None:
        """Turn on instruction tracing (optionally including the write-back)."""
        self.trace = True
        if stream is not None:
            self.trace_stream = stream
        self.trace_registers = registers

    def disable_trace(self) -> None:
        self.trace = False

    def _trace_interrupt(self, pc: int, cause: int) -> None:
        """Trace an interrupt taken at an instruction boundary."""
        if self.trace and self.trace_hook is None:
            target = self.csr.trap_target(cause, True)
            print(
                f"PC=0x{pc:08X}  INST=--------  ! interrupt {cause} "
                f"-> mtvec 0x{target:08X}",
                file=self.trace_stream,
            )

    def _trace_trap(self, pc: int, exception) -> None:
        """Trace a fault taken before any instruction could be traced."""
        if self.trace and self.trace_hook is None:
            print(
                f"PC=0x{pc:08X}  INST=--------  ! {exception.mnemonic} "
                f"-> mtvec 0x{self.csr.trap_target(exception.cause, False):08X}",
                file=self.trace_stream,
            )

    def _trace_line(self, pc: int, raw: int, inst, compressed_name: str | None = None) -> str:
        if compressed_name is not None:
            from .disasm import disassemble_compressed

            return f"PC=0x{pc:08X}  INST=0x{raw:04X}      {disassemble_compressed(compressed_name, inst)}"
        from .disasm import disassemble

        return f"PC=0x{pc:08X}  INST=0x{raw:08X}  {disassemble(raw, pc=pc, instruction=inst)}"

    # ------------------------------------------------------------------
    # Fetch / decode / execute
    # ------------------------------------------------------------------
    def fetch(self, pc: int | None = None) -> int:
        """Read the instruction word at ``pc`` (defaults to the current PC)."""
        addr = self.pc if pc is None else pc & MASK32
        if addr & 0x3:
            raise InstructionAddressMisaligned(tval=addr, pc=addr)
        if self.protection is not None and not self.protection.check(addr, 4, PERM_X):
            raise InstructionAccessFault(
                tval=addr,
                pc=addr,
                message=f"no execute permission at 0x{addr:08X}",
            )
        try:
            return self.memory.read_u32(addr)
        except MemoryError_ as exc:
            raise InstructionAccessFault(tval=addr, pc=addr, message=str(exc)) from None

    def _fetch_halfword(self, addr: int) -> int:
        """Like :meth:`fetch` but for one 16-bit halfword (only ever used
        under C, where IALIGN is 16, so alignment is not re-checked here)."""
        if self.protection is not None and not self.protection.check(addr, 2, PERM_X):
            raise InstructionAccessFault(
                tval=addr, pc=addr, message=f"no execute permission at 0x{addr:08X}"
            )
        try:
            return self.memory.read_u16(addr)
        except MemoryError_ as exc:
            raise InstructionAccessFault(tval=addr, pc=addr, message=str(exc)) from None

    def fetch_instruction(self, pc: int | None = None):
        """Fetch the instruction at ``pc``, expanding it if compressed.

        Returns ``(word, size, compressed_name, raw)``: ``word`` is always a
        plain RV32I-encoded 32-bit word, ready for :meth:`decode` -- a
        compressed instruction has no semantics of its own, so it is expanded
        to the 32-bit encoding of the real instruction it stands for
        (:func:`riscv.rv32c.expand`) before decoding ever sees it. ``size`` is
        how many bytes it actually occupies in memory (2 or 4), which is how
        far ``next_pc`` must advance. ``compressed_name`` is the canonical
        ``C.*`` mnemonic for display, or ``None`` for an ordinary instruction.
        ``raw`` is the literal bytes at ``pc`` (2 or 4 of them) -- what a
        trace or disassembly should show as ``INST=``, since it is *not* the
        same value as ``word`` once a compressed instruction is expanded.
        """
        addr = self.pc if pc is None else pc & MASK32
        if not self.compressed:
            word = self.fetch(addr)
            return word, 4, None, word
        if addr & 0x1:
            raise InstructionAddressMisaligned(tval=addr, pc=addr)
        half = self._fetch_halfword(addr)
        if (half & 0x3) == 0x3:
            high = self._fetch_halfword((addr + 2) & MASK32)
            word = (half | (high << 16)) & MASK32
            return word, 4, None, word
        from .rv32c import expand

        word, name = expand(half, pc=addr)
        return word, 2, name, half

    def decode(self, word: int, pc: int | None = None):
        """Decode a word into an :class:`~riscv.decoder.Instruction`."""
        return self.decoder.decode(word, pc if pc is not None else self.pc)

    def step(self):
        """Execute exactly one instruction; return the decoded instruction.

        Raises :class:`~riscv.exceptions.RiscvException` on a trap, leaving the
        PC on the faulting instruction.  Returns ``None`` when the step was
        consumed by taking a trap or an interrupt instead of retiring an
        instruction.

        Emulated time advances by one tick per call, whatever the outcome.
        """
        if self.timer is None:
            return self._step()
        try:
            return self._step()
        finally:
            self.timer.tick()

    def _step(self):
        if self.halted:
            raise EmulatorError("cannot step: CPU is halted (call reset() first)")

        # An interrupt is taken *between* instructions and nowhere else: this is
        # the instruction boundary, before anything has been fetched.
        if self.traps_architecturally:
            cause = self.pending_interrupt()
            if cause is not None:
                self._trace_interrupt(self.pc, cause)
                self.take_interrupt(cause)
                return None

        pc = self.pc
        try:
            word, size, compressed_name, raw = self.fetch_instruction(pc)
            inst = self.decoder.decode(word, pc)
        except RiscvException as exc:
            self.last_exception = exc
            if self.traps_architecturally:
                self._trace_trap(pc, exc)
                self.take_trap(exc)
                return None  # nothing retired; the handler runs next
            raise

        self.last_instruction = inst
        self.next_pc = (pc + size) & MASK32

        trace_prefix = None
        if self.trace:
            if self.trace_hook is not None:
                self.trace_hook(self, pc, raw, inst)
            else:
                trace_prefix = self._trace_line(pc, raw, inst, compressed_name)

        writes_rd = inst.fmt in ("R", "I", "U", "J", "AMO") and inst.rd != 0
        before = self.regs.read(inst.rd) if (self.trace_registers and writes_rd) else 0
        try:
            inst.definition.execute(self, inst)
        except HaltRequest as request:
            self.halt(request.code)
        except RiscvException as exc:
            self.last_exception = exc
            if trace_prefix is not None:
                print(f"{trace_prefix}   ! {exc.mnemonic}", file=self.trace_stream)
            if self.traps_architecturally:
                self.take_trap(exc)
                return None  # the faulting instruction does not retire
            raise

        if trace_prefix is not None:
            suffix = ""
            if self.trace_registers and writes_rd and self.regs.read(inst.rd) != before:
                suffix = f"   x{inst.rd}=0x{self.regs.read(inst.rd):08X}"
            print(trace_prefix + suffix, file=self.trace_stream)

        self.pc = self.next_pc
        self.instret += 1
        return inst

    def run(self, max_steps: int | None = None, *, stop_on_breakpoint: bool = True) -> int:
        """Execute up to ``max_steps`` instructions; return how many ran.

        Stops early when the CPU halts or hits an address in
        :attr:`breakpoints`.  ``max_steps=None`` means "until halted", which
        will spin forever on a program that never halts -- pass a bound when
        running untrusted code.
        """
        executed = 0
        while not self.halted and (max_steps is None or executed < max_steps):
            if stop_on_breakpoint and executed and self.pc in self.breakpoints:
                break
            self.step()
            executed += 1
        return executed

    def run_until(self, pc: int, max_steps: int = 1_000_000) -> int:
        """Run until the PC reaches ``pc`` (or the CPU halts / the bound is hit)."""
        executed = 0
        while not self.halted and self.pc != (pc & MASK32) and executed < max_steps:
            self.step()
            executed += 1
        return executed

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def state(self) -> dict:
        """A JSON-friendly snapshot of all architectural + emulator state."""
        return {
            "pc": self.pc,
            "registers": self.regs.snapshot(),
            "instret": self.instret,
            "halted": self.halted,
            "halt_code": self.halt_code,
            "last_instruction": None if self.last_instruction is None else self.last_instruction.name,
            "csrs": self.csr.snapshot(),
            "privilege": self.csr.privilege,
            "traps_taken": self.traps_taken,
            "interrupts_taken": self.interrupts_taken,
            "mtime": None if self.timer is None else self.timer.mtime,
        }

    def restore_state(self, state: dict) -> None:
        """Inverse of :meth:`state` (registers, PC and counters only)."""
        self.regs.restore(state["registers"])
        self.set_pc(state["pc"])
        self.instret = state.get("instret", 0)
        self.halted = state.get("halted", False)
        self.halt_code = state.get("halt_code", 0)
        if "csrs" in state:
            self.csr.restore(state["csrs"])
        if "privilege" in state:
            self.csr.privilege = state["privilege"]
        self.traps_taken = state.get("traps_taken", 0)
        self.interrupts_taken = state.get("interrupts_taken", 0)
        if self.timer is not None and state.get("mtime") is not None:
            self.timer.set_time(state["mtime"])

    def dump_registers(self, abi: bool = True) -> str:
        return self.regs.format(abi=abi)

    def dump(self, abi: bool = True, csrs: bool = False) -> str:
        """Full human-readable state dump."""
        header = (
            f"pc = 0x{self.pc:08X}   instret = {self.instret}   halted = {self.halted}"
        )
        if self.traps_taken:
            header += f"   traps = {self.traps_taken}"
        if self.interrupts_taken:
            header += f" ({self.interrupts_taken} interrupt(s))"
        text = f"{header}\n{self.regs.format(abi=abi)}"
        if csrs:
            text += "\ncsrs:\n" + self.csr.format()
            if self.timer is not None:
                text += "\ntimer:\n" + self.timer.format()
        return text

    def disassemble_range(self, addr: int, count: int) -> str:
        """Disassemble ``count`` instructions starting at ``addr``."""
        from .disasm import disassemble

        lines = []
        for index in range(count):
            pc = (addr + index * 4) & MASK32
            word = self.memory.read_u32(pc)
            try:
                text = disassemble(word, pc=pc, instruction_set=self.decoder.instruction_set)
            except IllegalInstruction:
                text = ".word 0x%08X" % word
            lines.append(f"0x{pc:08X}:  {word:08X}  {text}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<CPU pc=0x{self.pc:08X} instret={self.instret} halted={self.halted}>"
