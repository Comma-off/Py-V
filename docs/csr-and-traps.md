# CSR storage and trap flow

How Py-V stores control and status registers, and what happens between a
faulting instruction and the guest's trap handler.

## Where the state lives

All CSR knowledge is in [`riscv/csr.py`](../riscv/csr.py). Nothing else in the
emulator knows a CSR number or a field position: `riscv/zicsr.py` implements the
six access instructions in terms of `CSRFile.read`/`write`, and `riscv/cpu.py`
calls exactly two methods — `enter_trap` and `machine_return`.

```
riscv/csr.py       CSRFile: definitions, values, privilege, trap transitions
riscv/zicsr.py     CSRRW/S/C[I] + MRET, as an ISA-table extension
riscv/cpu.py       step() routes an exception to take_trap(); nothing more
```

A CSR is a `CSRDefinition`: number, name, reset value, write mask, and an
optional `sanitize` hook. Those four fields express the specification's field
rules without any per-register code:

| Rule | How it is expressed |
| --- | --- |
| **WPRI** (reserved bits) | bits outside `write_mask` keep their old value, so writing `0xFFFFFFFF` to `mstatus` sets only MIE, MPIE and MPP |
| **WARL** (write any, read legal) | `sanitize` maps an illegal field onto a legal one — `mtvec`'s reserved MODE, `mstatus.MPP` |
| **Read-only** | address bits `[11:10] == 0b11`, per the specification; writing raises an illegal instruction |
| **Not implemented** | absent from `definitions`, so access raises an illegal instruction rather than reading zero |

Adding a CSR is one `define(CSRDefinition(...))` call. Adding a whole privilege
mode means adding its registers and letting `privilege` take another value —
the transitions already read and write it.

## Implemented registers

| CSR | Number | Notes |
| --- | --- | --- |
| `mstatus` | 0x300 | MIE, MPIE, MPP. Every other field is WPRI and reads zero. MPP is WARL to machine mode. |
| `misa` | 0x301 | MXL=1 (XLEN 32) plus the extension letters of the machine's actual instruction set. Read-only here. |
| `mie` | 0x304 | MSIE/MTIE/MEIE only; nothing sets them yet. |
| `mtvec` | 0x305 | BASE + MODE. Direct and Vectored; reserved modes fall back to Direct. |
| `mscratch` | 0x340 | Plain 32-bit scratch for handlers. |
| `mepc` | 0x341 | Low two bits always zero (IALIGN = 32). |
| `mcause` | 0x342 | Bit 31 set for an interrupt. |
| `mtval` | 0x343 | Fault-specific value. |
| `mip` | 0x344 | Read-only zero: there are no interrupt sources. |
| `mvendorid` `marchid` `mimpid` `mhartid` | 0xF11–0xF14 | Read-only identification, all zero (`mhartid` 0 = single hart). |

## Trap flow

```
     guest instruction faults
              |
              v
   +--------------------------+
   | riscv/cpu.py  step()     |   the exception carries cause, tval and pc
   |   fetch / decode / exec  |   already -- it always has
   +--------------------------+
              |
              | traps_architecturally?
              |
      no  <---+--->  yes
       |             |
       v             v
  raise to      CSRFile.enter_trap(pc, cause, tval)
  the host        mepc   <- faulting pc      (not pc+4)
  (a Python       mcause <- cause
   exception)     mtval  <- tval
                  mstatus.MPIE <- mstatus.MIE
                  mstatus.MIE  <- 0
                  mstatus.MPP  <- current privilege
                  privilege    <- machine
                       |
                       v
                  pc <- mtvec BASE      instret unchanged: nothing retired
                       |
                       v
              guest trap handler runs
                  reads mcause / mepc / mtval
                  fixes up mepc
                       |
                       v
                  MRET -> CSRFile.machine_return()
                  privilege    <- mstatus.MPP
                  mstatus.MIE  <- mstatus.MPIE
                  mstatus.MPIE <- 1
                  mstatus.MPP  <- machine
                  pc <- mepc
                       |
                       v
              guest execution resumes
```

`mepc` points **at** the faulting instruction, not after it. That is what lets a
handler inspect, emulate or fix up the instruction and re-execute it — and it is
why a handler that forgets `mepc += 4` loops forever. A test asserts exactly
that failure mode.

## When a trap is taken architecturally

`CPU(trap_handling=...)` chooses:

| Value | Behaviour |
| --- | --- |
| `"auto"` (default) | architectural once the guest has written `mtvec`, host otherwise |
| `"architectural"` | always vector to `mtvec` |
| `"host"` | always raise the Python exception out of `step()` |

The default exists because a machine that has never installed a handler has
nowhere useful to go: vectoring to address 0 would silently turn a fault into
garbage execution. Under `auto`, such a program still reports its fault to you,
and a program that installs a handler gets real trap entry. Writing `mtvec` is
the guest saying "I will handle these myself".

`ECALL` is the one exception to the exception: if a **host** `ecall_handler` is
installed (which `riscv.py run`/`exec` do by default), it intercepts `ECALL`
before it becomes a trap, so the host ABI keeps working. `--no-host` turns that
off and lets the guest take `ECALL` as cause 11. A program can therefore handle
its own faults while still using the host to exit — which is what
`examples/elf/trap_handler.S` does.

## The machine timer

The timer lives in [`riscv/timer.py`](../riscv/timer.py). RISC-V puts it in the
*memory* map rather than in CSRs — there is no `mtimecmp` CSR — so it is a small
memory-mapped device using the standard CLINT layout:

| Address | Register |
| --- | --- |
| `0x02000000` | `msip` (software interrupt; **not implemented**) |
| `0x02004000` | `mtimecmp`, 64-bit |
| `0x0200BFF8` | `mtime`, 64-bit |

Both registers are genuinely 64 bits. An RV32 guest reaches them as two 32-bit
halves and the underlying value is never truncated — a shortcut that breaks the
moment `mtime` passes 2³².

```
   mtime, mtimecmp  (64-bit, memory-mapped)
            |
            |  mtime >= mtimecmp
            v
      MTIP (a wire, not a register)
            |
            |  registered via CSRFile.add_interrupt_source()
            v
      mip.MTIP        <- read live; guest writes to mip are dropped
            |
            |  & mie.MTIE        (per-source enable)
            |  & mstatus.MIE     (global enable)
            v
   deliverable  -> taken at the next instruction boundary
            |
            v
   CSRFile.enter_trap(pc, cause=7, tval=0, interrupt=True)
            |                     ^^^^^^^^^^^^^^^^^^^^^^^
            |                     the same entry point exceptions use
            v
   mtvec: BASE (Direct)  or  BASE + 4*7 (Vectored)
```

**MTIP is a wire, not a register.** The guest cannot set or clear it by writing
`mip`; the only way to resolve a timer interrupt is to move `mtimecmp` past
`mtime`. A handler that returns without doing so is re-entered immediately,
because `MRET` restores `mstatus.MIE` while the wire is still asserted — a test
asserts exactly that.

**Interrupts are taken between instructions.** The check happens at the top of
`_step`, before anything is fetched, so an instruction is never interrupted
part-way through. `mepc` is the instruction that has *not* run yet, so `MRET`
resumes by executing it — unlike an exception handler, a timer handler must not
advance `mepc`.

**Delivery priority** when several interrupts are pending is the privileged
specification's: external, then software, then timer.

### Deterministic time

`mtime` has nothing to do with the wall clock. It advances by `ticks_per_step`
for every CPU step — whether that step retired an instruction or took a trap —
so a program fires the timer at exactly the same instruction on every machine
and every run. A test can also drive time directly:

```python
cpu = CPU(timer=MachineTimer(ticks_per_step=0))   # freeze time
cpu.advance_time(1000)                            # ...and move it by hand
```

`riscv.py exec` attaches a timer by default (`--no-timer` disables it,
`--timer-ticks N` changes the rate). The Python `CPU` defaults to **no** timer:
a machine without one maps no device and lets no time pass, so it costs exactly
what it always did.

## Deliberately not implemented

This milestone is the machine-mode foundation, not the whole privileged spec.
Left for later, each with its seam already in place:

* **Software and external interrupts.** `mip`/`mie` carry MSIP and MEIP and the
  priority order handles them, but nothing asserts those wires: `msip` reads as
  zero and there is no PLIC. `CSRFile.add_interrupt_source()` is where such a
  device would attach, exactly as the timer does.
* **U and S modes.** `privilege` and `mstatus.MPP` are real and change on trap
  entry and `MRET`, but machine mode is the only implemented level, so MPP is
  WARL to 3. Adding a mode means adding its CSRs and relaxing that.
* **Delegation** (`medeleg`/`mideleg`), which only means something with S mode.
* **`WFI`, `SRET`, `SFENCE.VMA`**, counters (`mcycle`, `minstret`,
  `time`), PMP, and virtual memory.
* **Nested traps.** Taking a trap inside a handler overwrites `mepc`/`mcause`,
  exactly as the hardware would; there is no shadow state, so a handler that
  faults loses its return address. That is architectural behaviour, not a
  limitation — but it is worth knowing.
