# Py-V — a RISC-V emulator in pure Python

A complete, self-contained emulator for the **RV32I** base integer instruction
set, the **M**, **A** and **C extensions**, and **machine-mode CSRs and
traps**, written entirely in Python with **no dependencies outside the
standard library**. It loads and runs **real ELF32 RISC-V
executables** built by a normal RISC-V toolchain — including ones a real
`-march=rv32imac` compiler produced, mixed 2- and 4-byte instructions and
all — and, because it also ships its own assembler, linker-free ELF writer
and test suite, **no toolchain is required** to build or test it.

```
$ python riscv.py run examples/loops.asm --trace | head -4
PC=0x00000000  INST=0x00008137  LUI  x2,0x8
PC=0x00000004  INST=0x00010113  ADDI x2,x2,0
PC=0x00000008  INST=0x00000413  ADDI x8,x0,0
PC=0x0000000C  INST=0x00000913  ADDI x18,x0,0
```

* All **40 RV32I instructions**, with correct signed/unsigned behaviour, 32-bit
  wrapping, immediate decoding, sign extension, alignment rules and trap causes.
* The **8 RV32M instructions** (multiply/divide), with truncate-toward-zero
  division and the spec's defined divide-by-zero and overflow results — selected
  per run, and auto-detected from an ELF's recorded architecture.
* The **11 RV32A instructions**: `LR.W`/`SC.W` and the nine `AMO*.W`
  read-modify-write ops, with a real load-reserved/store-conditional
  reservation (invalidated by any trap, interrupt or reset) — enough to run a
  textbook lock-free compare-and-swap loop.
* **RV32C (compressed)**: mixed 2-/4-byte fetch and IALIGN=16, with every one
  of the 26 compressed encodings expanding straight to its real RV32I
  instruction and executing through the exact same code — verified against
  192 real `riscv64-linux-gnu-as` encodings and a GCC `-march=rv32imac`
  binary that actually runs to completion.
* **Zicsr and machine-mode traps**: the six CSR instructions, 13 machine CSRs,
  real architectural trap entry (`mepc`/`mcause`/`mtval`/`mstatus` → `mtvec`)
  and `MRET`. A guest can install its own trap handler and return from it.
* **Machine timer interrupts**: 64-bit `mtime`/`mtimecmp` mapped CLINT-style,
  MTIP gated by `mie.MTIE` and `mstatus.MIE`, delivered at an instruction
  boundary through the same trap path, in Direct or Vectored mode — on
  **deterministic emulated time**, never the wall clock.
* An **ELF32 little-endian RISC-V loader**: `PT_LOAD` segments, `.bss`
  zero-fill, entry point, segment R/W/X permissions, optional symbols.
* **623 automated tests** plus a **623-check built-in self-test**, both of which
  run without pytest and without any external toolchain.
* A **built-in assembler**, **ELF writer**, **disassembler** and
  **instruction tracer**.
* Modular by design: CPU, memory, decoding, execution and testing are separate,
  and the instruction set is a data table you can extend without touching the core.

---

## Requirements

Python 3.10 or newer. Nothing else. (pytest is optional — the bundled runner
uses only `unittest` from the standard library.)

## Quick start

```bash
git clone https://github.com/Comma-off/Py-V.git && cd Py-V

python riscv.py --self-test               # prove the emulator works
python riscv.py run examples/hello.asm    # Hello, RISC-V!
python riscv.py run examples/loops.asm --trace-regs
python -m pytest                          # or: python tests/run_tests.py

# Build the ELF examples (no toolchain needed) and run one:
python examples/elf/build.py
python riscv.py exec examples/elf/build/functions.elf
```

## Supported ISA and binary formats

| | |
| --- | --- |
| **ISA** | RV32I (40), RV32IM (+8), RV32A (+11), Zicsr + `MRET` (+7), and RV32C (+0 -- an alternate 16-bit *encoding* of instructions already counted) — 66 instructions at the widest (`rv32imac_zicsr`). No F/D, no U/S modes. |
| **Devices** | Machine timer (CLINT `mtime`/`mtimecmp`). No UART, PLIC or PMP. |
| **ISA selection** | `--isa` takes any of `rv32i`/`rv32im`/`rv32ia`/`rv32ima` crossed with an optional `c` (`rv32imac`, `rv32iac`, ...) and an optional `_zicsr` suffix -- 16 names in total (see `available_isas()`); default `auto`: what the ELF declares, else `rv32im_zicsr`. |
| **Binary formats** | ELF32 little-endian RISC-V (`EM_RISCV`, `ELFCLASS32`, `ELFDATA2LSB`); raw instruction words (`.bin`); hex words (`.hex`); assembly source (`.asm`, `.S`). |
| **Host interface** | One `ECALL` convention — see [docs/abi.md](docs/abi.md). |

## Command line

| Command | What it does |
| --- | --- |
| `python riscv.py exec PROG.elf` | Load and run an ELF32 RISC-V executable |
| `python riscv.py elf PROG.elf` | Inspect ELF headers and loadable segments |
| `python riscv.py disasm PROG` | Disassemble an ELF's code, or a flat binary |
| `python riscv.py run PROG` | Run a `.asm`/`.s`, `.bin` or `.hex` program |
| `python riscv.py asm SRC -o OUT` | Assemble to `.bin`, `.hex` or `.elf` |
| `python riscv.py --self-test` | Run the built-in validation suite |

(`assemble` and `dis` are accepted as aliases of `asm` and `disasm`.)

### Running an ELF

```bash
python riscv.py exec program.elf
python riscv.py exec program.elf --trace
python riscv.py exec program.elf --max-instructions 1000000
python riscv.py exec program.elf --stack 0x80000 --dump-regs
```

`exec` flags:

```
--trace / --trace-regs      instruction trace, optionally with write-back
--max-instructions N        instruction budget (default 1000000); stops runaway
                            programs instead of hanging the CLI
--isa NAME                  rv32i / rv32im / rv32ia / rv32ima, any of those
                            with a trailing c (rv32imac, ...) and/or _zicsr,
                            or auto
--stack ADDR                stack top (default: a page above the image + 64 KiB)
--stack-size N              stack size (default 0x10000)
--memory-size N             RAM size; grown automatically to fit the image
--no-protect                do not enforce the ELF segments' R/W/X permissions
--no-host                   leave ECALL as an architectural trap
--no-timer                  do not attach the machine timer (no CLINT, no time)
--timer-ticks N             mtime ticks per step (default 1; 0 freezes time)
--allow-misaligned          do not trap misaligned loads/stores
--breakpoint ADDR           stop before executing ADDR (repeatable)
--dump-regs / --dump-csrs   register / CSR + timer dump when execution stops
--dump-mem A:N              hexdump a memory range when execution stops
--verbose                   report the load layout
--quiet                     suppress the summary line
```

Inspecting and disassembling:

```bash
python riscv.py elf program.elf --symbols --sections
python riscv.py disasm program.elf --symbols --abi
```

`disasm` disassembles executable **sections** when the file has a section table,
because a linker maps the ELF header and read-only data into the same `R-X`
segment as the code; `--all` forces whole-segment disassembly instead. Both
`disasm` and `asm` also take `--isa`, which decides whether an `M` or CSR
encoding disassembles as an instruction or as a `.word`.

Useful `run` flags:

```
--trace                 one line per instruction (PC / INST / disassembly)
--trace-regs            ...and the register each instruction wrote
--max-steps N           instruction budget (default 1,000,000)
--isa NAME              instruction set (default auto: the widest available)
--dump-regs             print the register file when execution stops
--dump-csrs             print the CSRs and the timer when execution stops
--dump-mem 0x400:64     hexdump a memory range when execution stops
--breakpoint 0x1C       stop before executing this address (repeatable)
--base 0x8000           load address
--pc 0x8010             override the entry point
--memory-size 0x100000  RAM size
--allow-misaligned      do not trap misaligned loads/stores
--no-host               leave ECALL as an architectural trap (no syscalls)
--no-timer              do not attach the machine timer
--timer-ticks N         mtime ticks per step (default 1; 0 freezes time)
--quiet                 suppress the summary line
```

Exit status: `0` success, `1` guest exited non-zero, `2` bad input,
`3` the guest took an unhandled trap, `4` emulator misuse.

## Python API

Everything the emulator can do is a one-liner on `CPU`:

```python
from riscv import CPU, assemble

cpu = CPU()                                  # 1 MiB of RAM, PC = 0
program = assemble("""
    addi x1, x0, 10
    addi x2, x0, 20
    add  x3, x1, x2
""")

cpu.load_program(program.image)              # bytes or a list of words
cpu.run(3)                                   # execute N instructions
assert cpu.get_reg(3) == 30
```

| Need | Call |
| --- | --- |
| Load an ELF | `image = load_elf("program.elf")` |
| Place it in memory | `cpu.load_image(image)` (sets the PC to `image.entry`) |
| Enforce segment permissions | `cpu.load_image(image, enforce_permissions=True)` |
| Set up a stack | `cpu.setup_stack(top=0x80000, size=0x10000)` |
| Reset the CPU | `cpu.reset()` / `cpu.reset(pc=0x8000)` |
| Set the PC | `cpu.set_pc(addr)` (`cpu.get_pc()`, `cpu.pc`) |
| Read/write registers | `cpu.get_reg(3)`, `cpu.set_reg(3, 30)`, `cpu.read_register("sp")`, `cpu.write_register("a0", 1)` |
| Signed register view | `cpu.get_reg_signed(3)` |
| Write memory | `cpu.write_u8/u16/u32(addr, v)`, `cpu.write_bytes(addr, b"...")` |
| Read memory | `cpu.read_u8/u16/u32(addr)`, `cpu.read_bytes(addr, n)` |
| Execute one instruction | `cpu.step()` → the decoded instruction |
| Execute N instructions | `cpu.run(n)` → how many actually ran |
| Run to an address | `cpu.run_until(addr)` |
| Inspect CPU state | `cpu.state()` (dict), `cpu.dump()`, `cpu.dump_registers()` |
| Inspect memory | `cpu.memory.hexdump(addr, n)`, `cpu.disassemble_range(addr, n)` |
| Enable/disable tracing | `cpu.enable_trace(stream)`, `cpu.disable_trace()` |
| Breakpoints | `cpu.breakpoints.add(addr)` |
| Restore a snapshot | `cpu.restore_state(state)` |
| Read/write a CSR | `cpu.csr.get("mtvec")`, `cpu.csr.set("mtvec", 0x800)` |
| Inspect trap state | `cpu.csr.format()`, `cpu.dump(csrs=True)`, `cpu.traps_taken` |
| Attach a machine timer | `CPU(timer=MachineTimer())` / `cpu.attach_timer(t)` |
| Advance emulated time | `cpu.advance_time(ticks)` (never a wall clock) |
| Choose the ISA | `CPU(instruction_set=RV32IM_ZICSR)`, `get_isa("rv32im")` |

Running an ELF from Python, the way `riscv.py exec` does it:

```python
from riscv import CPU, HostEnvironment, load_elf
from riscv.timer import MachineTimer
from riscv.zicsr import RV32IM_ZICSR

image = load_elf("program.elf")
cpu = CPU(
    memory_size=1 << 20,
    instruction_set=RV32IM_ZICSR,                 # or get_isa(image.isa_name)
    ecall_handler=HostEnvironment(),
    timer=MachineTimer(),                         # CLINT at 0x02000000
)
cpu.load_image(image, enforce_permissions=True)   # segments + entry point
cpu.setup_stack(0x80000)                          # sp, and an RW stack region
cpu.run(1_000_000)                                # bounded: never hangs
print(cpu.halt_code)
```

A bare `CPU()` is a plain RV32I machine with no timer and no CSR extension —
pass `instruction_set` and `timer` to get the rest.

The loader is usable entirely on its own — it never touches a CPU:

```python
from riscv import load_elf, Memory

image = load_elf("program.elf")
print(hex(image.entry), len(image.segments))
for segment in image.segments:
    print(f"{segment.vaddr:#010x} {segment.filesz:#x}/{segment.memsz:#x} {segment.permissions}")
image.load_into(Memory(1 << 20))                  # no CPU involved
```

Fetch, decode and execute are also separable, which is what makes the decoder
independently testable:

```python
word = cpu.fetch()          # raises InstructionAccessFault / ...Misaligned
inst = cpu.decode(word)     # raises IllegalInstruction; pure function of `word`
inst.definition.execute(cpu, inst)
```

Traps surface as Python exceptions carrying the architectural cause code:

```python
from riscv import IllegalInstruction
try:
    cpu.step()
except IllegalInstruction as trap:
    print(trap.cause, hex(trap.tval), hex(trap.pc))   # 2, 0x0, 0x1c
```

## Testing

Three independent ways to run the tests, all self-contained:

```bash
python -m pytest              # 572 tests
python tests/run_tests.py     # same tests, no pytest needed
python riscv.py --self-test   # 289 internal checks + the full suite
```

Tests that need an external RISC-V toolchain skip themselves when one is not
installed; everything else runs anywhere Python does.

The bundled runner prints one line per test and a summary, and exits non-zero on
any failure:

```
[PASS] ADD
[PASS] ADDI
[PASS] LW/SW
[PASS] BEQ
[PASS] JAL
[PASS] Sign extension
[PASS] x0 immutability
...
Tests:  572
Passed: 572
Failed: 0
```

`python tests/run_tests.py -k branch` filters by name; `-q` prints only the summary.

### What is actually tested

| File | Coverage |
| --- | --- |
| `tests/test_bits.py` | masking, sign extension, signed/unsigned conversion |
| `tests/test_memory.py` | little-endian order, paging, bounds, lazy allocation |
| `tests/test_registers.py` | register reads/writes, **x0 immutability**, reset, ABI names |
| `tests/test_decoder.py` | field extraction, every immediate format, **illegal encodings** |
| `tests/test_encoder.py` | **32 encodings checked against real toolchain output**, exhaustive encode↔decode round-trips, extensibility |
| `tests/test_disasm.py` | disassembly text, trace-line format, disasm→asm round-trip |
| `tests/test_assembler.py` | syntax, labels, pseudo-instructions, directives, errors |
| `tests/test_cpu.py` | stepping, PC updates, reset, every trap cause, tracing, host API |
| `tests/test_instructions.py` | **at least one test per RV32I instruction**, plus wrapping/shift/sign edge cases |
| `tests/test_programs.py` | loops, branches, `JAL`/`JALR` calls, recursion, memcpy, strlen, and every `examples/` program |
| `tests/test_rv32m.py` | every M instruction, **truncation and sign rules**, divide-by-zero, overflow, the division identity over random operands, and ISA selection |
| `tests/test_csr.py` | all six CSR instructions, their read/write side-effect rules, every CSR's field behaviour, trap entry for all nine causes, `MRET`, and **a trap handler written in assembly and run on the emulated CPU** |
| `tests/test_timer.py` | 64-bit `mtime`/`mtimecmp`, low/high access, 2³² rollover, MTIP boundaries, MTIE/MIE gating, instruction-boundary delivery, Direct and Vectored vectoring, re-triggering, and **determinism across repeated runs** |
| `tests/test_elf.py` | ELF headers, immediates of every field, and **every way a file can be malformed** |
| `tests/test_elf_loader.py` | segment placement, `.bss` zero-fill, sparse behaviour, permission enforcement |
| `tests/test_elf_integration.py` | the whole path: ELF → loader → memory → CPU → `ECALL` → exit status, plus the CLI as a subprocess |
| `tests/test_real_binaries.py` | **differential tests against a real toolchain** — skipped when it is absent |

Tests assert architectural state (registers, PC, memory, trap causes) rather
than "it didn't crash", and several check state *between* instructions rather
than only at the end.

The encoder tests are worth singling out: 32 instruction encodings are compared
against machine code produced by a standard RISC-V toolchain, so the bit layout
is validated against something external, not merely against itself.

## Example programs

`examples/` contains the six programs below; each file's header comment states
the exact final register state, and `tests/test_programs.py` asserts it. (The
directory also holds `stress.asm` and `test.asm`, which are scratch programs —
they assemble and run, but no test pins their final state. `stress.asm` puts its
stack near the top of the address space, so it needs
`--memory-size 0x100000000`.)

| Program | Demonstrates |
| --- | --- |
| `hello.asm` | `la`, a data section, host `write`/`exit` syscalls |
| `arithmetic.asm` | arithmetic, logic, shifts, silent overflow, signed vs unsigned |
| `branches.asm` | all six conditional branches, taken and not taken |
| `loops.asm` | counted loops, nested loops, multiplication by repeated addition |
| `memory.asm` | all load/store widths, little-endianness, sign vs zero extension |
| `functions.asm` | calling convention, stack frames, recursion, indirect calls |

Each also ships as a `.hex` machine-code dump (regenerate with
`python examples/build_examples.py`), so the programs are runnable even with the
assembler completely out of the picture:

```bash
python riscv.py run examples/memory.hex --dump-regs
```

A test asserts the `.hex` files stay in sync with their `.asm` sources.

### ELF examples

`examples/elf/` holds the same kind of programs as real `.S` sources that build
**both ways** — with the bundled assembler and with a real RISC-V toolchain:

| Program | Demonstrates |
| --- | --- |
| `hello.S` | the minimum viable program: exit through the host ABI |
| `arithmetic.S` | register arithmetic, logic, signed vs unsigned comparison |
| `branches.S` | every conditional branch and a counted loop |
| `memory.S` | `.data`, `.bss` (loader zero-fill), the stack, all load/store widths |
| `functions.S` | `jal`, `jalr`, stack frames, `ra`, callee-saved registers, recursion |
| `trap.S` | deliberately takes a trap, to check it is reported correctly |
| `multiply.S` | all eight RV32M instructions, including every division edge case |
| `trap_handler.S` | a guest-defined trap handler: `mtvec`, three faults, `mcause`/`mepc`/`mtval`, `MRET` |
| `timer_interrupt.S` | a real timer interrupt: CLINT `mtimecmp`, MTIE/MIE, a **vectored** table, resolve, `MRET` |

`v0.2_test.S` also lives here — an end-to-end integration program covering ELF
loading, `.data`/`.bss`, the stack and calls. It assembles and exits 0, but it is
not in `build.py`'s program list and no test pins its state, so build it
explicitly:
`python riscv.py asm examples/elf/v0.2_test.S -o v0.2_test.elf --base 0x10000`.

```bash
python examples/elf/build.py                      # no toolchain needed
python riscv.py exec examples/elf/build/memory.elf

cd examples/elf && make                           # with a real toolchain
python riscv.py exec examples/elf/build/memory.elf
```

Each file's header comment states the exact expected final register state, and
`tests/test_elf_integration.py` asserts it for the bundled build while
`tests/test_real_binaries.py` asserts *the same values* for the toolchain build.

### Building with a real RISC-V toolchain

Nothing here requires one, but the emulator is meant to run what one produces:

```bash
riscv32-linux-gnu-as  -march=rv32i -mabi=ilp32 -mno-relax hello.S -o hello.o
riscv32-linux-gnu-ld  -m elf32lriscv -Ttext=0x10000 --no-relax hello.o -o hello.elf
python riscv.py exec hello.elf
```

Freestanding C works too:

```bash
riscv32-linux-gnu-gcc -march=rv32i -mabi=ilp32 -O2 -mno-relax \
    -nostdlib -nostartfiles -ffreestanding -e _start \
    -Wl,-Ttext=0x10000 -Wl,--build-id=none prog.c -o prog.elf
python riscv.py exec prog.elf
```

Any RV32I-capable toolchain works (`riscv32-unknown-elf-`, `riscv64-linux-gnu-`,
…); set `RISCV_TOOLPREFIX` to point the toolchain tests at yours.

**On linker relaxation:** with relaxation enabled, GNU `ld` rewrites
`la reg, symbol` into `gp`-relative addressing, which only works if the program
sets `gp` to `__global_pointer$` itself — normally `crt0.o`'s job. Freestanding
code should either do that in `_start` or link with `-mno-relax`/`--no-relax`.
See [docs/abi.md](docs/abi.md#gp-is-not-initialised).

---

## Architecture

```
riscv/
├── bits.py          32-bit primitives: mask32, to_signed, sign_extend
├── exceptions.py    architectural traps (with cause codes) + host errors
├── memory.py        sparse little-endian physical memory
├── registers.py     the 32-entry register file; x0 hardwired to zero
├── isa.py           InstructionSet/InstructionDef tables — the extension point
├── instructions.py  RV32I semantics: one function per instruction
├── rv32m.py         the M extension: multiply/divide, built on top of RV32I
├── zicsr.py         the Zicsr extension (CSR access) and MRET
├── csr.py           control and status registers + machine-mode trap state
├── timer.py         the machine timer: 64-bit mtime/mtimecmp, MTIP
├── decoder.py       word → Instruction (pure, memoised)
├── encoder.py       fields → word (exact inverse of the decoder)
├── disasm.py        Instruction → text (trace mode, `disasm` command)
├── assembler.py     two-pass assembler for .asm/.S sources
├── elf.py           ELF32 RISC-V reader/loader — independent of the CPU
├── elfwriter.py     ELF32 RISC-V writer (examples and test fixtures)
├── host.py          optional ECALL environment (print/exit) — not required
├── cpu.py           the hart: fetch/decode/execute, traps, interrupts, tracing
├── selftest.py      built-in validation suite
└── cli.py           command-line front end
```

### Loading a program

```
   ELF file
      |  riscv/elf.py — validates, parses program headers
      v
   LoadedImage        segments (vaddr, data, memsz, R/W/X) + entry point
      |  cpu.load_image() — a neutral description, no ELF knowledge needed
      v
   Memory             file bytes copied, .bss zero-filled (sparsely)
      |
      v
   CPU                pc = e_entry, sp set up, permissions installed
```

The executable format and the ISA never meet: `riscv/elf.py` imports nothing
from `riscv/cpu.py`, and `riscv/cpu.py` imports nothing from `riscv/elf.py`.
`CPU.load_image` consumes a *duck-typed* image — anything with `segments` and an
`entry` — so a future Mach-O or flat-image loader plugs in with no CPU changes,
and `riscv/instructions.py` remains entirely unaware that ELF exists.

Dependencies flow strictly downward — `bits` → `exceptions` → `memory`/`registers`
→ `isa` → `instructions` → `decoder`/`encoder` → `disasm`/`assembler` → `cpu` →
`cli`. There are no cycles, so any layer can be imported and tested alone.

### The execution loop

`CPU.step()` does exactly one instruction:

1. **Check PC alignment** → `InstructionAddressMisaligned` (cause 0)
2. **Fetch** 4 bytes at PC → `InstructionAccessFault` (cause 1)
3. **Decode** → `IllegalInstruction` (cause 2)
4. `next_pc = pc + 4`
5. **Execute** — the handler may overwrite `next_pc`, or trap
6. `pc = next_pc`; `instret += 1`

If a trap escapes steps 1–5, the PC is left **on the faulting instruction** —
the value a real machine would put in `mepc` — and `instret` is *not*
incremented, so you can inspect the exact machine state at the fault and, if you
want, emulate a handler by moving the PC yourself.

### Why these seams

* **Decoding is a pure function of the instruction word.** It never reads CPU or
  memory state, so it is trivially testable, position-independent, and safe to
  memoise (the decode cache is a plain dict, cleared on reset).
* **Instruction semantics talk to a narrow CPU interface** — `get_reg`,
  `set_reg`, `pc`, `next_pc`, `load`, `store`. No instruction knows how memory is
  paged, how traps are represented, or how tracing works. `x0`'s hardwiring lives
  in the register file, so no instruction has to remember it.
* **Alignment policy and trap causes live in the CPU**, not in memory, because
  only the CPU knows whether an access is a fetch, a load or a store — and each
  raises a different cause.
* **The ISA is a table, not a `match` statement.** Each entry is
  `(name, match, mask, format, execute)`. The decoder finds the entry whose
  `word & mask == match`; the encoder reads the same table backwards. They cannot
  drift apart, and adding instructions never touches the core.
* **The host environment is optional.** With no handler, `ECALL` is just trap
  cause 11. `riscv/host.py` is one callable that a CPU can be given; nothing in
  the core depends on it. That seam is where a supervisor would eventually live.

### Memory permissions

The loader preserves each segment's `R`/`W`/`X` flags, and
`riscv.memory.MemoryProtection` optionally enforces them. It is a sorted region
table, not an MMU — no translation, no page tables, no privilege levels:

* installed only when you ask (`cpu.load_image(..., enforce_permissions=True)`,
  which is what `exec` does unless `--no-protect`);
* consulted by the CPU on every fetch (`X`), load (`R`) and store (`W`), raising
  the corresponding access fault — cause 1, 5 or 7;
* addresses outside every region default to **read/write but not execute**, so
  jumping into the weeds faults at the fetch while scratch memory stays usable;
* on overlap, the region with the highest start address wins — a stated rule,
  not an accident of insertion order;
* the host API (`cpu.read_u32`, `cpu.write_bytes`, …) is **not** subject to it;
  a debugger has to be able to patch `.text`.

Cost when disabled is one `is not None` check per access.

### Porting to Luau

The design deliberately avoids anything hard to express in Luau:

* All state is plain integers, lists and dicts — no objects with exotic protocols.
* Values are unsigned 32-bit ints everywhere, with sign only applied explicitly
  at the point of use, so a language without a native 32-bit type behaves the
  same way (Luau's `bit32` library maps directly onto `riscv/bits.py`).
* The ISA table is data; in Luau it becomes an array of tables with a function
  field, and the decoder's mask/match loop translates line for line.
* The only Python-specific conveniences are decorators (`@RV32I.define`, purely
  sugar over `InstructionSet.add`) and `__slots__` classes (performance only).

---

## RV32I implementation decisions

Details that separate a correct core from an approximately-correct one, and how
each is handled here.

**Everything is unsigned internally.** Registers and memory hold Python ints in
`[0, 2**32)`. Signed interpretation happens explicitly via `to_signed()` at the
few places that need it (`SLT`, `SLTI`, `BLT`, `BGE`, `SRA`, `SRAI`). Python's
arbitrary-precision ints make it very easy to "accidentally" get 33-bit results,
so every arithmetic result is masked with `& 0xFFFFFFFF` at the point of write.

**Overflow wraps, silently.** RV32I has no trapping arithmetic:
`0x7FFFFFFF + 1 == 0x80000000`, and `0 - 1 == 0xFFFFFFFF`. Tested explicitly.

**`x0` is enforced in one place.** `RegisterFile.write` drops writes to index 0.
Instructions never special-case it — which matters, because writing `x0` is the
*normal* encoding of `nop` (`addi x0, x0, 0`), `j` (`jal x0, off`) and `ret`
(`jalr x0, 0(ra)`). `JAL`/`JALR` still compute and "write" the link value; it is
discarded by the register file.

**Shift amounts are 5 bits.** `SLL`/`SRL`/`SRA` use `rs2 & 0x1F`, so a shift of
32 is a shift of 0 and a shift of −1 is a shift of 31. `SLLI`/`SRLI`/`SRAI` pin
`funct7` in their decode mask, so a stray bit in `imm[11:5]` is an **illegal
instruction** rather than a silently-ignored encoding.

**`SRA` is arithmetic, `SRL` is logical.** `SRA` shifts the *signed*
interpretation (Python's `>>` on a negative int already replicates the sign) and
re-masks; `SRL` shifts the unsigned value.

**`SLTIU` compares against the sign-extended immediate.** The immediate is
sign-extended first and *then* compared as unsigned, so `sltiu rd, rs, -1`
compares against `0xFFFFFFFF` — this is what makes `seqz` (`sltiu rd, rs, 1`)
work. Same for `ANDI`/`ORI`/`XORI`: the immediate sign-extends, so
`andi rd, rs, -256` masks with `0xFFFFFF00`.

**`JALR` clears bit 0 *after* adding the immediate** (`(rs1 + imm) & ~1`), and
computes the target *before* writing `rd`, so `jalr ra, 0(ra)` works.

**Jump/branch alignment (IALIGN = 32).** With no compressed extension, a target
that is not 4-byte aligned raises `InstructionAddressMisaligned` **before** the
jump takes effect and before `rd` is written. An untaken branch never checks its
target at all.

**Misaligned data accesses trap by default** (`LoadAddressMisaligned` cause 4 /
`StoreAddressMisaligned` cause 6), matching hardware without the misaligned
extension. `CPU(allow_misaligned=True)` emulates a core that handles them.
Byte accesses are never misaligned.

**Load extension is per-instruction.** `LB`/`LH` sign-extend; `LBU`/`LHU`
zero-extend; `LW` fills the register. Stores truncate to their width and leave
neighbouring bytes untouched.

**`AUIPC` adds to the address of the `AUIPC` itself**, not to `pc + 4`.

**`LUI`'s immediate is already shifted** by the decoder (`word & 0xFFFFF000`),
so the executor is a plain move — that also makes `lui rd, 0` correctly clear
the low 12 bits of `rd`.

**Immediates are reconstructed exactly as the spec scrambles them.** B- and
J-type immediates are bit-shuffled so that the sign bit is always word bit 31;
`riscv/decoder.py` documents the layout and each format has its own tested
extractor.

**Unmapped memory reads as zero**, so fetching from a hole yields `0x00000000` —
which decodes as an illegal instruction, exactly the failure you want. Accesses
outside the configured address space raise access faults (causes 1/5/7).

**`FENCE` is a no-op** (single in-order hart), but it is decoded, counted
(`cpu.fences`) and its `pred`/`succ` fields are disassembled properly.

**`ECALL`/`EBREAK` are the only SYSTEM encodings.** Everything else in opcode
`0x73` — every CSR instruction, `MRET`, `WFI` — is an illegal instruction,
because CSRs are deliberately not implemented yet. Their encodings are free for
a future extension to claim.

## The M extension (RV32M)

Eight instructions — `MUL MULH MULHSU MULHU DIV DIVU REM REMU` — living in
[riscv/rv32m.py](riscv/rv32m.py). They are an *extension*, not an edit:
`RV32I` stays exactly 40 instructions and `RV32IM` is a copy with these added,
so the base ISA remains a faithful reference and `MUL` still traps as an illegal
instruction on an RV32I machine.

```bash
python riscv.py exec prog.elf                 # auto: uses what the ELF declares
python riscv.py exec prog.elf --isa rv32im    # force the M extension on
python riscv.py exec prog.elf --isa rv32i     # strict base ISA: M traps
python examples/elf/build.py multiply && python riscv.py exec examples/elf/build/multiply.elf
```

```python
from riscv import CPU, RV32IM
cpu = CPU(instruction_set=RV32IM)
```

### Architecture auto-detection

A toolchain records what it built for in the ELF's `.riscv.attributes` section
(`Tag_RISCV_arch`, e.g. `"rv32i2p1_m2p0_zmmul1p0"`). The loader decodes it, and
`--isa auto` — the default — uses it:

```
$ python riscv.py elf multiply.elf | head -5
ELF32 RISC-V (little-endian)
Type: EXEC
Entry: 0x00010000
Flags: 0x00000000 [soft-float]
Arch:  rv32i2p1_m2p0_zmmul1p0  -> rv32im
```

An `rv32i`-only binary therefore runs on a *strict* RV32I machine, so a
toolchain that wrongly emitted a `MUL` into it is caught rather than tolerated.
An architecture naming something unimplemented (`f`, `d`, `c`, …) maps to
`None` rather than pretending to support it; `m`, `a` and `zicsr`, in any
combination, resolve to the matching registered set (`rv32i2p1_a2p1` →
`rv32ia`, `rv32i2p1_m2p0_a2p1_zicsr2p0` → `rv32ima_zicsr`, and so on).
Our own ELF writer emits the same section, so files it produces are
self-describing too.

### Implementation decisions

**Division truncates toward zero; Python's `//` floors.** `-7 / 2` is `-3` in
RISC-V but `-4` in Python, and `-7 % 2` is `-1` in RISC-V but `1` in Python —
the remainder takes the sign of the *dividend*. `divide_signed` and
`remainder_signed` compute on magnitudes and reapply the sign rather than using
the operators. This is the single most likely bug in a hand-written RV32M, so it
is tested directly, against the identity `rs1 == rs2·DIV + REM` over random
operands, and against GCC's own constant folding.

**Division never traps.** RV32M has no divide-by-zero exception; the spec fixes
the result instead (spec table 7.1):

| Condition | `DIVU` | `REMU` | `DIV` | `REM` |
| --- | --- | --- | --- | --- |
| Divisor is zero | `2³²−1` | dividend | `−1` | dividend |
| Overflow (`−2³¹ / −1`) | — | — | `−2³¹` | `0` |

**The three high multiplies really are three.** On the same bit pattern
`0xFFFFFFFF × 0xFFFFFFFF`, `MULH` gives `0`, `MULHU` gives `0xFFFFFFFE` and
`MULHSU` gives `0xFFFFFFFF`. A test asserts all three at once, because an
implementation that conflates them passes any test that only checks one.
`MUL` itself is signedness-agnostic: the low half of a two's-complement product
is the same either way.

**High words come from Python's arithmetic shift.** `(product >> 32) & 0xFFFFFFFF`
is the two's-complement high word, because `>>` on a negative Python int floors
— which is exactly an arithmetic shift.

### How it was verified

Besides the unit tests, RV32M is checked against the real toolchain three ways:
`multiply.S` is assembled by GNU `as -march=rv32im` and must reach the same
state as the bundled build; our decoder is compared against `objdump -d`; and a
generated C program computes each result at runtime (through `volatile`) while
**GCC constant-folds the expected value at compile time**, so GCC's own model of
the ISA is the oracle. Breaking `DIV` to floor makes that test report
`check #3 disagreed with GCC's compile-time constant (operands (-7, 2))`.

## The A extension (RV32A)

Eleven instructions — `LR.W SC.W AMOSWAP.W AMOADD.W AMOXOR.W AMOOR.W AMOAND.W
AMOMIN.W AMOMAX.W AMOMINU.W AMOMAXU.W` — living in
[riscv/rv32a.py](riscv/rv32a.py), built the same way RV32M was: `RV32IA` and
`RV32IMA` are copies of `RV32I`/`RV32IM` with these added, so `AMOADD.W` still
traps as illegal on a machine built without it.

```bash
python riscv.py exec prog.elf --isa rv32ima     # M and A together
python riscv.py run cas.asm --isa rv32ia        # A alone, no multiply
```

```python
from riscv import CPU
from riscv.rv32a import RV32IMA
cpu = CPU(instruction_set=RV32IMA)
```

**Atomicity is free on one hart.** Every `AMO*.W` reads the old word into `rd`
and writes a new one back; the "atomic" part only matters when something else
could interleave between those two steps, and on a single-hart, strictly
in-order core nothing ever does. So each is just `cpu.load` then `cpu.store`
with no locking. The `aq`/`rl` ordering bits (`funct7` bits 26/25) are
consequently decoded as don't-cares — any value in those two bits still
matches the instruction — because ordering is already trivially total here.

**The reservation is the one piece of real state.** `LR.W` records the address
it read on `cpu.reservation`; `SC.W` stores only if that reservation still
names the address it's targeting, then consumes it either way (success or
failure). Any trap, interrupt or `reset()` also clears it — a context switch
between a guest's `LR.W` and its `SC.W` must make the pair fail, exactly as a
real hart's reservation would be lost to another hart's access in between:

```
retry:
    lr.w    t0, (a0)
    addi    t1, t0, 1
    sc.w    t2, t1, (a0)
    bnez    t2, retry        # someone else won the race -- try again
```

**New format, not a new field layout.** An AMO word happens to put `rd`,
`rs1`, `rs2` exactly where R-type does, so the decoder needed no changes at
all. What differs is how it *prints* and *parses*: `rd,rs2,(rs1)` for the
AMOs, `rd,(rs1)` for `LR.W` (there is no `rs2`), never `rd,rs1,rs2`. That got
its own format tag (`AMO_TYPE`) purely so the assembler and disassembler know
which operand order to use — the encoding math is shared with R-type.

## The C extension (RV32C)

Compressed, in [riscv/rv32c.py](riscv/rv32c.py), is architecturally different
from M and A in one important way: **it adds no instructions.** Every one of
its 26 16-bit encodings is, by the specification's own definition, shorthand
for an existing RV32I instruction — `C.ADDI` *is* `ADDI`, `C.MV` *is* `ADD`,
`C.J` *is* `JAL x0,...`. So rather than giving each one its own executor,
`expand()` decodes a compressed halfword straight into the 32-bit word of the
real instruction it stands for, and hands that to the *ordinary* decoder —
the `ADDI` executor runs with no idea it was reached via two bytes instead of
four. `len(RV32IC) == len(RV32I)`: C is an alternate *encoding*, not a new
mnemonic, exactly as the architecture describes it.

```bash
python riscv.py exec prog.elf --isa rv32imac    # auto picks this up from a real ELF already
python riscv.py exec prog.elf --isa auto        # Zaamo+Zalrsc+Zca in .riscv.attributes -> rv32imac
```

```python
from riscv import CPU
from riscv.rv32c import RV32IMAC
cpu = CPU(instruction_set=RV32IMAC)
```

**Mixed-width fetch.** A machine with no C still always fetches exactly 4
bytes, at zero extra cost — `cpu.compressed` is computed once at construction
and the whole fetch path is unchanged for it. One with C fetches 16 bits,
checks `word[1:0]`: `11` means "this is the low half of an ordinary 4-byte
instruction, fetch the other half", anything else means "this is the whole
instruction, expand it". That single reserved pattern is exactly why the
specification requires every *standard* 32-bit encoding to end in `11` — it
is what makes the two streams distinguishable at all. IALIGN also drops from
32 to 16 under C: a jump to a 2-byte-aligned (not 4-byte-aligned) address is
legitimate, and `next_pc` advances by 2 or 4 per instruction, not always 4.

**The scrambled immediates are the real risk.** Several C formats (`C.J`,
`C.JAL`, `C.BEQZ`, `C.BNEZ`, `C.ADDI16SP`, ...) scatter an immediate's bits
across the word in a deliberately non-contiguous order. Rather than a
hand-rolled chain of shifts, each is a declarative list of
`(destination_bit, source_bit)` pairs copied straight from the spec's own
`imm[...]` notation and assembled by one shared `_scramble()` helper — easy
to check by eye against the spec table, and easy to get a single bit wrong in
if written any other way.

**Verified against a real assembler, not just by eye.** `tests/test_rv32c.py`
carries 192 compressed/uncompressed instruction pairs captured once from
`riscv64-linux-gnu-as`/`objdump` (`c.addi s0,20` next to `addi s0,s0,20`) and
frozen as a table, so the test needs no toolchain to run but still catches a
wrong bit position immediately. The four PC-relative formats, which need a
real address to produce a meaningful offset, are checked the same way by
hand-derivation against the same toolchain's output. A GCC
`-march=rv32imac -O2` binary (mutual recursion, a loop, real compressed
`c.addi16sp`/`c.swsp`/`c.lw`/`c.beqz`-shaped control flow) runs to the
correct exit status end to end when the toolchain is available.

**Reserved vs. HINT.** The specification distinguishes encodings that must
trap (`C.ADDI4SPN`/`C.LUI`/`C.ADDI16SP` with an all-zero immediate, an
RV32-illegal shift amount, `C.JR x0`, the RV64-only `C.SUBW`/`C.ADDW` slot)
from ones that are merely HINTs — legal, and defined to do nothing beyond
what the base instruction already does (`C.NOP`, `C.LI x0,...`, a
zero-immediate `C.ADDI`, `C.MV`/`C.ADD` with `rd=x0`). Only the former are
special-cased; every HINT is left to expand and execute completely normally,
because writing to `x0` is already a no-op — "do nothing extra" is exactly
what running it for real gives you for free.

**Known limitation.** The bundled assembler has no *compressor*: it can
disassemble and execute compressed code, but `riscv.py asm` only ever emits
full-width words, so you cannot currently write `c.addi` directly in a `.asm`
source and have it assemble to two bytes. Reading real compressed binaries —
the reason to want C at all — is unaffected.

## CSRs and machine-mode traps

Design and full flow: **[docs/csr-and-traps.md](docs/csr-and-traps.md)**.

Seven instructions — `CSRRW CSRRS CSRRC CSRRWI CSRRSI CSRRCI` and `MRET` —
plus 13 machine CSRs and real architectural trap entry. As with RV32M, this is
an *extension*: `RV32I` is still exactly 40 instructions and `RV32IM` 48.

```bash
python examples/elf/build.py trap_handler
python riscv.py exec examples/elf/build/trap_handler.elf --dump-csrs
python riscv.py exec prog.elf --isa rv32im_zicsr
```

### Supported CSRs

| CSR | Number | Notes |
| --- | --- | --- |
| `mstatus` | 0x300 | MIE, MPIE, MPP; all other fields WPRI and read zero |
| `misa` | 0x301 | MXL=1 plus the machine's actual extension letters; read-only |
| `mie` | 0x304 | MSIE/MTIE/MEIE (nothing sets them yet) |
| `mtvec` | 0x305 | BASE + MODE, Direct and Vectored |
| `mscratch` | 0x340 | scratch for handlers |
| `mepc` | 0x341 | low two bits always zero (IALIGN = 32) |
| `mcause` | 0x342 | bit 31 set for interrupts |
| `mtval` | 0x343 | fault-specific value |
| `mip` | 0x344 | read-only zero — no interrupt sources |
| `mvendorid` `marchid` `mimpid` `mhartid` | 0xF11–0xF14 | read-only identification |

All CSR knowledge lives in [riscv/csr.py](riscv/csr.py): a `CSRDefinition`
carries a write mask (WPRI), an optional `sanitize` hook (WARL) and a read-only
flag taken from the address, so no register needs bespoke code. An
**unimplemented CSR raises an illegal instruction** rather than reading zero.

### Trap entry

Every existing fault now populates architectural state. On a synchronous
exception: `mepc` ← the faulting PC (not PC+4), `mcause` ← the cause, `mtval` ←
the fault value, `mstatus.MPIE` ← `MIE`, `MIE` ← 0, `MPP` ← the interrupted
privilege, and the PC transfers to `mtvec`. The faulting instruction does not
retire.

| Cause | Exception | `mtval` |
| --- | --- | --- |
| 0 | instruction address misaligned | the target address |
| 1 | instruction access fault | the faulting address |
| 2 | illegal instruction | the instruction word |
| 3 | breakpoint (`EBREAK`) | the PC |
| 4 / 6 | load / store address misaligned | the faulting address |
| 5 / 7 | load / store access fault | the faulting address |
| 11 | environment call (`ECALL`) | 0 |

`MRET` is a real state transition, not `pc = mepc`: privilege ← `MPP`,
`MIE` ← `MPIE`, `MPIE` ← 1, then the PC returns to `mepc`.

### When traps vector, and when they reach you

`CPU(trap_handling=...)` is `"auto"` by default: **architectural once the guest
writes `mtvec`**, and host-visible (a Python exception out of `step()`)
otherwise. A machine that never installed a handler has nowhere useful to go, so
its faults still reach you instead of vectoring into address zero — while a
program that installs a handler gets real trap entry. `"architectural"` and
`"host"` force either behaviour.

`ECALL` is the exception: an installed host `ecall_handler` intercepts it before
it becomes a trap, so the host ABI keeps working. Use `--no-host` to let the
guest take `ECALL` as cause 11. That combination — guest handles faults, host
handles exit — is what `examples/elf/trap_handler.S` uses.

### A guest handler, end to end

`examples/elf/trap_handler.S` runs entirely on the emulated CPU, with no Python
callback in the loop:

```asm
        la      t0, trap_handler
        csrw    mtvec, t0               # install the handler
        .word   0                       # illegal instruction -> cause 2
        addi    s5, s5, 0xA             # reached only because MRET came back
...
trap_handler:
        csrr    t2, mcause              # the guest reads its own trap state
        csrr    t3, mepc
        addi    t3, t3, 4               # step over the faulting instruction
        csrw    mepc, t3
        mret                            # ...and returns into the guest
```

It takes three different exceptions (causes 2, 4 and 6), and a test asserts that
removing the `mepc += 4` fix-up makes the fault repeat forever — which is the
point of `mepc` pointing *at* the faulting instruction.

### Deliberately deferred

Software and external interrupts (`mip`/`mie` carry MSIP and MEIP and the
priority order handles them, but nothing asserts those wires — `msip` reads zero
and there is no PLIC), U and S privilege modes, delegation
(`medeleg`/`mideleg`), `WFI`, `SRET`, counters, PMP and virtual memory. See
[docs/csr-and-traps.md](docs/csr-and-traps.md#deliberately-not-implemented).

## Machine timer interrupts

Design and full delivery path:
**[docs/csr-and-traps.md](docs/csr-and-traps.md#the-machine-timer)**.

```bash
python examples/elf/build.py timer_interrupt
python riscv.py exec examples/elf/build/timer_interrupt.elf --dump-csrs
python riscv.py exec prog.elf --timer-ticks 10     # slow the clock down
python riscv.py exec prog.elf --no-timer           # no timer at all
```

### `mtime` and `mtimecmp`

RISC-V puts the machine timer in the memory map, not in CSRs — there is no
`mtimecmp` CSR — so [riscv/timer.py](riscv/timer.py) implements it as a
CLINT-style device at the standard addresses:

| Address | Register | Width |
| --- | --- | --- |
| `0x02004000` | `mtimecmp` | 64-bit |
| `0x0200BFF8` | `mtime` | 64-bit |

Both are genuinely 64 bits and are never truncated to 32. An RV32 guest reads
and writes them as halves, and `examples/elf/timer_interrupt.S` uses the correct
idioms for both: re-reading the high word to detect a carry across a 64-bit
read, and writing `mtimecmp`'s low half to all-ones first so no intermediate
value briefly looks like an expired deadline.

### MTIP, MTIE, MIE

`MTIP` is a **wire, not a register**: the timer asserts it exactly while
`mtime >= mtimecmp` (a full 64-bit comparison), `mip` reports it live, and guest
writes to `mip` are dropped. The only way to resolve a timer interrupt is to
move `mtimecmp` past `mtime`.

An interrupt is delivered only when all three hold:

| Condition | Where |
| --- | --- |
| pending | `mip.MTIP` — the timer's wire |
| enabled | `mie.MTIE` — the per-source enable |
| permitted | `mstatus.MIE` — the global enable |

A handler that returns without resolving MTIP is re-entered immediately, because
`MRET` restores `mstatus.MIE` while the wire is still asserted. That is correct
behaviour, and a test asserts it.

### Delivery

Interrupts are taken **between instructions** — the check runs before anything
is fetched, so no instruction is ever interrupted part-way through. Delivery
goes through exactly the same `CSRFile.enter_trap` that exceptions use; the only
differences are that `mcause` gets bit 31 set, `mtval` is 0, and `mepc` is the
instruction that has **not** run yet. A timer handler therefore must *not*
advance `mepc` — unlike an exception handler, which must.

| `mtvec` mode | Timer interrupt (cause 7) | Synchronous exception |
| --- | --- | --- |
| Direct | BASE | BASE |
| Vectored | BASE + 4×7 | **BASE** |

Vectored mode offsets *interrupts only*; getting that backwards is a classic
bug, so it is tested in both directions and covered by a mutation.

### Deterministic time

`mtime` never consults a wall clock. It advances by `ticks_per_step` for every
CPU step, so a program fires the timer at the same instruction on every machine
and every run — a test runs the same program three times and asserts identical
register state, `mtime` and interrupt count.

```python
cpu = CPU(instruction_set=RV32IM_ZICSR, timer=MachineTimer(ticks_per_step=0))
cpu.advance_time(1000)        # drive time explicitly
```

`riscv.py exec` attaches a timer by default; the Python `CPU` does not, so a
machine without one maps no device and lets no time pass.

### The guest program

`examples/elf/timer_interrupt.S` does the whole loop on the emulated CPU, with
a **vectored** table so it proves both halves of the vectoring rule at once:

```asm
        la      t0, vector_table
        ori     t0, t0, 1               # MODE = Vectored
        csrw    mtvec, t0
        .word   0x00000000              # an exception -> BASE
        jal     ra, rearm_timer         # mtimecmp = mtime + 60
        csrs    mie, MIE_MTIE
        csrs    mstatus, MSTATUS_MIE    # ...now the timer can fire
...
timer_handler:                          # entered at BASE + 4*7
        csrr    s2, mcause              # 0x80000007
        csrr    s4, mip                 # MTIP asserted
        jal     ra, rearm_timer         # resolve it
        mret                            # mepc is untouched: resume, do not skip
```

Built with GNU `as`/`ld` it takes three real timer interrupts plus one
exception, and exits 0.

## Extending it

Adding an extension does not require touching the CPU, decoder or any existing
instruction — build a new instruction set and register into it. This is exactly
how [riscv/rv32m.py](riscv/rv32m.py) adds the M extension; the same shape works
for A, F or D:

```python
from riscv import CPU, encode
from riscv.instructions import RV32I
from riscv.isa import R_TYPE

RV32IM = RV32I.copy("RV32IM")

@RV32IM.define("MUL", match=0x02000033, mask=0xFE00707F, fmt=R_TYPE, extension="RV32M")
def exec_mul(cpu, inst):
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) * cpu.get_reg(inst.rs2)) & 0xFFFFFFFF)

cpu = CPU(instruction_set=RV32IM)      # decoder, encoder, assembler and
                                       # disassembler all pick MUL up for free
```

(That exact snippet is a test in `tests/test_encoder.py`, including a check that
the resulting encoding matches the RV32M specification. Register the result with
`register_isa("rv32im", RV32IM)` and `--isa rv32im` picks it up.)

Registering overlapping encodings raises immediately, so a mistake in a new
extension's `match`/`mask` is caught at import time rather than at run time.

**Deliberately not implemented** (each is a natural next step, none requires
restructuring): RV32A/F/D, U/S privilege modes, delegation, software and
external interrupt sources (`msip` reads zero and there is no PLIC), PMP,
virtual memory, dynamic linking and relocation processing (`ET_DYN`/`PT_DYNAMIC`
files are rejected rather than silently mis-run), and an OS interface beyond the
small ABI in `riscv/host.py`. The seams for them are: the ISA table (new
instructions), `CSRFile.add_interrupt_source()` (new interrupt sources),
`Memory.map_device()` (new memory-mapped devices), and `riscv/elf.py`'s
`build_image` (relocations).

## Assembly syntax

The built-in assembler accepts the usual GNU-flavoured syntax: `#`, `//` and
`/* ... */` comments, labels, ABI or `xN` register names, `offset(base)` memory
operands, decimal/hex/binary/octal/character literals and arithmetic
expressions. Block comments are what let the same `.S` file assemble here and
under GNU `as`.

* **Sections**: `.text .data .bss .section` — `.bss` reserves space with no file
  bytes, which is what becomes an ELF segment whose `p_memsz` exceeds `p_filesz`
* **Directives**: `.word .half .short .byte .string .asciz .ascii .space .zero
  .align .balign .p2align .equ .set .org .globl .global`, plus
  `.type .size .file .ident .option .attribute` accepted and ignored
* **Pseudo-instructions**: `nop mv not neg seqz snez sltz sgtz beqz bnez blez
  bgez bltz bgtz bgt ble bgtu bleu j jr ret jal jalr li la call tail`
* **CSR pseudo-instructions**: `csrr csrw csrs csrc csrwi csrsi csrci`, and CSRs
  may be named (`mtvec`) or numbered (`0x305`)
* **One deliberate deviation**: in a branch or jump, a *symbolic* target is an
  absolute address while a *purely numeric* one is a PC-relative offset
  (`beq x1, x2, -8` jumps back two instructions). This makes assembly and
  disassembly exact inverses, which a test enforces.

## Host ABI (`ECALL`)

Fully specified in **[docs/abi.md](docs/abi.md)**. The short version: `a7`
selects the call, `a0`–`a2` carry arguments, `a0` returns.

| `a7` | Call | Behaviour |
| --- | --- | --- |
| 1 | `print_int` | print `a0` as a signed decimal |
| 2 / 11 | `print_char` | print `a0` as a character |
| 4 | `print_string` | print the NUL-terminated string at `a0` |
| 10 | `exit` | halt with status 0 |
| 63 | `read` | unsupported; returns −1 |
| 64 | `write` | `write(fd=a0, buf=a1, len=a2)` |
| 93 | `exit` | halt with status `a0` |

```asm
    li  a0, 0
    li  a7, 93          # 93 = exit
    ecall
```

Numbers 64 and 93 match Linux/RISC-V; 1/2/4/10/11 match RARS. An unknown
syscall halts with status −1 rather than being silently ignored.

The host layer is **optional and outside the CPU**: with no `ecall_handler`,
`ECALL` is simply exception cause 11, which is what `--no-host` gives you.

### Startup state under `exec`

| Item | Value |
| --- | --- |
| PC | the ELF's `e_entry` |
| `sp` | top of the stack region, 16-byte aligned |
| `gp` | **0** — not initialised (see [docs/abi.md](docs/abi.md)) |
| Stack | 64 KiB by default, one page above the image; `--stack`/`--stack-size` override |
| Permissions | enforced from the `PT_LOAD` flags unless `--no-protect` |
| Timer | attached, CLINT at `0x02000000`, one tick per step (`--no-timer`, `--timer-ticks`) |

## Performance

Roughly **1 million instructions/second** on CPython 3.14 on an otherwise idle
machine — enough that a 750k-instruction loop finishes in under a second.
Clarity was prioritised over speed, but the decode cache means decoding happens
once per distinct instruction word rather than once per execution. Treat the
absolute figure as an order of magnitude: it moves with the CPython version and
with whatever else the machine is doing.

The optional layers each cost roughly **10–15%** when switched on, and nothing
when they are off:

| Layer | Cost when enabled | Cost when disabled |
| --- | --- | --- |
| Memory permissions | a region lookup per fetch/load/store | one `is not None` check |
| Machine timer | a tick per step, plus a device check per memory access | nothing — no device is mapped |

Loading is never the bottleneck: parsing an ELF and placing it in memory takes a
fraction of a millisecond, so the CPU dominates by orders of magnitude. Segment
data is handed to memory as a `memoryview` rather than copied, and `.bss` is
zeroed only in pages that already exist — a 64 MiB `.bss` allocates nothing.

## Project layout

```
riscv.py                  CLI entry point (python riscv.py --self-test)
riscv/                    the emulator package (see Architecture)
tests/                    572 tests + the standalone runner
examples/                 .asm programs, their .hex dumps, and the generator
examples/elf/             .S programs that build with or without a toolchain
docs/abi.md               the host ECALL ABI, specified
docs/csr-and-traps.md     CSR storage, trap flow and the machine timer
pytest.ini                pytest configuration
```
