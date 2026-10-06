# Py-V host ABI

The emulator implements **RV32I**, the optional **M extension**, and
**Zicsr, machine-mode traps and a machine timer**
([docs/csr-and-traps.md](csr-and-traps.md)).  There is no A/F/D, no U/S
privilege mode, no software or external interrupt source, and no operating
system. A guest program talks to its environment through
exactly one instruction, `ECALL`, and the small calling convention below.

This is deliberately minimal. It is the least an emulator can provide and still
run deterministic bare-metal test programs to completion with a meaningful exit
status — not an attempt to emulate Linux.

## Calling convention

| Register | Role |
| --- | --- |
| `a7` (`x17`) | syscall number — selects the call |
| `a0` (`x10`) | first argument, and the return value |
| `a1` (`x11`) | second argument |
| `a2` (`x12`) | third argument |

All other registers are preserved across an `ECALL`. The PC advances past the
`ECALL` normally, except for calls that halt the machine.

```asm
    li   a0, 0          # exit status
    li   a7, 93         # 93 = exit
    ecall               # does not return
```

## Calls

| `a7` | Name | Arguments | Behaviour |
| --- | --- | --- | --- |
| 1 | `print_int` | `a0` = value | Print `a0` as a signed decimal |
| 2 | `print_char` | `a0` = character | Print the low byte of `a0` |
| 4 | `print_string` | `a0` = address | Print the NUL-terminated string at `a0` |
| 10 | `exit` | — | Halt with status 0 |
| 11 | `put_char` | `a0` = character | Alias of 2 (RARS compatibility) |
| 63 | `read` | — | Not supported; returns −1 in `a0` |
| 64 | `write` | `a0` = fd, `a1` = buffer, `a2` = length | Write `a2` bytes; returns `a2` |
| 93 | `exit` | `a0` = status | Halt with status `a0` (Linux/newlib convention) |

Numbers 64 and 93 match the Linux RISC-V syscall numbering, so a program written
against `write`/`exit` behaves the same here as under `qemu-riscv32`. Numbers 1,
2, 4, 10 and 11 match the RARS/MARS convention, which is what most teaching
material for RISC-V assembly uses.

**An unrecognised syscall number halts the machine with status −1.** It is not
silently ignored, because silently ignoring it would let a broken program appear
to succeed.

## Exit status

`ECALL` with `a7 = 93` halts the machine and records `a0` as the exit status,
interpreted as a *signed* 32-bit value. That status is:

* available as `cpu.halt_code` in the Python API,
* reported by the CLI as `exit code N`,
* mapped to the CLI's own process exit status: `0` if the guest exited 0, `1` if
  the guest exited non-zero.

## Without a host

The host layer is optional and is **not** part of the CPU. A `CPU` built with no
`ecall_handler` treats `ECALL` as what the ISA says it is: an environment-call
exception, `mcause = 11`.

```python
cpu = CPU()                                # no host
cpu.step()                                 # raises EnvironmentCall (cause 11)

cpu = CPU(ecall_handler=HostEnvironment())  # syscalls above
cpu = CPU(ecall_handler=halt_on_ecall)      # any ECALL simply halts
```

The CLI installs `HostEnvironment` for `run` and `exec`; `--no-host` turns it
off, which is the right setting for testing that a program's traps are reported
correctly.

`EBREAK` behaves the same way: it raises `Breakpoint` (`mcause = 3`) unless an
`ebreak_handler` is installed.

## Startup state

When the CLI's `exec` command starts a program:

| Item | Value |
| --- | --- |
| ISA | what the ELF's `.riscv.attributes` declares, else RV32IM (`--isa` overrides) |
| PC | the ELF's `e_entry` |
| `sp` (`x2`) | top of the stack region, 16-byte aligned (RISC-V ABI) |
| `gp` (`x3`) | **0** — not initialised; see below |
| All other registers | 0 |
| Stack region | `[top − size, top)`, default size 64 KiB, readable and writable |
| Stack top (default) | one page above the image's highest address, plus the stack size |
| Segment permissions | enforced from the ELF's `PT_LOAD` flags |

Everything except the loaded segments and the stack region is zero.

### `gp` is not initialised

The emulator does not set up the global pointer, because `__global_pointer$` is
a *linker*-defined symbol and setting `gp` is part of a program's own startup
code — in a normal toolchain, `crt0.o` does it.

This matters in practice: if you link with relaxation enabled, GNU `ld` will
rewrite `la reg, symbol` into a `gp`-relative `addi`, and that will read from
address `0 + offset` if nothing set `gp` up. Either

* set `gp` yourself at `_start`:

  ```asm
      .option push
      .option norelax
      la      gp, __global_pointer$
      .option pop
  ```

* or link with relaxation off (`-mno-relax` when assembling, `--no-relax` when
  linking), which is what `examples/elf/Makefile` does so that the same sources
  also build with the bundled assembler.

Both work. The emulator executes whichever the linker produced.
