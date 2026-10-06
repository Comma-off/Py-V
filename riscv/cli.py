"""Command-line front end.

::

    python riscv.py run examples/loops.asm --trace
    python riscv.py exec program.elf --trace --max-instructions 100000
    python riscv.py elf program.elf
    python riscv.py disasm program.elf --symbols
    python riscv.py asm examples/hello.asm -o hello.elf --format elf
    python riscv.py --self-test

Accepted program formats: ELF32 little-endian RISC-V executables, ``.asm``/
``.s`` (assembled by the built-in assembler), ``.bin`` (raw little-endian
machine code) and ``.hex`` (one hex word per line, ``#`` comments allowed).

``run`` and ``exec`` are the same machine; ``exec`` defaults to ELF conventions
(permission enforcement, a set-up stack, an instruction budget) while ``run``
keeps the flat-image behaviour it has always had.
"""

import argparse
import sys

from . import __version__
from .assembler import assemble_file
from .cpu import CPU
from .disasm import disassemble_bytes, safe_disassemble
from .elf import ElfError, is_elf, load_elf, parse_elf
from .isa import available_isas, get_isa
from .exceptions import AssemblerError, EmulatorError, RiscvException
from .host import HostEnvironment
from .memory import DEFAULT_SIZE


# ---------------------------------------------------------------------------
# Program loading
# ---------------------------------------------------------------------------
def _parse_int(text):
    return int(text, 0)


def load_program_file(path, text_base=0, instruction_set=None):
    """Return ``(image_bytes, entry_address, program_or_None)`` for a file.

    ELF files are handled by :func:`command_exec` instead; this is the flat
    "load a blob at an address" path used by ``run``, ``asm`` and ``dis``.
    """
    lower = str(path).lower()
    if lower.endswith((".asm", ".s")):
        program = assemble_file(path, text_base=text_base, instruction_set=instruction_set)
        return program.image, program.entry, program
    if lower.endswith(".hex"):
        words = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    words.extend(int(token, 16) for token in line.split())
        blob = b"".join((word & 0xFFFFFFFF).to_bytes(4, "little") for word in words)
        return blob, text_base, None
    with open(path, "rb") as handle:
        return handle.read(), text_base, None


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def command_run(args) -> int:
    if is_elf(args.program):
        print(
            f"note: {args.program} is an ELF file; running it with ELF conventions "
            "(use 'riscv.py exec' directly to set --stack / --max-instructions)",
            file=sys.stderr,
        )
        args.max_instructions = args.max_steps
        args.stack = None
        args.stack_size = DEFAULT_STACK_SIZE
        args.no_protect = False
        args.verbose = False
        return command_exec(args)

    try:
        instruction_set, _isa_name = resolve_instruction_set(args.isa)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    try:
        image, entry, program = load_program_file(
            args.program, args.base, instruction_set=instruction_set
        )
    except AssemblerError as exc:
        print(f"error: assembly failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read {args.program}: {exc.strerror}", file=sys.stderr)
        return 2

    host = None if args.no_host else HostEnvironment()
    cpu = CPU(
        memory_size=args.memory_size,
        trace=args.trace or args.trace_regs,
        trace_registers=args.trace_regs,
        trace_stream=sys.stdout,
        allow_misaligned=args.allow_misaligned,
        ecall_handler=host,
        instruction_set=instruction_set,
        timer=None if args.no_timer else _make_timer(args),
    )
    cpu.load_program(image, addr=args.base)
    cpu.set_pc(args.pc if args.pc is not None else entry)
    for address in args.breakpoint or []:
        cpu.breakpoints.add(address)
    return _run_loaded(cpu, args, limit=args.max_steps)


def command_asm(args) -> int:
    try:
        instruction_set, _isa_name = resolve_instruction_set(args.isa)
        program = assemble_file(
            args.source, text_base=args.base, instruction_set=instruction_set
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    except AssemblerError as exc:
        print(f"assembly error: {exc}", file=sys.stderr)
        return 2
    if args.listing:
        print(program.format_listing())
        if program.symbols:
            print("\nsymbols:")
            for name, value in sorted(program.symbols.items(), key=lambda item: item[1]):
                print(f"  0x{value:08X}  {name}")
    if args.output:
        if args.format == "elf" or (args.format == "auto" and str(args.output).endswith(".elf")):
            from .elfwriter import program_to_elf

            blob = program_to_elf(program)
            with open(args.output, "wb") as handle:
                handle.write(blob)
            if not args.listing:
                print(
                    f"wrote {len(blob)} bytes of ELF32 RISC-V to {args.output} "
                    f"(entry 0x{program.entry:08X})",
                    file=sys.stderr,
                )
            return 0
        if args.hex:
            with open(args.output, "w", encoding="utf-8") as handle:
                for address, word, _line_no, text in program.listing:
                    handle.write(f"{word:08X}  # 0x{address:08X}  {text.strip()}\n")
        else:
            with open(args.output, "wb") as handle:
                handle.write(program.image)
        if not args.listing:
            print(f"wrote {len(program.image)} bytes to {args.output}", file=sys.stderr)
    elif not args.listing:
        for _address, word, _line_no, _text in program.listing:
            print(f"{word:08X}")
    return 0


def command_dis(args) -> int:
    try:
        image, _entry, _program = load_program_file(args.program, args.base)
    except OSError as exc:
        print(f"cannot read {args.program}: {exc}", file=sys.stderr)
        return 2
    for offset in range(0, len(image) - 3, 4):
        word = int.from_bytes(image[offset:offset + 4], "little")
        address = args.base + offset
        print(f"0x{address:08X}:  {word:08X}  {safe_disassemble(word, pc=address, abi=args.abi)}")
    return 0


#: Where the stack goes when ``exec`` is not told otherwise: just above the
#: highest address the ELF image occupies, rounded up to a page.  Deterministic,
#: always inside the address space, and never overlapping a loaded segment.
DEFAULT_STACK_SIZE = 0x10000  # 64 KiB


def stack_layout(image, stack_top=None, stack_size=DEFAULT_STACK_SIZE):
    """Return ``(stack_top, stack_size)`` for an image, honouring an override."""
    if stack_top is None:
        image_end = (image.max_address + 0xFFF) & ~0xFFF
        stack_top = image_end + stack_size
    return stack_top & ~0xF, stack_size


#: What ``--isa auto`` falls back to when nothing declares an architecture.
#: This is the most capable set the emulator implements, and each is a strict
#: superset of the last, so defaulting to it never changes how a base-ISA
#: program behaves -- it only means a program using MUL/DIV or a CSR runs
#: instead of trapping.  Pass ``--isa rv32i`` for a strict base-ISA machine.
DEFAULT_ISA = "rv32im_zicsr"


def resolve_instruction_set(name, image=None, verbose=False):
    """Pick an instruction set from ``--isa`` and, for ELF files, the image.

    ``auto`` uses what the ELF's ``.riscv.attributes`` says it was built for,
    which is how an ``rv32i``-only binary gets executed on a strict RV32I
    machine -- if it contains a MUL, that is a toolchain bug and should trap.
    """
    if name and name != "auto":
        return get_isa(name), name
    if image is not None and getattr(image, "isa_name", None):
        if verbose:
            print(
                f"note: {image.arch_string} -> running as {image.isa_name}",
                file=sys.stderr,
            )
        return get_isa(image.isa_name), image.isa_name
    return get_isa(DEFAULT_ISA), DEFAULT_ISA


def _make_timer(args):
    """Build the machine timer a command asked for."""
    from .timer import MachineTimer

    return MachineTimer(ticks_per_step=args.timer_ticks)


def command_elf(args) -> int:
    """Print an ELF file's header and program headers."""
    try:
        elf = parse_elf(args.program, require_executable=False)
    except ElfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(elf.describe())
    if args.symbols:
        symbols = sorted(elf.symbols, key=lambda symbol: symbol.value)
        print(f"\nSymbols: {len(symbols)}")
        for symbol in symbols:
            kind = "FUNC" if symbol.is_function else "    "
            print(f"  0x{symbol.value:08X}  {symbol.size:>6}  {kind}  {symbol.name}")
    if args.sections:
        try:
            sections = elf.sections
        except ElfError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"\nSections: {len(sections)}")
        for section in sections:
            print(
                f"  [{section.index:>2}] {section.name:<20} "
                f"addr=0x{section.addr:08X}  size=0x{section.size:06X}"
            )
    return 0


def command_exec(args) -> int:
    """Load an ELF32 RISC-V executable and run it."""
    try:
        image = load_elf(args.program)
    except ElfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stack_top, stack_size = stack_layout(image, args.stack, args.stack_size)
    needed = max(image.max_address, stack_top)
    memory_size = args.memory_size
    if memory_size < needed:
        memory_size = (needed + 0xFFFF) & ~0xFFFF  # grow to fit the program

    try:
        instruction_set, isa_name = resolve_instruction_set(
            args.isa, image, verbose=args.verbose
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2

    host = None if args.no_host else HostEnvironment()
    timer = None
    if not args.no_timer:
        from .timer import MachineTimer

        timer = MachineTimer(ticks_per_step=args.timer_ticks)
    cpu = CPU(
        memory_size=memory_size,
        trace=args.trace or args.trace_regs,
        trace_registers=args.trace_regs,
        trace_stream=sys.stdout,
        allow_misaligned=args.allow_misaligned,
        ecall_handler=host,
        instruction_set=instruction_set,
        timer=timer,
    )
    cpu.load_image(image, enforce_permissions=not args.no_protect)
    cpu.setup_stack(stack_top, stack_size)
    for address in args.breakpoint or []:
        cpu.breakpoints.add(address)

    if args.verbose:
        print(
            f"loaded {len(image.segments)} segment(s), entry 0x{image.entry:08X}, "
            f"sp 0x{cpu.get_reg(2):08X}, {memory_size >> 10} KiB of RAM, isa {isa_name}",
            file=sys.stderr,
        )

    return _run_loaded(cpu, args, limit=args.max_instructions)


def _run_loaded(cpu, args, limit) -> int:
    """Shared execution/reporting tail for ``run`` and ``exec``."""
    status = 0
    try:
        executed = cpu.run(limit)
    except RiscvException as exc:
        executed = cpu.instret
        print(f"\nerror: {exc}", file=sys.stderr)
        try:
            word = cpu.read_u32(cpu.pc)
            print(
                f"  faulting instruction: {safe_disassemble(word, pc=cpu.pc)}",
                file=sys.stderr,
            )
        except EmulatorError:
            pass
        status = 3
    except EmulatorError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 4

    if not args.quiet:
        summary = (
            f"\n{executed} instruction(s) executed; pc=0x{cpu.pc:08X}; halted={cpu.halted}"
        )
        if cpu.halted:
            summary += f"; exit code {cpu.halt_code}"
        print(summary, file=sys.stderr)
        if not cpu.halted and status == 0 and executed == limit:
            print(
                f"note: stopped at the instruction limit ({limit}); "
                "raise it with --max-instructions",
                file=sys.stderr,
            )
    if args.dump_regs or getattr(args, "dump_csrs", False):
        print(cpu.dump(csrs=getattr(args, "dump_csrs", False)))
    if args.dump_mem:
        address, length = args.dump_mem
        print(cpu.memory.hexdump(address, length))
    if status == 0 and cpu.halted and cpu.halt_code:
        status = 1
    return status


def _elf_code_regions(elf, whole_segments=False):
    """Byte ranges worth disassembling, as ``(label, address, data)``.

    Executable *sections* are preferred when the file has a section table: a
    linker typically maps the ELF header and read-only data into the same R-X
    segment as the code, and disassembling those as instructions is noise.
    Section headers are tooling metadata, so using them here is fine -- unlike
    execution, which only ever consults program headers.  Files with no section
    table (or ``--all``) fall back to whole executable segments.
    """
    from .elf import SHT_NOBITS

    regions = []
    if not whole_segments:
        try:
            sections = elf.sections
        except ElfError:
            sections = []
        for section in sections:
            if not section.is_executable or section.type == SHT_NOBITS or not section.size:
                continue
            if section.offset + section.size > len(elf.data):
                continue
            regions.append(
                (section.name or f"section{section.index}", section.addr,
                 bytes(elf.data[section.offset:section.offset + section.size]))
            )
    if not regions:
        for header in elf.loadable_segments():
            if header.flags & 0x1:  # PF_X
                regions.append(
                    (f"segment{header.index}", header.vaddr,
                     bytes(elf.data[header.offset:header.offset + header.filesz]))
                )
    return sorted(regions, key=lambda region: region[1])


def command_disasm(args) -> int:
    """Disassemble an ELF's executable code, or a flat binary."""
    path = args.program
    if is_elf(path):
        try:
            elf = parse_elf(path, require_executable=False)
            regions = _elf_code_regions(elf, whole_segments=args.all)
            instruction_set, _isa = resolve_instruction_set(args.isa, elf)
        except ElfError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return 2
        if not regions:
            print("error: no executable code found to disassemble", file=sys.stderr)
            return 2
        labels = {}
        if args.symbols:
            for symbol in elf.symbols:
                # Skip RISC-V mapping symbols ($x/$d), which mark instruction
                # vs data ranges and would otherwise shadow real names.
                if not symbol.name or symbol.name.startswith("$"):
                    continue
                previous = labels.get(symbol.value)
                if previous is None or (symbol.is_function and not previous[1]):
                    labels[symbol.value] = (symbol.name, symbol.is_function)
        first = True
        for name, base, data in regions:
            if not first:
                print()
            first = False
            print(f"Disassembly of {name}:")
            for address, size, raw, text in disassemble_bytes(
                data, base=base, instruction_set=instruction_set, abi=args.abi
            ):
                if address in labels:
                    print(f"\n{address:08x} <{labels[address][0]}>:")
                width = size * 2
                print(f"{address:08x}:  {raw:0{width}X}  {text}")
        return 0

    try:
        instruction_set, _isa = resolve_instruction_set(args.isa)
        image_bytes, _entry, _program = load_program_file(
            path, args.base, instruction_set=instruction_set
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    except AssemblerError as exc:
        print(f"error: assembly failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read {path}: {exc.strerror}", file=sys.stderr)
        return 2
    for address, size, raw, text in disassemble_bytes(
        image_bytes, base=args.base, instruction_set=instruction_set, abi=args.abi
    ):
        width = size * 2
        print(f"{address:08x}:  {raw:0{width}X}  {text}")
    return 0


def command_selftest(args) -> int:
    from .selftest import run_self_test

    print("=" * 64)
    print("Built-in validation suite")
    print("=" * 64)
    result = run_self_test(verbose=not args.quiet)
    failures = result.failed

    if not args.quick:
        try:
            from tests.run_tests import run_all  # noqa: PLC0415
        except ImportError:
            print("\n(tests/ package not found -- ran the internal suite only)")
        else:
            print()
            print("=" * 64)
            print("Full test suite (tests/)")
            print("=" * 64)
            failures += run_all(verbose=not args.quiet)

    print()
    if failures:
        print(f"SELF-TEST FAILED ({failures} failure(s))")
    else:
        print("SELF-TEST PASSED")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _memory_range(text):
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected ADDR:LENGTH, e.g. 0x400:64")
    return int(parts[0], 0), int(parts[1], 0)


def _add_isa_option(parser):
    """Give a subcommand the shared ``--isa`` flag."""
    parser.add_argument(
        "--isa",
        default="auto",
        metavar="NAME",
        help=(
            "instruction set: " + ", ".join(available_isas())
            + ", or 'auto' (default: what the ELF declares, else " + DEFAULT_ISA + ")"
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riscv",
        description="A self-contained RV32I emulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("::", 1)[1].strip() if "::" in __doc__ else None,
    )
    parser.add_argument("--version", action="version", version=f"riscv-emulator {__version__}")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in validation suite and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="execute a program")
    run.add_argument("program", help=".asm/.s, .bin or .hex file")
    run.add_argument("--trace", action="store_true", help="print one line per instruction")
    run.add_argument(
        "--trace-regs",
        action="store_true",
        help="trace, and show the register written by each instruction",
    )
    run.add_argument("--max-steps", type=_parse_int, default=1_000_000, help="instruction budget")
    run.add_argument("--base", type=_parse_int, default=0, help="load address (default 0)")
    run.add_argument("--pc", type=_parse_int, default=None, help="override the entry point")
    run.add_argument("--memory-size", type=_parse_int, default=DEFAULT_SIZE, help="bytes of RAM")
    run.add_argument("--dump-regs", action="store_true", help="print the register file at exit")
    run.add_argument("--dump-csrs", action="store_true", help="print the CSRs at exit")
    run.add_argument("--dump-mem", type=_memory_range, metavar="ADDR:LEN", help="hexdump at exit")
    run.add_argument("--breakpoint", type=_parse_int, action="append", metavar="ADDR", help="stop at ADDR")
    run.add_argument("--allow-misaligned", action="store_true", help="do not trap misaligned accesses")
    run.add_argument("--no-host", action="store_true", help="leave ECALL as an architectural trap")
    run.add_argument("--no-timer", action="store_true", help="do not attach the machine timer")
    run.add_argument(
        "--timer-ticks", type=_parse_int, default=1, metavar="N",
        help="mtime ticks per instruction step (default 1; 0 freezes time)",
    )
    run.add_argument("--quiet", action="store_true", help="suppress the summary line")
    _add_isa_option(run)
    run.set_defaults(func=command_run)

    for name in ("asm", "assemble"):
        asm = subparsers.add_parser(
            name, help="assemble a .asm file" if name == "asm" else argparse.SUPPRESS
        )
        asm.add_argument("source")
        asm.add_argument("-o", "--output", help="output file (default: hex words on stdout)")
        asm.add_argument("--hex", action="store_true", help="write hex text instead of raw binary")
        asm.add_argument(
            "--format",
            choices=("auto", "bin", "elf"),
            default="auto",
            help="output format; 'auto' picks ELF for a .elf output name",
        )
        asm.add_argument("--base", type=_parse_int, default=0, help="text base address")
        asm.add_argument("--listing", action="store_true", help="print an address/word/source listing")
        _add_isa_option(asm)
        asm.set_defaults(func=command_asm)

    exec_parser = subparsers.add_parser(
        "exec", help="load and run an ELF32 RISC-V executable"
    )
    exec_parser.add_argument("program", help="ELF32 little-endian RISC-V executable")
    exec_parser.add_argument("--trace", action="store_true", help="print one line per instruction")
    exec_parser.add_argument(
        "--trace-regs", action="store_true", help="trace, and show the register written"
    )
    exec_parser.add_argument(
        "--max-instructions", type=_parse_int, default=1_000_000,
        help="instruction budget; prevents an infinite loop from hanging (default 1000000)",
    )
    exec_parser.add_argument(
        "--stack", type=_parse_int, default=None,
        help="stack top address (default: one page above the image, plus --stack-size)",
    )
    exec_parser.add_argument(
        "--stack-size", type=_parse_int, default=DEFAULT_STACK_SIZE,
        help=f"stack size in bytes (default 0x{DEFAULT_STACK_SIZE:X})",
    )
    exec_parser.add_argument(
        "--memory-size", type=_parse_int, default=DEFAULT_SIZE,
        help="bytes of RAM; grown automatically if the image needs more",
    )
    exec_parser.add_argument(
        "--no-protect", action="store_true",
        help="do not enforce the ELF segments' R/W/X permissions",
    )
    exec_parser.add_argument("--dump-regs", action="store_true", help="print registers at exit")
    exec_parser.add_argument("--dump-csrs", action="store_true", help="print the CSRs at exit")
    exec_parser.add_argument("--dump-mem", type=_memory_range, metavar="ADDR:LEN", help="hexdump at exit")
    exec_parser.add_argument("--breakpoint", type=_parse_int, action="append", metavar="ADDR")
    exec_parser.add_argument("--allow-misaligned", action="store_true")
    exec_parser.add_argument("--no-host", action="store_true", help="leave ECALL as a trap")
    exec_parser.add_argument(
        "--no-timer", action="store_true",
        help="do not attach the machine timer (no CLINT mapping, no time)",
    )
    exec_parser.add_argument(
        "--timer-ticks", type=_parse_int, default=1, metavar="N",
        help="mtime ticks per instruction step (default 1; 0 freezes time)",
    )
    exec_parser.add_argument("--verbose", action="store_true", help="report the load layout")
    exec_parser.add_argument("--quiet", action="store_true", help="suppress the summary line")
    _add_isa_option(exec_parser)
    exec_parser.set_defaults(func=command_exec)

    elf_parser = subparsers.add_parser("elf", help="inspect an ELF file's headers")
    elf_parser.add_argument("program")
    elf_parser.add_argument("--symbols", action="store_true", help="also list the symbol table")
    elf_parser.add_argument("--sections", action="store_true", help="also list section headers")
    elf_parser.set_defaults(func=command_elf)

    for name in ("disasm", "dis"):
        dis = subparsers.add_parser(
            name,
            help=(
                "disassemble an ELF's executable segments, or a flat binary"
                if name == "disasm"
                else argparse.SUPPRESS
            ),
        )
        dis.add_argument("program")
        dis.add_argument("--base", type=_parse_int, default=0, help="address of the first word")
        dis.add_argument("--abi", action="store_true", help="use ABI register names (sp, a0, ...)")
        dis.add_argument("--symbols", action="store_true", help="label addresses with ELF symbols")
        dis.add_argument(
            "--all", action="store_true",
            help="disassemble whole executable segments, not just executable sections",
        )
        _add_isa_option(dis)
        dis.set_defaults(func=command_disasm)

    selftest = subparsers.add_parser("selftest", help="run the built-in validation suite")
    selftest.add_argument("--quick", action="store_true", help="internal checks only")
    selftest.add_argument("--quiet", action="store_true", help="summary output only")
    selftest.set_defaults(func=command_selftest)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        args.quick = getattr(args, "quick", False)
        args.quiet = getattr(args, "quiet", False)
        return command_selftest(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)
