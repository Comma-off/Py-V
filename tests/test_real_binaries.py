"""Differential tests against a real RISC-V toolchain.

Everything here is skipped when the toolchain is missing, so the suite passes on
a machine with no cross-compiler.  When one *is* present, these are the tests
that prove the emulator runs software it did not produce itself:

* the same ``.S`` sources assembled by GNU ``as`` and linked by GNU ``ld``,
  executed and checked against the same expected architectural state;
* our decoder checked instruction-by-instruction against ``objdump -d``;
* our ELF parser checked field-by-field against ``readelf``;
* a C program compiled by GCC at -O2, executed end to end.

Set ``RISCV_TOOLPREFIX`` to choose a toolchain (default: try a list of the usual
names).  ``readelf``/``objdump`` from the host binutils are used for ELF
structure checks, since those tools are architecture-independent readers.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.helpers import EmulatorTestCase, build_elf_example

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELF_EXAMPLES = os.path.join(PROJECT_ROOT, "examples", "elf")

#: Toolchain prefixes to try, in order.  Any RV32I-capable toolchain works.
CANDIDATE_PREFIXES = (
    os.environ.get("RISCV_TOOLPREFIX", ""),
    "riscv32-linux-gnu-",
    "riscv64-linux-gnu-",
    "riscv32-unknown-elf-",
    "riscv64-unknown-elf-",
    "riscv-none-elf-",
)
#: Extra directories to look in beyond PATH (a user-local toolchain install).
EXTRA_BIN_DIRS = (os.path.expanduser("~/.local/riscv-toolchain/usr/bin"),)


def _which(name):
    found = shutil.which(name)
    if found:
        return found
    for directory in EXTRA_BIN_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_toolchain():
    """Return ``(as, ld, objdump, gcc_or_None)`` for the first usable toolchain."""
    for prefix in CANDIDATE_PREFIXES:
        if not prefix:
            continue
        assembler = _which(prefix + "as")
        linker = _which(prefix + "ld")
        if assembler and linker:
            return assembler, linker, _which(prefix + "objdump"), _which(prefix + "gcc")
    return None, None, None, None


TOOL_AS, TOOL_LD, TOOL_OBJDUMP, TOOL_GCC = find_toolchain()
HAVE_TOOLCHAIN = bool(TOOL_AS and TOOL_LD)
HAVE_READELF = bool(_which("readelf"))

SKIP_REASON = (
    "no RISC-V toolchain found (looked for riscv32-linux-gnu-as and friends; "
    "set RISCV_TOOLPREFIX to point at one)"
)

#: The expected architectural state of each bundled ELF example.  Shared with
#: tests/test_elf_integration.py's expectations on purpose: the *same* program
#: built two different ways must produce the *same* result.
EXPECTED = {
    "hello": {"exit": 0},
    "arithmetic": {"exit": 42, "regs": {8: 30, 9: 60, 18: 42, 19: 0x0F00, 20: 0xFFF0,
                                        21: 0xF0F0, 22: 1, 23: 0}},
    "branches": {"exit": 0, "regs": {8: 55, 9: 6}},
    "memory": {"exit": 0, "regs": {8: 0x12345678, 9: 0, 18: 0xCAFEBABE, 19: 0xFFFFFFFF,
                                   20: 0xFF, 21: 0x5678, 22: 0xBEEF}},
    "functions": {"exit": 0, "regs": {8: 42, 9: 120, 18: 42, 19: 0x5A5A}},
    "trap": {"trap": 4},
    "multiply": {"exit": 0, "regs": {8: 14}, "arch": "rv32im"},
    "trap_handler": {
        "exit": 0,
        "regs": {8: 3, 9: 2, 18: 4, 19: 6, 20: 0x1001, 21: 0x1AA, 22: 8, 23: 0x1800},
        "arch": "rv32i_zicsr",
    },
    "timer_interrupt": {
        "exit": 0,
        "regs": {8: 3, 9: 1, 18: 0x80000007, 20: 0x80, 21: 0, 22: 0x1880, 23: 2},
        "arch": "rv32i_zicsr",
    },
}


def assemble_and_link(source_path, output_path, text_base=0x10000, relax=False, arch="rv32i"):
    """Build ``source_path`` into an ELF with the external toolchain."""
    object_path = output_path + ".o"
    as_flags = [f"-march={arch}", "-mabi=ilp32"] + ([] if relax else ["-mno-relax"])
    ld_flags = ["-m", "elf32lriscv", f"-Ttext=0x{text_base:X}", "--no-warn-rwx-segments"]
    if not relax:
        ld_flags.append("--no-relax")
    subprocess.run([TOOL_AS, *as_flags, source_path, "-o", object_path],
                   check=True, capture_output=True)
    subprocess.run([TOOL_LD, *ld_flags, object_path, "-o", output_path],
                   check=True, capture_output=True)
    os.unlink(object_path)
    return output_path


@unittest.skipUnless(HAVE_TOOLCHAIN, SKIP_REASON)
class TestToolchainBuiltExamples(EmulatorTestCase):
    """Run the examples as built by GNU as + ld."""

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix="pyv-gnu-")
        cls.built = {}
        for name, expected in EXPECTED.items():
            source = os.path.join(ELF_EXAMPLES, f"{name}.S")
            cls.built[name] = assemble_and_link(
                source, os.path.join(cls.tempdir, f"{name}.elf"),
                arch=expected.get("arch", "rv32i"),
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def check(self, name):
        from tests.test_elf_integration import run_elf

        expected = EXPECTED[name]
        cpu, _host, exception = run_elf(self.built[name])
        if "trap" in expected:
            self.assertIsNotNone(exception, f"{name}: expected a trap")
            self.assertEqual(exception.cause, expected["trap"])
            return cpu
        self.assertIsNone(exception, f"{name}: unexpected trap {exception}")
        self.assertTrue(cpu.halted, f"{name}: did not halt")
        self.assertEqual(cpu.halt_code, expected["exit"], f"{name}: wrong exit status")
        for register, value in expected.get("regs", {}).items():
            self.assertReg(cpu, register, value, f"{name}")
        return cpu

    def test_gnu_hello(self):
        """toolchain: GNU-built hello.S runs and exits 0"""
        self.check("hello")

    def test_gnu_arithmetic(self):
        """toolchain: GNU-built arithmetic.S"""
        self.check("arithmetic")

    def test_gnu_branches(self):
        """toolchain: GNU-built branches.S"""
        self.check("branches")

    def test_gnu_memory(self):
        """toolchain: GNU-built memory.S (.data and .bss from a real linker)"""
        self.check("memory")

    def test_gnu_functions(self):
        """toolchain: GNU-built functions.S"""
        self.check("functions")

    def test_gnu_trap(self):
        """toolchain: GNU-built trap.S takes the expected trap"""
        self.check("trap")

    def test_gnu_multiply(self):
        """toolchain: GNU-built multiply.S exercises all eight RV32M instructions"""
        cpu = self.check("multiply")
        self.assertEqual(cpu.halt_code, 0, "a non-zero exit names the failing check")

    def test_gnu_trap_handler(self):
        """toolchain: a GNU-built guest trap handler takes three traps and returns"""
        cpu = self.check("trap_handler")
        self.assertEqual(cpu.traps_taken, 3, "three architectural traps were taken")
        self.assertTrue(cpu.halted)

    def test_gnu_timer_interrupt(self):
        """toolchain: a GNU-built guest takes three real machine timer interrupts"""
        cpu = self.check("timer_interrupt")
        self.assertEqual(cpu.interrupts_taken, 3, "three interrupts were delivered")
        self.assertReg(cpu, 18, 0x80000007, "mcause: interrupt bit + machine timer")
        self.assertTrue(cpu.halted)

    def test_gnu_timer_binary_uses_vectored_mode(self):
        """toolchain: the GNU-built timer program really used vectored mtvec"""
        cpu = self.check("timer_interrupt")
        self.assertEqual(cpu.csr.mtvec_mode, 1, "mtvec MODE = Vectored")
        self.assertReg(cpu, 9, 1, "the exception entered at BASE")
        self.assertReg(cpu, 8, 3, "and the timer at BASE + 4*7")

    def test_trap_handler_binary_declares_zicsr(self):
        """toolchain: the Zicsr binary records its architecture, and we read it"""
        from riscv.elf import parse_elf

        elf = parse_elf(self.built["trap_handler"])
        self.assertIn("zicsr", elf.arch_string, f"arch was {elf.arch_string!r}")
        self.assertEqual(elf.isa_name, "rv32i_zicsr")

    def test_multiply_binary_declares_rv32im(self):
        """toolchain: the M binary records its architecture, and we read it"""
        from riscv.elf import parse_elf

        elf = parse_elf(self.built["multiply"])
        self.assertIn("m2p0", elf.arch_string, f"arch was {elf.arch_string!r}")
        self.assertEqual(elf.isa_name, "rv32im")
        base = parse_elf(self.built["branches"])
        self.assertEqual(base.isa_name, "rv32i", "a base binary must not claim M")

    def test_both_toolchains_agree(self):
        """toolchain: GNU-built and bundled-assembler builds reach the same state"""
        from tests.test_elf_integration import run_elf  # noqa: PLC0415

        for name in ("arithmetic", "branches", "memory", "functions", "multiply",
                     "trap_handler", "timer_interrupt"):
            gnu_cpu, _host, gnu_exc = run_elf(self.built[name])
            own_cpu, _host2, own_exc = run_elf(build_elf_example(name))
            self.assertIsNone(gnu_exc, name)
            self.assertIsNone(own_exc, name)
            self.assertEqual(gnu_cpu.halt_code, own_cpu.halt_code, name)
            # Compare the *computed* results, not every register: registers
            # holding addresses (what `la` produced) legitimately differ,
            # because GNU ld and the bundled assembler lay .data out at
            # different addresses.  Agreement on the computation is the claim
            # worth testing; agreement on pointer values would only be testing
            # that two linkers made the same layout choice.
            for register, value in EXPECTED[name].get("regs", {}).items():
                self.assertEqual(
                    gnu_cpu.get_reg(register), own_cpu.get_reg(register),
                    f"{name}: x{register} differs between the two builds "
                    f"(GNU 0x{gnu_cpu.get_reg(register):08X} vs "
                    f"bundled 0x{own_cpu.get_reg(register):08X})",
                )
                self.assertEqual(gnu_cpu.get_reg(register), value, f"{name}: x{register}")


@unittest.skipUnless(HAVE_TOOLCHAIN and TOOL_GCC, "no RISC-V GCC available")
class TestGccCompiledC(EmulatorTestCase):
    """Compile C with GCC at -O2 and run the result."""

    SOURCE = r"""
/* Freestanding C: no libc, no runtime, entry straight at _start. */
static int accumulate(int n)
{
    int total = 0;
    for (int i = 1; i <= n; i++)
        total += i;
    return total;
}

static int fib(int n)
{
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}

int global_initialised = 0x1234;
int global_zeroed;

void _start(void)
{
    global_zeroed = accumulate(10);          /* 55 */
    int f = fib(10);                         /* 55 */
    int status = (global_zeroed == 55 && f == 55 && global_initialised == 0x1234)
                 ? 0 : 1;
    register int a0 asm("a0") = status;
    register int a7 asm("a7") = 93;
    asm volatile("ecall" :: "r"(a0), "r"(a7));
    __builtin_unreachable();
}
"""

    def build(self, optimisation, stack_top=0x80000):
        tempdir = tempfile.mkdtemp(prefix="pyv-gcc-")
        self.addCleanup(shutil.rmtree, tempdir, True)
        source = os.path.join(tempdir, "prog.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(self.SOURCE)
        elf = os.path.join(tempdir, "prog.elf")
        result = subprocess.run(
            [
                TOOL_GCC, "-march=rv32i", "-mabi=ilp32", optimisation, "-mno-relax",
                "-nostdlib", "-nostartfiles", "-ffreestanding", "-Wl,-Ttext=0x10000",
                "-Wl,--no-warn-rwx-segments", "-Wl,--no-relax",
                # Fedora's GCC adds a build-id note by default, which collides
                # with an explicit -Ttext in a freestanding link.
                "-Wl,--build-id=none",
                "-e", "_start", source, "-o", elf,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.skipTest(f"GCC could not build the test program: {result.stderr.strip()[:300]}")
        return elf

    def test_gcc_o0(self):
        """toolchain: GCC -O0 C program runs to a correct exit status"""
        from tests.test_elf_integration import run_elf

        cpu, _host, exception = run_elf(self.build("-O0"), max_instructions=2_000_000)
        self.assertIsNone(exception, f"unexpected trap: {exception}")
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 0, "the C program's own self-checks failed")

    def test_gcc_o2(self):
        """toolchain: GCC -O2 C program runs to a correct exit status"""
        from tests.test_elf_integration import run_elf

        cpu, _host, exception = run_elf(self.build("-O2"), max_instructions=2_000_000)
        self.assertIsNone(exception, f"unexpected trap: {exception}")
        self.assertEqual(cpu.halt_code, 0)

    def test_gcc_uses_only_rv32i(self):
        """toolchain: every instruction GCC emitted decodes as RV32I"""
        from riscv.decoder import Decoder
        from riscv.elf import load_elf

        image = load_elf(self.build("-O2"))
        decoder = Decoder()
        count = 0
        for segment in image.executable_segments():
            data = bytes(segment.data)
            # Only disassemble what the symbol table says is code.
            for symbol in image.function_symbols():
                start = symbol.value - segment.vaddr
                if not 0 <= start < len(data):
                    continue
                size = symbol.size or 4
                for offset in range(start, min(start + size, len(data)) - 3, 4):
                    word = int.from_bytes(data[offset:offset + 4], "little")
                    decoder.decode(word)  # raises IllegalInstruction if not RV32I
                    count += 1
        self.assertGreater(count, 10, "expected a meaningful amount of code")


@unittest.skipUnless(HAVE_TOOLCHAIN and TOOL_GCC, "no RISC-V GCC available")
class TestRV32MAgainstGcc(EmulatorTestCase):
    """Differential test of RV32M semantics, with GCC as the oracle.

    The generated C program contains, for each operand pair, both a *runtime*
    computation (through ``volatile`` operands, so GCC must emit a real MUL /
    DIV / REM) and the *expected* value as a compile-time constant that GCC
    folded on the host using its own understanding of the RISC-V ISA.  The
    program exits non-zero at the first disagreement.

    That makes the comparison genuinely independent: if the emulator's division
    truncated the wrong way, GCC's constant would still be right and the
    program would report the mismatch.
    """

    OPERANDS = [
        (7, 6), (42, 6), (-7, 2), (7, -2), (-7, -2), (1, 1), (0, 5), (5, 0),
        (-1, 2), (1, -2), (0x7FFFFFFF, 2), (-2147483648, -1), (-2147483648, 2),
        (123456789, 1000), (-123456789, 1000), (65536, 65536), (-1, -1),
        (0x12345678, 0x1000), (99991, 977), (-99991, 977), (3, 7), (-3, 7),
    ]

    def build_source(self):
        lines = [
            "/* Generated by tests/test_real_binaries.py. */",
            "typedef unsigned int u32; typedef signed int s32;",
            "typedef unsigned long long u64; typedef signed long long s64;",
            "static volatile s32 sa, sb;",
            "static volatile u32 ua, ub;",
            "int fail_at;",
            "void _start(void) {",
            "  int check = 0;",
        ]
        for a, b in self.OPERANDS:
            ua, ub = a & 0xFFFFFFFF, b & 0xFFFFFFFF
            # Expected values, folded by GCC on the host at compile time.
            mul = (a * b) & 0xFFFFFFFF
            mulh = ((a * b) >> 32) & 0xFFFFFFFF
            mulhu = ((ua * ub) >> 32) & 0xFFFFFFFF
            div = "0" if b == 0 else f"((s32){a}) / ((s32){b})"
            rem = "0" if b == 0 else f"((s32){a}) %% ((s32){b})"
            lines += [
                f"  check++; sa = {a}; sb = {b}; ua = {ua}u; ub = {ub}u;",
                f"  if ((u32)(sa * sb) != {mul}u) {{ fail_at = check; goto out; }}",
                f"  if ((u32)(((s64)sa * (s64)sb) >> 32) != {mulh}u) {{ fail_at = check; goto out; }}",
                f"  if ((u32)(((u64)ua * (u64)ub) >> 32) != {mulhu}u) {{ fail_at = check; goto out; }}",
            ]
            if b != 0:
                # GCC folds the right-hand side; the emulator computes the left.
                lines += [
                    f"  if (sa / sb != ({div})) {{ fail_at = check; goto out; }}",
                    f"  if (sa %% sb != ({rem})) {{ fail_at = check; goto out; }}",
                ]
                if ub:
                    lines += [
                        f"  if (ua / ub != {ua // ub}u) {{ fail_at = check; goto out; }}",
                        f"  if (ua %% ub != {ua % ub}u) {{ fail_at = check; goto out; }}",
                    ]
        lines += [
            "out: ;",
            '  register int a0 asm("a0") = fail_at;',
            '  register int a7 asm("a7") = 93;',
            '  asm volatile("ecall" :: "r"(a0), "r"(a7));',
            "  __builtin_unreachable();",
            "}",
        ]
        return "\n".join(lines).replace("%%", "%")

    def build(self, optimisation):
        tempdir = tempfile.mkdtemp(prefix="pyv-rv32m-")
        self.addCleanup(shutil.rmtree, tempdir, True)
        source = os.path.join(tempdir, "m.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(self.build_source())
        elf = os.path.join(tempdir, "m.elf")
        result = subprocess.run(
            [
                TOOL_GCC, "-march=rv32im", "-mabi=ilp32", optimisation, "-mno-relax",
                "-nostdlib", "-nostartfiles", "-ffreestanding",
                "-Wl,-Ttext=0x10000", "-Wl,--no-warn-rwx-segments",
                "-Wl,--no-relax", "-Wl,--build-id=none",
                "-e", "_start", source, "-o", elf,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.skipTest(f"GCC could not build the RV32M test: {result.stderr.strip()[:300]}")
        return elf

    def run_and_check(self, optimisation):
        from tests.test_elf_integration import run_elf

        elf = self.build(optimisation)
        cpu, _host, exception = run_elf(elf, max_instructions=2_000_000)
        self.assertIsNone(exception, f"unexpected trap: {exception}")
        self.assertTrue(cpu.halted, "the program did not reach its exit ECALL")
        self.assertEqual(
            cpu.halt_code, 0,
            f"check #{cpu.halt_code} disagreed with GCC's compile-time constant "
            f"(operands {self.OPERANDS[cpu.halt_code - 1] if 0 < cpu.halt_code <= len(self.OPERANDS) else '?'})",
        )
        return elf

    def test_rv32m_matches_gcc_o0(self):
        """toolchain: RV32M results match GCC's constant folding (-O0)"""
        self.run_and_check("-O0")

    def test_rv32m_matches_gcc_o2(self):
        """toolchain: RV32M results match GCC's constant folding (-O2)"""
        self.run_and_check("-O2")

    def test_the_program_really_uses_rv32m(self):
        """toolchain: the generated program does contain MUL/DIV/REM"""
        from riscv.decoder import Decoder
        from riscv.elf import load_elf
        from riscv.rv32m import RV32IM, RV32M_INSTRUCTIONS

        image = load_elf(self.build("-O0"))
        decoder = Decoder(RV32IM)
        seen = set()
        for segment in image.executable_segments():
            data = bytes(segment.data)
            for offset in range(0, len(data) - 3, 4):
                word = int.from_bytes(data[offset:offset + 4], "little")
                try:
                    name = decoder.decode(word).name
                except Exception:  # noqa: BLE001 - not every word is an instruction
                    continue
                if name in RV32M_INSTRUCTIONS:
                    seen.add(name)
        self.assertTrue(
            {"MUL", "DIV", "REM"} <= seen,
            f"expected GCC to emit MUL/DIV/REM, saw {sorted(seen)}",
        )


@unittest.skipUnless(HAVE_TOOLCHAIN and TOOL_OBJDUMP, "no RISC-V objdump available")
class TestDifferentialDisassembly(EmulatorTestCase):
    """Compare our decoder against objdump, instruction by instruction."""

    OBJDUMP_LINE = re.compile(
        r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)\s*(.*)$"
    )
    #: objdump prints pseudo-instructions; map them back to the real encoding's
    #: mnemonic so the comparison is about the *decode*, not about idiom naming.
    PSEUDO = {
        "li": "ADDI", "mv": "ADDI", "nop": "ADDI", "not": "XORI", "neg": "SUB",
        "seqz": "SLTIU", "snez": "SLTU", "sltz": "SLT", "sgtz": "SLT",
        "beqz": "BEQ", "bnez": "BNE", "blez": "BGE", "bgez": "BGE",
        "bltz": "BLT", "bgtz": "BLT", "bgt": "BLT", "ble": "BGE",
        "bgtu": "BLTU", "bleu": "BGEU", "j": "JAL", "jal": "JAL", "jr": "JALR",
        "jalr": "JALR", "ret": "JALR", "call": "JALR", "tail": "JALR",
        "fence": "FENCE", "unimp": None, "zext.b": "ANDI",
        # Zicsr aliases: objdump prints the pseudo-instruction, we print the
        # canonical encoding.
        "csrr": "CSRRS", "csrw": "CSRRW", "csrs": "CSRRS", "csrc": "CSRRC",
        "csrwi": "CSRRWI", "csrsi": "CSRRSI", "csrci": "CSRRCI",
        "csrrw": "CSRRW", "csrrs": "CSRRS", "csrrc": "CSRRC",
        "csrrwi": "CSRRWI", "csrrsi": "CSRRSI", "csrrci": "CSRRCI",
        "mret": "MRET",
    }

    def test_decoder_matches_objdump(self):
        """toolchain: our decoder agrees with objdump on every instruction"""
        from riscv.decoder import Decoder
        from riscv.exceptions import IllegalInstruction
        from riscv.zicsr import RV32IM_ZICSR

        tempdir = tempfile.mkdtemp(prefix="pyv-diff-")
        self.addCleanup(shutil.rmtree, tempdir, True)
        elves = []
        for name in ("arithmetic", "branches", "memory", "functions", "multiply",
                     "trap_handler", "timer_interrupt"):
            elves.append(
                assemble_and_link(
                    os.path.join(ELF_EXAMPLES, f"{name}.S"),
                    os.path.join(tempdir, f"{name}.elf"),
                    arch=EXPECTED.get(name, {}).get("arch", "rv32i"),
                )
            )

        # Decode against the widest set, so RV32M and Zicsr encodings resolve.
        decoder = Decoder(RV32IM_ZICSR)
        compared = 0
        for elf in elves:
            result = subprocess.run(
                [TOOL_OBJDUMP, "-d", "--no-show-raw-insn" if False else "-d", elf],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.splitlines():
                match = self.OBJDUMP_LINE.match(line)
                if not match:
                    continue
                _address, word_text, mnemonic, _operands = match.groups()
                word = int(word_text, 16)
                if mnemonic == ".word":
                    continue  # data, not an instruction
                expected = self.PSEUDO.get(mnemonic, mnemonic.upper())
                if expected is None:
                    continue
                try:
                    decoded = decoder.decode(word)
                except IllegalInstruction:
                    self.fail(
                        f"objdump decoded 0x{word:08X} as {mnemonic!r} "
                        f"but our decoder called it illegal ({os.path.basename(elf)})"
                    )
                self.assertEqual(
                    decoded.name, expected,
                    f"0x{word:08X}: objdump says {mnemonic!r}, we say {decoded.name}",
                )
                compared += 1
        self.assertGreater(compared, 100, "expected to compare a meaningful number")


@unittest.skipUnless(HAVE_TOOLCHAIN and HAVE_READELF, "no readelf available")
class TestDifferentialElfParsing(EmulatorTestCase):
    """Compare our ELF parser against readelf, field by field."""

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix="pyv-readelf-")
        cls.elf = assemble_and_link(
            os.path.join(ELF_EXAMPLES, "memory.S"), os.path.join(cls.tempdir, "memory.elf")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    def readelf(self, *args):
        return subprocess.run(
            [_which("readelf"), *args, self.elf], capture_output=True, text=True, check=True
        ).stdout

    def test_header_matches_readelf(self):
        """toolchain: our ELF header agrees with readelf"""
        from riscv.elf import parse_elf

        text = self.readelf("-h")
        header = parse_elf(self.elf).header
        self.assertIn("ELF32", text)
        self.assertIn("RISC-V", text)
        entry = int(re.search(r"Entry point address:\s+0x([0-9a-f]+)", text).group(1), 16)
        self.assertEqual(header.entry, entry)
        phnum = int(re.search(r"Number of program headers:\s+(\d+)", text).group(1))
        self.assertEqual(header.phnum, phnum)
        shnum = int(re.search(r"Number of section headers:\s+(\d+)", text).group(1))
        self.assertEqual(header.shnum, shnum)

    def test_program_headers_match_readelf(self):
        """toolchain: our PT_LOAD segments agree with readelf"""
        from riscv.elf import parse_elf

        text = self.readelf("-l", "-W")
        reference = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 7 and fields[0] == "LOAD":
                reference.append(
                    (int(fields[1], 16), int(fields[2], 16), int(fields[4], 16), int(fields[5], 16))
                )
        self.assertTrue(reference, "readelf reported no LOAD segments")
        ours = [
            (header.offset, header.vaddr, header.filesz, header.memsz)
            for header in parse_elf(self.elf).loadable_segments()
        ]
        self.assertEqual(ours, reference)

    def test_bss_is_a_real_memsz_gap(self):
        """toolchain: the linker's .bss shows up as memsz > filesz"""
        from riscv.elf import load_elf

        image = load_elf(self.elf)
        self.assertTrue(
            any(segment.bss_size > 0 for segment in image.segments),
            "memory.S should produce a segment with a zero-filled tail",
        )


class TestToolchainReporting(unittest.TestCase):
    def test_report_toolchain_status(self):
        """toolchain: report whether an external toolchain was found"""
        if HAVE_TOOLCHAIN:
            self.assertTrue(os.path.exists(TOOL_AS))
        else:
            self.skipTest(SKIP_REASON)


if __name__ == "__main__":
    unittest.main()
