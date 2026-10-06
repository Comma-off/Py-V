"""End-to-end tests: ELF file -> loader -> memory -> CPU -> ECALL -> exit code.

These exercise the whole path rather than any single function, which is the
only way to catch a mistake that lives *between* the layers -- an image loaded
at the wrong address, an entry point ignored, a BSS that was never zeroed.
"""

import io
import os
import subprocess
import sys
import unittest

from tests.elf_fixtures import PF_R, PF_X, TEXT_BASE, assemble_elf
from tests.helpers import (
    ELF_EXAMPLE_NAMES,
    CPU,
    EmulatorTestCase,
    build_elf_example,
)

from riscv.elf import load_elf
from riscv.elfwriter import ElfWriter
from riscv.exceptions import RiscvException
from riscv.host import HostEnvironment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELF_EXAMPLES = os.path.join(PROJECT_ROOT, "examples", "elf")
DEFAULT_STACK = 0x00080000


def run_elf(blob_or_path, *, max_instructions=200000, enforce=True, capture=True,
            stack_top=DEFAULT_STACK, memory_size=1 << 20, isa="auto", timer=True):
    """Load and run an ELF exactly as ``riscv.py exec`` does.

    The instruction set is chosen with the CLI's own resolver, and a machine
    timer is attached, so the tests and the shipped command cannot drift apart:
    an ELF that declares ``rv32im`` runs on an RV32IM machine here for exactly
    the same reason it does there, and the CLINT is mapped here for the same
    reason too.

    Returns ``(cpu, host, exception)``; ``exception`` is the trap that stopped
    execution, or ``None`` if the program halted or ran out of budget.
    """
    from riscv.cli import resolve_instruction_set
    from riscv.timer import MachineTimer

    image = load_elf(blob_or_path)
    instruction_set, _isa_name = resolve_instruction_set(isa, image)
    needed = max(image.max_address, stack_top)
    if memory_size < needed:
        memory_size = (needed + 0xFFFF) & ~0xFFFF
    host = HostEnvironment(capture=capture, stdout=io.StringIO())
    cpu = CPU(
        memory_size=memory_size,
        ecall_handler=host,
        instruction_set=instruction_set,
        timer=MachineTimer() if timer else None,
    )
    cpu.load_image(image, enforce_permissions=enforce)
    cpu.setup_stack(stack_top)
    try:
        cpu.run(max_instructions)
        return cpu, host, None
    except RiscvException as exception:
        return cpu, host, exception


class TestExitStatus(EmulatorTestCase):
    def test_exit_zero(self):
        """integration: a program that exits 0"""
        cpu, _host, exception = run_elf(assemble_elf("_start:\n li a0,0\n li a7,93\n ecall\n"))
        self.assertIsNone(exception)
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 0)

    def test_exit_non_zero(self):
        """integration: a program that exits non-zero"""
        for code in (1, 42, 127):
            cpu, _host, _exc = run_elf(
                assemble_elf(f"_start:\n li a0,{code}\n li a7,93\n ecall\n")
            )
            self.assertTrue(cpu.halted)
            self.assertEqual(cpu.halt_code, code)

    def test_exit_code_is_signed(self):
        """integration: a negative exit status round-trips"""
        cpu, _host, _exc = run_elf(assemble_elf("_start:\n li a0,-1\n li a7,93\n ecall\n"))
        self.assertEqual(cpu.halt_code, -1)

    def test_write_syscall(self):
        """integration: the write syscall reaches the host"""
        blob = assemble_elf(
            """
            _start:
                la   a1, message
                li   a2, 5
                li   a0, 1
                li   a7, 64
                ecall
                li   a0, 0
                li   a7, 93
                ecall
                .data
            message:
                .ascii "hello"
            """
        )
        cpu, host, _exc = run_elf(blob)
        self.assertEqual(host.text(), "hello")
        self.assertEqual(cpu.halt_code, 0)


class TestComputation(EmulatorTestCase):
    def test_arithmetic_result(self):
        """integration: an arithmetic result survives the whole path"""
        cpu, _host, _exc = run_elf(
            assemble_elf(
                """
                _start:
                    li   t0, 10
                    li   t1, 20
                    add  s0, t0, t1
                    slli s1, s0, 1
                    mv   a0, s1
                    li   a7, 93
                    ecall
                """
            )
        )
        self.assertReg(cpu, 8, 30)
        self.assertReg(cpu, 9, 60)
        self.assertEqual(cpu.halt_code, 60)

    def test_branches_and_loop(self):
        """integration: a loop and its branches run to completion"""
        cpu, _host, _exc = run_elf(
            assemble_elf(
                """
                _start:
                    li   s0, 0
                    li   t0, 1
                    li   t1, 11
                loop:
                    bge  t0, t1, done
                    add  s0, s0, t0
                    addi t0, t0, 1
                    j    loop
                done:
                    mv   a0, s0
                    li   a7, 93
                    ecall
                """
            )
        )
        self.assertReg(cpu, 8, 55)
        self.assertEqual(cpu.halt_code, 55)

    def test_function_calls_and_stack(self):
        """integration: jal/jalr calls with a stack frame"""
        cpu, _host, _exc = run_elf(
            assemble_elf(
                """
                _start:
                    li   a0, 5
                    jal  ra, square
                    mv   s0, a0
                    li   a0, 0
                    li   a7, 93
                    ecall
                square:
                    addi sp, sp, -16
                    sw   ra, 12(sp)
                    mv   t0, a0
                    mv   t1, a0
                    li   a0, 0
                loop:
                    beqz t1, done
                    add  a0, a0, t0
                    addi t1, t1, -1
                    j    loop
                done:
                    lw   ra, 12(sp)
                    addi sp, sp, 16
                    ret
                """
            )
        )
        self.assertReg(cpu, 8, 25)
        self.assertEqual(cpu.get_reg(2), DEFAULT_STACK, "sp is restored")

    def test_data_and_bss(self):
        """integration: .data is loaded and .bss is zeroed"""
        cpu, _host, _exc = run_elf(
            assemble_elf(
                """
                _start:
                    la   t0, initialised
                    lw   s0, 0(t0)
                    la   t1, zeroed
                    lw   s1, 0(t1)
                    li   t2, 0x5A5A
                    sw   t2, 0(t1)
                    lw   s2, 0(t1)
                    li   a0, 0
                    li   a7, 93
                    ecall
                    .data
                initialised:
                    .word 0xABCD1234
                    .bss
                    .align 2
                zeroed:
                    .space 16
                """
            )
        )
        self.assertReg(cpu, 8, 0xABCD1234, "initialised data came from the file")
        self.assertReg(cpu, 9, 0, "bss was zero-filled")
        self.assertReg(cpu, 18, 0x5A5A, "bss is writable")

    def test_memory_round_trip_at_every_width(self):
        """integration: byte/half/word stores and loads through an ELF"""
        cpu, _host, _exc = run_elf(
            assemble_elf(
                """
                _start:
                    la   t0, buffer
                    li   t1, 0xFF
                    sb   t1, 0(t0)
                    lb   s0, 0(t0)
                    lbu  s1, 0(t0)
                    li   t2, 0x8000
                    sh   t2, 2(t0)
                    lh   s2, 2(t0)
                    lhu  s3, 2(t0)
                    li   a0, 0
                    li   a7, 93
                    ecall
                    .bss
                    .align 2
                buffer:
                    .space 16
                """
            )
        )
        self.assertRegs(cpu, {8: 0xFFFFFFFF, 9: 0xFF, 18: 0xFFFF8000, 19: 0x8000})


class TestEntryAndLayout(EmulatorTestCase):
    def test_entry_point_is_honoured(self):
        """integration: execution starts at e_entry, not at the segment start"""
        blob = assemble_elf(
            """
                li   s0, 1          /* skipped: entry points past this */
            _start:
                li   s1, 2
                li   a0, 0
                li   a7, 93
                ecall
            """
        )
        cpu, _host, _exc = run_elf(blob)
        self.assertReg(cpu, 8, 0, "the instruction before the entry point must not run")
        self.assertReg(cpu, 9, 2)

    def test_program_runs_at_a_high_load_address(self):
        """integration: a program linked high in the address space runs"""
        blob = assemble_elf(
            "_start:\n li a0,7\n li a7,93\n ecall\n", text_base=0x40000000
        )
        cpu, _host, _exc = run_elf(blob, stack_top=0x40100000, memory_size=0x40200000)
        self.assertEqual(cpu.halt_code, 7)

    def test_separate_segments_keep_their_permissions(self):
        """integration: a loaded image's segment permissions are enforced"""
        blob = assemble_elf(
            """
            _start:
                la   t0, value
                li   t1, 1
                sw   t1, 0(t0)      /* .data is writable: fine */
                la   t2, _start
                sw   t1, 0(t2)      /* .text is not: must fault */
                li   a0, 0
                li   a7, 93
                ecall
                .data
            value:
                .word 0
            """
        )
        cpu, _host, exception = run_elf(blob, enforce=True)
        self.assertIsNotNone(exception, "writing to .text must fault")
        self.assertEqual(exception.cause, 7, "store access fault")

    def test_permissions_can_be_disabled(self):
        """integration: --no-protect lets the same program through"""
        blob = assemble_elf(
            """
            _start:
                la   t2, _start
                li   t1, 1
                sw   t1, 0(t2)
                li   a0, 0
                li   a7, 93
                ecall
            """
        )
        cpu, _host, exception = run_elf(blob, enforce=False)
        self.assertIsNone(exception)
        self.assertEqual(cpu.halt_code, 0)


class TestTrapReporting(EmulatorTestCase):
    def test_unhandled_trap_is_reported(self):
        """integration: an unhandled trap stops execution and reports its cause"""
        blob = assemble_elf("_start:\n li t0, 0x1002\n lw t1, 0(t0)\n")
        cpu, _host, exception = run_elf(blob)
        self.assertIsNotNone(exception)
        self.assertEqual(exception.cause, 4, "load address misaligned")
        self.assertFalse(cpu.halted)

    def test_illegal_instruction_in_an_elf(self):
        """integration: an illegal instruction traps with cause 2"""
        writer = ElfWriter(entry=TEXT_BASE)
        writer.add_segment(TEXT_BASE, b"\x00\x00\x00\x00", PF_R | PF_X, align=0x1000)
        cpu, _host, exception = run_elf(writer.build(with_symbols=False))
        self.assertIsNotNone(exception)
        self.assertEqual(exception.cause, 2)

    def test_running_off_the_end_of_a_segment(self):
        """integration: falling off the end of .text faults rather than running wild"""
        blob = assemble_elf("_start:\n addi x1, x0, 1\n")
        cpu, _host, exception = run_elf(blob)
        self.assertIsNotNone(exception)
        self.assertIn(exception.cause, (1, 2), "fetch fault or illegal instruction")

    def test_instruction_limit_stops_an_infinite_loop(self):
        """integration: the instruction budget stops an infinite loop"""
        blob = assemble_elf("_start:\n j _start\n")
        cpu, _host, exception = run_elf(blob, max_instructions=5000)
        self.assertIsNone(exception)
        self.assertFalse(cpu.halted)
        self.assertEqual(cpu.instret, 5000, "stopped exactly at the budget")


class TestBundledElfExamples(EmulatorTestCase):
    """The programs in examples/elf/, built by the bundled assembler."""

    @classmethod
    def setUpClass(cls):
        cls.build_dir = os.path.join(ELF_EXAMPLES, "build")
        cls.paths = {name: build_elf_example(name) for name in ELF_EXAMPLE_NAMES}

    def test_hello(self):
        """elf example: hello.S exits 0"""
        cpu, _host, exception = run_elf(self.paths["hello"])
        self.assertIsNone(exception)
        self.assertEqual(cpu.halt_code, 0)

    def test_arithmetic(self):
        """elf example: arithmetic.S"""
        cpu, _host, _exc = run_elf(self.paths["arithmetic"])
        self.assertEqual(cpu.halt_code, 42)
        self.assertRegs(cpu, {8: 30, 9: 60, 18: 42, 19: 0x0F00, 20: 0xFFF0, 21: 0xF0F0,
                              22: 1, 23: 0})

    def test_branches(self):
        """elf example: branches.S"""
        cpu, _host, _exc = run_elf(self.paths["branches"])
        self.assertEqual(cpu.halt_code, 0)
        self.assertRegs(cpu, {8: 55, 9: 6})

    def test_memory(self):
        """elf example: memory.S (.data, .bss and the stack)"""
        cpu, _host, _exc = run_elf(self.paths["memory"])
        self.assertEqual(cpu.halt_code, 0)
        self.assertRegs(
            cpu,
            {8: 0x12345678, 9: 0, 18: 0xCAFEBABE, 19: 0xFFFFFFFF, 20: 0xFF,
             21: 0x5678, 22: 0xBEEF},
        )

    def test_functions(self):
        """elf example: functions.S (jal, jalr, frames, callee-saved)"""
        cpu, _host, _exc = run_elf(self.paths["functions"])
        self.assertEqual(cpu.halt_code, 0)
        self.assertRegs(cpu, {8: 42, 9: 120, 18: 42, 19: 0x5A5A})

    def test_trap(self):
        """elf example: trap.S takes the trap it advertises"""
        cpu, _host, exception = run_elf(self.paths["trap"])
        self.assertIsNotNone(exception)
        self.assertEqual(exception.cause, 4)
        self.assertFalse(cpu.halted)


class TestCommandLine(EmulatorTestCase):
    """The CLI is part of the deliverable, so exercise it as a subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.elf_path = os.path.join(
            PROJECT_ROOT, "examples", "elf", "build", "arithmetic.elf"
        )
        if not os.path.exists(cls.elf_path):
            build_elf_example("arithmetic")

    def riscv(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "riscv.py"), *args],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        )

    def test_exec_command(self):
        """cli: exec runs an ELF and reports the exit code"""
        result = self.riscv("exec", self.elf_path)
        self.assertEqual(result.returncode, 1, "guest exited 42, so the CLI exits 1")
        self.assertIn("exit code 42", result.stderr)

    def test_exec_trace(self):
        """cli: exec --trace prints one line per instruction"""
        result = self.riscv("exec", self.elf_path, "--trace", "--quiet")
        lines = [line for line in result.stdout.splitlines() if line.startswith("PC=")]
        self.assertGreater(len(lines), 5)
        self.assertRegex(lines[0], r"^PC=0x000100[0-9A-F]{2}  INST=0x[0-9A-F]{8}  ")

    def test_exec_instruction_limit(self):
        """cli: exec --max-instructions bounds an infinite loop"""
        import tempfile

        from tests.elf_fixtures import assemble_elf as build

        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as handle:
            handle.write(build("_start:\n j _start\n"))
            path = handle.name
        try:
            result = self.riscv("exec", path, "--max-instructions", "1000")
            self.assertEqual(result.returncode, 0)
            self.assertIn("1000 instruction(s) executed", result.stderr)
            self.assertIn("instruction limit", result.stderr)
        finally:
            os.unlink(path)

    def test_elf_command(self):
        """cli: elf prints headers and loadable segments"""
        result = self.riscv("elf", self.elf_path)
        self.assertEqual(result.returncode, 0)
        self.assertIn("ELF32 RISC-V", result.stdout)
        self.assertIn("Type: EXEC", result.stdout)
        self.assertIn("Entry: 0x00010000", result.stdout)
        self.assertIn("LOAD", result.stdout)
        self.assertIn("R-X", result.stdout)

    def test_elf_command_symbols(self):
        """cli: elf --symbols lists the symbol table"""
        result = self.riscv("elf", self.elf_path, "--symbols")
        self.assertIn("_start", result.stdout)

    def test_disasm_command(self):
        """cli: disasm shows addresses, words and decoded instructions"""
        result = self.riscv("disasm", self.elf_path)
        self.assertEqual(result.returncode, 0)
        self.assertRegex(result.stdout, r"00010000:  [0-9A-F]{8}  ADDI ")

    def test_disasm_symbols(self):
        """cli: disasm --symbols labels addresses"""
        result = self.riscv("disasm", self.elf_path, "--symbols")
        self.assertIn("<_start>:", result.stdout)

    def test_errors_are_clean(self):
        """cli: a malformed ELF gives an error, not a traceback"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as handle:
            handle.write(b"\x7fELF" + bytes(200))
            path = handle.name
        try:
            result = self.riscv("exec", path)
            self.assertEqual(result.returncode, 2)
            self.assertTrue(result.stderr.startswith("error:"), result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        finally:
            os.unlink(path)

    def test_not_an_elf_at_all(self):
        """cli: a non-ELF file passed to exec is reported cleanly"""
        result = self.riscv("exec", os.path.join(PROJECT_ROOT, "README.md"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
