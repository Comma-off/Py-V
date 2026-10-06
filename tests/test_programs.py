"""End-to-end tests: complete programs, checked at their final state and at
intermediate checkpoints.

These are the tests that would catch an emulator that gets every instruction
right in isolation but mis-sequences them.
"""

import unittest

from tests.helpers import (
    SCRATCH,
    CPU,
    EmulatorTestCase,
    assemble,
    encode,
    run_asm,
    run_example,
    run_words,
)

from riscv.host import halt_on_ecall


class TestCanonicalProgram(EmulatorTestCase):
    def test_add_three_numbers(self):
        """program: addi/addi/add leaves x3 == 30"""
        cpu = run_words(
            [
                0x00A00093,  # addi x1, x0, 10
                0x01400113,  # addi x2, x0, 20
                0x002081B3,  # add  x3, x1, x2
            ]
        )
        self.assertReg(cpu, 1, 10)
        self.assertReg(cpu, 2, 20)
        self.assertReg(cpu, 3, 30)
        self.assertPC(cpu, 12)
        self.assertEqual(cpu.instret, 3)

    def test_checkpoints_between_instructions(self):
        """program: state is correct after each individual instruction"""
        cpu = CPU()
        cpu.load_program(assemble("addi x1, x0, 10\naddi x2, x0, 20\nadd x3, x1, x2").image)

        cpu.step()
        self.assertRegs(cpu, {1: 10, 2: 0, 3: 0})
        self.assertPC(cpu, 4)

        cpu.step()
        self.assertRegs(cpu, {1: 10, 2: 20, 3: 0})
        self.assertPC(cpu, 8)

        cpu.step()
        self.assertRegs(cpu, {1: 10, 2: 20, 3: 30})
        self.assertPC(cpu, 12)


class TestLoops(EmulatorTestCase):
    def test_sum_1_to_10(self):
        """program: a counted loop sums 1..10"""
        cpu = run_asm(
            """
                li   t0, 0
                li   t1, 1
            loop:
                li   t2, 11
                bge  t1, t2, done
                add  t0, t0, t1
                addi t1, t1, 1
                j    loop
            done:
                mv   s0, t0
            """
        )
        self.assertReg(cpu, 8, 55)

    def test_countdown_loop(self):
        """program: a decrementing loop terminates exactly"""
        cpu = run_asm(
            """
                li   t0, 10
                li   t1, 0
            loop:
                beqz t0, done
                addi t0, t0, -1
                addi t1, t1, 1
                j    loop
            done:
            """
        )
        self.assertReg(cpu, 5, 0)
        self.assertReg(cpu, 6, 10, "the loop body ran exactly ten times")

    def test_nested_loops(self):
        """program: nested loops multiply by repeated addition"""
        cpu = run_asm(
            """
                li   s0, 0
                li   t0, 7          # outer
            outer:
                beqz t0, done
                li   t1, 6          # inner
            inner:
                beqz t1, next
                addi s0, s0, 1
                addi t1, t1, -1
                j    inner
            next:
                addi t0, t0, -1
                j    outer
            done:
            """
        )
        self.assertReg(cpu, 8, 42)

    def test_loop_writes_an_array(self):
        """program: a loop fills an array in memory"""
        cpu = run_asm(
            f"""
                li   t0, {SCRATCH}
                li   t1, 0          # i
                li   t2, 8          # n
            fill:
                bge  t1, t2, done
                slli t3, t1, 2      # i * 4
                add  t4, t0, t3
                slli t5, t1, 1      # i * 2
                sw   t5, 0(t4)
                addi t1, t1, 1
                j    fill
            done:
            """
        )
        for index in range(8):
            self.assertWord(cpu, SCRATCH + 4 * index, 2 * index)


class TestConditionals(EmulatorTestCase):
    def test_max_of_two_signed_values(self):
        """program: a signed max() implemented with BGE"""
        source = """
                li   a0, {a}
                li   a1, {b}
                bge  a0, a1, keep
                mv   a0, a1
            keep:
                mv   s0, a0
        """
        for a, b, expected in [(3, 7, 7), (7, 3, 7), (-5, 2, 2), (2, -5, 2), (-9, -2, -2)]:
            cpu = run_asm(source.format(a=a, b=b))
            self.assertReg(cpu, 8, expected & 0xFFFFFFFF, f"max({a}, {b})")

    def test_absolute_value(self):
        """program: abs() using BGEZ and SUB"""
        for value, expected in [(5, 5), (-5, 5), (0, 0), (-2147483647, 2147483647)]:
            cpu = run_asm(
                f"""
                    li   a0, {value}
                    bgez a0, positive
                    sub  a0, x0, a0
                positive:
                    mv   s0, a0
                """
            )
            self.assertReg(cpu, 8, expected)

    def test_branch_chain(self):
        """program: a chain of comparisons selects one path"""
        source = """
                li   a0, {value}
                li   t0, 10
                blt  a0, t0, small
                li   t0, 100
                blt  a0, t0, medium
                li   s0, 3
                j    done
            small:
                li   s0, 1
                j    done
            medium:
                li   s0, 2
            done:
        """
        for value, expected in [(5, 1), (50, 2), (500, 3)]:
            cpu = run_asm(source.format(value=value))
            self.assertReg(cpu, 8, expected, f"value {value}")


class TestFunctions(EmulatorTestCase):
    def test_leaf_call_and_return(self):
        """program: JAL/RET call and return"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                li   a0, 20
                li   a1, 22
                jal  ra, add2
                mv   s0, a0
                li   a7, 93
                ecall
            add2:
                add  a0, a0, a1
                ret
            """
        )
        self.assertReg(cpu, 8, 42)
        self.assertTrue(cpu.halted)

    def test_indirect_call(self):
        """program: an indirect call through JALR"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                la   t0, double
                li   a0, 21
                jalr ra, 0(t0)
                mv   s0, a0
                li   a7, 93
                ecall
            double:
                add  a0, a0, a0
                ret
            """
        )
        self.assertReg(cpu, 8, 42)

    def test_recursion_with_a_stack_frame(self):
        """program: recursive factorial using the stack"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                li   a0, 5
                jal  ra, fact
                mv   s0, a0
                li   a7, 93
                ecall

            fact:
                addi sp, sp, -16
                sw   ra, 12(sp)
                sw   a0, 8(sp)
                li   t0, 2
                blt  a0, t0, base
                addi a0, a0, -1
                jal  ra, fact
                lw   t1, 8(sp)
                jal  ra, mul
                j    restore
            base:
                li   a0, 1
            restore:
                lw   ra, 12(sp)
                addi sp, sp, 16
                ret

            mul:                    # a0 * t1 -> a0
                mv   t2, a0
                li   a0, 0
                mv   t3, t1
            mul_step:
                beqz t3, mul_ret
                add  a0, a0, t2
                addi t3, t3, -1
                j    mul_step
            mul_ret:
                ret
            """
        )
        self.assertReg(cpu, 8, 120)

    def test_stack_is_balanced_after_recursion(self):
        """program: recursion leaves the stack pointer where it started"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                mv   s1, sp
                li   a0, 6
                jal  ra, countdown
                sub  s0, sp, s1
                li   a7, 93
                ecall
            countdown:
                addi sp, sp, -16
                sw   ra, 12(sp)
                beqz a0, back
                addi a0, a0, -1
                jal  ra, countdown
            back:
                lw   ra, 12(sp)
                addi sp, sp, 16
                ret
            """
        )
        self.assertReg(cpu, 8, 0, "sp must return to its original value")

    def test_saved_registers_survive_a_call(self):
        """program: callee-saved registers survive a call that preserves them"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                li   s0, 0x1234
                li   s1, 0x5678
                jal  ra, clobber
                li   a7, 93
                ecall
            clobber:
                addi sp, sp, -16
                sw   s0, 12(sp)
                sw   s1, 8(sp)
                li   s0, 0
                li   s1, 0
                lw   s0, 12(sp)
                lw   s1, 8(sp)
                addi sp, sp, 16
                ret
            """
        )
        self.assertReg(cpu, 8, 0x1234)
        self.assertReg(cpu, 9, 0x5678)


class TestMemoryPrograms(EmulatorTestCase):
    def test_memcpy(self):
        """program: a byte-wise memcpy loop"""
        source_addr = SCRATCH
        dest_addr = SCRATCH + 0x100
        cpu = CPU(ecall_handler=halt_on_ecall)
        program = assemble(
            f"""
                li   a0, {dest_addr}
                li   a1, {source_addr}
                li   a2, 13
            copy:
                beqz a2, done
                lbu  t0, 0(a1)
                sb   t0, 0(a0)
                addi a0, a0, 1
                addi a1, a1, 1
                addi a2, a2, -1
                j    copy
            done:
                li   a7, 93
                ecall
            """
        )
        program.load(cpu.memory)
        cpu.write_bytes(source_addr, b"Hello, RISC-V")
        cpu.set_pc(program.entry)
        cpu.run(1000)
        self.assertEqual(cpu.read_bytes(dest_addr, 13), b"Hello, RISC-V")

    def test_string_length(self):
        """program: strlen over a NUL-terminated string"""
        cpu = CPU(ecall_handler=halt_on_ecall)
        program = assemble(
            f"""
                li   a0, {SCRATCH}
                li   s0, 0
            scan:
                lbu  t0, 0(a0)
                beqz t0, done
                addi s0, s0, 1
                addi a0, a0, 1
                j    scan
            done:
                li   a7, 93
                ecall
            """
        )
        program.load(cpu.memory)
        cpu.write_bytes(SCRATCH, b"emulator\x00")
        cpu.set_pc(program.entry)
        cpu.run(1000)
        self.assertReg(cpu, 8, 8)

    def test_sum_an_array(self):
        """program: sum an array of signed words"""
        values = [10, -20, 30, -40, 50]
        cpu = CPU(ecall_handler=halt_on_ecall)
        program = assemble(
            f"""
                li   a0, {SCRATCH}
                li   a1, {len(values)}
                li   s0, 0
            loop:
                beqz a1, done
                lw   t0, 0(a0)
                add  s0, s0, t0
                addi a0, a0, 4
                addi a1, a1, -1
                j    loop
            done:
                li   a7, 93
                ecall
            """
        )
        program.load(cpu.memory)
        for index, value in enumerate(values):
            cpu.write_u32(SCRATCH + 4 * index, value & 0xFFFFFFFF)
        cpu.set_pc(program.entry)
        cpu.run(1000)
        self.assertReg(cpu, 8, sum(values) & 0xFFFFFFFF)

    def test_word_and_byte_views_agree(self):
        """program: a word store is visible through byte loads"""
        cpu = run_asm(
            f"""
                li   t0, {SCRATCH}
                li   t1, 0x11223344
                sw   t1, 0(t0)
                lbu  s0, 0(t0)
                lbu  s1, 1(t0)
                lbu  s2, 2(t0)
                lbu  s3, 3(t0)
            """
        )
        self.assertRegs(cpu, {8: 0x44, 9: 0x33, 18: 0x22, 19: 0x11})


class TestArithmeticEdgeCases(EmulatorTestCase):
    def test_negative_immediates(self):
        """program: negative immediates throughout"""
        cpu = run_asm(
            """
                addi s0, x0, -1
                addi s1, x0, -2048
                addi s2, s0, -1
                andi s3, s0, -16
                ori  s4, x0, -1
                xori s5, x0, -1
                slti s6, s0, 0
            """
        )
        self.assertRegs(
            cpu,
            {
                8: 0xFFFFFFFF,
                9: 0xFFFFF800,
                18: 0xFFFFFFFE,
                19: 0xFFFFFFF0,
                20: 0xFFFFFFFF,
                21: 0xFFFFFFFF,
                22: 1,
            },
        )

    def test_integer_overflow_is_silent(self):
        """program: overflow wraps instead of trapping"""
        cpu = run_asm(
            """
                li   t0, 0x7FFFFFFF
                addi s0, t0, 1          # INT_MAX + 1
                li   t1, 0x80000000
                addi s1, t1, -1         # INT_MIN - 1
                add  s2, t0, t0         # INT_MAX + INT_MAX
                sub  s3, t1, t0         # INT_MIN - INT_MAX
            """
        )
        self.assertRegs(cpu, {8: 0x80000000, 9: 0x7FFFFFFF, 18: 0xFFFFFFFE, 19: 1})

    def test_signed_vs_unsigned_comparison(self):
        """program: signed and unsigned comparisons differ on the same bits"""
        cpu = run_asm(
            """
                li   t0, -1
                li   t1, 1
                slt  s0, t0, t1         # signed:   -1 < 1        -> 1
                sltu s1, t0, t1         # unsigned: huge < 1      -> 0
                slt  s2, t1, t0         # signed:   1 < -1        -> 0
                sltu s3, t1, t0         # unsigned: 1 < huge      -> 1
            """
        )
        self.assertRegs(cpu, {8: 1, 9: 0, 18: 0, 19: 1})

    def test_shift_edge_cases(self):
        """program: shifts by 0 and 31, logical vs arithmetic"""
        cpu = run_asm(
            """
                li   t0, -1
                srli s0, t0, 0
                srli s1, t0, 31
                srai s2, t0, 31
                slli s3, t0, 31
                li   t1, 0x80000000
                srli s4, t1, 31
                srai s5, t1, 31
            """
        )
        self.assertRegs(
            cpu,
            {8: 0xFFFFFFFF, 9: 1, 18: 0xFFFFFFFF, 19: 0x80000000, 20: 1, 21: 0xFFFFFFFF},
        )

    def test_register_shift_forms(self):
        """program: SLL/SRL/SRA with register shift amounts"""
        cpu = run_asm(
            """
                li   t0, -1
                li   t1, 4
                srl  s0, t0, t1         # logical  -> 0x0FFFFFFF
                sra  s1, t0, t1         # arith    -> 0xFFFFFFFF
                sll  s2, t0, t1         #          -> 0xFFFFFFF0
                li   t2, 0x80000000
                sra  s3, t2, t1         # arith    -> 0xF8000000
                srl  s4, t2, t1         # logical  -> 0x08000000
                li   t3, 36             # shift amounts use only the low 5 bits
                sll  s5, t0, t3         # 36 & 31 == 4
            """
        )
        self.assertRegs(
            cpu,
            {8: 0x0FFFFFFF, 9: 0xFFFFFFFF, 18: 0xFFFFFFF0, 19: 0xF8000000,
             20: 0x08000000, 21: 0xFFFFFFF0},
        )

    def test_jalr_alignment_and_low_bit(self):
        """program: JALR clears the target's low bit in a real call"""
        cpu = run_asm(
            """
                li   sp, 0x8000
                la   t0, target
                addi t0, t0, 1          # a deliberately odd function pointer
                jalr ra, 0(t0)
                li   a7, 93
                ecall
            target:
                li   s0, 0xABC
                ret
            """
        )
        self.assertReg(cpu, 8, 0xABC, "the odd bit must be masked off, not trap")

    def test_building_every_constant_class(self):
        """program: LUI/ADDI build constants across the whole 32-bit range"""
        for value in (0, 1, -1, 0x7FF, 0x800, -0x800, -0x801, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            cpu = run_asm(f"li s0, {value}")
            self.assertReg(cpu, 8, value & 0xFFFFFFFF, f"li s0, {value}")


class TestExamplePrograms(EmulatorTestCase):
    """The programs under examples/ must produce their documented results."""

    def test_hello_example(self):
        """example: hello.asm prints its message and exits 0"""
        cpu, host = run_example("hello", capture=True)
        self.assertEqual(host.text(), "Hello, RISC-V!\n")
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 0)

    def test_arithmetic_example(self):
        """example: arithmetic.asm"""
        cpu, _ = run_example("arithmetic")
        self.assertRegs(
            cpu,
            {
                8: 30,           # s0  = 10 + 20
                9: 0xFFFFFFF6,   # s1  = 10 - 20
                18: 0x80000000,  # s2  = 0x7FFFFFFF + 1
                19: 0x7FFFFFFF,  # s3  = srli -1, 1
                20: 0xFFFFFFFF,  # s4  = srai -1, 1
                21: 1,           # s5  = slt  -1, 10
                22: 0,           # s6  = sltu -1, 10
                23: 0x12345678,  # s7
                24: 0xDEADBEEF,  # s8
                25: 0x0000FF00,  # s9
                26: 0x00100000,  # s10 = 1 << 20
                27: 0xFFFFFFFF,  # s11 = -1
            },
        )

    def test_branches_example(self):
        """example: branches.asm passes all twelve checks"""
        cpu, _ = run_example("branches")
        self.assertReg(cpu, 8, 12, "every branch behaved as expected")
        self.assertReg(cpu, 10, 0, "exit code 0 means no check failed")

    def test_loops_example(self):
        """example: loops.asm"""
        cpu, _ = run_example("loops")
        self.assertRegs(cpu, {8: 55, 9: 42, 18: 10, 19: 385, 21: 6})

    def test_memory_example(self):
        """example: memory.asm"""
        cpu, _ = run_example("memory")
        self.assertRegs(
            cpu,
            {
                8: 0x12345678,
                9: 0x00000078,
                18: 0xFFFFFFFF,
                19: 0x000000FF,
                20: 0xFFFF8000,
                21: 0x00008000,
                22: 0x11,
                23: 0x44,
                24: 0x12345678,
            },
        )

    def test_functions_example(self):
        """example: functions.asm"""
        cpu, _ = run_example("functions")
        self.assertRegs(cpu, {8: 6, 9: 120, 18: 6, 19: 5})

    def test_every_example_halts_cleanly(self):
        """example: every example program halts with exit code 0"""
        for name in ("hello", "arithmetic", "branches", "loops", "memory", "functions"):
            cpu, _ = run_example(name, capture=True)
            self.assertTrue(cpu.halted, f"{name} did not halt")
            self.assertEqual(cpu.halt_code, 0, f"{name} exited non-zero")


class TestDeterminism(EmulatorTestCase):
    def test_reset_and_rerun_is_reproducible(self):
        """program: a program produces identical state on every run"""
        program = assemble(
            """
                li   t0, 0
                li   t1, 1
            loop:
                li   t2, 21
                bge  t1, t2, done
                add  t0, t0, t1
                addi t1, t1, 1
                j    loop
            done:
                li   a7, 93
                li   a0, 0
                ecall
            """
        )
        cpu = CPU(ecall_handler=halt_on_ecall)
        cpu.load_program(program.image)
        cpu.run(1000)
        first = cpu.state()
        cpu.reset()
        cpu.run(1000)
        self.assertEqual(cpu.state(), first)

    def test_two_cpus_agree(self):
        """program: two independent CPUs reach the same state"""
        words = assemble("li x1, 100\nli x2, 200\nadd x3, x1, x2\nsub x4, x2, x1").words
        one = run_words(words)
        two = run_words(words)
        self.assertEqual(one.state(), two.state())


if __name__ == "__main__":
    unittest.main()
