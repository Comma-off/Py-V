/*
 * multiply.S -- the RV32M extension: every one of the eight instructions,
 * including the cases that distinguish a correct implementation from a
 * plausible one.
 *
 * Each check bumps s0.  On the first mismatch the program exits with the
 * failing check's number, so a failure says exactly which one broke.
 *
 * Build:
 *   riscv32-linux-gnu-as -march=rv32im -mabi=ilp32 -mno-relax multiply.S -o multiply.o
 *   riscv32-linux-gnu-ld -m elf32lriscv -Ttext=0x10000 --no-relax multiply.o -o multiply.elf
 *   python riscv.py exec multiply.elf
 *
 * Expected: exit status 0, s0 == 14 (all thirteen checks passed).
 * Running it with --isa rv32i must fail with an illegal instruction instead.
 */
        .text
        .globl  _start
_start:
        li      s0, 1                   /* check number                     */

/* 1: MUL, the ordinary case */
        li      t0, 7
        li      t1, 6
        mul     t2, t0, t1
        li      t3, 42
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 2: MUL with a negative operand -- the low word is signedness-agnostic */
        li      t0, -7
        li      t1, 6
        mul     t2, t0, t1
        li      t3, -42
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 3: MULH -- 0x10000 * 0x10000 is 2^32, so the high word is 1 */
        li      t0, 0x10000
        mulh    t2, t0, t0
        li      t3, 1
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 4: MULHU -- 0xFFFFFFFF squared, unsigned */
        li      t0, -1
        mulhu   t2, t0, t0
        li      t3, 0xFFFFFFFE
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 5: MULHSU -- the same bits again, but rs1 signed and rs2 unsigned */
        li      t0, -1
        mulhsu  t2, t0, t0
        li      t3, -1
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 6: MULH of the same bits differs from both of the above */
        li      t0, -1
        mulh    t2, t0, t0
        bnez    t2, fail
        addi    s0, s0, 1

/* 7: DIV, the ordinary case */
        li      t0, 42
        li      t1, 6
        div     t2, t0, t1
        li      t3, 7
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 8: DIV truncates toward zero: -7 / 2 == -3, not -4 */
        li      t0, -7
        li      t1, 2
        div     t2, t0, t1
        li      t3, -3
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 9: REM takes the sign of the dividend: -7 % 2 == -1, not 1 */
        li      t0, -7
        li      t1, 2
        rem     t2, t0, t1
        li      t3, -1
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 10: DIVU treats the operands as unsigned */
        li      t0, -1                  /* 0xFFFFFFFF                       */
        li      t1, 2
        divu    t2, t0, t1
        li      t3, 0x7FFFFFFF
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 11: REMU likewise */
        li      t0, -1
        li      t1, 16
        remu    t2, t0, t1
        li      t3, 15
        bne     t2, t3, fail
        addi    s0, s0, 1

/* 12: division by zero is defined, not a trap: DIV -> -1, REM -> dividend */
        li      t0, 42
        li      t1, 0
        div     t2, t0, t1
        li      t3, -1
        bne     t2, t3, fail
        rem     t2, t0, t1
        bne     t2, t0, fail
        addi    s0, s0, 1

/* 13: the one signed overflow: INT32_MIN / -1 wraps to INT32_MIN, REM is 0 */
        li      t0, 0x80000000
        li      t1, -1
        div     t2, t0, t1
        bne     t2, t0, fail
        rem     t2, t0, t1
        bnez    t2, fail
        addi    s0, s0, 1

        li      a0, 0                   /* every check passed               */
        j       done
fail:
        mv      a0, s0                  /* exit with the failing check      */
done:
        li      a7, 93
        ecall
