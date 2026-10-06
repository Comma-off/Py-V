/*
 * arithmetic.S -- register arithmetic and logic, exiting with the result.
 *
 * Computes ((10 + 20) * 2 - 18) / 1 == 42 using only RV32I (the doubling is a
 * shift, since RV32I has no MUL), then exits with 42.
 *
 * Expected: exit status 42, s0 == 30, s1 == 60.
 */
        .text
        .globl  _start
_start:
        li      t0, 10
        li      t1, 20
        add     s0, t0, t1              /* s0 = 30                          */
        slli    s1, s0, 1               /* s1 = 60                          */
        addi    s2, s1, -18             /* s2 = 42                          */

        /* A few logic operations whose results the test also checks. */
        li      t2, 0xFF00
        li      t3, 0x0FF0
        and     s3, t2, t3              /* s3 = 0x0F00                      */
        or      s4, t2, t3              /* s4 = 0xFFF0                      */
        xor     s5, t2, t3              /* s5 = 0xF0F0                      */

        /* Signed vs unsigned comparison of the same bit pattern. */
        li      t4, -1
        li      t5, 1
        slt     s6, t4, t5              /* signed:   1                      */
        sltu    s7, t4, t5              /* unsigned: 0                      */

        mv      a0, s2                  /* exit with 42                     */
        li      a7, 93
        ecall
