/*
 * functions.S -- the calling convention: jal, jalr, stack frames, ra, and
 * callee-saved registers across a recursive call.
 *
 * Expected: exit status 0.
 *   s0 == 42     sum3(10, 14, 18) via jal
 *   s1 == 120    factorial(5), recursive, using the stack
 *   s2 == 42     sum3 again, called indirectly through jalr
 *   s3 == 0x5A5A callee-saved register preserved across all of the above
 */
        .text
        .globl  _start
_start:
        li      s3, 0x5A5A              /* must survive every call below    */

        li      a0, 10
        li      a1, 14
        li      a2, 18
        jal     ra, sum3
        mv      s0, a0                  /* 42                               */

        li      a0, 5
        jal     ra, factorial
        mv      s1, a0                  /* 120                              */

        la      t0, sum3                /* function pointer                 */
        li      a0, 10
        li      a1, 14
        li      a2, 18
        jalr    ra, 0(t0)
        mv      s2, a0                  /* 42                               */

        li      a0, 0
        li      a7, 93
        ecall

/* sum3(a0, a1, a2) -> a0.  Leaf: no frame needed. */
sum3:
        add     a0, a0, a1
        add     a0, a0, a2
        ret

/* factorial(a0) -> a0.  Recursive: builds a frame and saves ra. */
factorial:
        addi    sp, sp, -16
        sw      ra, 12(sp)
        sw      s4, 8(sp)               /* callee-saved: save before use    */
        mv      s4, a0                  /* keep n across the recursive call */

        li      t0, 2
        blt     a0, t0, factorial_base

        addi    a0, a0, -1
        jal     ra, factorial           /* a0 = factorial(n - 1)            */
        mv      t1, a0
        mv      t2, s4                  /* RV32I has no MUL: add n times    */
        li      a0, 0
factorial_mul:
        beqz    t2, factorial_done
        add     a0, a0, t1
        addi    t2, t2, -1
        j       factorial_mul

factorial_base:
        li      a0, 1

factorial_done:
        lw      s4, 8(sp)
        lw      ra, 12(sp)
        addi    sp, sp, 16
        ret
