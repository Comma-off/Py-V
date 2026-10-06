/*
 * branches.S -- conditional branches and a counted loop.
 *
 * Sums 1..10 (== 55) with a loop, then checks each branch kind.  Every check
 * that behaves correctly increments s1.
 *
 * Expected: exit status 0, s0 == 55, s1 == 6.
 */
        .text
        .globl  _start
_start:
        li      s0, 0                   /* accumulator                      */
        li      s1, 0                   /* passed-check counter             */
        li      t0, 1                   /* i                                */
        li      t1, 11                  /* limit                            */
sum_loop:
        bge     t0, t1, sum_done
        add     s0, s0, t0
        addi    t0, t0, 1
        j       sum_loop
sum_done:

        li      t2, 5
        li      t3, -1                  /* negative signed, huge unsigned   */

        beq     t2, t2, check_bne
        j       fail
check_bne:
        addi    s1, s1, 1
        bne     t2, t3, check_blt
        j       fail
check_blt:
        addi    s1, s1, 1
        blt     t3, t2, check_bge       /* -1 < 5 signed                    */
        j       fail
check_bge:
        addi    s1, s1, 1
        bge     t2, t2, check_bltu      /* 5 >= 5                           */
        j       fail
check_bltu:
        addi    s1, s1, 1
        bltu    t2, t3, check_bgeu      /* 5 < 0xFFFFFFFF unsigned          */
        j       fail
check_bgeu:
        addi    s1, s1, 1
        bgeu    t3, t2, all_passed      /* 0xFFFFFFFF >= 5 unsigned         */
        j       fail
all_passed:
        addi    s1, s1, 1
        li      a0, 0
        j       done
fail:
        li      a0, 1
done:
        li      a7, 93
        ecall
