# loops.asm -- counted loops, an accumulator, and a nested loop that
# implements multiplication with adds (RV32I has no MUL).
#
# Expected final state:
#   s0 == 55     sum of 1..10
#   s1 == 42     7 * 6, computed by the nested loop
#   s2 == 10     iterations of the sum loop
#   s5 == 6      iterations of the multiply outer loop
#   s3 == 385    sum of squares 1^2..10^2, using the multiply helper
#   a0 == 0

        .text
_start:
        li      sp, 0x8000              # stack top (nothing below it is used)

# ---- sum 1..10 -----------------------------------------------------------
        li      s0, 0                   # accumulator
        li      s2, 0                   # iteration counter
        li      t0, 1                   # i = 1
        li      t1, 11                  # limit
sum_loop:
        bge     t0, t1, sum_done        # while (i < 11)
        add     s0, s0, t0
        addi    t0, t0, 1
        addi    s2, s2, 1
        j       sum_loop
sum_done:

# ---- 7 * 6 via repeated addition (nested loop) ---------------------------
        li      s1, 0
        li      t0, 6                   # outer counter
        li      s5, 0                   # outer iteration counter
mul_outer:
        beqz    t0, mul_done
        li      t2, 7                   # inner counter
mul_inner:
        beqz    t2, mul_next
        addi    s1, s1, 1
        addi    t2, t2, -1
        j       mul_inner
mul_next:
        addi    t0, t0, -1
        addi    s5, s5, 1
        j       mul_outer
mul_done:

# ---- sum of squares 1..10 using the multiply helper ----------------------
        li      s3, 0
        li      s4, 1                   # i = 1
sq_loop:
        li      t3, 11
        bge     s4, t3, sq_done
        mv      a0, s4
        mv      a1, s4
        jal     ra, multiply            # a0 = a0 * a1
        add     s3, s3, a0
        addi    s4, s4, 1
        j       sq_loop
sq_done:

        li      a0, 0
        li      a7, 93
        ecall

# multiply(a0, a1) -> a0, by repeated addition.  Leaf function: no stack use.
multiply:
        mv      t0, a0
        li      a0, 0
        mv      t1, a1
mul_step:
        beqz    t1, mul_ret
        add     a0, a0, t0
        addi    t1, t1, -1
        j       mul_step
mul_ret:
        ret
