# functions.asm -- the calling convention: JAL/JALR, a stack frame, saving ra,
# recursion, and an indirect call through a function pointer.
#
# Expected final state:
#   s0 == 6      sum3(1, 2, 3)
#   s1 == 120    factorial(5), computed recursively
#   s2 == 6      sum3 called again, indirectly through a register
#   s3 == 5      leaf_depth: the recursion reached depth 5
#   a0 == 0

        .text
_start:
        li      sp, 0x8000              # initialise the stack pointer
        li      s3, 0

# ---- direct call ---------------------------------------------------------
        li      a0, 1
        li      a1, 2
        li      a2, 3
        jal     ra, sum3
        mv      s0, a0                  # 6

# ---- recursive call ------------------------------------------------------
        li      a0, 5
        jal     ra, factorial
        mv      s1, a0                  # 120

# ---- indirect call through a function pointer (JALR) ---------------------
        la      t0, sum3
        li      a0, 1
        li      a1, 2
        li      a2, 3
        jalr    ra, 0(t0)
        mv      s2, a0                  # 6

        li      a0, 0
        li      a7, 93
        ecall

# -------------------------------------------------------------------------
# sum3(a0, a1, a2) -> a0.  Leaf function: no frame needed.
# -------------------------------------------------------------------------
sum3:
        add     a0, a0, a1
        add     a0, a0, a2
        ret

# -------------------------------------------------------------------------
# factorial(a0) -> a0.  Recursive, so it builds a frame and saves ra.
# s3 tracks the deepest recursion level reached.
# -------------------------------------------------------------------------
factorial:
        addi    sp, sp, -16
        sw      ra, 12(sp)
        sw      a0, 8(sp)               # keep n for after the recursive call

        blt     s3, a0, record_depth    # s3 = max(s3, n)
        j       depth_done
record_depth:
        mv      s3, a0
depth_done:

        li      t0, 2
        blt     a0, t0, factorial_base  # n < 2 -> return 1

        addi    a0, a0, -1
        jal     ra, factorial           # a0 = factorial(n - 1)
        lw      t1, 8(sp)               # t1 = n (reloaded: t-regs are caller-saved)
        jal     ra, multiply_helper     # a0 = a0 * t1
        j       factorial_return

factorial_base:
        li      a0, 1

factorial_return:
        lw      ra, 12(sp)
        addi    sp, sp, 16
        ret

# multiply_helper(a0, t1) -> a0, by repeated addition.  Leaf function.
multiply_helper:
        mv      t2, a0
        li      a0, 0
        mv      t3, t1
multiply_step:
        beqz    t3, multiply_ret
        add     a0, a0, t2
        addi    t3, t3, -1
        j       multiply_step
multiply_ret:
        ret
