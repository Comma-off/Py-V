# branches.asm -- every RV32I conditional branch, taken and not taken.
#
# s0 counts branches that behaved correctly; each check adds 1.
#
# Expected final state:
#   s0 == 12   (6 branch kinds x {taken, not-taken})
#   a0 == 0    (exit status: 0 means every check passed)

        .text
_start:
        li      s0, 0                   # correct-check counter
        li      t0, 5
        li      t1, 5
        li      t2, -1                  # 0xFFFFFFFF: negative signed, huge unsigned

# ---- BEQ -----------------------------------------------------------------
        beq     t0, t1, beq_taken       # 5 == 5 -> taken
        j       fail
beq_taken:
        addi    s0, s0, 1
        beq     t0, t2, fail            # 5 != -1 -> not taken
        addi    s0, s0, 1

# ---- BNE -----------------------------------------------------------------
        bne     t0, t2, bne_taken
        j       fail
bne_taken:
        addi    s0, s0, 1
        bne     t0, t1, fail
        addi    s0, s0, 1

# ---- BLT (signed) --------------------------------------------------------
        blt     t2, t0, blt_taken       # -1 < 5 signed -> taken
        j       fail
blt_taken:
        addi    s0, s0, 1
        blt     t0, t2, fail            # 5 < -1 is false
        addi    s0, s0, 1

# ---- BGE (signed) --------------------------------------------------------
        bge     t0, t1, bge_taken       # 5 >= 5 -> taken
        j       fail
bge_taken:
        addi    s0, s0, 1
        bge     t2, t0, fail            # -1 >= 5 is false
        addi    s0, s0, 1

# ---- BLTU (unsigned) -----------------------------------------------------
        bltu    t0, t2, bltu_taken      # 5 < 0xFFFFFFFF unsigned -> taken
        j       fail
bltu_taken:
        addi    s0, s0, 1
        bltu    t2, t0, fail            # 0xFFFFFFFF < 5 is false
        addi    s0, s0, 1

# ---- BGEU (unsigned) -----------------------------------------------------
        bgeu    t2, t0, bgeu_taken      # 0xFFFFFFFF >= 5 unsigned -> taken
        j       fail
bgeu_taken:
        addi    s0, s0, 1
        bgeu    t0, t2, fail            # 5 >= 0xFFFFFFFF is false
        addi    s0, s0, 1

        li      a0, 0                   # all checks passed
        j       done
fail:
        li      a0, 1
done:
        li      a7, 93
        ecall
