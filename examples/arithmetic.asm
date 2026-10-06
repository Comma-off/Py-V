# arithmetic.asm -- integer arithmetic, logic, shifts and the edge cases that
# distinguish a correct RV32I core from a sloppy one.
#
# Expected final state:
#   s0  = 30            (10 + 20)
#   s1  = 0xFFFFFFF6    (10 - 20, i.e. -10)
#   s2  = 0x80000000    (0x7FFFFFFF + 1 wraps silently; no overflow trap)
#   s3  = 0x7FFFFFFF    (srli of -1 by 1: logical shift)
#   s4  = 0xFFFFFFFF    (srai of -1 by 1: arithmetic shift keeps the sign)
#   s5  = 1             (slt  -1 < 1 signed)
#   s6  = 0             (sltu -1 < 1 unsigned: 0xFFFFFFFF is huge)
#   s7  = 0x12345678    (li of a large constant: lui + addi)
#   s8  = 0xDEADBEEF    (li of a constant whose low half is negative)
#   s9  = 0x0000FF00    (and/or/xor combination)
#   s10 = 0x00100000    (1 << 20, shift amount taken from the low 5 bits of t1)
#   s11 = 0xFFFFFFFF    (addi x0, -1: negative immediates sign-extend)

        .text
_start:
        li      t0, 10
        li      t1, 20
        add     s0, t0, t1              # 30
        sub     s1, t0, t1              # -10

        li      t2, 0x7FFFFFFF
        addi    s2, t2, 1               # wraps to 0x80000000

        li      t3, -1
        srli    s3, t3, 1               # 0x7FFFFFFF
        srai    s4, t3, 1               # 0xFFFFFFFF
        slt     s5, t3, t0              # -1 <  10 signed   -> 1
        sltu    s6, t3, t0              # huge < 10 unsigned -> 0

        li      s7, 0x12345678
        li      s8, 0xDEADBEEF          # low 12 bits sign-extend negative

        li      t4, 0x0000FFFF
        li      t5, 0x0000FF00
        and     s9, t4, t5              # 0x0000FF00

        li      t6, 1
        sll     s10, t6, t1             # 1 << (20 & 31) = 0x00100000

        addi    s11, x0, -1             # 0xFFFFFFFF

        li      a0, 0
        li      a7, 93
        ecall
