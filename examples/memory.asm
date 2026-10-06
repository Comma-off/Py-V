# memory.asm -- loads and stores at every width, little-endian byte order, and
# the sign/zero-extension difference between LB/LBU and LH/LHU.
#
# Expected final state:
#   s0 == 0x12345678   word round-trip through sw/lw
#   s1 == 0x00000078   lb of the *lowest* byte (little-endian) -- positive
#   s2 == 0xFFFFFFFF   lb of a stored 0xFF: sign-extended to -1
#   s3 == 0x000000FF   lbu of the same byte: zero-extended
#   s4 == 0xFFFF8000   lh  of a stored 0x8000: sign-extended
#   s5 == 0x00008000   lhu of the same half: zero-extended
#   s6 == 0x11         highest byte of 0x11223344 (proves little-endian order)
#   s7 == 0x44         lowest byte of the same word
#   s8 == 0x12345678   load through a *negative* offset
#   a0 == 0

        .text
_start:
        la      s11, buffer             # s11 = base address of the buffer

# ---- word round-trip -----------------------------------------------------
        li      t0, 0x12345678
        sw      t0, 0(s11)
        lw      s0, 0(s11)              # 0x12345678
        lb      s1, 0(s11)              # 0x78 -- little-endian low byte

# ---- byte: signed vs unsigned load ---------------------------------------
        li      t1, 0xFF
        sb      t1, 4(s11)
        lb      s2, 4(s11)              # 0xFFFFFFFF
        lbu     s3, 4(s11)              # 0x000000FF

# ---- half: signed vs unsigned load ---------------------------------------
        li      t2, 0x8000
        sh      t2, 6(s11)
        lh      s4, 6(s11)              # 0xFFFF8000
        lhu     s5, 6(s11)              # 0x00008000

# ---- byte order of a stored word -----------------------------------------
        li      t3, 0x11223344
        sw      t3, 8(s11)
        lbu     s6, 11(s11)             # most significant byte -> 0x11
        lbu     s7, 8(s11)              # least significant byte -> 0x44

# ---- negative offsets ----------------------------------------------------
        addi    t4, s11, 16
        lw      s8, -16(t4)             # same as 0(s11) -> 0x12345678

        li      a0, 0
        li      a7, 93
        ecall

        .data
buffer:
        .space  32
