# RV32I integration stress test
# Expected final state:
#   x10 (a0) = 0x0000002A (42)
#   x11 (a1) = 0xFFFFFFF6 (-10)
#   x12 (a2) = 0x00000037 (55)
#   x13 (a3) = 0x0000000A (10)
#   x14 (a4) = 0x00000037 (55)
#   x15 (a5) = 0x00000001
#
# The program exits through the host environment.

.text
.org 0x00000000

start:
    # ------------------------------------------------------------
    # Build a stack near the top of the 32-bit address space.
    # ------------------------------------------------------------
    lui     sp, 0x10000

    # Save a known callee-saved value.
    li      s0, 0x12345678

    # ------------------------------------------------------------
    # Test signed arithmetic.
    # ------------------------------------------------------------
    li      a0, -20
    li      a1, 10
    add     a2, a0, a1          # -10
    sub     a3, a1, a0          # 30

    # ------------------------------------------------------------
    # Test a function call with a stack frame.
    # ------------------------------------------------------------
    li      a0, 10
    li      a1, 32
    call    add_and_double      # returns 84

    # ------------------------------------------------------------
    # Test array construction.
    # ------------------------------------------------------------
    addi    a0, x0, 0            # base = 0
    addi    a1, x0, 10           # count = 10
    call    fill_array

    # ------------------------------------------------------------
    # Sum the array.
    # 0+1+2+...+9 = 45
    # ------------------------------------------------------------
    addi    a0, x0, 0
    addi    a1, x0, 10
    call    sum_array

    # ------------------------------------------------------------
    # Factorial(5) = 120.
    # ------------------------------------------------------------
    addi    a0, x0, 5
    call    factorial

    # ------------------------------------------------------------
    # Test indirect call through JALR.
    # ------------------------------------------------------------
    la      t0, indirect_target
    jalr    ra, 0(t0)

    # ------------------------------------------------------------
    # Verify s0 survived all calls.
    # ------------------------------------------------------------
    li      t0, 0x12345678
    bne     s0, t0, failure

    # Test x0 remains immutable.
    li      t0, 123
    addi    x0, t0, 1
    bne     x0, x0, failure

    # ------------------------------------------------------------
    # Success.
    # ------------------------------------------------------------
success:
    li      a0, 0
    li      a7, 93              # host exit
    ecall

failure:
    li      a0, 1
    li      a7, 93
    ecall


# ================================================================
# add_and_double(a0, a1)
# return: a0 = (a0 + a1) * 2
# ================================================================
add_and_double:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      s0, 8(sp)

    add     a0, a0, a1
    slli    a0, a0, 1

    lw      s0, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    jalr    x0, 0(ra)


# ================================================================
# fill_array(a0=base, a1=count)
# Writes:
#   [0] = 0
#   [1] = 1
#   ...
# ================================================================
fill_array:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      s0, 8(sp)

    addi    s0, a0, 0
    addi    t0, x0, 0

fill_loop:
    bge     t0, a1, fill_done

    slli    t1, t0, 2
    add     t2, s0, t1
    sw      t0, 0(t2)

    addi    t0, t0, 1
    jal     x0, fill_loop

fill_done:
    lw      s0, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    jalr    x0, 0(ra)


# ================================================================
# sum_array(a0=base, a1=count)
# return: a0 = sum
# ================================================================
sum_array:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      s0, 8(sp)

    addi    s0, a0, 0
    addi    t0, x0, 0
    addi    a0, x0, 0

sum_loop:
    bge     t0, a1, sum_done

    slli    t1, t0, 2
    add     t2, s0, t1
    lw      t3, 0(t2)
    add     a0, a0, t3

    addi    t0, t0, 1
    jal     x0, sum_loop

sum_done:
    lw      s0, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    jalr    x0, 0(ra)


# ================================================================
# factorial(a0)
# Recursive factorial.
# factorial(0) = 1
# ================================================================
factorial:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      a0, 8(sp)

    addi    t0, x0, 1
    ble     a0, t0, factorial_base

    addi    a0, a0, -1
    call    factorial

    lw      t0, 8(sp)

    # RV32I has no MUL, so a0 = a0 * t0 by repeated addition.
    # t1/t2 are caller-saved and dead here, so no frame slots are needed.
    mv      t1, a0                  # t1 = factorial(n - 1)
    li      a0, 0                   # accumulator
    mv      t2, t0                  # counter = n
factorial_mul_loop:
    beqz    t2, factorial_mul_done
    add     a0, a0, t1
    addi    t2, t2, -1
    j       factorial_mul_loop
factorial_mul_done:
    jal     x0, factorial_done

factorial_base:
    addi    a0, x0, 1

factorial_done:
    lw      ra, 12(sp)
    addi    sp, sp, 16
    jalr    x0, 0(ra)


# ================================================================
# Indirect JALR target.
# ================================================================
indirect_target:
    li      t0, 55
    li      t1, 55
    bne     t0, t1, failure

    jalr    x0, 0(ra)