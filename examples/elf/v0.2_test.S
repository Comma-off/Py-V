# Py-V v0.2 integration test
#
# Tests:
#   - ELF32 loading
#   - .text / .data / .bss
#   - stack usage
#   - function calls / returns
#   - JALR
#   - RV32I arithmetic
#   - loads/stores
#   - BSS zero-fill
#   - high-ish stack address
#   - host exit syscall (a7 = 93)
#
# Expected exit code: 0
#
# Any failed check exits with a non-zero code.

.section .text
.globl _start

_start:
    # Set up stack.
    # Py-V's ELF runner provides the stack.
    addi sp, sp, -32

    # Save return address.
    sw ra, 28(sp)

    # ------------------------------------------------------------
    # Test 1: .data
    # ------------------------------------------------------------

    la t0, test_data
    lw t1, 0(t0)

    li t2, 0x12345678
    bne t1, t2, fail_data


    # ------------------------------------------------------------
    # Test 2: .bss must start at zero
    # ------------------------------------------------------------

    la t0, test_bss
    lw t1, 0(t0)

    bne t1, zero, fail_bss


    # ------------------------------------------------------------
    # Test 3: store/load
    # ------------------------------------------------------------

    li t1, 0xCAFEBABE
    sw t1, 0(t0)

    lw t2, 0(t0)

    bne t1, t2, fail_memory


    # ------------------------------------------------------------
    # Test 4: arithmetic
    # ------------------------------------------------------------

    li a0, 20
    li a1, 22

    add a2, a0, a1

    li t0, 42
    bne a2, t0, fail_arithmetic


    # ------------------------------------------------------------
    # Test 5: function call
    # ------------------------------------------------------------

    li a0, 7
    li a1, 6

    call multiply_by_addition

    li t0, 42
    bne a0, t0, fail_function


    # ------------------------------------------------------------
    # Test 6: indirect JALR
    # ------------------------------------------------------------

    la t0, indirect_target
    jalr ra, 0(t0)

    li t0, 42
    bne a0, t0, fail_jalr


    # ------------------------------------------------------------
    # Test 7: stack
    # ------------------------------------------------------------

    li t0, 0xDEADBEEF
    sw t0, 0(sp)

    lw t1, 0(sp)

    bne t0, t1, fail_stack


    # ------------------------------------------------------------
    # Test 8: branches
    # ------------------------------------------------------------

    li t0, 100
    li t1, 100

    beq t0, t1, branch_ok

    j fail_branch

branch_ok:


    # ------------------------------------------------------------
    # Test 9: x0 must remain zero
    # ------------------------------------------------------------

    li zero, 12345

    bne zero, zero, fail_x0


    # ------------------------------------------------------------
    # Everything passed
    # ------------------------------------------------------------

    lw ra, 28(sp)
    addi sp, sp, 32

    li a0, 0
    li a7, 93
    ecall


# ------------------------------------------------------------
# multiply_by_addition
#
# a0 = a
# a1 = b
# returns a0 = a * b
#
# RV32I only, no MUL.
# ------------------------------------------------------------

multiply_by_addition:
    addi sp, sp, -16

    sw ra, 12(sp)
    sw s0, 8(sp)

    addi s0, a0, 0

    addi a0, zero, 0

multiply_loop:
    beq a1, zero, multiply_done

    add a0, a0, s0
    addi a1, a1, -1

    j multiply_loop

multiply_done:
    lw s0, 8(sp)
    lw ra, 12(sp)

    addi sp, sp, 16

    jalr zero, 0(ra)


# ------------------------------------------------------------
# Indirect-call target
# ------------------------------------------------------------

indirect_target:
    li a0, 42
    jalr zero, 0(ra)


# ------------------------------------------------------------
# Failure handlers
#
# Each one returns a unique non-zero exit code.
# ------------------------------------------------------------

fail_data:
    li a0, 1
    j exit_failure

fail_bss:
    li a0, 2
    j exit_failure

fail_memory:
    li a0, 3
    j exit_failure

fail_arithmetic:
    li a0, 4
    j exit_failure

fail_function:
    li a0, 5
    j exit_failure

fail_jalr:
    li a0, 6
    j exit_failure

fail_stack:
    li a0, 7
    j exit_failure

fail_branch:
    li a0, 8
    j exit_failure

fail_x0:
    li a0, 9
    j exit_failure

exit_failure:
    li a7, 93
    ecall


.section .data

test_data:
    .word 0x12345678


.section .bss

.align 4

test_bss:
    .space 4