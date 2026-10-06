# hello.asm -- print a string through the host ECALL environment.
#
# Requires a host environment (riscv.host.HostEnvironment), which the CLI
# installs automatically:  python riscv.py run examples/hello.asm
#
# Expected output: "Hello, RISC-V!\n"
# Expected final state: a0 == 0 (exit code), CPU halted.

        .text
        .globl _start
_start:
        la      a1, message         # a1 = buffer address (auipc + addi)
        li      a2, message_len     # a2 = length
        li      a0, 1               # a0 = fd (1 = stdout)
        li      a7, 64              # a7 = syscall "write"
        ecall

        li      a0, 0               # exit status 0
        li      a7, 93              # a7 = syscall "exit"
        ecall

        .data
message:
        .string "Hello, RISC-V!\n"
        .equ    message_len, 15     # bytes, not counting the NUL
