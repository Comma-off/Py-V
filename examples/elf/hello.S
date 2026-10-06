/*
 * hello.S -- the minimum viable program: exit through the host ABI.
 *
 * Assembles with either the bundled assembler or a real RISC-V toolchain:
 *     riscv32-linux-gnu-as -march=rv32i -mabi=ilp32 hello.S -o hello.o
 *     riscv32-linux-gnu-ld -m elf32lriscv -Ttext=0x10000 hello.o -o hello.elf
 *     python riscv.py exec hello.elf
 *
 * Expected: prints nothing, exits with status 0.
 */
        .text
        .globl  _start
_start:
        li      a0, 0                   /* exit status                      */
        li      a7, 93                  /* host ABI: 93 = exit              */
        ecall
