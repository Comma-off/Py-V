/*
 * trap.S -- deliberately takes a trap, to prove the emulator reports it.
 *
 * Loads a word from a deliberately misaligned address, which RV32I defines as
 * a load-address-misaligned exception (mcause 4).  The program never reaches
 * its exit ECALL.
 *
 * Expected: no clean exit.  The emulator stops with
 *     error: load address misaligned (cause=4) at pc=...
 * and the CLI exits with status 3.
 */
        .text
        .globl  _start
_start:
        li      t0, 0x1002              /* deliberately not 4-byte aligned  */
        lw      t1, 0(t0)               /* traps here: mcause 4             */

        li      a0, 0                   /* never reached                    */
        li      a7, 93
        ecall
