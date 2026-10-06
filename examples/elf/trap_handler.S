/*
 * trap_handler.S -- a guest-defined machine-mode trap handler.
 *
 * This is the milestone's headline program: everything below runs on the
 * emulated CPU.  There is no Python callback anywhere in the loop -- the
 * emulator takes the trap architecturally, the guest's own handler reads the
 * trap CSRs, adjusts mepc, and MRET returns into the interrupted code.
 *
 *   set mtvec -> fault -> handler -> read mcause/mepc/mtval -> mepc += 4
 *             -> MRET -> execution resumes after the faulting instruction
 *
 * Three different exceptions are raised, so the handler is exercised against
 * three different causes and trap values.
 *
 * Build (needs Zicsr):
 *   riscv32-linux-gnu-as -march=rv32i_zicsr -mabi=ilp32 -mno-relax \
 *       trap_handler.S -o trap_handler.o
 *   riscv32-linux-gnu-ld -m elf32lriscv -Ttext=0x10000 --no-relax \
 *       trap_handler.o -o trap_handler.elf
 *   python riscv.py exec trap_handler.elf
 *
 * Expected final state: exit status 0, and
 *   s0 == 3            the handler ran three times
 *   s1 == 2            first  cause: illegal instruction
 *   s2 == 4            second cause: load address misaligned
 *   s3 == 6            third  cause: store address misaligned
 *   s4 == 0x00001001   mtval from the misaligned load (the bad address)
 *   s5 == 0x000001AA   0xA + 0xA0 + 0x100: execution resumed after all three
 *   s6 == 0x00000008   mstatus.MPIE seen inside the handler (MIE had been 1)
 *   s7 == 0x00001800   mstatus.MPP == 3 (machine mode) inside the handler
 */
        .text
        .globl  _start
_start:
        la      t0, trap_handler
        csrw    mtvec, t0               /* install the handler (direct mode) */

        li      s0, 0                   /* handler entry count               */
        li      s5, 0                   /* progress marker                   */

        /* Enable machine interrupts, so the handler can observe MPIE later.
           Nothing raises an interrupt here; this is about mstatus bookkeeping. */
        li      t0, 8                   /* mstatus.MIE                       */
        csrs    mstatus, t0

/* ---- exception 1: an illegal instruction ------------------------------- */
        .word   0x00000000              /* not a valid encoding -> cause 2   */
        addi    s5, s5, 0xA             /* reached only if MRET came back    */

/* ---- exception 2: a misaligned load ------------------------------------ */
        li      t0, 0x1001              /* deliberately not 4-byte aligned   */
        lw      t1, 0(t0)               /* -> cause 4, mtval = 0x1001        */
        addi    s5, s5, 0xA0

/* ---- exception 3: a misaligned store ----------------------------------- */
        li      t0, 0x2002
        sw      t1, 0(t0)               /* -> cause 6, mtval = 0x2002        */
        addi    s5, s5, 0x100           /* s5 now 0xA + 0xA0 + 0x100 == 0x1AA */

        li      a0, 0
        li      a7, 93
        ecall                           /* host syscall: exit 0              */

/* ------------------------------------------------------------------------
 * The trap handler.  Entered with the machine's trap CSRs already set:
 *   mepc   = address of the faulting instruction
 *   mcause = why we are here
 *   mtval  = the fault-specific value
 *   mstatus.MPIE = the MIE we interrupted, MIE = 0, MPP = the old privilege
 * ---------------------------------------------------------------------- */
        .align  2                       /* mtvec BASE must be 4-byte aligned */
trap_handler:
        addi    s0, s0, 1               /* count this entry                  */

        csrr    t2, mcause              /* what happened?                    */
        csrr    t3, mepc                /* where?                            */
        csrr    t4, mtval               /* with what value?                  */
        csrr    t5, mstatus

        /* Record the cause of each entry in a different register. */
        li      t6, 1
        beq     s0, t6, record_first
        li      t6, 2
        beq     s0, t6, record_second
        j       record_third
record_first:
        mv      s1, t2                  /* first cause                       */
        li      t6, 0x80
        and     s6, t5, t6              /* mstatus.MPIE (bit 7)              */
        srli    s6, s6, 4               /* report it in bit 3: 0x80 -> 0x08  */
        li      t6, 0x1800
        and     s7, t5, t6              /* mstatus.MPP (bits 12:11)          */
        j       resume
record_second:
        mv      s2, t2
        mv      s4, t4                  /* mtval of the misaligned load      */
        j       resume
record_third:
        mv      s3, t2

resume:
        /* Skip the faulting instruction and return.  mepc points *at* it, so
           the handler must advance past it or the fault repeats forever. */
        addi    t3, t3, 4
        csrw    mepc, t3
        mret
