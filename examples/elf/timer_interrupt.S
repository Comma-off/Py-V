/*
 * timer_interrupt.S -- a real machine timer interrupt, handled by the guest.
 *
 * Everything below runs on the emulated CPU.  The emulator delivers the
 * interrupt architecturally at an instruction boundary; the guest's own vector
 * table dispatches it, its handler reads the trap CSRs, resolves the timer
 * condition and executes MRET.  No Python callback takes part.
 *
 * mtvec is installed in VECTORED mode, so this also proves the two halves of
 * that rule at once:
 *
 *   - a synchronous exception vectors to BASE     (entry 0)
 *   - a timer interrupt vectors to BASE + 4 * 7   (entry 7)
 *
 * The timer registers are memory-mapped CLINT-style, as they are on real
 * hardware -- there is no mtimecmp CSR:
 *
 *   0x02004000  mtimecmp  (64-bit)
 *   0x0200BFF8  mtime     (64-bit)
 *
 * Build (needs Zicsr):
 *   riscv32-linux-gnu-as -march=rv32i_zicsr -mabi=ilp32 -mno-relax \
 *       timer_interrupt.S -o timer_interrupt.o
 *   riscv32-linux-gnu-ld -m elf32lriscv -Ttext=0x10000 --no-relax \
 *       timer_interrupt.o -o timer_interrupt.elf
 *   python riscv.py exec timer_interrupt.elf
 *
 * Expected final state: exit status 0, and
 *   s0 == 3            the timer handler ran three times
 *   s1 == 1            the exception handler ran once
 *   s2 == 0x80000007   mcause: interrupt bit set, cause 7 (machine timer)
 *   s4 == 0x00000080   mip at handler entry: MTIP asserted
 *   s5 == 0x00000000   mip after re-arming: MTIP resolved
 *   s6 == 0x00001880   mstatus in the handler: MPP=3, MPIE=1, MIE=0
 *   s7 == 0x00000002   mcause seen by the exception handler
 *   s3 != 0            mepc was the interrupted PC
 */

        .equ    CLINT_MTIMECMP, 0x02004000
        .equ    CLINT_MTIME,    0x0200BFF8
        .equ    MSTATUS_MIE,    0x008           /* mstatus.MIE  */
        .equ    MIE_MTIE,       0x080           /* mie.MTIE     */
        .equ    TICK_INTERVAL,  60

        .text
        .globl  _start
_start:
        /* ---- install the vector table in VECTORED mode (MODE = 1) ---- */
        la      t0, vector_table
        ori     t0, t0, 1
        csrw    mtvec, t0

        li      s0, 0                           /* timer interrupts     */
        li      s1, 0                           /* exceptions           */
        li      s8, 0                           /* progress counter     */

        /* ---- a synchronous exception first, with interrupts still off.
                In vectored mode this must land at BASE, not BASE + 4*2. ---- */
        .word   0x00000000                      /* illegal instruction  */

        /* ---- arm the timer, then enable it ---- */
        jal     ra, rearm_timer

        li      t0, MIE_MTIE
        csrs    mie, t0                         /* enable the timer source */
        li      t0, MSTATUS_MIE
        csrs    mstatus, t0                     /* enable machine interrupts */

        /* ---- do work until the timer has fired three times ---- */
wait_loop:
        addi    s8, s8, 1
        li      t0, 3
        blt     s0, t0, wait_loop

        /* ---- done: turn interrupts off and exit through the host ---- */
        li      t0, MSTATUS_MIE
        csrc    mstatus, t0
        li      a0, 0
        li      a7, 93
        ecall

/* ------------------------------------------------------------------------
 * rearm_timer: mtimecmp = mtime + TICK_INTERVAL, as a full 64-bit value.
 *
 * Both halves matter.  Reading a 64-bit counter through two 32-bit loads can
 * tear across a carry, so the high word is re-read and the pair retried; and
 * the comparator is written low-first-as-all-ones so that no intermediate
 * value can briefly look like a deadline that has already passed.
 * Clobbers t0-t4.  Returns nothing.
 * ---------------------------------------------------------------------- */
rearm_timer:
        li      t0, CLINT_MTIME
read_mtime:
        lw      t2, 4(t0)                       /* high                 */
        lw      t1, 0(t0)                       /* low                  */
        lw      t3, 4(t0)                       /* high again           */
        bne     t2, t3, read_mtime              /* carried? read again  */

        /* 64-bit add of TICK_INTERVAL */
        li      t4, TICK_INTERVAL
        add     t1, t1, t4                      /* low + interval       */
        sltu    t4, t1, t4                      /* carry out            */
        add     t2, t2, t4                      /* high + carry         */

        /* Write it back without ever creating a spurious deadline. */
        li      t0, CLINT_MTIMECMP
        li      t3, -1
        sw      t3, 0(t0)                       /* low  = all ones      */
        sw      t2, 4(t0)                       /* high = final value   */
        sw      t1, 0(t0)                       /* low  = final value   */
        ret

/* ------------------------------------------------------------------------
 * The vector table.  In vectored mode mtvec's BASE is the table itself and
 * interrupt cause N enters at BASE + 4*N, so every entry must be exactly one
 * instruction: a jump to the real handler.  Synchronous exceptions ignore the
 * offset entirely and enter at BASE.
 * ---------------------------------------------------------------------- */
        .align  2
vector_table:
        j       exception_handler               /*  0: exceptions       */
        j       unexpected                      /*  1                   */
        j       unexpected                      /*  2                   */
        j       unexpected                      /*  3: machine software */
        j       unexpected                      /*  4                   */
        j       unexpected                      /*  5                   */
        j       unexpected                      /*  6                   */
        j       timer_handler                   /*  7: machine timer    */
        j       unexpected                      /*  8                   */
        j       unexpected                      /*  9                   */
        j       unexpected                      /* 10                   */
        j       unexpected                      /* 11: machine external */

/* ---- machine timer interrupt (cause 7) -------------------------------- */
timer_handler:
        addi    s0, s0, 1                       /* count this interrupt */
        csrr    s2, mcause                      /* 0x80000007           */
        csrr    s3, mepc                        /* the interrupted PC   */
        csrr    s4, mip                         /* MTIP is asserted     */
        csrr    s6, mstatus                     /* MIE off, MPIE on     */

        jal     ra, rearm_timer                 /* resolve the condition */
        csrr    s5, mip                         /* ...MTIP now clear    */

        /* mepc already points at the instruction that has not run yet, so
           unlike an exception handler there is nothing to advance. */
        mret

/* ---- synchronous exception (entered at BASE) -------------------------- */
exception_handler:
        addi    s1, s1, 1
        csrr    s7, mcause                      /* 2: illegal instruction */
        csrr    t0, mepc
        addi    t0, t0, 4                       /* step over the fault  */
        csrw    mepc, t0
        mret

/* ---- anything else is a bug in the emulator or this program ----------- */
unexpected:
        li      a0, 1
        li      a7, 93
        ecall
