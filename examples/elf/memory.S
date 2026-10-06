/*
 * memory.S -- .data, .bss, the stack, and loads/stores at every width.
 *
 * This is the program that proves the loader does its job: `initialised` comes
 * from the file image, `zeroed` comes from the p_memsz > p_filesz tail that the
 * loader must zero-fill, and `scratch` is written at run time.
 *
 * Expected: exit status 0.
 *   s0 == 0x12345678   (initialised global, loaded from the file)
 *   s1 == 0            (.bss global, zero-filled by the loader)
 *   s2 == 0xCAFEBABE   (stored then reloaded from .bss)
 *   s3 == 0xFFFFFFFF   (lb of a stored 0xFF: sign-extended)
 *   s4 == 0x000000FF   (lbu of the same byte: zero-extended)
 *   s5 == 0x00005678   (lhu of the low half of the initialised global)
 *   s6 == 0x0000BEEF   (a stack round-trip)
 */
        .text
        .globl  _start
_start:
        la      t0, initialised
        lw      s0, 0(t0)               /* 0x12345678 from the file image   */
        lhu     s5, 0(t0)               /* 0x5678                           */

        la      t1, zeroed
        lw      s1, 0(t1)               /* 0: the loader zeroed .bss        */

        li      t2, 0xCAFEBABE
        sw      t2, 0(t1)
        lw      s2, 0(t1)               /* round-trip through .bss          */

        la      t3, scratch
        li      t4, 0xFF
        sb      t4, 0(t3)
        lb      s3, 0(t3)               /* sign-extended  -> 0xFFFFFFFF     */
        lbu     s4, 0(t3)               /* zero-extended  -> 0x000000FF     */

        /* A stack variable: push, reload, pop. */
        addi    sp, sp, -16
        li      t5, 0xBEEF
        sw      t5, 8(sp)
        lw      s6, 8(sp)
        addi    sp, sp, 16

        li      a0, 0
        li      a7, 93
        ecall

        .data
initialised:
        .word   0x12345678

        .bss
        .align  2
zeroed:
        .space  4
scratch:
        .space  4
