"""RV32C tests: compressed-instruction expansion, fetch/step, and tracing.

The ``TOOLCHAIN`` table is the heart of this file: 192 compressed halfwords,
each paired with the 32-bit word GNU ``as`` emits for the equivalent
full-width instruction (``c.addi s0,20`` next to ``addi s0,s0,20``), captured
once from ``riscv64-linux-gnu-as``/``objdump`` and frozen here so the test
needs no toolchain to run. :func:`riscv.rv32c.expand` must reproduce every one
of those 32-bit words exactly -- that is the same "GCC/GNU as is the oracle"
strategy :mod:`tests.test_rv32m` uses for multiply/divide, applied to the
trickier problem of compressed instructions' scrambled immediate encodings.
"""

import unittest

from tests.helpers import CPU, EmulatorTestCase, SCRATCH

from riscv.exceptions import IllegalInstruction
from riscv.isa import available_isas, get_isa
from riscv.rv32c import RV32IC, expand
from riscv.rv32m import RV32IM

MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# 192 compressed/uncompressed pairs straight from riscv64-linux-gnu-as.
# (compressed_half, uncompressed_word, compressed_text, uncompressed_text)
# ---------------------------------------------------------------------------
TOOLCHAIN = [
    (0x003c, 0x00810793, 'c.addi4spn a5, sp, 8', 'addi a5, sp, 8'),
    (0x0030, 0x00810613, 'c.addi4spn a2, sp, 8', 'addi a2, sp, 8'),
    (0x1fb8, 0x3F810713, 'c.addi4spn a4, sp, 1016', 'addi a4, sp, 1016'),
    (0x10d0, 0x06410613, 'c.addi4spn a2, sp, 100', 'addi a2, sp, 100'),
    (0x1fec, 0x3FC10593, 'c.addi4spn a1, sp, 1020', 'addi a1, sp, 1020'),
    (0x081c, 0x01010793, 'c.addi4spn a5, sp, 16', 'addi a5, sp, 16'),
    (0x10d8, 0x06410713, 'c.addi4spn a4, sp, 100', 'addi a4, sp, 100'),
    (0x080c, 0x01010593, 'c.addi4spn a1, sp, 16', 'addi a1, sp, 16'),
    (0x50b4, 0x0604A683, 'c.lw a3, 96(s1)', 'lw a3, 96(s1)'),
    (0xd0b4, 0x06D4A023, 'c.sw a3, 96(s1)', 'sw a3, 96(s1)'),
    (0x53ac, 0x0607A583, 'c.lw a1, 96(a5)', 'lw a1, 96(a5)'),
    (0xd3ac, 0x06B7A023, 'c.sw a1, 96(a5)', 'sw a1, 96(a5)'),
    (0x51c8, 0x0245A503, 'c.lw a0, 36(a1)', 'lw a0, 36(a1)'),
    (0xd1c8, 0x02A5A223, 'c.sw a0, 36(a1)', 'sw a0, 36(a1)'),
    (0x52a0, 0x0606A403, 'c.lw s0, 96(a3)', 'lw s0, 96(a3)'),
    (0xd2a0, 0x0686A023, 'c.sw s0, 96(a3)', 'sw s0, 96(a3)'),
    (0x4398, 0x0007A703, 'c.lw a4, 0(a5)', 'lw a4, 0(a5)'),
    (0xc398, 0x00E7A023, 'c.sw a4, 0(a5)', 'sw a4, 0(a5)'),
    (0x52c8, 0x0246A503, 'c.lw a0, 36(a3)', 'lw a0, 36(a3)'),
    (0xd2c8, 0x02A6A223, 'c.sw a0, 36(a3)', 'sw a0, 36(a3)'),
    (0x42d4, 0x0046A683, 'c.lw a3, 4(a3)', 'lw a3, 4(a3)'),
    (0xc2d4, 0x00D6A223, 'c.sw a3, 4(a3)', 'sw a3, 4(a3)'),
    (0x5354, 0x02472683, 'c.lw a3, 36(a4)', 'lw a3, 36(a4)'),
    (0xd354, 0x02D72223, 'c.sw a3, 36(a4)', 'sw a3, 36(a4)'),
    (0x51d4, 0x0245A683, 'c.lw a3, 36(a1)', 'lw a3, 36(a1)'),
    (0xd1d4, 0x02D5A223, 'c.sw a3, 36(a1)', 'sw a3, 36(a1)'),
    (0x418c, 0x0005A583, 'c.lw a1, 0(a1)', 'lw a1, 0(a1)'),
    (0xc18c, 0x00B5A023, 'c.sw a1, 0(a1)', 'sw a1, 0(a1)'),
    (0x1c6d, 0xFFBC0C13, 'c.addi s8, -5', 'addi s8, s8, -5'),
    (0x0c85, 0x001C8C93, 'c.addi s9, 1', 'addi s9, s9, 1'),
    (0x0e7d, 0x01FE0E13, 'c.addi t3, 31', 'addi t3, t3, 31'),
    (0x1e01, 0xFE0E0E13, 'c.addi t3, -32', 'addi t3, t3, -32'),
    (0x0cd1, 0x014C8C93, 'c.addi s9, 20', 'addi s9, s9, 20'),
    (0x1a7d, 0xFFFA0A13, 'c.addi s4, -1', 'addi s4, s4, -1'),
    (0x15ed, 0xFFB58593, 'c.addi a1, -5', 'addi a1, a1, -5'),
    (0x143d, 0xFEF40413, 'c.addi s0, -17', 'addi s0, s0, -17'),
    (0x0615, 0x00560613, 'c.addi a2, 5', 'addi a2, a2, 5'),
    (0x1381, 0xFE038393, 'c.addi t2, -32', 'addi t2, t2, -32'),
    (0x0001, 0x00000013, 'c.nop', 'addi zero, zero, 0'),
    (0x6141, 0x01010113, 'c.addi16sp sp, 16', 'addi sp, sp, 16'),
    (0x717d, 0xFF010113, 'c.addi16sp sp, -16', 'addi sp, sp, -16'),
    (0x6105, 0x02010113, 'c.addi16sp sp, 32', 'addi sp, sp, 32'),
    (0x713d, 0xFE010113, 'c.addi16sp sp, -32', 'addi sp, sp, -32'),
    (0x6165, 0x07010113, 'c.addi16sp sp, 112', 'addi sp, sp, 112'),
    (0x7101, 0xE0010113, 'c.addi16sp sp, -512', 'addi sp, sp, -512'),
    (0x617d, 0x1F010113, 'c.addi16sp sp, 496', 'addi sp, sp, 496'),
    (0x6149, 0x09010113, 'c.addi16sp sp, 144', 'addi sp, sp, 144'),
    (0x5801, 0xFE000813, 'c.li a6, -32', 'addi a6, zero, -32'),
    (0x4dd1, 0x01400D93, 'c.li s11, 20', 'addi s11, zero, 20'),
    (0x52fd, 0xFFF00293, 'c.li t0, -1', 'addi t0, zero, -1'),
    (0x5681, 0xFE000693, 'c.li a3, -32', 'addi a3, zero, -32'),
    (0x55fd, 0xFFF00593, 'c.li a1, -1', 'addi a1, zero, -1'),
    (0x4fd1, 0x01400F93, 'c.li t6, 20', 'addi t6, zero, 20'),
    (0x4bfd, 0x01F00B93, 'c.li s7, 31', 'addi s7, zero, 31'),
    (0x4e05, 0x00100E13, 'c.li t3, 1', 'addi t3, zero, 1'),
    (0x4b51, 0x01400B13, 'c.li s6, 20', 'addi s6, zero, 20'),
    (0x5cbd, 0xFEF00C93, 'c.li s9, -17', 'addi s9, zero, -17'),
    (0x6315, 0x00005337, 'c.lui t1, 5', 'lui t1, 5'),
    (0x6ffd, 0x0001FFB7, 'c.lui t6, 31', 'lui t6, 31'),
    (0x6a05, 0x00001A37, 'c.lui s4, 1', 'lui s4, 1'),
    (0x62fd, 0x0001F2B7, 'c.lui t0, 31', 'lui t0, 31'),
    (0x60fd, 0x0001F0B7, 'c.lui ra, 31', 'lui ra, 31'),
    (0x687d, 0x0001F837, 'c.lui a6, 31', 'lui a6, 31'),
    (0x66fd, 0x0001F6B7, 'c.lui a3, 31', 'lui a3, 31'),
    (0x627d, 0x0001F237, 'c.lui tp, 31', 'lui tp, 31'),
    (0x8215, 0x00565613, 'c.srli a2, 5', 'srli a2, a2, 5'),
    (0x8615, 0x40565613, 'c.srai a2, 5', 'srai a2, a2, 5'),
    (0x9a31, 0xFEC67613, 'c.andi a2, -20', 'andi a2, a2, -20'),
    (0x8241, 0x01065613, 'c.srli a2, 16', 'srli a2, a2, 16'),
    (0x8641, 0x41065613, 'c.srai a2, 16', 'srai a2, a2, 16'),
    (0x8a7d, 0x01F67613, 'c.andi a2, 31', 'andi a2, a2, 31'),
    (0x8341, 0x01075713, 'c.srli a4, 16', 'srli a4, a4, 16'),
    (0x8741, 0x41075713, 'c.srai a4, 16', 'srai a4, a4, 16'),
    (0x9b7d, 0xFFF77713, 'c.andi a4, -1', 'andi a4, a4, -1'),
    (0x80fd, 0x01F4D493, 'c.srli s1, 31', 'srli s1, s1, 31'),
    (0x84fd, 0x41F4D493, 'c.srai s1, 31', 'srai s1, s1, 31'),
    (0x88d1, 0x0144F493, 'c.andi s1, 20', 'andi s1, s1, 20'),
    (0x821d, 0x00765613, 'c.srli a2, 7', 'srli a2, a2, 7'),
    (0x861d, 0x40765613, 'c.srai a2, 7', 'srai a2, a2, 7'),
    (0x8a05, 0x00167613, 'c.andi a2, 1', 'andi a2, a2, 1'),
    (0x8395, 0x0057D793, 'c.srli a5, 5', 'srli a5, a5, 5'),
    (0x8795, 0x4057D793, 'c.srai a5, 5', 'srai a5, a5, 5'),
    (0x9bfd, 0xFFF7F793, 'c.andi a5, -1', 'andi a5, a5, -1'),
    (0x8315, 0x00575713, 'c.srli a4, 5', 'srli a4, a4, 5'),
    (0x8715, 0x40575713, 'c.srai a4, 5', 'srai a4, a4, 5'),
    (0x8b7d, 0x01F77713, 'c.andi a4, 31', 'andi a4, a4, 31'),
    (0x8d95, 0x40D585B3, 'c.sub a1, a3', 'sub a1, a1, a3'),
    (0x8db5, 0x00D5C5B3, 'c.xor a1, a3', 'xor a1, a1, a3'),
    (0x8dd5, 0x00D5E5B3, 'c.or a1, a3', 'or a1, a1, a3'),
    (0x8df5, 0x00D5F5B3, 'c.and a1, a3', 'and a1, a1, a3'),
    (0x8e81, 0x408686B3, 'c.sub a3, s0', 'sub a3, a3, s0'),
    (0x8ea1, 0x0086C6B3, 'c.xor a3, s0', 'xor a3, a3, s0'),
    (0x8ec1, 0x0086E6B3, 'c.or a3, s0', 'or a3, a3, s0'),
    (0x8ee1, 0x0086F6B3, 'c.and a3, s0', 'and a3, a3, s0'),
    (0x8f05, 0x40970733, 'c.sub a4, s1', 'sub a4, a4, s1'),
    (0x8f25, 0x00974733, 'c.xor a4, s1', 'xor a4, a4, s1'),
    (0x8f45, 0x00976733, 'c.or a4, s1', 'or a4, a4, s1'),
    (0x8f65, 0x00977733, 'c.and a4, s1', 'and a4, a4, s1'),
    (0x8c8d, 0x40B484B3, 'c.sub s1, a1', 'sub s1, s1, a1'),
    (0x8cad, 0x00B4C4B3, 'c.xor s1, a1', 'xor s1, s1, a1'),
    (0x8ccd, 0x00B4E4B3, 'c.or s1, a1', 'or s1, s1, a1'),
    (0x8ced, 0x00B4F4B3, 'c.and s1, a1', 'and s1, s1, a1'),
    (0x8c09, 0x40A40433, 'c.sub s0, a0', 'sub s0, s0, a0'),
    (0x8c29, 0x00A44433, 'c.xor s0, a0', 'xor s0, s0, a0'),
    (0x8c49, 0x00A46433, 'c.or s0, a0', 'or s0, s0, a0'),
    (0x8c69, 0x00A47433, 'c.and s0, a0', 'and s0, s0, a0'),
    (0x8d85, 0x409585B3, 'c.sub a1, s1', 'sub a1, a1, s1'),
    (0x8da5, 0x0095C5B3, 'c.xor a1, s1', 'xor a1, a1, s1'),
    (0x8dc5, 0x0095E5B3, 'c.or a1, s1', 'or a1, a1, s1'),
    (0x8de5, 0x0095F5B3, 'c.and a1, s1', 'and a1, a1, s1'),
    (0x8d0d, 0x40B50533, 'c.sub a0, a1', 'sub a0, a0, a1'),
    (0x8d2d, 0x00B54533, 'c.xor a0, a1', 'xor a0, a0, a1'),
    (0x8d4d, 0x00B56533, 'c.or a0, a1', 'or a0, a0, a1'),
    (0x8d6d, 0x00B57533, 'c.and a0, a1', 'and a0, a0, a1'),
    (0x8f89, 0x40A787B3, 'c.sub a5, a0', 'sub a5, a5, a0'),
    (0x8fa9, 0x00A7C7B3, 'c.xor a5, a0', 'xor a5, a5, a0'),
    (0x8fc9, 0x00A7E7B3, 'c.or a5, a0', 'or a5, a5, a0'),
    (0x8fe9, 0x00A7F7B3, 'c.and a5, a0', 'and a5, a5, a0'),
    (0x0306, 0x00131313, 'c.slli t1, 1', 'slli t1, t1, 1'),
    (0x029e, 0x00729293, 'c.slli t0, 7', 'slli t0, t0, 7'),
    (0x021e, 0x00721213, 'c.slli tp, 7', 'slli tp, tp, 7'),
    (0x091e, 0x00791913, 'c.slli s2, 7', 'slli s2, s2, 7'),
    (0x0a86, 0x001A9A93, 'c.slli s5, 1', 'slli s5, s5, 1'),
    (0x0a06, 0x001A1A13, 'c.slli s4, 1', 'slli s4, s4, 1'),
    (0x0916, 0x00591913, 'c.slli s2, 5', 'slli s2, s2, 5'),
    (0x097e, 0x01F91913, 'c.slli s2, 31', 'slli s2, s2, 31'),
    (0x0086, 0x00109093, 'c.slli ra, 1', 'slli ra, ra, 1'),
    (0x0486, 0x00149493, 'c.slli s1, 1', 'slli s1, s1, 1'),
    (0x4302, 0x00012303, 'c.lwsp t1, 0(sp)', 'lw t1, 0(sp)'),
    (0x5216, 0x06412203, 'c.lwsp tp, 100(sp)', 'lw tp, 100(sp)'),
    (0x5e7e, 0x0FC12E03, 'c.lwsp t3, 252(sp)', 'lw t3, 252(sp)'),
    (0x5b7e, 0x0FC12B03, 'c.lwsp s6, 252(sp)', 'lw s6, 252(sp)'),
    (0x4c12, 0x00412C03, 'c.lwsp s8, 4(sp)', 'lw s8, 4(sp)'),
    (0x4992, 0x00412983, 'c.lwsp s3, 4(sp)', 'lw s3, 4(sp)'),
    (0x4a92, 0x00412A83, 'c.lwsp s5, 4(sp)', 'lw s5, 4(sp)'),
    (0x4792, 0x00412783, 'c.lwsp a5, 4(sp)', 'lw a5, 4(sp)'),
    (0x4842, 0x01012803, 'c.lwsp a6, 16(sp)', 'lw a6, 16(sp)'),
    (0x5496, 0x06412483, 'c.lwsp s1, 100(sp)', 'lw s1, 100(sp)'),
    (0xd2be, 0x06F12223, 'c.swsp a5, 100(sp)', 'sw a5, 100(sp)'),
    (0xc26a, 0x01A12223, 'c.swsp s10, 4(sp)', 'sw s10, 4(sp)'),
    (0xdfb6, 0x0ED12E23, 'c.swsp a3, 252(sp)', 'sw a3, 252(sp)'),
    (0xde72, 0x03C12E23, 'c.swsp t3, 60(sp)', 'sw t3, 60(sp)'),
    (0xd2d2, 0x07412223, 'c.swsp s4, 100(sp)', 'sw s4, 100(sp)'),
    (0xc216, 0x00512223, 'c.swsp t0, 4(sp)', 'sw t0, 4(sp)'),
    (0xc81a, 0x00612823, 'c.swsp t1, 16(sp)', 'sw t1, 16(sp)'),
    (0xc236, 0x00D12223, 'c.swsp a3, 4(sp)', 'sw a3, 4(sp)'),
    (0x8c82, 0x000C8067, 'c.jr s9', 'jalr zero, s9, 0'),
    (0x9c82, 0x000C80E7, 'c.jalr s9', 'jalr ra, s9, 0'),
    (0x8302, 0x00030067, 'c.jr t1', 'jalr zero, t1, 0'),
    (0x9302, 0x000300E7, 'c.jalr t1', 'jalr ra, t1, 0'),
    (0x8602, 0x00060067, 'c.jr a2', 'jalr zero, a2, 0'),
    (0x9602, 0x000600E7, 'c.jalr a2', 'jalr ra, a2, 0'),
    (0x8782, 0x00078067, 'c.jr a5', 'jalr zero, a5, 0'),
    (0x9782, 0x000780E7, 'c.jalr a5', 'jalr ra, a5, 0'),
    (0x8b02, 0x000B0067, 'c.jr s6', 'jalr zero, s6, 0'),
    (0x9b02, 0x000B00E7, 'c.jalr s6', 'jalr ra, s6, 0'),
    (0x8082, 0x00008067, 'c.jr ra', 'jalr zero, ra, 0'),
    (0x9082, 0x000080E7, 'c.jalr ra', 'jalr ra, ra, 0'),
    (0x8202, 0x00020067, 'c.jr tp', 'jalr zero, tp, 0'),
    (0x9202, 0x000200E7, 'c.jalr tp', 'jalr ra, tp, 0'),
    (0x8582, 0x00058067, 'c.jr a1', 'jalr zero, a1, 0'),
    (0x9582, 0x000580E7, 'c.jalr a1', 'jalr ra, a1, 0'),
    (0x8ca2, 0x00800CB3, 'c.mv s9, s0', 'add s9, zero, s0'),
    (0x9ca2, 0x008C8CB3, 'c.add s9, s0', 'add s9, s9, s0'),
    (0x8cc6, 0x01100CB3, 'c.mv s9, a7', 'add s9, zero, a7'),
    (0x9cc6, 0x011C8CB3, 'c.add s9, a7', 'add s9, s9, a7'),
    (0x875a, 0x01600733, 'c.mv a4, s6', 'add a4, zero, s6'),
    (0x975a, 0x01670733, 'c.add a4, s6', 'add a4, a4, s6'),
    (0x811e, 0x00700133, 'c.mv sp, t2', 'add sp, zero, t2'),
    (0x911e, 0x00710133, 'c.add sp, t2', 'add sp, sp, t2'),
    (0x8a3e, 0x00F00A33, 'c.mv s4, a5', 'add s4, zero, a5'),
    (0x9a3e, 0x00FA0A33, 'c.add s4, a5', 'add s4, s4, a5'),
    (0x8942, 0x01000933, 'c.mv s2, a6', 'add s2, zero, a6'),
    (0x9942, 0x01090933, 'c.add s2, a6', 'add s2, s2, a6'),
    (0x896a, 0x01A00933, 'c.mv s2, s10', 'add s2, zero, s10'),
    (0x996a, 0x01A90933, 'c.add s2, s10', 'add s2, s2, s10'),
    (0x813a, 0x00E00133, 'c.mv sp, a4', 'add sp, zero, a4'),
    (0x913a, 0x00E10133, 'c.add sp, a4', 'add sp, sp, a4'),
    (0x8586, 0x001005B3, 'c.mv a1, ra', 'add a1, zero, ra'),
    (0x9586, 0x001585B3, 'c.add a1, ra', 'add a1, a1, ra'),
    (0x88ae, 0x00B008B3, 'c.mv a7, a1', 'add a7, zero, a1'),
    (0x98ae, 0x00B888B3, 'c.add a7, a1', 'add a7, a7, a1'),
    (0x9002, 0x00100073, 'c.ebreak', 'ebreak'),
]

#: C.J / C.JAL / C.BEQZ / C.BNEZ: the one bit-scramble (CJ/CB format) the table
#: above doesn't exercise, because a PC-relative immediate needs real
#: addresses, not just an operand list. These four are hand-verified against
#: the same toolchain's objdump output (see riscv/rv32c.py's module docstring
#: derivation): each is the real encoding of a 2-byte ``c.j``/``c.jal``/
#: ``c.beqz``/``c.bnez`` whose target is ``address + offset``.
BRANCH_TOOLCHAIN = [
    # (half, address, offset, base_name, base_kwargs)
    (0xb7c5, 0x20, -32, "JAL", {"rd": 0}),
    (0xdc79, 0x22, -34, "BEQ", {"rs1": 8, "rs2": 0}),
    (0xfc71, 0x24, -36, "BNE", {"rs1": 8, "rs2": 0}),
    (0x3fe9, 0x26, -38, "JAL", {"rd": 1}),
]


class TestExpandAgainstToolchain(unittest.TestCase):
    def test_192_pairs_match_gnu_as(self):
        """rv32c: 192 compressed/uncompressed pairs from riscv64-linux-gnu-as agree"""
        mismatches = []
        for half, uword, ctext, utext in TOOLCHAIN:
            try:
                word, _name = expand(half)
            except IllegalInstruction:
                mismatches.append(f"{ctext!r} (0x{half:04X}): raised IllegalInstruction")
                continue
            if word != uword:
                mismatches.append(
                    f"{ctext!r} (0x{half:04X}): got 0x{word:08X}, expected 0x{uword:08X} ({utext!r})"
                )
        self.assertEqual(mismatches, [])

    def test_branch_and_jump_offsets(self):
        """rv32c: C.J/C.JAL/C.BEQZ/C.BNEZ's scrambled PC-relative immediate"""
        from riscv.encoder import encode
        from riscv.instructions import RV32I

        for half, address, offset, name, kwargs in BRANCH_TOOLCHAIN:
            word, _name = expand(half, pc=address)
            expected = encode(name, imm=offset, instruction_set=RV32I, **kwargs)
            self.assertEqual(word, expected, f"0x{half:04X} at 0x{address:X}")

    def test_mnemonics_are_canonical(self):
        """rv32c: expand() reports the real C.* name for every table entry"""
        for half, _uword, ctext, _utext in TOOLCHAIN:
            _word, name = expand(half)
            expected = ctext.split(None, 1)[0].split(",")[0].upper()
            self.assertEqual(name, expected, f"0x{half:04X}")


# ---------------------------------------------------------------------------
# Reserved encodings: must trap, not silently misdecode
# ---------------------------------------------------------------------------
class TestReservedEncodings(unittest.TestCase):
    RESERVED = {
        "C.ADDI4SPN all-zero": 0x0000,
        "C.LUI nzimm=0 (rd=1, not 0 or 2)": 0x6081,
        "quadrant0 funct3=100": 0x8000,
        "C.LWSP rd=0": 0x4002,
        "C.JR rd=0 (and rs2=0)": 0x8002,
        "C.SRLI shamt[5]=1 (RV64-only)": 0x9015,
        "C.SRAI shamt[5]=1 (RV64-only)": 0x9415,
        "RV64-only C.SUBW/C.ADDW slot": 0x9C01,
    }

    def test_reserved_encodings_trap(self):
        """rv32c: encodings the spec reserves raise IllegalInstruction, not a misdecode"""
        failures = []
        for label, half in self.RESERVED.items():
            try:
                word, name = expand(half)
            except IllegalInstruction:
                continue
            failures.append(f"{label} (0x{half:04X}): decoded as {name} -> 0x{word:08X}")
        self.assertEqual(failures, [])

    def test_float_only_quadrants_are_illegal(self):
        """rv32c: C.FLD/C.FLW/C.FSDSP/C.FSWSP (D/F-only) are illegal with no D/F"""
        for half in (0x2000, 0x6000, 0xA002, 0xE002):
            self.assertRaises(IllegalInstruction, expand, half)

    def test_hints_are_legal_not_reserved(self):
        """rv32c: C.NOP, C.LI x0, C.MV x0,.., a zero-imm C.ADDI are HINTs, not illegal"""
        for half in (0x0001, 0x4001, 0x8002 | (0 << 7) | (1 << 2), 0x0081):
            expand(half)  # must not raise


# ---------------------------------------------------------------------------
# CPU integration: fetch/step with a mixed 2-/4-byte instruction stream
# ---------------------------------------------------------------------------
class TestCPUIntegration(EmulatorTestCase):
    def test_isa_names_registered(self):
        """rv32c: all eight combinations are registered, and add no new mnemonics"""
        for name in (
            "rv32ic", "rv32imc", "rv32iac", "rv32imac",
            "rv32ic_zicsr", "rv32imc_zicsr", "rv32iac_zicsr", "rv32imac_zicsr",
        ):
            self.assertIn(name, available_isas())
        self.assertTrue(get_isa("rv32imac").compressed)
        self.assertFalse(RV32IM.compressed)
        from riscv.instructions import RV32I

        self.assertEqual(len(RV32IC), len(RV32I), "C adds encodings, not instructions")

    def test_cpu_defaults_to_not_compressed(self):
        """rv32c: a plain CPU() never enters the compressed fetch path"""
        cpu = CPU()
        self.assertFalse(cpu.compressed)

    def test_compressed_word_executes_like_its_expansion(self):
        """c.addi s0,20 behaves exactly like addi s0,s0,20"""
        cpu = CPU(instruction_set=RV32IC)
        cpu.load_program([0x0451], addr=0)  # c.addi s0, 20  (from TOOLCHAIN)
        cpu.set_reg(8, 100)
        cpu.step()
        self.assertReg(cpu, 8, 120)

    def test_next_pc_advances_by_two(self):
        """A compressed instruction advances the PC by 2, not 4"""
        cpu = CPU(instruction_set=RV32IC)
        cpu.load_program([0x0451], addr=0)
        cpu.step()
        self.assertPC(cpu, 2)

    def test_mixed_width_stream(self):
        """A 2-byte and a 4-byte instruction back to back decode correctly"""
        from riscv.encoder import encode
        from riscv.zicsr import RV32IM_ZICSR

        full = encode("ADDI", rd=9, rs1=0, imm=7, instruction_set=RV32IM_ZICSR)
        cpu = CPU(instruction_set=get_isa("rv32imac_zicsr"))
        cpu.memory.write_u16(0, 0x0451)  # c.addi s0, 20 (2 bytes)
        cpu.memory.write_u32(2, full)  # addi s1, zero, 7 (4 bytes)
        cpu.set_reg(8, 0)
        cpu.step()
        self.assertPC(cpu, 2)
        self.assertReg(cpu, 8, 20)
        cpu.step()
        self.assertPC(cpu, 6)
        self.assertReg(cpu, 9, 7)

    def test_odd_pc_is_misaligned(self):
        """Jumping to an odd address still traps, even with IALIGN=16"""
        from riscv.exceptions import InstructionAddressMisaligned

        cpu = CPU(instruction_set=RV32IC)
        cpu.set_pc(1)
        self.assertRaises(InstructionAddressMisaligned, cpu.step)

    def test_two_byte_aligned_pc_is_fine(self):
        """A 4-byte instruction at a 2-byte-aligned (not 4-byte-aligned) PC runs"""
        from riscv.encoder import encode
        from riscv.zicsr import RV32IM_ZICSR

        full = encode("ADDI", rd=9, rs1=0, imm=7, instruction_set=RV32IM_ZICSR)
        cpu = CPU(instruction_set=get_isa("rv32imac_zicsr"))
        cpu.memory.write_u32(2, full)
        cpu.set_pc(2)
        cpu.step()
        self.assertReg(cpu, 9, 7)

    def test_illegal_compressed_instruction_traps(self):
        """A reserved compressed encoding raises IllegalInstruction from step()"""
        cpu = CPU(instruction_set=RV32IC)
        cpu.load_program([0x0000], addr=0)  # C.ADDI4SPN with nzuimm=0: reserved
        self.assertRaises(IllegalInstruction, cpu.step)

    def test_a_compressed_amo_adjacent_program_still_expands_atomics_separately(self):
        """RV32IMAC: a full-width AMO next to compressed arithmetic both work"""
        from riscv.rv32a import RV32IMA
        from riscv.encoder import encode

        isa = get_isa("rv32imac")
        cpu = CPU(instruction_set=isa)
        cpu.write_u32(SCRATCH, 10)
        amo = encode("AMOADD.W", rd=3, rs1=1, rs2=2, instruction_set=RV32IMA)
        cpu.memory.write_u16(0, 0x4e05)  # c.li t3, 1  (TOOLCHAIN-verified)
        cpu.memory.write_u32(2, amo)
        cpu.set_reg(1, SCRATCH)
        cpu.set_reg(2, 5)
        cpu.step()
        self.assertReg(cpu, 28, 1)
        cpu.step()
        self.assertReg(cpu, 3, 10)
        self.assertWord(cpu, SCRATCH, 15)


# ---------------------------------------------------------------------------
# Trace / disassembly: the real C.* mnemonic, not the expanded one
# ---------------------------------------------------------------------------
class TestTraceAndDisassembly(unittest.TestCase):
    def test_trace_shows_the_compressed_mnemonic(self):
        """--trace prints C.ADDI, not ADDI, for a compressed instruction"""
        import io

        cpu = CPU(instruction_set=RV32IC, trace=True, trace_stream=io.StringIO())
        cpu.load_program([0x0451], addr=0)  # c.addi s0, 20
        cpu.step()
        output = cpu.trace_stream.getvalue()
        self.assertIn("C.ADDI", output)
        self.assertIn("INST=0x0451", output)
        self.assertNotIn("INST=0x00000451", output)

    def test_disassemble_bytes_mixed_stream(self):
        """disassemble_bytes walks 2- and 4-byte instructions correctly"""
        from riscv.disasm import disassemble_bytes
        from riscv.encoder import encode
        from riscv.zicsr import RV32IM_ZICSR

        full = encode("ADDI", rd=9, rs1=0, imm=7, instruction_set=RV32IM_ZICSR)
        data = (0x0451).to_bytes(2, "little") + full.to_bytes(4, "little")
        results = list(disassemble_bytes(data, instruction_set=get_isa("rv32imac")))
        self.assertEqual([(addr, size) for addr, size, _raw, _text in results], [(0, 2), (2, 4)])
        self.assertIn("C.ADDI", results[0][3])
        self.assertIn("ADDI", results[1][3])

    def test_disassemble_bytes_without_c_is_plain_4_byte_stepping(self):
        """disassemble_bytes on a non-C set behaves exactly as before"""
        from riscv.disasm import disassemble_bytes
        from riscv.encoder import encode
        from riscv.instructions import RV32I

        word = encode("ADDI", rd=1, rs1=0, imm=5, instruction_set=RV32I)
        data = word.to_bytes(4, "little") * 2
        results = list(disassemble_bytes(data, instruction_set=RV32I))
        self.assertEqual([size for _addr, size, _raw, _text in results], [4, 4])


class TestEndToEndProgram(EmulatorTestCase):
    def test_compressed_loop(self):
        """A mixed compressed/full-width loop: sum 1..5 via c.li/c.add/c.addi + bne.

        This project's bundled assembler has no compressor (it only ever
        emits full-width words), so a genuinely compressed program can't come
        from ``assemble()`` yet -- see the RV32C section of the README. The
        2-byte words below are built directly from the CI/CR bit layouts
        (field placements, not scrambled immediates, so inverting them by
        hand is low-risk) and immediately cross-checked against
        :func:`riscv.rv32c.expand`, which is itself verified against 192 real
        GNU `as` encodings above -- so correctness here traces back to that,
        not to hand arithmetic.
        """
        from riscv.encoder import encode
        from riscv.instructions import RV32I
        from riscv.rv32c import expand

        def ci(funct3, rd, imm6):
            """C.ADDI/C.LI's bit layout: funct3 | imm[5] | rd | imm[4:0] | 01."""
            imm6 &= 0x3F
            return (funct3 << 13) | ((imm6 >> 5 & 1) << 12) | (rd << 7) | ((imm6 & 0x1F) << 2) | 0b01

        def cr_add(rd, rs2):
            """C.ADD's bit layout: 1001 | rd | rs2 | 10."""
            return (0b1001 << 12) | (rd << 7) | (rs2 << 2) | 0b10

        a0, a1, t0 = 10, 11, 5
        halves = {
            "li a0,0": ci(0b010, a0, 0),
            "li a1,1": ci(0b010, a1, 1),
            "li t0,6": ci(0b010, t0, 6),
            "add a0,a1": cr_add(a0, a1),
            "addi a1,1": ci(0b000, a1, 1),
        }
        expected = {
            "li a0,0": encode("ADDI", rd=a0, rs1=0, imm=0, instruction_set=RV32I),
            "li a1,1": encode("ADDI", rd=a1, rs1=0, imm=1, instruction_set=RV32I),
            "li t0,6": encode("ADDI", rd=t0, rs1=0, imm=6, instruction_set=RV32I),
            "add a0,a1": encode("ADD", rd=a0, rs1=a0, rs2=a1, instruction_set=RV32I),
            "addi a1,1": encode("ADDI", rd=a1, rs1=a1, imm=1, instruction_set=RV32I),
        }
        for key, half in halves.items():
            word, _name = expand(half)
            self.assertEqual(word, expected[key], key)

        # Layout: three 2-byte setup instructions, then a 2+2+4 byte loop body.
        bne = encode("BNE", rs1=a1, rs2=t0, imm=-4, instruction_set=RV32I)  # back to offset 6

        code = bytearray()
        for half in (halves["li a0,0"], halves["li a1,1"], halves["li t0,6"]):
            code += half.to_bytes(2, "little")
        for half in (halves["add a0,a1"], halves["addi a1,1"]):
            code += half.to_bytes(2, "little")
        code += bne.to_bytes(4, "little")

        cpu = CPU(instruction_set=get_isa("rv32imac"))
        cpu.load_program(bytes(code), addr=0)
        for _ in range(3 + 5 * 3):  # 3 setup + 5 loop iterations x 3 instructions
            cpu.step()
        self.assertReg(cpu, a0, 15)  # 1+2+3+4+5
        self.assertReg(cpu, a1, 6)
        self.assertPC(cpu, 14)  # stopped at the (not-taken) 5th BNE's fall-through


class TestArchStringMapping(unittest.TestCase):
    def test_plain_c(self):
        """isa: ELF architecture strings with 'c' map to the C-extension sets"""
        from riscv.elf import isa_name_from_arch_string as resolve

        self.assertEqual(resolve("rv32i2p1_c2p0"), "rv32ic")
        self.assertEqual(resolve("rv32i2p1_m2p0_c2p0"), "rv32imc")
        self.assertEqual(resolve("rv32i2p1_m2p0_a2p1_c2p0_zicsr2p0"), "rv32imac_zicsr")

    def test_split_sub_extensions(self):
        """isa: a current toolchain's Zaamo+Zalrsc/Zca map the same way"""
        from riscv.elf import isa_name_from_arch_string as resolve

        self.assertEqual(
            resolve("rv32i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0"),
            "rv32imac",
        )
        self.assertEqual(resolve("rv32i2p1_zca1p0"), "rv32ic")
        self.assertEqual(resolve("rv32i2p1_zaamo1p0_zalrsc1p0"), "rv32ia")

    def test_zaamo_alone_is_not_full_a(self):
        """isa: Zaamo without Zalrsc is a smaller extension than our A"""
        from riscv.elf import isa_name_from_arch_string as resolve

        self.assertIsNone(resolve("rv32i2p1_zaamo1p0"))


if __name__ == "__main__":
    unittest.main()
