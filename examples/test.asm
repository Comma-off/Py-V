# test.asm
# Expected final values:
# x3  = 42
# x6  = 84
# x8  = 0x12345000
# x9  = 0x12345042
# x10 = 0
# x11 = 1
# x12 = 10
# x13 = 20
# x14 = 30
# x15 = 0xDEADBEEF

# -------------------------
# Arithmetic
# -------------------------

addi x1, x0, 40
addi x2, x0, 2
add  x3, x1, x2        # 42
sub  x4, x3, x2        # 40
addi x5, x0, 2
slli x6, x3, 1         # 84

# -------------------------
# Logical operations
# -------------------------

addi x7, x0, 0x7F
andi x7, x7, 0x0F       # 15

ori  x7, x7, 0x10       # 31
xori x7, x7, 0x1        # 30

# -------------------------
# LUI / arithmetic
# -------------------------

lui  x8, 0x12345        # 0x12345000
addi x9, x8, 0x42        # 0x12345042

# -------------------------
# Comparisons
# -------------------------

addi x10, x0, 10
addi x11, x0, 20

slt x12, x10, x11        # 1
slt x13, x11, x10        # 0

# -------------------------
# Memory
# -------------------------

addi x14, x0, 30
sw   x14, 0(x0)

lw   x15, 0(x0)          # 30

# -------------------------
# Loop
# -------------------------

addi x16, x0, 0          # counter
addi x17, x0, 10         # limit

loop:
    addi x16, x16, 1
    blt  x16, x17, loop

# x16 should be 10

# -------------------------
# Branch tests
# -------------------------

addi x18, x0, 5
addi x19, x0, 5

beq x18, x19, equal

# Should never execute
addi x20, x0, 999

equal:
    addi x20, x0, 123

# -------------------------
# Negative number test
# -------------------------

addi x21, x0, -1

# x21 should be 0xFFFFFFFF

# -------------------------
# End
# -------------------------

ebreak