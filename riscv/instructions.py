"""RV32I instruction semantics.

Every executor has the signature ``execute(cpu, inst) -> None`` and communicates
with the rest of the machine through a deliberately narrow CPU interface:

    cpu.get_reg(i) / cpu.set_reg(i, v)   architectural registers (x0 is handled
                                          inside set_reg, never here)
    cpu.pc                                address of the *current* instruction
    cpu.next_pc                           address of the next one; pre-set to
                                          pc+4, overwritten by jumps/branches
    cpu.load_* / cpu.store_*              memory access with the correct trap
                                          causes and alignment policy

Values are unsigned 32-bit ints everywhere.  Signed operations convert
explicitly via :func:`riscv.bits.to_signed`, which is the only correct way to
get RISC-V's signed comparisons and arithmetic right in Python, where ints are
arbitrary precision and ``>>`` is already arithmetic.

Implementation decisions worth calling out:

* **Wrapping**: every arithmetic result is masked to 32 bits.  RV32I defines
  ADD/SUB/ADDI to wrap silently -- overflow is never trapped.
* **Shift amounts** come from the low 5 bits of the source (``rs2 & 0x1F`` or
  the immediate's ``shamt`` field), per spec; larger values wrap rather than
  producing zero.
* **SRA** shifts the *signed* interpretation, then re-masks.  Python's ``>>``
  on a negative int is already an arithmetic shift with infinite sign bits.
* **JALR** clears bit 0 of the computed target ("setting the LSB to zero"),
  *after* adding the immediate.
* **Link registers**: JAL/JALR write ``pc + 4`` even when ``rd`` is ``x0``,
  where the write is discarded -- that is how ``j`` and ``ret`` are encoded.
* **Branch/jump alignment**: with no compressed extension IALIGN is 32, so a
  target that is not 4-byte aligned raises an instruction-address-misaligned
  trap *before* the jump takes effect and before ``rd`` is written.
"""

from .bits import MASK32, to_signed
from .isa import (
    B_TYPE,
    FENCE_TYPE,
    I_TYPE,
    InstructionSet,
    J_TYPE,
    N_TYPE,
    R_TYPE,
    S_TYPE,
    U_TYPE,
    register_isa,
)

#: The base integer instruction set.  Kept strictly RV32I -- extensions build a
#: copy rather than mutating this, so it stays a faithful reference.
RV32I = register_isa("rv32i", InstructionSet("RV32I"))


def _check_target_alignment(cpu, target, inst):
    """Raise if a jump/branch target violates IALIGN (32, or 16 with C)."""
    mask = 0x1 if cpu.compressed else 0x3
    if target & mask:
        from .exceptions import InstructionAddressMisaligned

        raise InstructionAddressMisaligned(tval=target, pc=cpu.pc)
    return target


# ---------------------------------------------------------------------------
# U-type: LUI, AUIPC
# ---------------------------------------------------------------------------
@RV32I.define("LUI", match=0x00000037, mask=0x0000007F, fmt=U_TYPE)
def exec_lui(cpu, inst):
    """rd = imm[31:12] << 12 (the immediate already carries the shift)."""
    cpu.set_reg(inst.rd, inst.imm)


@RV32I.define("AUIPC", match=0x00000017, mask=0x0000007F, fmt=U_TYPE)
def exec_auipc(cpu, inst):
    """rd = pc + (imm[31:12] << 12)."""
    cpu.set_reg(inst.rd, (cpu.pc + inst.imm) & MASK32)


# ---------------------------------------------------------------------------
# Jumps: JAL, JALR
# ---------------------------------------------------------------------------
@RV32I.define("JAL", match=0x0000006F, mask=0x0000007F, fmt=J_TYPE)
def exec_jal(cpu, inst):
    """rd = pc + 4; pc += sext(imm)."""
    target = (cpu.pc + inst.imm) & MASK32
    _check_target_alignment(cpu, target, inst)
    cpu.set_reg(inst.rd, (cpu.pc + 4) & MASK32)
    cpu.next_pc = target


@RV32I.define("JALR", match=0x00000067, mask=0x0000707F, fmt=I_TYPE)
def exec_jalr(cpu, inst):
    """rd = pc + 4; pc = (rs1 + sext(imm)) & ~1."""
    target = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32 & ~1
    _check_target_alignment(cpu, target, inst)
    cpu.set_reg(inst.rd, (cpu.pc + 4) & MASK32)
    cpu.next_pc = target


# ---------------------------------------------------------------------------
# B-type conditional branches
# ---------------------------------------------------------------------------
def _branch(cpu, inst, taken):
    if taken:
        target = (cpu.pc + inst.imm) & MASK32
        _check_target_alignment(cpu, target, inst)
        cpu.next_pc = target


@RV32I.define("BEQ", match=0x00000063, mask=0x0000707F, fmt=B_TYPE)
def exec_beq(cpu, inst):
    _branch(cpu, inst, cpu.get_reg(inst.rs1) == cpu.get_reg(inst.rs2))


@RV32I.define("BNE", match=0x00001063, mask=0x0000707F, fmt=B_TYPE)
def exec_bne(cpu, inst):
    _branch(cpu, inst, cpu.get_reg(inst.rs1) != cpu.get_reg(inst.rs2))


@RV32I.define("BLT", match=0x00004063, mask=0x0000707F, fmt=B_TYPE)
def exec_blt(cpu, inst):
    _branch(cpu, inst, to_signed(cpu.get_reg(inst.rs1)) < to_signed(cpu.get_reg(inst.rs2)))


@RV32I.define("BGE", match=0x00005063, mask=0x0000707F, fmt=B_TYPE)
def exec_bge(cpu, inst):
    _branch(cpu, inst, to_signed(cpu.get_reg(inst.rs1)) >= to_signed(cpu.get_reg(inst.rs2)))


@RV32I.define("BLTU", match=0x00006063, mask=0x0000707F, fmt=B_TYPE)
def exec_bltu(cpu, inst):
    _branch(cpu, inst, cpu.get_reg(inst.rs1) < cpu.get_reg(inst.rs2))


@RV32I.define("BGEU", match=0x00007063, mask=0x0000707F, fmt=B_TYPE)
def exec_bgeu(cpu, inst):
    _branch(cpu, inst, cpu.get_reg(inst.rs1) >= cpu.get_reg(inst.rs2))


# ---------------------------------------------------------------------------
# Loads (I-type) -- rd = mem[rs1 + sext(imm)]
# ---------------------------------------------------------------------------
@RV32I.define("LB", match=0x00000003, mask=0x0000707F, fmt=I_TYPE)
def exec_lb(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.set_reg(inst.rd, cpu.load(addr, 1, signed=True))


@RV32I.define("LH", match=0x00001003, mask=0x0000707F, fmt=I_TYPE)
def exec_lh(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.set_reg(inst.rd, cpu.load(addr, 2, signed=True))


@RV32I.define("LW", match=0x00002003, mask=0x0000707F, fmt=I_TYPE)
def exec_lw(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.set_reg(inst.rd, cpu.load(addr, 4, signed=True))


@RV32I.define("LBU", match=0x00004003, mask=0x0000707F, fmt=I_TYPE)
def exec_lbu(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.set_reg(inst.rd, cpu.load(addr, 1, signed=False))


@RV32I.define("LHU", match=0x00005003, mask=0x0000707F, fmt=I_TYPE)
def exec_lhu(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.set_reg(inst.rd, cpu.load(addr, 2, signed=False))


# ---------------------------------------------------------------------------
# Stores (S-type) -- mem[rs1 + sext(imm)] = rs2
# ---------------------------------------------------------------------------
@RV32I.define("SB", match=0x00000023, mask=0x0000707F, fmt=S_TYPE)
def exec_sb(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.store(addr, 1, cpu.get_reg(inst.rs2))


@RV32I.define("SH", match=0x00001023, mask=0x0000707F, fmt=S_TYPE)
def exec_sh(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.store(addr, 2, cpu.get_reg(inst.rs2))


@RV32I.define("SW", match=0x00002023, mask=0x0000707F, fmt=S_TYPE)
def exec_sw(cpu, inst):
    addr = (cpu.get_reg(inst.rs1) + inst.imm) & MASK32
    cpu.store(addr, 4, cpu.get_reg(inst.rs2))


# ---------------------------------------------------------------------------
# Register-immediate ALU (I-type)
# ---------------------------------------------------------------------------
@RV32I.define("ADDI", match=0x00000013, mask=0x0000707F, fmt=I_TYPE)
def exec_addi(cpu, inst):
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) + inst.imm) & MASK32)


@RV32I.define("SLTI", match=0x00002013, mask=0x0000707F, fmt=I_TYPE)
def exec_slti(cpu, inst):
    cpu.set_reg(inst.rd, 1 if to_signed(cpu.get_reg(inst.rs1)) < inst.imm_signed else 0)


@RV32I.define("SLTIU", match=0x00003013, mask=0x0000707F, fmt=I_TYPE)
def exec_sltiu(cpu, inst):
    """Unsigned compare against the *sign-extended* immediate (spec quirk:
    ``sltiu rd, rs1, 1`` is the ``seqz`` idiom, and -1 compares as 0xFFFFFFFF)."""
    cpu.set_reg(inst.rd, 1 if cpu.get_reg(inst.rs1) < inst.imm else 0)


@RV32I.define("XORI", match=0x00004013, mask=0x0000707F, fmt=I_TYPE)
def exec_xori(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) ^ inst.imm)


@RV32I.define("ORI", match=0x00006013, mask=0x0000707F, fmt=I_TYPE)
def exec_ori(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) | inst.imm)


@RV32I.define("ANDI", match=0x00007013, mask=0x0000707F, fmt=I_TYPE)
def exec_andi(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) & inst.imm)


# Shift-immediates pin funct7, so a stray bit in imm[11:5] is an illegal
# instruction rather than a silently-ignored encoding.
@RV32I.define("SLLI", match=0x00001013, mask=0xFE00707F, fmt=I_TYPE)
def exec_slli(cpu, inst):
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) << inst.shamt) & MASK32)


@RV32I.define("SRLI", match=0x00005013, mask=0xFE00707F, fmt=I_TYPE)
def exec_srli(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) >> inst.shamt)


@RV32I.define("SRAI", match=0x40005013, mask=0xFE00707F, fmt=I_TYPE)
def exec_srai(cpu, inst):
    cpu.set_reg(inst.rd, (to_signed(cpu.get_reg(inst.rs1)) >> inst.shamt) & MASK32)


# ---------------------------------------------------------------------------
# Register-register ALU (R-type)
# ---------------------------------------------------------------------------
@RV32I.define("ADD", match=0x00000033, mask=0xFE00707F, fmt=R_TYPE)
def exec_add(cpu, inst):
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) + cpu.get_reg(inst.rs2)) & MASK32)


@RV32I.define("SUB", match=0x40000033, mask=0xFE00707F, fmt=R_TYPE)
def exec_sub(cpu, inst):
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) - cpu.get_reg(inst.rs2)) & MASK32)


@RV32I.define("SLL", match=0x00001033, mask=0xFE00707F, fmt=R_TYPE)
def exec_sll(cpu, inst):
    shift = cpu.get_reg(inst.rs2) & 0x1F
    cpu.set_reg(inst.rd, (cpu.get_reg(inst.rs1) << shift) & MASK32)


@RV32I.define("SLT", match=0x00002033, mask=0xFE00707F, fmt=R_TYPE)
def exec_slt(cpu, inst):
    left = to_signed(cpu.get_reg(inst.rs1))
    right = to_signed(cpu.get_reg(inst.rs2))
    cpu.set_reg(inst.rd, 1 if left < right else 0)


@RV32I.define("SLTU", match=0x00003033, mask=0xFE00707F, fmt=R_TYPE)
def exec_sltu(cpu, inst):
    cpu.set_reg(inst.rd, 1 if cpu.get_reg(inst.rs1) < cpu.get_reg(inst.rs2) else 0)


@RV32I.define("XOR", match=0x00004033, mask=0xFE00707F, fmt=R_TYPE)
def exec_xor(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) ^ cpu.get_reg(inst.rs2))


@RV32I.define("SRL", match=0x00005033, mask=0xFE00707F, fmt=R_TYPE)
def exec_srl(cpu, inst):
    shift = cpu.get_reg(inst.rs2) & 0x1F
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) >> shift)


@RV32I.define("SRA", match=0x40005033, mask=0xFE00707F, fmt=R_TYPE)
def exec_sra(cpu, inst):
    shift = cpu.get_reg(inst.rs2) & 0x1F
    cpu.set_reg(inst.rd, (to_signed(cpu.get_reg(inst.rs1)) >> shift) & MASK32)


@RV32I.define("OR", match=0x00006033, mask=0xFE00707F, fmt=R_TYPE)
def exec_or(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) | cpu.get_reg(inst.rs2))


@RV32I.define("AND", match=0x00007033, mask=0xFE00707F, fmt=R_TYPE)
def exec_and(cpu, inst):
    cpu.set_reg(inst.rd, cpu.get_reg(inst.rs1) & cpu.get_reg(inst.rs2))


# ---------------------------------------------------------------------------
# MISC-MEM and SYSTEM
# ---------------------------------------------------------------------------
@RV32I.define("FENCE", match=0x0000000F, mask=0x0000707F, fmt=FENCE_TYPE)
def exec_fence(cpu, inst):
    """No-op: this core executes strictly in order with a single hart."""
    cpu.fences += 1


@RV32I.define("ECALL", match=0x00000073, mask=0xFFFFFFFF, fmt=N_TYPE)
def exec_ecall(cpu, inst):
    """Hand control to the environment.

    With no host handler installed this raises the architectural trap; the CLI
    installs a small syscall handler so example programs can print and exit.
    """
    cpu.environment_call()


@RV32I.define("EBREAK", match=0x00100073, mask=0xFFFFFFFF, fmt=N_TYPE)
def exec_ebreak(cpu, inst):
    """Return control to a debugger; by default raises the breakpoint trap."""
    cpu.breakpoint()


#: Instruction count sanity check -- RV32I has exactly 40 instructions.
RV32I_INSTRUCTION_COUNT = 40
