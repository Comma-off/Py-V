"""A self-contained RV32I emulator written in pure Python.

Quick start::

    from riscv import CPU, assemble

    cpu = CPU()
    program = assemble("addi x1, x0, 10\\naddi x2, x0, 20\\nadd x3, x1, x2")
    cpu.load_program(program.image)
    cpu.run(3)
    assert cpu.get_reg(3) == 30

Module map
----------
``bits``          32-bit helpers: masking, sign extension, signed/unsigned views
``exceptions``    architectural traps (with RISC-V cause codes) + host errors
``memory``        sparse little-endian physical memory
``registers``     the 32-entry register file, x0 hardwired to zero
``isa``           instruction-set tables -- the extension point for RV32M/A/F
``instructions``  RV32I semantics, one function per instruction
``decoder``       word -> decoded instruction (pure, cacheable)
``encoder``       decoded fields -> word (exact inverse of the decoder)
``disasm``        decoded instruction -> text, used by trace mode
``cpu``           the hart: fetch/decode/execute, traps, tracing
``assembler``     two-pass assembler so ``.asm`` examples need no toolchain
``elf``           ELF32 RISC-V reader/loader (independent of the CPU)
``elfwriter``     ELF32 RISC-V writer, for building examples without a toolchain
``rv32m``         the M extension (multiply/divide), built on top of RV32I
``csr``           control and status registers, and machine-mode trap state
``zicsr``         the Zicsr extension (CSR access) and MRET
``host``          optional ECALL environment (printing/exit) for examples
``selftest``      built-in validation suite behind ``riscv.py --self-test``
"""

from .assembler import Assembler, Program, assemble, assemble_file
from .bits import MASK32, XLEN, mask32, sign_extend, to_signed, to_unsigned
from .cpu import CPU
from .csr import CSRFile, CSRDefinition, PRIV_MACHINE, csr_names
from .decoder import Decoder, Instruction
from .disasm import disassemble, safe_disassemble, trace_line
from .elf import ElfError, ElfFile, LoadedImage, Segment, is_elf, load_elf, parse_elf
from .elfwriter import ElfWriter, program_to_elf
from .encoder import encode
from .exceptions import (
    AssemblerError,
    Breakpoint,
    EmulatorError,
    EnvironmentCall,
    IllegalInstruction,
    InstructionAccessFault,
    InstructionAddressMisaligned,
    LoadAccessFault,
    LoadAddressMisaligned,
    RiscvException,
    StoreAccessFault,
    StoreAddressMisaligned,
)
from .host import HostEnvironment, halt_on_ecall
from .instructions import RV32I
from .isa import (
    InstructionDef,
    InstructionSet,
    available_isas,
    get_isa,
    register_isa,
)
from .memory import PERM_R, PERM_RW, PERM_RWX, PERM_RX, PERM_W, PERM_X, Memory, MemoryProtection
from .registers import ABI_NAMES, RegisterFile, register_number
from .rv32m import RV32IM
from .zicsr import RV32I_ZICSR, RV32IM_ZICSR

__version__ = "1.0.0"

__all__ = [
    "ABI_NAMES",
    "Assembler",
    "AssemblerError",
    "Breakpoint",
    "CPU",
    "CSRDefinition",
    "CSRFile",
    "Decoder",
    "ElfError",
    "ElfFile",
    "ElfWriter",
    "EmulatorError",
    "EnvironmentCall",
    "HostEnvironment",
    "IllegalInstruction",
    "Instruction",
    "InstructionAccessFault",
    "InstructionAddressMisaligned",
    "InstructionDef",
    "InstructionSet",
    "LoadAccessFault",
    "LoadAddressMisaligned",
    "LoadedImage",
    "MASK32",
    "Memory",
    "MemoryProtection",
    "PRIV_MACHINE",
    "PERM_R",
    "PERM_RW",
    "PERM_RWX",
    "PERM_RX",
    "PERM_W",
    "PERM_X",
    "Program",
    "RV32I",
    "RV32IM",
    "RV32IM_ZICSR",
    "RV32I_ZICSR",
    "RegisterFile",
    "RiscvException",
    "Segment",
    "StoreAccessFault",
    "StoreAddressMisaligned",
    "XLEN",
    "__version__",
    "assemble",
    "assemble_file",
    "available_isas",
    "csr_names",
    "disassemble",
    "encode",
    "get_isa",
    "halt_on_ecall",
    "is_elf",
    "load_elf",
    "parse_elf",
    "program_to_elf",
    "register_isa",
    "mask32",
    "register_number",
    "safe_disassemble",
    "sign_extend",
    "to_signed",
    "to_unsigned",
    "trace_line",
]
