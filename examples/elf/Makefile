# Build the ELF examples with a real RISC-V toolchain.
#
#     make                       # build every example into build/
#     make run-hello             # build and run one under the emulator
#     make TOOLPREFIX=riscv64-unknown-elf-   # use a different toolchain
#
# A toolchain is NOT required to use these examples or to run the test suite:
#   python examples/elf/build.py
# assembles the same sources with the emulator's own assembler.

TOOLPREFIX ?= riscv32-linux-gnu-
AS      := $(TOOLPREFIX)as
LD      := $(TOOLPREFIX)ld
OBJDUMP := $(TOOLPREFIX)objdump

# -mno-relax / --no-relax keep `la` as auipc+addi.  Without them the linker
# relaxes symbol references to gp-relative addressing, which only works if the
# program first sets gp to __global_pointer$ -- part of the ABI startup that
# bare-metal code without a C runtime has to do for itself.  Keeping relaxation
# off makes these sources build identically with the bundled assembler, which
# has no linker-defined symbols.  (tests/test_real_binaries.py separately builds
# a *relaxed* binary that does set up gp, to check that path works too.)
ASFLAGS := -march=rv32i -mabi=ilp32 -mno-relax
LDFLAGS := -m elf32lriscv -Ttext=0x10000 --no-warn-rwx-segments --no-relax

PROGRAMS := hello arithmetic branches memory functions trap multiply \
            trap_handler timer_interrupt
BUILD    := build
ELVES    := $(addprefix $(BUILD)/,$(addsuffix .elf,$(PROGRAMS)))

RISCV := python3 ../../riscv.py

all: $(ELVES)

$(BUILD):
	mkdir -p $(BUILD)

# Per-program architecture: multiply.S needs M, trap_handler.S needs Zicsr,
# everything else is base RV32I.
$(BUILD)/multiply.o:     ARCH := rv32im
$(BUILD)/trap_handler.o:    ARCH := rv32i_zicsr
$(BUILD)/timer_interrupt.o: ARCH := rv32i_zicsr
$(BUILD)/%.o: ARCH ?= rv32i

$(BUILD)/%.o: %.S | $(BUILD)
	$(AS) -march=$(ARCH) -mabi=ilp32 -mno-relax $< -o $@

$(BUILD)/%.elf: $(BUILD)/%.o
	$(LD) $(LDFLAGS) $< -o $@

run-%: $(BUILD)/%.elf
	$(RISCV) exec $<

dis-%: $(BUILD)/%.elf
	$(RISCV) disasm $< --symbols --abi

check: $(ELVES)
	@for elf in $(ELVES); do echo "== $$elf"; $(RISCV) exec $$elf || true; done

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
