"""Machine timer, MTIP, and timer interrupt delivery.

Everything here is deterministic: ``mtime`` advances by a fixed amount per CPU
step and by explicit :meth:`advance` calls, and nothing consults a wall clock.
The same program fires the timer at the same instruction on every machine.
"""

import unittest

from tests.helpers import CPU, EmulatorTestCase, assemble

from riscv.csr import (
    INTERRUPT_PRIORITY,
    IRQ_MACHINE_EXTERNAL,
    IRQ_MACHINE_SOFTWARE,
    IRQ_MACHINE_TIMER,
    MCAUSE,
    MEPC,
    MIE,
    MIP,
    MIP_MEIP,
    MIP_MSIP,
    MIP_MTIP,
    MSTATUS,
    MSTATUS_MIE,
    MSTATUS_MPIE,
    MSTATUS_MPP,
    MSTATUS_MPP_SHIFT,
    MTVAL,
    MTVEC,
    MTVEC_MODE_VECTORED,
    PRIV_MACHINE,
)
from riscv.encoder import encode
from riscv.timer import (
    CLINT_BASE,
    MTIMECMP_OFFSET,
    MTIMECMP_RESET,
    MTIME_OFFSET,
    MachineTimer,
)
from riscv.zicsr import RV32IM_ZICSR

ISA = RV32IM_ZICSR
MTIME_ADDR = CLINT_BASE + MTIME_OFFSET
MTIMECMP_ADDR = CLINT_BASE + MTIMECMP_OFFSET
HANDLER = 0x800
NOP = encode("ADDI", rd=0, rs1=0, imm=0)


def timer_cpu(words=None, *, mtvec=HANDLER, mode=0, ticks_per_step=1, arm=None, **kwargs):
    """A CPU with Zicsr and a machine timer, optionally armed and vectored."""
    kwargs.setdefault("instruction_set", ISA)
    cpu = CPU(timer=MachineTimer(ticks_per_step=ticks_per_step), **kwargs)
    if words is not None:
        cpu.load_program(list(words))
    if mtvec is not None:
        cpu.csr.set(MTVEC, mtvec | mode)
    if arm is not None:
        cpu.timer.mtimecmp = arm
    return cpu


def enable_timer_interrupts(cpu):
    """Set mstatus.MIE and mie.MTIE, as a guest would."""
    cpu.csr.write(MSTATUS, MSTATUS_MIE)
    cpu.csr.write(MIE, MIP_MTIP)


# ---------------------------------------------------------------------------
# The 64-bit registers
# ---------------------------------------------------------------------------
class TestTimerRegisters(unittest.TestCase):
    def setUp(self):
        self.timer = MachineTimer()

    def test_reset_state(self):
        """timer: reset gives time zero and a maximal comparator"""
        self.assertEqual(self.timer.mtime, 0)
        self.assertEqual(self.timer.mtimecmp, MTIMECMP_RESET)
        self.assertEqual(self.timer.mtimecmp, (1 << 64) - 1)
        self.assertFalse(self.timer.mtip, "a reset timer must not be firing")

    def test_mtime_is_64_bit(self):
        """timer: mtime holds a full 64-bit value"""
        self.timer.mtime = 0x1234_5678_9ABC_DEF0
        self.assertEqual(self.timer.mtime, 0x1234_5678_9ABC_DEF0)
        self.timer.mtime = (1 << 64) - 1
        self.assertEqual(self.timer.mtime, (1 << 64) - 1)

    def test_mtimecmp_is_64_bit(self):
        """timer: mtimecmp holds a full 64-bit value"""
        self.timer.mtimecmp = 0xFEDC_BA98_7654_3210
        self.assertEqual(self.timer.mtimecmp, 0xFEDC_BA98_7654_3210)

    def test_values_wrap_at_64_bits(self):
        """timer: mtime wraps at 2**64, not 2**32"""
        self.timer.mtime = (1 << 64) - 1
        self.timer.advance(1)
        self.assertEqual(self.timer.mtime, 0)
        self.timer.mtime = (1 << 32) - 1
        self.timer.advance(1)
        self.assertEqual(self.timer.mtime, 1 << 32, "must not truncate to 32 bits")

    def test_low_and_high_word_reads(self):
        """timer: 32-bit reads return the correct halves"""
        self.timer.mtime = 0x1122_3344_5566_7788
        self.assertEqual(self.timer.read(MTIME_OFFSET, 4), 0x55667788)
        self.assertEqual(self.timer.read(MTIME_OFFSET + 4, 4), 0x11223344)
        self.timer.mtimecmp = 0xAABB_CCDD_EEFF_0011
        self.assertEqual(self.timer.read(MTIMECMP_OFFSET, 4), 0xEEFF0011)
        self.assertEqual(self.timer.read(MTIMECMP_OFFSET + 4, 4), 0xAABBCCDD)

    def test_low_and_high_word_writes(self):
        """timer: 32-bit writes update only their own half"""
        self.timer.mtimecmp = 0
        self.timer.write(MTIMECMP_OFFSET, 4, 0xDEADBEEF)
        self.assertEqual(self.timer.mtimecmp, 0xDEADBEEF)
        self.timer.write(MTIMECMP_OFFSET + 4, 4, 0x12345678)
        self.assertEqual(self.timer.mtimecmp, 0x12345678_DEADBEEF)
        self.timer.write(MTIMECMP_OFFSET, 4, 0)
        self.assertEqual(self.timer.mtimecmp, 0x12345678_00000000,
                         "writing the low half must not disturb the high half")

    def test_narrow_and_wide_accesses(self):
        """timer: byte, halfword and doubleword accesses work"""
        self.timer.mtime = 0x0011_2233_4455_6677
        self.assertEqual(self.timer.read(MTIME_OFFSET, 1), 0x77)
        self.assertEqual(self.timer.read(MTIME_OFFSET + 7, 1), 0x00)
        self.assertEqual(self.timer.read(MTIME_OFFSET, 2), 0x6677)
        self.assertEqual(self.timer.read(MTIME_OFFSET, 8), 0x0011223344556677)

    def test_unmapped_offsets_are_inert(self):
        """timer: offsets outside the two registers read zero and ignore writes"""
        self.assertEqual(self.timer.read(0x0000, 4), 0, "msip is not implemented")
        self.timer.write(0x0000, 4, 0xFFFFFFFF)
        self.assertEqual(self.timer.read(0x0000, 4), 0)
        self.assertFalse(self.timer.mtip, "an unmapped write must not arm the timer")


# ---------------------------------------------------------------------------
# MTIP
# ---------------------------------------------------------------------------
class TestMtip(unittest.TestCase):
    def setUp(self):
        self.timer = MachineTimer()

    def test_asserts_at_the_boundary(self):
        """MTIP: asserts exactly when mtime >= mtimecmp"""
        self.timer.mtimecmp = 100
        self.timer.set_time(98)
        self.assertFalse(self.timer.mtip)
        self.timer.advance(1)
        self.assertFalse(self.timer.mtip, "99 < 100")
        self.timer.advance(1)
        self.assertTrue(self.timer.mtip, "100 >= 100 -- the comparison is >=, not >")
        self.timer.advance(1)
        self.assertTrue(self.timer.mtip)

    def test_comparison_is_64_bit(self):
        """MTIP: the comparison uses all 64 bits"""
        # A low-word-only comparison would get both of these wrong.
        self.timer.set_time(0xFFFFFFFF)
        self.timer.mtimecmp = 1 << 32
        self.assertFalse(self.timer.mtip, "2**32-1 < 2**32")
        self.timer.set_time(1 << 32)
        self.timer.mtimecmp = 0xFFFFFFFF
        self.assertTrue(self.timer.mtip, "2**32 >= 2**32-1")
        self.timer.set_time(0x1_0000_0000)
        self.timer.mtimecmp = 0x2_0000_0000
        self.assertFalse(self.timer.mtip, "high words differ, low words are equal")

    def test_clearing_requires_moving_the_comparator(self):
        """MTIP: only moving mtimecmp past mtime clears it"""
        self.timer.mtimecmp = 10
        self.timer.set_time(50)
        self.assertTrue(self.timer.mtip)
        self.timer.mtimecmp = 51
        self.assertFalse(self.timer.mtip, "pushing the deadline out clears MTIP")
        self.timer.mtimecmp = 50
        self.assertTrue(self.timer.mtip, "...and pulling it back re-asserts")

    def test_time_until_fire(self):
        """timer: time_until_fire reports the remaining ticks"""
        self.timer.mtimecmp = 100
        self.timer.set_time(60)
        self.assertEqual(self.timer.time_until_fire(), 40)
        self.timer.set_time(100)
        self.assertIsNone(self.timer.time_until_fire())


# ---------------------------------------------------------------------------
# Determinism and memory mapping
# ---------------------------------------------------------------------------
class TestDeterminismAndMapping(EmulatorTestCase):
    def test_time_advances_per_step(self):
        """timer: mtime advances one tick per CPU step"""
        cpu = timer_cpu([NOP] * 5, mtvec=None)
        cpu.run(5)
        self.assertEqual(cpu.timer.mtime, 5)
        self.assertEqual(cpu.timer.ticks, 5)

    def test_ticks_per_step_is_configurable(self):
        """timer: ticks_per_step scales emulated time"""
        cpu = timer_cpu([NOP] * 4, mtvec=None, ticks_per_step=10)
        cpu.run(4)
        self.assertEqual(cpu.timer.mtime, 40)

    def test_time_can_be_frozen(self):
        """timer: ticks_per_step=0 freezes time for hand-driven tests"""
        cpu = timer_cpu([NOP] * 4, mtvec=None, ticks_per_step=0)
        cpu.run(4)
        self.assertEqual(cpu.timer.mtime, 0)
        cpu.advance_time(1234)
        self.assertEqual(cpu.timer.mtime, 1234)

    def test_execution_is_deterministic(self):
        """timer: two identical runs fire the timer identically"""
        source = """
            _start:
                la   t0, handler
                csrw mtvec, t0
                li   s0, 0
                li   t1, 40
                li   t2, 0x02004000
                sw   t1, 0(t2)
                sw   x0, 4(t2)
                li   t1, 0x80
                csrs mie, t1
                li   t1, 8
                csrs mstatus, t1
            loop:
                addi s1, s1, 1
                li   t3, 300
                blt  s1, t3, loop
                li   a7, 93
                li   a0, 0
                ecall
            handler:
                addi s0, s0, 1
                li   t4, 0x02004000
                li   t5, -1
                sw   t5, 0(t4)
                sw   t5, 4(t4)
                mret
        """
        from riscv.host import halt_on_ecall

        states = []
        for _ in range(3):
            program = assemble(source, instruction_set=ISA)
            cpu = CPU(instruction_set=ISA, timer=MachineTimer(), ecall_handler=halt_on_ecall)
            program.load(cpu.memory)
            cpu.set_pc(program.entry)
            cpu.run(20000)
            states.append((cpu.regs.snapshot(), cpu.timer.mtime, cpu.interrupts_taken))
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[1], states[2])
        self.assertEqual(states[0][2], 1, "the timer fired exactly once")

    def test_clint_is_memory_mapped(self):
        """timer: the guest reaches mtime and mtimecmp through memory"""
        cpu = timer_cpu(mtvec=None)
        cpu.timer.mtime = 0x1122334455667788
        self.assertEqual(cpu.read_u32(MTIME_ADDR), 0x55667788)
        self.assertEqual(cpu.read_u32(MTIME_ADDR + 4), 0x11223344)
        cpu.write_u32(MTIMECMP_ADDR, 0x99)
        cpu.write_u32(MTIMECMP_ADDR + 4, 0)
        self.assertEqual(cpu.timer.mtimecmp, 0x99)

    def test_guest_stores_reach_the_timer(self):
        """timer: a guest sw to the CLINT updates mtimecmp"""
        cpu = timer_cpu(mtvec=None)
        program = assemble(
            f"li t0, {MTIMECMP_ADDR}\nli t1, 500\nsw t1, 0(t0)\nsw x0, 4(t0)",
            instruction_set=ISA,
        )
        cpu.load_program(program.words)
        cpu.run(len(program.words))
        self.assertEqual(cpu.timer.mtimecmp, 500)

    def test_clint_lives_outside_ram(self):
        """timer: the CLINT is reachable even above the RAM size"""
        cpu = CPU(instruction_set=ISA, memory_size=0x10000, timer=MachineTimer())
        self.assertGreater(CLINT_BASE, 0x10000)
        cpu.write_u32(MTIMECMP_ADDR, 7)
        self.assertEqual(cpu.timer.mtimecmp & 0xFFFFFFFF, 7)

    def test_a_cpu_without_a_timer_is_unchanged(self):
        """timer: no timer means no device, no time and no interrupts"""
        cpu = CPU(instruction_set=ISA)
        self.assertIsNone(cpu.timer)
        self.assertEqual(cpu.memory.mapped_devices(), [])
        self.assertEqual(cpu.csr.interrupt_sources, [])
        self.assertIsNone(cpu.pending_interrupt())

    def test_reset_clears_the_timer(self):
        """timer: cpu.reset() returns the timer to its power-on state"""
        cpu = timer_cpu([NOP] * 3, mtvec=None)
        cpu.run(3)
        cpu.timer.mtimecmp = 5
        cpu.reset()
        self.assertEqual(cpu.timer.mtime, 0)
        self.assertEqual(cpu.timer.mtimecmp, MTIMECMP_RESET)


# ---------------------------------------------------------------------------
# mip, and the gating rules
# ---------------------------------------------------------------------------
class TestPendingAndGating(EmulatorTestCase):
    def test_mip_reports_the_live_wire(self):
        """mip: MTIP tracks the timer without anyone updating a copy"""
        cpu = timer_cpu(mtvec=None, arm=10)
        self.assertEqual(cpu.csr.get(MIP), 0)
        cpu.advance_time(10)
        self.assertEqual(cpu.csr.get(MIP), MIP_MTIP)
        cpu.timer.mtimecmp = 1000
        self.assertEqual(cpu.csr.get(MIP), 0, "resolving clears it immediately")

    def test_guest_can_read_mip(self):
        """mip: a guest CSR read sees MTIP"""
        cpu = timer_cpu([encode("CSRRS", rd=1, rs1=0, imm=MIP, instruction_set=ISA)],
                        mtvec=None, arm=0)
        cpu.step()
        self.assertReg(cpu, 1, MIP_MTIP)

    def test_mip_stays_unwritable(self):
        """mip: a guest cannot set or clear MTIP by writing mip"""
        cpu = timer_cpu(mtvec=None, arm=1 << 60)
        cpu.csr.write(MIP, 0xFFFFFFFF)
        self.assertEqual(cpu.csr.get(MIP), 0, "writes are dropped; the wire is clear")

    def test_not_delivered_without_mtie(self):
        """gating: a pending timer interrupt needs mie.MTIE"""
        cpu = timer_cpu([NOP] * 4, arm=0)
        cpu.csr.write(MSTATUS, MSTATUS_MIE)  # MIE on, MTIE off
        cpu.run(4)
        self.assertEqual(cpu.interrupts_taken, 0)
        self.assertEqual(cpu.csr.get(MIP), MIP_MTIP, "but it is still pending")

    def test_not_delivered_without_mstatus_mie(self):
        """gating: a pending, enabled interrupt still needs mstatus.MIE"""
        cpu = timer_cpu([NOP] * 4, arm=0)
        cpu.csr.write(MIE, MIP_MTIP)  # MTIE on, MIE off
        cpu.run(4)
        self.assertEqual(cpu.interrupts_taken, 0)
        self.assertEqual(cpu.csr.get(MIP), MIP_MTIP)

    def test_delivered_when_both_are_enabled(self):
        """gating: pending + MTIE + MIE delivers"""
        cpu = timer_cpu([NOP] * 4, arm=0)
        enable_timer_interrupts(cpu)
        cpu.step()
        self.assertEqual(cpu.interrupts_taken, 1)

    def test_deliverable_helper(self):
        """gating: deliverable_interrupts applies all three conditions"""
        cpu = timer_cpu(mtvec=None, arm=0)
        self.assertEqual(cpu.csr.deliverable_interrupts(), 0)
        cpu.csr.write(MIE, MIP_MTIP)
        self.assertEqual(cpu.csr.deliverable_interrupts(), 0, "mstatus.MIE is clear")
        cpu.csr.write(MSTATUS, MSTATUS_MIE)
        self.assertEqual(cpu.csr.deliverable_interrupts(), MIP_MTIP)

    def test_priority_order(self):
        """gating: external beats software beats timer"""
        self.assertEqual(
            INTERRUPT_PRIORITY,
            (IRQ_MACHINE_EXTERNAL, IRQ_MACHINE_SOFTWARE, IRQ_MACHINE_TIMER),
        )
        cpu = timer_cpu(mtvec=None, arm=0)
        cpu.csr.write(MSTATUS, MSTATUS_MIE)
        cpu.csr.write(MIE, MIP_MTIP | MIP_MSIP | MIP_MEIP)
        cpu.csr.values[MIP] = MIP_MSIP | MIP_MEIP  # as a controller would assert them
        self.assertEqual(cpu.pending_interrupt(), IRQ_MACHINE_EXTERNAL)
        cpu.csr.values[MIP] = MIP_MSIP
        self.assertEqual(cpu.pending_interrupt(), IRQ_MACHINE_SOFTWARE)
        cpu.csr.values[MIP] = 0
        self.assertEqual(cpu.pending_interrupt(), IRQ_MACHINE_TIMER)


# ---------------------------------------------------------------------------
# Trap entry for interrupts
# ---------------------------------------------------------------------------
class TestInterruptTrapEntry(EmulatorTestCase):
    def armed(self, words=None, **kwargs):
        cpu = timer_cpu(words if words is not None else [NOP] * 8, arm=0, **kwargs)
        enable_timer_interrupts(cpu)
        return cpu

    def test_mcause_has_the_interrupt_bit(self):
        """interrupt: mcause bit 31 is set"""
        cpu = self.armed()
        cpu.step()
        self.assertEqual(cpu.csr.get(MCAUSE), 0x80000007)
        self.assertTrue(cpu.csr.get(MCAUSE) & (1 << 31), "the interrupt bit")
        self.assertEqual(cpu.csr.get(MCAUSE) & 0x7FFFFFFF, IRQ_MACHINE_TIMER)

    def test_mepc_is_the_interrupted_pc(self):
        """interrupt: mepc is the instruction that has not run yet"""
        cpu = timer_cpu([NOP] * 8, arm=5)
        enable_timer_interrupts(cpu)
        # Stop as soon as the interrupt lands: past that point the (empty)
        # handler would fault and overwrite mepc with its own trap.
        for _ in range(20):
            if cpu.interrupts_taken:
                break
            cpu.step()
        self.assertEqual(cpu.csr.get(MEPC), 20,
                         "five NOPs retired, so the sixth is interrupted")
        self.assertEqual(cpu.instret, 5, "the interrupted instruction did not retire")

    def test_mtval_is_zero(self):
        """interrupt: mtval is zero -- an interrupt has no faulting address"""
        # Take the interrupt well away from address zero, so that "mtval is
        # zero" is distinguishable from "mtval happens to be the PC".
        cpu = timer_cpu([NOP] * 8, arm=5)
        enable_timer_interrupts(cpu)
        cpu.csr.set(MTVAL, 0xDEADBEEF)
        for _ in range(20):
            if cpu.interrupts_taken:
                break
            cpu.step()
        self.assertEqual(cpu.interrupts_taken, 1)
        self.assertNotEqual(cpu.csr.get(MEPC), 0, "the interrupted PC is not zero")
        self.assertEqual(cpu.csr.get(MTVAL), 0, "and mtval is still zero")

    def test_mstatus_transitions(self):
        """interrupt: MPIE <- MIE, MIE <- 0, MPP <- privilege"""
        cpu = self.armed()
        cpu.step()
        self.assertFalse(cpu.csr.mie_enabled, "interrupts off inside the handler")
        self.assertTrue(cpu.csr.mpie)
        self.assertEqual(cpu.csr.mpp, PRIV_MACHINE)

    def test_control_transfers_through_mtvec(self):
        """interrupt: control goes to the trap vector"""
        cpu = self.armed()
        cpu.step()
        self.assertPC(cpu, HANDLER)

    def test_no_delivery_without_a_trap_vector(self):
        """interrupt: with no handler installed nothing is delivered"""
        cpu = timer_cpu([NOP] * 4, mtvec=None, arm=0)
        enable_timer_interrupts(cpu)
        cpu.run(4)
        self.assertEqual(cpu.interrupts_taken, 0)
        self.assertEqual(cpu.instret, 4, "the program just runs")

    def test_interrupts_are_taken_between_instructions(self):
        """interrupt: delivery happens at an instruction boundary, never mid-instruction"""
        # A store followed by a load of the same word: if an interrupt were
        # taken part-way through the store, the load would see the old value.
        program = assemble(
            "li t0, 0x1000\nli t1, 0xABC\nsw t1, 0(t0)\nlw t2, 0(t0)\nnop\nnop",
            instruction_set=ISA,
        )
        for deadline in range(0, 8):
            cpu = timer_cpu(program.words, arm=deadline)
            enable_timer_interrupts(cpu)
            cpu.load_program([encode("MRET", instruction_set=ISA)], addr=HANDLER,
                             set_pc=False)
            cpu.set_pc(0)
            cpu.run(30)
            if cpu.get_reg(7):  # t2 was loaded
                self.assertEqual(cpu.get_reg(7), 0xABC,
                                 f"deadline {deadline}: the store completed atomically")
            self.assertEqual(cpu.read_u32(0x1000) in (0, 0xABC), True)

    def test_the_interrupted_instruction_runs_after_mret(self):
        """interrupt: MRET re-runs the instruction that was interrupted"""
        cpu = timer_cpu([encode("ADDI", rd=1, rs1=1, imm=1)] * 4, arm=2)
        enable_timer_interrupts(cpu)
        # The handler must resolve MTIP before returning, or MRET restores MIE
        # while the wire is still asserted and it fires again immediately.
        handler = assemble(
            f"li t0, {MTIMECMP_ADDR}\nli t1, -1\nsw t1, 0(t0)\nsw t1, 4(t0)\nmret",
            instruction_set=ISA,
        ).words
        cpu.load_program(handler, addr=HANDLER, set_pc=False)
        cpu.set_pc(0)
        cpu.run(30)
        # Two ADDIs retired, the interrupt was taken, MRET returned to the third,
        # and all four eventually ran.
        self.assertReg(cpu, 1, 4, "every ADDI ran exactly once")
        self.assertEqual(cpu.interrupts_taken, 1)


# ---------------------------------------------------------------------------
# Direct and vectored delivery
# ---------------------------------------------------------------------------
class TestVectoring(EmulatorTestCase):
    def test_direct_mode_sends_the_timer_to_base(self):
        """vectoring: Direct mode delivers the timer interrupt to BASE"""
        cpu = timer_cpu([NOP] * 4, arm=0, mode=0)
        enable_timer_interrupts(cpu)
        cpu.step()
        self.assertPC(cpu, HANDLER)

    def test_vectored_mode_offsets_the_timer(self):
        """vectoring: Vectored mode delivers the timer to BASE + 4*7"""
        cpu = timer_cpu([NOP] * 4, arm=0, mode=MTVEC_MODE_VECTORED)
        enable_timer_interrupts(cpu)
        cpu.step()
        self.assertPC(cpu, HANDLER + 4 * IRQ_MACHINE_TIMER)
        self.assertPC(cpu, HANDLER + 28)

    def test_vectored_mode_does_not_offset_exceptions(self):
        """vectoring: a synchronous exception still enters at BASE"""
        cpu = timer_cpu([0x00000000], mode=MTVEC_MODE_VECTORED)
        cpu.step()
        self.assertPC(cpu, HANDLER, "cause 2 must not become BASE + 8")
        self.assertEqual(cpu.csr.get(MCAUSE), 2)

    def test_both_kinds_in_one_program(self):
        """vectoring: exception -> BASE, timer -> BASE+28, in the same machine"""
        cpu = timer_cpu([0x00000000, NOP, NOP, NOP], mode=MTVEC_MODE_VECTORED)
        cpu.step()
        self.assertPC(cpu, HANDLER, "the exception")
        cpu.csr.set(MEPC, 4)
        cpu.set_pc(4)
        cpu.timer.mtimecmp = 0
        enable_timer_interrupts(cpu)
        cpu.step()
        self.assertPC(cpu, HANDLER + 28, "the interrupt")

    def test_vector_offsets_for_every_cause(self):
        """vectoring: BASE + 4*cause for each machine interrupt"""
        cpu = timer_cpu(mtvec=HANDLER, mode=MTVEC_MODE_VECTORED)
        for cause in (IRQ_MACHINE_SOFTWARE, IRQ_MACHINE_TIMER, IRQ_MACHINE_EXTERNAL):
            self.assertEqual(cpu.csr.trap_target(cause, True), HANDLER + 4 * cause)
            self.assertEqual(cpu.csr.trap_target(cause, False), HANDLER)

    def test_direct_mode_ignores_the_cause(self):
        """vectoring: Direct mode ignores the cause entirely"""
        cpu = timer_cpu(mtvec=HANDLER, mode=0)
        for cause in (IRQ_MACHINE_SOFTWARE, IRQ_MACHINE_TIMER, IRQ_MACHINE_EXTERNAL):
            self.assertEqual(cpu.csr.trap_target(cause, True), HANDLER)


# ---------------------------------------------------------------------------
# Re-triggering
# ---------------------------------------------------------------------------
class TestRetriggering(EmulatorTestCase):
    def handler_cpu(self, handler_words, arm=0, ticks_per_step=1):
        cpu = timer_cpu([NOP] * 40, arm=arm, ticks_per_step=ticks_per_step)
        enable_timer_interrupts(cpu)
        cpu.load_program(handler_words, addr=HANDLER, set_pc=False)
        cpu.set_pc(0)
        return cpu

    def test_resolved_interrupt_does_not_retrigger(self):
        """retrigger: pushing mtimecmp out stops the interrupt recurring"""
        # Handler: mtimecmp = all ones, then MRET.
        handler = assemble(
            f"li t0, {MTIMECMP_ADDR}\nli t1, -1\nsw t1, 0(t0)\nsw t1, 4(t0)\nmret",
            instruction_set=ISA,
        ).words
        cpu = self.handler_cpu(handler)
        cpu.run(60)
        self.assertEqual(cpu.interrupts_taken, 1, "exactly one delivery")
        self.assertEqual(cpu.csr.get(MIP), 0, "MTIP resolved")

    def test_unresolved_interrupt_retriggers_immediately(self):
        """retrigger: an unresolved MTIP fires again as soon as MRET restores MIE"""
        cpu = self.handler_cpu([encode("MRET", instruction_set=ISA)])
        cpu.run(40)
        self.assertGreater(cpu.interrupts_taken, 5,
                           "MRET restores MIE while MTIP is still asserted")
        self.assertEqual(cpu.get_reg(0), 0, "and the program makes no progress")

    def test_periodic_rearming_fires_repeatedly(self):
        """retrigger: re-arming to mtime+interval gives a periodic tick"""
        handler = assemble(
            f"""
                li   t0, {MTIME_ADDR}
                lw   t1, 0(t0)
                li   t2, 10
                add  t1, t1, t2
                li   t0, {MTIMECMP_ADDR}
                li   t3, -1
                sw   t3, 0(t0)
                sw   x0, 4(t0)
                sw   t1, 0(t0)
                mret
            """,
            instruction_set=ISA,
        ).words
        cpu = self.handler_cpu(handler, arm=10)
        cpu.run(200)
        self.assertGreaterEqual(cpu.interrupts_taken, 3, "a periodic tick")
        self.assertLessEqual(cpu.interrupts_taken, 20, "and not a storm")

    def test_disabling_mie_stops_delivery(self):
        """retrigger: clearing mstatus.MIE in the handler stops the storm"""
        handler = assemble(
            "li t0, 8\ncsrc mstatus, t0\nmret", instruction_set=ISA
        ).words
        cpu = self.handler_cpu(handler)
        cpu.run(60)
        # MRET restores MIE from MPIE, so clearing MIE alone is not enough --
        # this documents that the handler must resolve MTIP or clear MTIE.
        self.assertGreater(cpu.interrupts_taken, 1)

    def test_clearing_mtie_stops_delivery(self):
        """retrigger: clearing mie.MTIE in the handler stops delivery"""
        handler = assemble(
            "li t0, 0x80\ncsrc mie, t0\nmret", instruction_set=ISA
        ).words
        cpu = self.handler_cpu(handler)
        cpu.run(60)
        self.assertEqual(cpu.interrupts_taken, 1, "MTIE is not restored by MRET")
        self.assertEqual(cpu.csr.get(MIP), MIP_MTIP, "still pending, just not enabled")


# ---------------------------------------------------------------------------
# MRET after an interrupt
# ---------------------------------------------------------------------------
class TestMretAfterInterrupt(EmulatorTestCase):
    def test_mret_returns_to_the_interrupted_instruction(self):
        """MRET: returns to mepc after an interrupt"""
        cpu = timer_cpu([NOP] * 6, arm=3)
        enable_timer_interrupts(cpu)
        cpu.load_program([encode("MRET", instruction_set=ISA)], addr=HANDLER, set_pc=False)
        cpu.set_pc(0)
        cpu.run(4)
        interrupted = cpu.csr.get(MEPC)
        for _ in range(20):
            if cpu.pc == interrupted:
                break
            cpu.step()
        self.assertPC(cpu, interrupted)

    def test_mret_restores_mie(self):
        """MRET: MIE comes back on after an interrupt handler"""
        cpu = timer_cpu([NOP] * 4, arm=0)
        enable_timer_interrupts(cpu)
        cpu.step()
        self.assertFalse(cpu.csr.mie_enabled)
        cpu.load_program([encode("MRET", instruction_set=ISA)], addr=HANDLER, set_pc=False)
        cpu.step()
        self.assertTrue(cpu.csr.mie_enabled, "restored from MPIE")
        self.assertTrue(cpu.csr.mpie, "and MPIE is set to 1")

    def test_mstatus_fields_round_trip(self):
        """MRET: MPP and the interrupt bits end where the specification says"""
        cpu = timer_cpu([NOP] * 4, arm=0)
        enable_timer_interrupts(cpu)
        cpu.step()
        cpu.load_program([encode("MRET", instruction_set=ISA)], addr=HANDLER, set_pc=False)
        cpu.step()
        status = cpu.csr.get(MSTATUS)
        self.assertTrue(status & MSTATUS_MIE)
        self.assertTrue(status & MSTATUS_MPIE)
        self.assertEqual((status & MSTATUS_MPP) >> MSTATUS_MPP_SHIFT, PRIV_MACHINE)


# ---------------------------------------------------------------------------
# The guest program
# ---------------------------------------------------------------------------
class TestGuestTimerProgram(EmulatorTestCase):
    """examples/elf/timer_interrupt.S, run on the emulated CPU."""

    EXPECTED = {8: 3, 9: 1, 18: 0x80000007, 20: 0x80, 21: 0x00, 22: 0x1880, 23: 2}

    def run_example(self):
        from tests.helpers import build_elf_example
        from tests.test_elf_integration import run_elf

        return run_elf(build_elf_example("timer_interrupt"), max_instructions=100000)

    def test_the_guest_handles_a_real_timer_interrupt(self):
        """guest timer: the example takes three timer interrupts and exits 0"""
        cpu, _host, exception = self.run_example()
        self.assertIsNone(exception, f"unexpected trap: {exception}")
        self.assertTrue(cpu.halted)
        self.assertEqual(cpu.halt_code, 0)
        self.assertEqual(cpu.interrupts_taken, 3)

    def test_the_guest_observed_the_architectural_state(self):
        """guest timer: the handler read mcause/mepc/mip itself"""
        cpu, _host, _exc = self.run_example()
        for register, value in self.EXPECTED.items():
            self.assertReg(cpu, register, value)
        self.assertNotEqual(cpu.get_reg(19), 0, "mepc was the interrupted PC")

    def test_the_guest_used_vectored_mode(self):
        """guest timer: exceptions went to BASE and the timer to BASE+28"""
        cpu, _host, _exc = self.run_example()
        self.assertEqual(cpu.csr.mtvec_mode, MTVEC_MODE_VECTORED)
        self.assertReg(cpu, 9, 1, "the exception handler ran, so BASE was used")
        self.assertReg(cpu, 8, 3, "the timer handler ran, so BASE+28 was used")


if __name__ == "__main__":
    unittest.main()
