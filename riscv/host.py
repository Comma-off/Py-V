"""An optional host environment for ECALL.

The core emulator is deliberately OS-free: with no handler installed, ``ECALL``
raises the architectural trap and nothing else happens.  Programs that want to
print or exit -- the bundled ``hello.asm``, for instance -- run under this small
handler, which the CLI installs.

Nothing in :mod:`riscv.cpu` depends on this module; it is one ``ecall_handler``
callable, which is exactly the seam a real supervisor would eventually occupy.

Calling convention (a7 selects the call, a0-a2 carry arguments, a0 returns):

===== ============== =================================================
a7    name           behaviour
===== ============== =================================================
1     print_int      print a0 as a signed decimal
2     print_char     print a0 as a character
4     print_string   print the NUL-terminated string at a0
10    exit           halt with code 0
11    put_char       alias of print_char (RARS compatibility)
63    read           unsupported; returns -1
64    write          write(a0=fd, a1=buf, a2=len); returns a2
93    exit           halt with code a0 (Linux/newlib convention)
===== ============== =================================================
"""

import sys

from .bits import to_signed
from .exceptions import HaltRequest

#: Register numbers of the argument registers used by the ABI.
A0, A1, A2, A7 = 10, 11, 12, 17


class HostEnvironment:
    """A minimal syscall layer, usable as ``cpu.ecall_handler``."""

    def __init__(self, stdout=None, capture: bool = False):
        self.stdout = stdout if stdout is not None else sys.stdout
        #: When ``capture`` is set, output is also collected here for tests.
        self.capture = capture
        self.output = []
        self.exit_code = None
        self.calls = 0

    # -- helpers --------------------------------------------------------
    def _write(self, text: str) -> None:
        if self.capture:
            self.output.append(text)
        self.stdout.write(text)

    def text(self) -> str:
        """Everything written so far (only populated when ``capture`` is set)."""
        return "".join(self.output)

    def _cstring(self, cpu, addr: int, limit: int = 4096) -> bytes:
        out = bytearray()
        for offset in range(limit):
            byte = cpu.memory.read_u8(addr + offset)
            if byte == 0:
                break
            out.append(byte)
        return bytes(out)

    # -- the handler ----------------------------------------------------
    def __call__(self, cpu) -> None:
        self.calls += 1
        number = cpu.get_reg(A7)
        if number == 1:
            self._write(str(to_signed(cpu.get_reg(A0))))
        elif number in (2, 11):
            self._write(chr(cpu.get_reg(A0) & 0xFF))
        elif number == 4:
            self._write(self._cstring(cpu, cpu.get_reg(A0)).decode("utf-8", "replace"))
        elif number == 10:
            self.exit_code = 0
            raise HaltRequest(0)
        elif number == 63:
            cpu.set_reg(A0, 0xFFFFFFFF)  # read() is not supported
        elif number == 64:
            length = cpu.get_reg(A2)
            data = cpu.memory.read_bytes(cpu.get_reg(A1), length)
            self._write(data.decode("utf-8", "replace"))
            cpu.set_reg(A0, length)
        elif number == 93:
            self.exit_code = to_signed(cpu.get_reg(A0))
            raise HaltRequest(self.exit_code)
        else:
            raise HaltRequest(-1)


def halt_on_ecall(cpu) -> None:
    """The simplest possible handler: any ECALL stops the machine.

    Useful as a program terminator in tests that do not care about I/O.
    """
    raise HaltRequest(to_signed(cpu.get_reg(A0)))
