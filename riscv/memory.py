"""Little-endian, sparsely-allocated physical memory.

The address space is a flat array of bytes of a configurable ``size``.  Storage
is allocated lazily one page at a time, so a CPU can be given a 4 GiB address
space without allocating 4 GiB of host RAM.  Pages that were never written read
as zero, which conveniently means fetching from unmapped memory yields the
all-zeros word -- an illegal instruction -- rather than silently succeeding.

Design notes
------------
* Alignment is *not* enforced here.  The CPU enforces it, because only the CPU
  knows whether a given access is a fetch, a load or a store, and each of those
  raises a different trap cause.
* Accesses outside ``[0, size)`` raise :class:`~riscv.exceptions.MemoryError_`,
  which the CPU converts into the appropriate access-fault trap.
* There is deliberately no notion of MMIO, caching or permissions.  A future
  MMIO layer can subclass :class:`Memory` and override :meth:`read_bytes` /
  :meth:`write_bytes`, which every other accessor funnels through in the slow
  path.
"""

import bisect

from .exceptions import MemoryError_

DEFAULT_SIZE = 1 << 20  # 1 MiB is plenty for the bundled example programs
DEFAULT_PAGE_BITS = 12  # 4 KiB pages


class Memory:
    """A sparse little-endian byte-addressable memory."""

    __slots__ = ("size", "page_bits", "page_size", "_page_mask", "_pages", "_devices")

    def __init__(self, size: int = DEFAULT_SIZE, page_bits: int = DEFAULT_PAGE_BITS):
        if size <= 0:
            raise ValueError("memory size must be positive")
        if not 4 <= page_bits <= 24:
            raise ValueError("page_bits must be between 4 and 24")
        self.size = size
        self.page_bits = page_bits
        self.page_size = 1 << page_bits
        self._page_mask = self.page_size - 1
        self._pages: dict[int, bytearray] = {}
        #: Memory-mapped devices, as ``(start, end, device, name)``.  Empty for
        #: an ordinary RAM-only machine, so the check below costs one truth
        #: test per access.
        self._devices: list = []

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop every allocated page; the whole space reads as zero again."""
        self._pages.clear()

    @property
    def allocated_pages(self) -> int:
        return len(self._pages)

    @property
    def allocated_bytes(self) -> int:
        return len(self._pages) * self.page_size

    def mapped_page_addresses(self) -> list[int]:
        """Base addresses of every page that has been written to, ascending."""
        return sorted(index << self.page_bits for index in self._pages)

    def _check(self, addr: int, length: int) -> None:
        if addr < 0 or length < 0 or addr + length > self.size:
            # A mapped device lives in the address space whether or not it
            # happens to fall below `size` -- the CLINT sits at 0x02000000,
            # well above a small RAM.  Only the failing path pays for this.
            if length >= 0 and addr >= 0 and self._devices and self.device_at(addr):
                return
            raise MemoryError_(
                f"access of {length} byte(s) at 0x{addr:08X} is outside "
                f"the 0x0..0x{self.size:X} address space"
            )

    def _page(self, index: int) -> bytearray:
        page = self._pages.get(index)
        if page is None:
            page = bytearray(self.page_size)
            self._pages[index] = page
        return page

    # ------------------------------------------------------------------
    # Memory-mapped devices
    # ------------------------------------------------------------------
    def map_device(self, start: int, length: int, device, name: str = "") -> None:
        """Map ``device`` over ``[start, start+length)``.

        A device implements ``read(offset, width) -> int`` and
        ``write(offset, width, value)``, with ``offset`` relative to ``start``
        and ``width`` in bytes.  Accesses inside the range go to the device
        instead of RAM; everything else is untouched, so a machine with no
        devices behaves exactly as before.
        """
        if length <= 0:
            raise ValueError("device length must be positive")
        self._devices.append((start, start + length, device, name))

    def device_at(self, addr: int):
        """``(device, offset)`` covering ``addr``, or ``None``."""
        for start, end, device, _name in self._devices:
            if start <= addr < end:
                return device, addr - start
        return None

    def mapped_devices(self):
        """``[(start, end, name), ...]`` for every mapped device."""
        return [(start, end, name) for start, end, _device, name in self._devices]

    # ------------------------------------------------------------------
    # Byte-string access (every slow path funnels through these two)
    # ------------------------------------------------------------------
    def read_bytes(self, addr: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``addr``."""
        self._check(addr, length)
        if length == 0:
            return b""
        out = bytearray(length)
        done = 0
        while done < length:
            cur = addr + done
            index = cur >> self.page_bits
            offset = cur & self._page_mask
            chunk = min(self.page_size - offset, length - done)
            page = self._pages.get(index)
            if page is not None:
                out[done:done + chunk] = page[offset:offset + chunk]
            # else: leave the zeros already in ``out``
            done += chunk
        return bytes(out)

    def write_bytes(self, addr: int, data) -> None:
        """Write the byte-like ``data`` starting at ``addr``.

        ``data`` may be a ``memoryview`` (an ELF segment, say); it is sliced
        per page rather than copied wholesale first.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        view = memoryview(data)
        length = view.nbytes
        self._check(addr, length)
        done = 0
        while done < length:
            cur = addr + done
            index = cur >> self.page_bits
            offset = cur & self._page_mask
            chunk = min(self.page_size - offset, length - done)
            self._page(index)[offset:offset + chunk] = view[done:done + chunk]
            done += chunk

    def zero_fill(self, addr: int, length: int) -> None:
        """Zero ``length`` bytes without densifying the address space.

        Pages that were never allocated already read as zero, so they are left
        untouched -- a multi-megabyte ``.bss`` costs nothing.  Only pages that
        already exist are actually cleared.
        """
        self._check(addr, length)
        done = 0
        while done < length:
            cur = addr + done
            index = cur >> self.page_bits
            offset = cur & self._page_mask
            chunk = min(self.page_size - offset, length - done)
            page = self._pages.get(index)
            if page is not None:
                page[offset:offset + chunk] = bytes(chunk)
            done += chunk

    # ------------------------------------------------------------------
    # Unsigned scalar access
    # ------------------------------------------------------------------
    def read_u8(self, addr: int) -> int:
        self._check(addr, 1)
        if self._devices:
            hit = self.device_at(addr)
            if hit is not None:
                return hit[0].read(hit[1], 1) & 0xFF
        page = self._pages.get(addr >> self.page_bits)
        return 0 if page is None else page[addr & self._page_mask]

    def read_u16(self, addr: int) -> int:
        return int.from_bytes(self._read_scalar(addr, 2), "little")

    def read_u32(self, addr: int) -> int:
        return int.from_bytes(self._read_scalar(addr, 4), "little")

    def _read_scalar(self, addr: int, width: int) -> bytes:
        self._check(addr, width)
        if self._devices:
            hit = self.device_at(addr)
            if hit is not None:
                return (hit[0].read(hit[1], width) & ((1 << (width * 8)) - 1)).to_bytes(
                    width, "little"
                )
        offset = addr & self._page_mask
        if offset + width <= self.page_size:  # fast path: single page
            page = self._pages.get(addr >> self.page_bits)
            if page is None:
                return b"\x00" * width
            return bytes(page[offset:offset + width])
        return self.read_bytes(addr, width)  # straddles a page boundary

    def write_u8(self, addr: int, value: int) -> None:
        self._check(addr, 1)
        if self._devices:
            hit = self.device_at(addr)
            if hit is not None:
                hit[0].write(hit[1], 1, value & 0xFF)
                return
        self._page(addr >> self.page_bits)[addr & self._page_mask] = value & 0xFF

    def write_u16(self, addr: int, value: int) -> None:
        self._write_scalar(addr, value & 0xFFFF, 2)

    def write_u32(self, addr: int, value: int) -> None:
        self._write_scalar(addr, value & 0xFFFFFFFF, 4)

    def _write_scalar(self, addr: int, value: int, width: int) -> None:
        self._check(addr, width)
        if self._devices:
            hit = self.device_at(addr)
            if hit is not None:
                hit[0].write(hit[1], width, value)
                return
        offset = addr & self._page_mask
        raw = value.to_bytes(width, "little")
        if offset + width <= self.page_size:  # fast path: single page
            self._page(addr >> self.page_bits)[offset:offset + width] = raw
        else:
            self.write_bytes(addr, raw)

    # ------------------------------------------------------------------
    # Signed convenience accessors (host-side inspection only; the CPU does
    # its own sign extension so that trap causes stay correct)
    # ------------------------------------------------------------------
    def read_i8(self, addr: int) -> int:
        value = self.read_u8(addr)
        return value - 0x100 if value & 0x80 else value

    def read_i16(self, addr: int) -> int:
        value = self.read_u16(addr)
        return value - 0x10000 if value & 0x8000 else value

    def read_i32(self, addr: int) -> int:
        value = self.read_u32(addr)
        return value - 0x100000000 if value & 0x80000000 else value

    # ------------------------------------------------------------------
    # Program loading / inspection
    # ------------------------------------------------------------------
    def load(self, addr: int, data) -> None:
        """Copy a program image (bytes) into memory at ``addr``."""
        self.write_bytes(addr, data)

    def load_words(self, addr: int, words) -> None:
        """Copy a sequence of 32-bit little-endian words into memory."""
        blob = bytearray()
        for word in words:
            blob += (word & 0xFFFFFFFF).to_bytes(4, "little")
        self.write_bytes(addr, blob)

    def hexdump(self, addr: int, length: int, per_line: int = 16) -> str:
        """Return a classic hex/ASCII dump of a memory range."""
        data = self.read_bytes(addr, length)
        lines = []
        for offset in range(0, length, per_line):
            chunk = data[offset:offset + per_line]
            hex_part = " ".join(f"{byte:02X}" for byte in chunk)
            text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"0x{addr + offset:08X}  {hex_part:<{per_line * 3 - 1}}  |{text}|")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Memory size=0x{self.size:X} pages={len(self._pages)} "
            f"page_size=0x{self.page_size:X}>"
        )


# ---------------------------------------------------------------------------
# Optional access permissions
# ---------------------------------------------------------------------------
#: Permission bits.  These deliberately use the same values as the ELF ``PF_*``
#: segment flags, so loader metadata maps straight through with no translation.
PERM_X = 0x1
PERM_W = 0x2
PERM_R = 0x4
PERM_RW = PERM_R | PERM_W
PERM_RX = PERM_R | PERM_X
PERM_RWX = PERM_R | PERM_W | PERM_X


def permission_text(flags: int) -> str:
    """Render permission bits as ``R-X``-style text."""
    return (
        ("R" if flags & PERM_R else "-")
        + ("W" if flags & PERM_W else "-")
        + ("X" if flags & PERM_X else "-")
    )


class MemoryProtection:
    """A small region table describing what may be read, written and executed.

    This is deliberately *not* an MMU: there is no translation, no page tables
    and no privilege levels -- just a sorted list of ``[start, end)`` regions
    carrying ``R``/``W``/``X`` bits, which is exactly what an ELF's ``PT_LOAD``
    flags describe.  The CPU consults it (when one is installed) on every fetch,
    load and store, and raises the corresponding access fault.

    Addresses outside every registered region take ``default_flags``, which is
    read+write but *not* execute.  That is what turns "jumped into the weeds"
    into an instruction access fault at the first bad instruction, while leaving
    scratch memory usable by programs that were never loaded from an ELF.

    Regions are expected not to overlap (the ELF specification requires that of
    ``PT_LOAD`` segments).  If they do, the region with the **highest start
    address** containing the byte wins -- a deterministic rule, not an accident
    of insertion order.
    """

    __slots__ = ("regions", "_starts", "default_flags", "_cache")

    def __init__(self, default_flags: int = PERM_RW, regions=()):
        self.default_flags = default_flags
        self.regions = []
        self._starts = []
        self._cache = None
        for region in regions:
            self.add_region(*region)

    def add_region(self, start: int, length: int, flags: int, name: str = "") -> None:
        """Register ``[start, start+length)`` with the given permission bits."""
        if length <= 0:
            return
        index = bisect.bisect_right(self._starts, start)
        self.regions.insert(index, (start, start + length, flags, name))
        self._starts.insert(index, start)
        self._cache = None

    def _lookup(self, addr: int):
        """Return ``(flags, lo, hi)``: the permissions in force at ``addr``,
        valid for the whole half-open range ``[lo, hi)``."""
        starts = self._starts
        index = bisect.bisect_right(starts, addr) - 1
        while index >= 0:
            region_start, region_end, flags, _name = self.regions[index]
            if addr < region_end:
                return flags, region_start, region_end
            index -= 1
        # In a gap: the default applies up to the next region's start.
        upper = bisect.bisect_right(starts, addr)
        hi = starts[upper] if upper < len(starts) else 1 << 32
        return self.default_flags, addr, hi

    def permissions_at(self, addr: int) -> int:
        """Permission bits covering a single address."""
        cache = self._cache
        if cache is not None and cache[1] <= addr < cache[2]:
            return cache[0]
        entry = self._lookup(addr)
        self._cache = entry
        return entry[0]

    def check(self, addr: int, length: int, want: int) -> bool:
        """True if every byte in ``[addr, addr+length)`` grants ``want``."""
        if length <= 0:
            return True
        cache = self._cache
        if cache is not None and cache[1] <= addr and addr + length <= cache[2]:
            return (cache[0] & want) == want  # fast path: one region covers it all
        end = addr + length
        position = addr
        while position < end:
            flags, _lo, hi = entry = self._lookup(position)
            if (flags & want) != want:
                return False
            self._cache = entry
            position = hi
        return True

    def describe(self) -> str:
        lines = [
            f"default: {permission_text(self.default_flags)} (addresses outside every region)"
        ]
        for start, end, flags, name in self.regions:
            label = f"  {name}" if name else ""
            lines.append(f"0x{start:08X}-0x{end:08X}  {permission_text(flags)}{label}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<MemoryProtection regions={len(self.regions)} "
            f"default={permission_text(self.default_flags)}>"
        )
