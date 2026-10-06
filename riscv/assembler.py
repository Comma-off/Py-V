"""A small two-pass RV32I assembler.

Its whole reason to exist is self-containment: the brief requires example
``.asm`` programs *and* a test suite that runs without an external RISC-V
toolchain.  Rather than shipping opaque hex blobs, the emulator assembles its
own examples, and the assembler is validated against the decoder/disassembler by
round-trip tests.

Supported syntax
----------------
::

    # comments start with '#' or '//'
    .text                       # section (data follows text by default)
    .data
    label:                      # labels
        addi  a0, zero, 10      # ABI or xN register names
        lw    t0, 8(sp)         # memory operand form
        beq   a0, a1, label     # symbolic branch target
        beq   a0, a1, -8        # ...or a literal PC-relative offset
        li    t1, 0x12345678    # pseudo-instructions
        la    t2, message
    message:
        .string "hi\n"
        .word  1, 2, 3+4
        .align 2

Directives: ``.text .data .section .word .half .byte .string .asciz .ascii
.space .zero .align .balign .equ .set .org .globl .global`` (the last two are
accepted and ignored).

Numeric-vs-symbolic branch targets: an operand containing a symbol is an
*absolute address*; a purely numeric one is a *PC-relative offset*.  That rule
keeps assembly and disassembly exact inverses of each other.
"""

import re

from .bits import to_signed
from .csr import csr_names
from .encoder import encode, is_shift_immediate
from .exceptions import AssemblerError
from .isa import (
    AMO_TYPE,
    B_TYPE,
    CSR_TYPE,
    FENCE_TYPE,
    I_TYPE,
    J_TYPE,
    N_TYPE,
    R_TYPE,
    S_TYPE,
    U_TYPE,
)
from .registers import REGISTER_ALIASES, register_number

DEFAULT_TEXT_BASE = 0x00000000

_MEMORY_OPERAND = re.compile(r"^\s*(?P<offset>[^()]*?)\s*\(\s*(?P<base>[^()]+?)\s*\)\s*$")
_LABEL = re.compile(r"^(?P<label>[.A-Za-z_][\w.$]*)\s*:\s*(?P<rest>.*)$")
_TOKEN = re.compile(
    r"""\s*(?:
        (?P<hex>0[xX][0-9A-Fa-f]+) |
        (?P<bin>0[bB][01]+) |
        (?P<oct>0[oO][0-7]+) |
        (?P<num>\d+) |
        (?P<char>'(?:\\.|[^'\\])') |
        (?P<reloc>%[a-z_]+) |
        (?P<sym>[.A-Za-z_][\w.$]*) |
        (?P<op><<|>>|[-+*/%&|^~()])
    )""",
    re.VERBOSE,
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------
class _Expr:
    """Tokenised expression with precedence-climbing evaluation."""

    _PRECEDENCE = {
        "|": 1, "^": 2, "&": 3,
        "<<": 4, ">>": 4,
        "+": 5, "-": 5,
        "*": 6, "/": 6, "%": 6,
    }

    def __init__(self, text, line_no=None):
        self.text = text
        self.line_no = line_no
        self.tokens = self._tokenise(text)

    @staticmethod
    def _tokenise(text):
        tokens = []
        position = 0
        while position < len(text):
            if text[position].isspace():
                position += 1
                continue
            match = _TOKEN.match(text, position)
            if not match or match.end() == position:
                raise AssemblerError(f"cannot parse expression {text!r}")
            position = match.end()
            kind = match.lastgroup
            value = match.group(kind)
            tokens.append((kind, value))
        if not tokens:
            raise AssemblerError("empty expression")
        return tokens

    @property
    def symbols(self):
        return [value for kind, value in self.tokens if kind == "sym"]

    @property
    def has_symbols(self):
        return any(kind in ("sym", "reloc") for kind, _ in self.tokens)

    def evaluate(self, symbols, pc=None):
        self._symbols = symbols
        self._pc = pc
        self._pos = 0
        value = self._parse(0)
        if self._pos != len(self.tokens):
            raise AssemblerError(f"trailing tokens in expression {self.text!r}")
        return value

    # -- recursive descent ------------------------------------------
    def _peek(self):
        return self.tokens[self._pos] if self._pos < len(self.tokens) else (None, None)

    def _parse(self, min_precedence):
        left = self._parse_unary()
        while True:
            kind, value = self._peek()
            if kind != "op" or value not in self._PRECEDENCE:
                break
            precedence = self._PRECEDENCE[value]
            if precedence < min_precedence:
                break
            self._pos += 1
            right = self._parse(precedence + 1)
            left = self._apply(value, left, right)
        return left

    @staticmethod
    def _apply(operator, left, right):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            if right == 0:
                raise AssemblerError("division by zero in expression")
            return int(left / right)
        if operator == "%":
            if right == 0:
                raise AssemblerError("division by zero in expression")
            return left % right
        if operator == "<<":
            return left << right
        if operator == ">>":
            return left >> right
        if operator == "&":
            return left & right
        if operator == "|":
            return left | right
        return left ^ right  # "^"

    def _parse_unary(self):
        kind, value = self._peek()
        if kind == "op" and value in ("-", "+", "~", "("):
            self._pos += 1
            if value == "(":
                inner = self._parse(0)
                nkind, nvalue = self._peek()
                if nkind != "op" or nvalue != ")":
                    raise AssemblerError(f"unbalanced parentheses in {self.text!r}")
                self._pos += 1
                return inner
            operand = self._parse_unary()
            if value == "-":
                return -operand
            if value == "~":
                return ~operand
            return operand
        return self._parse_atom()

    def _parse_atom(self):
        kind, value = self._peek()
        self._pos += 1
        if kind == "hex":
            return int(value, 16)
        if kind == "bin":
            return int(value, 2)
        if kind == "oct":
            return int(value, 8)
        if kind == "num":
            return int(value, 10)
        if kind == "char":
            body = value[1:-1]
            if body.startswith("\\"):
                return ord(_unescape(body))
            return ord(body)
        if kind == "reloc":
            return self._parse_reloc(value)
        if kind == "sym":
            if value == ".":
                if self._pc is None:
                    raise AssemblerError("'.' used where no location is known")
                return self._pc
            if value not in self._symbols:
                raise AssemblerError(f"undefined symbol {value!r}")
            return self._symbols[value]
        raise AssemblerError(f"unexpected token {value!r} in expression {self.text!r}")

    def _parse_reloc(self, name):
        kind, value = self._peek()
        if kind != "op" or value != "(":
            raise AssemblerError(f"{name} must be followed by '('")
        self._pos += 1
        inner = self._parse(0)
        kind, value = self._peek()
        if kind != "op" or value != ")":
            raise AssemblerError(f"unbalanced parentheses after {name}")
        self._pos += 1
        if name == "%hi":
            return ((inner + 0x800) >> 12) & 0xFFFFF
        if name == "%lo":
            return to_signed(inner & 0xFFF, 12)
        raise AssemblerError(f"unsupported relocation {name}")


def _unescape(text):
    out = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            if nxt == "x" and index + 3 < len(text) + 1:
                out.append(chr(int(text[index + 2:index + 4], 16)))
                index += 4
                continue
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                index += 2
                continue
            out.append(nxt)
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def split_operands(text):
    """Split an operand list on commas that are outside parens and quotes."""
    if not text.strip():
        return []
    parts = []
    current = []
    depth = 0
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            current.append(char)
            if char == "\\" and index + 1 < len(text):
                current.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if quote:
        raise AssemblerError(f"unterminated string in {text!r}")
    parts.append("".join(current).strip())
    return parts


def strip_block_comments(source):
    """Replace ``/* ... */`` comments with blank space, preserving line numbers.

    Block comments are what GNU ``as`` (and the C preprocessor, for ``.S``
    files) accept, so supporting them lets one source file assemble with either
    the bundled assembler or a real toolchain.  Newlines inside a comment are
    kept so that error messages still report the right line.
    """
    out = []
    index = 0
    length = len(source)
    quote = None
    while index < length:
        char = source[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "#" or source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end < 0 else end
            out.append(source[index:end])  # left for strip_comment() to remove
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise AssemblerError("unterminated /* comment")
            out.append("\n" * source.count("\n", index, end + 2))
            index = end + 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def strip_comment(line):
    """Remove ``#`` / ``//`` comments, respecting quoted strings."""
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:index]
        elif char == "/" and line.startswith("//", index):
            return line[:index]
        index += 1
    return line


# ---------------------------------------------------------------------------
# Pseudo-instructions
# ---------------------------------------------------------------------------
#: name -> (operand count, expansion builder).  Builders return a list of
#: ``(mnemonic, [operands])`` tuples; ``None`` operands are filled by the caller.
PSEUDO_SIMPLE = {
    "nop":   (0, lambda ops: [("addi", ["x0", "x0", "0"])]),
    "mv":    (2, lambda ops: [("addi", [ops[0], ops[1], "0"])]),
    "not":   (2, lambda ops: [("xori", [ops[0], ops[1], "-1"])]),
    "neg":   (2, lambda ops: [("sub", [ops[0], "x0", ops[1]])]),
    "seqz":  (2, lambda ops: [("sltiu", [ops[0], ops[1], "1"])]),
    "snez":  (2, lambda ops: [("sltu", [ops[0], "x0", ops[1]])]),
    "sltz":  (2, lambda ops: [("slt", [ops[0], ops[1], "x0"])]),
    "sgtz":  (2, lambda ops: [("slt", [ops[0], "x0", ops[1]])]),
    "beqz":  (2, lambda ops: [("beq", [ops[0], "x0", ops[1]])]),
    "bnez":  (2, lambda ops: [("bne", [ops[0], "x0", ops[1]])]),
    "blez":  (2, lambda ops: [("bge", ["x0", ops[0], ops[1]])]),
    "bgez":  (2, lambda ops: [("bge", [ops[0], "x0", ops[1]])]),
    "bltz":  (2, lambda ops: [("blt", [ops[0], "x0", ops[1]])]),
    "bgtz":  (2, lambda ops: [("blt", ["x0", ops[0], ops[1]])]),
    "bgt":   (3, lambda ops: [("blt", [ops[1], ops[0], ops[2]])]),
    "ble":   (3, lambda ops: [("bge", [ops[1], ops[0], ops[2]])]),
    "bgtu":  (3, lambda ops: [("bltu", [ops[1], ops[0], ops[2]])]),
    "bleu":  (3, lambda ops: [("bgeu", [ops[1], ops[0], ops[2]])]),
    "j":     (1, lambda ops: [("jal", ["x0", ops[0]])]),
    "jr":    (1, lambda ops: [("jalr", ["x0", "0(" + ops[0] + ")"])]),
    "ret":   (0, lambda ops: [("jalr", ["x0", "0(x1)"])]),
    # Zicsr pseudo-instructions (RISC-V assembly manual, table of CSR aliases).
    "csrr":  (2, lambda ops: [("csrrs", [ops[0], ops[1], "x0"])]),
    "csrw":  (2, lambda ops: [("csrrw", ["x0", ops[0], ops[1]])]),
    "csrs":  (2, lambda ops: [("csrrs", ["x0", ops[0], ops[1]])]),
    "csrc":  (2, lambda ops: [("csrrc", ["x0", ops[0], ops[1]])]),
    "csrwi": (2, lambda ops: [("csrrwi", ["x0", ops[0], ops[1]])]),
    "csrsi": (2, lambda ops: [("csrrsi", ["x0", ops[0], ops[1]])]),
    "csrci": (2, lambda ops: [("csrrci", ["x0", ops[0], ops[1]])]),
    "ecall_": (0, lambda ops: [("ecall", [])]),
}
#: Pseudo-instructions whose operand count selects the expansion.
PSEUDO_VARIADIC = {
    ("jal", 1): lambda ops: [("jal", ["x1", ops[0]])],
    ("jalr", 1): lambda ops: [("jalr", ["x1", "0(" + ops[0] + ")"])],
}
#: Pseudo-instructions expanded specially because their size depends on operands.
PSEUDO_MULTIWORD = ("li", "la", "call", "tail")


class Program:
    """The result of assembling a source file."""

    def __init__(self, segments, symbols, entry, listing, text_base, data_base,
                 bss_base=0, bss_size=0):
        #: ``[(address, bytes), ...]`` sorted by address.  File-backed only --
        #: ``.bss`` is not here, because it has no contents.
        self.segments = sorted(segments)
        self.symbols = symbols
        self.entry = entry
        #: ``[(address, word, line_no, source_text), ...]`` for every instruction.
        self.listing = listing
        self.text_base = text_base
        self.data_base = data_base
        #: Zero-initialised region: reserved address space with no file bytes,
        #: exactly like an ELF segment whose ``p_memsz`` exceeds ``p_filesz``.
        self.bss_base = bss_base
        self.bss_size = bss_size

    @property
    def words(self):
        """The text section as a list of 32-bit words."""
        blob = self.segments[0][1] if self.segments else b""
        return [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob) - 3, 4)]

    @property
    def image(self):
        """All segments as one blob starting at :attr:`base` (gaps zero-filled)."""
        if not self.segments:
            return b""
        start = self.segments[0][0]
        end = max(addr + len(data) for addr, data in self.segments)
        blob = bytearray(end - start)
        for addr, data in self.segments:
            blob[addr - start:addr - start + len(data)] = data
        return bytes(blob)

    @property
    def base(self):
        return self.segments[0][0] if self.segments else self.text_base

    @property
    def size(self):
        return len(self.image)

    def load(self, memory) -> None:
        """Write every segment into a :class:`~riscv.memory.Memory`.

        The ``.bss`` region is zeroed rather than written, so a large reserved
        area does not allocate pages it does not need.
        """
        for addr, data in self.segments:
            memory.write_bytes(addr, data)
        if self.bss_size:
            memory.zero_fill(self.bss_base, self.bss_size)

    def format_listing(self) -> str:
        lines = []
        for addr, word, line_no, text in self.listing:
            lines.append(f"0x{addr:08X}  {word:08X}  {line_no:>4}  {text.strip()}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Program base=0x{self.base:08X} size={self.size} symbols={len(self.symbols)}>"


class Assembler:
    """Two-pass assembler producing a :class:`Program`."""

    def __init__(self, text_base=DEFAULT_TEXT_BASE, data_base=None, instruction_set=None):
        if instruction_set is None:
            from .instructions import RV32I

            instruction_set = RV32I
        self.instruction_set = instruction_set
        self.text_base = text_base
        self.data_base = data_base

    # -- public ---------------------------------------------------------
    def assemble(self, source, filename=None) -> Program:
        items = self._parse(source)
        sizes, symbols_rel, sizes_by_section = self._measure(items)
        text_size = sizes_by_section["text"]
        data_size = sizes_by_section["data"]
        bss_size = sizes_by_section["bss"]

        text_base = self.text_base
        data_base = self.data_base
        if data_base is None:
            data_base = (text_base + text_size + 3) & ~3
        bss_base = (data_base + data_size + 3) & ~3
        bases = {"text": text_base, "data": data_base, "bss": bss_base}
        symbols = {
            name: bases[section] + offset for name, (section, offset) in symbols_rel.items()
        }
        symbols.update(self._absolutes)
        buffers, listing = self._emit(items, sizes, symbols, bases)

        segments = []
        if buffers["text"]:
            segments.append((text_base, bytes(buffers["text"])))
        if buffers["data"]:
            segments.append((data_base, bytes(buffers["data"])))
        entry = symbols.get("_start", symbols.get("main", text_base))
        return Program(
            segments, symbols, entry, listing, text_base, data_base,
            bss_base=bss_base, bss_size=bss_size,
        )

    # -- pass 0: parse --------------------------------------------------
    def _parse(self, source):
        items = []
        source = strip_block_comments(source)
        for line_no, raw in enumerate(source.splitlines(), start=1):
            line = strip_comment(raw).strip()
            while line:
                match = _LABEL.match(line)
                if not match:
                    break
                items.append(("label", match.group("label"), line_no, raw))
                line = match.group("rest").strip()
            if not line:
                continue
            parts = line.split(None, 1)
            head = parts[0]
            rest = parts[1].strip() if len(parts) > 1 else ""
            operands = split_operands(rest)
            if head.startswith("."):
                items.append(("directive", (head.lower(), operands), line_no, raw))
            else:
                items.append(("instruction", (head.lower(), operands), line_no, raw))
        return items

    # -- pass 1: measure ------------------------------------------------
    def _measure(self, items):
        sizes = {}
        symbols = {}
        self._absolutes = {}
        offsets = {"text": 0, "data": 0, "bss": 0}
        section = "text"
        for index, (kind, payload, line_no, raw) in enumerate(items):
            try:
                if kind == "label":
                    if payload in symbols or payload in self._absolutes:
                        raise AssemblerError(f"duplicate label {payload!r}")
                    symbols[payload] = (section, offsets[section])
                elif kind == "directive":
                    name, operands = payload
                    if name in (".text", ".data", ".bss"):
                        section = name[1:]
                        continue
                    if name == ".section":
                        section = _section_name(operands[0] if operands else ".text")
                        continue
                    if name in (".equ", ".set"):
                        if len(operands) != 2:
                            raise AssemblerError(f"{name} takes NAME, VALUE")
                        self._absolutes[operands[0]] = _Expr(operands[1], line_no).evaluate(
                            self._known(symbols, offsets, section)
                        )
                        continue
                    if section == "bss" and name in _DATA_DIRECTIVES:
                        raise AssemblerError(
                            f"{name} cannot emit data in .bss (use .space to reserve room)"
                        )
                    size, new_offset = self._directive_size(
                        name, operands, offsets[section], section, symbols, line_no
                    )
                    sizes[index] = size
                    offsets[section] = new_offset
                    continue
                else:
                    if section == "bss":
                        raise AssemblerError("instructions are not allowed in .bss")
                    size = self._instruction_size(payload, line_no)
                    sizes[index] = size
                    offsets[section] += size
            except AssemblerError as exc:
                raise AssemblerError(str(exc).split("\n")[0], line_no, raw) from None
        return sizes, symbols, offsets

    def _known(self, symbols, offsets, section):
        """Symbols usable during pass 1 (absolutes only -- addresses aren't final)."""
        return dict(self._absolutes)

    def _directive_size(self, name, operands, offset, section, symbols, line_no):
        if name in (".globl", ".global", ".type", ".size", ".file", ".ident", ".option", ".attribute"):
            return 0, offset
        if name == ".word":
            return 4 * len(operands), offset + 4 * len(operands)
        if name == ".half" or name == ".short":
            return 2 * len(operands), offset + 2 * len(operands)
        if name == ".byte":
            return len(operands), offset + len(operands)
        if name in (".string", ".asciz", ".ascii"):
            total = 0
            for operand in operands:
                total += len(_string_bytes(operand, nul=name != ".ascii"))
            return total, offset + total
        if name in (".space", ".zero"):
            count = _Expr(operands[0], line_no).evaluate(self._absolutes)
            return count, offset + count
        if name in (".align", ".balign", ".p2align"):
            boundary = _Expr(operands[0], line_no).evaluate(self._absolutes)
            boundary = boundary if name == ".balign" else 1 << boundary
            padding = (-offset) % boundary
            return padding, offset + padding
        if name == ".org":
            if section != "text":
                raise AssemblerError(".org is only supported in the text section")
            target = _Expr(operands[0], line_no).evaluate(self._absolutes) - self.text_base
            if target < offset:
                raise AssemblerError(f".org moves backwards (to 0x{target:X})")
            return target - offset, target
        raise AssemblerError(f"unknown directive {name}")

    def _instruction_size(self, payload, line_no):
        mnemonic, operands = payload
        if mnemonic in PSEUDO_MULTIWORD:
            if mnemonic == "li":
                if len(operands) != 2:
                    raise AssemblerError("li takes rd, immediate")
                expr = _Expr(operands[1], line_no)
                if expr.has_symbols:
                    return 8
                value = to_signed(expr.evaluate({}) & 0xFFFFFFFF)
                return 4 if -2048 <= value <= 2047 else 8
            return 8  # la / call / tail always use the two-instruction form
        expansions = self._expand_pseudo(mnemonic, operands, line_no)
        if expansions is not None:
            return 4 * len(expansions)
        self._lookup_mnemonic(mnemonic, line_no)
        return 4

    def _lookup_mnemonic(self, mnemonic, line_no):
        try:
            return self.instruction_set.by_name(mnemonic.upper())
        except KeyError:
            raise AssemblerError(f"unknown instruction {mnemonic!r}") from None

    def _expand_pseudo(self, mnemonic, operands, line_no):
        entry = PSEUDO_VARIADIC.get((mnemonic, len(operands)))
        if entry is not None:
            return entry(operands)
        entry = PSEUDO_SIMPLE.get(mnemonic)
        if entry is None:
            return None
        count, builder = entry
        if len(operands) != count:
            raise AssemblerError(f"{mnemonic} takes {count} operand(s), got {len(operands)}")
        return builder(operands)

    # -- pass 2: emit ---------------------------------------------------
    def _emit(self, items, sizes, symbols, bases):
        buffers = {"text": bytearray(), "data": bytearray(), "bss": bytearray()}
        listing = []
        section = "text"
        for index, (kind, payload, line_no, raw) in enumerate(items):
            try:
                if kind == "label":
                    continue
                if kind == "directive":
                    name, operands = payload
                    if name in (".text", ".data", ".bss"):
                        section = name[1:]
                        continue
                    if name == ".section":
                        section = _section_name(operands[0] if operands else ".text")
                        continue
                    if name in (".equ", ".set"):
                        continue
                    self._emit_directive(
                        name, operands, buffers[section], symbols, bases[section], line_no
                    )
                    continue
                pc = bases[section] + len(buffers[section])
                words = self._encode_instruction(payload, pc, symbols, line_no)
                expected = sizes[index]
                if 4 * len(words) != expected:
                    raise AssemblerError(
                        f"internal size mismatch: measured {expected}, emitted {4 * len(words)}"
                    )
                for offset, word in enumerate(words):
                    buffers[section] += word.to_bytes(4, "little")
                    listing.append((pc + 4 * offset, word, line_no, raw))
            except AssemblerError as exc:
                raise AssemblerError(str(exc).split("\n")[0], line_no, raw) from None
        return buffers, listing

    def _emit_directive(self, name, operands, buffer, symbols, base, line_no):
        if name in (".globl", ".global", ".type", ".size", ".file", ".ident", ".option", ".attribute"):
            return
        here = base + len(buffer)
        if name in (".word", ".half", ".short", ".byte"):
            width = {".word": 4, ".half": 2, ".short": 2, ".byte": 1}[name]
            for operand in operands:
                value = _Expr(operand, line_no).evaluate(symbols, here)
                buffer += (value & ((1 << (width * 8)) - 1)).to_bytes(width, "little")
            return
        if name in (".string", ".asciz", ".ascii"):
            for operand in operands:
                buffer += _string_bytes(operand, nul=name != ".ascii")
            return
        if name in (".space", ".zero"):
            count = _Expr(operands[0], line_no).evaluate(symbols, here)
            fill = _Expr(operands[1], line_no).evaluate(symbols, here) if len(operands) > 1 else 0
            buffer += bytes([fill & 0xFF]) * count
            return
        if name in (".align", ".balign", ".p2align"):
            boundary = _Expr(operands[0], line_no).evaluate(symbols, here)
            boundary = boundary if name == ".balign" else 1 << boundary
            buffer += bytes((-len(buffer)) % boundary)
            return
        if name == ".org":
            target = _Expr(operands[0], line_no).evaluate(symbols, here) - base
            buffer += bytes(max(0, target - len(buffer)))
            return
        raise AssemblerError(f"unknown directive {name}")

    # -- instruction encoding ------------------------------------------
    def _encode_instruction(self, payload, pc, symbols, line_no):
        mnemonic, operands = payload
        if mnemonic in PSEUDO_MULTIWORD:
            return self._encode_multiword(mnemonic, operands, pc, symbols, line_no)
        expansions = self._expand_pseudo(mnemonic, operands, line_no)
        if expansions is not None:
            words = []
            for offset, (real_mnemonic, real_operands) in enumerate(expansions):
                words.append(
                    self._encode_real(real_mnemonic, real_operands, pc + 4 * offset, symbols, line_no)
                )
            return words
        return [self._encode_real(mnemonic, operands, pc, symbols, line_no)]

    def _encode_multiword(self, mnemonic, operands, pc, symbols, line_no):
        if mnemonic == "li":
            rd = _register(operands[0], line_no)
            expr = _Expr(operands[1], line_no)
            value = expr.evaluate(symbols, pc) & 0xFFFFFFFF
            signed = to_signed(value)
            # A symbolic operand always takes the two-instruction form: pass 1
            # reserved 8 bytes for it because the value was not yet known.
            if -2048 <= signed <= 2047 and not expr.has_symbols:
                return [encode("ADDI", rd=rd, rs1=0, imm=signed, instruction_set=self.instruction_set)]
            upper = ((value + 0x800) >> 12) & 0xFFFFF
            lower = to_signed(value & 0xFFF, 12)
            return [
                encode("LUI", rd=rd, imm=upper, instruction_set=self.instruction_set),
                encode("ADDI", rd=rd, rs1=rd, imm=lower, instruction_set=self.instruction_set),
            ]
        if mnemonic == "la":
            rd = _register(operands[0], line_no)
            target = _Expr(operands[1], line_no).evaluate(symbols, pc)
            delta = (target - pc) & 0xFFFFFFFF
            upper = ((delta + 0x800) >> 12) & 0xFFFFF
            lower = to_signed(delta & 0xFFF, 12)
            return [
                encode("AUIPC", rd=rd, imm=upper, instruction_set=self.instruction_set),
                encode("ADDI", rd=rd, rs1=rd, imm=lower, instruction_set=self.instruction_set),
            ]
        if mnemonic in ("call", "tail"):
            link = 1 if mnemonic == "call" else 0
            target = _Expr(operands[-1], line_no).evaluate(symbols, pc)
            delta = (target - pc) & 0xFFFFFFFF
            upper = ((delta + 0x800) >> 12) & 0xFFFFF
            lower = to_signed(delta & 0xFFF, 12)
            scratch = 1 if mnemonic == "call" else 6  # t1 for tail, per the ABI
            return [
                encode("AUIPC", rd=scratch, imm=upper, instruction_set=self.instruction_set),
                encode("JALR", rd=link, rs1=scratch, imm=lower, instruction_set=self.instruction_set),
            ]
        raise AssemblerError(f"unhandled pseudo-instruction {mnemonic}")

    def _encode_real(self, mnemonic, operands, pc, symbols, line_no):
        definition = self._lookup_mnemonic(mnemonic, line_no)
        fmt = definition.fmt
        name = definition.name

        def need(count):
            if len(operands) != count:
                raise AssemblerError(
                    f"{mnemonic} takes {count} operand(s), got {len(operands)}"
                )

        if fmt == R_TYPE:
            need(3)
            return encode(
                name,
                rd=_register(operands[0], line_no),
                rs1=_register(operands[1], line_no),
                rs2=_register(operands[2], line_no),
                instruction_set=self.instruction_set,
            )
        if fmt == I_TYPE:
            if is_shift_immediate(definition):
                need(3)
                shamt = _Expr(operands[2], line_no).evaluate(symbols, pc)
                return encode(
                    name,
                    rd=_register(operands[0], line_no),
                    rs1=_register(operands[1], line_no),
                    imm=shamt,
                    instruction_set=self.instruction_set,
                )
            rd = _register(operands[0], line_no)
            if len(operands) == 2:  # rd, imm(rs1)
                offset, base = _memory_operand(operands[1], line_no)
                imm = _Expr(offset, line_no).evaluate(symbols, pc) if offset else 0
                rs1 = _register(base, line_no)
            elif len(operands) == 3:  # rd, rs1, imm
                rs1 = _register(operands[1], line_no)
                imm = _Expr(operands[2], line_no).evaluate(symbols, pc)
            else:
                raise AssemblerError(f"{mnemonic} takes 2 or 3 operands, got {len(operands)}")
            return encode(name, rd=rd, rs1=rs1, imm=imm, instruction_set=self.instruction_set)
        if fmt == S_TYPE:
            rs2 = _register(operands[0], line_no)
            if len(operands) == 2:  # rs2, imm(rs1)
                offset, base = _memory_operand(operands[1], line_no)
                imm = _Expr(offset, line_no).evaluate(symbols, pc) if offset else 0
                rs1 = _register(base, line_no)
            elif len(operands) == 3:  # rs2, rs1, imm
                rs1 = _register(operands[1], line_no)
                imm = _Expr(operands[2], line_no).evaluate(symbols, pc)
            else:
                raise AssemblerError(f"{mnemonic} takes 2 or 3 operands, got {len(operands)}")
            return encode(name, rs1=rs1, rs2=rs2, imm=imm, instruction_set=self.instruction_set)
        if fmt == B_TYPE:
            need(3)
            offset = self._branch_offset(operands[2], pc, symbols, line_no)
            return encode(
                name,
                rs1=_register(operands[0], line_no),
                rs2=_register(operands[1], line_no),
                imm=offset,
                instruction_set=self.instruction_set,
            )
        if fmt == U_TYPE:
            need(2)
            imm = _Expr(operands[1], line_no).evaluate(symbols, pc)
            return encode(
                name,
                rd=_register(operands[0], line_no),
                imm=imm,
                instruction_set=self.instruction_set,
            )
        if fmt == J_TYPE:
            need(2)
            offset = self._branch_offset(operands[1], pc, symbols, line_no)
            return encode(
                name,
                rd=_register(operands[0], line_no),
                imm=offset,
                instruction_set=self.instruction_set,
            )
        if fmt == CSR_TYPE:
            need(3)
            number = _csr_operand(operands[1], symbols, pc, line_no)
            if name in ("CSRRWI", "CSRRSI", "CSRRCI"):
                # The rs1 field carries a 5-bit unsigned immediate here.
                value = _Expr(operands[2], line_no).evaluate(symbols, pc)
                if not 0 <= value <= 31:
                    raise AssemblerError(
                        f"{mnemonic} immediate {value} out of range 0..31"
                    )
                source = value
            else:
                source = _register(operands[2], line_no)
            return encode(
                name,
                rd=_register(operands[0], line_no),
                rs1=source,
                imm=number,
                instruction_set=self.instruction_set,
            )
        if fmt == FENCE_TYPE:
            if not operands:
                pred = succ = 0xF
            elif len(operands) == 2:
                pred = _fence_field(operands[0], line_no)
                succ = _fence_field(operands[1], line_no)
            else:
                raise AssemblerError("fence takes no operands or pred, succ")
            return encode(name, imm=(pred << 4) | succ, instruction_set=self.instruction_set)
        if fmt == N_TYPE:
            need(0)
            return encode(name, instruction_set=self.instruction_set)
        if fmt == AMO_TYPE:
            # LR.W rd, (rs1) -- no rs2, no offset (there is no immediate field).
            # Everything else: AMOADD.W rd, rs2, (rs1), and likewise SC.W.
            if name.startswith("LR."):
                need(2)
                offset, base = _memory_operand(operands[1], line_no)
                if offset:
                    raise AssemblerError(f"{mnemonic} takes no offset")
                return encode(
                    name,
                    rd=_register(operands[0], line_no),
                    rs1=_register(base, line_no),
                    instruction_set=self.instruction_set,
                )
            need(3)
            offset, base = _memory_operand(operands[2], line_no)
            if offset:
                raise AssemblerError(f"{mnemonic} takes no offset")
            return encode(
                name,
                rd=_register(operands[0], line_no),
                rs2=_register(operands[1], line_no),
                rs1=_register(base, line_no),
                instruction_set=self.instruction_set,
            )
        raise AssemblerError(f"cannot assemble format {fmt}")

    def _branch_offset(self, operand, pc, symbols, line_no):
        """Symbolic operands are absolute addresses; numeric ones are offsets."""
        expr = _Expr(operand, line_no)
        value = expr.evaluate(symbols, pc)
        return value - pc if expr.has_symbols else value


#: Directives that emit bytes, and so are illegal inside ``.bss``.
_DATA_DIRECTIVES = frozenset(
    {".word", ".half", ".short", ".byte", ".string", ".asciz", ".ascii"}
)


def _section_name(operand):
    """Map a ``.section`` operand onto one of text/data/bss."""
    target = operand.strip().lstrip(".").lower()
    if target.startswith("bss"):
        return "bss"
    if target.startswith("data") or target.startswith("rodata"):
        return "data"
    return "text"


#: Symbolic CSR names the assembler accepts, matching what GNU as knows.
_CSR_NAMES = csr_names()


def _csr_operand(text, symbols, pc, line_no):
    """Resolve a CSR operand: a symbolic name, or any constant expression."""
    key = text.strip().lower()
    if key in _CSR_NAMES:
        return _CSR_NAMES[key]
    value = _Expr(text, line_no).evaluate(symbols, pc)
    if not 0 <= value <= 0xFFF:
        raise AssemblerError(f"CSR address {value} out of range 0..0xFFF")
    return value


def _register(text, line_no):
    key = text.strip().lower()
    if key not in REGISTER_ALIASES:
        raise AssemblerError(f"expected a register, got {text!r}")
    return register_number(key)


def _memory_operand(text, line_no):
    match = _MEMORY_OPERAND.match(text)
    if not match:
        raise AssemblerError(f"expected an offset(base) operand, got {text!r}")
    return match.group("offset"), match.group("base")


def _fence_field(text, line_no):
    text = text.strip().lower()
    if text in ("0", ""):
        return 0
    bits = 0
    for char in text:
        if char not in "iorw":
            raise AssemblerError(f"invalid fence field {text!r}")
        bits |= 1 << (3 - "iorw".index(char))
    return bits


def _string_bytes(operand, nul=True):
    text = operand.strip()
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        raise AssemblerError(f"expected a quoted string, got {operand!r}")
    data = _unescape(text[1:-1]).encode("utf-8")
    return data + (b"\x00" if nul else b"")


def assemble(source, text_base=DEFAULT_TEXT_BASE, data_base=None, instruction_set=None) -> Program:
    """Assemble ``source`` text into a :class:`Program`."""
    return Assembler(text_base, data_base, instruction_set).assemble(source)


def assemble_file(path, text_base=DEFAULT_TEXT_BASE, data_base=None, instruction_set=None) -> Program:
    """Assemble a ``.asm`` file into a :class:`Program`."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return Assembler(text_base, data_base, instruction_set).assemble(source, filename=str(path))
