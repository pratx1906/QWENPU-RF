"""Branch-uniform RV32 code generation helpers.

Kernels avoid data-dependent branches so every chunk of a stream has the same trace length;
that lets a batch of machines free-run in lockstep on one CUDA graph.
"""
from dataclasses import dataclass, field

DATA = 0x100  # first usable RAM byte (chipformer.rv.spec.DATA_BASE)
# Proof mode (default): every output/state word is loaded back before halt, so the neural trace itself
# carries each value. Production mode skips that readback; runs remain token-checked against the
# reference trace, but output values are then read from the reference machine's RAM, not the trace.
PROOF = True
RAM_BYTES = 1 << 15


@dataclass
class Layout:
    """Byte addresses of named RAM regions, allocated upward from DATA."""
    top: int = DATA
    regions: dict = field(default_factory=dict)

    def alloc(self, name: str, words: int) -> int:
        address = self.top
        self.regions[name] = (address, words)
        self.top += 4 * words
        if self.top > RAM_BYTES - 1024:  # keep 1 KB for the stack
            raise MemoryError(f"RAM layout exceeds 31 KB at {name}")
        return address

    def __getitem__(self, name):
        return self.regions[name][0]


class Asm:
    def __init__(self):
        self.lines = ["_start:"]
        self.serial = 0

    def __call__(self, *lines):
        self.lines.extend(lines)

    def fresh(self, stem: str) -> str:
        self.serial += 1
        return f"{stem}{self.serial}"

    def li(self, reg: str, value: int):
        self(f"li {reg}, {int(value)}")

    def round_shift(self, reg: str, shift: int, tmp: str = "t6"):
        """reg = (reg + 2^(shift-1)) >> shift, arithmetic (round half up)."""
        if shift:
            self(f"li {tmp}, {1 << (shift - 1)}", f"add {reg}, {reg}, {tmp}", f"srai {reg}, {reg}, {shift}")

    def loop(self, counter: str, count: int):
        """Open a fixed-count loop; returns the closing callable."""
        top = self.fresh("loop")
        self.li(counter, count)
        self(f"{top}:")
        opened = len(self.lines)

        def close():
            self(f"addi {counter}, {counter}, -1")
            # Each source line is at most two instructions (li); branches reach +-4 KB, so a
            # long body closes with a branch over an unconditional jump (reach +-1 MB).
            if 2 * (len(self.lines) - opened) < 1000:
                self(f"bnez {counter}, {top}")
            else:
                done = self.fresh("done")
                self(f"beqz {counter}, {done}", f"j {top}", f"{done}:")
        return close

    def readback(self, address: int, words: int):
        """Load every output word, so the trace itself carries each output value (proof mode)."""
        if not PROOF:
            return
        self.li("a6", address)
        close = self.loop("a7", words)
        self("lw t0, 0(a6)", "addi a6, a6, 4")
        close()

    def halt(self):
        self("li a0, 0", "li a7, 93", "ecall")

    def source(self) -> str:
        return "\n".join(self.lines) + "\n"


def round_half_up(x: int, shift: int) -> int:
    return (x + (1 << (shift - 1))) >> shift if shift else x


def word(v: int) -> int:
    return v & 0xFFFFFFFF


def signed(w: int) -> int:
    return w - (1 << 32) if w & 0x80000000 else w
