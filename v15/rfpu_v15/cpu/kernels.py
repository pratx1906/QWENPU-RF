"""Streaming Q15 RF kernels: RV32 program generators plus independent integer references.

A kernel processes one chunk: `history` context samples followed by `block` new samples.
Outputs are indexed by absolute input position; positions that emit nothing hold None.
Runtime parameters (taps, windows) are RAM data, not code, so one program serves any values.
"""
from dataclasses import dataclass
from .codegen import Asm, Layout, round_half_up, word, signed

Q = 15


@dataclass
class Built:
    source: str
    layout: Layout
    block: int


class Kernel:
    """Stateless kernels recompute `history` context samples per chunk, so chunks run in parallel.
    Stateful kernels (state_words > 0) keep words in a RAM region named "state": the program reads
    them at start, writes them back at the end and loads them in its readback, so the trace carries
    them; the runner passes the verified final state into the next chunk of the same stream."""
    name = "kernel"
    history = 0
    hop = 1  # chunk starts are multiples of hop, so emitting positions are block-invariant
    state_words = 0

    def build(self, block: int) -> Built: raise NotImplementedError
    def init_words(self, built: Built, context: list, start: int) -> dict: raise NotImplementedError
    def decode(self, built: Built, ram: list, start: int) -> list: raise NotImplementedError
    def reference(self, samples: list) -> list: raise NotImplementedError
    def describe(self) -> dict: return {"kernel": self.name}

    def initial_state(self) -> list:
        return [0] * self.state_words

    def put_state(self, built: Built, words: dict, state: list):
        for i, v in enumerate(state):
            words[built.layout["state"] + 4 * i] = word(v)

    def read_state(self, built: Built, ram: list) -> list:
        return [ram[(built.layout["state"] >> 2) + i] for i in range(self.state_words)]


def put_pairs(words: dict, address: int, pairs):
    for i, (re, im) in enumerate(pairs):
        words[address + 8 * i] = word(re)
        words[address + 8 * i + 4] = word(im)


def ram_word(ram: list, address: int) -> int:
    return signed(ram[address >> 2])


def mac(asm: Asm, acc_lo: str, acc_hi: str | None, a: str, b: str, subtract: bool = False):
    """acc (+/-)= a*b; 32-bit when acc_hi is None, else exact 64-bit with carry/borrow."""
    if acc_hi is None:
        asm(f"mul t5, {a}, {b}", f"{'sub' if subtract else 'add'} {acc_lo}, {acc_lo}, t5")
        return
    asm(f"mul t5, {a}, {b}", f"mulh t6, {a}, {b}")
    if subtract:
        asm(f"sltu t4, {acc_lo}, t5", f"sub {acc_lo}, {acc_lo}, t5", f"sub {acc_hi}, {acc_hi}, t6",
            f"sub {acc_hi}, {acc_hi}, t4")
    else:
        asm(f"add {acc_lo}, {acc_lo}, t5", f"sltu t4, {acc_lo}, t5", f"add {acc_hi}, {acc_hi}, t6",
            f"add {acc_hi}, {acc_hi}, t4")


def narrow64(asm: Asm, lo: str, hi: str, shift: int):
    """lo = round_half_up(hi:lo, shift), valid when the result fits 32 bits."""
    asm(f"li t5, {1 << (shift - 1)}", f"add {lo}, {lo}, t5", f"sltu t4, {lo}, t5", f"add {hi}, {hi}, t4",
        f"srli {lo}, {lo}, {shift}", f"slli t5, {hi}, {32 - shift}", f"or {lo}, {lo}, t5")


class FIR(Kernel):
    """y[n] = round_half_up(sum_k h[k] x[n-k], 15); complex taps and samples, zero initial history.

    Precision selection: 32-bit accumulation only when a range bound proves it cannot overflow,
    otherwise exact 64-bit accumulation. Coefficients up to 5 taps are held in registers."""
    name = "fir"

    def __init__(self, taps: list, name: str | None = None):
        if not 1 <= len(taps) <= 64:
            raise ValueError("1..64 taps")
        if any(not -32768 <= v <= 32767 for t in taps for v in t):
            raise ValueError("taps must be Q15 codes")
        self.taps = [tuple(map(int, t)) for t in taps]
        self.history = len(taps) - 1
        self.name = name or f"fir{len(taps)}"
        bound = sum(abs(a) + abs(b) for a, b in self.taps) * 32768
        self.wide = bound >= 1 << 31  # worst-case |accumulator| reaches 2^31

    def describe(self):
        return {"kernel": self.name, "taps": len(self.taps), "accumulator_bits": 64 if self.wide else 32}

    def build(self, block):
        L, T = Layout(), len(self.taps)
        L.alloc("in", 2 * (self.history + block))
        L.alloc("taps", 2 * T)
        L.alloc("out", 2 * block)
        a = Asm()
        cached = T <= 5
        regs = ["s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"]
        if cached:
            a.li("a1", L["taps"])
            for k in range(T):
                a(f"lw {regs[2*k]}, {8*k}(a1)", f"lw {regs[2*k+1]}, {8*k+4}(a1)")
        a.li("a0", L["in"] + 8 * self.history)
        a.li("a2", L["out"])
        a.li("a1", L["taps"])
        close = a.loop("a3", block)
        a("li s0, 0", "li s1, 0")
        if self.wide:
            a("li a4, 0", "li a5, 0")
        hr, hi = ("s0", "a4"), ("s1", "a5")
        for k in range(T):
            a(f"lw t0, {-8*k}(a0)", f"lw t1, {-8*k+4}(a0)")
            if cached:
                c0, c1 = regs[2*k], regs[2*k+1]
            else:
                a(f"lw t2, {8*k}(a1)", f"lw t3, {8*k+4}(a1)")
                c0, c1 = "t2", "t3"
            mac(a, "s0", "a4" if self.wide else None, "t0", c0)
            mac(a, "s0", "a4" if self.wide else None, "t1", c1, subtract=True)
            mac(a, "s1", "a5" if self.wide else None, "t0", c1)
            mac(a, "s1", "a5" if self.wide else None, "t1", c0)
        if self.wide:
            narrow64(a, "s0", "a4", Q)
            narrow64(a, "s1", "a5", Q)
        else:
            a.round_shift("s0", Q)
            a.round_shift("s1", Q)
        a("sw s0, 0(a2)", "sw s1, 4(a2)", "addi a2, a2, 8", "addi a0, a0, 8")
        close()
        a.readback(L["out"], 2 * block)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        put_pairs(w, built.layout["taps"], self.taps)
        return w

    def decode(self, built, ram, start):
        o = built.layout["out"]
        return [(ram_word(ram, o + 8*i), ram_word(ram, o + 8*i + 4)) for i in range(built.block)]

    def reference(self, samples):
        out = []
        for n in range(len(samples)):
            re = im = 0
            for k, (h0, h1) in enumerate(self.taps):
                if n - k < 0:
                    break
                x0, x1 = samples[n - k]
                re += x0 * h0 - x1 * h1
                im += x0 * h1 + x1 * h0
            out.append((round_half_up(re, Q), round_half_up(im, Q)))
        return out


class FFT4Power(Kernel):
    """Frames of 4 samples ending at positions e >= 3 with (e-3) % hop == 0; Q15 window
    xw = round_half_up(w*x, 15); exact 4-point DFT; power |X_k|^2 as exact 64-bit Q30."""
    name = "fft4_power"
    history = 3

    def __init__(self, window=(0, 24576, 24576, 0), hop=2):
        if hop not in (1, 2, 4):
            raise ValueError("hop 1, 2 or 4")
        self.window, self.hop = tuple(map(int, window)), hop

    def describe(self):
        return {"kernel": self.name, "window": list(self.window), "hop": self.hop}

    def frames(self, block):
        return [i for i in range(block) if (i - 3) % self.hop == 0]

    def build(self, block):
        if block % self.hop:
            raise ValueError("block must be a multiple of hop")
        L = Layout()
        frames = self.frames(block)
        L.alloc("in", 2 * (3 + block))
        L.alloc("window", 4)
        L.alloc("out", 8 * len(frames))
        a = Asm()
        a.li("a1", L["window"])
        a("lw s8, 0(a1)", "lw s9, 4(a1)", "lw s10, 8(a1)", "lw s11, 12(a1)")
        first = frames[0]  # frame ending at local i reads context samples i .. i+3
        a.li("a0", L["in"] + 8 * first)
        a.li("a2", L["out"])
        close = a.loop("a3", len(frames))
        names = [("s0", "s1"), ("s2", "s3"), ("s4", "s5"), ("s6", "s7")]
        for k, (r, i) in enumerate(names):
            w = ["s8", "s9", "s10", "s11"][k]
            a(f"lw {r}, {8*k}(a0)", f"lw {i}, {8*k+4}(a0)", f"mul {r}, {r}, {w}", f"mul {i}, {i}, {w}")
            a.round_shift(r, Q)
            a.round_shift(i, Q)
        # a=s0,s1 b=s2,s3 c=s4,s5 d=s6,s7
        a("add t0, s0, s4", "add t1, s1, s5",   # a+c
          "add t2, s2, s6", "add t3, s3, s7",   # b+d
          "sub s4, s0, s4", "sub s5, s1, s5",   # a-c
          "sub s6, s2, s6", "sub s7, s3, s7",   # b-d
          "add s0, t0, t2", "add s1, t1, t3",   # X0
          "sub s2, t0, t2", "sub s3, t1, t3")   # X2
        a("add t0, s4, s7", "sub t1, s5, s6",   # X1 = (a-c) - j(b-d)
          "sub t2, s4, s7", "add t3, s5, s6")   # X3 = (a-c) + j(b-d)
        bins = [("s0", "s1"), ("t0", "t1"), ("s2", "s3"), ("t2", "t3")]
        for k, (r, i) in enumerate(bins):
            a(f"mul a4, {r}, {r}", f"mulh a5, {r}, {r}", f"mul t5, {i}, {i}", f"mulh t6, {i}, {i}",
              "add a4, a4, t5", "sltu t4, a4, t5", "add a5, a5, t6", "add a5, a5, t4",
              f"sw a4, {8*k}(a2)", f"sw a5, {8*k+4}(a2)")
        a("addi a2, a2, 32", f"addi a0, a0, {8 * self.hop}")
        close()
        a.readback(L["out"], 8 * len(frames))
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for k, v in enumerate(self.window):
            w[built.layout["window"] + 4 * k] = word(v)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"], [None] * built.block
        for f, i in enumerate(self.frames(built.block)):
            powers = []
            for k in range(4):
                lo, hi = ram[(o + 32*f + 8*k) >> 2], ram[(o + 32*f + 8*k + 4) >> 2]
                powers.append(signed(hi) * (1 << 32) + lo)
            out[i] = tuple(powers) if start + i >= 3 else None
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(3, len(samples)):
            if (e - 3) % self.hop:
                continue
            x = [(round_half_up(re * w, Q), round_half_up(im * w, Q))
                 for (re, im), w in zip(samples[e - 3:e + 1], self.window)]
            powers = []
            for k in range(4):
                re = im = 0
                for n, (xr, xi) in enumerate(x):
                    c, s = [(1, 0), (0, -1), (-1, 0), (0, 1)][(k * n) % 4]
                    re += xr * c - xi * s
                    im += xr * s + xi * c
                powers.append(re * re + im * im)
            out[e] = tuple(powers)
        return out
