"""Filtering / mixing Q15 kernels: decimating FIR, IIR biquad, DC blocker, NCO mixer.

Same contract as kernels.py: all sample arithmetic is RV32 code with no data-dependent branches
(saturation and selection use sra masks); the host only writes RAM words (inputs, runtime
parameters, tables, state) and decodes words the program loaded back before halting.
Each kernel's `reference` is an independent pure-Python integer implementation of the math.
"""
import math

from .codegen import Asm, Layout, round_half_up, word, signed
from .kernels import FIR, Built, Kernel, Q, mac, narrow64, put_pairs, ram_word

INT16_MIN, INT16_MAX = -32768, 32767


def saturate_int16(v: int) -> int:
    return INT16_MAX if v > INT16_MAX else INT16_MIN if v < INT16_MIN else v


def q15(v: int, what: str) -> int:
    v = int(v)
    if not INT16_MIN <= v <= INT16_MAX:
        raise ValueError(f"{what} must be a 16-bit code")
    return v


def sat16(a: Asm, r: str, ta: str = "t2", tb: str = "t3"):
    """r = clamp(r, -32768, 32767), branch-free; valid for |r| < 2^30."""
    a(f"li {ta}, {INT16_MAX}", f"sub {tb}, {r}, {ta}", f"srai {r}, {tb}, 31",  # mask: r < 32767
      f"and {tb}, {tb}, {r}", f"add {r}, {ta}, {tb}")                           # min(r, 32767)
    a(f"li {ta}, {INT16_MIN}", f"sub {tb}, {r}, {ta}", f"srai {r}, {tb}, 31",  # mask: r < -32768
      f"not {r}, {r}", f"and {tb}, {tb}, {r}", f"add {r}, {ta}, {tb}")          # max(r, -32768)


def finish_stateful(a: Asm, L: Layout, regs: list, out_words: int):
    """Store final state registers, then load back every output and state word."""
    a.li("a1", L["state"])
    for i, r in enumerate(regs):
        a(f"sw {r}, {4*i}(a1)")
    a.readback(L["out"], out_words)
    a.readback(L["state"], len(regs))
    a.halt()


# ---------------------------------------------------------------------------------------------
class DecimatingFIR(FIR):
    """y[n] = round_half_up(sum_k h[k] x[n-k], 15) for n % decim == decim-1 only (others None).
    Complex Q15 taps, zero initial history; FIR's 32/64-bit precision selection."""
    name = "decim_fir"

    def __init__(self, taps: list, decim: int, name: str | None = None):
        super().__init__(taps, name)
        if not 1 <= int(decim) <= 64:
            raise ValueError("decim 1..64")
        self.decim = self.hop = int(decim)
        self.name = name or f"decim{self.decim}_fir{len(self.taps)}"

    def describe(self):
        return super().describe() | {"decim": self.decim}

    def build(self, block):
        if block % self.decim:
            raise ValueError("block must be a multiple of decim")
        L, T, D = Layout(), len(self.taps), self.decim
        outs = block // D
        L.alloc("in", 2 * (self.history + block))
        L.alloc("taps", 2 * T)
        L.alloc("out", 2 * outs)
        a = Asm()
        cached = T <= 5
        regs = ["s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"]
        if cached:
            a.li("a1", L["taps"])
            for k in range(T):
                a(f"lw {regs[2*k]}, {8*k}(a1)", f"lw {regs[2*k+1]}, {8*k+4}(a1)")
        acc_hi = ("a4", "a5") if self.wide else (None, None)
        a.li("a0", L["in"] + 8 * (self.history + D - 1))
        a.li("a2", L["out"])
        a.li("a1", L["taps"])
        close = a.loop("a3", outs)
        a("li s0, 0", "li s1, 0")
        if self.wide:
            a("li a4, 0", "li a5, 0")
        for k in range(T):
            a(f"lw t0, {-8*k}(a0)", f"lw t1, {-8*k+4}(a0)")
            if cached:
                c0, c1 = regs[2*k], regs[2*k+1]
            else:
                a(f"lw t2, {8*k}(a1)", f"lw t3, {8*k+4}(a1)")
                c0, c1 = "t2", "t3"
            mac(a, "s0", acc_hi[0], "t0", c0)
            mac(a, "s0", acc_hi[0], "t1", c1, subtract=True)
            mac(a, "s1", acc_hi[1], "t0", c1)
            mac(a, "s1", acc_hi[1], "t1", c0)
        if self.wide:
            narrow64(a, "s0", "a4", Q)
            narrow64(a, "s1", "a5", Q)
        else:
            a.round_shift("s0", Q)
            a.round_shift("s1", Q)
        a("sw s0, 0(a2)", "sw s1, 4(a2)", "addi a2, a2, 8")
        a.li("t0", 8 * D)
        a("add a0, a0, t0")
        close()
        a.readback(L["out"], 2 * outs)
        a.halt()
        return Built(a.source(), L, block)

    def decode(self, built, ram, start):
        o, D = built.layout["out"], self.decim
        out = [None] * built.block
        for j in range(built.block // D):
            i = j * D + D - 1
            out[i] = (ram_word(ram, o + 8*j), ram_word(ram, o + 8*j + 4))
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        for n in range(self.decim - 1, len(samples), self.decim):
            re = im = 0
            for k, (hr, hi) in enumerate(self.taps):
                if n - k >= 0:
                    xr, xi = samples[n - k]
                    re += hr * xr - hi * xi
                    im += hr * xi + hi * xr
            out[n] = (round_half_up(re, Q), round_half_up(im, Q))
        return out


# ---------------------------------------------------------------------------------------------
class IIRBiquad(Kernel):
    """Direct form I biquad, real Q14 coefficients, applied to I and Q independently:
    acc = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2 (exact 64-bit); y = sat16(round_half_up(acc, 14)).
    wide=True (PROTOCOL Amendment 6): y = round_half_up(acc, 14) as int32 (no saturation), so a filter
    with gain above 1 keeps its output; the reference raises if |y| reaches 2^31.
    out_frac=f (wide only, PROTOCOL Amendment 12): b0..b2 are stored x 2^f, so y carries f extra
    fractional bits (units of 2^-f input LSB) and the feedback uses them; the rounding floor drops 4^f.
    State (8 words): x1, x2, y1, y2 of I, then of Q."""
    name = "iir_biquad"
    state_words = 8
    SHIFT = 14

    def __init__(self, b0, b1, b2, a1, a2, name: str | None = None, wide: bool = False, out_frac: int = 0):
        self.coefs = tuple(q15(v, "coefficient") for v in (b0, b1, b2, a1, a2))
        self.wide = bool(wide)
        if out_frac and not wide or not 0 <= out_frac <= 8:
            raise ValueError("out_frac needs wide=True and 0..8 bits")
        self.out_frac = int(out_frac)
        if name:
            self.name = name
        elif wide:
            self.name = "iir_biquad_wide" + (f"_f{out_frac}" if out_frac else "")

    def _stored(self):
        f = self.out_frac
        b0, b1, b2, a1, a2 = self.coefs
        return b0 << f, b1 << f, b2 << f, a1, a2

    def describe(self):
        return {"kernel": self.name, "coefs_q14": list(self.coefs), "accumulator_bits": 64,
                "output": "int32" if self.wide else "sat16", "output_fraction_bits": self.out_frac}

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("coefs", 5)
        L.alloc("state", 8)
        L.alloc("out", 2 * block)
        a = Asm()
        b0, b1, b2, a1, a2 = "s2", "s3", "s4", "s5", "s6"
        comps = [("s7", "s8", "s9", "s10"), ("s11", "a6", "a4", "a5")]  # x1, x2, y1, y2
        state = [r for c in comps for r in c]
        a.li("a1", L["coefs"])
        for k, r in enumerate((b0, b1, b2, a1, a2)):
            a(f"lw {r}, {4*k}(a1)")
        a.li("a1", L["state"])
        for i, r in enumerate(state):
            a(f"lw {r}, {4*i}(a1)")
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", block)
        for c, (x1, x2, y1, y2) in enumerate(comps):
            a(f"lw t0, {4*c}(a0)", "li s0, 0", "li s1, 0")
            mac(a, "s0", "s1", "t0", b0)
            mac(a, "s0", "s1", x1, b1)
            mac(a, "s0", "s1", x2, b2)
            mac(a, "s0", "s1", y1, a1, subtract=True)
            mac(a, "s0", "s1", y2, a2, subtract=True)
            narrow64(a, "s0", "s1", self.SHIFT)  # the result fits 32 bits (wide: checked by the reference)
            if not self.wide:
                sat16(a, "s0")
            a(f"mv {x2}, {x1}", f"mv {x1}, t0", f"mv {y2}, {y1}", f"mv {y1}, s0", f"sw s0, {4*c}(a2)")
        a("addi a0, a0, 8", "addi a2, a2, 8")
        close()
        finish_stateful(a, L, state, 2 * block)
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for k, v in enumerate(self._stored()):
            w[built.layout["coefs"] + 4 * k] = word(v)
        return w

    def read_state(self, built, ram):
        return [signed(v) for v in super().read_state(built, ram)]

    def decode(self, built, ram, start):
        o = built.layout["out"]
        return [(ram_word(ram, o + 8*i), ram_word(ram, o + 8*i + 4)) for i in range(built.block)]

    def reference(self, samples):
        b0, b1, b2, a1, a2 = self._stored()
        ys = []
        for c in range(2):
            x1 = x2 = y1 = y2 = 0
            col = []
            for s in samples:
                x = s[c]
                acc = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                y = round_half_up(acc, self.SHIFT)
                if self.wide:
                    if not -(1 << 31) <= y < 1 << 31:
                        raise OverflowError("IIR wide output reaches 2^31")
                else:
                    y = saturate_int16(y)
                x2, x1, y2, y1 = x1, x, y1, y
                col.append(y)
            ys.append(col)
        return list(zip(*ys))


# ---------------------------------------------------------------------------------------------
class DCBlocker(Kernel):
    """y[n] = sat16(x[n] - x[n-1] + round_half_up(alpha y[n-1], 15)), I and Q independent.
    State (4 words): x[n-1], y[n-1] of I, then of Q."""
    name = "dc_blocker"
    state_words = 4

    def __init__(self, alpha: int = 32440, name: str | None = None):
        self.alpha = q15(alpha, "alpha")
        if name:
            self.name = name

    def describe(self):
        return {"kernel": self.name, "alpha_q15": self.alpha}

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("alpha", 1)
        L.alloc("state", 4)
        L.alloc("out", 2 * block)
        a = Asm()
        comps = [("s2", "s3"), ("s4", "s5")]  # (x1, y1)
        state = [r for c in comps for r in c]
        a.li("a1", L["alpha"])
        a("lw s6, 0(a1)")
        a.li("a1", L["state"])
        for i, r in enumerate(state):
            a(f"lw {r}, {4*i}(a1)")
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", block)
        for c, (x1, y1) in enumerate(comps):
            a(f"lw t0, {4*c}(a0)", f"sub t1, t0, {x1}", f"mul s0, s6, {y1}")
            a.round_shift("s0", Q)
            a("add t1, t1, s0")
            sat16(a, "t1")
            a(f"mv {x1}, t0", f"mv {y1}, t1", f"sw t1, {4*c}(a2)")
        a("addi a0, a0, 8", "addi a2, a2, 8")
        close()
        finish_stateful(a, L, state, 2 * block)
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        w[built.layout["alpha"]] = word(self.alpha)
        return w

    def read_state(self, built, ram):
        return [signed(v) for v in super().read_state(built, ram)]

    def decode(self, built, ram, start):
        o = built.layout["out"]
        return [(ram_word(ram, o + 8*i), ram_word(ram, o + 8*i + 4)) for i in range(built.block)]

    def reference(self, samples):
        cols = []
        for c in range(2):
            xp = yp = 0
            col = []
            for s in samples:
                y = saturate_int16(s[c] - xp + round_half_up(self.alpha * yp, Q))
                xp, yp = s[c], y
                col.append(y)
            cols.append(col)
        return list(zip(*cols))


# ---------------------------------------------------------------------------------------------
TABLE_BITS = 10
TABLE_SIZE = 1 << TABLE_BITS


def sine_table() -> list:
    """Q15 sin(2 pi i / 1024), rounded half up, clipped to 32767."""
    return [min(INT16_MAX, math.floor(32768 * math.sin(2 * math.pi * i / TABLE_SIZE) + 0.5))
            for i in range(TABLE_SIZE)]


class NCOMixer(Kernel):
    """Complex mix by a 32-bit phase-accumulator NCO with a 1024-entry Q15 sine table in RAM.
    idx = phase >> 22; s = T[idx], c = T[(idx + 256) & 1023];
    y = (rhu(xr c - xi s, 15), rhu(xr s + xi c, 15)); phase += freq_word (mod 2^32) per sample.
    interpolate=True (PROTOCOL Amendment 4): index rounded to nearest, pr = phase + 2^21, idx = pr >> 22,
    d = (pr & (2^22 - 1)) - 2^21 (fractional phase, delta = d 2 pi / 2^32 rad), m = d * 804 (804 ~ 2 pi 128);
    first-order Taylor: s += rhu(mulh(c, m), 7), c -= rhu(mulh(s, m), 7) (both from the table values).
    State (1 word): phase."""
    name = "nco_mixer"
    state_words = 1
    K = 804

    def __init__(self, freq_word: int, name: str | None = None, interpolate: bool = False):
        self.freq_word = int(freq_word) & 0xFFFFFFFF
        self.table = sine_table()
        self.interpolate = bool(interpolate)
        if name:
            self.name = name
        elif interpolate:
            self.name = "nco_mixer_interp"

    def describe(self):
        return {"kernel": self.name, "freq_word": self.freq_word, "table": TABLE_SIZE, "interpolate": self.interpolate}

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("freq", 1)
        L.alloc("state", 1)
        L.alloc("out", 2 * block)
        L.alloc("table", TABLE_SIZE)
        a = Asm()
        a.li("a1", L["freq"])
        a("lw s1, 0(a1)")
        a.li("a1", L["state"])
        a("lw s0, 0(a1)")
        a.li("s2", L["table"])
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        if self.interpolate:
            a.li("s7", 1 << 21)
            a.li("s8", (1 << 22) - 1)
            a.li("s9", self.K)
        close = a.loop("a3", block)
        if self.interpolate:
            a("add t4, s0, s7", f"srli t2, t4, {32 - TABLE_BITS}")                         # rounded index
        else:
            a(f"srli t2, s0, {32 - TABLE_BITS}")
        a("sh2add t3, t2, s2", "lw s3, 0(t3)",                                            # sin
          f"addi t2, t2, {TABLE_SIZE // 4}", f"andi t2, t2, {TABLE_SIZE - 1}",
          "sh2add t3, t2, s2", "lw s4, 0(t3)")                                            # cos
        if self.interpolate:
            # |d| <= 2^21, so |m| < 2^31; corrections |.| <= 101
            a("and t4, t4, s8", "sub t4, t4, s7", "mul t4, t4, s9",
              "mulh t5, s4, t4", "addi t5, t5, 64", "srai t5, t5, 7",
              "mulh t6, s3, t4", "addi t6, t6, 64", "srai t6, t6, 7",
              "add s3, s3, t5", "sub s4, s4, t6")
        a("lw t0, 0(a0)", "lw t1, 4(a0)",
          "mul s5, t0, s4", "mul t5, t1, s3", "sub s5, s5, t5",   # |.| <= sqrt2 * 2^30 < 2^31
          "mul s6, t0, s3", "mul t5, t1, s4", "add s6, s6, t5")
        a.round_shift("s5", Q)
        a.round_shift("s6", Q)
        a("sw s5, 0(a2)", "sw s6, 4(a2)", "add s0, s0, s1", "addi a0, a0, 8", "addi a2, a2, 8")
        close()
        finish_stateful(a, L, ["s0"], 2 * block)
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        w[built.layout["freq"]] = self.freq_word
        t = built.layout["table"]
        for i, v in enumerate(self.table):
            w[t + 4 * i] = word(v)
        return w

    def decode(self, built, ram, start):
        o = built.layout["out"]
        return [(ram_word(ram, o + 8*i), ram_word(ram, o + 8*i + 4)) for i in range(built.block)]

    def reference(self, samples):
        out, phase = [], 0
        for xr, xi in samples:
            pr = (phase + (1 << 21)) % (1 << 32) if self.interpolate else phase
            idx = pr >> 22
            s = min(32767, math.floor(32768 * math.sin(2 * math.pi * idx / 1024) + 0.5))
            ci = (idx + 256) % 1024
            c = min(32767, math.floor(32768 * math.sin(2 * math.pi * ci / 1024) + 0.5))
            if self.interpolate:
                m = ((pr & ((1 << 22) - 1)) - (1 << 21)) * self.K
                s, c = s + round_half_up((c * m) >> 32, 7), c - round_half_up((s * m) >> 32, 7)
            out.append((round_half_up(xr * c - xi * s, 15), round_half_up(xr * s + xi * c, 15)))
            phase = (phase + self.freq_word) % (1 << 32)
        return out
