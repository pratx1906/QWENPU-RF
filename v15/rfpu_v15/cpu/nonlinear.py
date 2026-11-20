"""Nonlinear RF kernels built on a branch-free integer CORDIC (vectoring mode).

CORDIC contract (both the RV32 helper `cordic_asm` and the Python reference `cordic_vector`):
  inputs x, y signed 32-bit with |x|, |y| < 2^29.
  Pre-rotation: if x < 0 then (x, y) = (-x, -y) and z0 = 2^31 (== pi), else z0 = 0.
  16 iterations i = 0..15, m = y >> 31 (arithmetic), d(v) = (v ^ m) - m:
      x' = x + d(y >> i),  y' = y - d(x >> i),  z' = z + d(ATAN[i]),
      ATAN[i] = round(atan(2^-i) / pi * 2^31).
  Outputs: magnitude = x_final = K * |x + jy| (+ truncation error), K = prod_i sqrt(1 + 2^-2i)
  ~= 1.64676 (the gain is NOT removed); angle = z_final as int32, binary angle units where
  2^31 == pi (wraps mod 2^32). With |x|,|y| < 2^29 every x, y stays below 2^31 in magnitude.
  The asm uses sign masks only: no data-dependent branch anywhere.
"""
import math

from .codegen import Asm, Layout, word, signed
from .kernels import Built, Kernel, mac, put_pairs, ram_word

ITERATIONS = 16
ATAN = [round(math.atan(2.0 ** -i) / math.pi * 2 ** 31) for i in range(ITERATIONS)]


def gain(iterations: int = ITERATIONS) -> float:
    return math.prod(math.sqrt(1 + 2.0 ** (-2 * i)) for i in range(iterations))


GAIN = gain()  # ~1.6467602
LIMIT = 1 << 29
Q15_SHIFT = 13  # Q15 sample << 13: |32768 << 13| = 2^28 < 2^29
FM_SHIFT = 3    # |Re, Im of x[n] conj(x[n-1])| <= 2 * 2^30 = 2^31  ->  >> 3 gives <= 2^28


def cordic_vector(x: int, y: int, iterations: int = ITERATIONS) -> tuple:
    """Independent reference: returns (magnitude, angle_int32)."""
    if not (-LIMIT < x < LIMIT and -LIMIT < y < LIMIT):
        raise ValueError("CORDIC inputs must satisfy |x|, |y| < 2^29")
    z = 0
    if x < 0:
        x, y, z = -x, -y, 1 << 31
    for i in range(iterations):
        if y < 0:
            x, y, z = x - (y >> i), y + (x >> i), z - ATAN[i]
        else:
            x, y, z = x + (y >> i), y - (x >> i), z + ATAN[i]
    return x, signed(word(z))


def cordic_asm(a: Asm, x: str = "t0", y: str = "t1", z: str = "t2", iterations: int = ITERATIONS, angle: bool = True):
    """Emit the unrolled branch-free CORDIC: x <- magnitude, z <- angle (angle=False: z is not computed).
    Clobbers t3, t4, t5."""
    a(f"srai t3, {x}, 31",                       # m = sign mask of x
      f"xor {x}, {x}, t3", f"sub {x}, {x}, t3",   # x = |x| (pre-rotation by pi when x < 0)
      f"xor {y}, {y}, t3", f"sub {y}, {y}, t3")
    if angle:
        a.li("t4", -(1 << 31))
        a(f"and {z}, t3, t4")                    # z0 = 2^31 & m
    for i in range(iterations):
        a(f"srai t3, {y}, 31",
          f"srai t4, {y}, {i}", "xor t4, t4, t3", "sub t4, t4, t3",
          f"srai t5, {x}, {i}", "xor t5, t5, t3", "sub t5, t5, t3",
          f"add {x}, {x}, t4", f"sub {y}, {y}, t5")
        if angle:
            a.li("t4", ATAN[i])
            a("xor t4, t4, t3", "sub t4, t4, t3", f"add {z}, {z}, t4")


def shift64(a: Asm, lo: str, hi: str, shift: int):
    """lo = (hi:lo) >> shift (arithmetic floor), valid when the result fits 32 bits."""
    a(f"srli {lo}, {lo}, {shift}", f"slli t5, {hi}, {32 - shift}", f"or {lo}, {lo}, t5")


def _decode_words(built, ram, per):
    o = built.layout["out"]
    return [tuple(ram_word(ram, o + 4 * (per * i + k)) for k in range(per)) for i in range(built.block)]


class _PerSample(Kernel):
    """Stateless per-sample CORDIC on (re << 13, im << 13)."""
    per = 2
    iterations = ITERATIONS

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("out", self.per * block)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", block)
        a("lw t0, 0(a0)", "lw t1, 4(a0)", f"slli t0, t0, {Q15_SHIFT}", f"slli t1, t1, {Q15_SHIFT}")
        cordic_asm(a, iterations=self.iterations, angle=self.per == 2)
        a("sw t0, 0(a2)")
        if self.per == 2:
            a("sw t2, 4(a2)")
        a(f"addi a2, a2, {4 * self.per}", "addi a0, a0, 8")
        close()
        a.readback(L["out"], self.per * block)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w


class MagPhase(_PerSample):
    """Per Q15 sample: (magnitude, angle) = CORDIC(re << 13, im << 13).
    magnitude ~= GAIN * 2^13 * |x|; angle in binary units (2^31 == pi)."""
    name = "mag_phase"

    def describe(self):
        return {"kernel": self.name, "pre_shift": Q15_SHIFT, "cordic_iterations": ITERATIONS, "gain": GAIN}

    def decode(self, built, ram, start):
        return _decode_words(built, ram, 2)

    def reference(self, samples):
        return [cordic_vector(re << Q15_SHIFT, im << Q15_SHIFT) for re, im in samples]


class AMEnvelope(_PerSample):
    """Per Q15 sample: magnitude only, ~= gain(iterations) * 2^13 * |x| (magnitude-only CORDIC: no angle)."""
    name = "am_envelope"
    per = 1

    def __init__(self, iterations: int = ITERATIONS):
        if not 1 <= iterations <= ITERATIONS:
            raise ValueError(f"iterations in 1..{ITERATIONS}")
        self.iterations = int(iterations)
        if iterations != ITERATIONS:
            self.name = f"am_envelope_k{iterations}"

    def describe(self):
        return {"kernel": self.name, "pre_shift": Q15_SHIFT, "cordic_iterations": self.iterations,
                "gain": gain(self.iterations)}

    def decode(self, built, ram, start):
        return [v[0] for v in _decode_words(built, ram, 1)]

    def reference(self, samples):
        return [cordic_vector(re << Q15_SHIFT, im << Q15_SHIFT, self.iterations)[0] for re, im in samples]


class FMDemod(Kernel):
    """Instantaneous phase difference: p = x[n] conj(x[n-1]) exact 64-bit, (Re, Im) >> 3
    (arithmetic floor; |.| <= 2^28), angle = CORDIC(p) as int32 binary angle. x[-1] = 0, so the
    first output of a stream is CORDIC(0, 0)'s angle = sum(ATAN) (meaningless but deterministic)."""
    name = "fm_demod"
    history = 1

    def describe(self):
        return {"kernel": self.name, "product_shift": FM_SHIFT, "cordic_iterations": ITERATIONS}

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * (1 + block))
        L.alloc("out", block)
        a = Asm()
        a.li("a0", L["in"] + 8)
        a.li("a2", L["out"])
        close = a.loop("a3", block)
        a("lw s4, 0(a0)", "lw s5, 4(a0)", "lw s6, -8(a0)", "lw s7, -4(a0)",  # a, b, c, d
          "li s0, 0", "li s1, 0", "li s2, 0", "li s3, 0")
        mac(a, "s0", "s1", "s4", "s6")                 # re = a c + b d
        mac(a, "s0", "s1", "s5", "s7")
        mac(a, "s2", "s3", "s5", "s6")                 # im = b c - a d
        mac(a, "s2", "s3", "s4", "s7", subtract=True)
        shift64(a, "s0", "s1", FM_SHIFT)
        shift64(a, "s2", "s3", FM_SHIFT)
        a("mv t0, s0", "mv t1, s2")
        cordic_asm(a)
        a("sw t2, 0(a2)", "addi a2, a2, 4", "addi a0, a0, 8")
        close()
        a.readback(L["out"], block)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        return [v[0] for v in _decode_words(built, ram, 1)]

    def reference(self, samples):
        out, prev = [], (0, 0)
        for a_, b in samples:
            c, d = prev
            re, im = a_ * c + b * d, b * c - a_ * d
            out.append(cordic_vector(re >> FM_SHIFT, im >> FM_SHIFT)[1])
            prev = (a_, b)
        return out


class CFOEstimate(Kernel):
    """Lag autocorrelation per window: windows end at e = window-1 + k*window (hop = window);
    R = sum_{n=e-window+1..e} x[n] conj(x[n-lag]) exact 64-bit (x[<0] = 0), (Re, Im) >> s with
    s = log2(window) + 3 (arithmetic floor; |.| <= window * 2^31 >> s = 2^28), then CORDIC.
    Output (angle, magnitude) at each window end; angle / lag is the per-sample phase rotation."""
    name = "cfo_estimate"

    def __init__(self, lag: int = 1, window: int = 64):
        if not 1 <= lag <= 16:
            raise ValueError("lag 1..16")
        if window < 1 or window > 256 or window & (window - 1):
            raise ValueError("window must be a power of two <= 256")
        self.lag, self.window = lag, window
        self.history, self.hop = lag, window
        self.shift = (window.bit_length() - 1) + 3
        self.name = f"cfo_lag{lag}_w{window}"

    def describe(self):
        return {"kernel": self.name, "lag": self.lag, "window": self.window, "shift": self.shift,
                "cordic_iterations": ITERATIONS}

    def build(self, block):
        if block % self.window:
            raise ValueError("block must be a multiple of window")
        L, W, lag = Layout(), self.window, self.lag
        frames = block // W
        L.alloc("in", 2 * (lag + block))
        L.alloc("out", 2 * frames)
        a = Asm()
        a.li("a0", L["in"] + 8 * lag)   # x[n] for the first sample of the first window
        a.li("a2", L["out"])
        outer = a.loop("a3", frames)
        a("li s0, 0", "li s1, 0", "li s2, 0", "li s3, 0")
        inner = a.loop("a4", W)
        a("lw s4, 0(a0)", "lw s5, 4(a0)", f"lw s6, {-8 * lag}(a0)", f"lw s7, {-8 * lag + 4}(a0)")
        mac(a, "s0", "s1", "s4", "s6")
        mac(a, "s0", "s1", "s5", "s7")
        mac(a, "s2", "s3", "s5", "s6")
        mac(a, "s2", "s3", "s4", "s7", subtract=True)
        a("addi a0, a0, 8")
        inner()
        shift64(a, "s0", "s1", self.shift)
        shift64(a, "s2", "s3", self.shift)
        a("mv t0, s0", "mv t1, s2")
        cordic_asm(a)
        a("sw t2, 0(a2)", "sw t0, 4(a2)", "addi a2, a2, 8")
        outer()
        a.readback(L["out"], 2 * frames)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"], [None] * built.block
        for f in range(built.block // self.window):
            out[(f + 1) * self.window - 1] = (ram_word(ram, o + 8 * f), ram_word(ram, o + 8 * f + 4))
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        x = lambda n: samples[n] if n >= 0 else (0, 0)  # noqa: E731
        for e in range(self.window - 1, len(samples), self.window):
            re = im = 0
            for n in range(e - self.window + 1, e + 1):
                (a_, b), (c, d) = x(n), x(n - self.lag)
                re += a_ * c + b * d
                im += b * c - a_ * d
            mag, ang = cordic_vector(re >> self.shift, im >> self.shift)
            out[e] = (ang, mag)
        return out
