"""Spectral RF kernels on the RV32 CPU: radix-2 FFT power/complex bins, Welch PSD, CA-CFAR, single-bin DFT.

Numeric contract (all integer, Q15, round_half_up = (x + 2^(s-1)) >> s, arithmetic):

* Frames of n samples end at positions e >= n-1 with (e-(n-1)) % hop == 0.
* Window: xw = round_half_up(w * x, 15) per component (w are runtime Q15 RAM codes).
* Twiddles: W_j = (clip(rnd(32768 cos(2 pi j/n))), clip(rnd(-32768 sin(2 pi j/n)))), rnd = floor(v + 1/2),
  clip to [-32768, 32767]; written to RAM as runtime data with a bit-reversal byte-offset table.
* Radix-2 decimation in time, in place: t = round_half_up(W*b, 15) (complex, each component from the exact
  32-bit sum of two products), a' = round_half_up(a + t, 1), b' = round_half_up(a - t, 1), so X = DFT/n.
  Range: |xw| <= 32768*sqrt(2) and each stage keeps the complex magnitude <= max(|a|,|b|) + 2, so every
  component stays below 46345 and every W*b component below 32768 * 46345 < 2^31 (32-bit exact).
* Power re^2 + im^2 < 2^32 is exact 64-bit (two words, low first).

Code is branch-uniform: fixed-count loops only; selection/argmax through sltu masks. The FFT is one
subroutine (call/ret) shared by every frame. Every output word is loaded back before halt.
"""
import math

from .codegen import Asm, Layout, round_half_up, word, signed
from .kernels import Built, Kernel, put_pairs, mac, narrow64

Q = 15
SIZES = (8, 16, 32, 64, 128, 256)


def q15(v: float) -> int:
    return max(-32768, min(32767, math.floor(32768 * v + 0.5)))


def twiddle(j: int, n: int) -> tuple:
    a = 2 * math.pi * j / n
    return q15(math.cos(a)), q15(-math.sin(a))


def hann(n: int) -> list:
    return [q15(0.5 - 0.5 * math.cos(2 * math.pi * i / n)) for i in range(n)]


def bitrev(i: int, bits: int) -> int:
    return int(format(i, f"0{bits}b")[::-1], 2) if bits else 0


def u64(ram: list, address: int) -> int:
    return signed(ram[(address >> 2) + 1]) * (1 << 32) + ram[address >> 2]


# ----------------------------------------------------------------------------------------------
# independent reference (recursive radix-2 DIT, no bit-reversal table)

def ref_fft(x: list, n: int) -> list:
    N = len(x)
    if N == 1:
        return list(x)
    E, O = ref_fft(x[0::2], n), ref_fft(x[1::2], n)
    out = [None] * N
    for k in range(N // 2):
        wr, wi = twiddle(k * (n // N), n)
        br, bi = O[k]
        tr = round_half_up(wr * br - wi * bi, Q)
        ti = round_half_up(wr * bi + wi * br, Q)
        ar, ai = E[k]
        out[k] = (round_half_up(ar + tr, 1), round_half_up(ai + ti, 1))
        out[k + N // 2] = (round_half_up(ar - tr, 1), round_half_up(ai - ti, 1))
    return out


def ref_frame(samples: list, e: int, n: int, window) -> list:
    x = [(round_half_up(re * w, Q), round_half_up(im * w, Q)) for (re, im), w in zip(samples[e - n + 1:e + 1], window)]
    return ref_fft(x, n)


def ref_powers(bins):
    return [re * re + im * im for re, im in bins]


def onesabs(v: int) -> int:
    """v ^ (v >> 31): |v| for v >= 0, |v| - 1 for v < 0."""
    return v if v >= 0 else ~v


def ref_frame_block(samples: list, e: int, n: int, window) -> tuple:
    """Block-floating-point FFT of one frame (iterative, written independently of the program).
    p = x*w exact; r = bitlen(OR|p| >> 14); xw = round_half_up(p, r). Stage k scales by 1/2 (rounded
    half up) only if its input has a component with onesabs >= 2^14. Returns (exponent, bins) with
    bins ~= DFT(x*w / 2^15) / n * 2^exponent, exponent = 15 + log2(n) - r - (number of scaled stages)."""
    L = n.bit_length() - 1
    p = [(re * w, im * w) for (re, im), w in zip(samples[e - n + 1:e + 1], window)]
    acc = 0
    for re, im in p:
        acc |= onesabs(re) | onesabs(im)
    r = (acc >> 14).bit_length()
    buf = [None] * n
    for i, (re, im) in enumerate(p):
        buf[bitrev(i, L)] = (round_half_up(re, r), round_half_up(im, r))
    scaled = 0
    for s in range(L):
        h = 1 << s
        acc = 0
        for re, im in buf:
            acc |= onesabs(re) | onesabs(im)
        sh = 1 if acc >> 14 else 0
        scaled += sh
        for g in range(0, n, 2 * h):
            for k in range(h):
                wr, wi = twiddle(k * (n // (2 * h)), n)
                (ar, ai), (br, bi) = buf[g + k], buf[g + k + h]
                pr, pi = wr * br - wi * bi, wr * bi + wi * br
                assert -2**31 <= pr < 2**31 and -2**31 <= pi < 2**31
                tr, ti = round_half_up(pr, Q), round_half_up(pi, Q)
                buf[g + k] = ((ar + tr + sh) >> sh, (ai + ti + sh) >> sh)
                buf[g + k + h] = ((ar - tr + sh) >> sh, (ai - ti + sh) >> sh)
    return 15 + L - r - scaled, buf


# ----------------------------------------------------------------------------------------------

class _Framed(Kernel):
    """Frames of n samples, hop between frames; frame ending at local i starts at context index i - (n-1) + history."""

    def _check(self, n, hop):
        if n not in SIZES:
            raise ValueError(f"n in {SIZES}")
        if hop < 1 or hop > n or hop & (hop - 1):
            raise ValueError("hop must be a power of two <= n")
        self.n, self.L = n, n.bit_length() - 1

    def _window(self, window):
        window = hann(self.n) if window is None else [int(v) for v in window]
        if len(window) != self.n or any(not -32768 <= v <= 32767 for v in window):
            raise ValueError("window: n Q15 codes")
        return window

    def max_block(self, limit=4096):
        """Largest power-of-two block (multiple of self.hop) whose RAM layout fits."""
        best, b = None, self.hop
        while b <= limit:
            try:
                self.layout(b)
                best = b
            except MemoryError:
                break
            b *= 2
        return best

    def emit_positions(self, block):
        return [i for i in range(block) if (i - self.history) % self.hop == 0]

    def build(self, block):
        if block % self.hop:
            raise ValueError("block must be a multiple of hop")
        return Built(self.program(block), self.layout(block), block)


class _FFTKernel(_Framed):
    def _fft_alloc(self, L, block):
        n = self.n
        L.alloc("in", 2 * (self.history + block))
        L.alloc("window", n)
        L.alloc("rev", n)
        L.alloc("tw", n)  # n/2 complex twiddles
        L.alloc("buf", 2 * n)

    def _fft_words(self, built, context):
        w, L = {}, built.layout
        put_pairs(w, L["in"], context)
        for i, v in enumerate(self.window):
            w[L["window"] + 4 * i] = word(v)
        for i in range(self.n):
            w[L["rev"] + 4 * i] = 8 * bitrev(i, self.L)
        put_pairs(w, L["tw"], [twiddle(j, self.n) for j in range(self.n // 2)])
        return w

    def _emit_fft(self, a: Asm, L: Layout):
        """Subroutine `fft`: windows the frame at s8 into buf (bit-reversed), then log2(n) stages in place.
        Clobbers t0-t6, s0-s7, a1-a5; preserves s8-s11, a0, a6, a7."""
        n = self.n
        a("fft:")
        a.li("s7", 1 << (Q - 1))
        a.li("a1", L["window"])
        a.li("a2", L["rev"])
        a.li("a3", L["buf"])
        a("mv a4, s8")
        close = a.loop("a5", n)
        a("lw t0, 0(a4)", "lw t1, 4(a4)", "lw t2, 0(a1)", "mul t0, t0, t2", "mul t1, t1, t2",
          "add t0, t0, s7", "srai t0, t0, 15", "add t1, t1, s7", "srai t1, t1, 15",
          "lw t3, 0(a2)", "add t3, t3, a3", "sw t0, 0(t3)", "sw t1, 4(t3)",
          "addi a4, a4, 8", "addi a1, a1, 4", "addi a2, a2, 4")
        close()
        self._emit_stages(a, L, block_fp=False)
        a("ret")

    def _emit_fft_block(self, a: Asm, L: Layout):
        """Subroutine `fft` with block-floating-point scaling (see ref_frame_block); stores the frame
        exponent to L["fexp"]. Registers: s6 = OR of onesabs, a2 = stage shift, a4 = exponent.
        Clobbers t0-t6, s0-s7, a1-a5; preserves s8-s11, a0, a6, a7."""
        n = self.n
        a("fft:")
        a.li("a1", L["window"])
        a.li("a2", L["rev"])
        a.li("a3", L["buf"])
        a("mv a4, s8", "li s6, 0")
        close = a.loop("a5", n)                                   # p = x*w, exact, bit-reversed store
        a("lw t0, 0(a4)", "lw t1, 4(a4)", "lw t2, 0(a1)", "mul t0, t0, t2", "mul t1, t1, t2",
          "srai t6, t0, 31", "xor t6, t0, t6", "or s6, s6, t6", "srai t6, t1, 31", "xor t6, t1, t6", "or s6, s6, t6",
          "lw t3, 0(a2)", "add t3, t3, a3", "sw t0, 0(t3)", "sw t1, 4(t3)",
          "addi a4, a4, 8", "addi a1, a1, 4", "addi a2, a2, 4")
        close()
        # r = bitlen(OR >> 14) (OR < 2^31, so at most 17 bits)
        a("srli t0, s6, 14", "li t1, 0")
        close = a.loop("a5", 17)
        a("snez t2, t0", "add t1, t1, t2", "srli t0, t0, 1")
        close()
        a(f"li a4, {15 + self.L}", "sub a4, a4, t1", "li t2, 1", "sll t2, t2, t1", "srli t2, t2, 1", "li s6, 0")
        a.li("a1", L["buf"])
        close = a.loop("a5", 2 * n)                               # xw = round_half_up(p, r)
        a("lw t0, 0(a1)", "add t0, t0, t2", "sra t0, t0, t1", "sw t0, 0(a1)",
          "srai t6, t0, 31", "xor t6, t0, t6", "or s6, s6, t6", "addi a1, a1, 4")
        close()
        a.li("s7", 1 << (Q - 1))
        self._emit_stages(a, L, block_fp=True)
        a.li("t0", L["fexp"])
        a("sw a4, 0(t0)", "ret")

    def _emit_stages(self, a: Asm, L: Layout, block_fp: bool):
        """The radix-2 stages in place on buf (a3), bit-identical to ref_fft / ref_frame_block. Stage 0
        uses W_0 = (32767, 0): rhu(32767 b, 15) = b + ((2^14 - b) >> 15); stage 1 also W_{n/4} = (0, -32768):
        t = (bi, -br). Later stages load each twiddle once (twiddle-outer loops; the last stage flat).
        Block mode: a2 = this stage's shift from the OR in s6 (then s6 restarts), a4 -= shift.
        Uses t0-t6, s0-s5, a5, a1 (twiddle pointer)."""
        n, Lg = self.n, self.L

        def finish(last, ar, ai, tr, ti, oa, ob):
            for op, x, t_, off in (("add", ar, tr, oa), ("sub", ar, tr, ob), ("add", ai, ti, oa + 4), ("sub", ai, ti, ob + 4)):
                a(f"{op} t5, {x}, {t_}")
                a("add t5, t5, a2", "sra t5, t5, a2") if block_fp else a("addi t5, t5, 1", "srai t5, t5, 1")
                a(f"sw t5, {off}(t2)")
                if block_fp and not last:
                    a("srai t6, t5, 31", "xor t6, t5, t6", "or s6, s6, t6")

        def w0(dst, src):  # rhu(32767 src, 15)
            a(f"sub {dst}, s1, {src}", f"srai {dst}, {dst}, 15", f"add {dst}, {dst}, {src}")

        def product():  # t3 + i t4 = rhu(W b, 15) per component; W in s0, s1; b at 8h(t2)
            a("mul t3, s0, s2", "mul t5, s1, s3", "sub t3, t3, t5",
              "mul t4, s0, s3", "mul t5, s1, s2", "add t4, t4, t5",
              "add t3, t3, s7", "srai t3, t3, 15", "add t4, t4, s7", "srai t4, t4, 15")

        for st in range(Lg):
            h, last = 1 << st, st == Lg - 1
            if block_fp:
                a("srli a2, s6, 14", "snez a2, a2", "sub a4, a4, a2", "li s6, 0")
            a("mv t2, a3")
            if st == 0:
                a.li("s1", 1 << 14)
                close = a.loop("a5", n // 2)
                a("lw s4, 0(t2)", "lw s5, 4(t2)", "lw s2, 8(t2)", "lw s3, 12(t2)")
                w0("t3", "s2")
                w0("t4", "s3")
                finish(last, "s4", "s5", "t3", "t4", 0, 8)
                a("addi t2, t2, 16")
                close()
            elif st == 1:
                close = a.loop("a5", n // 4)
                a("lw s4, 0(t2)", "lw s5, 4(t2)", "lw s2, 16(t2)", "lw s3, 20(t2)")
                w0("t3", "s2")
                w0("t4", "s3")
                finish(last, "s4", "s5", "t3", "t4", 0, 16)
                a("lw s4, 8(t2)", "lw s5, 12(t2)", "lw s2, 24(t2)", "lw s3, 28(t2)", "neg t4, s2")
                finish(last, "s4", "s5", "s3", "t4", 8, 24)
                a("addi t2, t2, 32")
                close()
            else:
                groups, tstep = n // (2 * h), 8 * (n // (2 * h))
                a.li("a1", L["tw"])
                close = a.loop("a5", h)
                a("lw s0, 0(a1)", "lw s1, 4(a1)")
                inner = a.loop("t0", groups) if groups > 1 else None
                a(f"lw s2, {8*h}(t2)", f"lw s3, {8*h + 4}(t2)")
                product()
                a("lw s4, 0(t2)", "lw s5, 4(t2)")
                finish(last, "s4", "s5", "t3", "t4", 0, 8 * h)
                if inner:
                    a(f"addi t2, t2, {16 * h}")
                    inner()
                    a(f"addi t2, t2, {8 - 8 * n}")
                else:
                    a("addi t2, t2, 8")
                a(f"addi a1, a1, {tstep}")
                close()

    @staticmethod
    def _power(a: Asm, re="t0", im="t1"):
        """t2:t3 = re^2 + im^2 (exact 64-bit); clobbers t4-t6."""
        a(f"mul t2, {re}, {re}", f"mulh t3, {re}, {re}", f"mul t4, {im}, {im}", f"mulh t5, {im}, {im}",
          "add t2, t2, t4", "sltu t6, t2, t4", "add t3, t3, t5", "add t3, t3, t6")

    def _frame_loop(self, a: Asm, L: Layout, frames: int, first_ctx: int, body, step_frames: int = 1):
        """for each of `frames` frames (s10 counter): call fft at s8, run body(), advance s8 by hop."""
        a.li("s8", L["in"] + 8 * first_ctx)
        close = a.loop("s10", frames)
        a("call fft")
        body()
        a.li("t0", 8 * self.hop * step_frames)
        a("add s8, s8, t0")
        close()


class FFTPower(_FFTKernel):
    """Per frame: n bin powers |X_k|^2 (exact 64-bit) or, output='complex', the int32 bins X_k = DFT(xw)_k / n.
    scaling='block': block-floating-point FFT (ref_frame_block); each frame's output is (exponent, bins)."""

    def __init__(self, n=16, window=None, hop=None, output="power", scaling="fixed"):
        hop = n if hop is None else hop
        self._check(n, hop)
        if output not in ("power", "complex"):
            raise ValueError("output 'power' or 'complex'")
        if scaling not in ("fixed", "block"):
            raise ValueError("scaling 'fixed' or 'block'")
        self.hop, self.output, self.history, self.scaling = hop, output, n - 1, scaling
        self.window = self._window(window)
        self.name = f"fft{n}_{output}" + ("_bfp" if scaling == "block" else "")

    def describe(self):
        return {"kernel": self.name, "n": self.n, "hop": self.hop, "output": self.output, "scaling": self.scaling,
                "window": list(self.window), "max_block": self.max_block()}

    def layout(self, block):
        L = Layout()
        self._fft_alloc(L, block)
        L.alloc("out", 2 * self.n * (block // self.hop))
        if self.scaling == "block":
            L.alloc("fexp", 1)
            L.alloc("exp", block // self.hop)
        return L

    def program(self, block):
        L, a, n = self.layout(block), Asm(), self.n
        frames = self.emit_positions(block)
        block_fp = self.scaling == "block"
        a.li("s9", L["out"])
        if block_fp:
            a.li("s11", L["exp"])

        def body():
            if block_fp:
                a.li("t0", L["fexp"])
                a("lw t0, 0(t0)", "sw t0, 0(s11)", "addi s11, s11, 4")
            a.li("a1", L["buf"])
            close = a.loop("a5", n)
            a("lw t0, 0(a1)", "lw t1, 4(a1)")
            if self.output == "power":
                self._power(a)
                a("sw t2, 0(s9)", "sw t3, 4(s9)")
            else:
                a("sw t0, 0(s9)", "sw t1, 4(s9)")
            a("addi a1, a1, 8", "addi s9, s9, 8")
            close()
        self._frame_loop(a, L, len(frames), frames[0] - (n - 1) + self.history, body)
        a.readback(L["out"], 2 * n * len(frames))
        if block_fp:
            a.readback(L["exp"], len(frames))
        a.halt()
        (self._emit_fft_block if block_fp else self._emit_fft)(a, L)
        return a.source()

    def init_words(self, built, context, start):
        return self._fft_words(built, context)

    def decode(self, built, ram, start):
        o, n, out = built.layout["out"], self.n, [None] * built.block
        for f, i in enumerate(self.emit_positions(built.block)):
            if start + i < n - 1:
                continue
            base = o + 8 * n * f
            if self.output == "power":
                out[i] = tuple(u64(ram, base + 8 * k) for k in range(n))
            else:
                out[i] = tuple((signed(ram[(base >> 2) + 2 * k]), signed(ram[(base >> 2) + 2 * k + 1]))
                               for k in range(n))
            if self.scaling == "block":
                out[i] = (signed(ram[(built.layout["exp"] >> 2) + f]), out[i])
        return out

    def reference(self, samples):
        out, n = [None] * len(samples), self.n
        for e in range(n - 1, len(samples), self.hop):
            if self.scaling == "block":
                exponent, bins = ref_frame_block(samples, e, n, self.window)
            else:
                bins = ref_frame(samples, e, n, self.window)
            out[e] = tuple(ref_powers(bins)) if self.output == "power" else tuple(bins)
            if self.scaling == "block":
                out[e] = (exponent, out[e])
        return out


class WelchPSD(_FFTKernel):
    """Average of `frames` consecutive FFT frame powers: exact 64-bit sum >> log2(frames), emitted at the end
    of the last frame, once every frames*hop samples (first at e = n-1 + (frames-1)*hop)."""

    def __init__(self, n=16, window=None, hop=None, frames=4):
        hop = n if hop is None else hop
        self._check(n, hop)
        if frames < 1 or frames & (frames - 1) or frames > 1024:
            raise ValueError("frames: power of two <= 1024")
        self.frame_hop, self.frames, self.shift = hop, frames, frames.bit_length() - 1
        self.hop = frames * hop
        self.history = n - 1 + (frames - 1) * hop
        self.window = self._window(window)
        self.name = f"welch{n}x{frames}"

    def describe(self):
        return {"kernel": self.name, "n": self.n, "hop": self.frame_hop, "frames": self.frames,
                "window": list(self.window), "max_block": self.max_block()}

    def layout(self, block):
        L = Layout()
        self._fft_alloc(L, block)
        L.alloc("acc", 2 * self.n)
        L.alloc("out", 2 * self.n * (block // self.hop))
        return L

    def program(self, block):
        L, a, n, sh = self.layout(block), Asm(), self.n, self.shift
        outs = self.emit_positions(block)
        a.li("s9", L["out"])
        a.li("s8", L["in"] + 8 * outs[0])  # earliest frame of the first output starts at context index outs[0]
        close_out = a.loop("s11", len(outs))
        a.li("a1", L["acc"])
        close = a.loop("a5", 2 * n)
        a("sw zero, 0(a1)", "addi a1, a1, 4")
        close()
        close_f = a.loop("s10", self.frames)
        a("call fft")
        a.li("a1", L["buf"])
        a.li("a2", L["acc"])
        close = a.loop("a5", n)
        a("lw t0, 0(a1)", "lw t1, 4(a1)")
        self._power(a)
        a("lw t0, 0(a2)", "lw t1, 4(a2)", "add t0, t0, t2", "sltu t6, t0, t2", "add t1, t1, t3",
          "add t1, t1, t6", "sw t0, 0(a2)", "sw t1, 4(a2)", "addi a1, a1, 8", "addi a2, a2, 8")
        close()
        a.li("t0", 8 * self.frame_hop)
        a("add s8, s8, t0")
        close_f()
        a.li("a2", L["acc"])
        close = a.loop("a5", n)
        a("lw t0, 0(a2)", "lw t1, 4(a2)")
        if sh:
            a(f"srli t0, t0, {sh}", f"slli t2, t1, {32 - sh}", "or t0, t0, t2", f"srai t1, t1, {sh}")
        a("sw t0, 0(s9)", "sw t1, 4(s9)", "addi a2, a2, 8", "addi s9, s9, 8")
        close()
        close_out()
        a.readback(L["out"], 2 * n * len(outs))
        a.halt()
        self._emit_fft(a, L)
        return a.source()

    def init_words(self, built, context, start):
        return self._fft_words(built, context)

    def decode(self, built, ram, start):
        o, n, out = built.layout["out"], self.n, [None] * built.block
        for f, i in enumerate(self.emit_positions(built.block)):
            if start + i >= self.history:
                out[i] = tuple(u64(ram, o + 8 * n * f + 8 * k) for k in range(n))
        return out

    def reference(self, samples):
        out, n = [None] * len(samples), self.n
        for e in range(self.history, len(samples), self.hop):
            total = [0] * n
            for j in range(self.frames):
                for k, p in enumerate(ref_powers(ref_frame(samples, e - j * self.frame_hop, n, self.window))):
                    total[k] += p
            out[e] = tuple(t >> self.shift for t in total)
        return out


class CFARDetect(_FFTKernel):
    """Cell-averaging CFAR across the bins of each FFT frame, circular indexing.

    noise_k = sum_{j=guard+1}^{guard+train} (P[(k+j) mod n] + P[(k-j) mod n]);
    bit k = P_k * (2*train*256) > scale_q8 * noise_k, exact unsigned 64-bit:
    P < 2^32, so P*512*train < 2^41*train and scale_q8*noise < 2^16 * 2*train*2^32 = 2^49*train <= 2^56
    (train <= 128); both sides fit 64 bits. Output per frame: (mask, peak) where mask has bit k set for
    detections (ceil(n/32) words in RAM) and peak = argmax P (lowest index on ties), all branch-free."""

    def __init__(self, n=16, window=None, hop=None, train=2, guard=1, scale_q8=1024):
        hop = n if hop is None else hop
        self._check(n, hop)
        if train < 1 or guard < 0 or 2 * (train + guard) + 1 > n:
            raise ValueError("need train >= 1, guard >= 0, 2*(train+guard)+1 <= n")
        if not 0 <= scale_q8 < 1 << 16:
            raise ValueError("scale_q8 in [0, 65536)")
        self.hop, self.history = hop, n - 1
        self.train, self.guard, self.scale_q8 = train, guard, scale_q8
        self.window = self._window(window)
        self.mask_words = (n + 31) // 32
        self.name = f"cfar{n}_t{train}g{guard}"

    def describe(self):
        return {"kernel": self.name, "n": self.n, "hop": self.hop, "train": self.train, "guard": self.guard,
                "scale_q8": self.scale_q8, "window": list(self.window), "max_block": self.max_block(),
                "compare_bits": 64}

    def layout(self, block):
        L = Layout()
        self._fft_alloc(L, block)
        L.alloc("pow", 2 * self.n)
        L.alloc("param", 1)
        L.alloc("out", (self.mask_words + 1) * (block // self.hop))
        return L

    def program(self, block):
        L, a, n = self.layout(block), Asm(), self.n
        T, g, m = self.train, self.guard, n - 1
        frames = self.emit_positions(block)
        a.li("s9", L["out"])

        def load(off):  # t1:t2 = P[(k + off) mod n]
            a(f"addi t0, s11, {off}", f"andi t0, t0, {m}", "slli t0, t0, 3", "add t0, t0, s0",
              "lw t1, 0(t0)", "lw t2, 4(t0)")

        def add64(lo, hi):
            a(f"add {lo}, {lo}, t1", f"sltu t3, {lo}, t1", f"add {hi}, {hi}, t2", f"add {hi}, {hi}, t3")

        def sub64(lo, hi):
            a(f"sltu t3, {lo}, t1", f"sub {lo}, {lo}, t1", f"sub {hi}, {hi}, t2", f"sub {hi}, {hi}, t3")

        def gt64(dst, ah, al, bh, bl):  # dst = (ah:al > bh:bl) unsigned; clobbers t6, t0
            a(f"sltu {dst}, {bh}, {ah}", f"xor t6, {ah}, {bh}", "seqz t6, t6", f"sltu t0, {bl}, {al}",
              "and t6, t6, t0", f"or {dst}, {dst}, t6")

        def body():
            a.li("s0", L["pow"])
            a.li("a1", L["buf"])
            a("mv a2, s0")
            close = a.loop("a5", n)
            a("lw t0, 0(a1)", "lw t1, 4(a1)")
            self._power(a)
            a("sw t2, 0(a2)", "sw t3, 4(a2)", "addi a1, a1, 8", "addi a2, a2, 8")
            close()
            for j in range(self.mask_words):
                a(f"sw zero, {4 * j}(s9)")
            # windows for k = 0: R = sum P[g+1..g+T], L = sum P[-g-1..-g-T]
            a("li a1, 0", "li a2, 0", "li a3, 0", "li a4, 0")
            for j in range(g + 1, g + T + 1):
                a(f"lw t1, {8 * (j % n)}(s0)", f"lw t2, {8 * (j % n) + 4}(s0)")
                add64("a3", "a4")
                a(f"lw t1, {8 * (-j % n)}(s0)", f"lw t2, {8 * (-j % n) + 4}(s0)")
                add64("a1", "a2")
            a.li("s6", 512 * T)
            a.li("t0", L["param"])
            a("lw s7, 0(t0)", "li s3, 0", "li s4, 0", "li s5, 0", "li s11, 0")
            close = a.loop("a5", n)
            a("slli t0, s11, 3", "add t0, t0, s0", "lw t1, 0(t0)", "lw t2, 4(t0)",
              "mul t3, t1, s6", "mulhu t4, t1, s6", "mul t5, t2, s6", "add t4, t4, t5",       # LHS t4:t3
              "add t5, a1, a3", "sltu t6, t5, a3", "add t6, t6, a2", "add t6, t6, a4",       # noise t6:t5
              "mul s1, t5, s7", "mulhu s2, t5, s7", "mul t6, t6, s7", "add s2, s2, t6")      # RHS s2:s1
            gt64("t5", "t4", "t3", "s2", "s1")
            a("andi t6, s11, 31", "sll t5, t5, t6", "srli t6, s11, 5", "slli t6, t6, 2", "add t6, t6, s9",
              "lw t0, 0(t6)", "or t0, t0, t5", "sw t0, 0(t6)")
            gt64("t5", "t2", "t1", "s4", "s3")
            a("neg t5, t5",
              "xor t6, s3, t1", "and t6, t6, t5", "xor s3, s3, t6",
              "xor t6, s4, t2", "and t6, t6, t5", "xor s4, s4, t6",
              "xor t6, s5, s11", "and t6, t6, t5", "xor s5, s5, t6")
            load(g + T + 1)
            add64("a3", "a4")
            load(g + 1)
            sub64("a3", "a4")
            load(-g)
            add64("a1", "a2")
            load(-g - T)
            sub64("a1", "a2")
            a("addi s11, s11, 1")
            close()
            a(f"sw s5, {4 * self.mask_words}(s9)", f"addi s9, s9, {4 * (self.mask_words + 1)}")
        self._frame_loop(a, L, len(frames), frames[0], body)
        a.readback(L["out"], (self.mask_words + 1) * len(frames))
        a.halt()
        self._emit_fft(a, L)
        return a.source()

    def init_words(self, built, context, start):
        w = self._fft_words(built, context)
        w[built.layout["param"]] = self.scale_q8
        return w

    def decode(self, built, ram, start):
        o, W, out = built.layout["out"] >> 2, self.mask_words, [None] * built.block
        for f, i in enumerate(self.emit_positions(built.block)):
            if start + i >= self.n - 1:
                base = o + (W + 1) * f
                mask = sum(ram[base + j] << (32 * j) for j in range(W))
                out[i] = (mask, signed(ram[base + W]))
        return out

    def reference(self, samples):
        out, n, T, g = [None] * len(samples), self.n, self.train, self.guard
        for e in range(n - 1, len(samples), self.hop):
            P = ref_powers(ref_frame(samples, e, n, self.window))
            mask = 0
            for k in range(n):
                noise = sum(P[(k + j) % n] + P[(k - j) % n] for j in range(g + 1, g + T + 1))
                if P[k] * 2 * T * 256 > self.scale_q8 * noise:
                    mask |= 1 << k
            out[e] = (mask, P.index(max(P)))
        return out


class DFTBin(_Framed):
    """Single-bin detector: X = round_half_up(sum_t x_t * C_t, 15) per component with C_t = W_n^(k t) Q15
    (runtime RAM table), exact 64-bit accumulation (|sum| < 2^40), then |X|^2 exact 64-bit (< 2^49).
    Frames as FFTPower (rectangular window); one output (two words) per frame."""

    def __init__(self, n=16, k=1, hop=None):
        hop = n if hop is None else hop
        self._check(n, hop)
        if not 0 <= k < n:
            raise ValueError("0 <= k < n")
        self.k, self.hop, self.history = k, hop, n - 1
        self.coef = [twiddle((k * t) % n, n) for t in range(n)]
        self.name = f"dft{n}_bin{k}"

    def describe(self):
        return {"kernel": self.name, "n": self.n, "bin": self.k, "hop": self.hop, "max_block": self.max_block(),
                "accumulator_bits": 64}

    def layout(self, block):
        L = Layout()
        L.alloc("in", 2 * (self.history + block))
        L.alloc("coef", 2 * self.n)
        L.alloc("out", 2 * (block // self.hop))
        return L

    def program(self, block):
        L, a, n = self.layout(block), Asm(), self.n
        frames = self.emit_positions(block)
        a.li("s9", L["out"])
        a.li("s8", L["in"] + 8 * frames[0])
        close_f = a.loop("s10", len(frames))
        a("mv a1, s8", "li s0, 0", "li a3, 0", "li s1, 0", "li a4, 0")
        a.li("a2", L["coef"])
        close = a.loop("a5", n)
        a("lw t0, 0(a1)", "lw t1, 4(a1)", "lw t2, 0(a2)", "lw t3, 4(a2)")
        mac(a, "s0", "a3", "t0", "t2")
        mac(a, "s0", "a3", "t1", "t3", subtract=True)
        mac(a, "s1", "a4", "t0", "t3")
        mac(a, "s1", "a4", "t1", "t2")
        a("addi a1, a1, 8", "addi a2, a2, 8")
        close()
        narrow64(a, "s0", "a3", Q)
        narrow64(a, "s1", "a4", Q)
        a("mul t2, s0, s0", "mulh t3, s0, s0", "mul t4, s1, s1", "mulh t5, s1, s1",
          "add t2, t2, t4", "sltu t6, t2, t4", "add t3, t3, t5", "add t3, t3, t6",
          "sw t2, 0(s9)", "sw t3, 4(s9)", "addi s9, s9, 8")
        a.li("t0", 8 * self.hop)
        a("add s8, s8, t0")
        close_f()
        a.readback(L["out"], 2 * len(frames))
        a.halt()
        return a.source()

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        put_pairs(w, built.layout["coef"], self.coef)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"], [None] * built.block
        for f, i in enumerate(self.emit_positions(built.block)):
            if start + i >= self.n - 1:
                out[i] = u64(ram, o + 8 * f)
        return out

    def reference(self, samples):
        out, n = [None] * len(samples), self.n
        for e in range(n - 1, len(samples), self.hop):
            re = im = 0
            for t, (xr, xi) in enumerate(samples[e - n + 1:e + 1]):
                ang = 2 * math.pi * ((self.k * t) % n) / n
                cr, ci = q15(math.cos(ang)), q15(-math.sin(ang))
                re += xr * cr - xi * ci
                im += xr * ci + xi * cr
            re, im = round_half_up(re, Q), round_half_up(im, Q)
            out[e] = re * re + im * im
        return out
