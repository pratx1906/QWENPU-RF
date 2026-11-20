"""FFT64 power spectrum with block floating point on the new VX instruction VBFLY (QwenPU branch
rfpu-vfir; needs that chipformer first on the path). PROTOCOL Amendment 6.

Per frame of 64 samples (hop 64):
  1. CPU: p = x w (Hann, Q15, exact), r = bitlen(max |p| >> 13), xw = round_half_up(p, r), so every
     component is at most 2^13; xw goes into VM by VPUT, x[i] to v0 pair i and x[i + 32] to v8 pair i.
  2. Six constant-geometry stages of VBFLY (twiddles TW[r] = W64^brev5(r) in the 32 RoPE pairs, loaded
     once; mask 2^s - 1). Stage s scales (sc = 1) only when the largest |component| of its input has
     bit length >= 14 (the CPU's for stage 0, else VGET 3 after the previous, tracked stage), which keeps
     every operand inside int16 (complex magnitude <= ~23,200). Stages 0..4 are tracked (4 butterflies
     an op); the last is not (8 an op) and writes bank A registers v2 / v3.
  3. Bin k's power |X_k|^2 by VPOW (n = 1) on output position brev6(k); exponent e = 21 - r - sum(sc),
     so the power approximates |DFT(x w / 2^15) / 64|^2 * 4^e.
Output per frame (at its last sample): (e, 64 powers in natural bin order).

scaling="fixed" (ablation, PROTOCOL Amendment 8 P8.6): r = 16 for every frame (xw = round_half_up(x w, 16),
which keeps every butterfly operand inside int16), every stage sc = 1, untracked (8 butterflies an op),
so e = -1 always: the int16 analogue of the CPU's fixed-scaling FFT.
"""
import math

from .codegen import Asm, Layout, round_half_up, signed, word
from .kernels import Built, Kernel, put_pairs

N, H, L = 64, 32, 6


def _brev(i, b):
    return int(format(i, f"0{b}b")[::-1], 2) if b else 0


def _q15(v):
    return max(-32768, min(32767, math.floor(32768 * v + 0.5)))


TW = [(_q15(math.cos(2 * math.pi * _brev(r, 5) / N)), _q15(-math.sin(2 * math.pi * _brev(r, 5) / N)))
      for r in range(32)]
HANN = [_q15(0.5 - 0.5 * math.cos(2 * math.pi * i / N)) for i in range(N)]


def _bfly_word(a, h, dst, woff, rhi, mask, n, keep, track):
    return a | h << 3 | dst << 6 | woff << 10 | rhi << 15 | mask << 16 | n << 22 | keep << 26 | track << 27


def ref_frame(samples, fixed=False):
    """Independent integer model of one frame (not vx.py): returns (exponent, powers)."""
    p = [(re * w, im * w) for (re, im), w in zip(samples, HANN)]
    r = 16 if fixed else (max(max(abs(a), abs(b)) for a, b in p) >> 13).bit_length()
    v = [(round_half_up(a, r), round_half_up(b, r)) for a, b in p]
    e = 15 + L - r
    bl = max(max(abs(a), abs(b)) for a, b in v).bit_length()
    for s in range(L):
        sc = 1 if fixed else int(bl >= 14)
        e -= sc
        y = [None] * N
        for i in range(H):
            (ar, ai), (br, bi) = v[i], v[i + H]
            assert all(-32768 <= x < 32768 for x in (ar, ai, br, bi))
            c, sn = TW[i & ((1 << s) - 1)]
            tr, ti = c * br - sn * bi, c * bi + sn * br
            rnd = 1 << (14 + sc)
            y[2 * i] = ((ar * 32768 + tr + rnd) >> (15 + sc), (ai * 32768 + ti + rnd) >> (15 + sc))
            y[2 * i + 1] = ((ar * 32768 - tr + rnd) >> (15 + sc), (ai * 32768 - ti + rnd) >> (15 + sc))
        v = y
        bl = max(max(abs(a), abs(b)) for a, b in v).bit_length()
    return e, tuple(v[_brev(k, L)][0] ** 2 + v[_brev(k, L)][1] ** 2 for k in range(N))


class VXFFT64(Kernel):
    name = "vx_fft64_bfp"
    hop = N

    def __init__(self, scaling="block"):
        if scaling not in ("block", "fixed"):
            raise ValueError("scaling 'block' or 'fixed'")
        self.fixed = scaling == "fixed"
        if self.fixed:
            self.name = "vx_fft64_fixed"

    def describe(self):
        return {"kernel": self.name, "n": N, "hop": N, "window": "hann", "scaling": "fixed" if self.fixed else "block",
                "instruction": "VBFLY (custom-2 funct7=1 funct3=2) + VPOW", "vx": True}

    def build(self, block):
        if block % N:
            raise ValueError("block must be a multiple of 64")
        frames = block // N
        L_ = Layout()
        L_.alloc("in", 2 * block)
        L_.alloc("window", N)
        L_.alloc("buf", 2 * N)
        L_.alloc("out", N * frames)
        L_.alloc("exp", frames)
        a = Asm()
        for r_, (c, s) in enumerate(TW):                       # the twiddle table, once
            a.li("t0", 512 + r_)
            a.li("t1", c)
            a.li("t2", s)
            a("vput t0, t1, t2")
        a.li("s8", L_["in"])
        a.li("s9", L_["out"])
        a.li("s11", L_["exp"])
        close_f = a.loop("s10", frames)
        # 1. p = x w (exact), OR of |p|
        a.li("a1", L_["window"])
        a.li("a4", L_["buf"])
        a("mv t4, s8", "li s6, 0")
        close = a.loop("a5", N)
        a("lw t0, 0(t4)", "lw t1, 4(t4)", "lw t2, 0(a1)", "mul t0, t0, t2", "mul t1, t1, t2",
          "sw t0, 0(a4)", "sw t1, 4(a4)",
          "srai t6, t0, 31", "xor t5, t0, t6", "sub t5, t5, t6", "or s6, s6, t5",
          "srai t6, t1, 31", "xor t5, t1, t6", "sub t5, t5, t6", "or s6, s6, t5",
          "addi t4, t4, 8", "addi a1, a1, 4", "addi a4, a4, 8")
        close()
        if self.fixed:
            a("li a2, 16")
        else:
            a("srli t0, s6, 13", "li a2, 0")                   # r = bitlen(OR >> 13)
            close = a.loop("a5", 18)
            a("snez t1, t0", "add a2, a2, t1", "srli t0, t0, 1")
            close()
        a("li a3, 1", "sll a3, a3, a2", "srli a3, a3, 1", "li s7, 0")   # rounding constant 2^(r-1)
        # xw = round_half_up(p, r) into VM: x[i] -> v0 pair i, x[i + 32] -> v8 pair i
        a.li("a4", L_["buf"])
        for base in (0, 256):
            a.li("t3", base)
            close = a.loop("a5", H)
            a("lw t1, 0(a4)", "lw t2, 4(a4)", "add t1, t1, a3", "sra t1, t1, a2", "add t2, t2, a3", "sra t2, t2, a2",
              "vput t3, t1, t2",
              "srai t6, t1, 31", "xor t5, t1, t6", "sub t5, t5, t6", "or s7, s7, t5",
              "srai t6, t2, 31", "xor t5, t2, t6", "sub t5, t5, t6", "or s7, s7, t5",
              "addi t3, t3, 1", "addi a4, a4, 8")
            close()
        if self.fixed:
            a("li s4, 15")                                     # every stage scales
        else:
            a("li s4, 0")                                      # s4 = bitlen(max |xw|)
            close = a.loop("a5", 15)
            a("snez t1, s7", "add s4, s4, t1", "srli s7, s7, 1")
            close()
        a(f"li a1, {15 + L}", "sub a1, a1, a2")               # a1 = exponent
        # 2. the stages
        for s in range(L):
            last = s == L - 1
            a("sltiu t0, s4, 14", "xori t0, t0, 1", "sub a1, a1, t0", "slli s5, t0, 21")
            src = s % 2
            track = not (last or self.fixed)
            n_b = 4 if track else 8
            for op in range(H // n_b):
                i0 = n_b * op
                if last:
                    dst = 2 if 2 * i0 < H else 3
                else:
                    dst = (1 - src) if 2 * i0 < H else 8 + (1 - src)
                w_ = _bfly_word(src, src, dst, (2 * i0) % 32, i0 >> 4, (1 << s) - 1, n_b, int(op > 0), int(track))
                a.li("t3", w_)
                a("or t3, t3, s5", "vbfly zero, t3, zero")
            if track:
                a("li t1, 3", "vget s4, t1, zero")
        # 3. powers in natural bin order, exponent
        for k in range(N):
            pos = _brev(k, L)
            reg, pair = (2, pos) if pos < H else (3, pos - H)
            a.li("t3", reg | 1 << 3 | (2 * pair) << 9)
            a("vpow t4, t3, zero", f"sw t4, {4 * k}(s9)")
        a(f"addi s9, s9, {4 * N}", "sw a1, 0(s11)", "addi s11, s11, 4", f"addi s8, s8, {8 * N}")
        close_f()
        a.readback(L_["out"], N * frames)
        a.readback(L_["exp"], frames)
        a.halt()
        return Built(a.source(), L_, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for i, v in enumerate(HANN):
            w[built.layout["window"] + 4 * i] = word(v)
        return w

    def decode(self, built, ram, start):
        out = [None] * built.block
        o, ex = built.layout["out"] >> 2, built.layout["exp"] >> 2
        for f in range(built.block // N):
            out[N * f + N - 1] = (signed(ram[ex + f]), tuple(ram[o + N * f + k] for k in range(N)))
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        for f in range(len(samples) // N):
            out[N * f + N - 1] = ref_frame(samples[N * f:N * f + N], self.fixed)
        return out
