"""Blind modulation classifier on QwenPU's CPU in weights (PROTOCOL Amendment 15).

Stage A, ModFeatures: one capture of N = 1024 Q15 complex samples -> NF = 30 integer features.
  a, b = I >> 4, Q >> 4                                   (|.| <= 2048)
  pass A over all samples n (p = a^2 + b^2):
    Sp  += p >> 4            Sp2 += (p >> 10)^2 (u64)     Sa += a, Sb += b
    Su  += (a^2 - b^2) >> 4  Sv  += (2ab) >> 4            E[n & 7] += p >> 4   (timing-phase energy)
    n >= 1: phi = CORDIC (10 iterations) angle of x[n] conj(x[n-1]) >> 16 (int16, 2^15 = pi)
            Sphi += phi, Sphi2 += phi^2 (u64); n >= 2: Sdd += |phi - phi_prev|
  pass B on the symbol-rate samples x[k* + 8m], m = 0..127, k* = first argmax E (SPS = 8 assumed):
    (r, ang) = CORDIC(a << 6, b << 6), th = ang >> 16 as u16 (2^16 = 2 pi); Sr += r
    second loop: mr = (Sr >> 7) + 1, bin = min((8 r) // mr, 15), H[bin] += 1
                 for M in 1,2,4,8,16,32: C_M, S_M += COS64/SIN64[((M th) mod 2^16) >> 10] (Q12 table)
                 w = r >> 10, Sw += w, WC_M, WS_M += w * table, M = 2, 4
  features (L = log2 in Q8, classify.log2_q8; L64 for u64):
    f0 = L64(Sp2) - 2 L(Sp)                          amplitude kurtosis
    f1 = L((Sa>>6)^2 + (Sb>>6)^2) - L(Sp)            residual carrier
    f2 = L((Su>>14)^2 + (Sv>>14)^2) - 2 L(Sp>>14)    |E x^2| / E|x|^2 (real-valued signals)
    f3 = Sphi >> 10, f4 = L64(Sphi2), f5 = L(Sdd)    instantaneous frequency: mean, power, roughness
    f6..f11  = L((C_M>>4)^2 + (S_M>>4)^2)            M-th power phase concentration
    f12, f13 = L((WC_M>>13)^2 + (WS_M>>13)^2) - 2 L(Sw)   amplitude-weighted, M = 2, 4
    f14..f29 = H[0..15]                              symbol amplitude histogram (ring structure)
Stage B, IntMLP: features (host copies the verified words) -> h = clamp(relu(rhu(W1 f, s1) + b1), 32767),
  s = rhu(W2 h, 12) + b2, class = first argmax. Weights are runtime RAM data (int16 codes).
All arithmetic is RV32 on the CPU in weights; both stages have independent integer references.
"""
import math

from .classify import LOG2_T, log2_q8
from .codegen import Asm, Layout, round_half_up, signed, word
from .kernels import Built, Kernel, mac, narrow64, put_pairs
from .detect import min_u32
from .nonlinear import cordic_asm, cordic_vector

N, SPS, NS, NF = 1024, 8, 128, 30
IF_ITER = 10  # CORDIC iterations for the instantaneous-frequency angle (resolution ~0.1 degree)
MS = (1, 2, 4, 8, 16, 32)
COS64 = [round(4096 * math.cos(2 * math.pi * k / 64)) for k in range(64)]
SIN64 = [round(4096 * math.sin(2 * math.pi * k / 64)) for k in range(64)]
FEATURE_NAMES = ["kurtosis", "carrier", "c20", "if_mean", "if_power", "if_rough"] + \
    [f"phase_m{m}" for m in MS] + ["wphase_m2", "wphase_m4"] + [f"ring{k}" for k in range(16)]


def log2_q8_64(v: int) -> int:
    hi = v >> 32
    if not hi:
        return log2_q8(v)
    n = hi.bit_length()
    return log2_q8(v >> n) + (n << 8)


def _sq2(x, y, s):
    return (x >> s) ** 2 + (y >> s) ** 2


def features(x):
    """Independent reference of ModFeatures on one capture of N (I, Q) Q15 pairs."""
    ab = [(i >> 4, q >> 4) for i, q in x]
    Sp = Sp2 = Sa = Sb = Su = Sv = Sphi = Sphi2 = Sdd = 0
    E, prev = [0] * SPS, 0
    for n, (a, b) in enumerate(ab):
        p = a * a + b * b
        Sp += p >> 4
        Sp2 += (p >> 10) ** 2
        Sa, Sb = Sa + a, Sb + b
        Su += (a * a - b * b) >> 4
        Sv += (2 * a * b) >> 4
        E[n & 7] += p >> 4
        if n >= 1:
            c, d = ab[n - 1]
            phi = cordic_vector(a * c + b * d, b * c - a * d, IF_ITER)[1] >> 16
            Sphi, Sphi2 = Sphi + phi, Sphi2 + phi * phi
            if n >= 2:
                Sdd += abs(phi - prev)
            prev = phi
    k = max(range(SPS), key=lambda j: (E[j], -j))
    sym = [cordic_vector(ab[k + SPS * m][0] << 6, ab[k + SPS * m][1] << 6) for m in range(NS)]
    r = [s[0] for s in sym]
    th = [(word(s[1]) >> 16) for s in sym]
    mr = (sum(r) >> 7) + 1
    H = [0] * 16
    C, S = {m: 0 for m in MS}, {m: 0 for m in MS}
    WC, WS, Sw = {2: 0, 4: 0}, {2: 0, 4: 0}, 0
    for rm, t in zip(r, th):
        H[min((rm << 3) // mr, 15)] += 1
        w = rm >> 10
        Sw += w
        for m in MS:
            i = ((m * t) & 0xFFFF) >> 10
            C[m], S[m] = C[m] + COS64[i], S[m] + SIN64[i]
            if m in WC:
                WC[m], WS[m] = WC[m] + w * COS64[i], WS[m] + w * SIN64[i]
    L = log2_q8
    f = [log2_q8_64(Sp2) - 2 * L(Sp), L(_sq2(Sa, Sb, 6)) - L(Sp), L(_sq2(Su, Sv, 14)) - 2 * L(Sp >> 14),
         Sphi >> 10, log2_q8_64(Sphi2), L(Sdd)]
    f += [L(_sq2(C[m], S[m], 4)) for m in MS]
    f += [L(_sq2(WC[m], WS[m], 13)) - 2 * L(Sw) for m in (2, 4)]
    return f + H


LOG2_SUBROUTINES = ("log2q8", "l64")


def _log_subroutines(a: Asm, table: int):
    # log2q8(a0 u32) -> a0 (Q8); 0 -> 0. Clobbers t0-t5 (as classify.TechClassifier).
    a("log2q8:", "bnez a0, lq1", "ret", "lq1:", "li t0, 31",
      "lq2:", "srl t1, a0, t0", "bnez t1, lq3", "addi t0, t0, -1", "j lq2",
      "lq3:", "li t1, 31", "sub t1, t1, t0", "sll t2, a0, t1",
      "srli t3, t2, 27", "andi t3, t3, 15", "srli t4, t2, 11", "li t5, 65535", "and t4, t4, t5")
    a.li("t5", table)
    a("slli t3, t3, 2", "add t5, t5, t3", "lw t1, 0(t5)", "lw t5, 4(t5)", "sub t5, t5, t1",
      "mul t5, t5, t4", "srli t5, t5, 16", "add t1, t1, t5", "slli t0, t0, 16", "add t0, t0, t1",
      "srai a0, t0, 8", "ret")
    # l64(a0 lo, a1 hi < 2^31) -> a0 = L64; clobbers t0-t5, a1-a3.
    a("l64:", "bnez a1, l64a", "j log2q8",
      "l64a:", "li t0, 0", "mv t1, a1", "l64b:", "beqz t1, l64c", "srli t1, t1, 1", "addi t0, t0, 1", "j l64b",
      "l64c:", "srl a0, a0, t0", "li t2, 32", "sub t2, t2, t0", "sll t1, a1, t2", "or a0, a0, t1",
      "mv a3, ra", "mv a2, t0", "call log2q8", "slli a2, a2, 8", "add a0, a0, a2", "mv ra, a3", "ret")


class ModFeatures(Kernel):
    name = "mod_features"
    hop = N

    def describe(self):
        return {"kernel": self.name, "samples": N, "sps": SPS, "symbols": NS, "features": NF, "phase_orders": list(MS), "if_cordic_iterations": IF_ITER}

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(N - 1, len(samples), N):
            out[e] = tuple(features(samples[e - N + 1:e + 1]))
        return out

    def build(self, block, base=None):
        if block != N:
            raise ValueError("one capture per block (block = 1024)")
        L = Layout() if base is None else Layout(top=base)
        L.alloc("in", 2 * N)
        L.alloc("table", 17)
        L.alloc("trig", 128)            # COS64 then SIN64
        L.alloc("E", SPS)
        L.alloc("acc", 16)              # pass A sums
        L.alloc("r", NS)
        L.alloc("th", NS)
        L.alloc("hist", 16)
        L.alloc("ph", 2 * len(MS) + 5)  # C_M, S_M pairs; WC2, WS2, WC4, WS4, Sw
        L.alloc("out", NF)
        A, PH, TRIG = L["acc"], L["ph"], L["trig"]
        a = Asm()
        # ---- pass A
        a.li("s11", L["in"])
        for r in ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "a0", "a1", "a2", "a6"):
            a(f"li {r}, 0")
        close = a.loop("a7", N)
        skip, skipdd = a.fresh("noif"), a.fresh("nodd")
        a("lw t0, 0(s11)", "lw t1, 4(s11)", "srai a3, t0, 4", "srai a4, t1, 4",
          "mul t0, a3, a3", "mul t1, a4, a4", "add t2, t0, t1", "sub t3, t0, t1",
          "srai t3, t3, 4", "add s5, s5, t3",
          "mul t3, a3, a4", "slli t3, t3, 1", "srai t3, t3, 4", "add s6, s6, t3",
          "srai t3, t2, 4", "add s0, s0, t3",
          "andi t4, a6, 7", "slli t4, t4, 2")
        a.li("t5", L["E"])
        a("add t4, t4, t5", "lw t5, 0(t4)", "add t5, t5, t3", "sw t5, 0(t4)",
          "srai t3, t2, 10")
        mac(a, "s1", "s2", "t3", "t3")
        a("add s3, s3, a3", "add s4, s4, a4", f"beqz a6, {skip}",
          "mul t0, a3, a1", "mul t1, a4, a2", "add t0, t0, t1",
          "mul t1, a4, a1", "mul t2, a3, a2", "sub t1, t1, t2")
        cordic_asm(a, iterations=IF_ITER)
        a("srai t2, t2, 16", "add s7, s7, t2")
        mac(a, "s8", "s9", "t2", "t2")
        a("li t6, 2", f"blt a6, t6, {skipdd}",
          "sub t3, t2, a0", "srai t4, t3, 31", "xor t3, t3, t4", "sub t3, t3, t4", "add s10, s10, t3",
          f"{skipdd}:", "mv a0, t2",
          f"{skip}:", "mv a1, a3", "mv a2, a4", "addi s11, s11, 8", "addi a6, a6, 1")
        close()
        a.li("t6", A)
        for i, r in enumerate(("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10")):
            a(f"sw {r}, {4 * i}(t6)")
        # ---- timing phase: first argmax of E
        a.li("t6", L["E"])
        a("lw s0, 0(t6)", "li s1, 0")
        for k in range(1, SPS):
            nx = a.fresh("eam")
            a(f"lw t0, {4 * k}(t6)", f"bge s0, t0, {nx}", "mv s0, t0", f"li s1, {k}", f"{nx}:")
        a.li("s11", L["in"])
        a("slli t0, s1, 3", "add s11, s11, t0")
        a.li("s9", L["r"])
        a.li("s10", L["th"])
        a("li s8, 0")
        close = a.loop("a7", NS)
        a("lw t0, 0(s11)", "lw t1, 4(s11)", "srai t0, t0, 4", "srai t1, t1, 4", "slli t0, t0, 6", "slli t1, t1, 6")
        cordic_asm(a)
        a("srli t2, t2, 16", "sw t0, 0(s9)", "sw t2, 0(s10)", "add s8, s8, t0",
          f"addi s11, s11, {8 * SPS}", "addi s9, s9, 4", "addi s10, s10, 4")
        close()
        # ---- histogram and phase sums
        a("srli s8, s8, 7", "addi s8, s8, 1")
        a.li("s9", L["r"])
        a.li("s10", L["th"])
        a.li("s7", L["hist"])
        a.li("s6", TRIG)
        a.li("s5", PH)
        close = a.loop("a7", NS)
        a("lw t0, 0(s9)", "lw t1, 0(s10)", "slli t2, t0, 3", "divu t2, t2, s8", "li t3, 15")
        min_u32(a, "t2", "t2", "t3")
        a("slli t2, t2, 2", "add t2, t2, s7", "lw t3, 0(t2)", "addi t3, t3, 1", "sw t3, 0(t2)",
          "srli s4, t0, 10", "lw t3, 64(s5)", "add t3, t3, s4", "sw t3, 64(s5)")  # w, Sw
        for j, m in enumerate(MS):
            a(f"slli t3, t1, {int(math.log2(m))}", "srli t3, t3, 10", "andi t3, t3, 63", "slli t3, t3, 2",
              "add t3, t3, s6", "lw t4, 0(t3)", "lw t5, 256(t3)",
              f"lw t2, {8 * j}(s5)", "add t2, t2, t4", f"sw t2, {8 * j}(s5)",
              f"lw t2, {8 * j + 4}(s5)", "add t2, t2, t5", f"sw t2, {8 * j + 4}(s5)")
            if m in (2, 4):
                o = 4 * (2 * len(MS)) + (0 if m == 2 else 8)
                a("mul t4, t4, s4", "mul t5, t5, s4",
                  f"lw t2, {o}(s5)", "add t2, t2, t4", f"sw t2, {o}(s5)",
                  f"lw t2, {o + 4}(s5)", "add t2, t2, t5", f"sw t2, {o + 4}(s5)")
        a("addi s9, s9, 4", "addi s10, s10, 4")
        close()
        # ---- features
        a.li("s11", L["out"])
        a.li("s10", A)

        def sq2(rx, ry, s, base):  # a0 = (x >> s)^2 + (y >> s)^2 from RAM words at base offsets
            a(f"lw t0, {rx}({base})", f"lw t1, {ry}({base})", f"srai t0, t0, {s}", f"srai t1, t1, {s}",
              "mul t0, t0, t0", "mul t1, t1, t1", "add a0, t0, t1")
        a("lw a0, 0(s10)", "call log2q8", "mv s0, a0")                       # L(Sp)
        a("lw a0, 4(s10)", "lw a1, 8(s10)", "call l64", "sub a0, a0, s0", "sub a0, a0, s0", "sw a0, 0(s11)")
        sq2(12, 16, 6, "s10")
        a("call log2q8", "sub a0, a0, s0", "sw a0, 4(s11)")
        sq2(20, 24, 14, "s10")
        a("call log2q8", "mv s1, a0", "lw a0, 0(s10)", "srli a0, a0, 14", "call log2q8",
          "sub s1, s1, a0", "sub s1, s1, a0", "sw s1, 8(s11)")
        a("lw t0, 28(s10)", "srai t0, t0, 10", "sw t0, 12(s11)")
        a("lw a0, 32(s10)", "lw a1, 36(s10)", "call l64", "sw a0, 16(s11)")
        a("lw a0, 40(s10)", "call log2q8", "sw a0, 20(s11)")
        a.li("s10", PH)
        for j in range(len(MS)):
            sq2(8 * j, 8 * j + 4, 4, "s10")
            a("call log2q8", f"sw a0, {24 + 4 * j}(s11)")
        a("lw a0, 64(s10)", "call log2q8", "mv s0, a0")                      # L(Sw)
        for i, o in enumerate((48, 56)):
            sq2(o, o + 4, 13, "s10")
            a("call log2q8", "sub a0, a0, s0", "sub a0, a0, s0", f"sw a0, {48 + 4 * i}(s11)")
        a.li("s10", L["hist"])
        for k in range(16):
            a(f"lw t0, {4 * k}(s10)", f"sw t0, {56 + 4 * k}(s11)")
        a.readback(L["out"], NF)
        a.halt()
        _log_subroutines(a, L["table"])
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for i, v in enumerate(LOG2_T):
            w[built.layout["table"] + 4 * i] = word(v)
        for i, v in enumerate(COS64 + SIN64):
            w[built.layout["trig"] + 4 * i] = word(v)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        out[N - 1] = tuple(signed(ram[o + i]) for i in range(NF))
        return out


class IntMLP(Kernel):
    """Integer MLP over NF features carried as the stream (f_k, 0), one vector per block of `width` samples."""
    name = "int_mlp"

    def __init__(self, w1, b1, w2, b2, s1=12, width=32):
        self.w1, self.w2 = [list(map(int, r)) for r in w1], [list(map(int, r)) for r in w2]
        self.b1, self.b2 = list(map(int, b1)), list(map(int, b2))
        self.H, self.C, self.NF, self.s1, self.width = len(self.w1), len(self.w2), len(self.w1[0]), int(s1), width
        self.hop = width
        if not (self.NF <= width and all(len(r) == self.NF for r in self.w1) and all(len(r) == self.H for r in self.w2)
                and 1 <= self.s1 <= 24):
            raise ValueError("shape")
        if any(not -32768 <= v <= 32767 for v in sum(self.w1, []) + sum(self.w2, [])):
            raise ValueError("int16 weight codes")
        if any(not -2**30 <= v < 2**30 for v in self.b1 + self.b2):
            raise ValueError("bias range")

    def describe(self):
        return {"kernel": self.name, "features": self.NF, "hidden": self.H, "classes": self.C, "s1": self.s1}

    def forward(self, f):
        h = [min(max(round_half_up(sum(w * v for w, v in zip(r, f)), self.s1) + b, 0), 32767)
             for r, b in zip(self.w1, self.b1)]
        s = [round_half_up(sum(w * v for w, v in zip(r, h)), 12) + b for r, b in zip(self.w2, self.b2)]
        return tuple(s), max(range(self.C), key=lambda c: (s[c], -c))

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(self.width - 1, len(samples), self.width):
            out[e] = self.forward([v[0] for v in samples[e - self.width + 1:e - self.width + 1 + self.NF]])
        return out

    def build(self, block, base=None):
        if block != self.width:
            raise ValueError("one vector per block")
        L = Layout() if base is None else Layout(top=base)
        L.alloc("in", 2 * block)
        L.alloc("params", self.NF * self.H + self.H + self.C * self.H + self.C)
        L.alloc("hid", self.H)
        L.alloc("out", self.C + 1)
        P = L["params"]
        W1, B1 = P, P + 4 * self.NF * self.H
        W2 = B1 + 4 * self.H
        B2 = W2 + 4 * self.C * self.H
        a = Asm()
        a.li("s6", L["in"])
        a.li("s3", L["hid"])
        a.li("s10", L["out"])
        for j in range(self.H):
            a("li s0, 0", "li s1, 0")
            for k in range(self.NF):
                a.li("t6", W1 + 4 * (self.NF * j + k))
                a("lw t2, 0(t6)", f"lw t0, {8 * k}(s6)")
                mac(a, "s0", "s1", "t0", "t2")
            narrow64(a, "s0", "s1", self.s1)
            a.li("t6", B1 + 4 * j)
            a("lw t2, 0(t6)", "add s0, s0, t2", "srai t2, s0, 31", "not t2, t2", "and s0, s0, t2", "li t6, 32767")
            min_u32(a, "s0", "s0", "t6")
            a(f"sw s0, {4 * j}(s3)")
        for c in range(self.C):
            a("li s0, 0", "li s1, 0")
            for j in range(self.H):
                a.li("t6", W2 + 4 * (self.H * c + j))
                a("lw t2, 0(t6)", f"lw t0, {4 * j}(s3)")
                mac(a, "s0", "s1", "t0", "t2")
            narrow64(a, "s0", "s1", 12)
            a.li("t6", B2 + 4 * c)
            a("lw t2, 0(t6)", "add s0, s0, t2", f"sw s0, {4 * c}(s10)")
            if c == 0:
                a("mv s2, s0", "li s4, 0")
            else:
                a("slt t3, s2, s0", "neg t3, t3", "xor t4, s2, s0", "and t4, t4, t3", "xor s2, s2, t4",
                  f"li t5, {c}", "xor t4, s4, t5", "and t4, t4, t3", "xor s4, s4, t4")
        a(f"sw s4, {4 * self.C}(s10)")
        a.readback(L["out"], self.C + 1)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for i, v in enumerate(sum(self.w1, []) + self.b1 + sum(self.w2, []) + self.b2):
            w[built.layout["params"] + 4 * i] = word(v)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        row = [signed(ram[o + c]) for c in range(self.C + 1)]
        out[self.width - 1] = (tuple(row[:self.C]), row[self.C])
        return out


def subroutine(built, prefix):
    """A kernel program (built with codegen.PROOF = False) as inline code for firmware: `_start` removed,
    the exit ecall replaced by a jump past the kernel's helper routines, every label prefixed."""
    import re
    lines = [ln for ln in built.source.splitlines() if ln != "_start:"]
    i = next(k for k in range(len(lines) - 2) if lines[k:k + 3] == ["li a0, 0", "li a7, 93", "ecall"])
    lines[i:i + 3] = [f"j {prefix}end"]
    labels = {ln[:-1] for ln in lines if re.fullmatch(r"\w+:", ln)}
    if not labels:
        return lines + [f"{prefix}end:"]
    pat = re.compile(r"\b(" + "|".join(sorted(labels, key=len, reverse=True)) + r")\b")
    return [pat.sub(lambda m: prefix + m.group(1), ln) for ln in lines] + [f"{prefix}end:"]
