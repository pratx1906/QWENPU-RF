"""Technology classifier on QwenPU's CPU in weights: log-spectral-shape features + integer MLP.

Per window of 1024 Q15 complex samples (hop 1024):
  window powers  w_j = sum over 64 samples of (|x|^2 >> 6), j = 0..15                 (u32)
  DFT16 powers   P_k = sum over F frames (16 samples, hop 1024/F) of |X_k|^2,
                 X_k = round_half_up(sum_n (x_n >> 5) W^{kn}, 15) with Q15 twiddles        (u64)
  features       L(v) = log2(v) in Q8 (16-entry table, linear interpolation; L(0) = 0)
                 f_k  = L(P_k >> 6) - mean_k L(P_k >> 6)   (spectral shape, gain-invariant)
                 f_16 = L(max w) - L(min w + 1)            (burstiness)
  MLP            h_j = clamp(relu((sum_k W1[j][k] f_k) >> 12 rounded + b1[j]), 32767)
                 s_c = (sum_j W2[c][j] h_j) >> 12 rounded + b2[c];   class = first argmax
Weights are runtime RAM data (int16 codes). All arithmetic is RV32 on the CPU in weights.

bandwidth=True (PROTOCOL Amendment 8, P8.2): 64 more features, the occupied-bandwidth shape. Stage A is
spectral.FFTPower(64, Hann, hop 64) on the window, in weights; the host copies its verified power words
(16 frames x 64 bins, u64) into this kernel (`set_fft`), which computes on the CPU
  B_k = (sum_f P_f[k]) >> 10      (= floor(mean over 16 frames) >> 6)
  g_k = L(B_k) - (sum_k L(B_k)) >> 6
and feeds the 81 features to the MLP.
"""
import math

from .codegen import Asm, Layout, round_half_up, signed, word
from .detect import min_u32
from .kernels import Built, Kernel, mac, narrow64, put_pairs

WIN, NB, BW = 1024, 16, 64
LOG2_T = [round(math.log2(1 + k / 16) * 65536) for k in range(17)]


def log2_q8(v: int) -> int:
    """Independent reference of the CPU's log2: Q8, 16-segment linear table; 0 -> 0."""
    if v <= 0:
        return 0
    e = v.bit_length() - 1
    m = (v << (31 - e)) & 0xFFFFFFFF if e <= 31 else v >> (e - 31)
    seg, pos = (m >> 27) & 15, (m >> 11) & 0xFFFF
    frac = LOG2_T[seg] + (((LOG2_T[seg + 1] - LOG2_T[seg]) * pos) >> 16)
    return ((e << 16) + frac) >> 8


class TechClassifier(Kernel):
    name = "tech_classifier"
    hop = WIN

    def __init__(self, w1, b1, w2, b2, frames=16, bandwidth=False):
        self.w1, self.b1, self.w2, self.b2 = [list(map(int, r)) for r in w1], list(map(int, b1)), \
            [list(map(int, r)) for r in w2], list(map(int, b2))
        self.H, self.C, self.F = len(self.w1), len(self.w2), frames
        self.bandwidth = bool(bandwidth)
        self.NF = NB + 1 + (BW if bandwidth else 0)
        self.fft = {}  # window start -> 16 x 64 verified FFT64 powers (bandwidth)
        if bandwidth:
            self.name = "tech_classifier_bw"
        if not (1 <= self.H <= 16 and 2 <= self.C <= 8 and frames in (8, 16, 32, 64)
                and all(len(r) == self.NF for r in self.w1) and all(len(r) == self.H for r in self.w2)):
            raise ValueError(f"{self.NF} features, 1..16 hidden, 2..8 classes, 8/16/32/64 frames")
        if any(not -32768 <= v <= 32767 for v in sum(self.w1, []) + sum(self.w2, []) + self.b1 + self.b2):
            raise ValueError("int16 weight codes")

    def describe(self):
        return {"kernel": self.name, "window": WIN, "dft": NB, "frames": self.F, "hidden": self.H, "classes": self.C,
                "bandwidth_features": self.bandwidth}

    @staticmethod
    def bandwidth_features(fft_frames):
        """Reference of the 64 bandwidth features from 16 frames of 64 FFT64 powers."""
        B = [sum(f[k] for f in fft_frames) >> 10 for k in range(BW)]
        L = [log2_q8(b) for b in B]
        m = sum(L) >> 6
        return [v - m for v in L]

    def set_fft(self, start, fft_frames):
        """Host copy of stage A's verified FFT64 powers for the window starting at `start`."""
        if len(fft_frames) != WIN // BW or any(len(f) != BW for f in fft_frames):
            raise ValueError("16 frames of 64 powers")
        self.fft[start] = [tuple(int(v) for v in f) for f in fft_frames]

    # ------------------------------------------------------------------ reference
    def features(self, x):
        win = [sum((a * a + b * b) >> 6 for a, b in x[j * 64:(j + 1) * 64]) for j in range(16)]
        P = [0] * NB
        step = WIN // self.F
        for f in range(self.F):
            fr = x[f * step:f * step + NB]
            for k in range(NB):
                re = im = 0
                for n, (a, b) in enumerate(fr):
                    c = round(math.cos(2 * math.pi * k * n / NB) * 32767)
                    s = round(-math.sin(2 * math.pi * k * n / NB) * 32767)
                    a5, b5 = a >> 5, b >> 5
                    re += a5 * c - b5 * s
                    im += a5 * s + b5 * c
                re, im = round_half_up(re, 15), round_half_up(im, 15)
                P[k] += re * re + im * im
        L = [log2_q8(p >> 6) for p in P]
        mean = sum(L) >> 4
        return [v - mean for v in L] + [log2_q8(max(win)) - log2_q8(min(win) + 1)]

    def all_features(self, x, start=0):
        f = self.features(x)
        return f + self.bandwidth_features(self.fft[start]) if self.bandwidth else f

    def forward(self, f):
        h = [min(max(round_half_up(sum(w * v for w, v in zip(r, f)), 12) + b, 0), 32767) for r, b in zip(self.w1, self.b1)]
        s = [round_half_up(sum(w * v for w, v in zip(r, h)), 12) + b for r, b in zip(self.w2, self.b2)]
        return tuple(s), max(range(self.C), key=lambda c: (s[c], -c))

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(WIN - 1, len(samples), WIN):
            out[e] = self.forward(self.all_features(samples[e - WIN + 1:e + 1], e - WIN + 1))
        return out

    # ------------------------------------------------------------------ RV32 program
    def build(self, block):
        if block % WIN:
            raise ValueError("block must be a multiple of 1024")
        L = Layout()
        nw = block // WIN
        L.alloc("in", 2 * block)
        L.alloc("table", 17)
        NF = self.NF
        if self.bandwidth and nw != 1:
            raise ValueError("bandwidth: one window per block (block = 1024)")
        L.alloc("params", NF * self.H + self.H + self.C * self.H + self.C)
        L.alloc("bins", 2 * NB)
        L.alloc("feat", NF)
        if self.bandwidth:
            L.alloc("fft", 2 * BW * (WIN // BW))
        L.alloc("hid", self.H)
        L.alloc("out", (self.C + 1) * nw)
        P = L["params"]
        W1, B1 = P, P + 4 * NF * self.H
        W2 = B1 + 4 * self.H
        B2 = W2 + 4 * self.C * self.H
        a = Asm()
        a.li("s11", L["in"])
        a.li("s10", L["out"])
        close_w = a.loop("a7", nw)
        # ---- window powers -> min (s8), max (s9)
        a("mv t6, s11", "li s8, -1", "li s9, 0")
        close_j = a.loop("a6", 16)
        a("li s0, 0")
        close_n = a.loop("a5", 64)
        a("lw t0, 0(t6)", "lw t1, 4(t6)", "mul t0, t0, t0", "mul t1, t1, t1", "add t0, t0, t1",
          "srli t0, t0, 6", "add s0, s0, t0", "addi t6, t6, 8")
        close_n()
        min_u32(a, "s8", "s8", "s0")
        a("sltu t4, s9, s0", "neg t4, t4", "xor t5, s9, s0", "and t5, t5, t4", "xor s9, s9, t5")  # max
        close_j()
        # ---- DFT16 power over F frames
        a.li("t6", L["bins"])
        for k in range(2 * NB):
            a(f"sw zero, {4 * k}(t6)")
        a("mv s7, s11")
        close_f = a.loop("a6", self.F)
        for n in range(NB):  # pre-shift the frame into s0.. via RAM reads per bin (registers are scarce)
            pass
        for k in range(NB):
            a("li s0, 0", "li s1, 0")
            for n in range(NB):
                c = round(math.cos(2 * math.pi * k * n / NB) * 32767)
                s = round(-math.sin(2 * math.pi * k * n / NB) * 32767)
                if not c and not s:
                    continue
                a(f"lw t0, {8 * n}(s7)", f"lw t1, {8 * n + 4}(s7)", "srai t0, t0, 5", "srai t1, t1, 5")
                if c:
                    a(f"li t2, {c}", "mul t3, t0, t2", "add s0, s0, t3", "mul t3, t1, t2", "add s1, s1, t3")
                if s:
                    a(f"li t2, {s}", "mul t3, t1, t2", "sub s0, s0, t3", "mul t3, t0, t2", "add s1, s1, t3")
            a.round_shift("s0", 15)
            a.round_shift("s1", 15)
            a("mul s0, s0, s0", "mul s1, s1, s1", "add s0, s0, s1")
            a.li("t6", L["bins"] + 8 * k)
            a("lw t0, 0(t6)", "lw t1, 4(t6)", "add t0, t0, s0", "sltu t2, t0, s0", "add t1, t1, t2",
              "sw t0, 0(t6)", "sw t1, 4(t6)")
        a(f"addi s7, s7, {8 * (WIN // self.F)}")
        close_f()
        # ---- features: L(P >> 6), mean, shape; burstiness
        a.li("s6", L["feat"])
        a("li s5, 0")  # sum of logs
        for k in range(NB):
            a.li("t6", L["bins"] + 8 * k)
            a("lw t0, 0(t6)", "lw t1, 4(t6)", "srli a0, t0, 6", "slli t1, t1, 26", "or a0, a0, t1",
              "call log2q8", f"sw a0, {4 * k}(s6)", "add s5, s5, a0")
        a("srai s5, s5, 4")
        for k in range(NB):
            a(f"lw t0, {4 * k}(s6)", "sub t0, t0, s5", f"sw t0, {4 * k}(s6)")
        a("mv a0, s9", "call log2q8", "mv s4, a0", "addi a0, s8, 1", "call log2q8", "sub s4, s4, a0",
          f"sw s4, {4 * NB}(s6)")
        if self.bandwidth:  # B_k = (sum_f P_f[k]) >> 10 (64-bit sum), L(B_k), minus the mean
            a("li s5, 0")
            for k in range(BW):
                a("li s0, 0", "li s1, 0")
                a.li("t6", L["fft"] + 8 * k)
                for f in range(WIN // BW):  # frames are 512 bytes apart: step the pointer (12-bit offsets)
                    a("lw t0, 0(t6)", "lw t1, 4(t6)", "add s0, s0, t0", "sltu t2, s0, t0",
                      "add s1, s1, t1", "add s1, s1, t2", f"addi t6, t6, {8 * BW}")
                a("srli a0, s0, 10", "slli t1, s1, 22", "or a0, a0, t1", "call log2q8",
                  f"sw a0, {4 * (NB + 1 + k)}(s6)", "add s5, s5, a0")
            a("srai s5, s5, 6")
            for k in range(BW):
                a(f"lw t0, {4 * (NB + 1 + k)}(s6)", "sub t0, t0, s5", f"sw t0, {4 * (NB + 1 + k)}(s6)")
        # ---- MLP layer 1 (64-bit accumulate)
        a.li("s3", L["hid"])
        for j in range(self.H):
            a("li s0, 0", "li s1, 0")
            for k in range(NF):
                a.li("t6", W1 + 4 * (NF * j + k))
                a("lw t2, 0(t6)", f"lw t0, {4 * k}(s6)")
                mac(a, "s0", "s1", "t0", "t2")
            narrow64(a, "s0", "s1", 12)
            a.li("t6", B1 + 4 * j)
            a("lw t2, 0(t6)", "add s0, s0, t2", "srai t2, s0, 31", "not t2, t2", "and s0, s0, t2", "li t6, 32767")
            min_u32(a, "s0", "s0", "t6")
            a(f"sw s0, {4 * j}(s3)")
        # ---- layer 2 + argmax
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
        a(f"sw s4, {4 * self.C}(s10)", f"addi s10, s10, {4 * (self.C + 1)}")
        a.li("t0", 8 * WIN)
        a("add s11, s11, t0")
        close_w()
        a.readback(L["out"], (self.C + 1) * nw)
        a.halt()
        # log2q8(a0 u32) -> a0 (Q8); 0 -> 0. Clobbers t0-t5.
        a("log2q8:", "bnez a0, lq1", "ret", "lq1:", "li t0, 31",
          "lq2:", "srl t1, a0, t0", "bnez t1, lq3", "addi t0, t0, -1", "j lq2",
          "lq3:", "li t1, 31", "sub t1, t1, t0", "sll t2, a0, t1",
          "srli t3, t2, 27", "andi t3, t3, 15", "srli t4, t2, 11", "li t5, 65535", "and t4, t4, t5")
        a.li("t5", L["table"])
        a("slli t3, t3, 2", "add t5, t5, t3", "lw t1, 0(t5)", "lw t5, 4(t5)", "sub t5, t5, t1",
          "mul t5, t5, t4", "srli t5, t5, 16", "add t1, t1, t5", "slli t0, t0, 16", "add t0, t0, t1",
          "srai a0, t0, 8", "ret")
        return Built(a.source(), L, block)

    def params(self):
        return sum(self.w1, []) + self.b1 + sum(self.w2, []) + self.b2

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for i, v in enumerate(LOG2_T):
            w[built.layout["table"] + 4 * i] = word(v)
        for i, v in enumerate(self.params()):
            w[built.layout["params"] + 4 * i] = word(v)
        if self.bandwidth:  # stage A's verified FFT64 powers (u64: low word, high word)
            for f, frame in enumerate(self.fft[start]):
                for k, pw in enumerate(frame):
                    a_ = built.layout["fft"] + 8 * (BW * f + k)
                    w[a_], w[a_ + 4] = pw & 0xFFFFFFFF, pw >> 32
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for k in range(built.block // WIN):
            row = [signed(ram[o + (self.C + 1) * k + c]) for c in range(self.C + 1)]
            out[k * WIN + WIN - 1] = (tuple(row[:self.C]), row[self.C])
        return out
