"""Runtime-weight inference on the CPU in weights: FFT4 power features -> dense -> ReLU -> dense -> argmax.

Per frame of 4 samples (frames end at e % 4 == 3): X = exact 4-point DFT, P_k = |X_k|^2 (64-bit),
f_k = min(P_k >> fshift, 32767); h_j = clamp(relu(round_half_up(sum_k W1[j][k] f_k, 12) + b1[j]), 32767);
s_c = round_half_up(sum_j W2[c][j] h_j, 12) + b2[c]; class = first argmax. Weights are Q12 codes
in RAM, so a different model needs no new program and no new processor weights.
"""
from .codegen import Asm, Layout, round_half_up, word, signed
from .kernels import Built, Kernel, put_pairs, mac, narrow64
from .detect import min_u32


class SpectralMLP(Kernel):
    name = "spectral_mlp"
    hop = 4

    def __init__(self, w1, b1, w2, b2, fshift=16):
        self.w1, self.b1, self.w2, self.b2 = [list(map(int, r)) for r in w1], list(map(int, b1)), \
            [list(map(int, r)) for r in w2], list(map(int, b2))
        self.H, self.C = len(self.w1), len(self.w2)
        if not (1 <= self.H <= 8 and 2 <= self.C <= 8 and all(len(r) == 4 for r in self.w1)
                and all(len(r) == self.H for r in self.w2) and 4 <= fshift < 32):
            raise ValueError("4 features, 1..8 hidden, 2..8 classes, fshift 4..31")
        if any(not -32768 <= v <= 32767 for v in sum(self.w1, []) + sum(self.w2, []) + self.b1 + self.b2):
            raise ValueError("weights are 16-bit codes")
        self.fshift = int(fshift)

    def describe(self):
        return {"kernel": self.name, "features": 4, "hidden": self.H, "classes": self.C, "fshift": self.fshift}

    def build(self, block):
        L = Layout()
        frames = block // 4
        L.alloc("in", 2 * block)
        L.alloc("params", 1 + 4 * self.H + self.H + self.C * self.H + self.C)
        L.alloc("hidden", self.H)
        L.alloc("out", (self.C + 1) * frames)
        P = L["params"]
        W1, B1 = P + 4, P + 4 + 16 * self.H
        W2 = B1 + 4 * self.H
        B2 = W2 + 4 * self.C * self.H
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        a.li("a1", P)
        a("lw s11, 0(a1)", "li t0, 32", "sub s10, t0, s11")  # fshift and 32 - fshift
        a.li("a5", L["hidden"])
        close = a.loop("a3", frames)
        # exact 4-point DFT of samples a..d (rectangular window)
        a("lw s0, 0(a0)", "lw s1, 4(a0)", "lw s2, 8(a0)", "lw s3, 12(a0)",
          "lw s4, 16(a0)", "lw s5, 20(a0)", "lw s6, 24(a0)", "lw s7, 28(a0)",
          "add t0, s0, s4", "add t1, s1, s5", "add t2, s2, s6", "add t3, s3, s7",
          "sub s4, s0, s4", "sub s5, s1, s5", "sub s6, s2, s6", "sub s7, s3, s7",
          "add s0, t0, t2", "add s1, t1, t3", "sub s2, t0, t2", "sub s3, t1, t3",
          "add t0, s4, s7", "sub t1, s5, s6", "sub t2, s4, s7", "add t3, s5, s6")
        # move X1, X3 out of the temporaries; features overwrite the dead (a-c), (b-d) registers
        a("mv s8, t0", "mv s9, t1", "mv a6, t2", "mv a7, t3")
        bins = [("s0", "s1"), ("s8", "s9"), ("s2", "s3"), ("a6", "a7")]
        feats = ["s4", "s5", "s6", "s7"]
        for k, (r, i) in enumerate(bins):
            # P < 2^35 and fshift >= 4, so the shifted power fits 32 bits before saturation
            a(f"mul t4, {r}, {r}", f"mulh t5, {r}, {r}", f"mul t6, {i}, {i}", f"mulh t2, {i}, {i}",
              "add t4, t4, t6", "sltu t3, t4, t6", "add t5, t5, t2", "add t5, t5, t3",
              "srl t4, t4, s11", "sll t5, t5, s10", f"or {feats[k]}, t4, t5", "li t6, 32767")
            min_u32(a, feats[k], feats[k], "t6")
        for j in range(self.H):
            a("li s8, 0", "li s9, 0")
            for k in range(4):
                a(f"lw t2, {W1 - P + 4 * (4 * j + k)}(a1)")
                mac(a, "s8", "s9", feats[k], "t2")
            narrow64(a, "s8", "s9", 12)
            a(f"lw t2, {B1 - P + 4 * j}(a1)", "add s8, s8, t2",
              "srai t2, s8, 31", "not t2, t2", "and s8, s8, t2", "li t6, 32767")  # ReLU
            min_u32(a, "s8", "s8", "t6")
            a(f"sw s8, {4 * j}(a5)")
        for c in range(self.C):
            a("li s8, 0", "li s9, 0")
            for j in range(self.H):
                a(f"lw t2, {W2 - P + 4 * (self.H * c + j)}(a1)", f"lw t3, {4 * j}(a5)")
                mac(a, "s8", "s9", "t3", "t2")
            narrow64(a, "s8", "s9", 12)
            a(f"lw t2, {B2 - P + 4 * c}(a1)", "add s8, s8, t2", f"sw s8, {4 * c}(a2)")
            if c == 0:
                a("mv s0, s8", "li s1, 0")  # best score, best class
            else:
                a("slt t3, s0, s8", "neg t3, t3", "xor t4, s0, s8", "and t4, t4, t3", "xor s0, s0, t4",
                  f"li t5, {c}", "xor t4, s1, t5", "and t4, t4, t3", "xor s1, s1, t4")
        a(f"sw s1, {4 * self.C}(a2)", f"addi a2, a2, {4 * (self.C + 1)}", "addi a0, a0, 32")
        close()
        a.readback(L["out"], (self.C + 1) * frames)
        a.halt()
        return Built(a.source(), L, block)

    def params(self):
        return [self.fshift] + sum(self.w1, []) + self.b1 + sum(self.w2, []) + self.b2

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        for i, v in enumerate(self.params()):
            w[built.layout["params"] + 4 * i] = word(v)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for f in range(built.block // 4):
            row = [signed(ram[o + (self.C + 1) * f + c]) for c in range(self.C + 1)]
            out[4 * f + 3] = (tuple(row[:self.C]), row[self.C])
        return out

    def features(self, frame):
        x = list(frame)
        f = []
        for k in range(4):
            re = im = 0
            for n, (xr, xi) in enumerate(x):
                c, s = [(1, 0), (0, -1), (-1, 0), (0, 1)][(k * n) % 4]
                re += xr * c - xi * s
                im += xr * s + xi * c
            f.append(min((re * re + im * im) >> self.fshift, 32767))
        return f

    def forward(self, f):
        h = [min(max(round_half_up(sum(w * x for w, x in zip(row, f)), 12) + b, 0), 32767)
             for row, b in zip(self.w1, self.b1)]
        s = [round_half_up(sum(w * x for w, x in zip(row, h)), 12) + b for row, b in zip(self.w2, self.b2)]
        return tuple(s), max(range(self.C), key=lambda c: (s[c], -c))

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(3, len(samples), 4):
            out[e] = self.forward(self.features(samples[e - 3:e + 1]))
        return out
