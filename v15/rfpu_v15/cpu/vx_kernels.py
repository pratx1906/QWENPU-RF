"""Kernels that use QwenPU's VX vector unit (chipformer/rv/vx.py, docs/vx.md) from the RV32 CPU.

The unit has its own memories (VM vector registers, the int8 activation vector VA, the int4
weight ROM WROM); the CPU reaches them only through custom-2 ops: VPUT (two registers into a VM
pair), VQ (VM -> int8 VA), VRED (sum / sum of squares of a VA chunk prefix), VMAT (VA chunk .
WROM row-chunk, times the row scale S_r), VGET (a tracker into a register). No VM element can be
read back directly, so results reach the scalar side through VRED/VGET (sums) or VMAT's argmax.

Every program here is branch-uniform (fixed-count loops only); every output word is stored to
RAM and loaded back before halt. Each kernel has an independent pure-Python reference of the
integer contract below (written from vx.py's documented semantics, without calling it).

VQ narrowing used here (vx.py VQ with r = 16384 or 32767):
    vq(v; sh, r) = clamp(sign(p) * ((|p| + 2^14) >> 15), -127, 127), p = clamp16(v >> sh) * r
    r = 16384: vq = clamp(sign(f) * ((|f| + 1) >> 1), +-127), f = clamp16(v >> sh)
               (f / 2 rounded half away from zero)
    r = 32767, 0 <= v <= 127, sh = 0: vq = v (identity; used to move scalar int8 values into VA)
"""
from .codegen import Asm, Layout, word, signed
from .kernels import Built, Kernel, put_pairs
from .detect import min_u32

CHUNK_CELLS = 72  # WROM row-chunk: 64 int4 cells (u = w + 8), 4 scale nibbles (LSB first), 4 unused


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def vq_ref(v: int, sh: int, r: int) -> int:
    """VQ's int8 quantizer (vx.py contract), re-implemented."""
    p = clamp(v >> sh, -32768, 32767) * r
    q = (abs(p) + (1 << 14)) >> 15
    return clamp(q if p >= 0 else -q, -127, 127)


def sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


# ---- instruction-field packers (vx.py FIELDS) ----

def vq_cfg(src, eoff, n, dst, sh):
    return src | eoff << 4 | n << 10 | dst << 17 | sh << 19


def vmat_cfg(dst, n, k, mode, keep, chunk):
    return dst | n << 4 | k << 11 | mode << 13 | keep << 14 | chunk << 15


def vred_cfg(n, mode, chunk):
    return n | mode << 7 | chunk << 8


def emit_vq(a: Asm, src, eoff, n, sh, off, r):
    a.li("t4", vq_cfg(src, eoff, n, 0, sh))
    a.li("t5", off)
    a.li("t6", r)
    a("vq t4, t5, t6")


class VXEnergy(Kernel):
    """Per window of W complex Q15 samples (ending at e with e % W == W-1, hop W):
        q = vq(x; shift, 16384) for each I and Q component (int8, see module doc)
        E = sum over the window of qI^2 + qQ^2           (0 <= E <= 2 W 127^2)
    Precision trade: each component keeps 8 bits (about 48 dB of per-sample dynamic range at
    shift 7: full scale 32767 -> 127 codes of 256 Q15 LSBs each); E is exact on those codes.
    Machine: per sample the CPU loads I, Q and VPUTs them into a VM pair; per window VQ narrows
    the 2W values into VA (one chunk per 64 values), VRED mode 0 sums squares per chunk, VGET
    reads each sum, the CPU adds and stores one word."""
    name = "vx_energy"

    def __init__(self, window=64, shift=7):
        if window not in (8, 16, 32, 64) or not 0 <= shift <= 15:
            raise ValueError("window 8, 16, 32 or 64; shift 0..15")
        self.W, self.shift, self.hop = window, shift, window
        self.chunks = (2 * window + 63) // 64
        self.per_chunk = 2 * window // self.chunks

    def describe(self):
        return {"kernel": self.name, "window": self.W, "shift": self.shift, "vx": True,
                "contract": "E = sum vq(x; shift, 16384)^2 over I and Q"}

    def build(self, block):
        if block % self.W:
            raise ValueError("block must be a multiple of the window")
        L = Layout()
        windows = block // self.W
        L.alloc("in", 2 * block)
        L.alloc("out", windows)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        a("li s3, 1")  # VGET selector: vsum
        close_w = a.loop("a3", windows)
        a("li a5, 0")  # VPUT target: VM register a5 >> 5, pair a5 & 31 (v0 then v1)
        close_s = a.loop("a4", self.W)
        a("lw t0, 0(a0)", "lw t1, 4(a0)", "vput a5, t0, t1", "addi a5, a5, 1", "addi a0, a0, 8")
        close_s()
        a("li s0, 0")
        for c in range(self.chunks):
            emit_vq(a, c, 0, self.per_chunk, self.shift, 64 * c, 16384)
            a.li("t4", vred_cfg(self.per_chunk, 0, c))
            a("vred t4, zero, zero", "vget s1, s3, zero", "add s0, s0, s1")
        a("sw s0, 0(a2)", "addi a2, a2, 4")
        close_w()
        a.readback(L["out"], windows)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for k in range(built.block // self.W):
            out[k * self.W + self.W - 1] = ram[o + k]
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(self.W - 1, len(samples), self.W):
            total = 0
            for re, im in samples[e - self.W + 1:e + 1]:
                for v in (re, im):
                    f = clamp(v >> self.shift, -32768, 32767)
                    q = clamp((abs(f) + 1) >> 1, 0, 127)
                    total += q * q  # the sign does not matter for the square
            out[e] = total
        return out


class ScalarEnergy(Kernel):
    """Scalar baseline for token comparison: per window E = sum |x|^2, exact 64-bit (2 words)."""
    name = "scalar_energy"

    def __init__(self, window=64):
        self.W = self.hop = window

    def build(self, block):
        L = Layout()
        windows = block // self.W
        L.alloc("in", 2 * block)
        L.alloc("out", 2 * windows)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close_w = a.loop("a3", windows)
        a("li s0, 0", "li s1, 0")
        close_s = a.loop("a4", self.W)
        a("lw t0, 0(a0)", "lw t1, 4(a0)", "mul t2, t0, t0", "mul t3, t1, t1", "add t2, t2, t3",
          "add s0, s0, t2", "sltu t4, s0, t2", "add s1, s1, t4", "addi a0, a0, 8")
        close_s()
        a("sw s0, 0(a2)", "sw s1, 4(a2)", "addi a2, a2, 8")
        close_w()
        a.readback(L["out"], 2 * windows)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for k in range(built.block // self.W):
            out[k * self.W + self.W - 1] = ram[o + 2 * k] | ram[o + 2 * k + 1] << 32
        return out

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(self.W - 1, len(samples), self.W):
            out[e] = sum(re * re + im * im for re, im in samples[e - self.W + 1:e + 1])
        return out


class VXDense:
    """y = W x + b on VMAT, for up to 63 int8 inputs in one VA chunk (lane BIAS_LANE holds the
    constant bias activation) and up to 64 outputs:
        acc_r = sum_i w[r][i] x_i + b[r] * bias_act      (w, b int4 in [-8, 7]; x int8)
        y_r   = sx31(acc_r * S_r mod 2^32)               (S_r < 2^16, the row's WROM scale)
    Rows live in WROM row-chunks (3 a row pair, k = 1 addressing): row r is chunk 3 base + r.
    Mode 0 writes y into a VM register (rows padded to an even count with zero rows); mode 1
    tracks only the argmax (first index wins ties; y must stay inside +-2^30)."""
    # Lane 8: the bias activation shares the inputs' VM register and is written by the same
    # chunk-aligned VQ. A separate 2-element VQ at VA offset 14 made the compiled circuit disagree
    # with vx.py on the next VMAT (v15/reports/vx-neural.json); upstream tests only aligned writes.
    BIAS_LANE = 8

    def __init__(self, weights, scales, bias=None):
        self.w = [list(map(int, r)) for r in weights]
        self.rows, self.inputs = len(self.w), len(self.w[0])
        self.s = list(map(int, scales))
        self.b = list(map(int, bias)) if bias is not None else [0] * self.rows
        if not (1 <= self.rows <= 64 and 1 <= self.inputs <= self.BIAS_LANE
                and all(len(r) == self.inputs for r in self.w)
                and len(self.s) == len(self.b) == self.rows):
            raise ValueError("1..64 rows of 1..8 inputs, one scale and bias per row")
        if any(not -8 <= v <= 7 for v in sum(self.w, []) + self.b) or any(not 0 <= s < 1 << 16 for s in self.s):
            raise ValueError("int4 weights / biases, scales < 2^16")

    def padded_rows(self):
        return self.rows + self.rows % 2

    def wrom_chunks(self):
        """Row-chunks as 72-cell byte strings (padding rows: all-zero weights, scale 0)."""
        out = []
        for r in range(self.padded_rows()):
            lanes = [0] * 64
            if r < self.rows:
                lanes[:self.inputs] = self.w[r]
                lanes[self.BIAS_LANE] = self.b[r]
            sc = self.s[r] if r < self.rows else 0
            out.append(bytes(v + 8 for v in lanes) + bytes(sc >> 4 * i & 15 for i in range(4)) + bytes(4))
        return out

    def emit(self, a: Asm, base: int, chunk: int, mode: int, dst: int = 0):
        n = self.padded_rows() if mode == 0 else self.rows
        a.li("t4", vmat_cfg(dst, n, 1, mode, 0, chunk))
        a.li("t5", base)  # doff 0, start 0
        a("vmat t4, t5, zero")

    def forward(self, x, bias_act):
        acc = [sum(w * v for w, v in zip(row, x)) + b * bias_act for row, b in zip(self.w, self.b)]
        return [sx(a * s, 31) for a, s in zip(acc, self.s)]

    @staticmethod
    def argmax(y):
        return max(range(len(y)), key=lambda c: (y[c], -c))


def wrom_source(chunks) -> str:
    lines = [".wrom"]
    for ch in chunks:
        for i in range(0, len(ch), 24):
            lines.append(".byte " + ", ".join(str(b) for b in ch[i:i + 24]))
    lines.append(".text")
    return "\n".join(lines)


class VXSpectralMLP(Kernel):
    """FFT4 power features -> VMAT dense -> VQ int8 -> scalar ReLU -> VMAT dense -> argmax.

    Per frame of 4 samples (frames end at e % 4 == 3):
      P_k = |X_k|^2 of the exact 4-point DFT (64-bit);  f_k = min(P_k >> fshift, 127)    (int8)
      y_j = sx31((sum_k W1[j][k] f_k + b1[j] A) * S1[j])            (VXDense, A = bias_act)
      h_j = max(vq(y_j; sh1, r1), 0)                                  (int8 in 0..127)
      z_c = (sum_j W2[c][j] h_j + b2[c] A) * S2[c]                    (|z| < 2^30 always)
      class = first argmax of z_c
    Output per frame: (h_0..h_{H-1}, class). W1, W2, b1, b2 are int4, S1, S2 < 2^16: they are the
    program's .wrom data, written by the assembler, not processor weights. (With 4 int8 features
    and int4 weights |acc| <= 5080, so y never wraps.)
    Machine: features on the scalar CPU; VPUT them into VM v0, VQ (identity) into VA chunk 0;
    VMAT mode 0 -> VM v2; VQ (sh1, r1) -> VA chunk 1; VRED mode 1 prefix sums + VGET read each h
    back (h_j = prefix_j - prefix_{j-1}); ReLU on the CPU; VPUT into VM v3, VQ (identity) into VA
    chunk 2; VMAT mode 1 (argmax) over the output rows; VGET varg & 63 is the class."""
    name = "vx_spectral_mlp"
    hop = 4

    def __init__(self, w1, s1, b1, w2, s2, b2, fshift=16, sh1=8, r1=16384, bias_act=64):
        self.l1, self.l2 = VXDense(w1, s1, b1), VXDense(w2, s2, b2)
        self.H, self.C = self.l1.rows, self.l2.rows
        if not (self.l1.inputs == 4 and 1 <= self.H <= 8 and 2 <= self.C <= 8 and self.l2.inputs == self.H
                and 4 <= fshift < 32 and 0 <= sh1 < 32 and 0 <= r1 < 1 << 15 and 0 <= bias_act <= 127):
            raise ValueError("4 features, 1..8 hidden, 2..8 classes, fshift 4..31, sh1 < 32, r1 < 2^15, bias_act <= 127")
        self.fshift, self.sh1, self.r1, self.A = int(fshift), int(sh1), int(r1), int(bias_act)
        self.base2 = (self.l1.padded_rows() + 2) // 3  # row pair of the output layer's first row

    def describe(self):
        return {"kernel": self.name, "features": 4, "hidden": self.H, "classes": self.C, "fshift": self.fshift,
                "sh1": self.sh1, "r1": self.r1, "bias_act": self.A, "vx": True}

    def wrom(self):
        c1 = self.l1.wrom_chunks()
        pad = [bytes([8] * 64 + [0] * 8)] * (3 * self.base2 - len(c1))
        return c1 + pad + self.l2.wrom_chunks()[:self.C]

    def build(self, block):
        L = Layout()
        frames = block // 4
        L.alloc("in", 2 * block)
        L.alloc("fshift", 1)
        L.alloc("out", (self.H + 1) * frames)
        a = Asm()
        a.li("a1", L["fshift"])
        a("lw s11, 0(a1)", "li t0, 32", "sub s10, t0, s11")  # fshift and 32 - fshift
        # the bias activation A: VM v0 and v3 element 8 (pair 4), carried into VA lane 8 of chunks 0
        # and 2 by the same aligned 16-element identity VQ that carries the features / hidden values
        a.li("t1", self.A)
        a("li t0, 4", "vput t0, t1, zero", "li t0, 100", "vput t0, t1, zero")
        a("li s3, 1", "li a4, 2")  # VGET selectors: vsum, varg
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", frames)
        a("lw s0, 0(a0)", "lw s1, 4(a0)", "lw s2, 8(a0)", "lw s3, 12(a0)",
          "lw s4, 16(a0)", "lw s5, 20(a0)", "lw s6, 24(a0)", "lw s7, 28(a0)",
          "add t0, s0, s4", "add t1, s1, s5", "add t2, s2, s6", "add t3, s3, s7",
          "sub s4, s0, s4", "sub s5, s1, s5", "sub s6, s2, s6", "sub s7, s3, s7",
          "add s0, t0, t2", "add s1, t1, t3", "sub s2, t0, t2", "sub s3, t1, t3",
          "add t0, s4, s7", "sub t1, s5, s6", "sub t2, s4, s7", "add t3, s5, s6",
          "mv s8, t0", "mv s9, t1", "mv a6, t2", "mv a7, t3")
        bins = [("s0", "s1"), ("s8", "s9"), ("s2", "s3"), ("a6", "a7")]
        feats = ["s4", "s5", "s6", "s7"]
        for k, (r, i) in enumerate(bins):
            a(f"mul t4, {r}, {r}", f"mulh t5, {r}, {r}", f"mul t6, {i}, {i}", f"mulh t2, {i}, {i}",
              "add t4, t4, t6", "sltu t3, t4, t6", "add t5, t5, t2", "add t5, t5, t3",
              "srl t4, t4, s11", "sll t5, t5, s10", f"or {feats[k]}, t4, t5", "li t6, 127")
            min_u32(a, feats[k], feats[k], "t6")
        a("li t0, 0", "vput t0, s4, s5", "li t0, 1", "vput t0, s6, s7")
        emit_vq(a, 0, 0, 16, 0, 0, 32767)
        self.l1.emit(a, 0, 0, 0, dst=2)
        emit_vq(a, 2, 0, 8, self.sh1, 64, self.r1)
        a("li s3, 1", "li s0, 0")  # vsum selector; previous prefix sum
        hregs = ["s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"][:self.H]
        a("mv a5, s10", "mv a1, s11")  # keep 32 - fshift and fshift while s10, s11 hold h
        for j, h in enumerate(hregs):
            a.li("t4", vred_cfg(j + 1, 1, 1))
            a("vred t4, zero, zero", "vget s1, s3, zero", f"sub {h}, s1, s0", "mv s0, s1",
              f"srai t0, {h}, 31", "not t0, t0", f"and {h}, {h}, t0", f"sw {h}, {4 * j}(a2)")
        for p in range((self.H + 1) // 2):
            hi = hregs[2 * p + 1] if 2 * p + 1 < self.H else "zero"
            a.li("t0", 96 + p)
            a(f"vput t0, {hregs[2 * p]}, {hi}")
        emit_vq(a, 3, 0, 16, 0, 128, 32767)
        self.l2.emit(a, self.base2, 2, 1)
        a("vget t0, a4, zero", "andi t0, t0, 63", f"sw t0, {4 * self.H}(a2)",
          f"addi a2, a2, {4 * (self.H + 1)}", "addi a0, a0, 32", "mv s10, a5", "mv s11, a1")
        close()
        a.readback(L["out"], (self.H + 1) * frames)
        a.halt()
        return Built(wrom_source(self.wrom()) + "\n" + a.source(), L, block)

    def init_words(self, built, context, start):
        w = {built.layout["fshift"]: word(self.fshift)}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for f in range(built.block // 4):
            row = [signed(ram[o + (self.H + 1) * f + c]) for c in range(self.H + 1)]
            out[4 * f + 3] = (tuple(row[:self.H]), row[self.H])
        return out

    # ---- independent reference ----
    def features(self, frame):
        f = []
        for k in range(4):
            re = im = 0
            for n, (xr, xi) in enumerate(frame):
                c, s = [(1, 0), (0, -1), (-1, 0), (0, 1)][(k * n) % 4]
                re += xr * c - xi * s
                im += xr * s + xi * c
            f.append(min((re * re + im * im) >> self.fshift, 127))
        return f

    def forward(self, f):
        y = self.l1.forward(f, self.A)
        h = [max(vq_ref(v, self.sh1, self.r1), 0) for v in y]
        z = self.l2.forward(h, self.A)
        return tuple(h), VXDense.argmax(z)

    def reference(self, samples):
        out = [None] * len(samples)
        for e in range(3, len(samples), 4):
            out[e] = self.forward(self.features(samples[e - 3:e + 1]))
        return out


class VXDenseFrames(Kernel):
    """VXDense alone, per frame of N = inputs consecutive samples (hop N, frames end at
    e % N == N-1): x_i = vq(I_i; sh_in, 16384) (Q components are ignored), y = VXDense(x),
    output (q_0..q_{R-1}, argmax_r y_r) with q_r = vq(y_r; sh_out, r_out). VM elements cannot be
    read by the CPU, so the int8 q_r come back through VRED mode-1 prefix sums + VGET, and the
    exact-y argmax through a second VMAT in argmax mode."""
    name = "vx_dense"

    def __init__(self, dense: VXDense, sh_in=8, sh_out=8, r_out=16384, bias_act=64):
        if dense.inputs not in (2, 4, 8) or dense.rows > 8:
            raise ValueError("2, 4 or 8 inputs and at most 8 rows")
        self.d, self.sh_in, self.sh_out, self.r_out, self.A = dense, sh_in, sh_out, r_out, bias_act
        self.hop = dense.inputs

    def build(self, block):
        L, R, N = Layout(), self.d.rows, self.d.inputs
        frames = block // N
        L.alloc("in", 2 * block)
        L.alloc("out", (R + 1) * frames)
        a = Asm()
        # bias A in v0 element 8 (pair 4), pre-scaled so vq(sh_in, 16384) yields exactly A
        a.li("t1", self.A << (self.sh_in + 1))
        a("li t0, 4", "vput t0, t1, zero")
        a("li s3, 1", "li a4, 2")
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", frames)
        for p in range(N // 2):
            a(f"lw t0, {16 * p}(a0)", f"lw t1, {16 * p + 8}(a0)")
            a.li("t2", p)
            a("vput t2, t0, t1")
        emit_vq(a, 0, 0, 16, self.sh_in, 0, 16384)
        self.d.emit(a, 0, 0, 0, dst=2)
        emit_vq(a, 2, 0, 8, self.sh_out, 64, self.r_out)
        a("li s0, 0")
        for j in range(R):
            a.li("t4", vred_cfg(j + 1, 1, 1))
            a("vred t4, zero, zero", "vget s1, s3, zero", "sub t0, s1, s0", "mv s0, s1", f"sw t0, {4 * j}(a2)")
        self.d.emit(a, 0, 0, 1)
        a("vget t0, a4, zero", "andi t0, t0, 63", f"sw t0, {4 * R}(a2)",
          f"addi a2, a2, {4 * (R + 1)}", f"addi a0, a0, {8 * N}")
        close()
        a.readback(L["out"], (R + 1) * frames)
        a.halt()
        return Built(wrom_source(self.d.wrom_chunks()) + "\n" + a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        R, N = self.d.rows, self.d.inputs
        o, out = built.layout["out"] >> 2, [None] * built.block
        for f in range(built.block // N):
            row = [signed(ram[o + (R + 1) * f + c]) for c in range(R + 1)]
            out[N * f + N - 1] = (tuple(row[:R]), row[R])
        return out

    def reference(self, samples):
        N, out = self.d.inputs, [None] * len(samples)
        for e in range(N - 1, len(samples), N):
            x = [clamp(sign_round(re >> self.sh_in), -127, 127) for re, _ in samples[e - N + 1:e + 1]]
            y = self.d.forward(x, self.A)
            out[e] = (tuple(vq_ref(v, self.sh_out, self.r_out) for v in y), VXDense.argmax(y))
        return out


def sign_round(f: int) -> int:
    """f / 2 rounded half away from zero (vq with r = 16384 before its clamp)."""
    q = (abs(f) + 1) >> 1
    return q if f >= 0 else -q
