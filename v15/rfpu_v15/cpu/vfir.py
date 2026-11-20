"""Complex Q15 FIR on the new VX instruction VFIR (QwenPU branch rfpu-vfir; needs that chipformer first
on the path). Taps are written once into VM register v8 (reversed), the filter state ping-pongs between
v0 and v1, and every input sample costs one `vfir` instruction instead of ~4N scalar multiply-adds.

Contract (VFIR's, which differs from kernels.FIR): each tap product is rounded on its own,
y[n] = sat16( sum_k round_half_up(h_k x[n-k], 15) ) per component (int16 output, packed in rd).
Stateless across chunks: the N-1 history samples are re-run at the start of each chunk.

guard=True (PROTOCOL Amendment 8, VFIR's guard bit): taps are stored as 4 h (|4 h| <= 32767), each tap
product is still rounded at 2^-15 (so at 1/4 output LSB), a rounding constant 2 enters once through the
chain pair, and rd = sat16(y >> 2): y[n] = sat16((2 + sum_k round_half_up(4 h_k x[n-k], 15)) >> 2). The
state block sits at pairs 16..31 of v0 / v1 (offset 32) with the constant at pair 15.

scale=s (PROTOCOL Amendment 12, non-guard only): taps are stored as 2^s h, so rd carries s extra
fractional bits (units of 2^-s output LSB) with 2^s less headroom; the caller picks s from a verified
block maximum of the guard output (adaptive output scale).
"""
from .codegen import Asm, Layout, round_half_up, signed
from .kernels import Built, Kernel, put_pairs


def sat16(v):
    return max(-32768, min(32767, v))


class VXFir(Kernel):
    name = "vx_fir"

    def __init__(self, taps, packed=True, guard=False, scale=0):
        if scale and (guard or not 0 <= scale <= 4):
            raise ValueError("scale: non-guard only, 0..4 bits")
        self.scale = int(scale)
        taps = [(int(r) << self.scale, int(i) << self.scale) for r, i in taps]
        if not 1 <= len(taps) <= 16 or any(not -32768 <= v <= 32767 for t in taps for v in t):
            raise ValueError("1..16 int16 complex taps (one VFIR block)")
        if guard and any(not -8192 <= v <= 8191 for t in taps for v in t):
            raise ValueError("guard: taps must satisfy |4 h| <= 32767")
        self.guard = bool(guard)
        self.taps = [tuple(map(int, t)) for t in taps]
        self.history = len(taps) - 1
        self.name = f"vx_fir{len(taps)}" + ("_packed" if packed else "") + ("_guard" if guard else "") + (f"_s{scale}" if scale else "")
        # packed: the host lays each sample out as one word (re | im << 16) and reads rd words back;
        # a layout choice only, so the CPU skips the pack/unpack instructions.
        self.packed = packed

    def describe(self):
        return {"kernel": self.name, "taps": len(self.taps), "instruction": "VFIR (custom-2 funct7=1)", "vx": True,
                "packed_io": self.packed, "guard": self.guard, "output_scale_bits": self.scale}

    def build(self, block):
        L = Layout()
        w = 1 if self.packed else 2
        L.alloc("in", w * (self.history + block))
        L.alloc("out", w * block)
        N, a = len(self.taps), Asm()
        sc, p0 = (4, 16) if self.guard else (1, 0)
        for j, (hr, hi) in enumerate(reversed(self.taps)):        # VM v8 pair p0 + j = sc h[N-1-j]
            a.li("t0", 8 * 32 + p0 + j)
            a.li("t1", sc * hr)
            a.li("t2", sc * hi)
            a("vput t0, t1, t2")
        cfg = 0 | 1 << 4 | 0 << 7 | N << 10        # dst v0, src v1, taps v8, n, eoff 0, no chain
        if self.guard:  # offset 32, chain from pair 15 (the rounding constant 2), rd = sat16(y >> 2)
            for reg in (0, 1):
                a.li("t0", 32 * reg + 15)
                a("li t1, 2", "li t2, 2", "vput t0, t1, t2")
            cfg |= 32 << 15 | 1 << 21 | 1 << 22
        a.li("s0", cfg)
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        close = a.loop("a3", self.history + block)
        if self.packed:
            a("lw t2, 0(a0)", "vfir t3, t2, s0", "sw t3, 0(a2)", "xori s0, s0, 17", "addi a0, a0, 4")
        else:
            a("lw t0, 0(a0)", "lw t1, 4(a0)", "slli t0, t0, 16", "srli t0, t0, 16", "slli t1, t1, 16", "or t2, t0, t1",
              "vfir t3, t2, s0", "slli t4, t3, 16", "srai t4, t4, 16", "srai t5, t3, 16",
              "sw t4, 0(a2)", "sw t5, 4(a2)", "xori s0, s0, 17", "addi a0, a0, 8")
        # outputs of the history samples land in a scratch slot (a2 stays put); a2 advances only after
        # a block sample (index >= history), when a0 already points past it
        a(f"li t6, {L['in'] + 4 * w * self.history}", "sltu t6, t6, a0", f"slli t6, t6, {2 if self.packed else 3}", "add a2, a2, t6")
        close()
        a.readback(L["out"], w * block)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        if self.packed:
            for i, (re, im) in enumerate(context):
                w[built.layout["in"] + 4 * i] = (re & 0xFFFF) | (im & 0xFFFF) << 16
        else:
            put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o = built.layout["out"] >> 2
        if self.packed:
            unpack = lambda v: v - 65536 if v & 0x8000 else v  # noqa: E731
            return [(unpack(ram[o + i] & 0xFFFF), unpack(ram[o + i] >> 16)) for i in range(built.block)]
        return [(signed(ram[o + 2 * i]), signed(ram[o + 2 * i + 1])) for i in range(built.block)]

    def reference(self, samples):
        out = []
        for n in range(len(samples)):
            re = im = 0
            for k, (hr, hi) in enumerate(self.taps):
                xr, xi = samples[n - k] if n - k >= 0 else (0, 0)
                if self.guard:
                    re += round_half_up(4 * hr * xr - 4 * hi * xi, 15)
                    im += round_half_up(4 * hr * xi + 4 * hi * xr, 15)
                else:
                    re += round_half_up(hr * xr - hi * xi, 15)
                    im += round_half_up(hr * xi + hi * xr, 15)
            if self.guard:
                re, im = (re + 2) >> 2, (im + 2) >> 2
            out.append((sat16(re), sat16(im)))
        return out
