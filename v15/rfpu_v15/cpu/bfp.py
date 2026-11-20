"""Block-floating-point front end on the CPU in weights (PROTOCOL Amendment 2).

Input: int32 codes at a fine scale (e.g. Q24), written by the host without clipping.
BlockMax: stateful running maximum of |component| over the stream (u32). The final value M is
copied, as a verified word, into AutoScale, which computes on the CPU s = max(0, bitlen(M) - 15)
and emits x_q = sat16(round_half_up(x, s)) per component, so the stream's peak uses the full Q15
range. The block exponent (fine fractional bits - s) is reported with the stream.
"""
from .codegen import Asm, Layout, round_half_up, signed, word
from .kernels import Built, Kernel, put_pairs


def shift_for(m: int) -> int:
    return max(0, m.bit_length() - 15)


class BlockMax(Kernel):
    name = "block_max"
    state_words = 1

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("state", 1)
        L.alloc("out", block)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        a.li("a1", L["state"])
        a("lw s0, 0(a1)")
        close = a.loop("a3", 2 * block)                       # every component
        a("lw t0, 0(a0)", "srai t1, t0, 31", "xor t0, t0, t1", "sub t0, t0, t1",   # |x|
          "sltu t2, s0, t0", "neg t2, t2", "xor t3, s0, t0", "and t3, t3, t2", "xor s0, s0, t3",
          "andi t4, a3, 1", "beqz t4, bm_skip", "sw s0, 0(a2)", "addi a2, a2, 4", "bm_skip:", "addi a0, a0, 4")  # store after Q (a3 odd)
        close()
        a("sw s0, 0(a1)")
        a.readback(L["out"], block)
        a.readback(L["state"], 1)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o = built.layout["out"] >> 2
        return [ram[o + i] for i in range(built.block)]

    def reference(self, samples):
        m, out = 0, []
        for re, im in samples:
            m = max(m, abs(re), abs(im))
            out.append(m)
        return out


class AutoScale(Kernel):
    """Stateless; M is a runtime word. Outputs (x_q pairs); the shift s is the program's first output word."""
    name = "auto_scale"

    def __init__(self, m: int):
        if not 0 <= m < 2**31:
            raise ValueError("M is a non-negative 31-bit magnitude")
        self.m = int(m)

    def describe(self):
        return {"kernel": self.name, "max_abs": self.m, "shift": shift_for(self.m)}

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("m", 1)
        L.alloc("out", 1 + 2 * block)
        a = Asm()
        a.li("t6", L["m"])
        # s = max(0, bitlen(M) - 15), computed on the CPU (loop over bit positions)
        a("lw t0, 0(t6)", "li s1, 0", "as_len:", "beqz t0, as_done", "srli t0, t0, 1", "addi s1, s1, 1", "j as_len",
          "as_done:", "addi s1, s1, -15", "srai t1, s1, 31", "not t1, t1", "and s1, s1, t1")   # s1 = s
        a.li("a2", L["out"])
        a("sw s1, 0(a2)", "addi a2, a2, 4")
        # rounding constant 2^(s-1) (0 when s = 0)
        a("li s2, 1", "sll s2, s2, s1", "srli s2, s2, 1", "li s3, 32767", "li s4, -32768")
        a.li("a0", L["in"])
        close = a.loop("a3", 2 * block)
        a("lw t0, 0(a0)", "add t0, t0, s2", "sra t0, t0, s1",
          "slt t1, s3, t0", "neg t1, t1", "xor t2, t0, s3", "and t2, t2, t1", "xor t0, t0, t2",      # min(.., 32767)
          "slt t1, t0, s4", "neg t1, t1", "xor t2, t0, s4", "and t2, t2, t1", "xor t0, t0, t2",      # max(.., -32768)
          "sw t0, 0(a2)", "addi a2, a2, 4", "addi a0, a0, 4")
        close()
        a.readback(L["out"], 1 + 2 * block)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {built.layout["m"]: word(self.m)}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o = (built.layout["out"] >> 2) + 1
        return [(signed(ram[o + 2 * i]), signed(ram[o + 2 * i + 1])) for i in range(built.block)]

    def reference(self, samples):
        s = shift_for(self.m)
        q = lambda v: max(-32768, min(32767, round_half_up(v, s)))  # noqa: E731
        return [(q(re), q(im)) for re, im in samples]
