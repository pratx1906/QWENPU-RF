"""Detection kernels: windowed energy with a minimum-statistics noise floor, burst edges, peak search.

The noise floor is the minimum window mean over the last M windows including the current one
(minimum statistics). It is a noise estimate only while bursts are shorter than M windows;
longer occupancy raises it. Peak search runs on the CPU in weights, not on the host.
"""
from .codegen import Asm, Layout, word
from .kernels import Built, Kernel, put_pairs

U32 = 0xFFFFFFFF


def gt_u64(asm: Asm, out: str, ahi: str, alo: str, bhi: str, blo: str):
    """out = (a > b) for unsigned 64-bit a, b; branch-free. Clobbers t4, t5."""
    asm(f"sltu {out}, {bhi}, {ahi}",          # hi_a > hi_b
        f"xor t4, {ahi}, {bhi}", "seqz t4, t4",  # hi equal
        f"sltu t5, {blo}, {alo}", "and t4, t4, t5", f"or {out}, {out}, t4")


def min_u32(asm: Asm, dst: str, a: str, b: str):
    """dst = min(a, b) unsigned, branch-free. Clobbers t4, t5."""
    asm(f"sltu t4, {a}, {b}", "neg t4, t4", f"xor t5, {a}, {b}", "and t5, t5, t4", f"xor {dst}, {b}, t5")


class EnergyDetector(Kernel):
    """Per window of W samples (ending at e with e % W == W-1): mean power P = sum|x|^2 >> log2 W,
    noise N = min of the last M window means, detect = P*256 > alpha_q8*N (exact 64-bit),
    start = detect and not previous, end = previous and not detect. State: M means + previous."""
    name = "energy_detector"

    def __init__(self, window=64, memory=8, alpha_q8=1024):
        if window & (window - 1) or not 4 <= window <= 256 or not 1 <= memory <= 16:
            raise ValueError("window power of two 4..256, memory 1..16")
        self.W, self.M, self.alpha = window, memory, int(alpha_q8)
        self.hop, self.state_words = window, memory + 1

    def describe(self):
        return {"kernel": self.name, "window": self.W, "memory_windows": self.M, "alpha_q8": self.alpha}

    def initial_state(self):
        return [U32] * self.M + [0]

    def build(self, block):
        L = Layout()
        windows = block // self.W
        L.alloc("in", 2 * block)
        L.alloc("state", self.state_words)
        L.alloc("alpha", 1)
        L.alloc("out", 5 * windows)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        a.li("a1", L["state"])
        a.li("t0", L["alpha"])
        a("lw s11, 0(t0)", f"lw s10, {4 * self.M}(a1)")  # alpha, previous detect
        close_w = a.loop("a3", windows)
        a("li s0, 0", "li s1, 0")  # 64-bit power sum
        close_s = a.loop("a4", self.W)
        a("lw t0, 0(a0)", "lw t1, 4(a0)", "mul t2, t0, t0", "mul t3, t1, t1", "add t2, t2, t3",
          "add s0, s0, t2", "sltu t4, s0, t2", "add s1, s1, t4", "addi a0, a0, 8")
        close_s()
        shift = self.W.bit_length() - 1
        a(f"srli s0, s0, {shift}", f"slli t4, s1, {32 - shift}", "or s0, s0, t4")  # mean power (u32)
        # shift the minimum-statistics memory and take the minimum
        for i in range(self.M - 1):
            a(f"lw t0, {4 * (i + 1)}(a1)", f"sw t0, {4 * i}(a1)")
        a(f"sw s0, {4 * (self.M - 1)}(a1)", "mv s2, s0")
        for i in range(self.M - 1):
            a(f"lw t0, {4 * i}(a1)")
            min_u32(a, "s2", "s2", "t0")
        # detect: P*256 > alpha*N, both exact unsigned 64-bit
        a("slli s3, s0, 8", "srli s4, s0, 24",            # lo, hi of P*256
          "mul s5, s11, s2", "mulhu s6, s11, s2")          # lo, hi of alpha*N
        gt_u64(a, "s7", "s4", "s3", "s6", "s5")
        a("xori t0, s10, 1", "and s8, s7, t0",            # start
          "xori t0, s7, 1", "and s9, s10, t0",            # end
          "mv s10, s7",
          "sw s0, 0(a2)", "sw s2, 4(a2)", "sw s7, 8(a2)", "sw s8, 12(a2)", "sw s9, 16(a2)", "addi a2, a2, 20")
        close_w()
        a(f"sw s10, {4 * self.M}(a1)")
        a.readback(L["out"], 5 * windows)
        a.readback(L["state"], self.state_words)
        a.halt()
        return Built(a.source(), L, block)

    def init_words(self, built, context, start):
        w = {built.layout["alpha"]: word(self.alpha)}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o, out = built.layout["out"] >> 2, [None] * built.block
        for k in range(built.block // self.W):
            out[k * self.W + self.W - 1] = tuple(ram[o + 5 * k + j] for j in range(5))
        return out

    def reference(self, samples):
        memory, previous, out = [U32] * self.M, 0, [None] * len(samples)
        for e in range(self.W - 1, len(samples), self.W):
            mean = sum(re * re + im * im for re, im in samples[e - self.W + 1:e + 1]) >> (self.W.bit_length() - 1)
            memory = memory[1:] + [mean]
            noise = min(memory)
            detect = int(mean * 256 > self.alpha * noise)
            out[e] = (mean, noise, detect, int(detect and not previous), int(previous and not detect))
            previous = detect
        return out

    def chunk_tail(self, samples):
        return len(samples) - len(samples) % self.W  # incomplete final windows emit nothing


class PeakTracker(Kernel):
    """Running argmax of |x|^2 (u32, exact) over the stream; first index wins ties.
    Each position emits the argmax index so far. State: best power, best index, position."""
    name = "peak_tracker"
    state_words = 3

    def build(self, block):
        L = Layout()
        L.alloc("in", 2 * block)
        L.alloc("state", 3)
        L.alloc("out", block)
        a = Asm()
        a.li("a0", L["in"])
        a.li("a2", L["out"])
        a.li("a1", L["state"])
        a("lw s0, 0(a1)", "lw s1, 4(a1)", "lw s2, 8(a1)")  # best, index, position
        close = a.loop("a3", block)
        a("lw t0, 0(a0)", "lw t1, 4(a0)", "mul t2, t0, t0", "mul t3, t1, t1", "add t2, t2, t3",
          "sltu t3, s0, t2", "neg t3, t3",                  # mask = best < p
          "xor t4, s0, t2", "and t4, t4, t3", "xor s0, s0, t4",
          "xor t4, s1, s2", "and t4, t4, t3", "xor s1, s1, t4",
          "sw s1, 0(a2)", "addi a2, a2, 4", "addi a0, a0, 8", "addi s2, s2, 1")
        close()
        a("sw s0, 0(a1)", "sw s1, 4(a1)", "sw s2, 8(a1)")
        a.readback(L["out"], block)
        a.readback(L["state"], 3)
        a.halt()
        return Built(a.source(), L, block)

    def initial_state(self):
        return [0, 0, 0]

    def init_words(self, built, context, start):
        w = {}
        put_pairs(w, built.layout["in"], context)
        return w

    def decode(self, built, ram, start):
        o = built.layout["out"] >> 2
        return [ram[o + i] for i in range(built.block)]

    def reference(self, samples):
        best, index, out = 0, 0, []
        for n, (re, im) in enumerate(samples):
            p = re * re + im * im
            if p > best:
                best, index = p, n
            out.append(index)
        return out
