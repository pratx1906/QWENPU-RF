"""Parallel streaming runner: one CPU-in-weights machine per chunk, batched on one CUDA graph.

Each chunk runs (1) on chipformer's RV32 reference machine, giving the expected trace, and
(2) on the free-running fused executor of the compiled circuit. Every generated token must
equal the reference token; the trace includes a load of every output (and state) word. Outputs
are then checked against the kernel's independent integer reference over the whole stream.

Stateless kernels: all chunks of all streams are independent rows. Stateful kernels: chunk j of
every stream forms one wave; the verified final state of chunk j seeds chunk j + 1.
"""
import time
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    outputs: list
    chunks: int
    tokens: int
    neural_seconds: float
    reference_seconds: float
    exact_tokens: bool
    exact_outputs: bool
    stats: dict = field(default_factory=dict)


class Runner:
    """Holds one compiled circuit and one free-running executor."""

    def __init__(self, vx: bool = False, batch: int = 8, proof: bool = True):
        from chipformer.rv import core
        self.core, self.vx, self.batch, self.proof = core, vx, batch, proof
        self.c = core.build(vx=vx)
        self.fr = None
        self.programs = {}

    def program(self, kernel, block):
        from chipformer.rv.asm import assemble
        # The cached entry keeps the kernel alive, so its id cannot be reused by another kernel.
        from . import codegen
        key = (id(kernel), block, self.proof)
        if key not in self.programs:
            codegen.PROOF = self.proof
            try:
                built = kernel.build(block)
            finally:
                codegen.PROOF = True
            self.programs[key] = (built, assemble(built.source), kernel)
        return self.programs[key][:2]

    def machine(self, prog, words):
        from chipformer.rv import ref
        m = ref.Machine.from_program(prog)
        for address, value in words.items():
            m.ram[address >> 2] = value
        return m

    def execute(self, prog, word_sets, neural=True, name="program"):
        """Run machines for each RAM image; returns (rams, token_count, ref_seconds, neural_seconds)."""
        import numpy as np
        import torch
        from chipformer.rv import spec as S
        rams, tokens, t_ref, t_neural = [], 0, 0.0, 0.0
        for first in range(0, len(word_sets), self.batch):
            group = word_sets[first:first + self.batch]
            traces = []
            t = time.perf_counter()
            for words in group:
                m = self.machine(prog, words)
                trace = m.run(10**8)
                if not m.halted or m.fault is not None or m.exit_code != 0:
                    raise RuntimeError(f"{name}: reference machine fault {m.fault}")
                traces.append(trace)
                rams.append(m.ram)
            t_ref += time.perf_counter() - t
            tokens += sum(len(tr) - 1 for tr in traces)
            if not neural:
                continue
            # Value records use 1..3 tokens by magnitude, so lengths differ by data; the batch
            # runs to the longest trace and each row is checked over its own complete trace.
            n = max(len(tr) for tr in traces) - 1
            if self.fr is None:
                from chipformer.rv import fusedexec
                self.fr = fusedexec.FreeRun3(self.c, self.batch, "cuda")
            for row in range(self.batch):
                self.core.snapshot(self.c, self.machine(prog, group[min(row, len(group) - 1)]))
                self.fr.load(row, S.tok("GO"))
            if getattr(self.fr, "graph", None) is None:
                self.fr.capture()
            torch.cuda.synchronize()
            t = time.perf_counter()
            got = np.asarray(self.fr.run(n))
            torch.cuda.synchronize()
            t_neural += time.perf_counter() - t
            for row, trace in enumerate(traces):
                m_ = len(trace) - 1
                if got[row][:m_].tolist() != trace[1:]:
                    bad = next(i for i in range(m_) if int(got[row][i]) != trace[i + 1])
                    raise AssertionError(f"{name}: neural token mismatch, row {first + row}, step {bad}")
        return rams, tokens, t_ref, t_neural

    @staticmethod
    def context(kernel, samples, start, block):
        H = kernel.history
        body = samples[start:start + block]
        ctx = [samples[i] if i >= 0 else (0, 0) for i in range(start - H, start)]
        return ctx + body + [(0, 0)] * (block - len(body)), len(body)

    def run_many(self, kernel, streams, block=256, neural=True, check_reference=True):
        if block % kernel.hop:
            raise ValueError("block must be a multiple of the kernel hop")
        built, prog = self.program(kernel, block)
        outputs = [[None] * len(s) for s in streams]
        jobs = [(k, start) for k, s in enumerate(streams) for start in range(0, len(s), block)]
        waves = [jobs] if not kernel.state_words else \
            [[(k, j * block) for k, s in enumerate(streams) if j * block < len(s)]
             for j in range(max((len(s) + block - 1) // block for s in streams))]
        state = [kernel.initial_state() for _ in streams]
        tokens = t_ref = t_neural = 0
        for wave in waves:
            word_sets, valid = [], []
            for k, start in wave:
                ctx, n = self.context(kernel, streams[k], start, block)
                words = kernel.init_words(built, ctx, start)
                if kernel.state_words:
                    kernel.put_state(built, words, state[k])
                word_sets.append(words)
                valid.append(n)
            rams, tok, a, b = self.execute(prog, word_sets, neural, kernel.name)
            tokens, t_ref, t_neural = tokens + tok, t_ref + a, t_neural + b
            for (k, start), ram, n in zip(wave, rams, valid):
                outputs[k][start:start + n] = kernel.decode(built, ram, start)[:n]
                if kernel.state_words:
                    state[k] = kernel.read_state(built, ram)
        exact = [kernel.reference(s) == o for s, o in zip(streams, outputs)] if check_reference else None
        total = sum(len(s) for s in streams)
        results = []
        for k, s in enumerate(streams):
            results.append(StreamResult(outputs[k], sum(1 for j in jobs if j[0] == k), tokens, t_neural, t_ref,
                                        neural, exact[k] if exact else None,
                                        {"block": block, "batch": self.batch, "streams": len(streams),
                                         "tokens_per_sample": tokens / max(total, 1), "vx_build": self.vx,
                                         "stateful": bool(kernel.state_words), "proof_mode": self.proof}))
        return results

    def run(self, kernel, samples, block=256, neural=True, check_reference=True):
        return self.run_many(kernel, [samples], block, neural, check_reference)[0]
