# v1.5 ablations (mandate Phase F, A1–A8)

These are controlled comparisons, compiled from measured runs; every number comes from a neural (token-verified) run unless marked otherwise. Evidence is linked per row. Where a controlled comparison could not be built, the entry is marked **N/A** or **blocked**, as the mandate requires.

## A1: Precision

The same kernels and inputs at different number representations.

**Real-IQ gate** (14 captures; criteria unchanged from v1.4)

| Representation | Passing captures | Evidence |
|---|---|---|
| v1.4, 4-bit narrowing | 0/14 | `v14` reports; [../research/NUMERICAL_FAILURE_ANALYSIS.md](../research/NUMERICAL_FAILURE_ANALYSIS.md) |
| Q15, inherited gain rule | 8/14 | [real-iq-q15.json](real-iq-q15.json) |
| Q2.13, fixed headroom (Amendment 1) | 13/14 | [real-iq-q13-headroom.json](real-iq-q13-headroom.json) |
| Q15 + block-floating-point front end in weights (Amendment 2) | **14/14** | [real-iq-bfp.json](real-iq-bfp.json) |

**FFT64 power:** arithmetic SQNR over −60…0 dBFS, and cost.

| Representation | SQNR | Tokens/sample | Evidence |
|---|---|---|---|
| Fixed per-stage scaling | 7.5–66.5 dB | 166 | [ERROR_BUDGET.md](ERROR_BUDGET.md), [fft-opt-neural.json](fft-opt-neural.json) |
| CPU block floating point | 74.6–76.3 dB | 230 | same |
| VBFLY block floating point | 69.5–71.5 dB | 127–133 | [amendment6.json](amendment6.json) |

**Other kernels** (arithmetic SQNR at −12 dBFS)

| Kernel | Narrower variant | Wider / more precise variant | Evidence |
|---|---|---|---|
| NCO | 1,024-entry table: 49.0 dB | + Taylor interpolation: 85.8 dB (53 → 72 tokens/sample) | [ERROR_BUDGET.md](ERROR_BUDGET.md) |
| IIR, +12 dB gain | int16 saturating: 35.1 dB | int32 `wide`: 82.3 dB | [amendment6.json](amendment6.json) |
| AM envelope (CORDIC) | 10 iterations: 121.6 dB, 185 tokens/sample | 16 iterations: 155 dB, 256 tokens/sample | [am-trim-neural.json](am-trim-neural.json) |
| Energy | VX int8: 35.2 dB | Scalar exact | [ERROR_BUDGET.md](ERROR_BUDGET.md) |
| FIR16 | VFIR per-tap rounding: 69.2 dB | Exact accumulation, one rounding: 81.3 dB | [amendment6.json](amendment6.json) |

## A2: State residency

**N/A on v1.5.** Program state lives in QwenPU's in-weights machine state (registers and RAM in the Gated DeltaNet states). There is no extraction and reinjection variant to compare.

The closest controlled measurement is stateful kernels, whose verified state words are copied between chunks by the host (runner `run_many`) against stateless kernels. That difference is a scheduling property, reported under A4.

## A3: Memory allocation

**N/A on v1.5.** RAM layouts are static per kernel (`codegen.Layout`); no lifetime-reuse allocator exists to ablate.

## A4: Scheduling (scalar vs vector; per-op vs batch packing)

**Scalar RV32 vs the VX vector unit**

| Workload | Scalar | VX | Ratio | Evidence |
|---|---|---|---|---|
| Spectral MLP | 416 tokens/sample, 60.2 ms | 125 tokens/sample, 18.0 ms | 3.3× | [vx-neural.json](vx-neural.json) |
| Energy W64 | 16.4 tokens/sample | 15.2 tokens/sample (VX int8, lower precision) | 1.08× | same |
| FIR16 vs VFIR16, packed | — | — | **6.73×** (pre-declared timing study, 95% CI [6.727, 6.738], p = 0.0078) | [timing-study.json](timing-study.json) |
| FIR 4 / 8 taps vs VFIR | — | — | 2.6× / 3.5× | [vfir-neural.json](vfir-neural.json) |
| FFT64 block floating point, CPU vs VBFLY | 34–38 ms/sample | 19–21 ms/sample | 1.8× | [amendment6.json](amendment6.json) |
| Energy from data already in VM, scalar vs VPOW | — | — | 8.8× (~1× when loading from RAM) | [../research/VX_RF_INSTRUCTIONS.md](../research/VX_RF_INSTRUCTIONS.md) |

**Batch packing**

| Kernel | Batch 1 | Batch 32 | Gain | Evidence |
|---|---|---|---|---|
| FIR16 | 7.4k tokens/s | 33.3k tokens/s | 4.5× | [DSP_METRICS.md](DSP_METRICS.md) systems table |
| FFT64 | similar | similar | similar | same |

## A5: Instruction fusion (unfused vs fused, identical numerical semantics)

| Program pair | Effect | Evidence |
|---|---|---|
| FFT stages: generic butterfly loop vs W₀/−j stages without multiplies, twiddle-outer loops | 17–19% fewer tokens, bit-identical | [fft-opt-neural.json](fft-opt-neural.json) |
| AM envelope: CORDIC with angle accumulation vs magnitude-only | 450 → 256 tokens/sample, bit-identical | [am-trim-neural.json](am-trim-neural.json) |
| VFIR: unpacked vs packed I/O | 16 taps 5.9× → 6.7× vs scalar | [vfir-neural.json](vfir-neural.json) |
| Stage 2 firmware: separate passes vs merged pass 1 and pre-shifted DFT16 | 252k → 93k tokens per call | [STAGE2.md](STAGE2.md) |
| Production mode (no readback) vs proof mode | −1.4% tokens; outputs then read from the reference RAM, not the trace | ledger |

## A6: Runtime (same processor weights and program semantics)

**Chip-token throughput by runtime**

| Runtime | Tokens/s | Evidence |
|---|---|---|
| HF saved layers, eager, one token per forward (PublishedCheckpoint `check_program`) | 35.6 | [speculative-verify.json](speculative-verify.json) |
| HF saved layers, compiled chip view, one CUDA graph per token (RvChat fast path, 19 GB merge) | ~400–420 (93k chip tokens in 223–247 s) | `stage2/fullgpu-sequential-newfw-stage2-chat.json` |
| Fused circuit executor (`fusedexec.FreeRun3`, CUDA graph), batch 1 | 4,363 | [speculative-fir64.json](speculative-fir64.json) |
| Fused circuit executor, batch 32 | 33.3k | [DSP_METRICS.md](DSP_METRICS.md) |
| Chunked verification on HF saved layers, torch kernels, batched decode | 4.4–6.0k | `speculative-fir64-batched.json` |
| Chunked verification on HF saved layers, flash-linear-attention kernels, GPU compare | 11.5–13.0k | `speculative-fir64-fla-gpucmp.json` |

Caveats:
- The eager and compiled HF rows use different merged checkpoints (published VX checkpoint vs 19 GB plain-CPU merge). Both have 13-layer chip circuits of the same design.
- Chunked verification is an audit mode: the reference drafts the trace and the layers verify it.
- Kernel swap (torch vs flash-linear-attention): every circuit token stays exact. Qwen's own text layers change wording (float).

## A7: Architecture (widths, depths, instruction encodings)

**Instruction encodings** (controlled: same circuit plus one instruction, same programs except the instruction)

| Instruction | Circuit cost | Effect | Evidence |
|---|---|---|---|
| VFIR | 13 → 13 layers, 137 → 137 heads, FFN +1.4% | 6.73× on FIR16 | [../research/VX_RF_INSTRUCTIONS.md](../research/VX_RF_INSTRUCTIONS.md) |
| VPOW | no new layers or heads | 8.8× when data is in VM | same |
| VBFLY | 13 layers, 137 heads, FFN +1.1% (+1.4% with track) | 1.8× on FFT64 at 70 dB | [amendment6.json](amendment6.json) |
| VBFLY with the tracker compare taken directly on the MAC output | **14 layers** | Rejected | [../research/RESEARCH_LOG.md](../research/RESEARCH_LOG.md) |

The rejected variant is a depth result: where a comparison sits in the datapath decides the layer count.

**Model width and depth: blocked.**
- Only Qwen3.5-4B has the head layout QwenPU's CPU circuit needs (2 value heads per key head).
- 0.8B and 2B fail to merge (`merge-qwen2b-rv1`, failed in the ledger).
- A different circuit (qwenGPU) targets 2B, but it is not the same processor, so the comparison would not be controlled.

## A8: Conventional computing

Same-contract integer baselines, resident inputs, best of 5.

| Kernel | Baseline | ns/sample | Neural slowdown |
|---|---|---:|---|
| lowpass16 FIR | NumPy int64 (CPU) | 47.8 | 4.5×10^5 (≈2×10^5 at full batch) |
| lowpass16 FIR | PyTorch int64 (CUDA) | 17.2 | 1.2×10^6 (≈6×10^5 at full batch) |
| FFT64 | cuFFT / NumPy float (not the same contract) | — | 3.7×10^6 to 1.5×10^7 |

Evidence: [DSP_METRICS.md](DSP_METRICS.md) conventional baselines. No real-time claim is made.
