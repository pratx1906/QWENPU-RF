# Exact, Verifiable RF Signal Processing Inside a Language Model's Weights

*Working draft (started 2026-11-04; covers Amendments 1–14; Amendment 15 is in the repository README, not yet folded in). Every number comes from a run recorded in `v15/results.md` and `v15/research/RESEARCH_LOG.md`; pre-declared gates are in `v15/research/PROTOCOL.md`.*

## Abstract

We show that a language model can carry out exact, independently verified radio-frequency (RF) signal processing and inference on real IQ recordings within its own weights. We use it as a tool and reason over the exact results.

**Substrate.** We build on QwenPU, which compiles an RV32 CPU and a vector unit into the weights of Qwen3.5-4B. On that substrate we implement 29 RF kernels as RV32/VX programs under a declared 16-bit fixed-point contract:
- filtering, FFT, Welch, CFAR, CORDIC demodulation;
- frequency-offset estimation, energy detection;
- a learned classifier.

**Verification.** Every token the network generates is checked against an RV32 reference machine, and every output against an independent integer reference.

**Accuracy.** A 14-capture real-IQ screen that the previous 4-bit design failed entirely (0/14) now passes 14/14 with a block-floating-point front end that runs in the weights. On a pre-declared expansion to 133 recordings from five independent datasets, the screen passes 132/133 (99.2%); a full 8-stage pipeline and a 10-kernel battery are token-exact on all 133. A pre-declared error budget puts the main kernels at 69–88 dB arithmetic SQNR. A block-floating-point FFT reaches 75 dB at every input level from −60 to 0 dBFS.

**New instructions.** We add three RF instructions to the compiled vector unit without new layers or attention heads:
- **VFIR** (complex FIR tap block): 6.73× over scalar code in a pre-declared timing study;
- **VPOW** (power sum);
- **VBFLY** (FFT butterfly with block-floating-point tracking): 1.8× faster at 70 dB.

**Integration.** Qwen calls the in-weight processor through its tool interface, entirely on one 24 GB GPU, with 0 reference-lock errors. On a pre-declared 96-question bank graded by a deterministic answer checker, 99.0% of the numbers in Qwen's answers trace back to the verified report. The checker flags the rest, all of them Qwen's own unit conversions. On 48 pre-declared held-out questions, free wording keeps every circuit token exact and every asked value quoted, but Qwen at first declined only 4 of 8 questions the report cannot answer. Having the in-weights firmware state what it does not measure raises this to 16/16 on new questions, while occasional misreadings of exact numbers remain: exact computation does not make the model's conclusions exact. Verifying the processor's tokens in parallel chunks against the reference lock is token-identical to sequential execution and cuts the time per question from 255–288 s to 47–54 s.

**Limits.** The system is about 10^5–10^6× slower than conventional DSP, and we make no real-time claim. Its value is auditable exactness in the model's own forward pass.

## 1. Introduction

Language models increasingly reason about the physical world through sensor data, including radio. Current systems put the signal processing outside the model: a conventional DSP front end, or an external tool whose output the model must trust. A separate line of work compiles general-purpose computers into transformer weights (Qwenputer, QwenPU, qwenGPU, qwenTPU, Loom, Percepta's transformer-vm). Those systems run generic programs and kernels, but none performs signal processing on radio data.

We ask whether a language model's own weights can perform exact, auditable RF signal processing on real recordings, and whether the model can then use those results. "Exact" has a precise meaning here (§3.1): bit-exact to a declared integer contract and token-for-token equal to a reference machine. It does not mean equal to floating point; we report the float error separately as an error budget.

**Contributions**
1. **An RF DSP and inference library for an in-weight computer.** 29 kernels (25 scalar, 4 vector) with independent integer references, plus a verification runner that checks every generated token (§3).
2. **Precision engineering for 16-bit in-weight arithmetic.** A failure analysis of the previous 4-bit design; a block-floating-point front end computed in the weights (0/14 → 14/14 on real IQ); a pre-declared error budget; and block-floating-point FFT and interpolated NCO kernels (§4.1–4.2).
3. **Three new RF instructions compiled into the vector unit** (VFIR, VPOW, VBFLY), with no new layers or heads, verified from an independent oracle through the compiled circuit and fused executor to saved checkpoint weights (§3.4, §4.3).
4. **Qwen-in-the-loop RF analysis.** Tool calls into the in-weight processor on one GPU; chunked verification that is token-identical to sequential execution; and a cheaper level-only tool the model selects when appropriate (§3.5, §4.5).
5. **An honest evaluation.** Pre-declared gates (some failed), ablations, conventional baselines and a public research log (§4–§6).

## 2. Background: a computer in the weights

QwenPU compiles an RV32IM+Zba CPU and a streaming vector unit (VX) into Qwen3.5-4B. Gated DeltaNet recurrent states serve as registers and RAM. Each executed instruction emits a short sequence of "chip tokens" (opcode families and 16-bit payloads), and the model's next-token prediction is the machine's next state transition.

The vector unit adds 16 vector registers of 64 elements, int8 activation memory, and fused operations (matrix-vector, elementwise, RoPE), all streaming results as tokens. A bridge captures `<tool_call>` text into the machine's memory, so firmware can parse the call and print a `<tool_response>` that the model then reads.

Two execution paths exist for the same compiled circuit:
- **Saved HF layers:** the merged checkpoint's own transformer layers, one token per forward pass.
- **Fused executor:** chipformer's CUDA-graph implementation of the circuit, about 100× faster than eager HF layers and about 11× faster than compiled HF layers (§7).

## 3. System

### 3.1 Numeric contract and verification

All signal arithmetic is RV32 integer code under one contract:
- Q15 samples and coefficients;
- exact 32- or 64-bit accumulation, with 64-bit chosen by a range proof;
- round-half-up shifts `(x + 2^(s−1)) >> s`;
- explicit saturation;
- branch-uniform code where batching needs it.

**What counts as a result.** A result counts only if both hold:
1. every generated token equals the RV32 reference machine's trace;
2. every output equals an independent integer reference, written separately from the program.

In proof mode, every output word is loaded back before halt, so the trace itself carries every output value. The host only writes inputs, copies verified words between stages, and decodes; all signal arithmetic runs in the weights.

### 3.2 Kernels

| Group | Kernels |
|---|---|
| Filtering | Complex FIR (4–64 taps, 32/64-bit accumulation), decimating FIR, IIR biquad (int16 saturating or int32 wide), DC blocker |
| Frequency | NCO mixer (table or Taylor-interpolated) |
| Spectral | Radix-2 FFT power or complex bins (fixed or block floating point), Welch PSD, single-bin DFT, CA-CFAR |
| Nonlinear | Branch-free CORDIC (magnitude/phase, AM envelope, FM discriminator), lag-L CFO estimator |
| Detection | Energy detector with minimum-statistics noise floor, peak tracker |
| Inference | Spectral MLP (synthetic task); real-data technology classifier (17 log-spectral features, integer MLP) |
| Vector unit | Energy, MLP, VFIR filter, VBFLY FFT64 |

Stateful kernels pass verified state words between chunks. A batched runner executes many machines in lock-step on one CUDA graph.

### 3.3 Block floating point in the weights

v1.4's 4-bit design failed every real capture. Our analysis attributes this to intermediate narrowing and wrap, not input quantization. At 16 bits the remaining failures were input clipping from a fixed gain rule.

We added two CPU kernels: BlockMax (a running maximum |component|) and AutoScale (the shift computed on the CPU, then a saturating round). Together they normalize each stream so its peak uses the full Q15 range.

The same idea inside the FFT normalizes the windowed frame and scales a stage by 1/2 only when its input needs it. This makes FFT accuracy independent of input level.

### 3.4 New RF instructions

All three use custom-2 opcode space (funct7 = 1) and reuse the vector unit's RoPE multiply-accumulate chain. A "NEW" flag in a configuration slot gates them off the existing datapaths.

**VFIR** (funct3 = 0): one input sample through a transposed-form complex FIR block of up to 16 taps.
- Taps are in a VM register; state ping-pongs between two registers.
- Each tap product is rounded on its own.

**VPOW** (funct3 = 1): Σ(ar² + ai²) over up to 32 VM elements into a register. The running sum comes back through the lag head.

**VBFLY** (funct3 = 2): up to 8 radix-2 butterflies of a constant-geometry FFT stage.
- Inputs: `a` from bank A, `b` from bank B, and the twiddle from a 32-pair table read at `i & (2^s − 1)`.
- The table stores `W^bitrev(r)`. We verified numerically that natural-order input with this mask gives the DFT in bit-reversed order, so one table serves every stage and frame.
- Optional tracking streams each output a second time from the token history, so the vector unit's max tracker records the stage's largest component. The CPU reads its bit length and chooses the next stage's scaling (block floating point).
- A first design compared the output directly; that added a 14th layer and was rejected.

**Circuit cost** (all three instructions): 13 layers and 137 heads unchanged; FFN units +1.4%; dims 6,675 → 6,984.

**Verification ladder** for each instruction:
1. an independent oracle;
2. the compiled circuit, teacher-forced, every token, on random programs that mix in the existing operations;
3. the fused executor, free-running;
4. a merged checkpoint's saved layers.

All three pass step 4 on a merged checkpoint (Qwen3.5-4B + VX + VFIR + VPOW + VBFLY): 5/5 programs token-exact, including a real-IQ FFT64 frame through VBFLY.

### 3.5 Qwen in the loop

`rf_analyze(capture)` is RV32 firmware.
- It parks on an exit ecall.
- On a tool call it parses the capture index from the captured text and analyses 1,024 stored samples: power, peak, 64-sample windows, an averaged DFT16 and the lag-1 CFO via CORDIC.
- It prints a report into `<tool_response>`.

`rf_levels(capture)` is the same firmware with a mode read from the call text. It reports level statistics only, at about one third of the cost.

**Fully on one GPU.** We merged the plain RV32 CPU into Qwen3.5-4B (19 GB) so the whole model runs on one 24 GB GPU. Chip tokens go through the model's own saved layers, and two independent checks run throughout:
- a reference lock compares every chip token with the RV32 reference;
- a head check compares the model's own tied LM-head prediction with the circuit's output code at every circuit-owned position.

**Chunked verification.** During a tool call the reference lock already knows the computer's upcoming tokens. We feed runs of them through the chip layers 1,024 at a time and require every position's prediction to equal the next drafted token. By induction this equals the sequential run, and it is checked empirically below. This is an audit mode, stated plainly: the reference drafts, the weights verify.

## 4. Evaluation

**Hardware:** one RTX 5090 Laptop GPU (24 GB), 63 GB RAM. All neural results are token-verified as in §3.1.

### 4.1 Accuracy on real IQ

**Screen:** 14 real captures (IDLab WiFi/LTE/DVB-T, NTU and EPFL LoRa, Tampere drone controllers). Criteria are unchanged from v1.4: FIR NMSE ≤ 0.01, spectral NMSE ≤ 0.01, matched-filter peak displacement ≤ 1.

| Configuration | Captures passing |
|---|---|
| v1.4, 4-bit | 0/14 |
| Q15, inherited gain | 8/14 |
| Q2.13 headroom | 13/14 (one peak near-tie) |
| **BFP front end in weights** | **14/14** (57,344 neural samples) |

Arithmetic NMSE is about 1e-9. A full 8-stage pipeline on all 14 captures (31.1M tokens) is exact on every token.

**Expanded screen (pre-declared Amendment 9).** The selection is stratified and seeded (seed 2026), drawn before any run: 133 recordings from every real-IQ dataset available.

| Dataset | Pass |
|---|---|
| IDLab WiFi/LTE/DVB-T | 53/53 |
| EPFL LoRaIQ | 32/32 |
| NTU RFSR over-the-air LoRa | 27/27 |
| Tampere drone RF | 18/19 |
| CAMRAS/Dwingeloo radio astronomy | 2/2 |
| **Total** | **132/133 (99.2%, Wilson 95% CI 95.9–99.9%)** |

- FIR NMSE median 7.5e-9 (max 3.7e-8); spectral NMSE median 4.9e-9.
- The single failure is a DJI 5G recording whose float matched-filter output has two maxima 2 samples apart that differ by 0.024%; 16-bit rounding flips the argmax. The arithmetic is correct, and the failure stands under the pre-declared criterion.

**Depth on the same 133 (Amendment 10).**
- The full 8-stage pipeline passes on all 133 streams: 288.6M tokens, every stage exact, 38.9 ms/sample.
- A 10-kernel battery (FFT64 block, Welch, CA-CFAR, CORDIC, FM, CFO, interpolated NCO, wide IIR, VFIR guard, VBFLY FFT) is token-exact on all 133. FFT64 block is ≥ 72.7 dB on every recording (pre-declared ≥ 65).
- **Regression after all changes (Amendment 13).** The 40-case neural fuzz (random kernels, random parameters, awkward stream lengths of 1, below the history, and non-multiples of the block) re-run on the final code is 40/40 token-exact (10.7M tokens). The only differences from the first run are 12.5% fewer tokens, all in the kernels shortened by bit-identical rewrites.

**Precision tails.** The battery's lowest IIR (55.9 dB) and VFIR (65.4 dB) values are not clipping:
- The rounding-error power is constant (≈0.95 and 0.35 LSB²), and SQNR tracks the filter's output power (corr 0.99).
- The worst recordings are wideband 200 MS/s drone captures that the test lowpass attenuates by 12–17 dB.
- Keeping 4 extra output bits in the IIR (pre-declared P12.5) raises its minimum to 80.0 dB (median 102.6) on all 133 recordings, exactly in the weights.
- An adaptive VFIR output scale (P12.6) reaches only 68.3 dB, below 70. The instruction contract rounds every tap product on its own, so output scaling cannot remove that floor. We declared this expected failure before the run.

### 4.2 Error budget (pre-declared)

Arithmetic SQNR compares each kernel with float64 on the same quantized input, with user parameters held at their integer values and internal constants (twiddles, sine table, CORDIC angles) exact.

| Kernel (−12 dBFS) | SQNR | Tokens/sample |
|---|---|---|
| lowpass16 FIR | 79.4 dB (design 77.3) | 319 |
| FIR64, 64-bit accumulation | 87.3 dB | 2,371 |
| Decimating FIR | 79.6 dB | 44 |
| IIR (+12 dB gain), int16 → int32 wide | 35.1 → **82.3 dB** | 172 → 130 |
| DC blocker | 69.2 dB | 84 |
| NCO, table → interpolated | 49.0 → **85.8 dB** | 53 → 72 |
| CORDIC magnitude, 16 → 10 iterations | 155 → 121.6 dB | 256 → 185 |
| FFT64 power, fixed → block FP | 56.1 → **74.6 dB** | 166 → 230 |
| VFIR16 (per-tap rounding) | 69.2 dB (**missed the 70 dB target**) | 110 |
| VFIR16 with the guard bit (Amendment 8) | **77.9 dB** (≥ 75 target; exact FIR 81.3) | 110 |
| Energy: exact scalar / VX int8 | exact / 35.2 dB | 16 / 15 |

**Level independence.** Across −60…0 dBFS, fixed-scaling FFT64 ranges from 7.5 to 66.5 dB; block floating point gives 74.6–76.3 dB on the CPU and 69.5–71.5 dB with VBFLY.

**Figures of merit:**
- FIR: 78.5 dB SQNR (12.7 ENOB).
- Energy detector: ROC matches chi-square theory.
- CA-CFAR: Pd ≥ 0.93 from −6 dB.
- Matched filter: 12.4 dB processing gain (theory 12.0).
- FM demodulation: 35–67 dB in-band SINAD.

### 4.3 New instructions

| Comparison | Result |
|---|---|
| VFIR16 vs scalar FIR16, pre-declared study (8 processes, A-B-B-A) | 6.73× geometric mean, 95% CI [6.727, 6.738], exact sign-flip p = 0.0078 |
| VFIR 4 / 8 taps vs scalar | 2.6× / 3.5× |
| VPOW vs scalar energy, data already in VM | 8.8× (~1× when loading from RAM) |
| VBFLY block FFT64 vs CPU block FFT64 (same circuit) | 19–21 vs 34–38 ms/sample (1.8×), 69.5–71.5 vs 74.6–75.2 dB |
| Vector MLP vs scalar MLP | 3.3× fewer tokens |

**Checkpoints.** VFIR is 5/5 token-exact on a merged checkpoint's saved layers, and a second merge with all three instructions is also 5/5 token-exact (VBFLY FFT64 on a real-IQ frame, VFIR, VBFLY and VPOW test programs).

### 4.4 Speed and verification

**Throughput by runtime** (chip tokens per second, same circuit)

| Runtime | Tokens/s |
|---|---|
| HF layers, eager, one token per forward | 35.6 |
| HF layers, compiled, one CUDA graph per token | ~400 |
| Fused executor, batch 1 (FIR64 verification workload) | 4,363 |
| Fused executor, batch 32 (FIR16) | 33,320 |
| Chunked verification on the saved layers (flash-linear-attention, GPU-side compare) | 13,000 |

The chunked verification run used a 107,023-token trace with 0 mismatches at every chunk size. Altering one drafted token is detected at that position.

Same-contract integer FIR on NumPy or CUDA is 2×10^5 to 6×10^5× faster than the neural path.

**Diagnosis worth reporting.** Chunked verification was initially *slower* than the executor. The cause was one host synchronization per position; batching the decode fixed it.

### 4.5 Qwen-in-the-loop

**Three demo questions, fully on one GPU**

| Mode | Seconds per question (computer part) | Checks |
|---|---|---|
| Sequential | 255–288 (223–247) | 0 lock errors, 0 head mismatches |
| Chunked | **47–54 (16)** | **Token-identical** to sequential, Qwen's answers included (93,449 / 100,884 / 100,984 tokens) |

On a synthetic capture with hidden ground truth, Qwen recovers the burst location, level and frequency from the exact report. Qwen's domain interpretation is not verified, and it can be wrong (e.g. it called a LoRa capture "narrowband").

**Choosing the cheaper tool.** With both tools offered, Qwen chose `rf_levels` for 2/2 level-only questions (26–29k chip tokens, 27–31 s) and `rf_analyze` for 1/1 frequency question.

**Real recordings, pre-declared (Amendment 7).** Six questions about six real captures (WiFi, LTE at 1 and 10 MS/s, DVB-T, LoRa, a drone controller), both tools offered. All 6 passed: 0 lock errors, 0 head mismatches, reports equal to an independent host model, and every requested number quoted exactly. Qwen chose `rf_levels` for the four level questions and `rf_analyze` for the spectral and CFO questions, at 29–42 s per question.

**Question bank with an automatic answer checker (pre-declared Amendment 11).**
- **Bank:** 96 questions (16 captures × 6 templates: levels, peak, activity, dominant bin, CFO, power share), both tools offered.
- **Checker:** deterministic. An asked field must appear verbatim, and every number in an answer must be supported by the verified report.

| Gate | Result | Target |
|---|---|---|
| Circuit tokens exact | 96/96 | 100% |
| Tool called | 96/96 | ≥ 95% |
| Asked fields quoted verbatim | 96.6% (all misses are trailing zeros, e.g. "-0.5" for "-0.5000") | ≥ 90% |
| Numbers supported | **99.0% (308/311)** | ≥ 95% |
| Right tool chosen | 96/96 | descriptive |

- **The 3 unsupported numbers** are Qwen's own conversions of the CFO to Hz. Two are correct and one is wrong by 1000× (−62.6 "Hz" for −62.6 kHz).
- **Qwen twice attributed the CFO to the spectral bin;** it is the lag-1 autocorrelation phase. The number checker cannot catch this.
- **Response (Amendment 12).** The firmware now computes `freq_hz` and `cfo_hz` in the weights and carries one-line method notes. The re-run gives: all four gates pass; fields quoted 100%, numbers supported 98.8% (330/334); **wrong Hz values fall from 1 to 0** (22/22 Hz values correct) and **CFO-method misattributions from 2 to 0**; the remaining unsupported numbers are correct kHz/MHz rescalings of values computed in the weights. Giving the model verified derived quantities removes the errors it made computing them itself.

**Held-out questions (pre-declared Amendment 13).** The bank uses six fixed wordings, so we wrote 48 new questions before any run: 32 answerable questions in free wording (2 per capture), 8 general-knowledge questions that need no tool, and 8 questions the report *cannot* answer (modulation, occupied bandwidth, transmitter count, LoRa spreading factor, BER, distance, angle of arrival, symbol rate). Same model, firmware and settings as the bank.

| Gate | Result | Target | Verdict |
|---|---|---|---|
| Circuit tokens exact (every question that ran the computer) | 40/40 | 100% | pass |
| Answerable: tool called and answered | 32/32 | ≥ 95% | |
| Answerable: asked fields quoted | 100% (44/44) | ≥ 90% | |
| Answerable: numbers supported | 89.3% (117/131) | ≥ 95% | **fail** |
| No-tool questions answered without a tool | 8/8 | ≥ 7/8 | pass |
| Unanswerable questions declined | **4/8** | ≥ 7/8 | **fail** |

- **Free wording does not hurt the verified part.** Qwen called a tool on every answerable question, picked `rf_levels` for the 16 level questions and `rf_analyze` for the 16 spectral ones, and quoted every asked field.
- **The numbers gate fails because the free wording invites arithmetic.** All 14 unsupported numbers are Qwen's own derivations, and on manual reading all 14 are arithmetically correct: differences ("9.9 dB above the noise floor"), percentages (3/16 = 18.75%), rescalings (75 MHz) and the sample rate written out in Hz. One carries a wrong unit ("1.1 dBFS" for a 1.1 dB difference). The checker cannot verify derived values, so they count against the gate.
- **The abstention gate exposes the real limit.** Qwen correctly said the report does not contain the spreading factor, BER, angle of arrival or the coding rate (4/8). On the other 4 it built a confident wrong answer on top of exact numbers: an occupied bandwidth of "0 MHz" from the dominant bin, "0 transmitters" from zero active windows, the 1 MS/s sample rate given as the symbol rate, and a drone distance of "about 100 m" from a path-loss calculation that treats dBFS as dBm.
- **Implication.** Verifying the computation guarantees the numbers, not the conclusions drawn from them. In this system the exact numbers are distinguishable from Qwen's inferences (every report is in the transcript, and the checker flags unsupported values), but nothing stops an unsupported inference. Abstention needs its own mechanism.

**Fix (pre-declared Amendment 14).** The firmware (v4) now ends every report with a line computed in the weights like the rest: `not_measured=modulation, coding rate, symbol rate, occupied bandwidth, number of transmitters, distance, direction of arrival, bit error rate, protocol or packet contents`; the system prompt adds one rule (answer only from the report, say what the tool does not measure, give dBFS differences in dB). Cost: +655 tokens per call (< 1%). We also tightened the checker (v3) so that correct simple derivations (differences, percentages, unit rescalings of report values) count as supported; a ×1.07 perturbation test bounds its leniency (1/12 perturbed values still pass on the development transcripts, 5/14 on the Amendment 14 run), and the fabricated distance calculation stays unsupported. On 36 *new* questions written before the run:

| Gate | Result | Target | Verdict |
|---|---|---|---|
| Circuit tokens exact | 34/34 | 100% | pass |
| Unanswerable declined, no invented numbers | **16/16 declined** (8/8 on quantities *not* on the list); one answer adds a hypothetical dBm example (5 numbers) | ≥ 14/16 and 0 | **fail** |
| Answerable: tool / fields / numbers | 16/16, 91.3%, 93.3% | 95% / 90% / 95% | **fail** |
| No-tool questions without a tool | 4/4 | ≥ 3/4 | pass |

- **Abstention is fixed and generalises:** declines rise from 4/8 to 16/16, including battery level, IQ imbalance, polarization and phase noise, which the not_measured line does not name. On the Amendment 13 set (descriptive, since it shaped the fix) declines rise from 4/8 to 8/8 and invented numbers from 45 to 0. One residual hedge: asked for a transmitter count, Qwen quotes the not_measured line yet still concludes "0 active transmitters detected".
- **Reading errors remain:** Qwen read `active_windows=0/16` as "16 active out of 64 (25%)" and dropped the sign of −22.5 MHz. The checker flags both; nothing prevents them.
- **Strict gates fail and we report them as failures.** The fix moves the system from "confidently wrong when asked the impossible" to "says what it cannot measure", but it does not make the model's reading of exact numbers exact.

### 4.6 Real-data classification

IDLab LTE/WiFi/DVB-T, 17 in-weight features, integer MLP.

| Setting | Recording accuracy |
|---|---|
| Fixed split (2 training locations) | 70.3% (chance 40%) |
| Six-location leave-one-out | 74.4% |
| 10 MS/s recordings, leave-one-recording-out | 84.6% (indicative, 26 recordings) |
| 10 MS/s + bandwidth features (81 features) | **96.2%** per recording, 78.8% per window (indicative); two-stage in weights, 26/26 neural-exact |

- The 10 MS/s model runs exactly in the weights.
- DVB-T vs LTE is the dominant error at 1 MS/s, where only a 1 MHz slice of 8–20 MHz signals is visible.
- Cyclic-prefix lag features did not help: this was pre-declared, evaluated once and reported as no improvement.

## 5. Ablations

Summary of `v15/reports/ABLATIONS.md`:

| Ablation | Finding |
|---|---|
| Precision (A1) | 4-bit → Q15 → BFP takes the real-IQ screen 0 → 8 → 14 of 14 |
| Scheduling (A4) | Vector over scalar: 3.3× (MLP), 6.7× (FIR), 1.8× (FFT); batch 32 over batch 1: 4.5× (FIR16, 7,436 → 33,320 tok/s) |
| Fusion (A5), bit-identical | 17–19% fewer FFT tokens; 43% for the AM envelope; firmware 252k → 93k tokens per call |
| Runtime (A6) | Eager HF → CUDA graph → fused executor → chunked verification: 35.6 → ~400 → 4.4k → 13k tokens/s |
| Architecture (A7) | Instruction encodings measured, and the 14-layer VBFLY variant rejected; model width and depth blocked (only the 4B hosts QwenPU's circuit) |
| State residency (A2), memory allocation (A3) | Not applicable to this substrate |

## 6. Related work

**Computers compiled into transformer weights** (Qwenputer, QwenPU, qwenGPU, qwenTPU, Percepta transformer-vm, Loom, Tracr/RASP and ALTA, looped transformers) establish the substrate. None performs DSP or radio processing; qwenGPU's kernels are generic (dot, matmul, saxpy, reduce, scan, histogram).

**Fourier and butterfly layers** (FNet, Butterfly-Net, learned DFTs) approximate transforms as layers rather than run an exact programmable DSP.

**RF and LLM agents** (RF-GPT, RF-Analyzer, SignalLLM, agentic RF intelligence) reason about RF with signal processing outside the model.

**Hardware DSP** (Versal, Hexagon, AIR-T) is the performance reference we do not compete with.

See `v15/research/NOVELTY_SEARCH.md` for dates and the claim's qualifiers.

## 7. Limitations

- **Speed:** 10^5–10^6× slower than conventional DSP. Not a real-time system.
- **Exactness is relative to a fixed-point contract,** not to real-number DSP. The float error is quantified separately (§4.2). Two pre-declared VFIR precision targets failed: 69.2 vs 70 dB (later fixed by the guard bit, 77.9 dB), and 68.3 vs 70 dB on strongly attenuated wideband outputs (P12.6).
- **Large campaigns ran on the fused executor** (the same circuit). Its agreement with the saved layers is checked on representative programs, not on every campaign.
- **Chunked verification is an audit mode:** the reference machine computes the draft.
- **Qwen's interpretation of the exact report is not verified.** We measure it: on the template bank 99% of numbers in answers are traceable to the verified report, and the rest are flagged. On held-out questions the report cannot answer, Qwen first declined only 4/8 times and otherwise drew confident wrong conclusions from exact numbers; an in-weights `not_measured` line plus one prompt rule raised this to 16/16 on new questions, but Qwen still misreads exact numbers occasionally (a fraction, a sign) and these gates fail strictly (§4.5). Exact computation does not make the model's reasoning exact.
- **Data limits:**
  - the classifier is limited by data (2 training locations at 1 MS/s; 26 recordings at 10 MS/s);
  - the real-IQ evidence is 133 recordings from 5 datasets;
  - the Stage 2 evidence is 96 bank questions plus 84 held-out questions on 16 captures;
  - timing is from one laptop.
- **Host model:** only Qwen3.5-4B hosts QwenPU's circuit. The vector-unit checkpoint (27 GB) does not fit the 24 GB GPU together with Qwen's own layers, so Stage 2 uses the plain-CPU merge.

## 8. Conclusion

A language model's own weights can perform exact, verified RF signal processing on real recordings, and can host new RF instructions without growing its depth or head count. The model can call that processor and use its exact results. The design space is now about precision engineering (block floating point recovers what 16 bits lose) and verification-aware runtimes (chunked verification gives a 5× faster chat with identical tokens), not raw speed.

## Reproducibility

See `v15/REPRODUCE.md` (environments, commits, checksums, commands, expected outputs) and `v15/research/RESEARCH_LOG.md` (every run, including failures).
