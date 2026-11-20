# v1.5 Stage 1 protocol (fixed before any v1.5 real-IQ run, 2026-11-03)

## Primary real-IQ fidelity gate

**Inputs**
- The 14 captures in `data/real-iq-selection.json`. Every file's SHA256 is verified before use.
- Preprocessing is the v1.4 rule:
  - calibrate on 512 samples;
  - gain = 0.75 / p95(|I|,|Q|);
  - evaluate the next 4,096 samples.
- Only the input format changes: Q15, with round half away from zero and clipping to [-32768, 32767].

**Pipelines** (identical to v1.4 in structure)
- FIR with h = [0.5, 0.25] (codes 16384, 8192).
- Matched filter with taps conj(h).
- FFT4 power with window [0, .75, .75, 0], hop 2, on frames ending at sample ≥ 3.

**Arithmetic.** Requantization uses the v1.5 contract (`v15/rfpu_v15/cpu/__init__.py`):
- round half up;
- exact 32/64-bit accumulation;
- exact 64-bit Q30 power.

**Criteria** (unchanged from v1.4)
- FIR NMSE ≤ 0.01.
- Spectral-power NMSE ≤ 0.01.
- Matched-filter peak displacement ≤ 1 sample.

All three are measured against NumPy float references, with the same formulas as `v14/scripts/real_iq.py`.

**Neural coverage.** Every one of the 4,096 samples per capture is executed on QwenPU's RV32 CPU compiled into weights, free-running on `fusedexec.FreeRun3`. A capture counts as passing only if both of these hold:
- every generated token equals the RV32 reference trace, which includes a load of every output word;
- the outputs equal the independent integer reference.

**Reporting.** The number of passing captures is reported as measured. Thresholds are not changed after viewing. If a capture fails, the cause is diagnosed and reported with its numbers.

## Performance measurements in Stage 1

These are descriptive (no significance claims). Measured per run:
- tokens per sample;
- neural seconds, with GPU synchronization, excluding reference-trace generation and executor construction;
- batch size and block size.

The confirmatory timing campaign belongs to Stage 3.

## Amendment 1: secondary headroom analysis (added 2026-11-03, after viewing primary per-capture results, before any secondary run)

Primary observation: every primary failure so far coincides with input clipping. The clipped component counts were 463, 109, 107 and 30, and each capture's input-quantization NMSE alone already equals its FIR NMSE. Every capture without clipping passes, with arithmetic NMSE of about 1e-9. The primary result stands as measured.

The secondary analysis is diagnostic and post hoc, and it is labelled as such. It keeps the same captures, gain rule, kernels, criteria and neural execution. Only the input fixed-point format changes, from Q15 (full scale ±1) to Q2.13 (codes = round(z·8192), full scale ±4).

Outputs are rescaled by the same factor:
- FIR and matched outputs: /8192.
- Power: /8192²·2^15·2^15/2^30, i.e. power codes/2^26.

This tests whether front-end headroom, rather than neural arithmetic, explains the failures. It is reported separately from the primary gate and never replaces it.

## Amendment 2: block-floating-point front end inside the weights (added 2026-11-04, before any run of it)

**Motivation.** The primary failures come from front-end clipping, and Amendment 1's fixed Q2.13 headroom was chosen after the fact. This amendment makes the scaling decision on the in-weights CPU.

**What stays fixed (unchanged from the primary gate)**
- The same 14 captures and the same 4,096-sample spans.
- The same gain rule, 0.75/p95.
- The same float references and criteria.
- The same FIR, matched-filter and FFT4 kernels.

**Pipeline**
1. The host writes each sample as an int32 Q24 code: round half away from zero, no clipping, since |z| < 128. This is a representation step only.
2. `BlockMax` (neural) computes M, the maximum |component| over the stream.
3. `AutoScale` (neural, with M copied in as a verified word) computes s = max(0, bitlen(M) − 15) on the CPU. It then emits x_q = sat16(round_half_up(x, s)) and the block exponent 24 − s.
4. FIR, matched filter and FFT4 run on x_q.
5. Output scale: FIR and matched outputs are divided by 2^(24−s); power is divided by 2^(2(24−s)).

**Reporting.** Reported separately from the primary gate (8/14) and from Amendment 1 (13/14), as the "BFP front end" configuration. Criteria are unchanged; failures are reported with diagnosis.

## Amendment 3: prespecified timing study, VFIR vs scalar FIR (added 2026-11-04, before any run)

**Arms** (same branch circuit, `core.build(vx=True)` from `rfpu-vfir`, fused executor, batch 16, proof mode):
- A = scalar RV32 complex FIR (`kernels.FIR`);
- B = `VXFir` packed;
- both with the same 16 taps (seed 4, |coefficient| < 6000).

**Input.** The first 4 real captures in `real-iq-selection.json`, 256 samples each (1,024 samples per arm run). Inputs and taps are fixed by seed and hashed in the record.

**Unit and design**
- Experimental unit: one fresh Python process.
- 8 processes, run sequentially on an otherwise idle GPU.
- Within each process: one warm-up run of each arm (excluded), then timed runs in A-B-B-A order.

**Endpoint.** Synchronized neural seconds per input sample, averaged within each process per arm. Tokens must equal the reference on every run; any inexact run invalidates the process.

**Analysis.**
- Per-process log ratio log(A/B).
- Geometric-mean ratio with a t interval (95%) across processes, plus an exact two-sided sign-flip test.
- Tokens per sample reported alongside.

**Scope.** One laptop and one session. This is not a multi-machine or multi-day replication.

## Amendment 4: error budget and precision targets (added 2026-11-04, before any implementation or run)

**Purpose.** Separate the error of every main kernel into (a) arithmetic error, which the in-weights CPU can reduce, and (b) design error (coefficient, table and filter-shape choices). Then reduce the two largest arithmetic errors found in `DSP_METRICS.md` (FFT64 power 56.1 dB, NCO 49.0 dB at −12 dBFS).

**Definitions.**
- Arithmetic SQNR: the kernel's neural output vs a float64 model that uses the *same quantized integer input and the same integer coefficients/tables/windows*. Only rounding, scaling and saturation inside the program count as error.
- Design error: the float64 model with quantized coefficients vs the ideal design (exact taps, exact sin/cos, exact window).
- Input: complex Gaussian, seed 1, 2,048 samples, Q15. Level −12 dBFS unless stated; FFT level sweep −60 to 0 dBFS in 6 dB steps.

**New kernels (options; existing kernels and their contracts stay unchanged).**
- `FFTPower(..., scaling="block")`: block-floating-point FFT. The windowed frame is normalized on the CPU (shift chosen from the OR of |x·w|), and each radix-2 stage scales by 1/2 only when the stage input has a component outside [−2^14, 2^14). A per-frame exponent word is output.
- `NCOMixer(..., interpolate=True)`: table index rounded to nearest, then first-order Taylor correction of sin/cos from the fractional phase, computed on the CPU.

**Targets (pass/fail; all neural, every token equal to the reference, outputs equal to the independent integer reference).**

| ID | Kernel | Target |
|---|---|---|
| P4.1 | FFT64 power, `scaling="block"` | Arithmetic SQNR ≥ 65 dB at every level −60…0 dBFS |
| P4.2 | FFT64 power, `scaling="block"` | Arithmetic SQNR ≥ 70 dB at −12 dBFS (baseline fixed scaling: reported alongside) |
| P4.3 | NCO, `interpolate=True` | Arithmetic SQNR ≥ 75 dB at −12 dBFS |
| P4.4 | All error-budget kernels | Neural-exact; each row reports arithmetic SQNR, design SQNR where defined, tokens/sample |

Token cost increases are reported, with no threshold. Failures are recorded as failures; targets are not changed after results.

**Amendment 4 clarification (added after CPU unit tests of the new kernels, before any neural or error-budget run).** "Tables" in the arithmetic-SQNR definition was ambiguous. The stricter reading applies: a kernel's *internal* constants (FFT twiddles, NCO sine table, CORDIC angle table, gain constants) are part of its arithmetic, so the float64 model uses exact trigonometry for them; only *user parameters* (filter taps, IIR coefficients, window codes) are held at their integer values, and their quantization is reported as design error. This is the reference the 49.0 dB (NCO) and 56.1 dB (FFT) baselines in `DSP_METRICS.md` were measured against. Targets unchanged.

## Amendment 5: token-reduction and verification-speed checks (added 2026-11-04, before these runs)

| ID | Item | Pass criterion |
|---|---|---|
| P5.1 | FFT stages rewritten (W_0 and −j stages without multiplies; one twiddle load per twiddle) | Neural-exact; outputs bit-identical to the unchanged contract (same independent references); tokens/sample reported vs the previous program |
| P5.2 | AM envelope with a magnitude-only CORDIC (no angle accumulation) and k = 10 iterations, k fixed by analysis (residual angle ≤ atan 2^-9, relative magnitude error ≤ 2e-6) before measurement | Neural-exact; arithmetic SQNR ≥ 90 dB at −12 dBFS (same definition as Amendment 4; gain constant for k iterations exact); tokens/sample reported vs 16 iterations |
| P5.3 | Chunked (speculative) verification on the published checkpoint's saved chip layers, FIR64 over 48 real samples (≥ 80k tokens) | 0 mismatches at every chunk size. Speed vs the fused executor (batch 1) reported; repeated with flash-linear-attention kernels (second environment), where exactness is the open question |

Exploratory items (no gate): none of these change an existing kernel's contract except P5.2, which is a new option.

## Amendment 6: precision-preserving VBFLY FFT and accuracy fixes (added 2026-11-04, before implementation or any run)

**A. VBFLY with block floating point.** VBFLY's outputs update the VX max tracker (vmx, as VEW's), so after each stage the CPU reads the bit length of the largest output component (VGET 3) and chooses the next stage's scaling: unscaled (sc = 0) when that bit length is ≤ 13, else sc = 1. This keeps every operand inside int16 (complex magnitude ≤ ~23,200) and is the int16 analogue of the CPU block FFT (Amendment 4). Input: window products normalized on the CPU to components below 2^13; frame exponent reported.

| ID | Item | Pass criterion |
|---|---|---|
| P6.1 | FFT64 power with VBFLY + block floating point (branch circuit, fused executor) | Neural-exact; arithmetic SQNR ≥ 68 dB at every level −60…0 dBFS (CPU block FFT: 74.6–76.3 dB, reported alongside); tokens/sample and ms/sample vs the CPU block FFT |

**B. Accuracy weak spots.**

| ID | Item | Pass criterion |
|---|---|---|
| P6.2 | IIR biquad `wide=True`: int32 state and output (no int16 saturation) | Neural-exact; arithmetic SQNR ≥ 65 dB at −12 and −6 dBFS (int16 version: 35.1 dB at −12) |
| P6.3 | VFIR (per-tap rounding) arithmetic SQNR, 16 taps, −12 dBFS | Measurement; target ≥ 70 dB. If met, VFIR's few-LSB difference from the exact FIR is reported as within budget |
| P6.4 | Real-data technology classifier | Exploratory: richer features under the same recording-level held-out protocol; an improvement is claimed only if held-out per-recording accuracy exceeds the recorded 70% |
| P6.5 | Energy | No new kernel: the exact scalar energy (16 tokens/sample, exact) is the precision path; the VX int8 energy (35 dB, 15 tokens/sample) is reported as a low-precision showcase only |

**Amendment 6, P6.4 method (added before any feature evaluation).** Diagnosis from the recorded confusion matrix: DVB-T windows go to LTE (398/640); WiFi 88%, LTE 95%. Candidate addition: OFDM cyclic-prefix lag correlations |R(L)| / R(0), R(L) = Σ x[n] conj(x[n−L]) over the 1,024-sample window, at L ∈ {64, 67, 128, 224, 448} (LTE useful symbol ≈ 66.7 µs; DVB-T 2K/8K 224/896 µs), computed in float on the host for screening. Selection uses only the calibration (training) recordings: leave-one-location-out cross-validation of the same MLP recipe, baseline (current 17 features) vs baseline + lag features. The held-out recordings are evaluated **once**, for the selected set only; if it does not beat the recorded 70.3% per-recording accuracy, the result is recorded as no improvement. An in-weights implementation follows only if the held-out result improves.

**Amendment 6, P6.4b: more training locations (added before the run).** The fixed split trains on 2 locations, and cross-location transfer is the observed limit. All 6 IDLab 1 MS/s locations are used in leave-one-location-out cross-validation (6 folds: train on 5, test on the held-out one; every recording tested exactly once). Metric: per-recording majority accuracy pooled over the folds, and per-window accuracy. Variants, all reported: {17 features, 17 + lag features} × {H = 8, H = 16}; primary = 17 + lags, H = 8 (declared now). This is a different evaluation protocol from the fixed manifest split and is reported alongside the recorded 70.3%, not in place of it. Host float screening; an in-weights version follows only if the primary variant beats the 17-feature H = 8 baseline under the same protocol.

**Amendment 6, P6.4c: 10 MS/s recordings (added before the run).** The 26 IDLab recordings at 10 MS/s (LTE 12, WiFi 9, DVB-T 5) show the signals' occupied bandwidth (DVB-T ~7.6 MHz, LTE 5–20 MHz, WiFi 20 MHz) within the band, which 1 MS/s cannot. Same 17 in-weights features (TechClassifier, 64 DFT16 frames) on 16 windows of 1,024 samples per recording, gain rule unchanged; same MLP recipe (H = 8). Evaluation: leave-one-recording-out (26 folds). Metric: per-recording majority accuracy and per-window accuracy, with the confusion matrix. Small sample (5 DVB-T recordings): reported as indicative, not as a general accuracy claim. An in-weights run follows if per-recording accuracy exceeds the 1 MS/s six-location result (74.4%).

## Amendment 7: Stage 2 on real recordings (added 2026-11-04, before the run)

Six questions about six real captures from the existing capture set: 1 IDLab WiFi 1 MS/s, 2 IDLab LTE 1 MS/s, 3 IDLab LTE 10 MS/s, 4 IDLab DVB-T 1 MS/s, 9 EPFL LoRa, 13 Tampere drone controller 5.8 GHz. Both tools (`rf_analyze`, `rf_levels`) are offered. Setup: fully on one GPU (19 GB merge), chunked verification C = 1024, main environment.

Pass criteria, per question:
1. 0 lock errors and 0 head mismatches.
2. Every report field equals the independent host model within the firmware tolerances (`report_check`).
3. Each numeric field the question asks for appears in Qwen's answer exactly as printed in the report.

Not graded: Qwen's domain interpretation (e.g. what the signal "is"), which is reported verbatim.

## Amendment 8: precision fix, classifier bandwidth feature, regression campaign, paper ablations (added 2026-11-04, before implementation or any run)

P6.3 stays recorded as failed (VFIR 69.2 dB vs 70). The items below are new and do not replace it.

| ID | Item | Pass criterion |
|---|---|---|
| P8.1 | VFIR guard-bit mode: taps pre-scaled by 2^g on the host (g = 2 for \|c\| < 6000), each tap product still rounded at 2^-15 of the scaled taps, a rounding constant 2^(g−1) placed in the chain pair, and rd = sat16(y >> g) (new 2-bit field g, a circuit change on branch `rfpu-vfir`) | Circuit token-exact (teacher-forced test plus regression); neural; arithmetic SQNR ≥ 75 dB at −12 dBFS, 16 taps (seed 4, \|c\| < 6000) |
| P8.2 | Classifier at 10 MS/s: add occupied-bandwidth features (64-bin FFT64 power shape, a CPU-in-weights kernel), leave-one-recording-out as in P6.4c | Exploratory: reported vs 84.6% under the same protocol; the claim only if higher |
| P8.3 | Stage 1 regression on today's code: real-IQ BFP gate, 14-capture pipeline, synthetic pipeline, neural fuzz | Identical pass outcomes to the recorded runs (14/14; PASS; PASS; 40/40 exact) |
| P8.4 | Stage 2 regression: robust suite with chunked verification | 0 lock errors, 0 head mismatches; recorded behaviour on the adversarial questions (out-of-range index, negative index, no-tool question) unchanged |
| P8.5 | Ablation: Stage 2 chunk size C ∈ {1, 64, 256, 1024, 4096} on one fixed question | Every C token-identical to C = 1 (sequential); seconds per question reported |
| P8.6 | Ablation: VBFLY FFT64 tracked block floating point vs untracked fixed scaling | Both neural-exact; SQNR over −60…0 dBFS and tokens/sample reported |

## Amendment 9: expanded real-IQ screen (added 2026-11-04, before selection was drawn or anything run)

**Purpose.** Test the Amendment 2 configuration (BFP front end in weights, then FIR, matched filter and FFT4 power, every sample neural) on far more real recordings than the 14-capture screen, across every real-IQ dataset on D:.

**Method.** Unchanged from Amendment 2: same kernels, same preprocessing (`real_iq_q15.inputs`: 512-sample gain prefix, 4,096 samples), same criteria (FIR NMSE ≤ 0.01, spectral NMSE ≤ 0.01, matched-peak displacement ≤ 1).

**Selection.** Stratified and deterministic (seed 2026, drawn by `v15/scripts/build_expanded_selection.py` before any run):

| Dataset | Selection | Count |
|---|---|---|
| IDLab | Up to 2 recordings per (technology × sample rate × location) | ~60 |
| EPFL LoRaIQ | 1 SigMF recording per (session × receiver), sampled | up to 32 |
| NTU RFSR over-the-air | 9 experiments × 3 receiver gains (lowest, middle, highest) | 27 |
| Tampere drone RF | Every file, one 4,608-sample segment at a seeded offset inside the file | 19 |
| CAMRAS / Dwingeloo radio astronomy | Both recordings | 2 |

Captures already in the 14-capture screen may appear; that subset is reported separately.

**Outputs**
- **Primary:** pass count and rate overall, with a Wilson 95% interval.
- **Secondary:** pass rate per dataset; the NMSE distributions (median, 95th percentile, maximum); every failure listed with its failing criterion.

**Thresholds** are fixed as above; no recording is removed after the run, except an unreadable file, which is reported as such. This is a single GPU job, about 6 hours.

## Amendment 10: deep DSP testing on the 133 expanded recordings (added 2026-11-04, before any run)

The selection is `v15/reports/real-iq-selection-expanded.json` (Amendment 9). Inputs are Q2.13, using `real_iq_q15.inputs` with the headroom rule as in the pipeline (Amendment 1). Every stage and kernel runs neurally (fused executor; branch circuit for VFIR and VBFLY).

**10a. Full 8-stage pipeline**
- Stages: DC blocker → lowpass16 → energy/noise/burst detector, FFT16 + CFAR, spectral MLP, CFO, peak tracker, FM demod.
- Input: 2,048 samples per recording, all 133 streams batched.
- **Pass:** every stage of every stream token-exact, with outputs equal to the independent references (as the 14-capture PASS).
- Per-stream summaries are reported descriptively: bursts, CFAR detections, classes, CFO, peak.

**10b. Kernel battery**
- Kernels:
  - FFT64 power, block floating point;
  - Welch PSD 64×4;
  - CA-CFAR 64;
  - CORDIC magnitude/phase;
  - FM discriminator;
  - CFO estimate (lag 1, window 64);
  - NCO, Taylor-interpolated;
  - IIR biquad, wide;
  - VFIR16 guard;
  - VBFLY FFT64, block floating point.
- Input: 256 samples per recording, all 133 recordings as batched streams.
- **Pass:** every kernel on every recording token-exact, with outputs equal to the independent references.
- Arithmetic SQNR (Amendment 4 definition) per recording, for kernels with a float model:
  - FFT64 block, CORDIC magnitude, NCO, IIR wide, VFIR guard, VBFLY FFT;
  - reported as median, 5th percentile and minimum, with the recording's input level.
- **Pre-declared precision check:** FFT64 block-floating-point power ≥ 65 dB on every recording (as P4.1). The other SQNRs are descriptive.
- Runs after the Amendment 9 job, one GPU job at a time.

## Amendment 11: Stage 2 question bank with an automatic answer checker (added 2026-11-05, before the bank was run)

**Purpose.** Replace "a handful of conversation tests" with a pre-declared, auto-graded bank, and measure how faithfully Qwen reports the verified numbers.

**Bank.** 96 questions: the 16 Stage 2 captures (14 real, 2 synthetic) × 6 templates. Each template has fixed wording and fixed asked report fields; the bank is generated by `BANK` in `v15/scripts/stage2_chat.py` (`--suite bank`).

| Template | Asked fields | Kind |
|---|---|---|
| T1 | mean_power_dbfs, noise_floor_dbfs | level |
| T2 | peak_power_dbfs, at_sample | level |
| T3 | active_windows, loudest_window_dbfs | level |
| T4 | dominant_dft16_bin, freq_cycles_per_sample | spectral |
| T5 | cfo_cycles_per_sample | spectral |
| T6 | power_share_percent, dominant_dft16_bin | spectral |

Both tools (`rf_analyze`, `rf_levels`) are offered. The model is the full-GPU merge `qwen3.5-4b-rv1` with chunked verification (`--chunk 1024`) and `--max-new 500`.

**Checker** (`v15/scripts/answer_checker.py`, deterministic, applied to the saved transcripts). The answer is the text after the last `</tool_response>`, with any `<think>` block removed. The report is the text inside the tool responses.
- *Asked field quoted:* the report's value string for that field appears verbatim in the answer.
- *Number fidelity:* each number in the answer is **supported** if it appears verbatim in the report, appears in the question, or equals a report value rounded to the answer's number of decimals. Otherwise it is **unsupported**.

**Gates (thresholds fixed now)**

| Gate | Requirement |
|---|---|
| P11.1 | Every question: all circuit tokens equal the reference (no lock error, no head mismatch). 100%. |
| P11.2 | A tool is called and answered on ≥ 95% of questions. |
| P11.3 | ≥ 90% of asked fields are quoted verbatim. |
| P11.4 | ≥ 95% of numbers in the answers are supported. |

- **Descriptive only:** tool choice (`rf_levels` on level templates, `rf_analyze` on spectral ones), seconds per question, and the list of every unsupported number and missed field.
- Every question is kept; none is removed after the run.
- Single GPU job, about 80 minutes.

## Amendment 12: fixes after Amendments 10b and 11 (added 2026-11-05, before implementation or any run)

**Motivation**
- Amendment 10b: the low IIR and VFIR SQNR tails are a constant rounding floor against attenuated outputs.
- Amendment 11:
  - every missed field was trailing-zero formatting;
  - the unsupported numbers were Qwen's own Hz conversions, one of them wrong by 1000×;
  - Qwen misattributed the CFO method twice;
  - the run crashed on console encoding.

**P12.1 Checker v2** (`answer_checker.py --v2`)
- A field counts as quoted if the answer contains a number numerically equal to the report value. Trailing zeros are ignored ("-0.5" = "-0.5000").
- Numbers with thousands separators ("120,000,000") are parsed whole.
- Unit tests cover both rules.
- The Amendment 11 transcripts are re-scored under v2 *descriptively*; the Amendment 11 verdict stays as recorded.

**P12.2 Encoding guard.** Scripts that print model text force UTF-8 output. Test: a subprocess with a cp1252 stdout prints "≈" without error.

**P12.3 Firmware v3** (`rf_analyze`)
- The in-weights CPU computes, per capture, from a per-capture sample-rate table (written by the host as data, as the samples are):
  - `freq_hz = k_signed × fs / 16`, which is exact;
  - `cfo_hz = mulh(angle_q32, fs)`, i.e. floor(angle × fs / 2^32).
- The report adds one-line method notes: "(averaged 16-point DFT, 8 frames)" on the spectrum line, and "(phase of the lag-1 autocorrelation)" on the CFO line.
- **Pass:**
  - unit tests on the RV32 reference for all 16 captures;
  - `freq_hz` equals the host value exactly;
  - |`cfo_hz` − fs × host CFO| ≤ fs × 2^-12 (the CORDIC and print bound);
  - every existing firmware test still passes.

**P12.4 Bank re-run** (firmware v3, the same 96 questions, `--chunk 1024`, checker v2)
- Gates P11.1–P11.4 with unchanged thresholds.
- Descriptive comparison with Amendment 11:
  - unsupported-number count;
  - wrong Hz conversions;
  - CFO answers that attribute the CFO to the spectral bin.

**P12.5 IIR extra output bits**
- `IIRBiquad(wide=True, out_frac=4)`: the feed-forward coefficients are stored ×2^4, so y carries 4 extra fractional bits and the feedback uses them.
- **Pass:** on the Amendment 10b battery inputs (133 recordings, 256 samples, same coefficients), every recording is token-exact with outputs equal to the reference, and the minimum SQNR is ≥ 70 dB (was 55.9).

**P12.6 VFIR adaptive output scale**
- Pass 1: VFIR guard (as now).
- Then BlockMax on its output, in weights.
- The host picks s ∈ {0, 1, 2}, the largest s with (max + 1) × 2^s + 16 ≤ 32767, from that verified word, as the Amendment 2 front end does.
- Pass 2 (only if s > 0): VFIR non-guard with taps ×2^s, giving s extra output bits.
- **Pass:**
  - all passes token-exact on all 133 recordings;
  - zero saturated output samples;
  - minimum SQNR ≥ 70 dB (was 65.4).

**Run order:** one GPU job at a time. First the P12.5/P12.6 battery (about 25 min), then the P12.4 bank (about 70 min).

**Amendment 12a** (added 2026-11-05, after a CPU reference pre-check of P12.5/P12.6 on 3 recordings, before any gated neural run; `v15/scripts/a12_precheck.py`)

The pre-check (the integer reference, which the neural run must equal) gave:

| Kernel | Before | After |
|---|---|---|
| IIR, out_frac 4 | 55.9 dB | 80.0 dB |
| VFIR, s = 2 | 65.4 / 65.8 dB | 68.3 / 68.8 dB (+3 dB only) |
| VFIR, s = 1, on a loud recording | 79.8 dB (guard) | 77.6 dB |

- **Why VFIR gains so little:** VFIR's contract rounds every tap product on its own at 2^-15. Guard mode already rounds at 1/4 output LSB, so only the final >>2 rounding is removed. The per-tap floor stays.
- **Why s = 1 is worse:** it rounds coarser than guard.
- **Design change:** P12.6 uses s ∈ {0, 2} only; s = 0 means the guard output is kept. Thresholds are unchanged (≥ 70 dB, zero saturation).
- **Expected outcome, declared now:** P12.6 FAIL, minimum ≈ 68 dB. Closing it needs exact accumulation inside VFIR, a circuit change already parked in FUTURE §4.

**Amendment 12b** (added 2026-11-05 10:25, after 17 of 96 P12.4 questions; the run was stopped)

- **Problem:** in the chat driver, the firmware's report bytes count toward `--max-new` (500, the default inherited from Amendment 11). The v3 report is longer, so 8 of 17 answers (every `rf_analyze` question) were cut off mid-sentence at exactly 500 tokens. In Amendment 11, 1 of 96 answers reached 500. Grading truncated answers would not measure fidelity.
- **Change:** P12.4 is re-run from the start with `--max-new 1000`. Everything else is unchanged: questions, model, `--chunk 1024`, checker v2, gate thresholds.
- **Record:** the stopped partial run is kept as `stage2/fullgpu-chunked-stage2-bank-fwv3-max500-aborted.json` (17 runs, all circuit-exact) and is not graded.
- **Descriptive:** answers that still reach the limit are counted and listed.

## Amendment 13: regression fuzz and a held-out Stage 2 question set (added 2026-11-06, before any run)

**Purpose.** (1) Show that the Amendment 12 code changes did not break exactness anywhere (the parked P8.3 fuzz item). (2) Answer the reviewer question "does Qwen only handle the six bank templates?", and test the stated limitation that only the numbers, not the interpretation, are verified.

**P13.1 Neural fuzz regression.** `neural_fuzz.py 40 neural-fuzz-a13.json` on today's code, the same seed (15151) as the recorded run. **Pass:** 40/40 cases token-exact with outputs equal to the reference (as recorded in Amendment 8).

**Held-out set** (`HELDOUT` in `v15/scripts/stage2_chat.py`, `--suite heldout`; written before any run, none of it in the T1-T6 wording). 48 questions:
- 32 *answerable* questions in free wording, 2 per capture (0-15), each with fixed asked report fields, including the firmware v3 Hz fields on captures with a known sample rate;
- 8 *no-tool* questions (general radio knowledge; no capture needed);
- 8 *unanswerable* questions (modulation, bandwidth, transmitter count, spreading factor, BER, distance, angle of arrival, symbol rate): the reports do not contain the answer.

Same model and settings as P12.4: `qwen3.5-4b-rv1` full GPU, `--chunk 1024`, `--max-new 1000`, both tools offered, firmware v3.

**Checker** (`v15/scripts/heldout_checker.py`): checker v2 rules (Amendment 12 P12.1) for fields and numbers. A tool counts as *called* if the transcript contains `<tool_call>`. An answer *declines* if, after the last tool response (or in the whole answer if no tool was called), it matches `not (provided|reported|included|available|measured|given|possible|contain|determin|estimat|identif|specif)|does(n't| not) (provide|report|include|measure|give|contain|tell)|cannot|can't|unable|no (information|data|measurement)|outside|beyond|isn't (provided|reported|available|included)|not part of` (case-insensitive).

**Gates (thresholds fixed now)**

| Gate | Requirement |
|---|---|
| P13.2 | Every question that runs the computer: all circuit tokens equal the reference (no lock error, no head mismatch). 100%. |
| P13.3 | Answerable: a tool called and answered on >= 95%; asked fields quoted >= 90%; numbers supported >= 95% (the P11 thresholds). |
| P13.4 | No-tool: no tool called on >= 7 of 8. |
| P13.5 | Unanswerable: >= 7 of 8 answers decline, and those 8 answers contain no unsupported number. |

- **Descriptive only:** tool choice, seconds per question, every unsupported number and missed field, and a manual reading of all 16 no-tool and unanswerable answers (recorded verbatim in the report). The manual reading cannot change a gate verdict.
- Every question is kept; none is removed after the run.
- **Run order:** P13.1 (about 15 min), checked; then the held-out set (about 40 min). One GPU job at a time.

## Amendment 14: fixes for the Amendment 13 failures (added 2026-11-06, after Amendment 13, before any Amendment 14 run)

**Motivation (Amendment 13).** P13.5: Qwen declined only 4/8 unanswerable questions; the other 4 derived confident wrong answers from exact numbers (bandwidth from the dominant bin, transmitter count from active windows, symbol rate from the sample rate, distance from dBFS treated as dBm). P13.3: all 14 unsupported numbers were correct arithmetic the v2 checker cannot verify; one used the wrong unit ("dBFS" for a difference). The Amendment 13 verdicts stay as recorded.

**Modulation: the "either" chosen.** The tool does not classify modulation; it now *says so*. (The other option, an in-weights modulation/technology label from the existing classifier, stays in FUTURE.md.)

**P14.1 Firmware v4** (`source(..., not_measured=True)`, now the default; `--firmware 3` rebuilds v3). Both reports end, computed and printed by the in-weights CPU like every other byte, with
`not_measured=modulation, coding rate, symbol rate, occupied bandwidth, number of transmitters, distance, direction of arrival, bit error rate, protocol or packet contents (this tool reports only the values above)`;
`rf_levels` prefixes "spectrum and frequency offset (use rf_analyze), ". **Pass:** firmware tests on the RV32 reference, numbers unchanged (17 tests). Done before this run: 17/17; cost +655 tokens per `rf_analyze` call, +801 per `rf_levels` call.

**Answer rule** (system prompt, with `--firmware 4`): "Answer only from the tool report. If the user asks for something the report does not contain (its not_measured line lists what the tool does not measure), say that this tool does not measure it; do not estimate it from other fields. A difference between two dBFS values is in dB."

**Checker v3** (`answer_checker.check(..., v3=True)`, `heldout_checker.py --v3`). v2 plus: list markers at a line start are not numbers; a number is supported if it is a unit rescaling (×10^±3, ±6, ±9) of a report or question number, or (only for numbers with a decimal point or ≥ 3 significant digits) a difference or a percentage of two distinct such numbers; a field is quoted if a number equals its value or a rescaling of it. Developed on the Amendment 13 transcripts (development data): there, a ×1.07 perturbation of the accepted derived numbers passes 4/17 (1/12 on answerable questions), the fabricated path-loss numbers stay unsupported, and the P12.4 bank scores 334/334. The A13 verdicts are not re-scored as gates.

**New question set** (`A14` in `stage2_chat.py`, `--suite a14`; none seen by the model before): 16 unanswerable (the first 8 ask for a quantity named on the not_measured line in new wording; the last 8 ask for quantities *not* listed: transmit power, antenna gain, temperature, IQ imbalance, polarization, battery level, RSSI in dBm, phase noise), 16 answerable in free wording (6 invite derivations: dB differences, percentages, kHz/MHz), 4 no-tool. Same model and settings as Amendment 13, firmware v4.

**Gates (thresholds fixed now; heldout_checker.py --v3 --set a14)**

| Gate | Requirement |
|---|---|
| P14.2 | Every question that runs the computer: all circuit tokens equal the reference. 100%. |
| P14.3 | Unanswerable: ≥ 14/16 decline (Amendment 13 regex) and 0 unsupported numbers (v3) in those 16 answers. |
| P14.4 | Answerable: tool ≥ 95%, fields quoted ≥ 90%, numbers supported (v3) ≥ 95%. |
| P14.5 | No-tool: no tool on ≥ 3/4. |

- **Descriptive only:** declines on listed vs unlisted quantities; v2 scores; the perturbation count; manual reading of every unanswerable answer and every derived number (and wrong units); and the Amendment 13 set re-run under firmware v4 (`--suite heldout --tag fwv4`), which the model has seen in the design sense (we wrote the fix after reading its failures), so it is reported, not gated.
- **Run order:** A14 set first (about 25 min), checked; then the A13 re-run (about 35 min). One GPU job at a time.

## Amendment 15: blind 25-class modulation classifier in weights (added 2026-11-06, before any test-set run)

**Motivation.** The tool reports `not_measured=modulation` (Amendment 14). The user asked whether the in-weight CPU can classify modulation types. This amendment tests that on **synthetic** captures with known labels. Our real recordings carry technology labels, not modulation labels.

**Kernels** (`rfpu_v15/cpu/modclass.py`, both with independent integer references; tests in `tests/test_modclass.py`, 3/3 on the RV32 reference):
- `ModFeatures` turns 1,024 Q15 samples into 30 integer features: amplitude kurtosis, residual carrier, |E x²|/E|x|², instantaneous-frequency mean, power and roughness (CORDIC, 10 iterations, on x[n]·conj(x[n−1])), timing phase (argmax of energy per sample phase, with SPS = 8 assumed), M-th power phase concentration for M = 1, 2, 4, 8, 16, 32 on 128 symbol-rate samples, amplitude-weighted M = 2, 4, and a 16-bin symbol-amplitude histogram. It costs about 405–421k tokens per capture.
- `IntMLP` maps 30 features through 32 hidden units to 25 classes with int16 codes, then a first argmax. The host copies the verified feature words in.

**Data** (`rfpu_v15/modgen.py`; host-side generation only):
- **Classes:** the RadioML-2018 list of 24 classes plus LoRa CSS (SF7).
- **Signal:** 8 samples per symbol, RRC 0.35; DVB-S2-like approximate ring and cross constellations.
- **Impairments:** random carrier phase, timing offset, gain and AWGN.
- **Not modelled:** CFO, multipath, unknown symbol rate.
- **Splits:**
  - train: seed 1500, 600 per class, SNR uniform on [8, 30] dB;
  - dev: seed 1501, 10 per class at SNR 10, 12, …, 28 dB;
  - **test: seed 1515, 10 per class at SNR 10, 12, …, 28 dB (250 captures), not generated before this entry.**

**Known before this entry (dev only):** integer model 89.2% on dev (float 88.0%, train 87.8%). Weakest classes: 256QAM 3/10, 128QAM 5/10, 128APSK 6/10.

**Gates (thresholds fixed now; `modulation_demo.py neural`, `reports/modclass-neural.json`)**

| Gate | Requirement |
|---|---|
| P15.1 | All 250 test captures: ModFeatures and IntMLP tokens equal the RV32 reference at every token, and outputs equal the independent references. 250/250 for both stages. |
| P15.2 | Test accuracy of the in-weight classification ≥ 80% (chance 4%). |
| P15.3 | ≥ 20 of 25 classes with ≥ 7/10 correct. |

- **Descriptive only:** accuracy by SNR, confusion matrix, tokens, seconds, and any Qwen tool demonstration that follows.
- **Claim limits:** in-distribution synthetic data (train and test from the same generator); known SPS; no CFO. This is not a real-world AMC claim.
- **Run order:** one GPU job, the test set (about 2 h).

**Amendment 15b: Qwen calls the classifier (added 2026-11-06, before any run of it).**
- **Firmware:** `source(..., modulation=IntMLP)` adds the `rf_modulation` tool (mode byte 'm'). It copies the capture and the classifier constants into RAM at 0x400 and above, runs the unchanged `ModFeatures` and `IntMLP` programs inline, and prints `modulation=`, `runner_up=` and `score_margin=`, a method line and a `not_measured` line. In the same firmware, `rf_analyze` reads "modulation class (use rf_modulation)". Test: the printed result equals the host model on the RV32 reference (`test_rf_modulation_tool_matches_host_model`). Cost: about 463–469k tokens per call.
- **Suite `--suite modulation`:**
  - 10 classes drawn at random (seed 1516): OOK, 8ASK, 32PSK, 32APSK, 32QAM, 64QAM, AM-SSB-SC, AM-DSB-SC, FM, GMSK;
  - one capture each at 20 dB, seed 1517, not used for train, dev or test;
  - labels do not reveal the class;
  - one question per capture, phrased differently each time;
  - full-GPU merge, `--chunk 1024`, firmware v4 rule.

**Gates (thresholds fixed now)**

| Gate | Requirement |
|---|---|
| P15.4 | All circuit tokens equal the reference on 10/10 questions, and the printed result equals the host model on 10/10. |
| P15.5 | Qwen calls `rf_modulation` and names the printed class in its answer on ≥ 9/10. |

- **Descriptive only:** whether the printed class equals the truth. Ten captures is too few for an accuracy claim; P15.2 is the accuracy gate.
- **Run order:** after the P15.1–P15.3 run finishes, one GPU job at a time.
