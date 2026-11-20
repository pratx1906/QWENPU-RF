# v1.5 research log

This is a chronological record of every run and result, including failures, negative results and caveats. Each entry names its evidence.
- **Ledger runs** (`v15/scripts/run.py`) are in [../results.md](../results.md) and `../results/raw/`.
- **Gates** are in [../reports/gates.json](../reports/gates.json).
- **Pre-declarations** are in [PROTOCOL.md](PROTOCOL.md).

Labels used below:
- **Neural:** every token was checked against the RV32 reference, and outputs against the independent reference.
- **Reference only:** run on the RV32 reference machine; not a neural claim.
- **Circuit:** the compiled circuit was run teacher-forced, with every token checked.

---

## 2026-11-03 to 2026-11-04 night (overnight queue)

| Run (ledger name) | Kind | Result | Evidence |
|---|---|---|---|
| `real-iq-bfp-gate` | Neural | 14/14 real captures pass (Amendment 2, BFP front end in weights) | `reports/real-iq-bfp.json` |
| `vfir-neural-packed` | Neural | VFIR packed I/O: 2.6× / 3.5× / 6.7× vs scalar FIR (4 / 8 / 16 taps) | `reports/vfir-neural.json` |
| `stage2-demo-fastfw` | Neural (executor) | Cheaper firmware: 92.7–100.1k chip tokens/call, 121–163 s/question, lock errors 0 | `reports/stage2/stage2-chat.json` |
| `stage2-fullgpu-fastfw` | Neural (saved layers, 19 GB merge) | 271–299 s/question, lock errors 0, head mismatches 0 | `reports/stage2/fullgpu-stage2-chat-oldfw.json` |
| `timing-study` | Neural, pre-declared (Amendment 3) | VFIR16 packed vs scalar FIR16: 6.73× geometric mean, 95% CI [6.727, 6.738], exact sign-flip p = 0.0078 | `reports/timing-study.json` |
| `speculative-verify` | Neural (saved layers), exploratory | lowpass16 FIR, 32 samples (~10k tokens): chunked verification 0 mismatches at C = 64/256/1024; 74× vs free-running **on the HF path** (35.6 tok/s) | `reports/speculative-verify.json` |
| `vpow-executor` | Neural (executor) | VPOW: 8/8 random programs token-exact | `reports/vpow-executor.json` |

Tests: 173 passed, 11 skipped; branch `test_vfir` 10/10.

---

## 2026-11-04 morning: error budget and precision (Amendment 4)

The protocol was declared before implementation. One definitional clarification (the stricter reading) was added after the CPU unit tests and before any neural run.

**New kernels**
- `FFTPower(scaling="block")`: a block-floating-point FFT on the CPU.
- `NCOMixer(interpolate=True)`: Taylor interpolation of the sine table.

**Neural run `error-budget`:** 34 rows across 11 kernels. All gates pass.

| Gate | Target | Result |
|---|---|---|
| P4.1: block FFT64 power, every level −60…0 dBFS | ≥ 65 dB | 74.6–76.3 dB (fixed scaling: 7.5–66.5) |
| P4.2: block FFT64 at −12 dBFS | ≥ 70 dB | 74.6 dB (fixed: 56.1) |
| P4.3: interpolated NCO | ≥ 75 dB | 85.8 dB (table only: 49.0) |
| P4.4: every row neural-exact | all rows | 34/34 |

**Token costs:** FFT64 201 → 266 tokens/sample (+32%); NCO 53 → 72 (+36%).

**Notable rows (arithmetic SQNR at −12 dBFS unless stated)**

| Kernel | Arithmetic SQNR |
|---|---|
| lowpass16 FIR | 79.4 dB (design SQNR 77.3) |
| FIR16 / FIR64, 64-bit accumulator | 81.3 / 87.3 dB |
| Decimating FIR | 79.6 dB |
| IIR (+12 dB DC gain) | 35.1 dB at −12 dBFS (saturates); 70.2 dB at −24 dBFS |
| DC blocker | 69.2 dB |
| CORDIC magnitude | 155 dB |
| Exact energy | inf (exact) |
| VX int8 energy | 35.2 dB |

Evidence: `reports/ERROR_BUDGET.md`, `reports/error-budget.json`.

---

## 2026-11-04 midday: token reduction and verification speed (Amendment 5)

| Run | Kind | Result |
|---|---|---|
| `fft-opt-neural` (P5.1) | Neural | FFT stages rewritten, outputs bit-identical. Tokens/sample: FFT16 152 → 125, FFT64 201 → 166, block FFT64 268 → 230, FFT256 251 → 206, Welch 412 → 343. ms timings may be slightly inflated: CPU tests ran concurrently. |
| `am-trim-neural` (P5.2) | Neural | AM envelope with a magnitude-only CORDIC: 450 → 256 tokens/sample, bit-identical; k = 10 iterations: 185 tokens/sample at 121.6 dB (target ≥ 90) |
| `speculative-fir64` (P5.3) | Neural (saved layers) | FIR64 × 48 real samples, 107,023 tokens: 0 mismatches at C = 256/1024/4096. **Negative result:** 2.7–3.4k tok/s, *slower* than the fused executor (4,363 tok/s, batch 1). |
| `speculative-fir64-fla` | Neural | Flash-linear-attention kernels: still 0 mismatches, but 1.5–2.3k tok/s (slower). Diagnosis: the per-position `ck.decode` host sync (~107k GPU syncs) dominated. |
| `speculative-fir64-batched` | Neural | Batched decode (one copy per chunk): 4.4–6.0k tok/s |
| `speculative-fir64-fla-batched` | Neural | FLA + batched decode: 12.7k tok/s (C = 1024); C = 16384 was slower (3.4k) |
| `speculative-fir64-fla-gpucmp` | Neural | FLA + GPU-side code compare: 11.5–13.0k tok/s = 3.0× the executor, ~360× HF free-running |
| `speculative-mutation` | Neural | One drafted token altered (position 50,000): detected, 2 mismatches, first at 49,999 |

The P5.3 gate passes on exactness. The 74× figure from the night was against the HF path only; against the executor the gain is 3.0×.

---

## 2026-11-04 afternoon: chunked verification inside Stage 2, and `rf_levels`

**Firmware profile** (reference only, one `rf_analyze` call, 92,570 tokens)

| Part | Share |
|---|---|
| Pass 1 per-sample loop (power, peak, windows, exact 64-bit CFO) | 61.7% |
| DFT16 × 8 frames | 33.6% |
| Parse and print | ~4.7% |

Recorded in `reports/STAGE2.md`.

**`rf_levels` tool** (a second function on the same firmware, selected from the call text)
- Covers power, peak, noise floor and windows.
- Unit tests show identical level numbers to `rf_analyze` at ~28k instead of ~93k tokens.
- First implementation bug: the levels branch jumped past pass 1's initialization. Fixed with an entry label; caught by the tests.

**Chat runs** (all on one GPU, 19 GB merge)

| Run | Kind | Result |
|---|---|---|
| `stage2-fullgpu-chunked` | Neural, FLA env, **old firmware** (the module had been imported before the firmware edit) | 0 lock errors, 0 head mismatches; chip 11–28 s/question. Qwen's final answer wording differs from the earlier run (FLA kernels in Qwen's float layers); every circuit token is exact. |
| `stage2-fullgpu-chunked-nofla` | Neural, main env, new firmware | 47–54 s/question, chip 16 s |
| `stage2-fullgpu-sequential-newfw` | Neural, main env, new firmware (control) | 255–288 s/question, chip 223–247 s |

**Token identity:** chunked vs sequential with the same firmware and environment: all 3 questions identical (93,449 / 100,884 / 100,984 tokens, including Qwen's answers).

**Confound found and resolved:** the first comparison (chunked vs the night's run) differed at the first chip token. Cause: the firmware itself changed (the mode scan added for `rf_levels`), not the chunking.

**`stage2-levels-chunked`** (neural, both tools offered)

| Question | Tool Qwen chose | Chip tokens | Seconds |
|---|---|---:|---:|
| Noise floor and loudest window (capture 14) | `rf_levels` | 25,769 | 31 |
| Mean and peak power (capture 12) | `rf_levels` | 28,541 | 27 |
| Burst frequency (capture 14) | `rf_analyze` | 92,913 | 48 |

0 lock errors, 0 head mismatches; correct answers.

Evidence: `reports/stage2/fullgpu-chunked-*.json`, `fullgpu-sequential-newfw-stage2-chat.json`, `fullgpu-chunked-stage2-levels.json`. Tests: 193 passed, 11 skipped.

---

## 2026-11-04 afternoon: VBFLY (FFT butterfly instruction) on branch `rfpu-vfir`

**Design checks**
- **Existing VX ops (VEW ROPE + ADD) as an FFT:** estimated ~45 tokens/butterfly, floor rounding with Q14 twiddles. Not pursued: barely better than the CPU and less precise.
- **Constant-geometry twiddle formula, verified numerically:** natural-order input, `TW[r] = W^brev5(r)`, row = `i & (2^s − 1)`, output bit-reversed. One 32-entry table serves all stages and frames.

**Reference and circuit**

| Step | Kind | Result |
|---|---|---|
| Commit `15bb774`: reference semantics, decode (uop 83, custom-2 funct7 = 1 funct3 = 2), `vbfly` mnemonic | Reference only | Oracle tests: full FFT64 bit-exact vs an independent integer constant-geometry FFT, within 2L LSB of numpy; 40 random single ops; contract violations raise. 68/68 VX reference tests. |
| Commit `824c694`: circuit | Circuit | Bank A/B read pair `hoff/2 + 16 rhi + j` (L1 forms); RoPE table 4 → 32 pairs read at `i & mask`; butterfly on the RoPE MAC. `test_vbfly`: 6/6 seeds, every token, first try. Regression: `test_rv_vx` 27/27. Size: 13 layers, 137 heads, dims 6,906 → 6,950, FFN units 80,295 → 81,199 (+1.1%). |

**Precision-preserving VBFLY** (Amendment 6, declared before implementation): track max |y| so the CPU can choose per-stage scaling (block floating point).

- **First attempt (negative result):** comparing on the MAC output directly made the circuit **14 layers**. Reverted.
- **Second design** (as VEW): with `track`, each output streams as VE VE (MAC), then as a word from the lags, with the compare on lag bits. 13 layers. Cost: 16 tokens/butterfly when tracked (n ≤ 4), 8 untracked.
- **First circuit test of the second design failed** at the 3rd and 4th tokens of every tracked output. Cause: lag indices off by one (the circuit predicts the next token, so "lag 1" is the current input token). Fixed. **Rerun: `test_vbfly` 6/6 seeds, every token (tracked and untracked ops, VGET 3 after tracked stages, chained stages), 13 layers.**

**FFT64 kernel `VXFFT64`** (`v15/rfpu_v15/cpu/vfft.py`): CPU windowing and normalization, 6 VBFLY stages with tracked BFP scaling, bins read by VPOW.

Development check, **reference only, not neural**, complex Gaussian input, 512 samples:

| Level | Exact vs independent model | Tokens/sample | Arithmetic SQNR |
|---|---|---:|---:|
| −60 dBFS | yes | 127 | 70.8 dB |
| −30 dBFS | yes | 129 | 70.0 dB |
| −12 dBFS | yes | 129 | 70.1 dB |
| 0 dBFS | yes | 133 | 72.2 dB |

For comparison, both from the neural runs above:
- CPU block FFT64: 230 tokens/sample at 74.6–76.3 dB.
- CPU fixed FFT64: 166 tokens/sample at 56 dB.

Expected trade-off, pending the neural run: about 1.8× fewer tokens than the CPU block FFT, at about 4 dB less precision. The P6.1 neural gate (≥ 68 dB at every level −60…0) is not yet run.

---

## 2026-11-04 afternoon: Amendment 6 results

Commit `b72623e` (branch): VBFLY track mode. Regression: 96 VX circuit and reference tests pass; 13 layers, 137 heads, dims 6,984, FFN units 81,438.

Ledger run `amendment6` (neural, branch circuit, fused executor):

| Gate | Target | Result | Pass |
|---|---|---|---|
| P6.1 VBFLY block FFT64 power | ≥ 68 dB at every level −60…0 dBFS | 69.5–71.5 dB; 127–133 tokens / 19–21 ms per sample. CPU block FFT, same run: 74.6–75.2 dB, 229–233 tokens / 34–38 ms | **PASS** |
| P6.2 IIR `wide=True` (+12 dB DC gain) | ≥ 65 dB at −12 and −6 dBFS | 82.3 / 88.2 dB (int16 saturating: 35.1 / 14.8); 130–134 vs 172–176 tokens/sample | **PASS** |
| P6.3 VFIR 16 taps | ≥ 70 dB at −12 dBFS | **69.2 dB** (exact-accumulation scalar FIR16: 81.3) | **FAIL** (by 0.8 dB; the per-tap rounding contract) |
| P6.5 energy | documentation | exact scalar energy is the precision path | n/a |

Evidence: `reports/amendment6.json`.

## 2026-11-04 afternoon: P6.4 classifier screening (ledger `classifier-screen`, host float)

- **Diagnosis:** DVB-T → LTE confusion (398/640 DVB-T windows); WiFi 88%, LTE 95%.
- **Candidate:** OFDM cyclic-prefix lag correlations |R(L)|/R(0) at L = 64, 67, 128, 224, 448.
- **Leave-one-location-out CV**, calibration recordings only (just 2 locations):

  | Feature set | Window accuracy | Recording accuracy |
  |---|---:|---:|
  | Baseline | 43.1% | 47.6% |
  | Baseline + lags | 46.2% | 52.4% |

  Baseline + lags was selected.
- **Held-out, evaluated once:** 67.2% windows / 70.3% recordings, the same as the recorded in-weights model (70.3%). **No improvement**; DVB-T is still mostly called LTE (399/640). Not implemented in weights.
- **Reading:** CV accuracy across the two training locations is ~45–52%, so the model does not transfer between locations. The limit is the data (2 training locations, 1 MS/s capture of wideband signals), not the arithmetic.

Evidence: `reports/classifier-screen.json`.

## 2026-11-04 afternoon: P6.4b more training locations (ledger `classifier-lolo6`, host float)

Leave-one-location-out over all 6 IDLab 1 MS/s locations (164 recordings, each tested once):

| Variant | Recording accuracy | Window accuracy |
|---|---:|---:|
| 17 features, H = 8 | **74.4%** | 70.1% |
| 17 features, H = 16 | 73.2% | 69.3% |
| 17 + lag features, H = 8 (primary) | 73.8% | 70.5% |
| 17 + lag features, H = 16 | 72.6% | 68.7% |

- **More training locations:** +4 points over the fixed-split 70.3% (a different protocol; reported alongside).
- **Lag features:** do not help. The primary does not beat the baseline, so there is no in-weights change.
- **Per-location accuracy:** 50% (uz) to 93% (reep).
- **DVB-T is still the error:** 40% correct, 402/960 windows predicted LTE.
- **Reading:** at 1 MS/s only a 1 MHz slice of 8–20 MHz signals is seen, so DVB-T and LTE look alike to these features.

Evidence: `reports/classifier-lolo6.json`.

## 2026-11-04 late afternoon: P6.4c classifier at 10 MS/s; VBFLY checkpoint merge; ablation table

**Ledger runs**
- `classifier-10msps`: first run keyed recordings by file name, so two recordings that share a name were counted as one (25 units).
- `classifier-10msps-fixed`, first attempt: **failed** with out-of-memory; the 27 GB merge was running concurrently.
- `classifier-10msps-fixed`, second attempt: passed. 26 recordings, leave-one-recording-out:
  - **84.6% recordings / 70.9% windows**, for both the float model and the exact integer in-weights model;
  - DVB-T windows 36/80;
  - final model neural on one window per recording: 26/26 token-exact, outputs equal the integer model.
  - Indicative only (5 DVB-T recordings). Evidence: `reports/classifier-10msps.json`.

**`merge-qwen4b-vx-rf`:** `chipformer.rv.merge --vx` from the branch at `b72623e` (VFIR, VPOW, VBFLY) to `$RFPU_DATA/checkpoints/rfpu-v15/qwen3.5-4b-vx-rf`. Passed.

**Ablations A1–A8** compiled from measured runs: `reports/ABLATIONS.md`.
- A2 and A3: N/A.
- A7: width and depth blocked (only the 4B hosts the circuit); instruction encodings measured, including the rejected 14-layer VBFLY variant.

**`vx-rf-checkpoint`** (saved weights of `qwen3.5-4b-vx-rf`, free-running, every token checked): **5/5 exact**, loading 48 s.

| Program | Tokens | Time | Notes |
|---|---:|---:|---|
| VXFFT64 on one real-IQ frame | 8,696 | 191 s | outputs = independent model |
| VXFir16 | 2,816 | 64 s | |
| `vbfly_program` seed 0 | 3,744 | 85 s | |
| `vbfly_program` seed 1 | 3,962 | 85 s | |
| `vpow_program` seed 0 | 2,075 | 45 s | |

Evidence: `reports/vx-rf-checkpoint.json`. The VFIR checkpoint result from 2026-11-03 also stands.

**`stage2-real-chunked`** (Amendment 7: real recordings): **6/6 pass.** 0 lock errors, 0 head mismatches, reports equal the host model, asked numbers quoted exactly. Qwen chose `rf_levels` for the 4 level questions and `rf_analyze` for the spectral and CFO questions; 29–42 s per question. Evidence: `reports/stage2/fullgpu-chunked-stage2-real.json`, `reports/STAGE2.md`.

**Documents written:** `reports/ABLATIONS.md`, `REPRODUCE.md`, `configs/env-main.txt`, `configs/env-fla.txt`, `paper/PAPER_DRAFT.md`.

## 2026-11-04 evening: Amendment 8

- **VFIR guard bit** (branch commit `2037c86`):
  - Circuit test 4/4 every token; VX regression 106 passed; 13 layers.
  - **P8.1 (`amendment8`, neural): 77.9 dB at −12 dBFS, target ≥ 75 → PASS**, at 110.5 tokens/sample.
  - For comparison: per-tap VFIR 69.2 dB (P6.3 stays recorded as failed); exact scalar FIR16 81.3 dB at 624.8 tokens/sample.
- **P8.6 ablation:**

  | VBFLY FFT64, −60…0 dBFS | SQNR | Tokens/sample (−12 dBFS) |
  |---|---|---:|
  | Fixed scaling | 6.6–64.1 dB | 103 |
  | Block floating point | 69.5–71.5 dB | 129 |

- **P8.3a (`regress-real-iq-bfp`):** 14/14 again, 57,344 samples.
  - The run was slowed by concurrent CPU work. Exactness only; no timing claimed.
- **P8.2 screening (`classifier-10msps-bandwidth`, host float):** 10 MS/s + FFT64 bandwidth shape, leave-one-recording-out.
  - **96.2% recordings / 78.8% windows** (was 84.6% / 70.9%); DVB-T windows 62/80 (was 36/80).
  - The first attempt **failed**: the classifier constructor accepts only 17 features. The in-weights version followed (`TechClassifier(bandwidth=True)`: stage A FFT64 in weights, host copies the verified powers, stage B classifier with 81 features); tests pass.
- **Systems tightening:**
  - An AST lint found 7 unused imports; removed, now clean.
  - New tests: `test_vfft.py` (both modes), the VFIR guard (`test_vfir.py`), the bandwidth classifier (`test_detect_infer.py`).
  - A uniqueness assert on recording keys.

- **`classifier-10msps-bandwidth-neural`:**
  - The first attempt passed in the ledger but took a **stale early-return path**, so it produced the screening output without the integer model or the neural check. Fixed.
  - Rerun: the exact integer in-weights model scores **96.2% recordings / 78.8% windows** (identical to float).
  - Two-stage neural run (FFT64 in weights, verified powers copied, classifier in weights): **26/26 token-exact**, predictions equal the integer model.

- **P8.4 (`regress-stage2-robust-chunked`):** passed.
  - Same behaviour as the recorded robust run; deterministic.
  - The answer no longer cites a "truth value in the capture description" (the label leak was removed 2026-11-03).
- **P8.5 (`stage2-chunk-sweep`):** chunk sizes 0, 64, 256, 1024 and 4096 are all token-identical (92,913 chip tokens).

  | Chunk size | 0 | 64 | 256 | 1024 | 4096 |
  |---|---:|---:|---:|---:|---:|
  | Seconds per question | 277.8 | 110.3 | 58.1 | 47.6 | 46.1 |

- **Stakeholder page** published (private); source text `v15/STAKEHOLDER_OVERVIEW.md`.

- **P8.3b (`regress-pipeline-14`):** PASS. All 14 streams' outputs (detections, classes, peaks) are identical to 2026-11-03, every token exact. 30.35M tokens vs 31.12M (−2.5%, from the FFT stage rewrite); 38.1 ms/sample.
- **`regress-pipeline-synthetic`:** PASS, outputs identical to 2026-11-03 (8.33M tokens).
- **Amendment 9 (expanded real-IQ screen)** declared, then the selection was drawn: 133 recordings (IDLab 53, EPFL 32, NTU 27, Tampere 19, CAMRAS 2), all readable. The run `real-iq-expanded-133` was launched (~5.5 h single GPU job).
- **Parked:** the neural fuzz regression, and all limitation fixes and ideas, in [../FUTURE.md](../FUTURE.md).

- **Amendment 9 result (`real-iq-expanded-133`, finished 22:43):** **132/133 PASS (99.2%, Wilson 95% CI 95.9–99.9%)**. Every sample neural (0 CPU-only), every kernel token-exact against the reference.

  | Dataset | Pass |
  |---|---|
  | IDLab | 53/53 |
  | EPFL LoRaIQ | 32/32 |
  | NTU RFSR | 27/27 |
  | Tampere drones | 18/19 |
  | CAMRAS radio astronomy | 2/2 |

  - Arithmetic NMSE: FIR median 7.5e-9 (max 3.7e-8); spectral median 4.9e-9 (max 2.4e-8). Threshold 0.01.
  - The 6 recordings shared with the 14-capture screen: 6/6.
  - **The one failure:** Tampere `DJI_matrice_210_5G_1of2.bin`, on the matched-peak criterion. The displacement is 2 samples against a limit of 1, while FIR NMSE is 1.1e-8 and spectral NMSE is 2.4e-9. CPU diagnosis: the float matched-filter output has a near-tie. Sample 2735 is within 0.024% of the maximum at 2737, so the Q15 rounding (relative error ~1e-5) flips the argmax. The arithmetic is correct, but the pre-declared criterion fails, and it stays a failure. Thresholds are not changed.
- **Amendment 10a (`pipeline-expanded-133`)** launched 22:45.

- **Amendment 10a result (`pipeline-expanded-133`, finished 01:56 on 2026-11-05): PASS.** All 133 streams have every one of the 8 stages token-exact, with outputs equal to the independent references. 2,048 samples per stream; 288.6M chip tokens; 1,060 tokens per sample; 38.9 ms per sample neural (10,585 s neural, 11,519 s wall). Per-stream summaries are in `v15/reports/pipeline-expanded.json` (descriptive only).
- **Amendment 10b (`kernel-battery-133`)** launched 01:57.

- **Amendment 10b result (`kernel-battery-133`, finished 03:38 on 2026-11-05): PASS.** Both gates pass: all 10 kernels are token-exact with outputs equal to the references on all 133 recordings (256 samples each), and FFT64 block reaches at least 65 dB on every recording.

  Arithmetic SQNR (dB) across recordings:

  | Kernel | Median | 5th percentile | Minimum |
  |---|---:|---:|---:|
  | FFT64 block | 75.7 | 74.1 | 72.7 |
  | CORDIC | 150.4 | 148.6 | 142.5 |
  | NCO Taylor | 81.1 | 79.9 | 74.1 |
  | IIR wide | 78.5 | 66.5 | 55.9 |
  | VFIR16 guard | 73.4 | 71.1 | 65.4 |
  | VBFLY FFT64 block | 71.5 | 70.1 | 68.6 |

  Welch, CFAR, FM and CFO are exact-only (no float model in the battery).
  - **Caveat, diagnosed** (`v15/scripts/battery_tail_diag.py`, CPU integer reference, which equals the neural outputs; it reproduces the battery SQNRs to 0.0 dB; output in `v15/reports/battery-tail-diag.json`):
    - The low tails (IIR 55.9 dB, VFIR 65.4 dB) are **not clipping**. My first guess was wrong: IIR `wide` has no saturation, and the worst recordings have 0 saturated samples, with peak outputs of 1.5k–4.9k against 32,767.
    - **Cause:** the rounding-error power is a constant floor, about 0.66–1.28 LSB² for IIR (median 0.95) and 0.29–0.47 LSB² for VFIR (median 0.35), independent of the recording. SQNR therefore tracks the filter's *output* power: corr(SQNR, log output power) = 0.994 for IIR and 0.992 for VFIR.
    - The worst recordings are wideband DJI 5G captures at 200 MS/s. The test lowpass rejects most of their energy (−12 to −17 dB output/input), so the fixed rounding floor is relatively larger. By contrast, IDLab Wi-Fi gains +10 dB through the same filter and reaches 89.6 dB.
    - This is expected fixed-point behaviour (absolute error bounded, relative error grows when the filter attenuates the signal), not an arithmetic fault.
    - **Possible fix (parked):** keep 2–4 extra fractional bits in the wide IIR output (and the VFIR guard), worth roughly +12–24 dB on attenuated outputs.

- **Amendment 11 declared** (2026-11-05, before the run): a Stage 2 question bank of 16 captures × 6 templates = 96 questions, graded by the deterministic `v15/scripts/answer_checker.py` (asked fields quoted verbatim; every number in the answer supported by the verified report). Gates are P11.1–P11.4.
  - The checker was dry-run on the earlier 6-question real suite: 6/6, 17/17 numbers supported. This is consistent with the record and is a tooling check, not a result.
  - `stage2-bank-96` launched (full GPU, `--chunk 1024`).

- **`stage2-bank-96` crashed after 77/96 questions** (07:52). Cause: printing Qwen's answer containing "≈" to a cp1252 console (`UnicodeEncodeError`). Not a computation fault, and the 77 saved runs are complete. Fix: `PYTHONIOENCODING=utf-8`, plus a new `--resume` flag that keeps the saved runs and skips answered questions. Each question is independent, so resuming does not change any result. A backup of the 77-run file is kept in the scratchpad. `stage2-bank-96-resume` launched for the remaining 19.

- **Amendment 11 result (`stage2-bank-96` + resume, finished 08:09): all four gates PASS** (`v15/reports/stage2/fullgpu-chunked-stage2-bank-a11-checked.json`).

  | Gate | Result | Target |
  |---|---|---|
  | P11.1: circuit tokens equal the reference, no head mismatch | 96/96 | 100% |
  | P11.2: tool called and answered | 96/96 | ≥ 95% |
  | P11.3: asked fields quoted verbatim | 170/176 = 96.6% | ≥ 90% |
  | P11.4: numbers supported by the verified report | 308/311 = 99.0% | ≥ 95% |

  - **Tool choice, descriptive:** 48/48 level templates used `rf_levels` and 48/48 spectral templates used `rf_analyze`. Median 37 s per question.
  - **All 6 missed fields are trailing-zero formatting**, for example "-0.5" for the report's "-0.5000" and "0.0" for "0.0000". They are numerically equal, but they count as misses under the declared verbatim rule.
  - **The 3 unsupported numbers are all Qwen's own unit conversions of the CFO into Hz:**
    - 0.0088 cyc/sample → "≈ 88 kHz at 10 MS/s": correct.
    - −0.1950 × 120 MS/s → "≈ −23.4 MHz": correct.
    - −0.0626 × 1 MS/s → "≈ −62.6 Hz": **wrong**; it should be −62.6 kHz, a unit error by 1000×. The checker flagged it as unverified.
  - **Interpretation error not caught by the number checker:** in two answers Qwen claimed the CFO "is derived from the dominant spectral bin analysis". It is in fact the lag-1 autocorrelation phase. This is evidence for verified labels and explanations (FUTURE 1b).
  - **Checker caveat:** numbers with thousands separators ("120,000,000") are split into parts; parts such as "000" pass by rounding. No case here changes a verdict, but fix the tokenizer before reuse.

- **Amendment 12 declared** (fixes 1–8). Implemented `IIRBiquad(out_frac)` and `VXFir(scale)`.
  - CPU reference pre-check on 3 recordings (`a12_precheck.py`):
    - IIR: 55.9 → 80.0 dB.
    - VFIR s = 2: 65.4 → 68.3 dB only. VFIR rounds each tap product on its own, so output scaling cannot remove that floor.
    - VFIR s = 1 is worse than guard.
  - **Amendment 12a** records this before the gated run: s ∈ {0, 2}, threshold kept, expected P12.6 FAIL.

- **Amendment 12 implementation (2026-11-05 morning):**
  - **P12.1 checker v2** (`answer_checker.py --v2`): numbers compare by value; thousands separators are parsed whole; Hz claims and CFO-attribution flags are listed. Tests are in `test_answer_checker.py` (4).
    - The Amendment 11 bank re-scored under v2 (descriptive; the A11 verdict is unchanged): fields 100%, numbers 98.7% (4 unsupported: the three Hz conversions plus "120,000,000"), 6 Hz claims, 2 CFO-to-bin attributions.
  - **P12.2 encoding guard:** `rfpu_v15/textio.utf8_stdout()`. The Amendment 11 crash was actually in `run.py`, whose echo of the child log printed to a cp1252 pipe; the guard is now installed there and in `stage2_chat.py`. `test_textio.py` (2) shows that a cp1252 stdout fails without the guard and works with it.
  - **P12.3 firmware v3:** per-capture sample-rate table (host data); `freq_hz` (exact) and `cfo_hz` (mulh) computed by the RV32 CPU; method notes on the DFT and CFO lines. **PASS** on the reference machine for all 16 Stage 2 captures (`firmware_v3_check.py`, `reports/firmware-v3-check.json`):
    - every `freq_hz` equals the host value;
    - every `cfo_hz` is within fs·2^-12;
    - the 6 captures with "fs=unknown" print no Hz fields.

    Firmware tests: 14 pass (5 new).
  - Kernels: `IIRBiquad(out_frac)` and `VXFir(scale)`, with tests.
  - Full CPU suite: 208 passed, 21 skipped (the skips are branch-only tests, run separately: VFIR 14/14).
  - `a12-battery` (P12.5 / P12.6) is running on the GPU.

- **`a12-battery` result (finished 10:03):**
  - **P12.5 PASS:** IIR `out_frac=4` is 133/133 token-exact. SQNR median 102.6, 5th percentile 90.5, minimum 80.0 dB (was 78.5 / 66.5 / 55.9).
  - **P12.6 FAIL, as pre-declared in 12a:** VFIR adaptive scale (guard pass + in-weights BlockMax + scaled pass on 123/133 recordings) has every pass token-exact and 0 saturated samples, but the minimum is 68.3 dB, below the 70 dB target. The median rose from 73.4 to 76.7 and the 5th percentile from 71.1 to 74.4. The lowest recordings are still the wideband DJI 5G ones; the floor is VFIR's per-tap rounding, which output scaling cannot remove. Report: `v15/reports/a12-battery.json`.
- The Amendment 11 bank files were renamed `*-bank-a11*.json` (references updated) before `stage2-bank-96-fwv3` (P12.4) was launched at 10:05.

- **P12.4 first attempt stopped (10:25) after 17/96 questions.** All 17 were circuit-exact, but 8/17 answers were truncated at `--max-new 500`: the longer v3 report counts toward the generation budget, and every `rf_analyze` answer was cut. Recorded as **Amendment 12b**. The file is kept as `*-bank-fwv3-max500-aborted.json` and not graded. Relaunched with `--max-new 1000`.

- **P12.4 result (`stage2-bank-96-fwv3-max1000`, finished 11:21):** **All gates pass** (checker v2): 96/96 circuit-exact (0 head mismatches), tool 100%, **asked fields 100%**, **numbers supported 98.8% (330/334)**, right tool 96/96, median 43 s (was 37: the longer report). No answer reached the 1000-token limit. Versus Amendment 11: **wrong Hz values 1 → 0** (22/22 Hz values correct, against 5/6; `hz_audit.py`) and **CFO attributed to the spectral bin 2 → 0**. The 4 unsupported numbers are all correct unit rescalings of in-weights values (4,947,600 Hz → "4.95 MHz"; 437,500 Hz → "437.5 kHz"; 70,198,244 Hz → "70.2 MHz"; 75,000,000 Hz → "75 MHz"); checker v2 does not rescale units. Under checker v1 the rate is 92.8%, because v1 splits "88,726" into two numbers; v2 is the declared P12.4 checker. Files: `stage2/fullgpu-chunked-stage2-bank.json`, `-checked-v2.json`, `hz-audit.jsonl`.
- **Amendment 12 summary:**
  - P12.1 pass; P12.2 pass; P12.3 pass; **P12.4 pass**; **P12.5 pass**;
  - **P12.6 FAIL**, pre-declared in 12a.

- **Supervisor technical report** written as a private document, "RFPU v1.5 Technical Report"  private). It covers background, the field table, what we built with a diagram, results, novelty, early-mover position, limits and sources. Project facts were checked online on 2026-11-05: GitHub stars Qwenputer 660, QwenPU 120, qwenGPU 117, qwenTPU 18, Percepta 243; Loom arXiv 2604.08816; MIT weight-space programming (VentureBeat, 2026-11-02).

- **Supervisor report as Word:** `v15/RFPU_v15_Technical_Report.docx` (4 pages, A4) plus a PDF copy; generator in `v15/paper/supervisor_report/build.js`.
  - Traction verified 2026-11-05 from live APIs:
    - **GitHub stars / forks:** Qwenputer 660/79, QwenPU 120/14, qwenGPU 117/19, qwenTPU 18/6, Percepta 243/50, Loom 7/2.
    - **Hacker News (points / comments):** Qwenputer 705/147, MIT "Astra … weights" 2,947/890, Percepta 330/130; QwenPU 2 points and qwenGPU 4 points.
    - **Hugging Face (likes / 30-day downloads):** Qwenputer 166/1,421; QwenPU 10/794.

- **Supervisor report v2:** 7 pages with 7 figures: the architecture, accuracy by dataset, FFT precision vs level, precision per tool, time per question, community traction, and the Hacker News trend over 2026. Generated by `v15/paper/supervisor_report/plots.py` and `build.js`.
  - The trend data comes from the HN Algolia API (5 Nov 2026), title-matched and stored in `hn_trend.json`. Before 28 Oct, the only notable post was Percepta (March, 330 points); after it came Qwenputer (705), MIT Astra (2,947) and a rebuttal post (1,029).
  - GitHub star timestamps need authentication, so they are not used.

## Open / pending (as of this entry)

See [../FUTURE.md](../FUTURE.md).

## 2026-11-06: Amendment 13 (regression fuzz, held-out Stage 2 set)

- **Amendment 13 pre-declared** in PROTOCOL.md before any run: P13.1 fuzz regression; P13.2-P13.5 a 48-question held-out Stage 2 set (32 free-wording answerable, 8 no-tool, 8 unanswerable). New `HELDOUT` / `--suite heldout` in `stage2_chat.py`, new `heldout_checker.py` (smoke-tested on fabricated transcripts only, not graded). `neural_fuzz.py` takes an optional output filename so the 2026-11-03 record is not overwritten.
- **P13.1 PASS:** `neural-fuzz-a13.json`, same seed 15151, 40/40 cases token-exact with outputs equal to the reference, 28 kernel configurations, 10.66M tokens, 727 s. Same kernels and configurations as the 2026-11-03 run; total tokens 12.5% lower. The 13 cases whose token count changed are all FFT / Welch / CFAR / AM-envelope, i.e. exactly the kernels shortened by Amendment 5 (bit-identical rewrites), which postdate the original fuzz run. Every other case has an identical token count.
- **Held-out Stage 2 set run** (`a13-stage2-heldout`; `fullgpu-chunked-stage2-heldout.json`, graded `-heldout-checked.json`). 48 questions, `--chunk 1024 --max-new 1000`, about 33 min.
  - **P13.2 PASS:** 40/40 questions that ran the computer, 0 lock errors, 0 head mismatches.
  - **P13.3 FAIL (numbers):** tool 32/32, right tool 32/32 (16 `rf_levels` on level questions, 16 `rf_analyze` on spectral), fields 44/44 = 100%, but numbers supported 89.3% (117/131) < 95%. Manual reading: all 14 unsupported numbers are Qwen's own derived values and all are arithmetically correct (peak-minus-floor differences 1.1 / 9.9 dB, 3/16 = 18.75%, 1.74% of fs, 75 MHz = 0.375 × 200 MS/s, -23.4 MHz, fc + CFO = 5,240,088,726 Hz, sample rates written as 1,000,000 / 120,000,000 / 200,000,000). One wrong unit: "1.1 dBFS" for a dB difference. The free wording ("the gap between", "how far above", "what fraction") invited arithmetic, which checker v2 cannot verify. Median 41 s per answerable question.
  - **P13.4 PASS:** 8/8 no-tool questions answered without a tool. Manual reading: content broadly correct; one factual slip ("LoRa is a LoRaWAN protocol": LoRa is the physical layer, LoRaWAN the MAC).
  - **P13.5 FAIL:** declined 4/8 (spreading factor, BER, angle of arrival, modulation/coding: correct). The other 4 gave confident wrong answers built on exact numbers: occupied bandwidth "0.0000 MHz" (from the dominant bin), "0 separate transmitters" (from 0 active windows), symbol rate "1 MS/s" (the sample rate), and drone distance "approximately 100 meters" from a path-loss calculation treating -12.3 dBFS as dBm (the model even writes "Wait, let me recalculate" and continues). The 45 unsupported numbers in these 8 answers are mostly list numbering plus the fabricated path-loss figures.
  - **Reading:** exact computation guarantees the numbers, not the conclusions. This is the strongest evidence yet for the stated limitation "interpretation is not verified". Not re-run with a changed prompt; any fix (e.g. firmware printing what it does not measure, or a system-prompt rule) must be a new pre-declared amendment.
  - Tests: `test_heldout_checker.py` 2 passed.

## 2026-11-06: Amendment 14 (fixes for the Amendment 13 failures)

- User asked to fix the errors, "modulation either or". Chosen: the tool states it does not measure modulation (the classifier-label option stays parked).
- **Firmware v4** (`firmware.py`: `NOT_MEASURED`, `source(..., not_measured=True)`): both reports end with an in-weights-printed `not_measured=...` line. RV32 reference tests 17/17 (2 new). Cost on capture 0: `rf_analyze` 100,336 -> 100,991 tokens, `rf_levels` 28,159 -> 28,960. First edit hit the heredoc newline-escaping pitfall again (fixed with the Edit tool); one new test initially failed because "bit error rate" contains "error" (test fixed, not firmware).
- **Answer rule** in the system prompt (`--firmware 4`, default); `--firmware 3` rebuilds the exact v3 prompt and firmware. `--tag` keeps earlier output files.
- **Checker v3** (`answer_checker.py`), developed on the A13 transcripts:
  - first version (products, ratios, all derivations for all numbers) was far too permissive: 31/42 derived numbers still passed after a x1.07 perturbation, and it accepted the fabricated drone path-loss figures. Rejected;
  - second (no products or plain ratios): still 29/42, but the perturbation test itself was flawed (x1.07 of a small integer rounds back to the same integer);
  - final: rescalings for all numbers; differences / percentages of distinct values only for numbers with a decimal point or >= 3 significant digits; trivial perturbations excluded. A13: 4/17 perturbed pass (1/12 on answerable), fabricated path-loss numbers stay unsupported, A13 answerable numbers 99.2% (130/131; the remaining one is fc + CFO = 5,240,088,726 Hz), P12.4 bank 334/334. Descriptive only; A13 verdicts unchanged.
- **Amendment 14 pre-declared** (PROTOCOL.md) before any run: 36 new questions (16 unanswerable: 8 listed on not_measured, 8 unlisted; 16 answerable; 4 no-tool), gates P14.2-P14.5; A13 re-run under v4 descriptive only.
- Tests: firmware + checkers 24 passed.
- **A14 new set run** (`a14-stage2-new`; `fullgpu-chunked-stage2-a14.json`, graded `-a14-checked-v3.json`). 36 questions, firmware v4 + answer rule.
  - **P14.2 PASS:** 34/34 questions that ran the computer exact (2 unanswerable questions were declined without calling a tool).
  - **P14.3 FAIL (strict):** declined **16/16** (8/8 listed on not_measured, **8/8 unlisted**, e.g. battery level, IQ imbalance, polarization, phase noise; A13 was 4/8). It fails only on the zero-invented-numbers clause: the RSSI-in-dBm answer declines correctly, then adds a hypothetical example ("if the ADC's full-scale power is -100 dBm, RSSI ≈ -114.1 dBm") and calls -100/-120 dBm a "typical" reference. That answer also hit the 1000-token budget at its last sentence.
  - **P14.4 FAIL:** tool 16/16, fields 91.3% (21/23), numbers 93.3% (70/75) < 95%. Two genuine errors: capture 3 report `active_windows=0/16`, answer "16 active windows out of 64 → 25%" (misread, primed by "64-sample windows" in the question); capture 12 `freq_hz=-22500000`, answer "22.5 MHz" (sign dropped). Three unsupported numbers are checker artefacts (the constants 100 and 1,000,000 in a shown conversion). Derived numbers accepted by v3: 15 (5/14 still pass a ×1.07 perturbation).
  - **P14.5 PASS:** 4/4 no tool.
  - **Timing not comparable:** chip time median 37 s vs 16 s in A13 for +2.5% tokens and text 76 vs 55 ms/token: the laptop GPU ran in a lower-power state. Not used for any timing claim.
  - **Reading:** the not_measured line + answer rule fix the main failure (abstention 4/8 → 16/16, and it generalises to quantities not on the list). It does not fix reading errors on exact numbers (a misread fraction, a dropped sign), which a checker catches but the model does not avoid.
- A13 set re-run under firmware v4 launched (descriptive, `--tag fwv4`). Launched with `nohup … &` by mistake instead of a tracked background job; a results.md watcher reports completion.
- **A13 set re-run under firmware v4** (`a14-stage2-a13-rerun-fwv4`, `fullgpu-chunked-stage2-heldout-fwv4*.json`; descriptive, the fix was written after reading these failures). 48 questions, about 42 min. Exact 40/40; tool 32/32 (right tool 32/32); fields 100%; numbers 115/115 under v3 (89.6% under v2: all 12 are derived values); no-tool 8/8; unanswerable declined 8/8 with 0 unsupported numbers (was 4/8 and 45 numbers). The drone distance, symbol-rate, bandwidth answers now say "does not measure". Manual reading: the transmitter-count answer quotes the not_measured line but still ends "0 active transmitters detected" (a hedge the regex counts as a decline; A14 capture 7 is similar). Median 55 s per answerable question (GPU state differs between runs; not a timing claim).
- Full CPU tests after Amendment 14: 214 passed, 21 skipped.

### 2026-11-06: Amendment 15, blind modulation classifier in weights (user: "show me 25 IQ modulations, output from the Qwen system")

- **No modulation labels in our data.** Our recordings carry technology labels (Wi-Fi, LTE, DVB-T, LoRa…), not modulation labels, so this experiment uses synthetic captures with known ground truth (`rfpu_v15/modgen.py`). The classes are the 24 of RadioML 2018 plus LoRa CSS.
- **Built `rfpu_v15/cpu/modclass.py`:** `ModFeatures` (30 integer features, about 405–421k tokens per capture) and `IntMLP` (30-32-25 integer network). Both match their independent references on the RV32 reference machine on the first build.
  - I cut the instantaneous-frequency CORDIC from 16 to 10 iterations before any data run, which reduced the cost from about 545k to 410k tokens.
  - Tests: `test_modclass.py`, 3/3.
- **Training (host, CPU reference features):** train seed 1500, 600 per class.
  - Integer model: dev 89.2% (float 88.0%), against 4% chance.
  - Weak classes on dev: 256QAM 3/10, 128QAM 5/10, 128APSK 6/10, 64QAM 7/10, 64APSK 7/10.
  - File: `reports/modclass-model.json`.
- **Pre-declared Amendment 15** (P15.1 exact 250/250, P15.2 accuracy ≥ 80%, P15.3 ≥ 20/25 classes ≥ 7/10) before the test set (seed 1515) was generated.
- **Launched `modclass-neural`:** 250 captures, about 103M tokens, about 2 h, one GPU job.
- **Firmware `rf_modulation` (Amendment 15b, built while the run was going; CPU-only):**
  - It reuses the unchanged kernel programs inline, via `modclass.subroutine` (labels prefixed, exit replaced by a jump). The heredoc `\b` pitfall hit again and was fixed.
  - On the reference machine, the printed class, runner-up and margin equal the host model (4 captures, about 465k tokens per call). One of them was 64QAM printed as 256QAM: the firmware is exact, the classification is wrong.
  - Tests: firmware 21/21 including the new test.
  - Added `--suite modulation` to `stage2_chat.py` (10 classes drawn at random, seed 1516). P15.4 and P15.5 pre-declared.
- **Result `modclass-neural` (2026-11-06, 4825 s, one GPU job):** 250 test captures (seed 1515), every token checked.
  - P15.1 PASS: features 250/250 exact, MLP 250/250 exact; 100.6M + 5.5M tokens (neural 4369 s for features).
  - P15.2 PASS: accuracy 86.4% (216/250; chance 4%). By SNR: 10 dB 68%, 12 76%, 14 84%, 16 84%, 18 96%, 20 88%, 22 100%, 24 92%, 26 88%, 28 88%.
  - P15.3 PASS: 21/25 classes ≥ 7/10. Below: 32QAM 5, 64QAM 5, 128QAM 5, 256QAM 6. 18 classes ≥ 9/10.
  - Errors are order confusions within a family: 256QAM→64QAM 4, 32QAM→128QAM 3, 16PSK↔32PSK 3/2, 64QAM→16/256QAM, APSK/QAM high orders. Only cross-family slips: AM-SSB-WC→AM-DSB-WC 1, GMSK→FM 1.
  - Caveat: synthetic in-distribution generator, known SPS, no CFO; accuracy is not a real-world AMC claim.
- **`stage2-modulation` first launch crashed at startup** (NameError: the modulation setup block was missing from `stage2_chat.py`; no GPU work done). Fixed; CPU dry run assembles. Host model on the 10 demo captures (descriptive): 8/10 equal truth (32QAM→64QAM, 64QAM→16QAM). Relaunched.
- **Result `stage2-modulation` (Amendment 15b, full GPU 19 GB merge, chunked C=1024):** 10 questions, 80–122 s each, 446–469k chip tokens per call.
  - P15.4 PASS: 10/10 every chip token equal to the reference lock, 0 head mismatches; printed class/runner-up/margin = host model 10/10.
  - P15.5 PASS: 10/10 Qwen called `rf_modulation` (only that tool) and named the printed class; runner-up and margin quoted correctly when asked.
  - Truth (descriptive): 8/10. Capture 4 (32QAM) printed 64QAM, capture 5 (64QAM) printed 16QAM; Qwen faithfully reported the wrong classes (exact computation, wrong classification).
  - Caveat: on Q2 Qwen called margin 1604 "very confident"; the margin is an uncalibrated integer score, so that wording is unsupported.
- Full CPU test suite after Amendment 15: 218 passed, 21 skipped (branch-only).
- Reports updated with Amendment 15 results: README (headline rows, limitation, tests 218), STAGE2.md, DSP_METRICS.md, FUTURE.md (ranked accuracy fixes).
- Published reports updated with Amendment 15: supervisor doc 'RFPU v1.5 Technical Report' (2 result rows + 1 limit) and 'RFPU Radio Intelligence' page v4 (86% KPI, accuracy-by-SNR chart, limit, status 6 Nov). Ranked accuracy fixes parked in FUTURE.md §4.
- 2026-11-12: supervisor brief PDF (v15/reports/supervisor/RFPU_v15_Supervisor_Brief.pdf, 5 pages): concept/history of in-weight computers, project origin, results, testing method and failures, why not yet a main-track paper, why release online. Novelty re-check same day (NOVELTY_SEARCH.md addendum): still no in-weight DSP/RF work; 40+ Qwenputer derivatives since 2026-11-03 incl. domain units (QwenQPU) and 9B/27B/Nemotron hosts, so the 'only 4B' wording stays qualified.
- 2026-11-12: supervisor brief: added subsection 'Many people are building on the Qwen system, and radio is still ours' (40+ derivatives by direction; radio only RFPU; honest wording notes).
- 2026-11-12: supervisor PDF rewritten as a 4-page technical brief (generator v15/reports/supervisor/make_brief.py): architecture diagram, software stack, 8 charts, what we proved, stress tests, positioning; paper-publishability section removed at user request.
- 2026-11-12: brief v3: 'What that sentence means' box, 'How it started' timeline (Qwenputer 19 Oct, QwenPU 31 Oct, our v1.0 2 Nov → v1.5), results explained in DSP/AI and plain terms, section 7 'What this can become' (verifiable RF intelligence). 6 pages.
- 2026-11-12: **Internals visualizer spike (throwaway, `v15/viz/`).** `record.py` = ChunkedRvChat on rv1 (full GPU, C=1024, reference lock on) with the chip CUDA graph disabled so forward hooks fire; records every layer's MLP activations (down_proj input), per-value-head output norms and residual norms, plus the circuit's next-token code at every position; chip units/heads named from the compiled IR (emit order, `allocate`), IR dataflow edges in graph.json. `serve.py` + `index.html` = local viewer (43 layer columns, every neuron a pixel, heads as dots, live IR edges between active nodes, hover names, pinned neuron sparklines, timeline). Run: LEVELS[0] (capture 14) → Qwen called rf_levels, 625 text + 26,581 chip tokens, 106 s (hooks, no graph, no fla), lock error None, answer "-48.9 dBFS / -10.4 dBFS" (correct). Data $RFPU_DATA/viz/levels-q0 (3.3 GB + int8 cache). Own check: generated positions 27,207, circuit-owned 27,144, equal to emitted 27,144, mismatch 0. Caveats/failures: (1) first two runs crashed at save (off-by-one when run hits max_new; zero-unit chip layer C3) and the first used max_new 400 which cut the answer; (2) circuit fires at 2 prompt positions (literal `</tool_call>` in tool instructions; prompt tokens are given, so unchecked: status 2); (3) exit-layer ballast-clear units overflow fp16 (1.74M inf in C11 MLP) - a storage artifact, clamped for display; recorder now keeps chip MLP in fp32; (4) Qwen neurons have no names; Qwen-layer "flow" particles are decorative residual-stream markers, only chip-layer edges are real IR dataflow; (5) frame serving ~0.2 s per 32-token batch.

## 2026-11-14: disk cleanup of $RFPU_DATA (regenerable items deleted, user request)

Deleted ~71 GB (D: 52 GB free -> 123 GB free):
- `cache/uv`, `cache/wheels`, `cache/triton`, `cache/huggingface`, `cache/tools` (package/build caches).
- `viz/levels-q0` (visualizer recording; re-record with `v15/viz/record.py`, ~2 min GPU).
- `checkpoints/qwen3.5-2b` (only used for the failed 2B merge; failure already logged).
- `checkpoints/rfpu-v15/qwen3.5-4b-vx-rf` (27 GB) and `qwen3.5-4b-vx-vfir` (27 GB): rebuildable with
  `chipformer.rv.merge --vx` from `checkpoints/qwen3.5-4b` at branch commits b72623e / c5d55c1 (see REPRODUCE.md).
  Their provenance (shard SHA256), chipformer.json and config.json were copied first to
  `v15/reports/checkpoint-provenance/`, so a rebuild can be checked bit-for-bit. Recorded results are unchanged.
Kept: rv1 merge, QwenPU-published, qwen3.5-4b base, all datasets, environments, v11-v14 checkpoints, releases,
`rfpu-local-chip` (v1.0, only 1/5 shards present; left untouched).

## 2026-11-20: D: drive lost; copy at $RFPU_DATA verified

After a system-wide issue, D: is empty (2.9 GB used, no RFPU-data). A copy exists at `$RFPU_DATA` (created
2026-11-18 13:25-13:30), matching the post-cleanup state of 2026-11-14. Verified by exact path against recorded hashes:
- Datasets: camras 4/4, IDLab 190/190, Tampere 25/25 (sha256); rfsr 416 records, all 320 files on disk OK (md5 + size);
  LoRa IQ: all 372 extracted files OK (size + crc32 vs archive index; only a subset was ever extracted).
- QwenPU-published 17/17 (HF LFS sha256 + size). Qwen3.5-4B base: all files match the HF API at 851bf6e (sha256/size).
- v12/v14 checkpoints (10) and bitwise-I16-W64-v1: model.safetensors sha256 matches construction.json.
- releases/rfpu-v14 zip sha256 matches.
- NOT hash-verifiable: `rfpu-v15/qwen3.5-4b-rv1` (no rfpu-provenance.json exists, although REPRODUCE.md says so; no shard
  hashes recorded anywhere). Needs a functional check (GPU run) or a rebuild from the base model.
- Code still hardcodes `$RFPU_DATA` in 9 v15 files; must be made configurable.

## 2026-11-20: data path setting, rv1 functional check, doc audit corrections (public-release prep)

- **`RFPU_DATA`** (new `v15/rfpu_v15/paths.py`): replaces the hard-coded `$RFPU_DATA` in 9 files; recorded paths in
  selections/manifests are mapped with `paths.local()`. Default: first existing of `<repo>/rfpu-data`, `~/RFPU-data`,
  `$RFPU_DATA`, `$RFPU_DATA`. Full CPU suite after the change: **218 passed, 21 skipped** (unchanged).
- **rv1 functional check** (`viz-record-levels-q0-rv1check`, record.py on `$RFPU_DATA/.../qwen3.5-4b-rv1`): Qwen called
  `rf_levels` on capture 14, 625 text + 26,581 chip tokens, 101 s; **27,144/27,144 circuit tokens equal to emitted,
  0 mismatches**; identical counts to the 2026-11-12 run; answer −48.9 / −10.4 dBFS (correct). The copy's weights behave
  identically. Shard SHA-256s computed now for the Hugging Face upload.
- **Doc audit** (read-only reviewer, each item re-checked against the evidence JSON) and corrections:
  README pipeline evidence link pointed to `pipeline.json`, which the 2026-11-04 synthetic regression overwrote; the
  14-capture run is `pipeline-20261103.json`. Modulation classes ≥ 9/10 are **16**, not 18 (also stated wrongly in the
  Amendment 15 entry above). Test counts unified (218/21; test_vfir 14). Executor throughput labels (FIR16 7.4k→33k,
  4.5×; 4.4k is the FIR64 verification workload). REPRODUCE asset table (rv1 had no hash file; VX merges deleted
  2026-11-14). VX_RF_INSTRUCTIONS marked superseded, funct3 mapping as built (VPOW 1, VBFLY 2), VPOW checkpoint result.
  FUTURE items already done. Stale status date. results.md header links. Superseded limitations annotated. Paper draft:
  scope note, executor speed ratio, CPU block FFT range 74.6–75.2, checker perturbation 1/12 (dev) and 5/14 (A14).
  STAGE2 profile is capture 14, not 1. Not changed: gates.json (append-only ledger; its early scope notes are superseded
  by later entries in the same file).

## 2026-11-20: public release on GitHub (pratx1906/QWENPU-RF, commit a402288)

- Built by `v15/release/export_public.py` into `Desktop/QWENPU-RF` (fresh git history, author = GitHub noreply address,
  no co-author lines): `v15/` as is, minus `prompt-v15.txt`, the .docx, `paper/supervisor_report`, `reports/supervisor`,
  `results/raw/*.source.zip`; machine paths rewritten (`<repo>`, `$RFPU_DATA`), private claude.ai links removed.
  Inputs that lived outside v15 moved to `data/` (gate selection, IDLab manifest, per-dataset manifests).
- Files > 25 MB (6 Stage 2 transcripts, LoRaIQ archive index; 469 MB) listed in `data/LARGE_FILES.json` with SHA-256,
  for the Hugging Face release; each transcript has a `-summary.json` (raw token list replaced by length + SHA-256).
- `upstream/rfpu-vx.patch`: the rfpu-vfir branch as one diff on QwenPU 5af1fc0; checked to apply cleanly and reproduce
  the branch (CRLF aside). `upstream/setup_upstream.py` fetches and patches.
- Viewer: new `&poster` theme (dark, bold) used for `docs/demo.gif` (22 real frames of the 2026-11-20 rf_levels run).
  Figures regenerated from evidence by `docs/make_figures.py`.
- Checks on the public tree: 218 passed / 21 skipped; 0 broken relative links; 0 leftover local paths.
