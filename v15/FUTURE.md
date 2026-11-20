# RFPU v1.5: future work (parked 2026-11-04, updated 2026-11-20)

Every idea, fix and pending item not done yet, in one place. Each entry gives its value, a rough effort, and whether this laptop is enough. Current results are in [README.md](README.md). Every run so far is in [research/RESEARCH_LOG.md](research/RESEARCH_LOG.md).

Rule for anything picked up from here: declare the gate in [research/PROTOCOL.md](research/PROTOCOL.md) before running it, record failures, and log every run.

---

## 1. Fixes for the stated limitations

### 1a. "Exact" means exact to a 16-bit fixed-point contract

| Item | Gain | Effort |
|---|---|---|
| Reword the claim everywhere to "bit-exact to a declared fixed-point contract, with a measured error budget" | Accurate framing, matching how DSP hardware is described | Minutes |
| Formal worst-case error bounds per kernel (proved, then tested at the extremes) | "Never off by more than X" in the paper | ~1 day |
| Higher-precision mode: 32-bit samples and coefficients for FIR, FFT and NCO (Q31, 64-bit accumulation) | Rounding error ~1e-4 → ~1e-9 relative; costs 2–4× tokens | 1–2 days |
| A general precision-selection pass (pick 32- or 64-bit accumulation per kernel from range proofs, as FIR already does) | Precision and cost chosen automatically | ~1 day |

### 1b. The AI's interpretation is not verified, only its numbers

| Item | Gain | Effort |
|---|---|---|
| Firmware reports conclusions too: deterministic labels such as signal type (the in-weights classifier), bursty/continuous and narrowband/wideband. Could be an `rf_classify` tool using the 81-feature two-stage classifier | Qwen repeats verified labels instead of guessing | ~½ day |
| ~~Automatic answer checker~~ **Done:** `answer_checker.py` (v2: value comparison, thousands separators); labels are not covered yet | — | — |
| Interpretation test set with known answers (synthetic plus labelled real recordings) | Published interpretation accuracy | ~½ day |
| ~~Abstention~~ **Done (Amendment 14):** declined 16/16 on new questions. Remaining: a reading check for misread exact numbers (0/16 read as 16/64, dropped sign), e.g. the host re-checks every quoted field and asks the model to correct; and hypothetical examples after a decline. Original item: **(from Amendment 13 P13.5, 4/8 declined):** firmware prints a "not measured: bandwidth, modulation, symbol rate, distance, direction, BER" line, and/or a system-prompt rule; re-test on a *new* pre-declared unanswerable set (not the A13 8, which are now seen) | Qwen says "not measured" instead of inventing | ~2 h + 30 min GPU |
| ~~Checker v3~~ **Done (Amendment 14).** Was: verify simple derived values (differences, ratios, unit rescalings of report values) so correct arithmetic is not counted unsupported (A13 P13.3: 14/14 correct) | Fair number-fidelity score on free wording | ~1 h |

### 1c. Modest evidence base

| Item | Gain | Effort / needs |
|---|---|---|
| ~~Real-IQ screen grown to ~100+ captures~~ **Done (Amendments 9–10):** 133 recordings, 132/133; pipeline and 10-tool battery exact on all 133 | — | — |
| Synthetic Monte-Carlo with known truth: thousands of signals with set frequency, level, burst position and SNR | Detection and estimation accuracy with error bars | ~½ day |
| ~~Stage 2 conversation bank, auto-graded~~ **Done (Amendments 11–12):** 96 questions, answer checker v1/v2, firmware v3 re-run | — | — |
| Classifier: more recordings and locations, especially 10 MS/s DVB-T (only 5 recordings now) | 96.2% currently indicative | Needs more data, or a download |
| Timing replication on a second machine or on different days | Removes "one laptop" | **Needs another GPU (cloud or a colleague's)** |

---

## 2. Pending from the current campaign

- ~~P8.3 neural fuzz regression~~: done as Amendment 13 P13.1, 40/40 exact on the final code.
- ~~Synthetic pipeline regression~~: PASS, 8.33M tokens (`regress-pipeline-synthetic`).
- **Paper:**
  - figures: precision vs level, speed ladder, verification ladder, chunk-size curve, layer map;
  - LaTeX conversion (template `v14/paper/`);
  - final read-through;
  - fold in Amendment 15 (modulation classifier, `rf_modulation`); Amendment 8 is already in.
- **Docs:** keep README and the plain-language overview in sync when numbers change.

---

## 3. Systems and speed

| Idea | Expected gain | Notes |
|---|---|---|
| Vector-unit checkpoint that fits one 24 GB GPU together with Qwen (smaller VX, or offload Qwen's text layers) | Stage 2 could use VFIR, VPOW and VBFLY | The VX merge is 27 GB |
| Streaming (weight-stationary) VFIR: taps and state stay in VM across chunks | ~1.3–1.5× more on FIR | Best idea from the qwenTPU study |
| More RF instructions: complex multiply / NCO mix, CORDIC step, CFAR window | Faster demodulation and detection | Same verification ladder as VFIR / VBFLY |
| VBFLY improvements: track without the second word copy (if a shallower comparison is found); load and read out samples through the vector unit | VBFLY currently 1.8×; the butterflies alone are ~7× | Depth must stay at 13 layers |
| Chunked verification inside the executor path (published 26.7 GB checkpoint) | Faster Stage 2 on the VX model | Executor is free-running; would need a teacher-forced mode |
| Install `causal_conv1d` in the FLA environment | Faster chunked verification (fallback kernel now) | Separate environment only |
| Stage 2 chunked verification with flash-linear-attention while keeping Qwen's text identical (run Qwen's text layers on torch kernels) | 11–28 s computer time with identical wording | Small engineering task |
| Firmware: fewer DFT frames, a 32-bit CFO accumulation with a proven bound, more `rf_*` tool modes | 1.3–2× per call | Contract change; host model must follow |
| Cheaper rounding in scalar kernels; loop unrolling and fusion of hot loops (FIR, CORDIC) | 10–30% | Same bit-identical approach as the FFT rewrite |
| Production mode (no readback) in more places | ~1.4% | Low value |
| Multi-token prediction (MTP) for chip tokens | — | Evaluated and rejected: low value |

---

## 4. Precision and accuracy follow-ups

- VFIR with full exact accumulation (wider VM state) to close the last 3.4 dB to the exact FIR (77.9 vs 81.3 dB).
- VBFLY precision closer to the CPU block FFT (70 vs 75 dB): a finer scaling rule or a wider intermediate.
- IIR `wide` mode as the default where downstream kernels accept int32.
- VX int8 energy: retire it or add an int16 path (exact scalar energy is the precision path now).
- Classifier ideas not tried yet:
  - features at 10 MS/s beyond bandwidth (spectral flatness, frame-power statistics);
  - an in-weights two-stage `rf_classify` tool (see 1b).
- IIR low tail: **done** (`out_frac=4`, minimum 80.0 dB, P12.5). VFIR low tail: output scaling only gives +3 dB (68.3 dB, P12.6 FAIL), so it needs exact accumulation inside VFIR (circuit change; see the first bullet).
- Cyclic-prefix lag features are on record as no improvement; don't retry without a new reason.

- **Modulation classifier (Amendment 15, 86.4%; weak high-order QAM).** Ranked fixes:
  1. Decision-directed QAM features after blind phase/gain recovery: grid-snap residual (EVM) for each candidate order 16/32/64/128/256, cross vs square corner occupancy. Targets 32–256QAM, which is most of the error.
  2. Longer observation (2048–4096 samples → 256–512 symbols) for the same features; halves the variance of the M-th power and ring statistics.
  3. Finer amplitude histogram (32–64 rings) and moments C40/C42/C61/C63; PSK 16 vs 32 via M=16 vs M=32 ratio with more symbols.
  4. Hierarchical classifier: family first (ASK/PSK/APSK/QAM/analog/CSS), then an order head per family; or a larger MLP (64 hidden) with more training data at low SNR.
  5. Calibration: map score margin to probability (temperature on the dev set) so Qwen can state confidence honestly; abstain ("QAM, order uncertain") below a margin threshold.
  6. Robustness before any real-world claim: CFO, unknown symbol rate (cyclostationary estimate), multipath, real labelled data (RadioML 2018 itself or our own SDR captures).

---

## 5. Research directions

- **Larger models:** QwenPU's circuit needs 2 value heads per key head (only Qwen3.5-4B among the small sizes). qwenGPU-style compilation reaches 2B; bigger models need a different construction.
- **Formal guarantees** beyond token-by-token checking: prove kernel programs correct against their references.
- **Exact plus approximate reasoning:** the model chooses which analysis to run, then reasons over verified results; measure how verified labels change answer quality.
- **Other sensors:** audio, vibration, biomedical, industrial: the same kernel library and verification.
- **Remaining ablations:** A2/A3 are N/A on this substrate; A7 model width and depth is blocked (only one host size).

---

## 6. Applications to explore

- Explainable spectrum monitoring ("what is on the air here?" with provable numbers).
- Auditable drone and interference detection, with a verifiable trail behind every alert.
- Teaching and certification tools that show exactly how each number was computed.
- Regulator and emergency-service decision support.
