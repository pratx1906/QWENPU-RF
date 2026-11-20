# Stage 2: Qwen calls RF analysis on its own in-weights CPU

Model: the published QwenPU checkpoint (Qwen3.5-4B's layers plus 13 compiled chip layers in one file; all 11 shards SHA256-verified against the publisher, revision `0ea39d7`, [stage2-checkpoint.json](stage2-checkpoint.json)).
Driver: [stage2_chat.py](../scripts/stage2_chat.py), which uses chipformer's `RvChat` in executor mode.

**How it works**
- **Qwen decides to call the tool.** It writes a `<tool_call>` to `rf_analyze`; the bridge captures that text into the CPU's RAM.
- **The firmware runs inside the same model.** [firmware.py](../rfpu_v15/stage2/firmware.py) runs free-running on the in-weights RV32 CPU and analyses one of 16 IQ captures stored in its read-only data memory (14 real, 2 synthetic with known truth).
- **The report comes back as Qwen tokens.** Its printed output is echoed into Qwen's context as `<tool_response>`, and Qwen answers from it.
- **Every computer token is verified.** Each one is checked live against chipformer's RV32 reference machine; this is RvChat's lock, and a mismatch would stop generation.

**Deviations from a single full-GPU forward pass (26.7 GB does not fit 24 GB)**
- Chip tokens run on QwenPU's Triton executor, which is the same circuit as the chip layers; that equivalence is checked by the 14/14 result in qwen-checkpoint-final.
- Qwen's 32 layers and its 248k vocabulary rows run as the saved HF modules on the GPU.
- The prompt is prefilled token by token through the executor and Qwen's layers.
- Logits cover Qwen's vocabulary only, so RvChat's lm_head head-check is not available.

| Question | Capture | Text tokens | Computer tokens | Seconds | Computer tokens equal reference |
|---|---:|---:|---:|---:|---|
| Capture 14 is a stored IQ recording. Use the RF analyzer: is there a b… | 14 | 605 | 252,647 | 239 | yes |
| Analyze capture 12 (a 2.4 GHz drone controller recording). What is the… | 12 | 712 | 263,266 | 255 | yes |
| Run the RF analyzer on capture 8 and summarize what kind of signal act… | 8 | 580 | 263,496 | 222 | yes |

Every field of every report Qwen received matches the independent host model of the same analysis ([test_stage2_firmware.py](../tests/test_stage2_firmware.py): 0.1 dB display, ±0.07 dB, exact counts and bins, CFO ±2e-4).

## Run 1: capture 14

**User:** Capture 14 is a stored IQ recording. Use the RF analyzer: is there a burst in it, roughly which samples does it occupy, how far above the noise floor is it, and what is its frequency in cycles per sample?

```text
<tool_call>
<function=rf_analyze>
<parameter=capture>
14
</parameter>
</function>
</tool_call>
<tool_response>
capture=14 SYNTHETIC tone burst (truth: samples 320-640, 0.19 cycles/sample) fs=1MS/s
samples=1024 (Q15)
mean_power_dbfs=-15.5
peak_power_dbfs=-10.2 at_sample=535
noise_floor_dbfs=-48.9 (quietest 64-sample window)
loudest_window_dbfs=-10.4
active_windows=5/16 (power > 4x quietest) first_active_sample=320 last_active_sample_end=640
dominant_dft16_bin=3 freq_cycles_per_sample=0.1875 power_share_percent=99
cfo_cycles_per_sample=0.1899
</tool_response>
<think>

</think>

Based on the analysis of capture 14:
*   **Burst:** Yes, there is a burst.
*   **Location:** It occupies samples **320 to 640**.
*   **Power:** The peak power is **-10.2 dBFS**, which is roughly **37.7 dB** above the noise floor (-48.9 dBFS).
*   **Frequency:** The dominant spectral bin indicates a frequency of **0.1875 cycles per sample**. The carrier frequency offset (CFO) is measured at **0.1899 cycles per sample**.
```

## Run 2: capture 12

**User:** Analyze capture 12 (a 2.4 GHz drone controller recording). What is the signal level relative to the noise floor, and what is the carrier frequency offset in kHz? Show the conversion.

```text
<tool_call>
<function=rf_analyze>
<parameter=capture>
12
</parameter>
</function>
</tool_call>
<tool_response>
capture=12 Tampere drone-controller fs=120MS/s fc=2440MHz
samples=1024 (Q15)
mean_power_dbfs=-14.1
peak_power_dbfs=-6.9 at_sample=477
noise_floor_dbfs=-16.7 (quietest 64-sample window)
loudest_window_dbfs=-12.3
active_windows=0/16 (power > 4x quietest)
dominant_dft16_bin=13 freq_cycles_per_sample=-0.1875 power_share_percent=59
cfo_cycles_per_sample=-0.1950
</tool_response>
<think>

</think>

Based on the analysis of capture 12:

**Signal Level Relative to Noise Floor:**
*   **Peak Power:** -6.9 dBFS
*   **Noise Floor:** -16.7 dBFS
*   **Difference:** -6.9 dBFS - (-16.7 dBFS) = **9.8 dB**

The signal is approximately 9.8 dB above the noise floor.

**Carrier Frequency Offset (CFO):**
*   **CFO (cycles/sample):** -0.1950 cycles/sample
*   **Sample Rate:** 120 MS/s (120,000,000 samples/second)
*   **CFO (Hz):** -0.1950 cycles/sample × 120,000,000 samples/second = **-23,400,000 Hz**
*   **CFO (kHz):** -23,400,000 Hz / 1,000 = **-23,400 kHz**

The carrier frequency offset is **-23,400 kHz** (or -23.4 MHz).

*Note: The analysis indicates that no windows were considered "active" (0/16), meaning the signal power did not exceed 4x the quietest window. This suggests the signal might be intermittent or the threshold for "active" was too high for this specific capture.*
```

## Run 3: capture 8

**User:** Run the RF analyzer on capture 8 and summarize what kind of signal activity it shows.

```text
<tool_call>
<function=rf_analyze>
<parameter=capture>
8
</parameter>
</function>
</tool_call>
<tool_response>
capture=8 EPFL-LoRaIQ LoRa fs=unknown
samples=1024 (Q15)
mean_power_dbfs=-14.1
peak_power_dbfs=-10.4 at_sample=41
noise_floor_dbfs=-14.9 (quietest 64-sample window)
loudest_window_dbfs=-13.3
active_windows=0/16 (power > 4x quietest)
dominant_dft16_bin=1 freq_cycles_per_sample=0.0625 power_share_percent=24
cfo_cycles_per_sample=0.0038
</tool_response>
<think>

</think>

The RF analyzer on capture 8 (EPFL-LoRaIQ LoRa) shows the following signal characteristics:
- **Mean power**: -14.1 dBFS
- **Peak power**: -10.4 dBFS (at sample 41)
- **Noise floor**: -14.9 dBFS (quietest 64-sample window)
- **Active windows**: 0 out of 16 (no windows exceed 4x the noise floor)
- **Dominant spectral bin**: Bin 1 at 0.0625 cycles/sample with 24% power share
- **Carrier frequency offset (CFO)**: 0.0038 cycles/sample

The signal appears to be a weak LoRa transmission with very low activity (no active windows detected). The dominant spectral bin at 0.0625 cycles/sample suggests a narrowband signal, consistent with LoRa modulation. The small CFO indicates a minor frequency offset.
```

## Assessment

- **Exact numbers.** In all three runs Qwen's answers quote the report's numbers, and its arithmetic is correct: 9.8 dB above the floor; −0.1950 × 120 MS/s = −23.4 MHz.
- **Synthetic ground truth recovered.** On capture 14 Qwen reports the burst at samples 320–640, 37.7 dB above the floor, at 0.1875 cycles/sample (spectral bin) and 0.1899 (CFO); the truth is 320–640 at 0.19.
- **Interpretation is Qwen's own and can be wrong.** For the LoRa capture, Qwen calls the signal narrowband "consistent with LoRa", but LoRa is chirp spread-spectrum. The report itself is exact; the 4B model's domain reasoning is not verified.
- **Cost.** About 250–263k computer tokens per call, roughly 2 min on the executor, against about 25 s for Qwen's text tokens.

## Full model on one GPU (2026-11-04)

**The model.** QwenPU's own merge tool built `$RFPU_DATA/checkpoints/rfpu-v15/qwen3.5-4b-rv1`:
- Qwen3.5-4B (SHA-verified, revision `851bf6e`) plus the plain RV32 CPU, without VX.
- 11 chip layers, hidden size 3968, 19 GB on disk.

**How it ran.** Stock `RvChat` loads every layer onto the GPU, with a peak of 19.1 GB. Chip tokens run through the saved HF chip layers on the compiled fast path; the executor is not used.

**Two independent checks per run.**
- **Lock:** every chip token is compared with the RV32 reference.
- **Head check:** at every circuit-owned position, the model's own tied lm_head prediction is compared with the circuit's output code.

**Capture labels.** Synthetic labels no longer contain the ground truth (the earlier labels leaked it).

| Capture | Chip tokens | Seconds | Lock errors | Head mismatches |
|---:|---:|---:|---|---:|
| 14 | 252,375 | 715 | 0 | 0 |
| 12 | 263,288 | 734 | 0 | 0 |
| 8 | 263,510 | 713 | 0 | 0 |

**Answers.** Qwen recovers the hidden synthetic truth from the report alone: burst at samples 320–640, about 38 dB above the floor, 0.1875 cycles/sample (truth 0.19).
The drone answer (9.8 dB, −23.4 MHz) matches the earlier QwenPU-published run.

**Throughput.** About 400 chip tokens/s on the saved layers against about 2,000 on the executor.

**Architecture note.** Qwen3.5-0.8B and 2B cannot host *QwenPU's* CPU circuit unchanged: they have 16 value heads per 16 key heads, and the circuit needs 2 value heads per key head (the merge fails with an assertion). Only the 4B has that layout among the small models. qwenGPU shows that a different compiled circuit can target Qwen3.5-2B ([qwenGPU](https://github.com/vixhal-baraiya/qwenGPU)).

## Cheaper firmware re-runs (2026-11-04)

The firmware was rewritten: one merged pass for power, peak, window and lag-1 CFO; DFT16 on a pre-shifted frame; dB by log2 table, rounded to the nearest tenth. Cost per call fell from ~252–263k to ~93–100k chip tokens. Same questions, same captures.

| Capture | Chip tokens | Executor (26.7 GB published; chip on executor) s | Full GPU (19 GB merge) s | Lock errors | Head mismatches |
|---:|---:|---:|---:|---|---:|
| 14 | 92,702 | 121 | 271 | 0 | 0 |
| 12 | 100,002 | 163 | 296 | 0 | 0 |
| 8 | 100,127 | 158 | 299 | 0 | 0 |

Chip time per call: ~52–58 s on the executor, ~231–247 s on the saved layers (full GPU). Answers are unchanged in substance. Evidence: [stage2/stage2-chat.json](stage2/stage2-chat.json), [stage2/fullgpu-stage2-chat.json](stage2/fullgpu-stage2-chat.json); ledger runs `stage2-demo-fastfw`, `stage2-fullgpu-fastfw`.

## Where a call's tokens go (reference-machine profile, 2026-11-04)

One `rf_analyze` call on capture 14 (the synthetic 320–640 burst): 92,570 tokens.

| Part | Tokens | Share |
|---|---:|---:|
| Pass 1 per-sample loop (power, peak, 64-sample windows, exact 64-bit lag-1 CFO sum) | 57,149 | 61.7% |
| DFT16 over 8 frames | 31,100 | 33.6% |
| Parsing, dB/log2, printing, bin search | ~4,300 | ~4.7% |

Pass 1 is ~56 tokens/sample, of which the exact 64-bit complex CFO accumulation is about half. Levers: lazy analysis (skip the CFO and DFT when the question does not need them: up to ~2–3× on such calls), fewer DFT frames, or a 32-bit CFO accumulation with a proven bound (a contract change; the host model must follow). Not done yet; each needs a Stage 2 re-run.

## Chunked verification inside the chat and `rf_levels` (2026-11-04)

**Chunked verification** (`rfpu_v15/stage2/chunked_chat.py`, `stage2_chat.py --full-gpu ... --chunk 1024`). During a tool call the reference lock already knows the computer's next tokens; runs of chip-input tokens go through the saved chip layers 1,024 at a time and every position's predicted token must equal the reference's. Tokens that feed text (Qwen's own, the printed report) keep the one-token path. The model's own cache already holds the tool-call text, so no snapshot is needed. Audit mode: the reference drafts, the layers verify.

| Capture | One token at a time (s, chip s) | Chunked (s, chip s) | Tokens identical | Lock errors | Head mismatches |
|---:|---:|---:|---|---:|---:|
| 14 | 255, 223 | 47, 16.6 | yes (93,449) | 0 | 0 |
| 12 | 284, 242 | 54, 16.5 | yes (100,884) | 0 | 0 |
| 8 | 288, 247 | 53, 16.2 | yes (100,984) | 0 | 0 |

Same firmware and environment for both columns. With flash-linear-attention kernels (second environment) the chip part is 11–28 s, every circuit token is still exact, but Qwen's own answer wording changes slightly (its float layers use different kernels).

**`rf_levels`**: a second tool on the same firmware (mode read from the call text): power, peak, noise floor and windows only, ~28k instead of ~93k tokens, identical level numbers. With both tools offered:

| Question | Tool Qwen chose | Chip tokens | Seconds |
|---|---|---:|---:|
| Noise floor and loudest window of capture 14 | rf_levels | 25,769 | 31 |
| Mean and peak power of capture 12 | rf_levels | 28,541 | 27 |
| Burst frequency of capture 14 | rf_analyze | 92,913 | 48 |

All with 0 lock errors and 0 head mismatches; answers quote the report correctly.

## Real recordings (Amendment 7, 2026-11-04)

Six pre-declared questions about six real captures; both tools offered; fully on one GPU with chunked verification. Evidence: [stage2/fullgpu-chunked-stage2-real.json](stage2/fullgpu-chunked-stage2-real.json).

| Capture | Question asks for | Tool chosen | Chip tokens | s | Qwen's answer |
|---|---|---|---:|---:|---|
| 1 IDLab WiFi 1 MS/s | mean power, noise floor | rf_levels | 27,663 | 38 | −13.3 dBFS, −14.2 dBFS |
| 2 IDLab LTE 1 MS/s | peak power and index | rf_levels | 28,088 | 33 | −5.8 dBFS at sample 46 |
| 3 IDLab LTE 10 MS/s | dominant bin, frequency | rf_analyze | 97,214 | 42 | bin 8, −0.5000 cycles/sample |
| 4 IDLab DVB-T 1 MS/s | active windows, loudest window | rf_levels | 27,826 | 29 | 0/16, −14.2 dBFS |
| 9 EPFL LoRa | CFO | rf_analyze | 100,639 | 40 | 0.0666 cycles/sample |
| 13 Tampere drone 5.8 GHz | mean and peak power | rf_levels | 28,776 | 32 | −14.1 dBFS, −7.0 dBFS |

All six had 0 lock errors and 0 head mismatches. Every report field equals the independent host model within the firmware tolerances, and every asked number appears in the answer exactly as printed in the report.

## Question bank with an automatic answer checker (Amendment 11, 2026-11-05)

**Setup**
- **Questions:** 96 = 16 captures × 6 templates (mean and noise floor; peak and its index; active windows and loudest window; dominant bin and frequency; CFO; power share). Both tools offered. Full-GPU merge, `--chunk 1024`.
- **Grading:** `v15/scripts/answer_checker.py`, which is deterministic:
  - an asked field must be quoted verbatim from the verified report;
  - every number in the answer must be supported by the report (verbatim, in the question, or a rounding of a report value).

| Gate | Result | Target |
|---|---|---|
| P11.1: circuit tokens equal the reference | 96/96 | 100% |
| P11.2: tool called and answered | 96/96 | ≥ 95% |
| P11.3: asked fields quoted verbatim | 170/176 = 96.6% | ≥ 90% |
| P11.4: numbers supported | 308/311 = 99.0% | ≥ 95% |

**Descriptive**
- Tool choice was right on all 96: `rf_levels` 48/48 on level templates, `rf_analyze` 48/48 on spectral ones.
- Median 37 s per question.

**Misses**
- All 6 field misses are trailing zeros ("-0.5" for "-0.5000"); under checker v2 the rate is 100%.
- The 3 unsupported numbers are Qwen's own CFO→Hz conversions: 88 kHz ✓, −23.4 MHz ✓, and −62.6 "Hz" ✗ (should be kHz).
- Two answers attribute the CFO to the spectral bin. It is in fact the lag-1 autocorrelation phase, and the number checker cannot catch this.

**Run note:** the run crashed after 77 questions on a console-encoding error in `run.py`'s log echo ("≈"). It was resumed with `--resume`; each question is independent.

Files: `stage2/fullgpu-chunked-stage2-bank-a11.json`, `-a11-checked.json`, `-a11-checked-v2.json`.

## Firmware v3 and the bank re-run (Amendment 12, 2026-11-05)

**Firmware v3**
- The host writes each capture's sample rate into a table, as data, as it does the samples.
- The in-weights CPU prints:
  - `freq_hz = k_signed·fs/16` (exact);
  - `cfo_hz = mulh(angle_q32, fs)`;
  - method notes: "(averaged 16-point DFT, 8 frames)" and "(phase of the lag-1 autocorrelation)".
- Captures whose label has `fs=unknown` get no Hz fields.
- On the reference machine (`firmware_v3_check.py`, all 16 captures), every `freq_hz` equals the host value, every `cfo_hz` is within fs·2^-12, and no Hz fields appear when the rate is unknown.

**Bank re-run (P12.4):** **All gates pass** (checker v2): 96/96 circuit-exact (0 head mismatches), tool 100%, **asked fields 100%**, **numbers supported 98.8% (330/334)**, right tool 96/96, median 43 s (was 37: the longer report). No answer reached the 1000-token limit. Versus Amendment 11: **wrong Hz values 1 → 0** (22/22 Hz values correct, against 5/6; `hz_audit.py`) and **CFO attributed to the spectral bin 2 → 0**. The 4 unsupported numbers are all correct unit rescalings of in-weights values (4,947,600 Hz → "4.95 MHz"; 437,500 Hz → "437.5 kHz"; 70,198,244 Hz → "70.2 MHz"; 75,000,000 Hz → "75 MHz"); checker v2 does not rescale units.

## Held-out questions (Amendment 13, 2026-11-06)

48 pre-declared questions not in the bank's wording: 32 answerable (2 per capture), 8 general-knowledge, 8 the report cannot answer. Same model and settings as the P12.4 bank. Graded by `v15/scripts/heldout_checker.py`; report [stage2/fullgpu-chunked-stage2-heldout-checked.json](stage2/fullgpu-chunked-stage2-heldout-checked.json).

| Gate | Result | Target | Verdict |
|---|---|---|---|
| P13.2 circuit tokens exact | 40/40 | 100% | PASS |
| P13.3 tool / fields / numbers | 32/32, 44/44, **89.3%** (117/131) | 95% / 90% / 95% | **FAIL** (numbers) |
| P13.4 no tool on general questions | 8/8 | ≥ 7/8 | PASS |
| P13.5 unanswerable declined, no invented numbers | **4/8** | ≥ 7/8 and 0 | **FAIL** |

- Right tool on 32/32 (`rf_levels` for all 16 level questions, `rf_analyze` for all 16 spectral). Median 41 s per answerable question.
- The 14 unsupported numbers are Qwen's own arithmetic on report values; all 14 are correct. One wrong unit ("1.1 dBFS" for a difference).
- Declined correctly: spreading factor, BER, angle of arrival, modulation and coding rate. Wrong: occupied bandwidth "0 MHz", "0 transmitters", symbol rate = sample rate, drone distance "~100 m" via dBFS treated as dBm.
- Exact computation fixes the numbers, not the conclusions. Fix ideas are parked in `FUTURE.md` §1b and need a new pre-declared set.

## Fix: firmware v4 and the answer rule (Amendment 14, 2026-11-06)

Firmware v4 ends both reports with an in-weights-printed `not_measured=modulation, coding rate, symbol rate, occupied bandwidth, number of transmitters, distance, direction of arrival, bit error rate, protocol or packet contents (this tool reports only the values above)` (+655 tokens per `rf_analyze` call); the system prompt adds an answer rule. Checker v3 accepts simple derivations of report values. 36 new questions: [stage2/fullgpu-chunked-stage2-a14-checked-v3.json](stage2/fullgpu-chunked-stage2-a14-checked-v3.json).

| Gate | Result | Verdict |
|---|---|---|
| P14.2 exact | 34/34 | PASS |
| P14.3 unanswerable | declined 16/16 (listed 8/8, unlisted 8/8); 5 numbers in one hypothetical dBm example | **FAIL** (strict) |
| P14.4 answerable | tool 16/16, fields 91.3%, numbers 93.3% (2 misreadings, 3 conversion constants) | **FAIL** |
| P14.5 no-tool | 4/4 | PASS |

A13 set re-run under v4 (descriptive; [stage2/fullgpu-chunked-stage2-heldout-fwv4-checked-v3.json](stage2/fullgpu-chunked-stage2-heldout-fwv4-checked-v3.json)): exact 40/40, numbers 115/115 (v3), declined 8/8 with 0 unsupported numbers (was 4/8, 45). Residual hedge: the transmitter-count answer quotes not_measured but still concludes "0 active transmitters detected". Timings in these two runs are not comparable (GPU power state).

## Modulation tool `rf_modulation` (Amendment 15b, 2026-11-06)

Mode byte 'm' runs the unchanged `ModFeatures` and `IntMLP` programs inline in the firmware (classifier RAM from 0x400) and prints `modulation= runner_up= score_margin=`, a method line and a `not_measured` line. 10 classes drawn at random (seed 1516), one capture each at 20 dB (seed 1517). Full GPU (19 GB merge), chunked verification C=1024: [stage2/fullgpu-chunked-stage2-modulation.json](stage2/fullgpu-chunked-stage2-modulation-summary.json).

| Gate | Result | Verdict |
|---|---|---|
| P15.4 exact | 10/10 chip tokens = reference lock, 0 head mismatches; printed = host model 10/10; 446–469k chip tokens, 80–122 s per question | PASS |
| P15.5 Qwen | called `rf_modulation` (only) and named the printed class 10/10; runner-up/margin quoted when asked | PASS |
| Truth (descriptive) | 8/10: capture 4 32QAM → 64QAM (margin 986), capture 5 64QAM → 16QAM (margin 863); true class was runner-up in both | — |

Caveat: on one answer Qwen called margin 1604 "very confident"; the margin is an uncalibrated integer score.
