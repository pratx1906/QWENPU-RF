# RFPU-VX v1.5: RF DSP and AI on QwenPU's CPU inside the weights

v1.5 runs radio signal processing as real programs on the **RISC-V CPU and vector unit that QwenPU compiles into Qwen3.5's weights**. Each RF kernel is an RV32 program. It runs free-running on the compiled circuit, and every generated token is checked against chipformer's RV32 reference machine. Every output is also checked against an independent integer reference.

Arithmetic is 16-bit fixed point (Q15) with exact 32/64-bit accumulation. This replaces v1.4's 4-bit processor, which failed every real capture.

**Status (2026-11-06; docs corrected 2026-11-20):**
- **Stage 1 is complete:** DSP and AI running in the weights.
- **Stage 2 is complete:** Qwen calls the RF processor in its own weights through `<tool_call>` and answers from the exact report; every computer token matches the reference machine. See [reports/STAGE2.md](reports/STAGE2.md).
- **Evidence campaign is complete:**
  - 133 real recordings from 5 datasets (screen, full pipeline, 10-tool battery);
  - a 96-question Stage 2 bank with an automatic answer checker;
  - fixes from Amendment 12.
- **Paper draft in progress:** [paper/PAPER_DRAFT.md](paper/PAPER_DRAFT.md).
- **Three new RF vector instructions are built into QwenPU's VX circuit** (branch `rfpu-vfir`, 13 layers, no extra heads), all token-exact on saved checkpoint weights:
  - **VFIR:** 6.7× faster than scalar for 16 taps (pre-declared timing study, 95% CI 6.727–6.738×);
  - **VPOW:** power sum;
  - **VBFLY:** FFT butterfly with block-floating-point tracking, 1.8× faster than the CPU block FFT.

The design is in [research/V15_DESIGN.md](research/V15_DESIGN.md) and the protocol in [research/PROTOCOL.md](research/PROTOCOL.md). The live ledger is [results.md](results.md) and the gates are in [reports/gates.json](reports/gates.json).

## Headline results (all measured on one RTX 5090 Laptop GPU)

| Question | Result | Evidence |
|---|---|---|
| Accurate on real signals? (14 captures; criteria unchanged from v1.4) | **8/14 pass** primary (v1.4: 0/14). Every failure is input clipping from the inherited gain rule; arithmetic NMSE is ~1e-9 | [reports/real-iq-q15.json](reports/real-iq-q15.json) |
| ... with 4× input headroom? (pre-declared secondary diagnostic) | **13/14 pass**, all NMSE ~1e-8. The one failure is a peak near-tie (1.4e-4 relative gap) in an OFDM record | [reports/real-iq-q13-headroom.json](reports/real-iq-q13-headroom.json), [reports/peak-tie-diagnosis.json](reports/peak-tie-diagnosis.json) |
| ... with a block-floating-point front end computed in the weights? (pre-declared Amendment 2) | **14/14 pass**, 57,344 neural samples, criteria unchanged | [reports/real-iq-bfp.json](reports/real-iq-bfp.json) |
| **Expanded real-IQ screen: 133 recordings, 5 datasets** (Amendment 9) | **132/133 pass (99.2%, Wilson 95% CI 95.9–99.9%)**. IDLab 53/53, EPFL LoRa 32/32, NTU LoRa 27/27, Tampere drones 18/19, CAMRAS radio astronomy 2/2. NMSE median ~5–8e-9 (threshold 1e-2). The one failure is a 0.024% matched-filter peak near-tie (displacement 2 > 1); the arithmetic is correct | [reports/real-iq-expanded.json](reports/real-iq-expanded.json) |
| Full 8-stage pipeline on all 133 (Amendment 10a) | **PASS**: 133/133 streams, every stage token-exact, 288.6M tokens, 38.9 ms/sample | [reports/pipeline-expanded.json](reports/pipeline-expanded.json) |
| 10-tool battery on all 133 (Amendment 10b) | **PASS**: 10 kernels × 133 recordings exact. SQNR median / minimum: FFT64 block 75.7 / 72.7, CORDIC 150 / 143, NCO 81.1 / 74.1, IIR 78.5 / 55.9, VFIR guard 73.4 / 65.4, VBFLY 71.5 / 68.6 dB. The low tails are a constant rounding floor against strongly attenuated outputs (corr 0.99), not clipping | [reports/kernel-battery.json](reports/kernel-battery.json), [reports/battery-tail-diag.json](reports/battery-tail-diag.json) |
| IIR with 4 extra output bits (Amendment 12, P12.5) | **PASS**: 133/133 exact, minimum **80.0 dB** (was 55.9), median 102.6 | [reports/a12-battery.json](reports/a12-battery.json) |
| VFIR adaptive output scale (P12.6, expected FAIL declared before the run) | **FAIL**: exact, 0 saturation, minimum 68.3 dB < 70 (median 73.4 → 76.7). VFIR rounds each tap product on its own; closing it needs exact accumulation in the circuit | [reports/a12-battery.json](reports/a12-battery.json) |
| VFIR guard bit (Amendment 8, P8.1) | **77.9 dB** (≥ 75 target; P6.3's 69.2 dB miss fixed); exact FIR 81.3 dB; same 110 tokens | [reports/amendment8.json](reports/amendment8.json) |
| Error budget and precision fixes (pre-declared Amendment 4) | Block-floating-point FFT: **74.6–76.3 dB at every level from −60 to 0 dBFS** (was 7.5–66.5); interpolated NCO **85.8 dB** (was 49.0); +32% / +36% tokens | [reports/ERROR_BUDGET.md](reports/ERROR_BUDGET.md) |
| Token reductions (Amendment 5) | FFT program rewrite bit-identical, **17–19% fewer tokens** (FFT64 201→166); AM envelope magnitude-only CORDIC 450→256 bit-identical, **185 with 10 iterations at 121.6 dB** (2.4×) | [reports/fft-opt-neural.json](reports/fft-opt-neural.json), [reports/am-trim-neural.json](reports/am-trim-neural.json) |
| New FFT instruction VBFLY with block floating point (Amendment 6) | **69.5–71.5 dB at every level −60…0 dBFS, 1.8× faster** than the CPU block FFT (127–133 vs 229–233 tokens/sample); 13 layers, no new heads | [reports/amendment6.json](reports/amendment6.json) |
| IIR without saturation (`wide=True`) | 82.3 / 88.2 dB at −12 / −6 dBFS (was 35.1 / 14.8) | [reports/amendment6.json](reports/amendment6.json) |
| VFIR precision (pre-declared ≥ 70 dB) | **69.2 dB: failed by 0.8 dB** (per-tap rounding); exact FIR16 81.3 dB | [reports/amendment6.json](reports/amendment6.json) |
| Why did v1.4 fail? | Four-bit intermediate narrowing and wrap, not input quantization | [research/NUMERICAL_FAILURE_ANALYSIS.md](research/NUMERICAL_FAILURE_ANALYSIS.md) |
| How much RF DSP runs inside the weights? | **25 CPU kernels + 4 vector-unit kernels**, all neural-exact | [reports/capability-campaign.json](reports/capability-campaign.json), [reports/vx-neural.json](reports/vx-neural.json) |
| Full pipeline on real data? | **PASS**: 8 stages × 14 captures × 2,048 samples, 31.1M tokens, every token exact, 38.2 ms/sample for the whole chain (clean run) | [reports/pipeline-20261103.json](reports/pipeline-20261103.json) |
| Full pipeline with ground truth? | **PASS**: bursts found at the correct windows, peak inside the burst, tone frames 131 vs 128 true | [reports/pipeline-synthetic.json](reports/pipeline-synthetic.json) |
| Learned AI stage? | Trained 3-class MLP: 82.3% float, **82.0% exact integer**, neural-exact (synthetic task only) | [configs/spectral-mlp.json](configs/spectral-mlp.json) |
| Vector unit (NPU) vs scalar CPU | Classifier **3.3× fewer tokens** (18.0 vs 60.2 ms/sample); energy 1.08× | [reports/vx-neural.json](reports/vx-neural.json) |
| Same programs on the actual published QwenPU checkpoint (HF layers)? | **14/14 kernels exact**, free-running, including both VX kernels; ~19–30 ms/token on the HF path | [reports/qwen-checkpoint-final.json](reports/qwen-checkpoint-final.json) |
| DSP figures of merit | FIR 78.5 dB SQNR (12.7 ENOB) at −12 dBFS; FFT64 power 56 dB; energy ROC matches chi-square theory; CFAR Pd ≥ 0.93 from −6 dB; matched-filter gain 12.4 dB (theory 12.0); FM in-band SINAD 35–67 dB; 2–6×10^5× slower than same-contract integer NumPy/CUDA | [reports/DSP_METRICS.md](reports/DSP_METRICS.md) |
| Qwen calls the in-weights RF tool (Stage 2) | Correct answers from the exact report; 0 lock errors, 0 head mismatches. Firmware cut from ~252k to **~93–100k chip tokens per call**: 121–163 s per question on the executor, 271–299 s fully on one GPU (19 GB merge) | [reports/STAGE2.md](reports/STAGE2.md) |
| Stage 2 with chunked verification and `rf_levels` | **Token-identical** to the one-token run; **47–54 s per question** (was 255–288 s; computer part 16 s, was 223–247 s). Level-only questions: Qwen picks `rf_levels`, **27–31 s** | [reports/STAGE2.md](reports/STAGE2.md) |
| Qwen on real recordings (Amendment 7) | **6/6** real captures: exact reports, asked numbers quoted exactly, right tool chosen, 29–42 s per question | [reports/STAGE2.md](reports/STAGE2.md) |
| **Stage 2 question bank, 96 questions, automatic answer checker** (Amendment 11) | **All gates pass**: 96/96 circuit-exact, tool called 100%, asked fields quoted 96.6% (all misses are trailing zeros; 100% under checker v2), **99.0% of numbers traceable to the verified report**, right tool 96/96, median 37 s. The 3 unsupported numbers are Qwen's own Hz conversions, one wrong by 1000×, and the checker flags it | [reports/stage2/fullgpu-chunked-stage2-bank-a11-checked.json](reports/stage2/fullgpu-chunked-stage2-bank-a11-checked.json) |
| Bank re-run with firmware v3: Hz values computed in the weights, method notes (Amendment 12, P12.4) | **All gates pass**: 96/96 exact, fields 100%, numbers supported 98.8%, wrong Hz values 1 → 0 (22/22 correct), CFO-method misattributions 2 → 0; the 4 unsupported numbers are correct kHz/MHz rescalings of in-weights values | [reports/stage2/fullgpu-chunked-stage2-bank-checked-v2.json](reports/stage2/fullgpu-chunked-stage2-bank-checked-v2.json) |
| **Held-out Stage 2 questions, 48 in free wording** (Amendment 13) | Circuit-exact 40/40; tool 32/32, right tool 32/32, asked fields 44/44; no-tool questions 8/8. **Two gates fail:** numbers supported 89.3% < 95% (all 14 unsupported are Qwen's own derived values, all arithmetically correct); **unanswerable questions declined only 4/8**: the other 4 drew confident wrong conclusions from exact numbers (e.g. a drone distance from dBFS treated as dBm). Fuzz regression on final code 40/40 exact | [reports/stage2/fullgpu-chunked-stage2-heldout-checked.json](reports/stage2/fullgpu-chunked-stage2-heldout-checked.json) |
| **Fix: firmware v4 `not_measured` line + answer rule** (Amendment 14), 36 new questions | Exact 34/34. **Unanswerable declined 16/16** (was 4/8), incl. 8/8 quantities not on the list. Two gates still fail strictly: one decline adds a hypothetical dBm example; answerable numbers 93.3% (checker v3) with 2 real misreadings (0/16 read as 16/64; a dropped minus sign). A13 set re-run under v4 (descriptive): declined 8/8, 0 invented numbers | [reports/stage2/fullgpu-chunked-stage2-a14-checked-v3.json](reports/stage2/fullgpu-chunked-stage2-a14-checked-v3.json) |
| VBFLY / VPOW on saved checkpoint weights | **5/5 programs token-exact** (incl. a real-IQ FFT64 frame) | [reports/vx-rf-checkpoint.json](reports/vx-rf-checkpoint.json) |
| Classifier with more data | 6 locations at 1 MS/s: 74.4%; 10 MS/s: 84.6%; **10 MS/s with bandwidth features: 96.2% per recording** (78.8% per window; indicative, 26 recordings). Two-stage in weights (FFT64, then the 81-feature classifier), 26/26 neural-exact | [reports/classifier-10msps.json](reports/classifier-10msps.json) |
| **Blind 25-class modulation classifier in weights** (Amendment 15; RadioML-2018 set + LoRa, synthetic) | **All gates pass.** 250/250 captures exact in both stages (30 integer features, then a 30-32-25 integer MLP; 106M tokens). **Accuracy 86.4%** (216/250, chance 4%): 68% at 10 dB, 84% at 14–16 dB, 88–100% from 18 dB. 21/25 classes ≥ 7/10, 16 ≥ 9/10; weak: 32/64/128QAM 5/10, 256QAM 6/10. Errors are order confusions inside a family (QAM orders, 16/32PSK, APSK orders); only 2 cross-family slips | [reports/modclass-neural.json](reports/modclass-neural.json) |
| Qwen calls `rf_modulation` (Amendment 15b, ~465k chip tokens per call) | **10/10 exact**, printed class = host model 10/10, Qwen called the tool and named the printed class 10/10, 80–122 s per question. Truth 8/10 (descriptive): 32QAM→64QAM and 64QAM→16QAM were reported faithfully as printed. One unsupported phrase ("very confident" for an uncalibrated margin) | [reports/stage2/fullgpu-chunked-stage2-modulation-summary.json](reports/stage2/fullgpu-chunked-stage2-modulation-summary.json) |
| Real-data technology classifier (IDLab WiFi/LTE/DVB-T) | 67% per window, 70% per recording (chance 40%), neural-exact | [configs/tech-classifier.json](configs/tech-classifier.json) |
| VFIR vs scalar FIR16, pre-declared timing study | **6.73×** geometric mean, 8 processes, 95% CI [6.727, 6.738], exact sign-flip p = 0.0078 | [reports/timing-study.json](reports/timing-study.json) |
| Chunked (speculative) verification on the saved layers (Amendment 5) | 107k-token trace, **0 mismatches** in every configuration; with flash-linear-attention and GPU-side compare **13.0k tokens/s = 3.0× the fused executor** (~360× one-token-at-a-time on the same layers); an altered drafted token is caught. Audit mode: the host drafts the trace | [reports/gates.json](reports/gates.json) (V15-P5.3) |
| Tests | **218 passed**, 21 skipped (branch-only); branch `test_vfir` 14/14 | `v15/tests` |

## Speed

| Measurement | Value |
|---|---|
| QwenPU fused executor, this laptop | FIR16: 7.4k tokens/s at batch 1, 33k tokens/s with 32 machines in parallel (4.5×) ([DSP_METRICS](reports/DSP_METRICS.md) systems table) |
| v1.4 | 2.2 ms/step |
| Hugging Face layers | ~19 ms/token |
| Cheap kernels | 3–10 ms per sample |
| Heavy kernels | 64-tap wide FIR 176 ms/sample (batch 16, half-filled) |
| Energy cost | ~1–1.6 J per sample at ~160 W |
| Chunked verification (audit mode) | 13.0k tokens/s on the saved layers with flash-linear-attention (same FIR64 workload on the fused executor at batch 1: 4.4k; free-running HF: 35.6) |
| vs a conventional DSP | About 10^5–10^6× slower. No real-time claim is made |

## Capability matrix

Measured on real IQ, 2,048 samples, batch 16. "ms/sample" is synchronized neural time divided by samples.

| Group | Kernel | Tokens/sample | ms/sample |
|---|---|---:|---:|
| Filtering | Complex FIR 4 / 16 / 32 taps | 103 / 353 / 670 | 6.8 / 23.7 / 45.1 |
| Filtering | Complex FIR 64 taps, 64-bit accumulator | 2621 | 175.7 |
| Filtering | Lowpass 16 taps (windowed sinc) | 319 | 21.3 |
| Filtering | Decimating FIR (8 taps, ÷4) | 48 | 3.2 |
| Filtering | IIR biquad / DC blocker (stateful) | 176 / 86 | 94.8 / 52.9 |
| Frequency | NCO mixer (stateful phase, sine table) | 54 | 28.9 |
| Spectral | FFT4 / 16 / 64 / 256 power | 94 / 153 / 404 / 253 | 6.3 / 10.3 / 27.2 / 17.0 |
| Spectral | FFT64 complex bins, Welch PSD 64×4, single-bin DFT | 190 / 416 / 181 | 12.8 / 28.1 / 12.2 |
| Detection | CA-CFAR (64 bins), energy + min-statistics noise + burst edges, peak search on the CPU | 603 / 19 / 30 | 40.3 / 9.9 / 16.2 |
| Demodulation | CORDIC magnitude/phase, AM envelope, FM discriminator | 457 / 450 / 482 | 30.5 / 30.0 / 32.4 |
| Synchronization | CFO estimate (lag 1, window 64 / lag 8, window 256) | 48 / 44 | 3.2 / 2.9 |
| Inference | Spectral MLP (Q12 weights, ReLU, argmax on the CPU) | 266 | 17.8 |
| Vector unit | VX energy / VX spectral MLP (scalar equivalents: 16 tok, 2.3 ms / 416 tok, 60.2 ms) | 15 / 125 | 2.1 / 18.0 |

## Numerical contract

Defined in [rfpu_v15/cpu/\_\_init\_\_.py](rfpu_v15/cpu/__init__.py):

- **Inputs:** Q15 (or Q2.13 with headroom).
- **Arithmetic:** products exact; accumulation 32-bit when a range bound proves it safe, otherwise exact 64-bit (FIR picks this automatically); requantization is round-half-up; saturation is branch-free.
- **FFT:** radix-2, ½ scaling per stage.
- **Power:** exact 64-bit (Q30).
- **CORDIC:** 16 iterations, binary angle.
- **Program structure:** programs avoid data-dependent branches.
- **Host's role:** writes input and parameter words into RAM, copies verified outputs between stages, and reads output words that the machine itself loaded. It does no arithmetic on signal values.

## Layout

| Path | Role |
|---|---|
| `rfpu_v15/cpu/codegen.py` | RV32 code generation, RAM layout, loops, readback |
| `rfpu_v15/cpu/kernels.py` | Kernel protocol (stateless and stateful), FIR, FFT4 power |
| `rfpu_v15/cpu/filters.py` | Decimating FIR, IIR biquad, DC blocker, NCO mixer |
| `rfpu_v15/cpu/spectral.py` | FFT 8–256 power/complex, Welch, CFAR, DFT bin |
| `rfpu_v15/cpu/nonlinear.py` | CORDIC magnitude/phase, AM, FM, CFO |
| `rfpu_v15/cpu/detect.py` | Energy detector with noise floor and burst edges, peak tracker |
| `rfpu_v15/cpu/infer.py` | Spectral MLP with runtime weights |
| `rfpu_v15/cpu/vx_kernels.py` | VX vector-unit energy, dense layer, MLP |
| `rfpu_v15/cpu/runner.py` | Parallel verified runner: batches CPUs on one CUDA graph and checks every token |
| `rfpu_v15/cpu/pipeline.py` | End-to-end DSP-AI pipeline |
| `rfpu_v15/neural.py`, `runtime.py` | Earlier bit-sliced I16/W64 ALU checkpoint (secondary; 10/10 opcodes exact) |
| `scripts/` | Gate, campaign, VX, pipeline, metrics (`dsp_metrics.py`), training, checkpoint scripts; `run.py` is the durable ledger |
| `tests/` | 218 CPU-reference tests (+21 that need the RF branch on `PYTHONPATH`) |

## Run it

From the RFPU root:

```powershell
.venv/Scripts/python.exe -m pytest v15/tests -q -p no:cacheprovider          # reference-machine tests
.venv/Scripts/python.exe v15/scripts/pipeline_run.py --synthetic              # one command, pass/fail
.venv/Scripts/python.exe v15/scripts/pipeline_run.py --captures 0 12          # real captures
.venv/Scripts/python.exe v15/scripts/real_iq_q15.py                           # 14-capture gate
.venv/Scripts/python.exe v15/scripts/dsp_metrics.py precision summary        # DSP figures of merit
.venv/Scripts/python.exe v15/scripts/qwen_checkpoint_final.py                 # actual published checkpoint
```

Wrap any of these in `v15/scripts/run.py <name> -- <python> <script>` to log them to the ledger. Pass the interpreter as an **absolute** path; relative paths fail on Windows. Real-capture runs need the datasets under `$RFPU_DATA/datasets` (see [../data/README.md](../data/README.md)).

## Findings worth knowing

- **Input scaling, not arithmetic, limits accuracy at 16-bit.** No single fixed gain rule both avoids clipping on peaky signals and keeps resolution on weak ones. The dynamic-range sweep shows the usable ≥40 dB FIR window is −48 to −12 dBFS for Q15 and −36 to 0 dBFS for Q2.13.
- **QwenPU's compiled vector unit can disagree with its reference model** (`vx.py`). A 2-element VQ write at a non-chunk-aligned VA offset (14), followed by VMAT, gave a 1-bit difference in the VMAT result in both the fused executor and the exact circuit IR. QwenPU's own VMAT tests (chunk-aligned writes) pass. We now use only aligned writes.
- **Stateful kernels serialize per stream.** Throughput comes from running many streams at once (batch 32 gives 4.5× the throughput of batch 1).
- **Design limits:**
  - The 16-tap lowpass has −3.3 dB DC gain, the price of its guaranteed no-overflow scaling.
  - The IIR test filter has +12 dB DC gain, so it saturates in Q15 at −12 dBFS (35 dB SQNR vs 69 dB with Q2.13); fixed later by `wide=True` (82.3 dB, Amendment 6) and `out_frac=4` (minimum 80.0 dB, Amendment 12).
  - The NCO's 1024-entry table limits it to 49 dB; the interpolated NCO (Amendment 4) reaches 85.8 dB.
  - FFT per-stage scaling costs about 6 bits.

## Ledger note

Between about 12:00 and 12:52 UTC on 2026-11-03, a background job chain that I had meant to stop kept running. It duplicated the vx-neural, pipeline, checkpoint and test runs. Those duplicates are in [results.md](results.md); the checkpoint attempt among them failed on a script bug that was fixed afterwards. Every run was token-exact, but timings from that window overlapped other GPU work. The `*-clean` runs repeat the timing-sensitive steps in isolation, and the numbers above come from clean runs.

## Limitations

- **Speed:** far from real time (~10^5–10^6× slower than conventional DSP).
- **"Exact"** means bit-exact to a declared 16-bit fixed-point contract, with a measured error budget per tool, not infinite precision.
- **Qwen's interpretation is not verified, only its numbers.** The answer checker measures it: 99% of numbers in answers trace back to the verified report, and the rest are flagged. Explanations of method can still be wrong (two cases in Amendment 11, fixed by firmware v3), and conclusions drawn from exact numbers can be wrong (Amendment 13); firmware v4's `not_measured` line reduces this but does not remove it (Amendment 14).
- **Evidence base:**
  - 133 real recordings for DSP;
  - 26 recordings for the 10 MS/s classifier (indicative);
  - 96 Stage 2 bank questions on 16 captures, plus 84 held-out questions (Amendments 13–14) and 10 modulation questions;
  - a single laptop GPU (no second-machine timing).
- **Modulation classifier:** synthetic data from our own generator (8 samples/symbol known, no CFO, no multipath, no unknown symbol rate); 86.4% is an in-distribution figure, not a real-world AMC claim. High-order QAM (32–256) is the weak spot. The score margin is not a calibrated confidence.
- **Peak criterion:** the global matched-filter argmax is fragile on near-ties (the only expanded-screen failure).
- **VFIR precision on strongly attenuated outputs** stays at 68 dB, because of per-tap rounding in the instruction contract (P12.6 FAIL).
- **Executor vs checkpoint layers:** large campaigns ran on the fused executor. Equivalence with the saved HF layers was verified on representative programs (14 on the published checkpoint, 5 + 5 on the two VX merges, plus the Stage 2 full-GPU runs), not on every campaign run.
- **VX model size:** the VX/VBFLY merge is 27 GB and does not fit the 24 GB GPU together with Qwen, so Stage 2 uses the plain-CPU merge.
- **Chunked verification** is an audit mode: the host computes the trace too, and the saved layers verify it.
- **Production mode** (no readback) saves only 1.4% of tokens.
- **Not implemented** (see [FUTURE.md](FUTURE.md)):
  - a general precision-selection pass;
  - weight-stationary streaming VFIR;
  - exact-accumulation VFIR;
  - ablations A2, A3 and A7.
