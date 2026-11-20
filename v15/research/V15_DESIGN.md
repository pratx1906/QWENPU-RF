# RFPU-VX v1.5 — design

Status: draft for approval, 2026-11-03. Mandate: the original project brief (not published; it ends inside Phase H). This document fixes how the mandate is carried out on this repository and hardware. Where it says "builder's call", the implementer decides and records the decision.

## 0. Stages and Stage 1 acceptance (approved by the user, 2026-11-03)

The research goal is **Qwen itself with an attached, weight-compiled RF DSP–AI processor**: Qwen calls the processor, the DSP and inference run exactly inside the same weights, and Qwen reads and reasons over the results. Stand-alone layer-type processors (v1.2–v1.5 so far) are the means, not the goal. Work proceeds in three stages:

1. **Stage 1 — DSP–AI system working 100%** (this plan). Done only when every item below is measured:
   1. The complete pipeline IQ → complex FIR (up to 32–64 taps) → FFT (up to 64) → power / correlation / activity → runtime-weight inference → scores executes in neural layers at 16-bit precision. The host performs no numeric operation on DSP values (§8 host boundary).
   2. Every neural output bit-matches the independent reference, on synthetic tests and on real captures.
   3. The real-IQ screen passes on the frozen 14 captures. The criteria are FIR NMSE ≤ 0.01, spectral NMSE ≤ 0.01 and matched-peak displacement ≤ 1 sample. Thresholds are not changed after viewing. If any capture fails, the cause is diagnosed and reported honestly; it is not relaxed.
   4. One fixed checkpoint runs everything. Coefficients and model weights are runtime data.
   5. One command runs the full pipeline end to end and reports pass/fail.
2. **Stage 2 — attach to Qwen3.5-4B.** Processor layers go in front of Qwen's unchanged layers via a Qwenputer-style `<tool_call>` bridge. Qwen calls the RF pipeline and reasons over exact results. The v1.5 hidden size (2560) already matches Qwen3.5-4B; the 4B base weights (~9 GB) must be downloaded. Designed separately after Stage 1.
3. **Stage 3 — evidence campaign and paper.** Ablations, timing protocol, novelty audit, manuscript and release (§4 items 7–9). Designed in detail after Stage 2.

Within Stage 1, the §4 items 1–6 apply. The QwenPU-as-precision-candidate question and the time budget are deferred to Stage 3; there is no fixed deadline for Stage 1.

## 0.1 Architecture revision (2026-11-03, user direction: "programmable CPU-like behavior as in Qwenputer; use QwenPU to make it fast")

**Supersedes §3.1–3.3.** The primary v1.5 target becomes QwenPU's own programmable RV32IM+Zba CPU and VX vector unit. It is compiled into weights by the pinned `chipformer` and has a PC, branches, ROM/RAM in Gated DeltaNet state, and assembled RV32 programs. It is extended with new RF vector instructions and executed free-running on QwenPU's fused executor (`chipformer/rv/fusedexec.FreeRun3`). Every token is checked against the RV32 reference machine. Equivalence with the saved-checkpoint layers is established separately, in lockstep (`execcheck`/`vxcheck` style), and is not assumed.

The bit-sliced I16/W64 ALU checkpoint stays as a measured, exact (10/10 opcodes) secondary artifact, kept for comparison.

Spike measurements on this laptop, made with throwaway scratch scripts (to be re-measured through the ledger):

| Measurement | Result |
|---|---|
| QwenPU fused executor, plain build | 7,188 tok/s at batch 1 (0.139 ms/step), exact |
| v1.0-compiled Q15 16-tap complex FIR, 32 samples, VX build | 20,059 tokens, exact, 4,244 tok/s ≈ 6.8 outputs/s (v1.0 HF path ≈ 19 ms/token) |
| Q15 Hann → FFT8 → density PSD | 1,947 tokens, exact |
| Batch scaling | 13,136 tok/s at batch 8; 14,893 tok/s at batch 32 |
| v1.5 ALU with `flash-linear-attention` (separate D: env) | 5.4–8.4k ops/s (was 3k) |

Stage 1 build order:

1. **Q15 RF program library on the free-running CPU.** Reuse and extend the v1.0 compiler: FIR, FFT up to 64, correlation, activity, MLP. Rerun the 14-capture screen at Q15 with every token verified.
2. **Batch-parallel streaming.** Split long streams into overlapping chunks across machines; results must be exact and independent of how the stream is split.
3. **RF vector instructions in VX** (new custom op family): int16 complex MAC/FIR, butterfly/twiddle, power. Each gets a reference in `vx.py` style, a circuit, random-program tests and a checkpoint lockstep check. The aim is to cut tokens per output by an order of magnitude.
4. **Saved-checkpoint equivalence** for the extended CPU, plus a one-command end-to-end runner.

## 1. Intent

v1.4 showed exact neural execution but useless RF numerics: 0/14 real captures passed the screen (FIR NMSE ≤ 0.01, spectral-power NMSE ≤ 0.01, matched-peak displacement ≤ 1 sample) at Q3. v1.5 answers one question with measured evidence: **can a programmable neural processor reach that numerical screen on real IQ, and what does the precision cost in parameters, forwards and time?** A rigorously supported "no", or "yes at cost X", are both acceptable outcomes. The output is a paper-grade evidence package (a later decision on publication is the user's).

## 2. Starting facts (measured 2026-11-03)

- v1.5 checkpoint `$RFPU_DATA/checkpoints/rfpu-v15/bitwise-I16-W64-v1`: 15 stock Qwen3.5 layers, 807,933,280 parameters, 1.62 GB, SHA256 `eb6ae816…`. Built from `v15/rfpu_v15/neural.py`.
- It is a **host-sequenced, stateless SIMD neural ALU**: operands are 64-bit two's-complement bit vectors injected into two recurrent heads; the token selects the opcode; one forward computes one opcode over a batch. Opcodes: ADD64, SUB64, MUL (signed 16×16 → 32, sign-extended), CMPGE, rounded arithmetic right shift by 1..30 (ties toward +∞). No neural PC, ROM or RAM.
- Saved-checkpoint exactness: all 10 tested opcodes exact on 256 edge+random pairs (`v15/reports/native-validation.json`).
- Throughput: ≈3,000 element-ops/s, **flat** across batch 32/128/512 (10.9 / 42.6 / 172.8 ms per forward); peak 2.3 / 4.7 / 14.1 GiB. The reference PyTorch DeltaNet/conv fallbacks are in use because `flash-linear-attention` and `causal_conv1d` are not installed.

## 3. Architecture decisions

1. **Two neural targets, one per precision class.**
   - *P1 control* = the frozen v1.4 autonomous Q3/W16 processor, unchanged.
   - *v1.5 ALU* = the bitwise I16/W64 checkpoint, carrying P2, P3, P4, P5 and P6.
   
   The paper must state plainly that the v1.5 target trades v1.4's autonomous control and neural RAM for precision. The host sequences instructions and holds inter-instruction values as exact integer codes. No arithmetic, comparison, shift or narrowing on DSP values happens on the host. Bit packing/unpacking, gather/copy and opcode selection are declared representation and control operations.
2. **Precision candidates on the v1.5 ALU:**
   - **P2:** Q7 and Q15 operands with W64 accumulation.
   - **P3:** multiword. Q31 operands from 16-bit limbs (limb products by MUL, recombination by shift+ADD), reconstructed and verified against the 64-bit result.
   - **P4:** block floating point. The shared exponent is found with neural CMPGE against power-of-two thresholds. The host may branch on those neural comparison bits (data-dependent dispatch, declared) but never computes magnitudes itself.
   - **P5:** mixed precision chosen by the compiler.
   - **P6:** saturation and selection built from CMPGE, MUL and ADD (composed). Wrap is native mod 2^64.
3. **Possible second v1.5 checkpoint (builder's call).** If composed SAT/SELECT/ABS dominate cost, emit a separately versioned checkpoint with native versions and compare it (A7 instruction-encoding ablation). Never overwrite `bitwise-I16-W64-v1`.
4. **Faster runtime as an ablation.** Try installing `flash-linear-attention` and `causal_conv1d` in a **separate** environment on D:; the pinned root `.venv` is not modified. Report it as A6 (same weights, same semantics). If the kernels are unavailable or change results, record it and use the fallback.

## 4. Work, in order

1. **Baseline freeze.** Verify the v1.4 source/checkpoint hashes (history-freeze already exists), rerun two representative v1.4 programs, and record the environment.
2. **Phase A (CPU only).** Error decomposition of the v1.4 pipelines on the 14 captures, switching on one error source at a time (input quantization, wrap, shift, overflow, accumulated depth), with high-precision references. Deliverable: `research/NUMERICAL_FAILURE_ANALYSIS.md`.
3. **Phase B.** Implement and verify P2–P6 on the v1.5 ALU. Classify each from "analytically proposed" to "complete-pipeline verified". Record the resource table from the mandate §4. Q15 directly on the v1.4 control-flow architecture is reported with its exact limitation (v1.2 dense estimate 37.4 GB), not attempted. Deliverable: `research/PRECISION_ARCHITECTURE.md`.
4. **Phase C. New precision-typed IR and compiler in `v15/rfpu_v15/`.** Values carry signedness, width, scale/exponent, complex/real, overflow and rounding policy, lifetime and storage class. Passes:
   - range analysis (interval);
   - scale propagation;
   - precision selection under explicit numerical constraints;
   - overflow proof or detection;
   - requantization insertion;
   - lowering to v1.5 ALU op streams, with batch packing of independent elementwise ops;
   - lifetime allocation of host value slots;
   - dependency-safe scheduling;
   - verification.

   "Program size" = op-stream length. ROM validation applies to P1 programs, which keep using the v1.4 lowering. Cost terms (forwards, overhead, bytes moved) are calibrated from measurements on a training workload set and evaluated on held-out workloads.
5. **Phase D workloads on the v1.5 ALU:**
   - complex FIR with 4/8/16/32/64 taps;
   - radix-2 FFT 4/8/16/32/64 with windows, overlap, runtime twiddles and power;
   - matched filter on injected waveforms with controlled SNR, timing, CFO, amplitude and phase (host peak search declared; neural argmax via CMPGE tree is a builder's call);
   - activity detection with an explicit noise estimate from a declared noise-only segment or minimum-statistics, and burst start/end;
   - complete IQ → FIR → FFT/correlation features → runtime-weight MLP → scores.
   
   One checkpoint for all of them.
6. **Phase E fidelity gate**, prespecified in `research/PROTOCOL.md` before any v1.5 real-IQ run.
   - Same 14 captures, same preprocessing and thresholds as `v14/reports/real-iq.json`.
   - **Primary comparison:** the same v1.4 pipeline configuration (taps2, FFT4) at each verified precision.
   - **Secondary:** the larger workloads.
   - A capture passes only if (a) neural execution on its predeclared window bit-matches the CPU fixed-point oracle and (b) that oracle meets all three criteria on the declared evaluation span. Neural and CPU sample counts are reported separately.
   - The ≤1-sample peak criterion on arbitrary records is kept, not relaxed; its appropriateness is discussed separately.
7. **Phase F ablations A1–A8.**
   - A1 (precision), A5 (fusion), A6 (runtime) and A8 (conventional CPU/CUDA, separated timing scopes) on v1.5.
   - A2/A3 are defined for P1; on the stateless v1.5 ALU they are marked not applicable unless a register-residency variant is built (builder's call).
   - A4 on v1.5 = batch packing vs per-op dispatch.
   - A7 per §3.3, else blocked.
   - Also a **controlled baseline against the unchanged published QwenPU checkpoint** on identical programs, with matched runtime (eager vs eager, or graph vs graph) and the speedup decomposed into forwards per sample vs time per forward.
8. **Phase G main timing campaign.** Prespecified protocol: ≥16 fresh processes, balanced order, excluded warmups, telemetry, fixed hashes. Primary endpoint: warm complete-stream time per IQ sample, for numerically passing implementations only.
9. **Phase H and write-up.**
   - Novelty audit extending `v14/research/RELATED_WORK_MATRIX.md`. Add: Percepta transformer-vm, ALTA, Tracr, deep unfolding, Butterfly-Net, Versal AIE, agentic-RF/LLM-RF works, precision-aware and BFP DSP compilation, FPGA DSP.
   - LaTeX paper under the mandate's working title, built like v14.
   - Claim–evidence matrix, publication-readiness audit, local release zip and verification on D:.

## 5. Budget and feasibility

At ≈3,000 ops/s, a 32-tap complex FIR costs ≈260 ops per output (≈11 s per 128 outputs) and one FFT64 frame with power ≈2,500 ops (≈1 s). The neural window per capture is therefore fixed small and declared in advance; working estimate 128–256 complex samples per capture per workload. Full-capture numerics come from the bit-exact CPU oracle and are labelled CPU coverage. The whole campaign is multi-day wall time on one laptop GPU, run sequentially.

## 6. Non-goals and anti-goals

**Non-goals:**
- real-time operation;
- trained-classifier accuracy;
- operational UAV detection;
- autonomous control on the v1.5 ALU;
- modifying upstream QwenPU;
- changing any v1.0–v1.4 file or measurement.

**Anti-goals (count as failure even if tests pass):**
- host arithmetic on DSP intermediates;
- CPU-only numerics reported as neural results;
- thresholds or endpoints changed after viewing results;
- incomparable timing scopes presented as speedups;
- overwriting a frozen checkpoint.

## 7. Constraints

RTX 5090 Laptop GPU (24 GB), one host. Source and reports on C:, weights/data/environments on D:. GPU jobs sequential, with existing VRAM admission and WDDM guards. All experiments go through `v15/scripts/run.py` (append-only ledger and source snapshots). The interpreter must be passed as an absolute path; the failed relative-path attempt stays in the ledger.

## 8. Settled after review

- **Mandate completeness.** The user confirmed the supplied text is the whole mandate. Phase H's list is completed from this session's prior-art search. The paper and release are included because the mandate title names them.
- **Host boundary (strict).** The host may only:
  - pack and unpack whole words to and from bits;
  - copy, gather and pad words;
  - select opcodes;
  - branch on neural comparison outputs.

  Anything that changes a value's numeric meaning must run in neural layers. That includes left/right shifts, limb splitting, sign extension, narrowing, saturation and scaling. Because the current ALU lacks a left shift and limb or narrowing ops, a **second, separately versioned checkpoint** (`bitwise-I16-W64-v2`) adds native SHL, SPLIT16 (low/high limb), SAT/NARROW16, SELECT and ABS. v1 vs v2 on identical programs is the A7 instruction-encoding ablation.
- **Input preprocessing.**
  - Primary: the frozen v1.4 rule. The gain is 0.75 / p95 of the first 512 samples and the full-scale clip is ±1. Only the Q scale changes (×2^(b−1), clip to the format range).
  - Secondary: input scaling chosen by the compiler (P4/P5), reported separately.
- **Evaluation span.**
  - The oracle criteria use the v1.4 4096-sample CPU span, for comparability.
  - The neural window must bit-match the oracle on its samples.
- **If nothing passes the gate.**
  - Phase G still runs, with a fallback primary endpoint prespecified in PROTOCOL.md: warm time per IQ sample for all bit-exact implementations, labelled "not numerically passing".
  - The near-tie sensitivity of the peak criterion is analysed separately, and the criterion is not changed.

## 9. Builder's calls already made (for review)

- v1.5 is host-sequenced, and the paper says so (§3.1).
- P1 stays on v1.4; everything wider runs on the v1.5 ALU.
- The primary fidelity comparison uses the v1.4 pipeline configuration; larger workloads are secondary.
- A published-QwenPU controlled baseline is added to Phase F.
- The fast DeltaNet kernels go in a separate environment.
- The mandate's truncated Phase H list is completed from this session's prior-art search; paper and release are included because the mandate title names them.
