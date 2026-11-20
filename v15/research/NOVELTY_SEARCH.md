# Novelty search for the v1.5 claim (2026-11-04)

## The claim, as it can be defended

> To our knowledge, this is the first demonstration of a language model performing exact, independently verified radio-frequency signal processing and inference on real IQ recordings within its own weights, and the first RF-specific instruction added to a computer compiled into a language model's weights.

"Exact" is defined below, and the qualifiers that follow are part of the claim.

## Qualifiers that must accompany the claim

1. **In-weight computers are prior art, not ours.** Qwenputer, QwenPU, qwenGPU, qwenTPU, Percepta transformer-vm and Loom all compile computers or interpreters into transformer weights. Our contribution is the RF DSP and inference built on QwenPU's CPU and vector unit, the Qwen-in-the-loop use, the real-IQ evaluation, and the new VFIR instruction.
2. **Generic numeric kernels already exist in that work.** qwenGPU runs dot products, matmul, saxpy, reductions, scans and histograms. None of these projects shows FFT, FIR, spectral, detection or demodulation work, or any radio or IQ data.
3. **What "exact" means.** Results are bit-exact to a declared Q15 fixed-point contract, and every token is checked against an RV32 reference machine. They are not exact relative to floating point; the float error is reported separately, for example 78.5 dB SQNR for the FIR.
4. **What "within its own weights" means.**
   - On the 19 GB merged Qwen3.5-4B, every chip token runs through the model's own saved layers on one GPU, with 0 lock errors and 0 head mismatches.
   - Large campaigns used QwenPU's executor (the same circuit). Its agreement with the saved layers was checked on representative programs.
5. **Scope of the search.** It was focused, not systematic. It is recent: several sibling projects appeared within the last week, and the field is moving fast. It covers repositories and preprints, not peer-reviewed literature alone.

## Method (2026-11-04)

**Web searches.** Queries combined "compiled transformer weights" or "in the weights" with FFT/FIR/signal processing; named in-weight computer projects with DSP, radio or IQ; LLMs executing programs in the forward pass with RF or spectrum analysis; and compiled or analytically constructed transformers with FFT or FIR.

**GitHub API and README checks.** For each repository we recorded creation and push dates and grepped the README for fft, fourier, fir, filter, signal processing, radio, spectrum, dsp and IQ. qwenGPU's `kernels/` directory was listed in full.

**Earlier review.** This session's prior-art review (see the PRIOR_ART and RELATED_WORK matrices) covered Loom, ALTA, Tracr/RASP, Giannou et al., looped ReLU computers, deep unfolding, Butterfly-Net, FNet-style Fourier layers, AMD Versal/Hexagon/AIR-T, RF-GPT, RF-Analyzer, SignalLLM, WirelessAgent, Agentic RF Intelligence and SensorBench.

## Findings

| Work | Date | What it shows | DSP / RF? |
|---|---|---|---|
| [Qwenputer](https://github.com/divagr18/qwenputer) | created 2026-10-19 | RV32 CPU inside Qwen3.5-4B; calc tool bridge | No DSP or RF in the README |
| [QwenPU](https://github.com/divagr18/QwenPU) | created 2026-10-31 | Vector unit; stories260K inside Qwen3.5-4B | No |
| [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) | created 2026-11-01 | SIMT GPU (45 compiled layers) inside Qwen3.5-2B, running CUDA/PTX | Kernels are generic: dot, matmul, saxpy, reduce, scan, histogram, vector ops. **No FFT, FIR, complex, RF or IQ** |
| [qwenTPU](https://github.com/Bobliuuu/qwentpu) | created 2026-11-02 | Matrix-multiply array in the weights | No |
| [Percepta transformer-vm](https://github.com/Percepta-Core/transformer-vm) | created 2026-03-25 | WebAssembly interpreter compiled into weights | No |
| [Loom](https://arxiv.org/abs/2604.08816) | 2026-04 | Analytical neural computer: C programs, 22 opcodes | No |
| FNet, SPECTRE, Butterfly-Net, trained-DFT nets | — | Fourier layers that are learned or fixed, not an exact programmable DSP | Approximate; not RF analysis by an LLM |
| Agentic RF / RF-GPT / RF-Analyzer / SignalLLM | 2025–26 | LLMs reasoning about RF | DSP runs outside the model (external tools or a conventional front end) |

## Correction to an earlier statement

We wrote that only Qwen3.5-4B "can host the CPU". The precise version: among the Qwen3.5 sizes, only the 4B has the head layout (2 value heads per key head) that **QwenPU's CPU circuit** needs unchanged. qwenGPU compiles a different circuit into Qwen3.5-2B, so smaller hosts are possible with a different construction.


## Re-check (2026-11-12)

Method: web search (extended) for in-weight computers + DSP/RF/FFT, arXiv 2026, RF+LLM modulation work; GitHub API: qwenputer/QwenPU commits and forks, repositories mentioning Qwenputer created after 2026-11-01 (45), and keyword searches (radio/DSP/FFT + compiled into weights: 0 hits).

**Result: still no in-weight signal processing or RF work found.** No peer-reviewed or arXiv paper on a computer compiled into a chat LLM found (Qwenputer added an ORCID and software citation on 2026-11-05; no paper).

New since 2026-11-04 (all generic or other domains):
- Domain units: QwenQPU (quantum circuits, Qwen3.5-4B, 2026-11-03), qwenFS (filesystem, 2B), nesformer (NES CPU, 2026-10-30), pixelcore (game engine), qwentris (Tetris as a direct circuit), graph-qwenputer, VEC4 (vector unit in 0.8B/2B with a tool-call bridge).
- Other hosts: rpn-gpt (Forth machine in NVIDIA Nemotron), qwenlab (Qwen3.5-9B), concept-rv32im and Qwenputer on Qwen3.8-27B, qwenpu-bonsai (QwenPU port to Bonsai-27B). **Our "only Qwen3.5-4B can host the CPU" wording must stay qualified as "QwenPU's circuit unchanged among small Qwen3.5 sizes"; other constructions now reach 0.8B–27B.**
- Tool use: qwenputer-handover fine-tunes Qwen to hand arithmetic to the in-weights CPU; QwenFX runs a coding agent on it. Our "Qwen calls the in-weights tool" is not unique as a mechanism; the RF tools and verified RF reports are.
- RF + LLM (learned, not exact): RF-GPT (2602.14833), Seeing Radio (2601.13157, 91.2% AMC with a VLM), DiSC-AMC (2510.00316, 83% AMC with a 7B LLM), RF-Analyzer (2605.04676). Unrelated "RF computing of weights": AIR-LLM (2610.00465), MIWEN (2506.12210).

Implication: the RF/DSP application remains open, but the substrate has 40+ derivatives in about 10 days; a domain-specific port (as QwenQPU did for quantum) is the obvious next move for others, so the priority window is short.
