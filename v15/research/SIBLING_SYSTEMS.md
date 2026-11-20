# Sibling "compute compiled into LLM weights" systems vs RFPU v1.5 (2026-11-04)

This was read-only research. Nothing was cloned or run. Sources were READMEs, docs and source files read on GitHub, raw.githubusercontent.com and the GitHub API, plus Hugging Face cards, arXiv and blog posts.

- **[D]** marks a documented fact, quoted or read from source.
- **[E]** marks my estimate. Every [E] number shows its arithmetic and has an uncertainty of about ±50% or more.

Our reference numbers (measured, from v15 README / DSP_METRICS / VX_RF_INSTRUCTIONS):

| Kernel | Tokens/sample |
|---|---:|
| Scalar complex FIR16 | 353–633 |
| VFIR 16 taps | 113 |
| FFT64 power | 404 |
| Energy | 19 (VX: 15) |
| CA-CFAR 64 bins | 603 |
| Spectral MLP | scalar 266–416, VX 125 |

Speed is 7.4k tok/s at batch 1 and 33k tok/s at batch 32 on the fused executor, and ~19 ms/token on HF layers (RTX 5090 Laptop).

Unit for comparison is **tokens per real MAC**:
- scalar FIR16: 353–633 / 64 = 5.5–9.9;
- VFIR16: 113 / 64 = 1.8.

## 1. Systems found

| System | URL | Base model | What is compiled in | License | Maturity |
|---|---|---|---|---|---|
| **qwenGPU (qgpu)** | https://github.com/vixhal-baraiya/qwenGPU | Qwen3.5-2B [D] | 4-lane SIMT GPU that runs a CUDA→PTX subset; 45 circuit layers in front of Qwen's 24 [D] | MIT code, Apache-2.0 weights [D] | "experimental"; 7 commits; no published checkpoint found (`just build` merges locally) [D] |
| **qwenTPU** | https://github.com/Bobliuuu/qwentpu | Qwen3.5-0.8B/2B/4B [D] | Weight-stationary matvec array; 4 appended layers [D] | MIT [D] | 7 commits, 1 star; no release; "small machine today" [D] |
| **Qwenputer** (parent of QwenPU) | https://github.com/divagr18/qwenputer, https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer | Qwen3.5-4B [D] | RV32IM+Zba+bignum CPU, 11 layers; `npu` branch adds VDOT (8×int8·int4) and SMAQA [D] | MIT / Apache-2.0 [D] | riscv-tests pass; 18.8 GB bf16; ~3.2k tok/s (Triton) and ~5.5k tok/s (fused), both on L4 [D] |
| **QwenPU** (our host) | https://github.com/divagr18/QwenPU | Qwen3.5-4B [D] | Qwenputer CPU plus VX vector unit (int8 activations, int4 weights; 13 layers) [D] | MIT / Apache-2.0 | ~3.3k tok/s on L4 [D]; 26.7 GB checkpoint |
| Forks (olmputer, TheFausap, kustomzone, rpruizc, aripiprazole) | e.g. https://github.com/JGalego/olmputer | Qwen3.5-4B | Copies of Qwenputer; no documented differences [D] | same | forks |
| **Percepta transformer-vm** | https://github.com/Percepta-Core/transformer-vm, blog https://www.percepta.ai/blog/can-llms-be-computers (2026-03-11) | none (standalone analytic transformer, not an LLM) | WebAssembly interpreter, 35 opcodes; optional "specialize" mode bakes the program into FFN weights (Futamura) [D] | Apache-2.0 [D] | 1 commit, ~240 stars [D] |
| **Loom** | arXiv 2604.08816, https://github.com/mkturkcan/Loom | none (looped 8-layer transformer, ~4.7M params) [D] | 22-opcode ISA including SHL, SHR and a MULACC "step"; state lives in one tensor; one forward pass per instruction; ran on an Alveo U200 FPGA and in WebGPU [D] | Apache-2.0 [D] | 6 commits, 7 stars [D] |

I searched for TPU, GPU, FPGA and DSP variants. The only other hit was the Qwen3 FPGA runtime (https://huggingface.co/Tyler01/qwen3-0p6b-fpga-q4-runtime), which is not compute compiled into weights. I found no RF or DSP-specific sibling.

## 2. Architecture details (documented)

### qwenGPU

| Item | Fact [D] | Source |
|---|---|---|
| Machine | SIMT, 15 usable registers/lane (16, r0 = zero), 8 predicates, program ROM of 32 blocks, global memory 126 words (`ABITS = 7`, address 127 = SINK), <32 warps, ≤128 threads | README, `docs/qgpu-cuda.md`, `qgpu/circuit/kernel.py` |
| Lane count | Merged default `lanes=4, stages=4` (`merge.py`); circuit supports 2/4/8 (`LANES = 8` constant) | `qgpu/merge.py`, `kernel.py` |
| Pipeline | 4 stages: `STAGES[4] = ({2},{1},{3})`, i.e. mul only in stage 2, logic in stage 1, atomics in stage 3. **At most ~1 multiply per block per warp** | `merge.py` |
| Data type | 32-bit int, two's complement, wraps. No float, no 64-bit. Division only by a launch constant | `docs/qgpu-cuda.md` |
| Multiply | `mad rd = ra*rb + rc` (low 32 bits); `mulhi` (unsigned high 32; signed via correction); "64-bit multiplier" built from nibble columns | `kernel.py`, `lower.py` |
| Shifts | `shl` → mad by 2^k; `shr.u32` → 1 `mulhi`; **`shr.s32` ≈ 5 machine instructions**; `min`/`max` are 1 instruction each; `selp` is 2 | `lower.py` |
| Token stream | Host: `PROG(b,s) f0 f1 PROGI(b,s) i0 i1` = 6 tokens/instruction; `MEM(a) lo16 hi16` = 3 tokens/word; then `EXEC(0,0,True)`. Execution: `EXEC(b,w)` followed by **2 VAL tokens per lane per live-out register** (16-bit halves) | `runtime.py`, `kernel.py` |
| Results | The model writes MEM records; the host reads arrays from recurrent state with no arithmetic | README |
| Exactness and verification | "Every value exact in bf16"; `--check` replays each launch on a reference interpreter that matched a Tesla T4 on 49/49 kernels; text KL ~7e-4 | README |
| Speed | **10–15 tok/s on MPS (Apple, bf16, HF path)**. No CUDA or fused-executor figure. Needs ~16 GB | README |
| Kernels | addk, bitops, branchy, collatz, colsum, div*, dot, histogram, matmul, mean, mulk, reduce, saxpy, scan, shfl, vadd/vsub/vmul/vmax, warpsum (all int) | GitHub tree API |

### qwenTPU

| Item | Fact [D] |
|---|---|
| Datapath | Weights sit in Gated DeltaNet state, one row per input. Products are bit-ANDs of activation bits and weight bits, and accumulators count bit planes. SwiGLU does ReLU, clamp and requantise |
| Data types | Activations u3 (0..7); weights s3, or s2 at 16×16. Accumulation is "exact i32", but readout is bounded by a comparator window of **\|y\| ≲ 256** and requantised to 3 bits. Wider operands (carry-save adder tree) are "designed, not built" |
| Tokens | `ADDR WVAL` per weight, loaded once; `CLS`; one `ACT(i,v)` per input; one `READ(c)`→`OUT(v)` per output. **KR + NC + 1 tokens per matvec**: 8×8 = 17 (3.8 MAC/token); 16×16 = 33 (7.8 MAC/token) |
| Speed | Teacher-forced/replayed, not free-running. 0.8B 8×8: 2.9k tok/s on A100 at batch 1, 38.8k at batch 16; 4B 8×8: 2.3k / 13.1k. "Free-running single stream is not faster"; no fused executor |
| Verification | 0 mismatches vs NumPy (11,760 outputs at 0.8B on CPU; "100k+ replayed"); mutation test; text KL 3–8e-4; `HybridSession` restores the cache after each chip run |

### transformer-vm and Loom

**transformer-vm**
- WASM i32 with byte-level carry arithmetic. MUL, DIV, AND, OR, XOR and the shifts are **lowered at compile time** rather than run as native ops. [D]
- "At most 5 tokens per instruction"; >30k tok/s on a CPU with the C++ engine. [D]
- Hard-max attention with an O(log n) convex-hull KV cache. [D]

**Loom**
- One forward pass per instruction. Its multiply is the MULACC "step" (iterative). [D]
- Word width is not stated. [D]

## 3. Can our workloads run on them? (estimates)

Cost model for qwenGPU [E], derived from the documented token format:
- A block (≤4 instructions, ≤1 multiply) costs `1 + 2·L·LO` tokens, where L = 4 lanes and LO = live-out registers (assume 2).
- That gives 17 tokens per warp-block, or **~4.25 tokens per lane-block**.
- A multiply-bound kernel therefore costs ≈ 4.25 tokens per multiply per lane.

### Expressibility and estimated tokens/sample

| Workload | qwenGPU | qwenTPU | transformer-vm | Ours (measured) |
|---|---|---|---|---|
| Complex Q15 FIR16 | Expressible: 4 `mad` per tap (exact 16×16→32, low word). Round-half-up by add 2^14 then `shr.s32` (~5 instructions), or bias-to-unsigned + 1 `mulhi`. Saturate with min/max. 32-bit acc is safe only if the range bound holds (16 complex taps can reach 2^35; 64-bit acc needs `mulhi` pairs, ~2× cost). Memory: 126 words = 32 tap words + 4n + 30 ≤ 126 → **≤16 outputs per launch**. Est: 64 mul-blocks × 4.25 = 272, + round/sat ~17, + I/O 6+6, + amortised program/taps ~(240+96)/16 = 21 → **~320 tok/sample (range 250–450) [E]** | Not expressible: u3×s3 with \|y\| ≲ 256. Q15 needs ~6×6 = 36 digit-plane products and host recombination, which our contract forbids | Expressible in C, but i32.mul is lowered in software. A shift-add multiply of ~32 iterations × ~6 instructions × ≤5 tokens ≈ 1,000 tokens/multiply → **~6×10^4 tok/sample [E, order of magnitude]** | 353–633 (scalar), **113 (VFIR)** |
| Radix-2 FFT64, Q15 twiddles, power | 64 complex samples = 128 words > 126, so it **does not fit** unless re/im are packed into one word. Unpacking costs ~6 instructions per half (`shr.s32`). If packed: 192 butterflies / 4 lanes = 48 warp-groups × ~9 blocks × 17 = 7.3k, plus power ~13/bin, plus I/O ~10 → **~140–250 tok/sample [E]**. Needs `__syncthreads` per stage (supported) | Only as a DFT matvec, and precision is impossible (see FIR) | ~10^4+ [E] | 404 scalar; VBFLY designed at ~27–56 [E, ours] |
| Energy / power | Unpacked I/Q: MEM 6 + ~2.5 blocks × 4.25 ≈ **16 tok/sample [E]** | ✗ | — | 19 (VX 15) |
| CA-CFAR 64 bins | 64 bins + 64 flags = 128 > 126 words, so it needs 2 launches. A per-cell sum of 16 training cells ~10 blocks × 4.25 + I/O ≈ **~50 tok/bin [E]** | ✗ | — | 603 (per sample as reported) |
| Integer MLP | `matmul` kernel exists; ~4.25 tok/MAC (int32 exact) [E] | Native, but only u3 activations and s2/s3 weights: **0.13–0.26 tok/MAC [D]** | — | VX 125 / scalar 266–416 per sample |

### Exactness against our Q15 contract

| Requirement | qwenGPU | qwenTPU | transformer-vm |
|---|---|---|---|
| Exact 16×16→32 signed product | Yes (`mad` low word) [D] | No (3-bit) [D] | Yes, slowly [D/E] |
| Complex | Via 4 mads [E] | No | Via software |
| 64-bit accumulation | Only via `mulhi` + carry emulation (no 64-bit type) [D] | No | i64 not stated |
| Round-half-up, arithmetic shift | ~5–6 instructions [D] | No | Lowered |
| Saturation | min/max, 1 instruction each [D] | Clamp to 7 only | Branches |

**Wall clock.** qwenGPU's only documented speed is 10–15 tok/s on MPS through HF. On our HF path (~19 ms/token, about 50 tok/s) an estimated 320 tok/sample would take ~6 s/sample. Even a hypothetical fused executor at our 7k tok/s gives ~45 ms/sample, which is **no better than our scalar FIR and 4× worse than VFIR (9.1 ms)**.

## 4. Could their approach beat us at equal accuracy?

- **SIMT lanes (qwenGPU).** Tokens scale with *lanes × live-outs*. VAL tokens are per lane, so lanes amortise only the 1-token EXEC header. The real saving is that intermediates inside a block are never emitted, roughly 1 token per lane-instruction versus our ~2–4 per scalar RV32 instruction. That is the same principle as VX fused ops, and VFIR already captures it better (1.8 tok/MAC vs ~4.25 [E]).
  - **Verdict:** no speed-up at Q15 exactness for FIR. FFT could beat our *scalar* FFT (est. 140–250 vs 404) but not VBFLY (est. 27–56). It is limited by the 126-word memory and the 2B base model.
- **Systolic, weight-stationary matvec (qwenTPU).** **KR + NC + 1 tokens per matvec**: MACs cost no tokens, only inputs and outputs do. It is the most token-efficient structure seen (up to 7.8 MAC/token), but only for 2–3-bit operands with \|y\| ≲ 256 in bf16. Q15 would need multi-digit planes and carry-save trees that are not built.
  - **Verdict:** not usable for our contract today. The *idea* is the strongest one to borrow (see below).
- **DFT as a matvec.** For N=64 this is 128×128 = 16,384 real MACs per frame. Even at qwenTPU's 7.8 MAC/token that is 2,100 tokens per frame = 33 tok/sample [E], at 3-bit precision. Q15 via 6×6 digit planes is ×36 ≈ 1,200 tok/sample [E]. Radix-2 VBFLY (N log N) is better at N ≥ 16. A DFT-as-VMAT is only competitive for N ≤ 8–16 bins or a few Goertzel/single-bin DFTs.
- **transformer-vm / Loom.** These are general-purpose machines with software multiply, so they are orders of magnitude worse per sample. Neither is an LLM checkpoint. Speed (30k tok/s CPU) comes from a tiny model, which our 4B host cannot copy.

## 5. Ideas worth borrowing into QwenPU/VX

| # | Idea (source) | Applied to us | Expected gain [E] | Feasibility | Rank score (gain × feasibility) |
|---|---|---|---|---|---|
| 1 | **Weight-stationary streaming** (qwenTPU: weights resident in DeltaNet, 1 token in, 1 token out) | "VFIR-stream": taps and delay line persist in VM/DeltaNet across chunks. One packed input token and one packed output record per sample. Removes the N−1 history re-run and ~20 pack/unpack tokens we measured | 113 → ~40–60 tok/sample (2–3×) | High: extends a verified op; needs persistent VM state across chunks | **Highest** |
| 2 | **Emit only live-outs per block** (qwenGPU) | Finish VBFLY and VPOW as fused ops that emit only stage outputs. FFT64 404 → ~27–56 (our design estimate). VPOW: energy 19 → ~10 | 7–15× on FFT; ~1.5–2× on energy | Medium: VBFLY twiddle ROM must grow and the layer budget is tight | High |
| 3 | **Program/constant specialisation into weights** (transformer-vm "specialize", Futamura) | Bake fixed FIR taps and FFT twiddles as circuit constants (ROPE/twiddle rows) rather than loading them per run. Saves tap-load tokens and frees VM | Small per sample (amortised loads); larger for short chunks | Medium: needs a per-filter checkpoint, conflicts with runtime taps | Medium |
| 4 | **Bias-to-unsigned rounding** (qwenGPU `shr.s32` cost) | In scalar RV32 kernels, use `(x + 2^14 + 2^31) >>> 15` minus a constant instead of arithmetic-shift fixups, where the codegen does extra work | ~5–10% on scalar kernels | High: codegen only | Medium-low |
| 5 | **Cache snapshot/restore around chip runs** (qwenTPU `HybridSession`) | Stage 2 tool calls: restore Qwen's cache after a computer run to remove the text-perturbation risk | Quality, not speed | High | Side benefit |
| 6 | **Small-N DFT as matvec** | A Q15-weight VMAT (16-bit weights via two int8 digit passes) for single-bin/Goertzel or N ≤ 16 | ≤2× on FFT16 | Low-medium: VMAT is int8×int4; widening needs digit splitting | Low |
| 7 | Port to qwenGPU | Run our kernels on the SIMT GPU | ≈1× on FIR (est. 320 vs 113–633); maybe 2× on FFT vs scalar only | Low: 126-word memory, no 64-bit, no fused executor, 2B base, 10–15 tok/s documented | Lowest |
| 8 | Port to qwenTPU / transformer-vm / Loom | — | <1× | Very low: precision or software multiply | Reject |

**Bottom line.** No sibling system runs Q15 complex DSP faster than our VFIR path. qwenGPU is exact for 32-bit integer math but costs about 4 tokens per MAC [E] and holds only 126 words. qwenTPU is the most token-efficient per MAC, but only at 3-bit precision. The best transfer is architectural: qwenTPU's weight-stationary streaming applied to VFIR, then the planned VBFLY/VPOW fused ops.
