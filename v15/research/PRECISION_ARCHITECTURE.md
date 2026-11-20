# Precision architecture: candidates and verification level

Verification levels (mandate §4), lowest to highest:
1. Analytically proposed
2. CPU-reference verified
3. Neural-circuit verified
4. Saved-checkpoint verified
5. Complete-pipeline verified

Only levels 4 and 5 support claims about neural execution.

**"Complete-pipeline verified" for v1.5** means every token of a multi-stage RF pipeline was generated free-running by the compiled circuit on QwenPU's fused executor and equals the RV32 reference. The equivalence of that executor with the saved checkpoint's own Hugging Face layers is established separately, by the published-checkpoint run (`v15/reports/qwen-checkpoint-final.json`).

| Candidate | Realization | Highest level reached | Evidence | Numerical outcome |
|---|---|---|---|---|
| P1 Q3/W16 (control) | v1.4 autonomous 11-layer processor | Saved-checkpoint + complete pipeline | `v14/reports/*` | 0/14 real-IQ screen |
| P2 Q15 | QwenPU RV32IM CPU: 16-bit codes in 32-bit registers, exact MUL | Complete pipeline (executor) + saved published checkpoint | `real-iq-q15.json`, `capability-campaign.json`, `qwen-checkpoint-final.json` | 8/14 primary; arithmetic NMSE ~1e-9 |
| P2 Q7 | v1.0 compiler on the published checkpoint | Saved-checkpoint (v1.0) | `reports/GPU_COMPLETION.md` | Not re-screened in v1.5 |
| P3 multiword | 64-bit accumulators and powers from MUL/MULH with SLTU carry/borrow (FIR > safe bound, power, CFAR, IIR, CFO) | Complete pipeline | `capability-campaign.json` (fir64_wide, fft*, cfar, iir) | Exact |
| P3 bit-sliced ALU | v1.5 15-layer I16/W64 ALU checkpoint (host-sequenced) | Saved-checkpoint, per opcode (10/10 exact) | `native-validation.json` | Not used for pipelines: host sequencing and ~3–8k op/s |
| P4 block floating point | Shared exponent per block | Analytically proposed only | — | Not implemented |
| P5 mixed precision | Per-stage formats, listed below | Complete pipeline | `pipeline.json`, `vx-neural.json` | 13/14 with Q2.13 inputs (secondary) |
| P6 saturation vs wrap | Branch-free saturation and round-half-up everywhere | Complete pipeline | `test_filters.py`, `capability-campaign.json` | Behaviour verified; no controlled wrap-vs-saturate study |

The P5 per-stage formats are:
- Q15 or Q2.13 IQ samples;
- 32/64-bit accumulators;
- Q30 64-bit power;
- Q14 IIR coefficients;
- Q12 MLP weights;
- int8 activations with int4 weights on the VX vector unit.

## Precision selection in the compiler (implemented scope)

- **FIR (and DecimatingFIR):** a range bound, Σ(|h_re| + |h_im|)·2^15 < 2^31, picks exact 32-bit accumulation. Otherwise the kernel uses exact 64-bit accumulation. Tested both ways (`test_precision_selection`).
- **Static bound arguments in each kernel's docstring** prove the absence of overflow for:
  - FFT butterflies (|W·b| < 2^31);
  - CFAR (left-hand side < 2^41·train);
  - IIR (accumulator < 2^33);
  - CORDIC inputs (< 2^29);
  - MLP accumulators (64-bit).
- **Not implemented:**
  - a general range-analysis and precision-selection pass over a multi-stage IR;
  - the mandate's calibrated cost function. Per-kernel token costs are measured (`capability-campaign.json`), which is the input such a pass would need.

## Resource notes

| Artifact | Size | Notes |
|---|---|---|
| QwenPU published checkpoint, chip-only path | Peak ~12.5 GB on GPU (v1.0 measurement) | Hidden size 5120, 13 chip layers + Qwen's 32 |
| Fused executor (chipformer) | Weight-identical arithmetic on the live structure only | 0.03–0.07 ms per token at batch 16 on this laptop |
| v1.5 I16/W64 ALU | 807,933,280 parameters, 1.62 GB | — |
| v1.4 | 309,343,456 parameters | — |

The cost of moving from Q3 to Q15 on the CPU path is **zero extra parameters**: the same published weights run both. The cost is in tokens. A 4-tap complex FIR takes about 100 tokens per sample (see `capability-campaign.json`).
