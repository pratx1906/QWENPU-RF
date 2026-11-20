# Numerical failure analysis: v1.4 (Q3) vs v1.5 (Q15 / Q2.13)

## Data sources

| Version | Report | Execution |
|---|---|---|
| v1.4 | `v14/reports/real-iq.json` | CPU fixed-point oracle |
| v1.5 primary | `v15/reports/real-iq-q15.json` | Every sample on the CPU in weights, every token verified |
| v1.5 secondary | `v15/reports/real-iq-q13-headroom.json` | Every sample on the CPU in weights, every token verified (PROTOCOL Amendment 1) |

Common setup across all three:
- The same 14 captures and the same 4,096-sample spans.
- The same float references.
- The same criteria: FIR NMSE ≤ 0.01, spectral-power NMSE ≤ 0.01, matched-peak displacement ≤ 1.

## Results by capture

Column groups: v1.4 Q3; v1.5 primary Q15; v1.5 secondary Q2.13.

| Capture | Q3 input NMSE | Q3 FIR | Q3 spectral | Q3 peak | Q15 clipped comps | Q15 FIR | Q15 spectral | Q15 peak | Q2.13 FIR | Q2.13 peak |
|---|---|---|---|---|---|---|---|---|---|---|
| IDLab WiFi | 0.004 | 0.049 | 0.960 | 2907 | 0 | 3.3e-09 | 2.3e-09 | 0 | 5.0e-08 | 0 |
| IDLab WiFi | 0.001 | 0.019 | 1.312 | 360 | 0 | 1.6e-09 | 9.5e-10 | 0 | 2.1e-08 | 0 |
| IDLab LTE | 0.017 | 0.038 | 1.191 | 44 | 463 | 1.6e-02 | 4.8e-02 | 0 | 1.5e-08 | 0 |
| IDLab LTE | 0.001 | 0.020 | 1.385 | 314 | 0 | 1.5e-09 | 1.0e-09 | 0 | 4.0e-08 | 0 |
| IDLab DVB-T | 0.002 | 0.052 | 1.330 | 1230 | 0 | 2.1e-09 | 1.7e-09 | 0 | 3.4e-08 | 850 |
| IDLab DVB-T | 0.001 | 0.025 | 1.469 | 518 | 0 | 1.5e-09 | 1.2e-09 | 0 | 2.6e-08 | 0 |
| NTU-RFSR LoRa | 0.012 | 0.074 | 1.002 | 2308 | 109 | 2.3e-03 | 1.2e-02 | 1119 | 3.1e-08 | 0 |
| NTU-RFSR LoRa | 0.012 | 0.075 | 1.006 | 2117 | 107 | 2.2e-03 | 1.3e-02 | 2117 | 3.1e-08 | 0 |
| EPFL LoRa | 0.006 | 0.042 | 1.033 | 789 | 0 | 1.2e-09 | 1.5e-09 | 0 | 1.9e-08 | 0 |
| EPFL LoRa | 0.008 | 0.067 | 0.932 | 919 | 30 | 1.6e-04 | 7.5e-04 | 0 | 3.1e-08 | 0 |
| EPFL LoRa | 0.012 | 0.067 | 1.044 | 1572 | 104 | 2.4e-03 | 1.2e-02 | 23 | 3.0e-08 | 0 |
| EPFL LoRa | 0.004 | 0.027 | 1.521 | 3184 | 0 | 8.2e-10 | 1.1e-09 | 0 | 1.3e-08 | 0 |
| Tampere drone-ctrl | 0.015 | 0.067 | 1.007 | 877 | 152 | 3.7e-03 | 3.5e-02 | 3 | 2.7e-08 | 0 |
| Tampere drone-ctrl | 0.013 | 0.159 | 1.064 | 0 | 108 | 4.1e-03 | 1.4e-02 | 0 | 6.9e-08 | 0 |
| **Screen passes** | | | | **0/14** | | | | **8/14** | | **13/14** |

## Findings (each supported by the table)

1. **v1.4 failed mainly through intermediate narrowing, not input quantization.**
   - Q3 input NMSE was 0.001–0.017, but FIR NMSE was 0.019–0.159, typically 5–20× larger. Four-bit requantization and wrap after each multiply-accumulate added most of the error.
   - Spectral power NMSE was about 1.0 on every capture, so it carried no information. Squared four-bit bins wrap, and v1.4 already recorded negative spectral power. Peak displacements reached hundreds to thousands of samples.
2. **At Q15, the arithmetic contributes about 1e-9 NMSE.** On the eight captures without clipping, FIR and spectral NMSE are 8e-10 to 3e-9 and the peaks match exactly. That is about seven orders of magnitude below v1.4. Exact 32/64-bit accumulation and exact 64-bit power remove the narrowing and wrap errors.
3. **Every Q15 primary failure comes from front-end saturation.**
   - The v1.4 gain rule (0.75/p95) puts high-crest-factor captures (LTE, LoRa, drone links) beyond full scale. The six failing captures clip 30–463 components.
   - On each of them, input-quantization NMSE alone equals the FIR NMSE, so the neural pipeline adds nothing measurable.
4. **With 4× headroom (Q2.13), the clipping disappears and 13/14 pass.** The cost is about 4× coarser input resolution; all errors stay around 1e-8.
5. **The one Q2.13 failure is a near-tie, not an arithmetic error** (`v15/reports/peak-tie-diagnosis.json`).
   - On IDLab DVB-T, the two largest float matched-filter peaks differ by 1.4e-4 relative. That is below one Q2.13 output step.
   - FIR and spectral NMSE are about 3e-8 there. A global argmax over a noise-like OFDM record is a fragile criterion; it is kept, not relaxed.

## Conclusions

- **Precision.** 16-bit operands with exact wide accumulation fix the v1.4 arithmetic failure entirely. The remaining failures are front-end design choices (input scaling and headroom) and criterion fragility.
- **Scope of the gain-rule choice.** No single fixed gain rule both avoids clipping on peaky signals and keeps resolution on weak ones. Choosing the input format per signal class (P5 mixed precision at the input) is the natural next step. It is not claimed here.
- **Not measured.**
  - A per-operation error decomposition of every intermediate (shift, wrap, depth) for v1.4. The table separates input and pipeline error only.
  - Errors propagated into inference accuracy on real data. No labelled real data exist; the synthetic classifier result is in `v15/configs/spectral-mlp.json`.
