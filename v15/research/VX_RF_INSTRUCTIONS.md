# New RF instructions for QwenPU's VX unit: design (2026-11-04)

> **Status:** this top part is the original design study, now superseded. VFIR, VPOW and VBFLY were all built afterwards; see the implemented sections below, [amendment6.json](../reports/amendment6.json), [amendment8.json](../reports/amendment8.json) and [vx-rf-checkpoint.json](../reports/vx-rf-checkpoint.json).

The study was read-only: `core.build(vx=True)` was probed on the CPU. The build happens in a separate QwenPU worktree on branch `rfpu-vfir`; the pinned upstream checkout is not modified.

## Expected gains (tokens per sample)

| Workload | Scalar RV32 today | With the new ops | Gain |
|---|---|---|---|
| FIR, 4 taps | ~100 | ~34 | 3× |
| FIR, 16 taps | ~650 | ~82 | 8× |
| FIR, 64 taps | ~2600 | ~300 | 8.7× |
| FFT64 power | ~400 | ~56 | 7× |
| Energy | ~20 | ~14 | 1.4× (limited by the cost of loading samples into VM) |

A later 16-bit VM element mode would roughly double these.

## Facts that constrain the design

**Encoding is free.**
- `ref.rom_word` pre-decodes instructions to 7-bit uops, and uops 81–127 are unused (`ref.py:66-73`).
- `core.py:456-460` builds the uop nodes from `RF.UOPS` automatically.
- Proposed encoding: custom-2 `0x5B` with funct7=1, where funct3 0, 1 and 2 select VFIR, VPOW and VBFLY (as built; the study first proposed VBFLY = 1). Only `ref.decode` (`ref.py:146`) needs to change.

**Circuit budget.**
- The circuit has 13 layers, and MLP levels 10–16 are full (8,616 = 9,216 − 600 reserve).
- The L2 heads are full (16/16); L1 has 5 free heads.
- So the new ops must reuse the ROPE multiply chain (`vx.mac.*`, levels 4–22), which does not add depth.

**Addressing and readback.**
- Addresses must be concatenations of fixed functions of the element counter (`docs/vx.md:92-95`).
- VM is never read back to the CPU directly. Results return as a value record into rd, the way DIV does (`core.py:1121-1130`).

**Flags and gating.**
- Slot 1 bit 11 is 0 in every existing op, so it can serve as a "new op" flag.
- VEW's gates (`vxcirc.py:425-426`) must be extended with `[1-NEW]` so they don't fire during the new ops.

**Shared datapath.**
`out = ((A<<15) ± (W1·F1 + W2·F2) + 2^(14+sc)) >> (15+sc)`, with |A|, |out| < 2^23 (ROPE columns are 40 bits wide).

## VFIR: transposed-form complex Q15 FIR, one input sample per call

**Fields**
- `rd` receives the output: packed `sat16(re) | sat16(im)<<16`, or NT when rd = x0.
- `rs1` holds the packed input sample x.
- `rs2` holds: dst, src (bank A state), h (bank B taps, stored reversed), n−1, eoff, chain/cin, fin.

**Semantics**, for j < n:
- `z = state[j-1]` (for j = 0: the chained block's last pair, or 0);
- `y_j = z + round_half_up(h_j·x / 2^15)`, computed per component as a complex product;
- `y_j` is written to `dst[j]`;
- the stream carries 2 words, which is 4 tokens per tap.

**Behaviour**
- Ping-ponging src and dst gives the identity `rd = Σ_k round_half_up(h_k·x_{n−k} / 2^15)`. That is an independent oracle for testing.
- n ≤ 16 fits in one op. 64 taps take 4 chained ops.
- Cost is about 4N + 18 tokens per sample.

## VBFLY: one 8-butterfly block of a constant-geometry radix-2 stage

The algorithm was checked numerically in float for N=64 at design time. As built (track mode for block floating point): 69.5–71.5 dB at every level −60…0 dBFS, 1.8× faster than the CPU block FFT ([amendment6.json](../reports/amendment6.json)).

**Semantics**
- Each butterfly computes `y[2i] = sat16((a·2^15 + w·b + r) >> (15+sc))` and `y[2i+1]` with `−w·b`, where a = x[i] and b = x[i+N/2].
- Twiddles come from `TW[r] = W64^brev5(r)` with index `i & mask_s`. They live in the ROPE head, grown from 4 rows to 32 inside the same L2 head.
- Output is bit-reversed.

**Cost:** about 27 tokens per sample for FFT64.

## VPOW

- Computes `acc = Σ(ar² + ai²) mod 2^32` over a VM register and puts it into rd.
- Uses the running sum through lags, like VMAT's `pacc` (`vxcirc.py:331`), with wt = W_SUM.
- Costs about 3 + 2n tokens.

## Risk

**Width and layers.** The new ops need roughly:
- ~1.5k units at level 2;
- ~0.3k in the full levels 8–16;
- ~0.5k at levels 20–24.

That gives a medium-low risk of a 14th layer, which must be measured.

**Open questions**
- Whether the rd commit (`reg_beta`, `core.py:492-508`) accepts F-records after a VX stream.
- Why MAX_STREAM is 96. The design keeps each op at ≤ 64 stream tokens.
- Executor rounding with the A<<15 column terms.

## Implementation plan (VFIR first)

1. **Reference.** Add the VFIR reference to `vx.py`, with a pytest against the independent direct-form oracle (random taps including ±32768, n ∈ {1, 2, 4, 8, 16}, chaining).
2. **Decode and assembler.** Update `ref.py` (CUSTOM 81, decode and dispatch) and `asm.py` (VX2_ASM).
3. **Circuit, in `vxcirc.py`.**
   - Add the NEW flag and `[1-NEW]` gating, then confirm `test_rv_vx.py` shows no regression.
   - Add the shifted-pair query, the fir cases in pick/f1/f2/sft, the A<<15 terms and the rounding constant.
   - Add the exit cases, lag head 0–3, packing and the value-record cases.
4. **Circuit tests.** `run_check` on random VFIR programs for seeds 0–3. Record `circ.stats()` before and after.
5. **Kernel and pipeline.** Add a `VXFir` kernel in v15, measure tokens against scalar, then run free-running on the executor, then merge into a checkpoint and run a checkpoint check.
6. **Then the others:** VPOW next, then VBFLY.

## VFIR: implemented and verified (2026-11-04)

Branch `rfpu-vfir`, worktree `upstream/QwenPU-rf`, commits `1dad4e4`, `6cd06ad`, `91a781d` and `c5d55c1`.

**Circuit size.** `circ.stats()` for the VX build:

| | Layers | Dims | Heads | FFN units | Max units |
|---|---:|---:|---:|---:|---:|
| Before | 13 | 6,675 | 137 | 78,460 | 8,616 |
| After | 13 | 6,828 | 137 | 79,560 | 8,616 |

No new layer and no new head.

**Verification ladder**

| Level | Result |
|---|---|
| Reference vs independent direct-form oracle | 44 tests |
| No regression | `test_rv_vx` 21/21, `test_rv_core` 94 passed, ref tests 174 |
| Circuit, teacher-forced IR (`run_check`, every token) | 4 random programs, 52 VFIR calls (chained, rd = x0 and rd ≠ 0); a mutation test (rounding removed) is caught |
| Fused GPU executor, free-running | 4/4 exact |
| **Saved checkpoint** `rfpu-v15/qwen3.5-4b-vx-vfir` (Qwen3.5-4B + VX + VFIR, 27 GB, merged by `chipformer.rv.merge --vx` from the branch) | Chip layers on the GPU (HF modules), free-running: 4 VFIR programs plus a 16-tap `VXFir` on real IQ, **5/5 token-exact** ([vfir-checkpoint.json](../reports/vfir-checkpoint.json)) |

The checkpoint's circuit metadata matches the circuit rebuilt from the branch.

**Real IQ, neural executor, both kernels token-exact** ([vfir-neural.json](../reports/vfir-neural.json))

| Taps | Scalar RV32 (tokens/sample, ms/sample) | VFIR (tokens/sample, ms/sample) | Gain |
|---|---:|---:|---:|
| 4 | 104.3, 8.74 | 57.7, 4.90 | 1.8× |
| 8 | 195.3, 16.67 | 75.2, 6.44 | 2.6× |
| 16 | 633.5, 53.55 | 112.7, 9.11 | **5.9×** |

**Contract difference.** VFIR rounds each tap product separately and saturates rd to int16. The scalar FIR uses exact accumulation and a single rounding. Outputs therefore differ by a few LSB, so VXFir has its own independent reference.

**Below the design estimate.** The gain is lower than the ~8× estimated for 16 taps because:
- each chunk re-runs the N−1 history samples;
- packing and unpacking cost about 20 tokens per sample.

A packed-input layout or a persistent VM state across chunks would remove most of that overhead.

## VFIR packed I/O and timing study (2026-11-04)

The host lays out each input as one word `re | im<<16`, removing the per-sample pack/unpack. Real IQ, executor, both arms token-exact ([vfir-neural.json](../reports/vfir-neural.json)):

| Taps | Scalar tokens/sample | VFIR packed tokens/sample | Time gain |
|---|---:|---:|---:|
| 4 | 104.3 | 40.4 | 2.6× |
| 8 | 195.3 | 57.7 | 3.5× |
| 16 | 633.5 | 94.8 | **6.7×** |

Pre-declared timing study (PROTOCOL Amendment 3, 8 independent processes, A-B-B-A): geometric-mean speedup **6.73×**, 95% t-CI [6.727, 6.738], exact sign-flip p = 0.0078 (the minimum possible with 8 units) ([timing-study.json](../reports/timing-study.json)).

## VPOW: implemented and verified (2026-11-04)

Encoding: custom-2 (0x5B), funct7 = 1, **funct3 = 1** (VFIR is funct3 = 0), uop 82. It shares VFIR's MAC chain and slot-1 NEW flag. No new layer or head. Commits `4255937`, `86e3385`, `0e64cfb`.

| Level | Result |
|---|---|
| Reference vs oracle | `test_rv_vpow_ref.py` |
| Circuit, teacher-forced (every token) | `test_rv_vx.py::test_vpow`, random n 1..32, extremes ±32768, interleaved with VFIR/VEW/VQ/VPUT |
| Fused executor, free-running | **8/8 programs token-exact**, ~2k tokens each ([vpow-executor.json](../reports/vpow-executor.json)) |
| Saved checkpoint | **Exact** on `qwen3.5-4b-vx-rf` (`vpow_program`, 2,075 tokens; 5/5 programs with VBFLY), [vx-rf-checkpoint.json](../reports/vx-rf-checkpoint.json) |

Gain: 8.8× vs scalar energy when the samples are already in VM; ~1× when they must first be loaded from RAM with VPUT, because loading dominates. It helps only inside a VX chain (e.g. after VFIR).
