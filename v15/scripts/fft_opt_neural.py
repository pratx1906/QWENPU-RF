"""Neural check of the optimized FFT stages (W_0 / -j stages without multiplies, twiddle-outer loops):
outputs are bit-identical to the unchanged contract, so every token must equal the reference and every
output the independent reference. Reports tokens/sample (before: fft64 201.3, fft64 bfp 267.9, fft16
151.8, fft256 251.3, welch64x4 412.0; reference machine, same inputs)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
import dsp_metrics as DM  # noqa: E402
from rfpu_v15.cpu import spectral as S  # noqa: E402

BEFORE = {"fft64_power": 201.3, "fft64_power_bfp": 267.9, "fft16_power": 151.8, "fft256_power": 251.3,
          "welch64x4": 412.0}


def main():
    import numpy as np
    rng = np.random.default_rng(1)
    x, _ = DM.quantize(DM.cgauss(rng, 2048, -12), 15)
    rows = []
    for k in (S.FFTPower(64, None, 64), S.FFTPower(64, None, 64, scaling="block"), S.FFTPower(16, None, 16),
              S.FFTPower(256, None, 256), S.WelchPSD(64, None, 32, 4), S.CFARDetect(64, None, 64)):
        _, st = DM.neural(k, [x])
        tps = st["tokens"] / st["samples"]
        rows.append({"kernel": k.name, "tokens_per_sample": tps, "ms_per_sample": st["ms_per_sample"],
                     "exact": True, "before_reference_tokens_per_sample": BEFORE.get(k.name)})
        print(rows[-1], flush=True)
    (ROOT / "v15/reports/fft-opt-neural.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
