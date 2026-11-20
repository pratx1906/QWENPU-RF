"""End-to-end RF DSP-AI pipeline on QwenPU's CPU in weights.

IQ -> DC blocker -> 16-tap lowpass FIR -> {energy/noise/burst detector, FFT power + CFAR peak,
spectral MLP classifier, CFO estimate, running peak, FM demodulation}. Every stage is a verified
neural run (every token checked against the RV32 reference; outputs equal the independent integer
reference). Between stages the host only copies verified output words into the next program's RAM.
"""
import math

from .detect import EnergyDetector, PeakTracker
from .filters import DCBlocker
from .infer import SpectralMLP
from .kernels import FIR
from .nonlinear import CFOEstimate, FMDemod


def lowpass_taps(n=16, cutoff=0.25):
    """Windowed-sinc lowpass in Q15 with sum |h| <= 32767, so the output provably stays in int16."""
    mid = (n - 1) / 2
    h = [(math.sin(2 * math.pi * cutoff * (k - mid)) / (math.pi * (k - mid)) if k != mid else 2 * cutoff)
         * (0.54 - 0.46 * math.cos(2 * math.pi * k / (n - 1))) for k in range(n)]
    scale = 32767 / sum(abs(v) for v in h)
    codes = [int(math.floor(v * scale)) if v >= 0 else -int(math.floor(-v * scale)) for v in h]
    assert sum(abs(c) for c in codes) <= 32767
    return [(c, 0) for c in codes]


def stages(mlp_config: dict):
    from .spectral import CFARDetect
    return {
        "dc_block": DCBlocker(32440),
        "lowpass": FIR(lowpass_taps(), "lowpass16"),
        "energy": EnergyDetector(window=64, memory=8, alpha_q8=1024),
        "cfar": CFARDetect(16, [32767] * 16, 16, 2, 1, 1024),
        "classifier": SpectralMLP(mlp_config["w1"], mlp_config["b1"], mlp_config["w2"], mlp_config["b2"],
                                  mlp_config["fshift"]),
        "cfo": CFOEstimate(lag=1, window=64),
        "peak": PeakTracker(),
        "fm": FMDemod(),
    }


def run_pipeline(runner, streams, mlp_config, fm_samples=512, block=256, neural=True):
    """Run all stages over a list of IQ streams (int pairs); returns per-stage results per stream."""
    S = stages(mlp_config)
    out = {}
    dc = runner.run_many(S["dc_block"], streams, block, neural)
    x1 = [r.outputs for r in dc]
    lp = runner.run_many(S["lowpass"], x1, block, neural)
    x2 = [r.outputs for r in lp]
    out["dc_block"], out["lowpass"] = dc, lp
    for name in ("energy", "cfar", "classifier", "cfo", "peak"):
        out[name] = runner.run_many(S[name], x2, block, neural)
    out["fm"] = runner.run_many(S["fm"], [x[:fm_samples] for x in x2], block, neural)
    return S, x1, x2, out
