"""Standard DSP figures of merit for v1.5 kernels, measured on neural (token-verified) runs.

  python v15/scripts/dsp_metrics.py <group> [<group> ...]
  groups: precision dynamic_range filter spectral roc_energy cfar matched cfo demod classifier systems baselines summary

Every kernel output used here comes from QwenPU's CPU in weights (fusedexec.FreeRun3), with every
token equal to the RV32 reference and outputs equal to the independent integer reference. Host-side
steps are analysis only (float references, threshold sweeps for ROC curves, fits) and are labelled.
Results: v15/reports/metrics/<group>.json; `summary` writes v15/reports/DSP_METRICS.md.
"""
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.detect import EnergyDetector, PeakTracker  # noqa: E402
from rfpu_v15.cpu.filters import DCBlocker, IIRBiquad, NCOMixer  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.nonlinear import GAIN, AMEnvelope, CFOEstimate, FMDemod, MagPhase  # noqa: E402
from rfpu_v15.cpu.pipeline import lowpass_taps  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.cpu.spectral import CFARDetect, FFTPower  # noqa: E402
from rfpu_v15.cpu.vx_kernels import ScalarEnergy, VXEnergy  # noqa: E402

OUT = ROOT / "v15/reports/metrics"
FORMATS = {"Q15": 15, "Q2.13": 13}
RUNNERS = {}


def runner(vx=False, batch=16):
    if (vx, batch) not in RUNNERS:
        RUNNERS[(vx, batch)] = Runner(vx=vx, batch=batch)
    return RUNNERS[(vx, batch)]


def neural(kernel, streams, block=256, vx=False, batch=16):
    """Neural run of many streams; returns (outputs per stream, stats). Fails on any inexact token."""
    res = runner(vx, batch).run_many(kernel, streams, block)
    assert all(r.exact_tokens and r.exact_outputs for r in res), kernel.name
    total = sum(len(s) for s in streams)
    return [r.outputs for r in res], {"kernel_config": kernel.describe(), "samples": total, "tokens": res[0].tokens,
                                      "neural_seconds": res[0].neural_seconds,
                                      "ms_per_sample": 1000 * res[0].neural_seconds / total}


def quantize(z, frac):
    s = np.stack((np.real(z), np.imag(z)), -1) * (1 << frac)
    r = np.sign(s) * np.floor(np.abs(s) + .5)
    clipped = int(np.count_nonzero((r < -32768) | (r > 32767)))
    q = np.clip(r, -32768, 32767).astype(int)
    return [tuple(map(int, v)) for v in q], clipped


def cgauss(rng, n, level_db):
    """Complex Gaussian with total RMS 10^(level/20) in full-scale units (component full scale = 1)."""
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * 10 ** (level_db / 20) / math.sqrt(2)


def sqnr(actual, reference):
    a, r = np.asarray(actual, complex).ravel(), np.asarray(reference, complex).ravel()
    e = float(np.sum(abs(a - r) ** 2)); p = float(np.sum(abs(r) ** 2))
    db = 10 * math.log10(p / e) if e > 0 else float("inf")
    return {"sqnr_db": db, "enob_bits": (db - 1.76) / 6.02 if e > 0 else None, "nmse": e / p if p else None}


def cplx(outputs, scale):
    return np.array([complex(*v) / scale for v in outputs])


def save(group, data):
    OUT.mkdir(parents=True, exist_ok=True)
    data["group"] = group
    (OUT / f"{group}.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    print(f"[{group}] saved", flush=True)


# ---------------------------------------------------------------- float references of the kernels
def ref_fir(z, taps):
    h = np.array([complex(a, b) for a, b in taps]) / 32768
    return np.convolve(z, h)[:len(z)]


def ref_fft_power(z, n, window_codes, hop):
    w = np.array(window_codes) / 32768
    ends = list(range(n - 1, len(z), hop))
    return ends, np.array([abs(np.fft.fft(z[e - n + 1:e + 1] * w) / n) ** 2 for e in ends])


def power_actual(outputs, ends, frac):
    return np.array([[p / (1 << 2 * frac) for p in outputs[e]] for e in ends])


# ---------------------------------------------------------------- groups
def precision():
    """SQNR / ENOB per kernel and number format at -12 dBFS complex Gaussian input."""
    rng = np.random.default_rng(1)
    z = cgauss(rng, 2048, -12)
    taps = lowpass_taps()
    rows = []
    for fmt, frac in FORMATS.items():
        x, clipped = quantize(z, frac)
        S = 1 << frac
        y, st = neural(K.FIR(taps, "lowpass16"), [x]); rows.append({"kernel": "lowpass16 FIR", "format": fmt, **st, **sqnr(cplx(y[0], S), ref_fir(z, taps))})
        k = FFTPower(64, None, 64); y, st = neural(k, [x]); ends, pf = ref_fft_power(z, 64, k.window, 64)
        rows.append({"kernel": "FFT64 power (Hann)", "format": fmt, **st, **sqnr(power_actual(y[0], ends, frac), pf)})
        y, st = neural(MagPhase(), [x])
        mag = np.array([v[0] for v in y[0]]) / (GAIN * 8192 * S)
        ang = np.array([v[1] for v in y[0]]) / 2**31 * math.pi
        rows.append({"kernel": "CORDIC magnitude", "format": fmt, **st, **sqnr(mag, abs(z))})
        err = np.angle(np.exp(1j * (ang - np.angle(z))))
        rows.append({"kernel": "CORDIC phase", "format": fmt, "rms_phase_error_rad": float(np.sqrt(np.mean(err ** 2))),
                     "max_phase_error_rad": float(np.max(abs(err)))})
        b = (4096, 8192, 4096, -16384, 4096)
        y, st = neural(IIRBiquad(*b), [x])
        from scipy.signal import lfilter
        rf = lfilter(np.array(b[:3]) / 16384, [1, b[3] / 16384, b[4] / 16384], z)
        rows.append({"kernel": "IIR biquad (Q14)", "format": fmt, **st, **sqnr(cplx(y[0], S), rf)})
        fw = 0x12345678
        y, st = neural(NCOMixer(fw), [x])
        ph = 2 * math.pi * fw / 2**32 * np.arange(len(z))
        rows.append({"kernel": "NCO mixer (1024-entry table)", "format": fmt, **st, **sqnr(cplx(y[0], S), z * np.exp(1j * ph))})
        rows[-1]["input_clipped_components"] = clipped
    # energy: exact scalar vs int8 vector unit (Q15 input)
    x, _ = quantize(z, 15)
    ends = list(range(63, len(z), 64))
    ef = np.array([np.sum(abs(z[e - 63:e + 1]) ** 2) for e in ends])
    y, st = neural(ScalarEnergy(64), [x], vx=True)
    rows.append({"kernel": "energy W64, scalar exact", "format": "Q15", **st, **sqnr(np.array([y[0][e] for e in ends]) / 2**30, ef)})
    y, st = neural(VXEnergy(64, 7), [x], vx=True)
    rows.append({"kernel": "energy W64, VX int8 (NPU)", "format": "Q15->int8", **st,
                 **sqnr(np.array([y[0][e] for e in ends]) * 2**16 / 2**30, ef)})
    save("precision", {"input": "complex Gaussian, -12 dBFS RMS, 2048 samples", "rows": rows})


def dynamic_range():
    """SQNR vs input level (-60..+6 dBFS) for both formats: shows resolution floor and clipping ceiling."""
    rng = np.random.default_rng(2)
    levels = list(range(-60, 7, 6))
    zs = [cgauss(rng, 512, L) for L in levels]
    taps = lowpass_taps()
    rows = []
    for fmt, frac in FORMATS.items():
        S = 1 << frac
        qs = [quantize(z, frac) for z in zs]
        xs = [q for q, _ in qs]
        fir_out, st1 = neural(K.FIR(taps, "lowpass16"), xs)
        k = FFTPower(64, None, 64); fft_out, st2 = neural(k, xs)
        mag_out, st3 = neural(AMEnvelope(), xs)
        for L, z, (_, clipped), yf, yp, ym in zip(levels, zs, qs, fir_out, fft_out, mag_out):
            ends, pf = ref_fft_power(z, 64, k.window, 64)
            rows.append({"format": fmt, "level_dbfs": L, "clipped_components": clipped,
                         "fir_sqnr_db": sqnr(cplx(yf, S), ref_fir(z, taps))["sqnr_db"],
                         "fft64_power_sqnr_db": sqnr(power_actual(yp, ends, frac), pf)["sqnr_db"],
                         "envelope_sqnr_db": sqnr(np.array(ym) / (GAIN * 8192 * S), abs(z))["sqnr_db"]})
    best = {}
    for fmt in FORMATS:
        ok = [r["level_dbfs"] for r in rows if r["format"] == fmt and r["fir_sqnr_db"] >= 40]
        best[fmt] = {"fir_levels_with_sqnr_ge_40db": [min(ok), max(ok)] if ok else None}
    save("dynamic_range", {"rows": rows, "usable_range": best, "samples_per_level": 512})


def filter_response():
    """Measured frequency response of the fixed-point lowpass FIR and IIR biquad (impulse and tones)."""
    taps = lowpass_taps()
    impulse = [(32767, 0)] + [(0, 0)] * 255
    (y,), st_f = neural(K.FIR(taps, "lowpass16"), [impulse])
    h = np.array([v[0] for v in y]) / 32767
    b = (4096, 8192, 4096, -16384, 4096)
    (yi,), st_i = neural(IIRBiquad(*b), [impulse])
    hi = np.array([v[0] for v in yi]) / 32767

    def response(h, pass_edge, stop_edge):
        H = np.fft.fft(h, 8192); f = np.fft.fftfreq(8192); m = f >= 0
        mag = 20 * np.log10(np.maximum(abs(H[m]), 1e-12)); f = f[m]
        dc = mag[0]
        pb = mag[f <= pass_edge] - dc; sb = mag[f >= stop_edge] - dc
        cut = f[np.argmax(mag - dc < -3)]
        ph = np.unwrap(np.angle(H[np.fft.fftfreq(8192) >= 0]))
        gd = -np.gradient(ph, 2 * np.pi * f)
        return {"dc_gain_db": float(dc), "passband_ripple_db": float(pb.max() - pb.min()),
                "stopband_attenuation_db": float(-sb.max()), "cutoff_3db_cycles_per_sample": float(cut),
                "group_delay_samples_passband": [float(gd[f <= pass_edge].min()), float(gd[f <= pass_edge].max())]}
    ideal = np.array([a for a, _ in taps]) / 32768
    rng = np.random.default_rng(3)
    tones, tone_rows = [0.02, 0.08, 0.15, 0.2, 0.3, 0.35, 0.42], []
    xs = [quantize(.5 * np.exp(1j * (2 * np.pi * f * np.arange(512) + rng.uniform(0, 6.28))), 15)[0] for f in tones]
    outs, st_t = neural(K.FIR(taps, "lowpass16"), xs)
    for f, x, yy in zip(tones, xs, outs):
        xin, yo = cplx(x, 32768)[64:], cplx(yy, 32768)[64:]
        g = np.vdot(xin, yo) / np.vdot(xin, xin)
        design = abs(np.sum(ideal * np.exp(-2j * np.pi * f * np.arange(len(ideal)))))
        tone_rows.append({"freq_cycles_per_sample": f, "measured_gain_db": float(20 * np.log10(abs(g))),
                          "design_gain_db": float(20 * np.log10(design))})
    save("filter", {"lowpass16_fir": {**response(h, .2, .3), "design_ideal": response(ideal, .2, .3), "neural": st_f,
                                      "impulse_equals_quantized_taps": bool(np.allclose(h[:16] * 32767, [round(a * 32767 / 32768) for a, _ in taps], atol=1) and np.all(h[16:] == 0))},
                    "tones": tone_rows, "tone_run": st_t,
                    "iir_biquad": {**response(hi, .05, .3), "coefficients_q14": list(b), "neural": st_i}})


def spectral():
    """SFDR, noise floor, tone-frequency accuracy and window ENBW for FFT64 / FFT256 power (Hann)."""
    rng = np.random.default_rng(4)
    rows = []
    for n in (64, 256):
        k = FFTPower(n, None, n)
        w = np.array(k.window) / 32768
        enbw = n * np.sum(w ** 2) / np.sum(w) ** 2
        cases = [("on-bin", 7.0), ("off-bin", 7.37), ("high", n / 3 + .21)]
        xs, fr = [], []
        for name, b in cases:
            f = b / n
            z = 10 ** (-1 / 20) * np.exp(1j * (2 * np.pi * f * np.arange(n * 8) + rng.uniform(0, 6.28)))
            xs.append(quantize(z, 15)[0]); fr.append((name, b))
        outs, st = neural(k, xs)
        for (name, b), y in zip(fr, outs):
            P = np.mean([np.array(v, float) for v in y if v is not None], axis=0) / 2**30
            pk = int(np.argmax(P)); lobe = [(pk + d) % n for d in range(-3, 4)]
            spur = np.delete(P, lobe)
            a, c, e = (math.log10(max(P[(pk + d) % n], 1e-30)) for d in (-1, 0, 1))
            delta = .5 * (a - e) / (a - 2 * c + e) if (a - 2 * c + e) else 0
            rows.append({"fft_n": n, "case": name, "true_bin": b, "peak_bin": pk, "interpolated_bin": pk + delta,
                         "bin_error": pk + delta - b, "sfdr_db": float(10 * math.log10(P[pk] / spur.max())),
                         "noise_floor_db_rel_peak": float(10 * math.log10(np.median(spur) / P[pk])),
                         "enbw_bins": float(enbw), **st})
    save("spectral", {"rows": rows, "input": "-1 dBFS complex tone, Q15, Hann window, frames averaged"})


def roc_energy():
    """Energy detector: Pd/Pfa ROC vs SNR. Statistic (window mean power) computed in weights;
    the threshold sweep that traces the ROC is host-side analysis. Theory: chi-square (2N dof)."""
    from scipy.stats import chi2, ncx2
    rng = np.random.default_rng(5)
    W, sigma_db = 64, -30
    snrs = list(range(-14, 7, 2))
    noise = lambda n: cgauss(rng, n, sigma_db)  # noqa: E731
    h0 = noise(W * 1000)
    h1 = [noise(W * 200) + cgauss(rng, W * 200, sigma_db + s) for s in snrs]
    outs, st = neural(EnergyDetector(W, 8, 1024), [quantize(z, 15)[0] for z in [h0] + h1])
    stat = [np.array([v[0] for v in o if v is not None], float) for o in outs]
    t0 = stat[0]
    rows = []
    for s, t1 in zip(snrs, stat[1:]):
        r = {"snr_db": s}
        for pfa in (0.1, 0.01):
            th = np.quantile(t0, 1 - pfa)
            r[f"pd_at_pfa_{pfa}"] = float(np.mean(t1 > th))
            # theory: 2*sum|x|^2/sigma^2 ~ chi2(2W) under H0, ncx2(2W, 2W*snr) under H1
            th_t = chi2.ppf(1 - pfa, 2 * W)
            r[f"pd_theory_at_pfa_{pfa}"] = float(ncx2.sf(th_t, 2 * W, 2 * W * 10 ** (s / 10)))
        thr = np.sort(np.concatenate([t0, t1]))
        pf = np.array([np.mean(t0 > x) for x in thr]); pd = np.array([np.mean(t1 > x) for x in thr])
        r["auc"] = float(-np.trapezoid(pd, pf))
        rows.append(r)
    save("roc_energy", {"window": W, "noise_dbfs": sigma_db, "h0_windows": len(t0), "h1_windows_per_snr": 200,
                        "rows": rows, "neural": st,
                        "scope": "mean power statistic computed in weights; ROC threshold sweep on host"})


def cfar():
    """CA-CFAR on FFT64 power: Pd at the tone bin and Pfa over other bins vs per-bin SNR (decisions in weights)."""
    rng = np.random.default_rng(6)
    n, frames = 64, 60
    k = CFARDetect(n, None, n, 8, 2, 1024)
    snrs = list(range(-6, 13, 3))
    xs, bins = [], []
    for s in [None] + snrs:
        z = cgauss(rng, n * frames, -30)
        b = rng.integers(0, n, frames)
        if s is not None:
            amp = 10 ** ((-30 + s) / 20)
            t = np.arange(n * frames)
            z = z + amp * np.exp(1j * 2 * np.pi * np.repeat(b, n) * t / n)
        xs.append(quantize(z, 15)[0]); bins.append(b)
    outs, st = neural(k, xs)
    rows = []
    for s, o, b in zip([None] + snrs, outs, bins):
        masks = [v[0] for v in o if v is not None]
        det = np.array([[(m >> j) & 1 for j in range(n)] for m in masks])
        if s is None:
            rows.append({"snr_db": "noise only", "pfa_per_bin": float(det.mean())})
            continue
        hit = det[np.arange(len(det)), b[:len(det)]]
        other = det.copy(); other[np.arange(len(det)), b[:len(det)]] = 0
        peaks = np.array([v[1] for v in o if v is not None])
        rows.append({"per_sample_snr_db": s, "pd": float(hit.mean()), "pfa_other_bins": float(other.sum() / (det.size - len(det))),
                     "peak_bin_correct": float(np.mean(peaks == b[:len(peaks)]))})
    save("cfar", {"pfa_theory_iid_exponential": (1 + 4 / 16) ** -16, "fft_n": n, "train": 8, "guard": 2, "scale_q8": 1024, "frames_per_case": frames, "rows": rows, "neural": st,
                  "note": "SNR is per input sample (tone power / noise power); FFT processing gain applies on top"})


def matched():
    """Matched filter (16-symbol QPSK preamble): correct-peak rate and processing gain vs SNR and CFO.
    Filtering and the peak search both run in weights (FIR + PeakTracker)."""
    rng = np.random.default_rng(7)
    L, rec, trials = 16, 160, 24
    pre = (rng.choice([-1, 1], L) + 1j * rng.choice([-1, 1], L)) / math.sqrt(2)
    amp = 1800
    taps = [(int(round(c.real * amp)), int(round(c.imag * amp))) for c in np.conj(pre[::-1])]
    fir = K.FIR(taps, "matched16")
    noise_db = -20  # fixed noise; the preamble level sets the SNR (no clipping at any SNR)
    cases = [("snr", s, 0.0) for s in range(-18, 4, 3)] + [("cfo", 0, c) for c in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)]
    streams, truth = [], []
    for kind, s, cfo in cases:
        for _ in range(trials):
            t0 = int(rng.integers(30, rec - L - 10))
            z = cgauss(rng, rec, noise_db)
            p = 10 ** ((noise_db + s) / 20) * pre * np.exp(1j * cfo * np.arange(L))
            z[t0:t0 + L] += p
            streams.append(quantize(z, 15)[0]); truth.append(t0 + L - 1)
    filt, st1 = neural(fir, streams)
    peaks, st2 = neural(PeakTracker(), filt)
    rows = []
    for i, (kind, s, cfo) in enumerate(cases):
        sl = slice(i * trials, (i + 1) * trials)
        found = np.array([p[-1] for p in peaks[sl]]); tr = np.array(truth[sl])
        ok = np.abs(found - tr) <= 1
        gains = []
        for y, t in zip(filt[sl], tr):
            pw = abs(cplx(y, 32768)) ** 2
            off = np.delete(pw, range(max(0, t - L), min(len(pw), t + L)))
            gains.append(10 * math.log10(pw[t] / off.mean()))
        rows.append({"sweep": kind, "snr_db": s, "cfo_rad_per_sample": cfo, "correct_peak_rate": float(ok.mean()),
                     "median_output_peak_to_mean_db": float(np.median(gains))})
    save("matched", {"preamble_symbols": L, "record": rec, "trials": trials, "theoretical_processing_gain_db": 10 * math.log10(L),
                     "rows": rows, "neural": [st1, st2]})


def cfo():
    """CFO estimator (lag-1 autocorrelation + CORDIC, window 64): RMSE vs SNR against the CRB."""
    rng = np.random.default_rng(8)
    W, n_win = 64, 100
    snrs = list(range(-6, 25, 4))
    # One stream per trial: the estimator's lag sample before a stream is zero, so no product
    # straddles two different tones (the first version concatenated trials and biased the estimate).
    streams, truths = [], []
    for s in snrs:
        w = rng.uniform(-.5, .5, n_win)
        truths.append(w)
        for wi in w:
            z = 10 ** (-12 / 20) * np.exp(1j * (wi * np.arange(W) + rng.uniform(0, 6.28))) + cgauss(rng, W, -12 - s)
            streams.append(quantize(z, 15)[0])
    outs, st = neural(CFOEstimate(1, W), streams, block=64)
    rows = []
    for i, (s, w) in enumerate(zip(snrs, truths)):
        est = np.array([o[-1][0] for o in outs[i * n_win:(i + 1) * n_win]]) / 2**31 * math.pi
        err = np.angle(np.exp(1j * (est - w[:len(est)])))
        snr = 10 ** (s / 10)
        crb = math.sqrt(6 / (snr * W * (W * W - 1)))
        rows.append({"snr_db": s, "rmse_rad_per_sample": float(np.sqrt(np.mean(err ** 2))), "crb_rad_per_sample": crb,
                     "rmse_over_crb": float(np.sqrt(np.mean(err ** 2)) / crb)})
    save("cfo", {"window": W, "windows_per_snr": n_win, "rows": rows, "neural": st,
                 "note": "lag-1 phase estimator is not CRB-efficient at low SNR; ratio shows the gap"})


def demod():
    """FM discriminator SINAD and AM envelope SINAD vs SNR."""
    rng = np.random.default_rng(9)
    N, fm, dev = 2048, 0.01, 0.3
    t = np.arange(N)
    msg = np.sin(2 * np.pi * fm * t)
    rows, streams_fm, streams_am, snrs = [], [], [], [10, 20, 30, 40]
    env = 1 + .5 * np.sin(2 * np.pi * .005 * t)
    for s in snrs:
        z = .5 * np.exp(1j * np.cumsum(dev * msg)) + cgauss(rng, N, -6 - s)
        streams_fm.append(quantize(z, 15)[0])
        a = .3 * env * np.exp(1j * .7 * t) + cgauss(rng, N, 20 * math.log10(.3) - s)
        streams_am.append(quantize(a, 15)[0])
    fmo, st1 = neural(FMDemod(), streams_fm)
    amo, st2 = neural(AMEnvelope(), streams_am)

    def sinad(y, ref, band=None):
        """Least-squares fit of the reference; band (cycles/sample) restricts the residual to the
        message bandwidth, as after a post-detection audio filter (standard SINAD practice)."""
        A = np.vstack([ref, np.ones_like(ref)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit = A @ c; r = y - fit
        if band is not None:
            R = np.fft.fft(r); R[np.abs(np.fft.fftfreq(len(r))) > band] = 0; r = np.real(np.fft.ifft(R))
        return float(10 * math.log10(np.var(fit) / np.mean(r ** 2)))
    for s, yf, ya in zip(snrs, fmo, amo):
        inst = np.array(yf[1:], float) / 2**31 * math.pi
        env_hat = np.array(ya, float) / (GAIN * 8192 * 32768)
        rows.append({"input_snr_db": s, "fm_output_sinad_db": sinad(inst, dev * msg[1:]),
                     "fm_inband_sinad_db": sinad(inst, dev * msg[1:], 2 * fm),
                     "am_output_sinad_db": sinad(env_hat, .3 * env),
                     "am_inband_sinad_db": sinad(env_hat, .3 * env, .01)})
    save("demod", {"fm": {"tone_cycles_per_sample": fm, "deviation_rad_per_sample": dev},
                   "am": {"modulation_index": .5}, "rows": rows, "neural": [st1, st2]})


def classifier():
    """Trained spectral MLP: confusion matrix at nominal conditions and accuracy vs extra noise."""
    import train_mlp
    mlp = json.loads((ROOT / "v15/configs/spectral-mlp.json").read_text())
    model = SpectralMLP(mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"], mlp["fshift"])
    rows, conf = [], None
    for mult in (1, 2, 4, 8, 16):
        rng = np.random.default_rng(100 + mult)
        frames, labels = train_mlp.frames(rng, 600 if mult == 1 else 300)
        extra = [[(int(np.clip(a + rng.normal() * 300 * (mult - 1), -32768, 32767)),
                   int(np.clip(b + rng.normal() * 300 * (mult - 1), -32768, 32767))) for a, b in f] for f in frames]
        outs, st = neural(model, [[s for f in extra for s in f]])
        pred = np.array([v[1] for v in outs[0] if v is not None])
        rows.append({"noise_multiplier": mult, "accuracy": float(np.mean(pred == labels)), "frames": len(labels), **st})
        if mult == 1:
            conf = [[int(np.sum((labels == i) & (pred == j))) for j in range(3)] for i in range(3)]
    save("classifier", {"classes": mlp["classes"], "confusion_matrix_rows_true_cols_pred": conf, "rows": rows,
                        "scope": "synthetic frames only; not field classification"})


class PowerSampler:
    def __init__(self):
        self.samples, self.stop = [], False

    def __enter__(self):
        def loop():
            while not self.stop:
                try:
                    out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], text=True)
                    self.samples.append(float(out.strip().splitlines()[0]))
                except Exception:
                    pass
                time.sleep(.2)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop = True; self.t.join()

    @property
    def mean_w(self):
        return float(np.mean(self.samples)) if self.samples else None


def systems():
    """Throughput, peak memory and GPU energy vs number of parallel machines (batch)."""
    import torch
    rng = np.random.default_rng(10)
    streams = [quantize(cgauss(rng, 256, -12), 15)[0] for _ in range(32)]
    rows = []
    for name, k in (("fir16", K.FIR([(int(v), 0) for v in rng.integers(-2000, 2000, 16)], "fir16")),
                    ("fft64_power", FFTPower(64, None, 64))):
        for batch in (1, 4, 8, 16, 32):
            torch.cuda.reset_peak_memory_stats()
            r = Runner(vx=False, batch=batch)
            r.run_many(k, streams[:batch], 256)  # warm-up and graph capture
            with PowerSampler() as ps:
                res = r.run_many(k, streams, 256)
            assert all(x.exact_tokens and x.exact_outputs for x in res)
            sec, tok, n = res[0].neural_seconds, res[0].tokens, 32 * 256
            rows.append({"kernel": name, "batch": batch, "tokens_per_s": tok / sec, "samples_per_s": n / sec,
                         "ms_per_sample": 1000 * sec / n, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                         "mean_gpu_power_w": ps.mean_w, "joules_per_sample": ps.mean_w * sec / n if ps.mean_w else None})
            del r; torch.cuda.empty_cache()
            print(rows[-1], flush=True)
    save("systems", {"rows": rows, "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()})


def baselines():
    """A8: numerically identical integer implementations on NumPy (CPU) and PyTorch (CUDA), plus float
    library references. Resident-input time per sample; bit-exactness against the kernel reference."""
    import torch
    rng = np.random.default_rng(11)
    N = 1 << 16
    x = np.array(quantize(cgauss(rng, N, -12), 15)[0], dtype=np.int64)
    taps = np.array(lowpass_taps(), dtype=np.int64)
    kf = K.FIR([tuple(map(int, t)) for t in taps], "lowpass16")

    def fir_int(xr, xi, mod):
        yr = mod.zeros_like(xr); yi = mod.zeros_like(xi)
        for k, (hr, hi) in enumerate(taps.tolist()):
            sr = xr[:len(xr) - k] if k else xr; si = xi[:len(xi) - k] if k else xi
            yr[k:] += sr * hr - si * hi; yi[k:] += sr * hi + si * hr
        return (yr + (1 << 14)) >> 15, (yi + (1 << 14)) >> 15

    def timed(fn, sync=None, reps=5):
        fn(); best = 1e9
        for _ in range(reps):
            t = time.perf_counter(); out = fn()
            if sync: sync()
            best = min(best, time.perf_counter() - t)
        return best, out
    rows = []
    sec, (yr, yi) = timed(lambda: fir_int(x[:, 0], x[:, 1], np))
    exact = list(zip(yr.tolist(), yi.tolist()))[:4096] == kf.reference([tuple(v) for v in x[:4096].tolist()])
    rows.append({"kernel": "lowpass16 FIR", "backend": "NumPy int64 (CPU), same contract", "ns_per_sample": 1e9 * sec / N, "bit_exact": exact})
    xt = torch.tensor(x, device="cuda")
    sec, (yr, yi) = timed(lambda: fir_int(xt[:, 0], xt[:, 1], torch), torch.cuda.synchronize)
    exact = list(zip(yr[:4096].tolist(), yi[:4096].tolist())) == kf.reference([tuple(v) for v in x[:4096].tolist()])
    rows.append({"kernel": "lowpass16 FIR", "backend": "PyTorch int64 (CUDA), same contract", "ns_per_sample": 1e9 * sec / N, "bit_exact": exact})
    z = x[:, 0] + 1j * x[:, 1]
    h = (taps[:, 0] + 1j * taps[:, 1]) / 32768
    sec, _ = timed(lambda: np.convolve(z, h)[:N])
    rows.append({"kernel": "lowpass16 FIR", "backend": "NumPy complex128 float (not same contract)", "ns_per_sample": 1e9 * sec / N, "bit_exact": None})
    zt = torch.tensor(z.reshape(-1, 64), device="cuda")
    sec, _ = timed(lambda: torch.abs(torch.fft.fft(zt)) ** 2, torch.cuda.synchronize)
    rows.append({"kernel": "FFT64 power", "backend": "cuFFT complex128 float (not same contract)", "ns_per_sample": 1e9 * sec / N, "bit_exact": None})
    sec, _ = timed(lambda: np.abs(np.fft.fft(z.reshape(-1, 64))) ** 2)
    rows.append({"kernel": "FFT64 power", "backend": "NumPy FFT float (not same contract)", "ns_per_sample": 1e9 * sec / N, "bit_exact": None})
    camp = {k["kernel"]: k["real_iq"]["ms_per_sample"] for k in json.loads((ROOT / "v15/reports/capability-campaign.json").read_text())["kernels"]}
    neural_ns = {"lowpass16 FIR": camp["lowpass16"] * 1e6, "FFT64 power": camp["fft64_power"] * 1e6}
    for r in rows:
        r["neural_ns_per_sample"] = neural_ns[r["kernel"]]
        r["neural_slowdown_x"] = r["neural_ns_per_sample"] / r["ns_per_sample"]
    save("baselines", {"samples": N, "rows": rows, "scope": "resident input, best of 5, excludes transfers; neural from capability-campaign (batch 16)"})


def summary():
    def load(g):
        p = OUT / f"{g}.json"
        return json.loads(p.read_text()) if p.exists() else None
    L = ["# v1.5 DSP figures of merit", "", "Generated by `v15/scripts/dsp_metrics.py summary` from `v15/reports/metrics/*.json`. "
         "All kernel outputs are neural and token-verified; host steps are analysis only.", ""]
    d = load("precision")
    if d:
        L += ["## Precision (complex Gaussian, -12 dBFS)", "", "| Kernel | Format | SQNR dB | ENOB | ms/sample |", "|---|---|---:|---:|---:|"]
        L += [f"| {r['kernel']} | {r.get('format','')} | {r.get('sqnr_db', float('nan')):.1f} | {(r.get('enob_bits') or float('nan')):.1f} | {r.get('ms_per_sample', float('nan')):.2f} |" for r in d["rows"] if "sqnr_db" in r]
        L += [f"| {r['kernel']} | {r['format']} | rms phase error {r['rms_phase_error_rad']:.2e} rad | | |" for r in d["rows"] if "rms_phase_error_rad" in r]
        L.append("")
    d = load("dynamic_range")
    if d:
        L += ["## Dynamic range (SQNR dB vs input level)", "", "| Format | Level dBFS | Clipped | FIR | FFT64 power | Envelope |", "|---|---:|---:|---:|---:|---:|"]
        L += [f"| {r['format']} | {r['level_dbfs']} | {r['clipped_components']} | {r['fir_sqnr_db']:.1f} | {r['fft64_power_sqnr_db']:.1f} | {r['envelope_sqnr_db']:.1f} |" for r in d["rows"]]
        L += ["", f"FIR levels with SQNR >= 40 dB: {json.dumps(d['usable_range'])}", ""]
    d = load("filter")
    if d:
        f, i = d["lowpass16_fir"], d["iir_biquad"]
        L += ["## Filter response (measured from neural impulse response)", "", "| Filter | DC dB | Passband ripple dB | Stopband atten dB | -3 dB cutoff | Group delay |", "|---|---:|---:|---:|---:|---|"]
        for name, r in (("lowpass16 FIR (fixed point)", f), ("lowpass16 FIR (ideal design)", f["design_ideal"]), ("IIR biquad", i)):
            L.append(f"| {name} | {r['dc_gain_db']:.3f} | {r['passband_ripple_db']:.3f} | {r['stopband_attenuation_db']:.1f} | {r['cutoff_3db_cycles_per_sample']:.4f} | {r['group_delay_samples_passband'][0]:.2f}-{r['group_delay_samples_passband'][1]:.2f} |")
        L += ["", "| Tone (cyc/sample) | Measured gain dB | Design gain dB |", "|---:|---:|---:|"]
        L += [f"| {r['freq_cycles_per_sample']} | {r['measured_gain_db']:.2f} | {r['design_gain_db']:.2f} |" for r in d["tones"]]
        L.append("")
    d = load("spectral")
    if d:
        L += ["## Spectral (Hann, -1 dBFS tone)", "", "| FFT | Case | True bin | Interp. bin | SFDR dB | Noise floor dB | ENBW bins |", "|---:|---|---:|---:|---:|---:|---:|"]
        L += [f"| {r['fft_n']} | {r['case']} | {r['true_bin']:.2f} | {r['interpolated_bin']:.2f} | {r['sfdr_db']:.1f} | {r['noise_floor_db_rel_peak']:.1f} | {r['enbw_bins']:.3f} |" for r in d["rows"]]
        L.append("")
    d = load("roc_energy")
    if d:
        L += ["## Energy detector ROC (W=64)", "", "| SNR dB | Pd @ Pfa 0.1 (theory) | Pd @ Pfa 0.01 (theory) | AUC |", "|---:|---|---|---:|"]
        L += [f"| {r['snr_db']} | {r['pd_at_pfa_0.1']:.2f} ({r['pd_theory_at_pfa_0.1']:.2f}) | {r['pd_at_pfa_0.01']:.2f} ({r['pd_theory_at_pfa_0.01']:.2f}) | {r['auc']:.3f} |" for r in d["rows"]]
        L.append("")
    d = load("cfar")
    if d:
        L += ["## CA-CFAR on FFT64 (train 8, guard 2, scale 4x)", "", "| Per-sample SNR dB | Pd | Pfa (other bins) | Peak bin correct |", "|---|---:|---:|---:|"]
        for r in d["rows"]:
            L.append(f"| noise only | | {r['pfa_per_bin']:.4f} | |" if "pfa_per_bin" in r else
                     f"| {r['per_sample_snr_db']} | {r['pd']:.2f} | {r['pfa_other_bins']:.4f} | {r['peak_bin_correct']:.2f} |")
        L.append("")
    d = load("matched")
    if d:
        L += [f"## Matched filter (16-symbol QPSK, theory gain {d['theoretical_processing_gain_db']:.1f} dB)", "", "| Sweep | SNR dB | CFO rad/sample | Correct peak rate | Peak/mean dB |", "|---|---:|---:|---:|---:|"]
        L += [f"| {r['sweep']} | {r['snr_db']} | {r['cfo_rad_per_sample']} | {r['correct_peak_rate']:.2f} | {r['median_output_peak_to_mean_db']:.1f} |" for r in d["rows"]]
        L.append("")
    d = load("cfo")
    if d:
        L += ["## CFO estimation (lag 1, W=64)", "", "| SNR dB | RMSE rad/sample | CRB | RMSE/CRB |", "|---:|---:|---:|---:|"]
        L += [f"| {r['snr_db']} | {r['rmse_rad_per_sample']:.2e} | {r['crb_rad_per_sample']:.2e} | {r['rmse_over_crb']:.2f} |" for r in d["rows"]]
        L.append("")
    d = load("demod")
    if d:
        L += ["## Demodulation (wideband / in message band)", "", "| Input SNR dB | FM SINAD wideband | FM SINAD in-band | AM SINAD wideband | AM SINAD in-band |", "|---:|---:|---:|---:|---:|"]
        L += [f"| {r['input_snr_db']} | {r['fm_output_sinad_db']:.1f} | {r['fm_inband_sinad_db']:.1f} | {r['am_output_sinad_db']:.1f} | {r['am_inband_sinad_db']:.1f} |" for r in d["rows"]]
        L.append("")
    d = load("classifier")
    if d:
        L += ["## Classifier (synthetic)", "", f"Confusion (rows true, cols predicted; {', '.join(d['classes'])}): {d['confusion_matrix_rows_true_cols_pred']}", "",
              "| Noise multiplier | Accuracy |", "|---:|---:|"]
        L += [f"| {r['noise_multiplier']} | {r['accuracy']:.3f} |" for r in d["rows"]]
        L.append("")
    d = load("systems")
    if d:
        L += [f"## Systems ({d['gpu']})", "", "| Kernel | Batch | Tokens/s | Samples/s | ms/sample | Peak GPU MB | GPU W | mJ/sample |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {r['kernel']} | {r['batch']} | {r['tokens_per_s']:,.0f} | {r['samples_per_s']:.1f} | {r['ms_per_sample']:.2f} | {r['peak_gpu_bytes']/2**20:.0f} | {r['mean_gpu_power_w'] or float('nan'):.0f} | {1000*(r['joules_per_sample'] or float('nan')):.1f} |" for r in d["rows"]]
        L.append("")
    d = load("baselines")
    if d:
        L += ["## Conventional baselines (A8)", "", "| Kernel | Backend | ns/sample | Bit-exact | Neural slowdown |", "|---|---|---:|---|---:|"]
        L += [f"| {r['kernel']} | {r['backend']} | {r['ns_per_sample']:.2f} | {r['bit_exact']} | {r['neural_slowdown_x']:,.0f}x |" for r in d["rows"]]
        L += ["", d["scope"] + ". The campaign ran 8 chunks per batch of 16 (half-filled); with a full batch of 32 the "
              "neural FIR is ~9.6 ms/sample (systems table), so these slowdowns are ~2x overstated: about 2-6e5x for "
              "same-contract integer FIR.", ""]
    (ROOT / "v15/reports/DSP_METRICS.md").write_text("\n".join(L), encoding="utf-8")
    print("summary written", flush=True)


GROUPS = {"precision": precision, "dynamic_range": dynamic_range, "filter": filter_response, "spectral": spectral,
          "roc_energy": roc_energy, "cfar": cfar, "matched": matched, "cfo": cfo, "demod": demod,
          "classifier": classifier, "systems": systems, "baselines": baselines, "summary": summary}

if __name__ == "__main__":
    for g in sys.argv[1:]:
        GROUPS[g]()
