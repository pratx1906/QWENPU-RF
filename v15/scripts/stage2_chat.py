"""Stage 2: Qwen3.5 (QwenPU-published merged checkpoint) calls RF analysis on its own in-weights CPU.

  python v15/scripts/stage2_chat.py                      # the three prepared questions
  python v15/scripts/stage2_chat.py --ask "..." [--capture-list]

Model: $RFPU_DATA/checkpoints/QwenPU-published (all shards SHA256-verified against the publisher,
v15/reports/stage2-checkpoint.json). It is Qwen3.5-4B's own layers plus 13 compiled chip layers in
one checkpoint. Decoding follows chipformer.rv.chat.RvChat in executor mode: text tokens run through
the chip circuit (QwenPU's Triton executor) and then Qwen's own saved layers; chip tokens run through
the circuit only; `<tool_call>` text is captured into the CPU's RAM and the firmware's putchar output
is echoed back as Qwen tokens. Every circuit-owned token is checked live against chipformer's RV32
reference machine (bridge.Session lock); any difference stops generation and is reported.

Memory (24 GB GPU): Qwen's 32 layers and the first 248,077 embedding rows (Qwen's own vocabulary,
tied with lm_head) are on the GPU; the chip layers stay in host RAM because the executor computes the
same circuit. Consequently logits cover Qwen's vocabulary only, so RvChat's lm_head-vs-circuit head
check is not available here (ownership is decided by the circuit's output code, as in RvChat).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.stage2.firmware import (N, RF_LEVELS_TOOL, RF_MOD_TOOL, RF_TOOL, analyze, modulation_report,  # noqa: E402
                                      sample_rate, source)

from rfpu_v15.paths import DATA  # noqa: E402
CK = DATA / "checkpoints/QwenPU-published"
OUT = ROOT / "v15/reports/stage2"


SYNTHETIC_TRUTH = {14: "tone burst, samples 320-640, 0.19 cycles/sample", 15: "steady tone, -0.0625 cycles/sample"}


def capture_set():
    """14 real captures (1,024 samples each, v1.4 preprocessing with 4x headroom: gain 0.25/p95,
    Q15) plus two synthetic captures with known ground truth."""
    import real_iq_q15 as R
    rows = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"]
    caps = []
    for row in rows:
        z, _, meta = R.inputs(row)
        z = z[:N] / 3  # 0.75/p95 -> 0.25/p95: headroom so peaky captures do not clip
        q = np.clip(np.round(np.stack((z.real, z.imag), -1) * 32768), -32768, 32767).astype(int)
        md = row.get("metadata", {})
        fs, fc = md.get("sample_rate"), md.get("center_frequency_hz")
        label = (f"{row['dataset']} {row['emitter']}" + (f" fs={fs / 1e6:g}MS/s" if fs else " fs=unknown")
                 + (f" fc={fc / 1e6:g}MHz" if fc else ""))
        caps.append((label, [tuple(map(int, v)) for v in q]))
    rng = np.random.default_rng(15)
    t = np.arange(N)
    noise = (rng.normal(size=N) + 1j * rng.normal(size=N)) * .003
    burst = noise.copy(); burst[320:640] += .3 * np.exp(2j * np.pi * .19 * t[320:640])
    tone = .4 * np.exp(2j * np.pi * -.0625 * t) + noise
    # Ground truth stays on the host (SYNTHETIC_TRUTH); labels must not leak it to Qwen.
    for label, z in (("synthetic test signal A fs=1MS/s", burst), ("synthetic test signal B fs=1MS/s", tone)):
        q = np.clip(np.round(np.stack((z.real, z.imag), -1) * 32768), -32768, 32767).astype(int)
        caps.append((label, [tuple(map(int, v)) for v in q]))
    return caps


class SplitRFChat:
    """RvChat (chipformer.rv.chat) in executor mode, with a token-by-token executor prefill."""

    def __new__(cls, path=CK):
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        from chipformer.q35.fast import exact_matmuls
        from chipformer.q35.ir import Target
        from chipformer.rv import bridge as BR, core
        from chipformer.rv.chat import RvChat
        exact_matmuls()
        chat = RvChat.__new__(RvChat)
        chat.path, chat.dev = Path(path), "cuda"
        chat.meta = json.loads((chat.path / "chipformer.json").read_text())
        t = time.perf_counter()
        model = Qwen3_5ForCausalLM.from_pretrained(chat.path, dtype=torch.bfloat16).eval()  # host RAM
        lm, n = model.model, chat.meta["chip_layers"]
        chat.v0 = chat.meta["chip_token_offset"]
        E = lm.embed_tokens.weight[:chat.v0].detach().to("cuda")         # Qwen's vocabulary rows
        lm.embed_tokens = torch.nn.Embedding.from_pretrained(E, freeze=True)
        head = torch.nn.Linear(E.shape[1], E.shape[0], bias=False, device="meta")
        head.weight = torch.nn.Parameter(E, requires_grad=False)                  # tied: same tensor
        model.lm_head = head
        lm.norm.to("cuda"); lm.rotary_emb.to("cuda")
        for i in range(n, len(lm.layers)):
            lm.layers[i].to("cuda")
        torch.cuda.synchronize()
        chat.load_seconds = time.perf_counter() - t
        chat.model, chat.lm = model, lm
        chat.tok = AutoTokenizer.from_pretrained(chat.path)
        chat.T = BR.Text(chat.path / "tokenizer.json")
        cfg = model.config
        chat.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.intermediate_size),
                            chat.T, vx=chat.meta.get("vx", False))
        chat.n_chip = n
        chat.code_dims = torch.tensor(chat.meta["code_dims"], device="cuda")
        chat.eos = {248046, 248044}
        chat.fast, chat.executor, chat.verify = False, True, False
        chat._build_exec()
        chat.model = _Prefill(chat, model)
        return chat


class _Prefill:
    """Stands in for `self.model(input_ids=x, use_cache=True)` in RvChat.generate: the prompt runs
    through a fresh executor row from zero states (as the HF chip layers would), and each token's
    post-chip residual through Qwen's layers one position at a time (RvChat._qwen_step)."""

    def __init__(self, chat, model):
        self.chat, self.inner = chat, model
        self.config, self.lm_head = model.config, model.lm_head

    @torch.no_grad()
    def __call__(self, input_ids, use_cache=True):
        from chipformer.rv.chat import ExecRow
        from transformers import DynamicCache
        chat, ids = self.chat, input_ids[0].tolist()
        row = ExecRow(chat)
        row._capture()
        for st in row.heads:
            st["S"].zero_(); st["hist"].zero_()
        cache = DynamicCache(config=self.config)
        logits = None
        for pos, i in enumerate(ids):
            keep = row.step(i)
            logits = chat._qwen_step(chat._exec_hc(i, keep), cache, pos)
        del row
        return SimpleNamespace(logits=logits.view(1, 1, -1), past_key_values=cache)


QUESTIONS = [
    (14, "Capture 14 is a stored IQ recording. Use the RF analyzer: is there a burst in it, roughly which "
         "samples does it occupy, how far above the noise floor is it, and what is its frequency in cycles per sample?"),
    (12, "Analyze capture 12 (a 2.4 GHz drone controller recording). What is the signal level relative to the "
         "noise floor, and what is the carrier frequency offset in kHz? Show the conversion."),
    (8, "Run the RF analyzer on capture 8 and summarize what kind of signal activity it shows."),
]


# Both tools offered: level-only questions should use the cheaper rf_levels, spectral ones rf_analyze.
LEVELS = [
    (14, "What is the noise floor of capture 14 in dBFS, and how loud is its loudest 64-sample window?"),
    (12, "Is capture 12 strong or weak? Give me its mean and peak power in dBFS."),
    (14, "What is the frequency of the burst in capture 14, in cycles per sample?"),
]
# Amendment 7: real captures; (capture, question, report fields the answer must quote)
REAL = [
    (1, "Capture 1: what are its mean power and noise floor in dBFS?", ("mean_power_dbfs", "noise_floor_dbfs")),
    (2, "For capture 2, report the peak power in dBFS and the sample index where it occurs.", ("peak_power_dbfs", "at_sample")),
    (3, "What is the dominant spectral bin of capture 3 and its frequency in cycles per sample?", ("dominant_dft16_bin", "freq_cycles_per_sample")),
    (4, "How many of capture 4's 64-sample windows are active, and what is its loudest window power in dBFS?", ("active_windows", "loudest_window_dbfs")),
    (9, "Give the carrier frequency offset of capture 9 in cycles per sample.", ("cfo_cycles_per_sample",)),
    (13, "Capture 13: what are its mean and peak power in dBFS?", ("mean_power_dbfs", "peak_power_dbfs")),
]
# Amendment 11: 16 captures x 6 templates; (capture, question, asked report fields)
TEMPLATES = [
    ("Capture {c}: what are its mean power and noise floor in dBFS?", ("mean_power_dbfs", "noise_floor_dbfs")),
    ("What is the peak power of capture {c} in dBFS, and at which sample index does it occur?", ("peak_power_dbfs", "at_sample")),
    ("How many 64-sample windows of capture {c} are active, and what is its loudest window power in dBFS?", ("active_windows", "loudest_window_dbfs")),
    ("What is the dominant spectral bin of capture {c}, and its frequency in cycles per sample?", ("dominant_dft16_bin", "freq_cycles_per_sample")),
    ("Estimate the carrier frequency offset of capture {c} in cycles per sample.", ("cfo_cycles_per_sample",)),
    ("For capture {c}, which spectral bin dominates and what percentage of the power does it hold?", ("power_share_percent", "dominant_dft16_bin")),
]
BANK = [(c, t.format(c=c), f) for c in range(16) for t, f in TEMPLATES]
# Amendment 13: held-out questions in free wording (not the T1-T6 templates); (capture, question, asked fields, kind)
# kind: "para" = answerable from the report, "notool" = general knowledge, no tool needed,
# "unans" = the report does not contain the answer (the model should say so, not invent a value)
HELDOUT = [
    (0, "How loud is capture 0 overall? Just give me the average level in dBFS.", ("mean_power_dbfs",), "para"),
    (0, "Where in frequency does most of capture 0's energy sit? Answer in Hz.", ("freq_hz",), "para"),
    (1, "I need the strongest single sample of capture 1: its power in dBFS and its index.", ("peak_power_dbfs", "at_sample"), "para"),
    (1, "Is capture 1 offset from its tuned centre? Give the offset in Hz.", ("cfo_hz",), "para"),
    (2, "Give me the gap between the average power and the quietest stretch of capture 2, with both numbers in dBFS.", ("mean_power_dbfs", "noise_floor_dbfs"), "para"),
    (2, "Which DFT bin wins in capture 2, and how much of the power is in it?", ("dominant_dft16_bin", "power_share_percent"), "para"),
    (3, "Out of the 16 windows in capture 3, how many show activity?", ("active_windows",), "para"),
    (3, "Tell me the dominant tone of capture 3 in hertz and in cycles per sample.", ("freq_hz", "freq_cycles_per_sample"), "para"),
    (4, "What's the background noise level in capture 4, in dBFS?", ("noise_floor_dbfs",), "para"),
    (4, "Measure the frequency error of capture 4 in cycles per sample and in Hz.", ("cfo_cycles_per_sample", "cfo_hz"), "para"),
    (5, "What's the loudest 64-sample chunk of capture 5, in dBFS?", ("loudest_window_dbfs",), "para"),
    (5, "Capture 5 - what fraction of the spectrum's power lands in the biggest bin?", ("power_share_percent",), "para"),
    (6, "Check capture 6: how far above its noise floor does its peak sample reach? Give both values in dBFS.", ("peak_power_dbfs", "noise_floor_dbfs"), "para"),
    (6, "Work out the carrier offset of capture 6 as a fraction of the sample rate.", ("cfo_cycles_per_sample",), "para"),
    (7, "What is capture 7's average power level, in dBFS?", ("mean_power_dbfs",), "para"),
    (7, "In capture 7, which of the 16 frequency bins is strongest?", ("dominant_dft16_bin",), "para"),
    (8, "Give me the active window count and the noise floor for capture 8.", ("active_windows", "noise_floor_dbfs"), "para"),
    (8, "What normalised frequency carries the most energy in capture 8?", ("freq_cycles_per_sample",), "para"),
    (9, "How hot does capture 9 peak, and at which sample?", ("peak_power_dbfs", "at_sample"), "para"),
    (9, "Estimate the frequency offset of capture 9 (in cycles per sample).", ("cfo_cycles_per_sample",), "para"),
    (10, "Quick level check on capture 10: average and loudest-window power in dBFS, please.", ("mean_power_dbfs", "loudest_window_dbfs"), "para"),
    (10, "For capture 10, report the dominant bin and the share of power it holds.", ("dominant_dft16_bin", "power_share_percent"), "para"),
    (11, "What is the quietest-window power of capture 11, in dBFS?", ("noise_floor_dbfs",), "para"),
    (11, "Is capture 11 frequency-shifted? Give its offset in cycles per sample.", ("cfo_cycles_per_sample",), "para"),
    (12, "How strong is the drone controller in capture 12 at its peak, in dBFS?", ("peak_power_dbfs",), "para"),
    (12, "Capture 12 is sampled at 120 MS/s. What is its carrier offset in Hz?", ("cfo_hz",), "para"),
    (13, "Roughly how busy is capture 13 - how many active windows does it have?", ("active_windows",), "para"),
    (13, "Where does the dominant energy of capture 13 sit, in Hz?", ("freq_hz",), "para"),
    (14, "Signal A (capture 14): what are its loudest-window power and its noise floor in dBFS?", ("loudest_window_dbfs", "noise_floor_dbfs"), "para"),
    (14, "At what frequency, in Hz, is the strongest component of capture 14?", ("freq_hz",), "para"),
    (15, "Signal B (capture 15): what is its mean power in dBFS?", ("mean_power_dbfs",), "para"),
    (15, "What is the frequency of capture 15's dominant component in cycles per sample, and which bin is it?", ("freq_cycles_per_sample", "dominant_dft16_bin"), "para"),
    (None, "In general terms, what is a carrier frequency offset and why does it happen?", (), "notool"),
    (None, "Explain the difference between dBFS and dBm in two sentences.", (), "notool"),
    (None, "What is a noise floor in a radio receiver?", (), "notool"),
    (None, "Why do people use a 16-point DFT rather than a much longer one for a quick spectral check?", (), "notool"),
    (None, "What does Q15 fixed-point format mean?", (), "notool"),
    (None, "What is LoRa, briefly?", (), "notool"),
    (None, "What is the Nyquist frequency of a signal sampled at 10 MS/s?", (), "notool"),
    (None, "In one sentence, what is IQ data?", (), "notool"),
    (0, "What modulation scheme and coding rate is capture 0 using? Base your answer on the measured data.", (), "unans"),
    (2, "What is the occupied bandwidth of capture 2 in MHz, according to the analyzer?", (), "unans"),
    (4, "How many separate transmitters are present in capture 4, according to the measurements?", (), "unans"),
    (6, "What is the LoRa spreading factor of capture 6? Use the analyzer's output.", (), "unans"),
    (9, "What is the bit error rate of capture 9, according to the analyzer?", (), "unans"),
    (12, "How far away was the drone controller in capture 12, in metres, according to the measurements?", (), "unans"),
    (13, "From which direction (angle of arrival) did capture 13's signal come, according to the analyzer?", (), "unans"),
    (15, "What is the symbol rate of capture 15, according to the analyzer?", (), "unans"),
]
# Amendment 14: new held-out questions for firmware v4 + the answer rule (none seen before the run)
A14 = [
    (1, "Which modulation does capture 1 use, going by the analyzer's numbers?", (), "unans"),
    (3, "How wide is the signal in capture 3, in MHz of occupied bandwidth?", (), "unans"),
    (5, "What symbol rate is the DVB-T signal in capture 5 running at, per the measurements?", (), "unans"),
    (7, "How many different LoRa devices are transmitting in capture 7, according to the analyzer?", (), "unans"),
    (10, "What is the bit error rate of the LoRa link in capture 10, based on the measurements?", (), "unans"),
    (11, "How far from the receiver was the transmitter in capture 11, in metres?", (), "unans"),
    (14, "From which bearing, in degrees, did signal A in capture 14 arrive, per the analyzer?", (), "unans"),
    (0, "What packet type or MAC address appears in capture 0, according to the analyzer?", (), "unans"),
    (2, "What is the transmitter's output power in watts for capture 2, according to the measurements?", (), "unans"),
    (4, "What is the receiver's antenna gain for capture 4, based on the analyzer?", (), "unans"),
    (6, "What is the temperature of the LoRa device in capture 6, according to the analyzer?", (), "unans"),
    (8, "What is the IQ imbalance (gain and phase) of capture 8, according to the analyzer?", (), "unans"),
    (9, "What is the signal's polarization in capture 9, according to the analyzer?", (), "unans"),
    (12, "What is the battery level of the drone controller in capture 12, according to the measurements?", (), "unans"),
    (13, "What is the RSSI of capture 13 in dBm, according to the analyzer?", (), "unans"),
    (15, "What is the phase noise of signal B (capture 15) in dBc/Hz, according to the analyzer?", (), "unans"),
    (0, "By how many dB does capture 0's peak exceed its mean power?", ("peak_power_dbfs", "mean_power_dbfs"), "para"),
    (1, "What is capture 1's loudest-window level relative to its noise floor, in dB?", ("loudest_window_dbfs", "noise_floor_dbfs"), "para"),
    (2, "Give capture 2's frequency offset in kilohertz.", ("cfo_hz",), "para"),
    (3, "What share of the 64-sample windows in capture 3 are active, as a percentage?", ("active_windows",), "para"),
    (4, "What is the dominant frequency of capture 4 in kHz?", ("freq_hz",), "para"),
    (5, "Report the mean power and the peak sample index for capture 5.", ("mean_power_dbfs", "at_sample"), "para"),
    (6, "How strong is the noise floor of capture 6 compared with its loudest window? Give both in dBFS.", ("noise_floor_dbfs", "loudest_window_dbfs"), "para"),
    (7, "What is the carrier offset of capture 7, expressed in cycles per sample?", ("cfo_cycles_per_sample",), "para"),
    (8, "Which DFT bin is strongest in capture 8, and what share of power does it carry?", ("dominant_dft16_bin", "power_share_percent"), "para"),
    (9, "Give the dominant tone of capture 9 in cycles per sample.", ("freq_cycles_per_sample",), "para"),
    (10, "What's the peak power of capture 10 and where does it occur?", ("peak_power_dbfs", "at_sample"), "para"),
    (11, "How many windows are active in capture 11?", ("active_windows",), "para"),
    (12, "Express capture 12's dominant frequency in MHz.", ("freq_hz",), "para"),
    (13, "What is the carrier offset of capture 13 in MHz?", ("cfo_hz",), "para"),
    (14, "How far above its noise floor is signal A's loudest window, in dB?", ("loudest_window_dbfs", "noise_floor_dbfs"), "para"),
    (15, "What is signal B's frequency offset in kHz?", ("cfo_hz",), "para"),
    (None, "What is the difference between a sample rate and a symbol rate?", (), "notool"),
    (None, "Why can't dBFS be converted to dBm without calibration?", (), "notool"),
    (None, "What does a CFAR detector do, in one or two sentences?", (), "notool"),
    (None, "What is the Doppler effect in radio, briefly?", (), "notool"),
]
def modulation_set():
    """Amendment 15 demo: 10 classes drawn at random (seed 1516, not hand-picked), one capture each at 20 dB SNR,
    from a seed (1517) not used for training, dev or test. Labels do not reveal the class; truth stays on the host."""
    from rfpu_v15.modgen import CLASSES, capture
    rng = np.random.default_rng(1516)
    classes = [CLASSES[i] for i in sorted(rng.choice(len(CLASSES), 10, replace=False))]
    g = np.random.default_rng(1517)
    return [(f"synthetic capture {k} fs=1MS/s", capture(c, g, 20)) for k, c in enumerate(classes)], classes


MODQ = [(0, "What modulation does capture 0 use?"),
        (1, "Classify the modulation of capture 1."),
        (2, "Which modulation scheme is in capture 2, and how confident is the classifier?"),
        (3, "Capture 3: what kind of modulation is this?"),
        (4, "Identify the modulation type of capture 4."),
        (5, "What is the modulation of capture 5, and what was the second choice?"),
        (6, "Tell me the modulation class of capture 6."),
        (7, "Capture 7 - what modulation?"),
        (8, "Determine the modulation used in capture 8."),
        (9, "What modulation is capture 9? Answer in one sentence.")]


RULE = (" Answer only from the tool report. If the user asks for something the report does not contain (its "
        "not_measured line lists what the tool does not measure), say that this tool does not measure it; do not "
        "estimate it from other fields. A difference between two dBFS values is in dB.")
ROBUST = [
    (None, "Analyze capture 20 with the RF analyzer."),
    (None, "Call rf_analyze with capture -1 and tell me exactly what the tool returned."),
    (None, "Compare captures 0 and 13: run the RF analyzer on both and tell me which has the higher mean power and by how many dB."),
    (None, "In one sentence, what does dBFS mean? Do not use any tools."),
    (14, "Capture 14: is there a burst, and at what frequency in cycles per sample?"),
    (14, "Capture 14: is there a burst, and at what frequency in cycles per sample?"),
]


def report_check(text, cap, fs=0):
    """Compare the report Qwen received (inside <tool_response>) with the host model."""
    m = re.search(r"<tool_response>(.*?)</tool_response>", text, re.S)
    if not m:
        return None
    got, want = {}, analyze(cap, fs=fs)
    for k in want:
        g = re.search(k + r"=(-?[0-9.]+)", m.group(1))
        if g:
            got[k] = float(g.group(1))
    return {"report": m.group(1).strip(), "fields": {k: {"firmware": got.get(k), "host_model": want[k]} for k in want}}


def main():
    from rfpu_v15.textio import utf8_stdout
    utf8_stdout()  # Qwen may write non-cp1252 symbols
    from chipformer.rv import bridge as BR, ref
    from chipformer.rv.asm import assemble
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask")
    ap.add_argument("--capture", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--suite", choices=["demo", "robust", "levels", "real", "bank", "heldout", "a14", "modulation"], default="demo")
    ap.add_argument("--full-gpu", help="merged checkpoint that fits the GPU: stock RvChat, every layer on the GPU, "
                    "chip tokens through the saved HF chip layers (compiled fast path), head check on")
    ap.add_argument("--chunk-sweep", default=None, help="with --full-gpu: comma list of chunk sizes (0 = the "
                    "unchunked RvChat path) run on the same first question; tokens compared with 0 (Amendment 8 P8.5)")
    ap.add_argument("--chunk", type=int, default=0, help="with --full-gpu: verify the computer's tokens C at a time "
                    "(chunked verification against the reference lock; rfpu_v15.stage2.chunked_chat)")
    ap.add_argument("--resume", action="store_true", help="append to the existing suite output, skipping questions "
                    "already answered (used after a crash; each question is independent)")
    ap.add_argument("--firmware", type=int, choices=[3, 4], default=4, help="4 (Amendment 14): not_measured line "
                    "plus the answer rule in the system prompt; 3: as in Amendments 12-13")
    ap.add_argument("--tag", default="", help="suffix for the output file name (keeps earlier runs)")
    a = ap.parse_args()
    mod = a.suite == "modulation"
    mlp = mod_truth = None
    if mod:  # Amendment 15b: modulation captures + the rf_modulation tool
        from rfpu_v15.cpu.modclass import IntMLP
        d = json.loads((ROOT / "v15/reports/modclass-model.json").read_text(encoding="utf-8"))["mlp"]
        mlp = IntMLP(d["w1"], d["b1"], d["w2"], d["b2"], s1=d["s1"])
        caps, mod_truth = modulation_set()
    else:
        caps = capture_set()
    prog = assemble(source(caps, not_measured=a.firmware == 4, modulation=mlp))
    listing = "\n".join(f"{i}: {label}" for i, (label, _) in enumerate(caps))
    system = (("You can analyze stored radio IQ captures with the rf_analyze, rf_levels and rf_modulation tools, "
               "which run exact signal processing and modulation classification on the processor built into this model."
               if mod else "You can analyze stored radio IQ captures with the rf_analyze tool, which runs exact signal "
               "processing on the processor built into this model.") + (RULE if a.firmware == 4 else "")
              + " Available captures:\n" + listing)
    if a.full_gpu:
        from chipformer.rv.chat import RvChat
        t = time.perf_counter()
        if a.chunk or a.chunk_sweep:
            from rfpu_v15.stage2.chunked_chat import ChunkedRvChat
            chat = ChunkedRvChat(a.full_gpu, fast=True, executor=False, chunk=a.chunk or 1024)
        else:
            chat = RvChat(a.full_gpu, fast=True, executor=False)
        chat.load_seconds = time.perf_counter() - t
    else:
        chat = SplitRFChat()
    OUT.mkdir(parents=True, exist_ok=True)
    fields = {q: f for _, q, f in REAL + BANK}
    fields.update({q: f for _, q, f, kind in HELDOUT + A14 if kind == "para"})
    qs = [(a.capture, a.ask)] if a.ask else {"robust": ROBUST, "levels": LEVELS,
                                             "real": [(c, q) for c, q, _ in REAL],
                                             "bank": [(c, q) for c, q, _ in BANK],
                                             "heldout": [(c, q) for c, q, _, _ in HELDOUT],
                                             "a14": [(c, q) for c, q, _, _ in A14], "modulation": MODQ}.get(a.suite, QUESTIONS)
    name = f"stage2-{a.suite}.json" if a.suite != "demo" and not a.ask else "stage2-chat.json"
    tools = (RF_TOOL, RF_LEVELS_TOOL) if a.suite in ("levels", "real", "bank", "heldout", "a14") else (RF_TOOL,)
    if mod:
        tools = (RF_TOOL, RF_LEVELS_TOOL, RF_MOD_TOOL)
    if a.full_gpu:
        name = ("fullgpu-chunked-" if a.chunk else "fullgpu-") + name
    if a.tag:
        name = name.replace(".json", f"-{a.tag}.json")
    summary = {"model": a.full_gpu or str(CK), "peak_gpu_bytes": None, "load_seconds": chat.load_seconds,
               "checkpoint_verification": "v15/reports/stage2-checkpoint.json", "runs": []}
    if a.chunk_sweep:  # P8.5: one question, several chunk sizes, one model load
        cap, q = qs[0]
        sweep = []
        for C in [int(v) for v in a.chunk_sweep.split(",")]:
            m = ref.Machine.from_program(prog); BR.park(m)
            m2 = ref.Machine.from_program(prog); BR.park(m2)
            ids = chat.prompt_ids([{"role": "system", "content": system}, {"role": "user", "content": q}], tools=(RF_TOOL,))
            lock = BR.Session(chat.T, m2)
            lock.history(ids)
            chat.chunk, chat.n_chunked = max(C, 1), 0
            t = time.perf_counter()
            gen = RvChat.generate if C == 0 else type(chat).generate
            r = gen(chat, ids, m, a.max_new, max_chip=2_000_000, check_head=True, lock=lock)
            sweep.append({"chunk": C, "seconds": time.perf_counter() - t, "chip_seconds": r["chip_seconds"],
                          "text_seconds": r["text_seconds"], "chip_tokens": r["chip_tokens"], "lock_error": r["lock_error"],
                          "head_mismatch": r["head_mismatch"], "tokens": r["tokens"]})
            print({k_: v for k_, v in sweep[-1].items() if k_ != "tokens"}, flush=True)
        base = sweep[0]["tokens"]
        for row in sweep:
            row["identical_to_first"] = row["tokens"] == base
        out = {"protocol": "PROTOCOL.md Amendment 8 P8.5", "question": q, "capture": cap,
               "rows": [{k_: v for k_, v in row.items() if k_ != "tokens"} for row in sweep]}
        (OUT / "fullgpu-chunk-sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out["rows"], indent=1))
        return
    done = set()
    if a.resume and (OUT / name).exists():
        old = json.loads((OUT / name).read_text(encoding="utf-8"))
        summary["runs"], done = old["runs"], {r["question"] for r in old["runs"]}
        summary["resumed_after"] = len(done)
    for k, (cap, q) in enumerate(qs):
        if q in done:
            continue
        m = ref.Machine.from_program(prog); BR.park(m)
        m2 = ref.Machine.from_program(prog); BR.park(m2)
        ids = chat.prompt_ids([{"role": "system", "content": system}, {"role": "user", "content": q}], tools=tools)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
        t = time.perf_counter()
        r = chat.generate(ids, m, a.max_new, max_chip=2_000_000, check_head=bool(a.full_gpu), verbose=True, lock=lock)
        text = chat.text_of(r["tokens"])
        run = {"question": q, "capture": cap, "prompt_tokens": len(ids), "answer": text,
               "text_tokens": r["text_tokens"], "chip_tokens": r["chip_tokens"], "seconds": time.perf_counter() - t,
               "chip_seconds": r["chip_seconds"], "text_seconds": r["text_seconds"],
               "reference_lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"],
               "chip_path": ("saved HF chip layers on GPU, chunked verification C=%d" % a.chunk if a.chunk else
                             "saved HF chip layers on GPU") if a.full_gpu else "Triton executor",
               "chunks": r.get("chunks"),
               "all_circuit_tokens_equal_reference": r["lock_error"] is None,
               "tool_calls": text.count("<tool_call>"), "tool_responses": text.count("</tool_response>"),
               "tools_called": re.findall(r"<function=(\w+)>", text),
               "tokens": r["tokens"],
               "report_check": report_check(text, caps[cap][1], sample_rate(caps[cap][0])) if cap is not None and not mod else None}
        if mod and cap is not None:
            c_, r_, m_ = modulation_report(caps[cap][1], mlp)
            g = re.search(r"modulation=(\S+) runner_up=(\S+) score_margin=(-?\d+)", text)
            run["modulation_check"] = {"truth": mod_truth[cap], "host_model": [c_, r_, m_],
                                       "printed": [g.group(1), g.group(2), int(g.group(3))] if g else None,
                                       "printed_equals_host_model": bool(g) and [g.group(1), g.group(2), int(g.group(3))] == [c_, r_, m_],
                                       "answer_names_printed_class": bool(g) and g.group(1) in text.split("</tool_response>")[-1]}
        if q in fields:
            m = re.search(r"<tool_response>(.*?)</tool_response>", text, re.S)
            answer = text.split("</tool_response>")[-1]
            quoted = {}
            for f in fields[q]:
                v = re.search(f + r"=(-?[0-9.]+)", m.group(1)) if m else None
                quoted[f] = {"report": v.group(1) if v else None, "in_answer": bool(v) and v.group(1) in answer}
            run["asked_fields_quoted"] = quoted
        summary["runs"].append(run)
        if a.suite == "robust" and k == len(qs) - 1:
            summary["determinism_identical_tokens"] = summary["runs"][-1]["tokens"] == summary["runs"][-2]["tokens"]
        summary["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        (OUT / name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("=" * 80, f"\nQ{k}: {q}\n---\n{text}\n[{r['text_tokens']} text + {r['chip_tokens']} chip tokens, "
              f"{run['seconds']:.0f}s, lock error: {r['lock_error']}]", flush=True)


if __name__ == "__main__":
    main()
