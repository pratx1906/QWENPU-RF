"""RF-analyst firmware for QwenPU's RV32 CPU, driven through the <tool_call> bridge.

Protocol (as chipformer/rv/env/calc/calc.c): park on an exit ecall; on GO read the call's text
(token i at 0x8000 + 16 i), take the first parameter's value (after the second '>' up to '<') as a
capture index, analyse that capture from DROM and print

    \\n<tool_response>\\nREPORT</tool_response>\\n<think>\\n\\n</think>\\n

then park again. Every value in REPORT is computed by RV32 instructions on the in-weights CPU:

  mean/peak power (exact u32/u64 sums), dBFS via a 16-entry log2 table (log2 to 1/256 bit, printed to the nearest 0.1 dB, within 0.07 dB), quietest and
  loudest 64-sample windows, active windows (> 4x the quietest), averaged DFT16 power spectrum
  (8 frames, x>>5 pre-scaling, Q15 twiddles) with dominant bin and its power share, and the
  lag-1 autocorrelation CFO (exact 64-bit sum, >>13, 16-step CORDIC) in cycles/sample.
  Firmware v3 (PROTOCOL Amendment 12): when the label carries "fs=<x>MS/s" the host writes that sample
  rate (Hz) into a per-capture table as data, and the CPU prints freq_hz = k_signed fs / 16 (exact) and
  cfo_hz = mulh(angle_q32, fs) = floor(angle fs / 2^32), plus one-line method notes.

`analyze()` is an independent float/integer host model of the same quantities used to check the
printed report; token-level correctness comes from the RV32 reference lockstep in the chat driver.
"""
import math
import re

from ..cpu.nonlinear import cordic_asm

N, W, FN = 1024, 64, 16
FRAMES = 8  # DFT16 frames averaged (hop N / FRAMES); was 32, which dominated the call's token cost
TOOL_RESPONSE, TOOL_RESPONSE_END, THINK, THINK_END = 256, 257, 258, 259
LOG2_T = [round(math.log2(1 + k / 16) * 65536) for k in range(17)]

RF_TOOL = {"type": "function", "function": {
    "name": "rf_analyze",
    "description": "Analyse a stored complex IQ radio capture on the exact RF processor built into "
                   "this model. Returns mean and peak power (dBFS), the noise floor and loudest "
                   "64-sample windows, how many windows are active, the dominant spectral bin of a "
                   "16-point spectrum with its frequency and power share, and the carrier frequency "
                   "offset in cycles per sample. Multiply cycles/sample by the capture's sample rate "
                   "for Hz.",
    "parameters": {"type": "object", "properties": {
        "capture": {"type": "integer", "description": "capture index (see the capture list)"}},
        "required": ["capture"]}}}
LEVELS = 108  # 'l': the 4th byte after the call's first '=' ("<function=rf_levels>")
RF_LEVELS_TOOL = {"type": "function", "function": {
    "name": "rf_levels",
    "description": "Fast level check of a stored complex IQ radio capture on the exact RF processor built "
                   "into this model: mean and peak power (dBFS), the noise floor and loudest 64-sample "
                   "windows, and which windows are active. No spectrum or frequency offset (use "
                   "rf_analyze for those); about a third of rf_analyze's cost.",
    "parameters": {"type": "object", "properties": {
        "capture": {"type": "integer", "description": "capture index (see the capture list)"}},
        "required": ["capture"]}}}


MODULATION = 109  # 'm': "<function=rf_modulation>"
RF_MOD_TOOL = {"type": "function", "function": {
    "name": "rf_modulation",
    "description": "Blind modulation classification of a stored complex IQ capture on the exact processor built "
                   "into this model: 30 signal features and a small neural network, both computed in the model's "
                   "weights, choose one of 25 classes (OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, "
                   "32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, "
                   "AM-DSB-SC, FM, GMSK, OQPSK, LoRa). Returns the class, the runner-up and the score margin.",
    "parameters": {"type": "object", "properties": {
        "capture": {"type": "integer", "description": "capture index (see the capture list)"}},
        "required": ["capture"]}}}
MOD_BASE = 0x400  # classifier RAM above the firmware's .data
MOD_NOT_MEASURED = ("coding rate, symbol rate, occupied bandwidth, number of transmitters, distance, direction of "
                    "arrival, bit error rate, protocol or packet contents, power levels and spectrum (use rf_analyze)")


def _modpath(mlp, rodata):
    """rf_modulation (PROTOCOL Amendment 15): the capture (DROM halves at s2) and the classifier's constants are
    copied into RAM, then cpu.modclass.ModFeatures and IntMLP run inline, unchanged, and the result is printed."""
    from ..cpu import codegen
    from ..cpu.modclass import COS64, NF, SIN64, LOG2_T as MLOG2, ModFeatures, subroutine
    from ..modgen import CLASSES
    if mlp.C != len(CLASSES) or mlp.NF != NF:
        raise ValueError("IntMLP must map 30 features to the 25 classes")
    codegen.PROOF = False
    try:
        fb = ModFeatures().build(N, base=MOD_BASE)
        mb = mlp.build(mlp.width, base=fb.layout.top)
    finally:
        codegen.PROOF = True
    FL, ML = fb.layout, mb.layout
    a = ["modpath:", "mv t0, s2"]
    a += [f"li t1, {FL['in']}", f"li t2, {2 * N}",
          "mcp0:", "lh t3, 0(t0)", "sw t3, 0(t1)", "addi t0, t0, 2", "addi t1, t1, 4", "addi t2, t2, -1", "bnez t2, mcp0"]
    params = sum(mlp.w1, []) + mlp.b1 + sum(mlp.w2, []) + mlp.b2
    for k, (lab, addr, vals) in enumerate((("mod_log2", FL["table"], MLOG2), ("mod_trig", FL["trig"], COS64 + SIN64),
                                           ("mod_params", ML["params"], params))):
        rodata += [f"{lab}:", ".word " + ", ".join(map(str, vals))]
        a += [f"la t0, {lab}", f"li t1, {addr}", f"li t2, {len(vals)}",
              f"mcp{k + 1}:", "lw t3, 0(t0)", "sw t3, 0(t1)", "addi t0, t0, 4", "addi t1, t1, 4", "addi t2, t2, -1", f"bnez t2, mcp{k + 1}"]
    a += ["li t0, 0", f"li t1, {FL['E']}", f"li t2, {FL['out'] - FL['E']}",  # zero the accumulators (RAM persists between calls)
          "mz0:", "sw zero, 0(t1)", "addi t1, t1, 4", "addi t2, t2, -4", "bnez t2, mz0"]
    a += subroutine(fb, "mf_")
    a += [f"li t0, {FL['out']}", f"li t1, {ML['in']}", f"li t2, {mlp.width}", "li t4, 0",
          "mcp4:", f"slti t3, t4, {NF}", "beqz t3, mcp5", "lw t3, 0(t0)", "j mcp6", "mcp5:", "li t3, 0", "mcp6:",
          "sw t3, 0(t1)", "sw zero, 4(t1)", "addi t0, t0, 4", "addi t1, t1, 8", "addi t4, t4, 1", "bne t4, t2, mcp4"]
    a += subroutine(mb, "mm_")
    out, C = ML["out"], mlp.C
    a += [f"li s6, {out}", f"lw s0, {4 * C}(s6)", "li s1, 0", "li s3, -2147483648", "li t1, 0",
          "mru:", "beq t1, s0, mrn", "slli t2, t1, 2", "add t2, t2, s6", "lw t3, 0(t2)", "bge s3, t3, mrn",
          "mv s3, t3", "mv s1, t1",
          "mrn:", "addi t1, t1, 1", f"li t2, {C}", "bne t1, t2, mru",
          "slli t2, s0, 2", "add t2, t2, s6", "lw t3, 0(t2)", "sub s4, t3, s3"]
    _puts(a, "modulation=")
    a += ["la t0, modnames", "slli t1, s0, 2", "add t0, t0, t1", "lw a0, 0(t0)", "call print_str"]
    _puts(a, " runner_up=")
    a += ["la t0, modnames", "slli t1, s1, 2", "add t0, t0, t1", "lw a0, 0(t0)", "call print_str"]
    _puts(a, " score_margin=")
    a += ["mv a0, s4", "call print_int"]
    _puts(a, "\nmethod=30 blind features (amplitude, carrier, x^2/x^4 symmetry, instantaneous frequency, M-th power "
             "phase, ring histogram) and a 30-32-25 integer MLP, all in weights; trained on synthetic captures "
             "with 8 samples/symbol and no frequency offset\nnot_measured=" + MOD_NOT_MEASURED + NOTE)
    a += ["li a0, 10", "li a7, 1", "ecall", "j done"]
    rodata += ["modnames:", ".word " + ", ".join(f"modn{i}" for i in range(C))]
    for i, name in enumerate(CLASSES):
        rodata += [f"modn{i}:", ".byte " + ", ".join(str(b) for b in name.encode()) + ", 0"]
    rodata += [".align 2"]
    return a


def modulation_report(x, mlp):
    """Independent host model of rf_modulation's result: (class, runner_up, margin)."""
    from ..cpu.modclass import features
    from ..modgen import CLASSES
    s, c = mlp.forward(features(x))
    r = max((k for k in range(len(s)) if k != c), key=lambda k: (s[k], -k))
    return CLASSES[c], CLASSES[r], s[c] - s[r]


def _put(a, ch):
    a.append(f"li a0, {ch}")
    a.append("li a7, 1")
    a.append("ecall")


def _puts(a, text):
    for ch in text.encode():
        _put(a, ch)


# Firmware v4 (PROTOCOL Amendment 14): each report ends with what the tool does NOT measure, so the model can
# say "not measured" instead of deriving an answer from unrelated fields.
NOT_MEASURED = ("modulation, coding rate, symbol rate, occupied bandwidth, number of transmitters, distance, "
                "direction of arrival, bit error rate, protocol or packet contents")
NOT_MEASURED_LEVELS = "spectrum and frequency offset (use rf_analyze), " + NOT_MEASURED
NOTE = " (this tool reports only the values above)"


def sample_rate(label):
    """Sample rate in Hz from a capture label ("fs=10MS/s"), or 0 when unknown."""
    m = re.search(r"fs=([0-9.]+)MS/s", label)
    return int(round(float(m.group(1)) * 1e6)) if m else 0


def source(captures, frames=FRAMES, not_measured=True, modulation=None):
    """captures: list of (label, [(i, q)] * N) Q15 pairs. Returns RV32 assembly.
    not_measured=False gives firmware v3 (Amendments 12-13), without the not_measured line.
    modulation: a cpu.modclass.IntMLP; adds the rf_modulation tool (PROTOCOL Amendment 15)."""
    if not 1 <= len(captures) <= 24 or any(len(x) != N for _, x in captures):
        raise ValueError(f"1..24 captures of {N} samples")
    if any(not 0 <= sample_rate(lab) < 1 << 31 for lab, _ in captures):
        raise ValueError("sample rate must fit int32")
    a = ["_start:", "park:", "li a0, 0", "li a7, 93", "ecall", "call analyze", "j park"]
    # ---------------- parse the capture index from the tool-call text
    a += ["analyze:", "addi sp, sp, -16", "sw ra, 0(sp)",
          # mode: the 4th non-zero byte after the first '=' (rf_analyze -> 'a', rf_levels -> 'l')
          "li t0, 32784", "li t1, 34064", "li t2, 0", "li s3, 0",
          "fscan:", "bgeu t0, t1, fdone", "lbu t3, 0(t0)", "addi t0, t0, 1", "beqz t3, fskip",
          "bnez t2, fcnt", "li t4, 61", "bne t3, t4, fscan", "li t2, 1", "j fscan",
          "fcnt:", "addi t2, t2, 1", "li t4, 5", "bne t2, t4, fscan", "mv s3, t3", "j fdone",
          "fskip:", "andi t3, t0, 15", "beqz t3, fscan", "addi t0, t0, 15", "andi t0, t0, -16", "j fscan",
          "fdone:", "la t4, mode", "sw s3, 0(t4)",
          "li t0, 32784", "li t1, 34064", "li t2, 0", "li s0, 0", "li s1, 0", "li a5, 0",  # gt count, value, digits, invalid
          "pscan:", "bgeu t0, t1, pdone", "lbu t3, 0(t0)", "addi t0, t0, 1",
          "beqz t3, pskip",
          "li t4, 2", "bge t2, t4, pval", "li t4, 62", "bne t3, t4, pscan", "addi t2, t2, 1", "j pscan",
          "pval:", "li t4, 60", "beq t3, t4, pdone",
          # whitespace is skipped; any other non-digit invalidates the value (e.g. "-1", "abc")
          "li t4, 32", "beq t3, t4, pscan", "li t4, 10", "beq t3, t4, pscan", "li t4, 13", "beq t3, t4, pscan", "li t4, 9", "beq t3, t4, pscan",
          "addi t3, t3, -48", "li t4, 10", "bltu t3, t4, pdig", "li a5, 1", "j pscan",
          "pdig:", "mul s0, s0, t4", "add s0, s0, t3", "addi s1, s1, 1", "j pscan",
          "pskip:", "andi t3, t0, 15", "beqz t3, pscan", "addi t0, t0, 15", "andi t0, t0, -16", "j pscan",
          "pdone:"]
    a += ["\n".join([]), "li a0, 10", "li a7, 1", "ecall", f"li a0, {TOOL_RESPONSE}", "li a7, 1", "ecall",
          "li a0, 10", "li a7, 1", "ecall"]
    a += [f"li t4, {len(captures)}", "bnez a5, bad1", "beqz s1, bad1", "li t3, 5", "bltu t3, s1, bad1", "bltu s0, t4, oki", "bad1:", "j bad", "oki:",
          "la t0, captab", "slli t1, s0, 2", "add t0, t0, t1", "lw s2, 0(t0)",        # s2 = capture base
          "la t0, fstab", "add t0, t0, t1", "lw t2, 0(t0)", "la t0, capfs", "sw t2, 0(t0)",  # this capture's fs (Hz)
          "la t0, labtab", "add t0, t0, t1", "lw a0, 0(t0)", "call print_str", "li a0, 10", "li a7, 1", "ecall"]
    _puts(a, f"samples={N} (Q15)\n")
    if modulation is not None:
        a += ["la t4, mode", "lw t4, 0(t4)", f"li t5, {MODULATION}", "bne t4, t5, nomod", "j modpath", "nomod:"]
    # ---------------- pass 1: mean power, peak, 16 window powers, and the lag-1 CFO sum
    # R = sum x[n] conj(x[n-1]) (exact 64-bit; x[-1] = 0) in the same pass, sharing loads and the loop
    a += ["la t4, mode", "lw t4, 0(t4)", f"li t5, {LEVELS}", "beq t4, t5, p1inL"]
    a += _pass1("", cfo=True) + ["j p1end"] + _pass1("L", cfo=False) + ["p1end:"]
    _puts(a, "mean_power_dbfs=")
    a += ["mv a0, s3", "call dbfs10", "li a1, 1", "call print_fixed"]
    _puts(a, "\npeak_power_dbfs=")
    a += ["mv a0, s5", "call dbfs10", "li a1, 1", "call print_fixed"]
    _puts(a, " at_sample=")
    a += ["mv a0, s6", "call print_int"]
    # windows: min, max, count above 4x min, first, last
    a += ["la t6, winpow", "li s3, -1", "li s4, 0", "li t0, 0",
          "w1:", "lw t1, 0(t6)", "bgeu t1, s3, w1a", "mv s3, t1", "w1a:", "bgeu s4, t1, w1b", "mv s4, t1", "w1b:",
          "addi t6, t6, 4", "addi t0, t0, 1", f"li t2, {N // W}", "bne t0, t2, w1",
          "la t6, winpow", "li s5, 0", "li s6, -1", "li s7, -1", "li t0, 0",
          "w2:", "lw t1, 0(t6)", "srli t1, t1, 2", "bgeu s3, t1, w2n", "addi s5, s5, 1",
          "bgez s6, w2f", "mv s6, t0", "w2f:", "mv s7, t0",
          "w2n:", "addi t6, t6, 4", "addi t0, t0, 1", f"li t2, {N // W}", "bne t0, t2, w2"]
    _puts(a, "\nnoise_floor_dbfs=")
    a += ["mv a0, s3", "call dbfs10", "li a1, 1", "call print_fixed"]
    _puts(a, " (quietest 64-sample window)\nloudest_window_dbfs=")
    a += ["mv a0, s4", "call dbfs10", "li a1, 1", "call print_fixed"]
    _puts(a, "\nactive_windows=")
    a += ["mv a0, s5", "call print_int"]
    _puts(a, f"/{N // W} (power > 4x quietest)")
    a += ["bltz s6, noact"]
    _puts(a, " first_active_sample=")
    a += ["slli a0, s6, 6", "call print_int"]
    _puts(a, " last_active_sample_end=")
    a += ["addi a0, s7, 1", "slli a0, a0, 6", "call print_int", "noact:",
          "la t4, mode", "lw t4, 0(t4)", f"li t5, {LEVELS}", "bne t4, t5, full", "j fin", "full:"]
    # ---------------- pass 2: averaged DFT16 power (`frames` frames, hop N / frames)
    # Each frame is pre-shifted (x >> 5) once into scratch RAM; the four twiddle magnitudes live in
    # s5..s8, so a DFT term is 2 loads + up to 4 multiply-adds (no reloads, no constant loads).
    mags = {32767: "s5", 30273: "s6", 23170: "s7", 12539: "s8"}
    a += ["la t6, binpow", "li t0, 0", "dz:", "sw zero, 0(t6)", "addi t6, t6, 4", "addi t0, t0, 1", "li t1, 32", "bne t0, t1, dz",
          "mv s10, s2", "li s11, 0", "la s9, frame"] + [f"li {r}, {m}" for m, r in mags.items()] + ["df:"]
    for n in range(FN):
        a += [f"lh t0, {4 * n}(s10)", f"lh t1, {4 * n + 2}(s10)", "srai t0, t0, 5", "srai t1, t1, 5",
              f"sw t0, {8 * n}(s9)", f"sw t1, {8 * n + 4}(s9)"]
    for k in range(FN):
        a += ["li s3, 0", "li s4, 0"]
        for n in range(FN):
            c = round(math.cos(2 * math.pi * k * n / FN) * 32767)
            s = round(-math.sin(2 * math.pi * k * n / FN) * 32767)
            a += [f"lw t0, {8 * n}(s9)", f"lw t1, {8 * n + 4}(s9)"]
            if c:  # re += xr c ; im += xi c
                r, op = mags[abs(c)], "add" if c > 0 else "sub"
                a += [f"mul t3, t0, {r}", f"{op} s3, s3, t3", f"mul t3, t1, {r}", f"{op} s4, s4, t3"]
            if s:  # re -= xi s ; im += xr s
                r = mags[abs(s)]
                a += [f"mul t3, t1, {r}", f"{'sub' if s > 0 else 'add'} s3, s3, t3",
                      f"mul t3, t0, {r}", f"{'add' if s > 0 else 'sub'} s4, s4, t3"]
        a += ["srai s3, s3, 15", "srai s4, s4, 15", "mul s3, s3, s3", "mul s4, s4, s4", "add s3, s3, s4",
              "la t6, binpow", f"addi t6, t6, {8 * k}", "lw t0, 0(t6)", "lw t1, 4(t6)", "add t0, t0, s3", "sltu t2, t0, s3",
              "add t1, t1, t2", "sw t0, 0(t6)", "sw t1, 4(t6)"]
    a += [f"addi s10, s10, {4 * (N // frames)}", "addi s11, s11, 1", f"li t0, {frames}", "beq s11, t0, dfend", "j df", "dfend:"]  # body > 4 KB
    # dominant bin (64-bit compare) and share = bin / total * 100 (both >> 10)
    a += ["la t6, binpow", "li s3, 0", "li s4, 0", "li s5, 0", "li s6, 0", "li s7, 0", "li t0, 0",
          "db:", "lw t1, 0(t6)", "lw t2, 4(t6)", "add s6, s6, t1", "sltu t3, s6, t1", "add s7, s7, t3", "add s7, s7, t2",
          "bltu t2, s4, dbn", "bltu s4, t2, dby", "bgeu s3, t1, dbn",
          "dby:", "mv s3, t1", "mv s4, t2", "mv s5, t0",
          "dbn:", "addi t6, t6, 8", "addi t0, t0, 1", f"li t3, {FN}", "bne t0, t3, db",
          "srli s3, s3, 14", "slli t0, s4, 18", "or s3, s3, t0", "srli s6, s6, 14", "slli t0, s7, 18", "or s6, s6, t0"]
    _puts(a, "\ndominant_dft16_bin=")
    a += ["mv a0, s5", "call print_int"]
    _puts(a, " freq_cycles_per_sample=")
    a += ["mv a0, s5", "li t0, 8", "blt a0, t0, dbpos", "addi a0, a0, -16", "dbpos:",
          "li t0, 625", "mul a0, a0, t0", "li a1, 4", "call print_fixed"]          # k/16 = k*625/10^4
    _puts(a, " power_share_percent=")
    a += ["li t0, 100", "mul a0, s3, t0", "beqz s6, dbz", "divu a0, a0, s6", "j dbp", "dbz:", "li a0, 0", "dbp:", "call print_int"]
    a += ["la t0, capfs", "lw t3, 0(t0)", "beqz t3, nofs1"]
    _puts(a, " freq_hz=")
    a += ["mv t0, s5", "li t1, 8", "blt t0, t1, fhp", "addi t0, t0, -16", "fhp:", "la t1, capfs", "lw t3, 0(t1)",
          "mul a0, t0, t3", "li t1, 16", "div a0, a0, t1", "call print_int", "nofs1:"]           # k fs / 16, exact
    _puts(a, f" (averaged 16-point DFT, {frames} frames)")
    # ---------------- CFO from the pass-1 sum
    a += ["la t2, cfosum", "lw s3, 0(t2)", "lw s4, 4(t2)", "lw s7, 8(t2)", "lw s8, 12(t2)",
          "srli t0, s3, 13", "slli t1, s4, 19", "or t0, t0, t1",       # Re >> 13
          "srli t1, s7, 13", "slli t2, s8, 19", "or t1, t1, t2"]        # Im >> 13
    asm_c = _Collector()
    cordic_asm(asm_c, "t0", "t1", "t2")
    a += asm_c.lines
    a += ["mv s3, t2"]                                                                  # angle, Q32 cycles
    _puts(a, "\ncfo_cycles_per_sample=")
    a += ["li t0, 10000", "mulh a0, s3, t0", "li a1, 4", "call print_fixed"]          # angle/2^32 * 10^4
    a += ["la t0, capfs", "lw t3, 0(t0)", "beqz t3, nofs2"]
    _puts(a, " cfo_hz=")
    a += ["la t0, capfs", "lw t3, 0(t0)", "mulh a0, s3, t3", "call print_int", "nofs2:"]  # floor(angle fs / 2^32)
    _puts(a, " (phase of the lag-1 autocorrelation)")
    a += ["fin:"]
    if not_measured:
        nm = NOT_MEASURED if modulation is None else NOT_MEASURED.replace("modulation, ", "modulation class (use rf_modulation), ")
        a += ["la t4, mode", "lw t4, 0(t4)", f"li t5, {LEVELS}", "beq t4, t5, nmlev", "j nmfull", "nmlev:"]
        _puts(a, "\nnot_measured=" + "spectrum and frequency offset (use rf_analyze), " + nm + NOTE)
        a += ["j nmend", "nmfull:"]
        _puts(a, "\nnot_measured=" + nm + NOTE)
        a += ["nmend:"]
    a += ["li a0, 10", "li a7, 1", "ecall", "j done"]
    rodata = []
    if modulation is not None:
        a += _modpath(modulation, rodata)
    a += ["bad:"]
    _puts(a, f"error: capture must be an integer 0..{len(captures) - 1}\n")
    a += ["done:", f"li a0, {TOOL_RESPONSE_END}", "li a7, 1", "ecall", "li a0, 10", "li a7, 1", "ecall",
          f"li a0, {THINK}", "li a7, 1", "ecall", "li a0, 10", "li a7, 1", "ecall", "li a0, 10", "li a7, 1", "ecall",
          f"li a0, {THINK_END}", "li a7, 1", "ecall", "li a0, 10", "li a7, 1", "ecall",
          "lw ra, 0(sp)", "addi sp, sp, 16", "ret"]
    a += _library()
    a += [".data", "winpow:", f".space {4 * (N // W)}", "binpow:", f".space {8 * FN}", "frame:", f".space {8 * FN}", "cfosum:", ".space 16", "numbuf:", ".space 16", "mode:", ".space 4", "capfs:", ".space 4",
          ".rodata", "log2t:", ".word " + ", ".join(map(str, LOG2_T)),
          "fstab:", ".word " + ", ".join(str(sample_rate(lab)) for lab, _ in captures),
          "captab:", ".word " + ", ".join(f"cap{i}" for i in range(len(captures))),
          "labtab:", ".word " + ", ".join(f"lab{i}" for i in range(len(captures)))] + rodata
    for i, (label, x) in enumerate(captures):
        a.append(f"lab{i}:")
        a.append(".byte " + ", ".join(str(b) for b in f"capture={i} {label}".encode()) + ", 0")
        a.append(".align 2")
    for i, (label, x) in enumerate(captures):
        a.append(f"cap{i}:")
        flat = [v for s in x for v in s]
        for j in range(0, len(flat), 32):
            a.append(".half " + ", ".join(str(v) for v in flat[j:j + 32]))
    return "\n".join(line for line in a if line) + "\n"


def _pass1(sfx, cfo):
    """Pass 1 over the N samples: mean power, peak, 16 window powers and (cfo) the lag-1 CFO sum
    R = sum x[n] conj(x[n-1]) (exact 64-bit; x[-1] = 0), sharing loads and the loop."""
    a = [f"p1in{sfx}:", "mv t6, s2", "li s3, 0", "li s4, 0", "li s5, 0", "li s6, 0", "li s7, 0",  # sum lo/hi, peak, peak idx, n
         "la s8, winpow", "li s9, 0",
         "li a1, 0", "li a2, 0", "li a3, 0", "li a4, 0", "li a5, 0", "li a6, 0",
         f"p1{sfx}:", "lh s10, 0(t6)", "lh s11, 2(t6)"]
    if cfo:
        for acc_lo, acc_hi, x, y, sub in (("a1", "a2", "s10", "a5", False), ("a1", "a2", "s11", "a6", False),
                                           ("a3", "a4", "s11", "a5", False), ("a3", "a4", "s10", "a6", True)):
            a += [f"mul t4, {x}, {y}", f"mulh t5, {x}, {y}"]
            if sub:
                a += [f"sltu t3, {acc_lo}, t4", f"sub {acc_lo}, {acc_lo}, t4", f"sub {acc_hi}, {acc_hi}, t5", f"sub {acc_hi}, {acc_hi}, t3"]
            else:
                a += [f"add {acc_lo}, {acc_lo}, t4", f"sltu t3, {acc_lo}, t4", f"add {acc_hi}, {acc_hi}, t5", f"add {acc_hi}, {acc_hi}, t3"]
        a += ["mv a5, s10", "mv a6, s11"]
    a += ["mul t0, s10, s10", "mul t1, s11, s11", "add t0, t0, t1",
          "add s3, s3, t0", "sltu t2, s3, t0", "add s4, s4, t2",
          "srli t2, t0, 6", "add s9, s9, t2",          # window mean = sum of p >> 6 (no overflow)
          f"bgeu s5, t0, p1np{sfx}", "mv s5, t0", "mv s6, s7", f"p1np{sfx}:",
          "addi s7, s7, 1", "andi t2, s7, 63", f"bnez t2, p1nw{sfx}", "sw s9, 0(s8)", "addi s8, s8, 4", "li s9, 0",
          f"p1nw{sfx}:", "addi t6, t6, 4", f"li t2, {N}", f"bne s7, t2, p1{sfx}",
          "srli s3, s3, 10", "slli t2, s4, 22", "or s3, s3, t2"]           # mean power u32
    if cfo:
        a += ["la t2, cfosum", "sw a1, 0(t2)", "sw a2, 4(t2)", "sw a3, 8(t2)", "sw a4, 12(t2)"]  # print routines use a1
    return a


class _Collector(list):
    """Adapter so cordic_asm (which calls asm(...) and asm.li(...)) can append raw lines."""
    @property
    def lines(self):
        return list(self)

    def __call__(self, *lines):
        self.extend(lines)

    def li(self, reg, value):
        self.append(f"li {reg}, {int(value)}")


def _library():
    """print_str(a0), print_int(a0 signed), print_fixed(a0 value*10^a1, a1 decimals), dbfs10(a0 u32)."""
    return [
        "print_str:", "mv t6, a0", "ps1:", "lbu a0, 0(t6)", "beqz a0, ps2", "li a7, 1", "ecall", "addi t6, t6, 1", "j ps1", "ps2:", "ret",
        "print_int:", "bgez a0, pi1", "mv t5, a0", "li a0, 45", "li a7, 1", "ecall", "neg a0, t5",
        "pi1:", "la t6, numbuf", "li t4, 10", "li t3, 0",
        "pi2:", "remu t5, a0, t4", "addi t5, t5, 48", "add t2, t6, t3", "sb t5, 0(t2)", "addi t3, t3, 1", "divu a0, a0, t4", "bnez a0, pi2",
        "pi3:", "addi t3, t3, -1", "add t2, t6, t3", "lbu a0, 0(t2)", "li a7, 1", "ecall", "bnez t3, pi3", "ret",
        # print_fixed: sign, integer part, '.', a1 fraction digits (zero padded)
        "print_fixed:", "addi sp, sp, -16", "sw ra, 0(sp)", "sw s0, 4(sp)", "sw s1, 8(sp)",
        "bgez a0, pf1", "mv t5, a0", "li a0, 45", "li a7, 1", "ecall", "neg a0, t5",
        "pf1:", "li s1, 1", "mv t0, a1", "pf2:", "beqz t0, pf3", "li t1, 10", "mul s1, s1, t1", "addi t0, t0, -1", "j pf2",
        "pf3:", "remu s0, a0, s1", "divu a0, a0, s1", "sw a1, 12(sp)", "call print_int", "li a0, 46", "li a7, 1", "ecall",
        "lw t0, 12(sp)", "pf4:", "beqz t0, pf5", "li t1, 10", "divu s1, s1, t1", "divu a0, s0, s1", "remu s0, s0, s1",
        "addi a0, a0, 48", "li a7, 1", "ecall", "addi t0, t0, -1", "j pf4",
        "pf5:", "lw ra, 0(sp)", "lw s0, 4(sp)", "lw s1, 8(sp)", "addi sp, sp, 16", "ret",
        # dbfs10: 10 * 10*log10(P / 2^30) as an integer (tenths of dB); P = 0 -> -999
        "dbfs10:", "bnez a0, db1", "li a0, -999", "ret",
        "db1:", "li t0, 31", "db2:", "srl t1, a0, t0", "bnez t1, db3", "addi t0, t0, -1", "j db2",   # t0 = msb index
        "db3:", "li t1, 31", "sub t1, t1, t0", "sll t2, a0, t1",                                     # normalized mantissa
        "srli t3, t2, 27", "andi t3, t3, 15", "srli t4, t2, 11", "li t5, 65535", "and t4, t4, t5",   # segment, position
        "la t5, log2t", "slli t3, t3, 2", "add t5, t5, t3", "lw t1, 0(t5)", "lw t5, 4(t5)", "sub t5, t5, t1",
        "mul t5, t5, t4", "srli t5, t5, 16", "add t1, t1, t5", "slli t0, t0, 16", "add t0, t0, t1",  # log2(P) Q16
        "li t1, 1966080", "sub t0, t0, t1", "srai t0, t0, 8",                                        # (log2 P - 30) Q8
        "li t1, 30103", "mul t0, t0, t1", "li t1, 128000", "bgez t0, dbr", "neg t1, t1", "dbr:", "add t0, t0, t1",
        "li t1, 256000", "div a0, t0, t1", "ret",                                                     # nearest tenth                   # tenths of dB
    ]


def analyze(x, frames=FRAMES, fs=0):
    """Independent host model of the report (float where the firmware is fixed point); fs (Hz) adds
    freq_hz and cfo_hz as firmware v3 prints them when the sample rate is known."""
    import numpy as np
    z = np.array([complex(i, q) for i, q in x]) / 32768
    p = abs(z) ** 2
    win = np.floor(np.abs(np.array(x, float)) ** 2 @ [1, 1] / 64).reshape(-1, W).sum(1) / 2**30  # firmware: sum of p >> 6
    db = lambda v: 10 * math.log10(v) if v > 0 else -99.9  # noqa: E731
    act = np.nonzero(win / 4 > win.min())[0]
    fr = np.array([z[f * (N // frames):f * (N // frames) + FN] for f in range(frames)])
    spec = (abs(np.fft.fft(fr, axis=1)) ** 2).sum(0)
    k = int(np.argmax(spec))
    r = np.sum(z[1:] * np.conj(z[:-1]))
    return {"mean_power_dbfs": db(p.mean()), "peak_power_dbfs": db(p.max()), "at_sample": int(np.argmax(p)),
            "noise_floor_dbfs": db(win.min()), "loudest_window_dbfs": db(win.max()), "active_windows": len(act),
            "dominant_dft16_bin": k, "freq_cycles_per_sample": (k - 16 if k >= 8 else k) / 16,
            "power_share_percent": 100 * spec[k] / spec.sum(), "cfo_cycles_per_sample": float(np.angle(r) / (2 * math.pi)),
            **({"freq_hz": (k - 16 if k >= 8 else k) * fs // 16, "cfo_hz": float(np.angle(r) / (2 * math.pi)) * fs} if fs else {})}
