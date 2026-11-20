"""PROTOCOL Amendment 12 P12.3: firmware v3 on the 16 Stage 2 captures (RV32 reference machine):
freq_hz equal to the host value, |cfo_hz - fs * host CFO| <= fs 2^-12, method notes present.
Writes v15/reports/firmware-v3-check.json."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
sys.stdout.reconfigure(encoding="utf-8")


def main():
    from chipformer.rv import bridge as BR, ref
    from chipformer.rv.asm import assemble
    from rfpu_v15.stage2.firmware import analyze, sample_rate, source
    from stage2_chat import capture_set
    sys.path.insert(0, str(ROOT / "v15/tests"))
    from test_stage2_firmware import CALL, rows
    caps = capture_set()
    prog = assemble(source(caps))
    out = []
    for i, (label, x) in enumerate(caps):
        m = ref.Machine.from_program(prog); BR.park(m)
        m.halted = False; m.text = rows(CALL.format(i)); m.run(10**8)
        rep = m.out.decode()
        fs = sample_rate(label)
        want = analyze(x, fs=fs)
        notes = "(averaged 16-point DFT" in rep and "(phase of the lag-1 autocorrelation)" in rep
        if not fs:  # unknown sample rate: no Hz fields may be printed
            row = {"capture": i, "label": label, "fs": 0, "freq_ok": "freq_hz" not in rep, "cfo_ok": "cfo_hz" not in rep, "notes_ok": notes}
            out.append(row); print(row)
            continue
        got = {k: int(re.search(k + r"=(-?[0-9]+)", rep).group(1)) for k in ("freq_hz", "cfo_hz")}
        row = {"capture": i, "label": label, "fs": fs, **{f"{k}_firmware": v for k, v in got.items()},
               "freq_hz_host": want["freq_hz"], "cfo_hz_host": want["cfo_hz"],
               "freq_ok": got["freq_hz"] == want["freq_hz"], "cfo_ok": abs(got["cfo_hz"] - want["cfo_hz"]) <= fs * 2**-12,
               "notes_ok": "(averaged 16-point DFT" in rep and "(phase of the lag-1 autocorrelation)" in rep}
        out.append(row); print(row)
    ok = all(r["freq_ok"] and r["cfo_ok"] and r["notes_ok"] for r in out)
    (ROOT / "v15/reports/firmware-v3-check.json").write_text(json.dumps({"protocol": "PROTOCOL.md Amendment 12 P12.3",
                                                                       "pass": ok, "captures": out}, indent=2), encoding="utf-8")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
