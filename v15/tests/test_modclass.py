import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfpu_v15.cpu.modclass import NF, IntMLP, ModFeatures, features  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.modgen import CLASSES, CONST, capture  # noqa: E402


def test_constellation_sizes_and_capture_range():
    size = {"OOK": 2, "BPSK": 2, "QPSK": 4}
    for c, pts in CONST.items():
        assert len(pts) == (size.get(c) or int("".join(ch for ch in c if ch.isdigit()))), c
    rng = np.random.default_rng(0)
    for c in CLASSES:
        x = capture(c, rng, 20)
        assert len(x) == 1024 and all(-32768 <= v <= 32767 for p in x for v in p)


def test_mod_features_program_matches_reference():
    rng = random.Random(5)
    R = Runner(vx=False, batch=1)
    k = ModFeatures()
    caps = [capture("LoRa", np.random.default_rng(1), 15), capture("AM-DSB-WC", np.random.default_rng(2), 25),
            [(rng.randint(-32768, 32767), rng.randint(-32768, 32767)) for _ in range(1023)] + [(-32768, -32768)],
            [(0, 0)] * 1024]
    for x in caps:
        r = R.run(k, x, block=1024, neural=False)
        assert r.exact_outputs and len(r.outputs[-1]) == NF


def test_int_mlp_program_matches_reference():
    rng = random.Random(8)
    H, C = 6, 25
    m = IntMLP([[rng.randint(-32768, 32767) for _ in range(NF)] for _ in range(H)], [rng.randint(-99999, 99999) for _ in range(H)],
               [[rng.randint(-32768, 32767) for _ in range(H)] for _ in range(C)], [rng.randint(-999, 999) for _ in range(C)], s1=13)
    f = features(capture("16QAM", np.random.default_rng(3), 20))
    r = Runner(vx=False, batch=1).run(m, [(v, 0) for v in f] + [(0, 0)] * (32 - NF), block=32, neural=False)
    assert r.exact_outputs and r.outputs[-1] == m.forward(f)
