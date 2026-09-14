#!/usr/bin/env python3
"""Does the ACTUATION MODEL, by itself, move the latency penalty?

The google_robot arms had to be run with --actuation native (3 Hz actuation,
27 Hz arrival grid) while the widowx arms in RESULTS.txt were run with
--actuation fine (25 Hz actuation, deltas rescaled). A difference-in-differences
between the two embodiments therefore confounds the embodiment with the model,
unless the model is shown not to matter.

This runs the SAME widowx task (eggplant), the SAME seeds (100-104) and the SAME
GPU (the local TITAN RTX) under both models, so everything except the model is
held fixed. The penalty is a within-run paired quantity, and the two models share
seeds, so the model difference is paired too -- t(4) intervals.
"""
from __future__ import annotations
import json, glob, math, collections
from pathlib import Path

HERE = Path(__file__).parent.parent
TC = {4: 2.776, 9: 2.262}
SEEDS = [100, 101, 102, 103, 104]
ARMS = ["pipe110", "serial283"]

def load(d):
    out = {}
    for f in glob.glob(str(HERE / d / "egg_*_rng*" / "summary.json")):
        n = Path(f).parent.name
        head, seed = n.rsplit("_rng", 1)
        arm = head.split("_", 1)[1]
        s = json.load(open(f))
        eps = s["episodes"]
        out[(arm, int(seed))] = 100.0 * sum(e["success"] for e in eps) / len(eps)
    return out

def ci(d, df=None):
    n = len(d); mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n); tc = TC[n - 1]
    return mu, mu - tc * se, mu + tc * se, n

fine, nat = load("actfine_widowx"), load("actnative_widowx")
print("eggplant, seeds 100-104, local TITAN RTX, 24 configs -- SAME GPU, SAME SEEDS\n")
print(f"{'':11s} {'lat0 SR':>18s} {'arm SR':>18s} {'penalty vs lat0':>26s}")
pen = {}
for tag, D in (("fine", fine), ("native", nat)):
    print(f"-- --actuation {tag}")
    l0 = [D[("lat0", s)] for s in SEEDS]
    print(f"{'lat0':11s} {sum(l0)/5:17.1f}% {'':18s}")
    for arm in ARMS:
        a = [D[(arm, s)] for s in SEEDS]
        d = [D[(arm, s)] - D[("lat0", s)] for s in SEEDS]
        pen[(tag, arm)] = d
        mu, lo, hi, n = ci(d)
        print(f"{arm:11s} {'':18s} {sum(a)/5:17.1f}% {mu:+8.1f} [{lo:+6.1f},{hi:+6.1f}] n={n}")
print("\nMODEL EFFECT (paired on seed): penalty(native) - penalty(fine)")
for arm in ARMS:
    d = [n - f for n, f in zip(pen[("native", arm)], pen[("fine", arm)])]
    mu, lo, hi, n = ci(d)
    verdict = "no evidence the model moves it" if lo <= 0 <= hi else "MODEL MATTERS"
    print(f"  {arm:11s} {mu:+6.1f} [{lo:+6.1f},{hi:+6.1f}] n={n}   {verdict}")
print("\nAWS 10-seed fine reference (RESULTS.txt): pipe110 +13.3 [+0.6,+26.0], "
      "serial283 -32.9 [-43.8,-22.0]")
print("Caveat: 5 seeds. The interval on the model effect is wide -- this rules out\n"
      "a LARGE model artifact, not a small one.")
