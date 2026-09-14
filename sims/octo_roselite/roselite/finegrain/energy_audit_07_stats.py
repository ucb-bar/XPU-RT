#!/usr/bin/env python3
"""AUDIT PASS 7 -- is the between-arm signal larger than the n=24 noise, and does it
survive (a) de-spiking and (b) dividing out mission length?

Bootstrap CI on the median ratio vs the ideal arm, 20000 resamples, per cell.
Also repeats the 'is it a clock?' regression on the de-spiked series.
"""
import json
from pathlib import Path
import numpy as np

def flag_transients(pt, thr=25.0, nb=5.0):
    """Isolated single-tick excursions in sum_i tau_i^2.

    A tick is flagged when its sum tau^2 exceeds `thr` x the episode median AND BOTH
    neighbouring ticks are below `nb` x the median -- i.e. the excursion lasts exactly
    one tick and the signal is back to baseline immediately before and after. A contact
    reaction is sustained while the arm pushes, so this cannot flag one; a PhysX
    inverse-dynamics transient is exactly this shape. Verified by eye against
    egg_fp32_555 ep07 t=425 (tau 1.6 -> 608 -> 1.7 N.m) and egg_p130w275 ep16 t=389.
    """
    import numpy as np
    m = float(np.median(pt))
    if m <= 0:
        return np.zeros(len(pt), bool)
    big = pt > thr * m
    prev = np.r_[True, pt[:-1] < nb * m]
    nxt = np.r_[pt[1:] < nb * m, True]
    return big & prev & nxt
from scipy.stats import pearsonr

rng = np.random.default_rng(0)
HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]
VLIM = {8: np.array([np.pi]*6 + [1.0, 1.0]),
        11: np.array([2., 2., 2., 2.5, 3., 3., 3., 1.3, 1.3, 2., 2.])}
B = 20000

D = {}
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        t2, t2c, dur = [], [], []
        for e in s["episodes"]:
            qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
            w = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qvel.npy")
            pt = (qf ** 2).sum(1)
            bad = flag_transients(pt)
            t2.append(pt.sum() * dt); t2c.append(pt[~bad].sum() * dt)
            dur.append(len(qf) * dt)
        D[cell] = dict(t2=np.array(t2), t2c=np.array(t2c), dur=np.array(dur),
                       sr=100.0 * s["n_success"] / s["n_episodes"])

def ci(num, den):
    """bootstrap CI on median(num)/median(den), resampling both cells."""
    i = rng.integers(0, len(num), (B, len(num)))
    j = rng.integers(0, len(den), (B, len(den)))
    r = np.median(num[i], 1) / np.median(den[j], 1)
    return np.percentile(r, [2.5, 97.5])

print("=" * 108)
print("Median ratio vs the ideal arm, with a 20 000-resample bootstrap 95% CI.")
print("  t2      as published            t2*  de-spiked            rate*  de-spiked, per second")
print("=" * 108)
for t in TASKS:
    b = D[f"{t}_lat0"]
    print(f"\n  {t}")
    print(f"    {'arm':<12}{'t2 x [95% CI]':>26}{'t2* x [95% CI]':>26}"
          f"{'rate* x [95% CI]':>26}{'succ%':>7}")
    for a in ARMS:
        c = D[f"{t}_{a}"]
        l1, h1 = ci(c["t2"], b["t2"]); l2, h2 = ci(c["t2c"], b["t2c"])
        l3, h3 = ci(c["t2c"] / c["dur"], b["t2c"] / b["dur"])
        f = lambda v, l, h: f"{v:.2f} [{l:.2f},{h:.2f}]"
        print(f"    {a:<12}"
              f"{f(np.median(c['t2'])/np.median(b['t2']), l1, h1):>26}"
              f"{f(np.median(c['t2c'])/np.median(b['t2c']), l2, h2):>26}"
              f"{f(np.median(c['t2c']/c['dur'])/np.median(b['t2c']/b['dur']), l3, h3):>26}"
              f"{c['sr']:>7.1f}")

print()
print("=" * 108)
print("'Is it a clock?' repeated on the DE-SPIKED series (216 episodes per task)")
print("=" * 108)
for t in TASKS:
    x = np.concatenate([D[f"{t}_{a}"]["dur"] for a in ARMS])
    y = np.concatenate([D[f"{t}_{a}"]["t2c"] for a in ARMS])
    k = (y * x).sum() / (x ** 2).sum()
    r2 = 1 - ((y - k * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {t:<8} r(t2*,dur) = {pearsonr(x,y)[0]:+.3f}   "
          f"R^2 of the one-parameter model t2* = {k:.2f}*duration : {r2:.3f}")

print()
print("=" * 108)
print("How much of the published SPREAD across the 9 arms survives each correction?")
print("  spread = max/min of the 9 median values")
print("=" * 108)
print(f"{'task':<9}{'published t2':>15}{'de-spiked':>13}{'per-second':>13}"
      f"{'per-second, de-spiked':>24}")
for t in TASKS:
    v1 = np.array([np.median(D[f"{t}_{a}"]["t2"]) for a in ARMS])
    v2 = np.array([np.median(D[f"{t}_{a}"]["t2c"]) for a in ARMS])
    v3 = np.array([np.median(D[f"{t}_{a}"]["t2"] / D[f"{t}_{a}"]["dur"]) for a in ARMS])
    v4 = np.array([np.median(D[f"{t}_{a}"]["t2c"] / D[f"{t}_{a}"]["dur"]) for a in ARMS])
    print(f"{t:<9}" + "".join(f"{v.max()/v.min():>15.2f}x" if i == 0 else
                              f"{v.max()/v.min():>12.2f}x"
                              for i, v in enumerate([v1, v2, v3, v4])))
