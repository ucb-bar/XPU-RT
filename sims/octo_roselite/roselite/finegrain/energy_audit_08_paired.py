#!/usr/bin/env python3
"""AUDIT PASS 8 -- why the medians are unstable, and the paired test that fixes it.

All t2 values here are DE-SPIKED (isolated single-tick solver transients removed).

Every cell of a task uses init_rng=100 and episode_ids 0..23, so episode i of
`egg_lat0` and episode i of `egg_cpu685` START FROM THE SAME SCENE. The published
figure nevertheless compares UNPAIRED medians, which throws that away -- and the
per-cell distribution is BIMODAL (a success terminates the episode early; a failure
runs to the horizon), so the median sits on the boundary between the two modes and
moves whenever the success count moves by one.
"""
import json
from pathlib import Path
import numpy as np

def flag_transients(pt, thr=25.0, nb=5.0):
    """Isolated single-tick excursions in sum_i tau_i^2 (PhysX inverse-dynamics
    transients -- see energy_audit_05_spikes.py). A contact reaction is sustained, so
    this shape cannot flag one."""
    m = float(np.median(pt))
    if m <= 0:
        return np.zeros(len(pt), bool)
    return (pt > thr * m) & np.r_[True, pt[:-1] < nb * m] & np.r_[pt[1:] < nb * m, True]


rng = np.random.default_rng(0)
HERE = Path(__file__).parent
SRC = HERE / "traces_torque2"
ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300", "pipe200fix",
        "serial283", "fp32_555", "cpu685"]
TASKS = ["egg", "spoon", "coke", "drawer"]

D = {}
for t in TASKS:
    for a in ARMS:
        cell = f"{t}_{a}"
        s = json.load(open(SRC / cell / "summary.json"))
        dt = s["tick_ms"] / 1000.0
        rec = {}
        for e in s["episodes"]:
            qf = np.load(SRC / cell / f"ep{e['episode_id']:02d}_qf.npy")
            pt = (qf ** 2).sum(1)
            pt = pt[~flag_transients(pt)]          # drop solver transients
            rec[e["episode_id"]] = (pt.sum() * dt, len(qf) * dt, e["success"])
        D[cell] = rec

print("=" * 96)
print("1.  The distribution behind each published median (egg / spoon, ideal arm)")
print("=" * 96)
for cell in ["egg_lat0", "spoon_lat0", "egg_cpu685", "coke_lat0"]:
    v = np.array([D[cell][k][0] for k in sorted(D[cell])])
    d = np.array([D[cell][k][1] for k in sorted(D[cell])])
    sc = np.array([D[cell][k][2] for k in sorted(D[cell])])
    o = np.argsort(v)
    print(f"\n  {cell}   median {np.median(v):.1f}   mean {v.mean():.1f}")
    print("    t2 sorted : " + " ".join(f"{x:.0f}" for x in v[o]))
    print("    dur       : " + " ".join(f"{x:.0f}" for x in d[o]))
    print("    success   : " + " ".join("S" if x else "." for x in sc[o]))
    print(f"    -> episodes that SUCCEEDED: t2 median {np.median(v[sc]):.1f} "
          f"(dur {np.median(d[sc]):.1f} s);  FAILED: {np.median(v[~sc]):.1f} "
          f"(dur {np.median(d[~sc]):.1f} s)")

print()
print("=" * 96)
print("2.  PAIRED per-episode ratio vs the ideal arm  (same init_rng, same episode id)")
print("    reported as the median of the 24 per-episode ratios, with a paired-bootstrap")
print("    95% CI -- compare with the unpaired numbers the figure prints.")
print("=" * 96)
B = 20000
for t in TASKS:
    base = D[f"{t}_lat0"]
    print(f"\n  {t:<8}{'arm':<12}{'unpaired x':>12}{'PAIRED x [95% CI]':>28}"
          f"{'paired, per-second x':>24}")
    for a in ARMS:
        c = D[f"{t}_{a}"]
        ks = sorted(set(c) & set(base))
        r = np.array([c[k][0] / base[k][0] for k in ks])
        rr = np.array([(c[k][0]/c[k][1]) / (base[k][0]/base[k][1]) for k in ks])
        i = rng.integers(0, len(r), (B, len(r)))
        lo, hi = np.percentile(np.median(r[i], 1), [2.5, 97.5])
        j = rng.integers(0, len(rr), (B, len(rr)))
        l2, h2 = np.percentile(np.median(rr[j], 1), [2.5, 97.5])
        up = np.median([c[k][0] for k in ks]) / np.median([base[k][0] for k in ks])
        print(f"  {'':<8}{a:<12}{up:>12.2f}"
              f"{f'{np.median(r):.2f} [{lo:.2f},{hi:.2f}]':>28}"
              f"{f'{np.median(rr):.2f} [{l2:.2f},{h2:.2f}]':>24}")
