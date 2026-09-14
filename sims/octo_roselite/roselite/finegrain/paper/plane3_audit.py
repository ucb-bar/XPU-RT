#!/usr/bin/env python3
"""Repeatable numeric audit of the in-flight plane sweep.

Everything the figures assert, checked as numbers so two snapshots can be diffed.
Run it after each g5grid/fetch_plane3.sh; the caveats it prints are the ones that
have to clear before the plane's energy panels can be quoted.
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
import plane3_lib as L

HERE = Path(__file__).parent
FLOOR = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}
NEAR_ZERO = 100.0

D = L.load()
T = L.table()
plane = L.arm_plane()
old = json.load(open(HERE / "grid_e2e_success.json"))
ncell = sum(len(s) for s in D.values())
print(f"=== PLANE AUDIT: {ncell}/1760 cells, "
      f"{sum(d['n_episodes'] for s in D.values() for d in s.values()):,} episodes ===\n")

LAT = np.array([plane[a][0] for a in plane]); PER = np.array([plane[a][1] for a in plane])
print(f"plane geometry  Spearman(latency, period) = {spearmanr(LAT, PER).statistic:+.3f}\n")

print("INTEGRITY")
dup = ncell - len({(t, a, s) for (t, a), sd in D.items() for s in sd})
zero = [(t, a, s) for (t, a), sd in D.items() for s, d in sd.items() if d["n_success"] == 0]
neps = collections.Counter(d["n_episodes"] for s in D.values() for d in s.values())
pairs = len(D)
ns = [len(s) for s in D.values()]
print(f"  duplicate cells {dup} | zero-success cells {len(zero)} | n_episodes {dict(neps)}")
print(f"  (task,arm) pairs {pairs}/176 | seeds/pair min {min(ns)} med {int(np.median(ns))} max {max(ns)}\n")

print(f"{'task':7s} {'succ rho(per)':>14s} {'E rho(per) med':>15s} {'E rho(per) mean':>16s} "
      f"{'rho(succ,E)':>12s} {'repl r':>8s} {'offset':>8s} {'SEM/spread':>11s} {'zero-E %':>9s}")
flag = []
for task, _, _ in L.TASKS:
    arms = sorted(T[task])
    per = np.array([plane[a][1] for a in arms])
    suc = np.array([T[task][a]["success"] for a in arms])
    emed = np.array([T[task][a]["energy"] for a in arms])
    emean = np.array([np.mean([np.mean([e[L.ECH] for e in d["episodes"]])
                               for d in D[(task, a)].values()]) for a in arms])
    r_m, r_a = spearmanr(per, emed).statistic, spearmanr(per, emean).statistic
    o = np.array([old[task][a]["rate"] if a in old[task] else np.nan for a in arms])
    src = np.array([old[task].get(a, {}).get("source", "") for a in arms])
    ok = np.isfinite(o) & (src == "measured")
    pr = pearsonr(o[ok], suc[ok]); bias = float(np.mean(suc[ok] - o[ok]))
    sem = np.mean([np.std(T[task][a]["success_seeds"], ddof=1) / np.sqrt(T[task][a]["n_seeds"])
                   for a in arms if T[task][a]["n_seeds"] > 1])
    ep = np.array([x[L.ECH] for (t, a), sd in D.items() if t == task
                   for d in sd.values() for x in d["episodes"]])
    zf = 100 * float((ep < NEAR_ZERO).mean())
    print(f"{task:7s} {spearmanr(per, suc).statistic:+14.3f} {r_m:+15.3f} {r_a:+16.3f} "
          f"{spearmanr(suc, emed).statistic:+12.3f} {pr.statistic:+8.3f} "
          f"{bias:+6.1f}pt {np.std(suc, ddof=1)/sem:11.2f} {zf:9.1f}")
    if r_m * r_a < 0:
        flag.append(f"{task}: energy trend SIGN-FLIPS between estimators "
                    f"(median {r_m:+.3f} vs mean {r_a:+.3f}) -- do not quote")
    if np.std(suc, ddof=1) / sem < 1.3:
        flag.append(f"{task}: across-arm success spread is only "
                    f"{np.std(suc, ddof=1)/sem:.2f}x the per-arm SEM -- surface not resolved")

print("\nENERGY ESTIMATOR: seed-to-seed CV of the per-cell statistic (lower is better)")
for task, _, _ in L.TASKS:
    cvm, cva = [], []
    for (t, a), s in D.items():
        if t != task or len(s) < 3:
            continue
        m = [np.median([e[L.ECH] for e in d["episodes"]]) for d in s.values()]
        v = [np.mean([e[L.ECH] for e in d["episodes"]]) for d in s.values()]
        cvm.append(np.std(m, ddof=1) / np.mean(m)); cva.append(np.std(v, ddof=1) / np.mean(v))
    print(f"  {task:7s} median {np.mean(cvm):.3f}  mean {np.mean(cva):.3f}  "
          f"-> {'mean' if np.mean(cva) < np.mean(cvm) else 'median'} is quieter")

print("\nENERGY OUTLIERS (|z|>3 on log10 within task; these are often the panel denominator)")
n_out = 0
for task, _, _ in L.TASKS:
    e = np.log10([v["energy"] for v in T[task].values()]); z = (e - e.mean()) / e.std()
    for (a, v), zi in zip(T[task].items(), z):
        if abs(zi) > 3:
            print(f"  {task}/{a}: z={zi:+.2f} n_seeds={v['n_seeds']} E={v['energy']:.3g}"); n_out += 1
print(f"  {n_out} outliers")

print("\nCAVEATS OUTSTANDING" if flag else "\nNo caveats outstanding.")
for f in flag:
    print("  !", f)
