#!/usr/bin/env python3
"""Two Pareto views on the corrected metric: success vs energy, and mission time vs energy.

Both axes of every point come from the SAME rollouts, which matters because the harness is
nondeterministic (~20% of byte-identical invocations diverge) -- pairing a success rate
from one run with an energy figure from another would describe no episode that happened.

ENERGY is `t2_drive_arm_sus` from trace_eval2.py: the PD drive torque PhysX actually
applies, reconstructed per substep and validated against the articulation's equation of
motion to 0.024 N.m. The previous Pareto figures used get_qf(), which is gravity +
Coriolis and blind to contact and stall (ENERGY_AUDIT.md) -- it compressed spoon's 45x
effect into 2.6x, so those frontiers were drawn on a proxy that could not separate arms.

ROW 1  success (up is better) vs energy (left is better).
ROW 2  mission time (down is better) vs energy (left is better) -- and note this row is
       CONDITIONED on success, because a failure has no completion time. That biases slow
       arms downward on the time axis: they only finish what they can still do. Arms with
       fewer than 20 contributing episodes are drawn hollow and labelled.

Energy is UNCONDITIONED in both rows: conditioning would keep only the episodes an arm
happened to win (1 in 60 for eggplant cpu685) and hide the flailing that is the effect.

n=10 seeds x 24 episodes per point; error bars are bootstrap 95% CIs over seeds.
"""
from __future__ import annotations
import json, glob, re, collections
from pathlib import Path

import numpy as np
np.random.seed(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SRC = HERE.parent / "traces_torque3"
OUT = HERE / "fig_pareto3.png"

ARMS = [("lat0", "ideal", "#7f8c8d"), ("pipe110fix", "pipe110", "#0e6655"),
        ("p105w300", "p105/300", "#148f77"), ("p130w275", "p130/275", "#17a589"),
        ("p150w300", "p150/300", "#45b39d"), ("pipe200fix", "pipe200", "#d68910"),
        ("serial283", "serial", "#e67e22"), ("fp32_555", "fp32", "#c0392b"),
        ("cpu685", "cpu685", "#7b241c")]
TASKS = [("egg", "eggplant"), ("spoon", "spoon"), ("coke", "coke can"), ("drawer", "drawer")]

D = collections.defaultdict(dict)
for f in glob.glob(str(SRC / "*" / "energy2.json")):
    k = Path(f).parent.name
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", k)
    D[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))


def ci(v, B=2000, seed=0):
    if len(v) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed); a = np.asarray(v, float)
    return tuple(np.percentile([r.choice(a, a.size, True).mean() for _ in range(B)], [2.5, 97.5]))


def frontier(pts, maximise_y):
    """pts = [(x_energy, y, ...)]. Lower x always better; y direction per row."""
    keep = []
    for i, (x, y, *_r) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        dom = False
        for j, (x2, y2, *_z) in enumerate(pts):
            if j == i or not np.isfinite(x2) or not np.isfinite(y2):
                continue
            better_y = (y2 >= y) if maximise_y else (y2 <= y)
            strict = (x2 < x) or (y2 > y if maximise_y else y2 < y)
            if x2 <= x and better_y and strict:
                dom = True; break
        if not dom:
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


fig, axes = plt.subplots(2, len(TASKS), figsize=(20.0, 10.4), dpi=140,
                         gridspec_kw={"hspace": 0.40, "wspace": 0.24})
for c, (task, tlab) in enumerate(TASKS):
    arms = [a for a, _, _ in ARMS if (task, a) in D]
    en = {a: [np.median([e["t2_drive_arm_sus"] for e in d["episodes"]])
              for d in D[(task, a)].values()] for a in arms}
    base = np.mean(en["lat0"])
    sr = {a: [100 * d["n_success"] / d["n_episodes"] for d in D[(task, a)].values()] for a in arms}
    mt, nsucc = {}, {}
    for a in arms:
        v = [e["duration_s"] for d in D[(task, a)].values() for e in d["episodes"] if e["success"]]
        mt[a] = v; nsucc[a] = len(v)

    for row, (ymap, ylab, maximise) in enumerate(
            [(sr, "success rate (%)", True),
             (mt, "mission time on successes (s)", False)]):
        ax = axes[row, c]
        pts = []
        for a, alab, col in ARMS:
            if a not in arms:
                continue
            yv = ymap[a]
            pts.append((np.mean(en[a]) / base, np.mean(yv) if yv else np.nan, alab, col, a))
        fr = frontier(pts, maximise)
        fx = [pts[i][0] for i in fr]; fy = [pts[i][1] for i in fr]
        ax.step(fx, fy, where="post" if maximise else "pre",
                color="#16161C", lw=1.5, alpha=0.5, zorder=2)
        for i, (xv, yv, alab, col, a) in enumerate(pts):
            if not np.isfinite(yv):
                continue
            thin = row == 1 and nsucc[a] < 20
            on = i in fr
            ax.errorbar(xv, yv,
                        xerr=[[xv - ci(en[a])[0] / base], [ci(en[a])[1] / base - xv]],
                        yerr=[[yv - ci(ymap[a])[0]], [ci(ymap[a])[1] - yv]] if len(ymap[a]) > 1 else None,
                        fmt="none", ecolor="#9a9aa4", lw=1.0, capsize=2, zorder=3)
            ax.scatter(xv, yv, s=300 if on else 170,
                       facecolor="white" if thin else col, edgecolor=col if thin else
                       ("#16161C" if on else "#9a9aa4"),
                       linewidth=2.4 if (on or thin) else 1.0, zorder=4)
            ax.annotate(alab + (f"\nn={nsucc[a]}" if thin else ""), (xv, yv),
                        textcoords="offset points", xytext=(0, 13 if i % 2 == 0 else -20),
                        ha="center", fontsize=7.0,
                        fontweight="bold" if on else "normal",
                        color="#16161C" if on else "#6B6B78", zorder=5)
        if max(p[0] for p in pts) / max(min(p[0] for p in pts), 1e-9) > 8:
            ax.set_xscale("log")
        ax.axvline(1.0, color="#16161C", lw=1.0, ls="--", alpha=0.45, zorder=1)
        ax.set_xlabel("actuator energy / ideal arm", fontsize=8.6)
        ax.set_title(f"{tlab}  ·  {len(fr)} of {len(pts)} on the frontier", fontsize=9.6)
        ax.grid(True, alpha=0.25, zorder=0)
        if c == 0:
            ax.set_ylabel(ylab, fontsize=8.8)

fig.suptitle("Pareto: success vs energy (top) and mission time vs energy (bottom)  —  "
             "corrected drive-torque metric, 10 seeds, preference is LEFT and "
             "(top) UP / (bottom) DOWN", fontsize=12.6, y=0.972)
fig.subplots_adjust(top=0.900, bottom=0.145)
fig.text(0.5, 0.088,
         "Both axes of a point come from the SAME rollouts — the harness is "
         "nondeterministic, so mixing runs would describe no episode that happened. Energy "
         "is the PD drive torque PhysX applies (validated to 0.024 N·m against the "
         "equation of motion), UNCONDITIONED on success.\n"
         "The previous Pareto figures used get_qf(), which is gravity + Coriolis and blind "
         "to contact — it compressed spoon's 45× spread into 2.6×. Mission time (bottom "
         "row only) is CONDITIONED on success, which biases slow arms downward; points "
         "with n<20 contributing episodes are hollow and labelled.\n"
         "Error bars are bootstrap 95% CIs over 10 seeds. Ratios are meaningful; absolute "
         "energy is not calibrated to joules.",
         ha="center", va="top", fontsize=7.9, style="italic", color="#555")
fig.savefig(OUT, dpi=140)
print(f"[ok] {OUT}")
for task, _ in TASKS:
    arms = [a for a, _, _ in ARMS if (task, a) in D]
    en = {a: np.mean([np.median([e["t2_drive_arm_sus"] for e in d["episodes"]])
                      for d in D[(task, a)].values()]) for a in arms}
    b = en["lat0"]
    sr = {a: np.mean([100 * d["n_success"] / d["n_episodes"] for d in D[(task, a)].values()]) for a in arms}
    p = [(en[a] / b, sr[a], a) for a in arms]
    fr = frontier(p, True)
    print(f"  {task:7s} success-vs-energy frontier: " +
          ", ".join(f"{p[i][2]}({p[i][1]:.0f}%,{p[i][0]:.2f}x)" for i in fr))
