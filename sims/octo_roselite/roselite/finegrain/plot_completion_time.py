#!/usr/bin/env python3
"""Task completion time by schedule, over SUCCESSFUL episodes only.

Source: the full fine-grain sweep (g5fine/runs), 240 episodes per arm per task.
An episode terminates the tick the evaluator reports success, so `sim_ms` on a
successful episode IS the completion time. Failures have no completion time --
they run to the horizon -- so they are excluded, not counted as the horizon.

THE SELECTION EFFECT, stated up front: conditioning on success means each arm's
distribution is drawn from a different subpopulation. A high-latency arm only
succeeds on the easy configs, so its completion time is biased DOWNWARD relative
to a like-for-like comparison. Where an arm has few successes the estimate is
both noisy and selected; n is printed on every box for that reason.
"""
from __future__ import annotations
import json, glob, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS = HERE / "g5fine" / "runs"
OUT = HERE / "completion_time.png"

ARMS = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
ALAB = {"lat0": "ideal\n0 ms", "pipe110": "PIPE\n110", "pipe200": "PIPE\n200",
        "serial283": "SERIAL\n283", "fp32_555": "CPU fp32\n555", "cpu685": "CPU int8\n685"}
CAT = {"lat0": "#7f8c8d", "pipe110": "#16a085", "pipe200": "#1a9e8f",
       "serial283": "#e67e22", "fp32_555": "#c0392b", "cpu685": "#96281b"}
TASKS = [("egg", "eggplant in basket"), ("spoon", "spoon on towel"),
         ("coke", "pick coke can"), ("drawer", "close drawer")]


def load():
    d = collections.defaultdict(list)
    for f in glob.glob(str(RUNS / "*" / "*" / "summary.json")):
        try: s = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name
        if "_rng" not in name: continue
        head, _ = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        for e in s.get("episodes", []):
            if e.get("success"):
                d[(task, arm)].append(e.get("sim_ms", np.nan) / 1000.0)
    return d


data = load()
present = [(k, lab) for k, lab in TASKS if any((k, a) in data for a in ARMS)]
fig, axes = plt.subplots(1, len(present), figsize=(4.6*len(present), 5.4), dpi=140,
                         sharey=False)
axes = np.atleast_1d(axes)
for ax, (task, lab) in zip(axes, present):
    series, cols, ticks = [], [], []
    for a in ARMS:
        v = [x for x in data.get((task, a), []) if np.isfinite(x)]
        series.append(v if v else [np.nan]); cols.append(CAT[a]); ticks.append(ALAB[a])
    bp = ax.boxplot(series, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color="black", lw=1.7))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("black")
    for i, v in enumerate(series):
        n = 0 if (len(v) == 1 and not np.isfinite(v[0])) else len(v)
        ax.scatter(np.random.normal(i+1, 0.055, n), v[:n], s=7, c="black",
                   alpha=0.30, zorder=4)
        ax.text(i+1, ax.get_ylim()[1], f"n={n}", ha="center", va="top", fontsize=7.5,
                color="#333" if n >= 20 else "#c0392b",
                fontweight="normal" if n >= 20 else "bold")
    ax.set_xticks(range(1, len(ARMS)+1)); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_title(lab, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel("task completion time (s of sim time)\nSUCCESSFUL episodes only")
fig.suptitle("Time to complete the task, by schedule — fine-grain sweep, successful episodes only "
             "(240 episodes/arm attempted)", fontsize=13, y=1.0)
fig.text(0.5, -0.045,
         "Conditioned on success: a high-latency arm only succeeds on the configs it can still do, so its "
         "completion time is biased DOWNWARD relative to a like-for-like comparison.\n"
         "Counts in red (n<20) are too few to read as a distribution. Failures are excluded, not scored at "
         "the horizon.", ha="center", fontsize=8.3, style="italic", color="#555")
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}\n")
for task, lab in present:
    print(f"{lab}:")
    for a in ARMS:
        v = np.array([x for x in data.get((task, a), []) if np.isfinite(x)])
        if len(v) == 0: print(f"  {a:11s} n=0"); continue
        print(f"  {a:11s} n={len(v):3d}  median {np.median(v):5.1f}s  "
              f"mean {v.mean():5.1f}s  IQR [{np.percentile(v,25):.1f},{np.percentile(v,75):.1f}]")
