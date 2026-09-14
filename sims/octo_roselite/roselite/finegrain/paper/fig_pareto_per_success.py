#!/usr/bin/env python3
"""Actuator energy PER SUCCESSFUL TRAJECTORY vs success rate.

This asks a different question from fig_pareto_energy.py. That one plots the cost of
running an episode; this one plots the cost of getting a task DONE -- every failed
attempt's energy is charged to the successes it did not produce:

    E_per_success = (sum of tau^2 energy over ALL 24 episodes) / (number of successes)

TWO PROPERTIES OF THIS METRIC THAT MUST BE STATED, because both are easy to misread:

1. IT IS UNDEFINED AT ZERO SUCCESSES. Four cells have none (egg/spoon x fp32_555,
   cpu685). They are NOT dropped -- dropping the worst arms would flatter the result.
   They are drawn as up-arrows at the axis top, meaning "burned energy, produced nothing,
   cost per success is infinite". That is the honest reading and it is arguably the
   strongest single fact on the chart.

2. THE AXES ARE NOT INDEPENDENT. Success rate appears in the denominator of the y-value,
   so the two are anti-correlated BY CONSTRUCTION -- part of the leftward-and-upward
   sweep is arithmetic, not physics. The chart is still meaningful (it is a cost-per-
   useful-outcome, which is what a deployment cares about) but it is NOT independent
   evidence on top of the success-vs-energy chart. Read them together, not as two
   corroborating results.

y is log-scaled: the spread runs from ~100 to infinity, and on a linear axis every
schedule that works collapses onto the floor.

Energy model, caveats and provenance: see ENERGY_RESULTS.md. 24 episodes, ONE seed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "fig_pareto_per_success.png"
CACHE = json.load(open(HERE / "energy_cache.json"))

ARMS = [("lat0", "ideal 0 ms", "#7f8c8d"),
        ("pipe110fix", "pipe110", "#0e6655"), ("p105w300", "p105/300", "#148f77"),
        ("p130w275", "p130/275", "#17a589"), ("p150w300", "p150/300", "#45b39d"),
        ("pipe200fix", "pipe200", "#d68910"), ("serial283", "serial283", "#e67e22"),
        ("fp32_555", "fp32 555", "#c0392b"), ("cpu685", "cpu 685", "#7b241c")]
TASKS = [("egg", "eggplant  ·  GRASP"), ("spoon", "spoon  ·  GRASP"),
         ("coke", "coke can  ·  GRASP"), ("drawer", "close drawer  ·  PUSH")]
BAND = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}


def frontier(pts):
    """Minimise energy-per-success, maximise success rate."""
    keep = []
    for i, (s, e, *_r) in enumerate(pts):
        if np.isinf(e):
            continue
        if not any((s2 >= s and e2 <= e and (s2 > s or e2 < e))
                   for j, (s2, e2, *_x) in enumerate(pts) if j != i and not np.isinf(e2)):
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


fig, axes = plt.subplots(1, 4, figsize=(20.0, 6.6), dpi=150,
                         gridspec_kw={"wspace": 0.24})
assert len(axes) == len(TASKS)
for ax, (task, lab) in zip(axes, TASKS):
    pts = []
    for arm, alab, col in ARMS:
        c = CACHE.get(f"{task}_{arm}")
        if c is None:
            continue
        tot = float(np.sum(c["t2"]))
        ns = c["sr"] / 100.0 * c["nep"]
        pts.append((c["sr"], tot / ns if ns else np.inf, alab, col, tot))
    fin = [p[1] for p in pts if not np.isinf(p[1])]
    top = max(fin) * 2.6
    fr = frontier(pts)

    fx = [pts[i][0] for i in fr]; fy = [pts[i][1] for i in fr]
    ax.step(fx, fy, where="pre", color="#16161C", lw=1.6, alpha=0.55, zorder=2)
    for i, (s, e, alab, col, tot) in enumerate(pts):
        if np.isinf(e):                       # zero successes: cost is undefined
            ax.annotate("", xy=(s, top), xytext=(s, top / 2.1),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6,
                                        mutation_scale=20), zorder=4)
            ax.scatter(s, top / 2.1, s=190, c=col, edgecolor="#16161C", lw=1.4, zorder=5)
            ax.annotate(f"{alab}\nno successes", (s, top / 2.1),
                        textcoords="offset points", xytext=(9, -4), ha="left",
                        va="top", fontsize=7.0, color=col, fontweight="bold", zorder=6)
            continue
        on = i in fr
        ax.scatter(s, e, s=300 if on else 170, c=col, zorder=4,
                   edgecolor="#16161C" if on else "#9a9aa4",
                   linewidth=2.4 if on else 1.0, alpha=1.0 if on else 0.72)
        ax.annotate(alab, (s, e), textcoords="offset points",
                    xytext=(0, 13 if i % 2 == 0 else -19), ha="center", fontsize=7.4,
                    fontweight="bold" if on else "normal",
                    color="#16161C" if on else "#6B6B78", zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(min(fin) * 0.55, top * 1.25)
    ax.set_xlabel("success rate (%)", fontsize=9)
    ax.set_title(f"{lab}\n{len(fr)} of {len(pts)} arms on the frontier"
                 f"{'   ·   ' + str(sum(np.isinf(p[1]) for p in pts)) + ' at zero successes' if any(np.isinf(p[1]) for p in pts) else ''}",
                 fontsize=9.4)
    ax.grid(True, alpha=0.25, which="both", zorder=0)
    ax.errorbar(min(p[0] for p in pts), min(fin) * 0.72, xerr=BAND[task],
                color="#6B6B78", capsize=4, lw=1.4, zorder=3)
    ax.text(min(p[0] for p in pts), min(fin) * 0.62, f"±{BAND[task]:.2f}",
            ha="center", va="top", fontsize=6.6, color="#6B6B78")
axes[0].set_ylabel(r"energy per SUCCESS  $\int\sum\tau_i^2 dt$ / n$_{success}$  (log)",
                   fontsize=8.8)

fig.suptitle("Cost of getting the task DONE: actuator energy per successful trajectory  "
             "—  every failed attempt is charged to the successes it did not produce",
             fontsize=12.4, y=0.975)
fig.text(0.5, 0.118,
         "Arrows = arms with ZERO successes: they burned energy and produced nothing, so "
         "cost per success is infinite. They are shown rather than dropped — dropping the "
         "worst arms would flatter the result.\n"
         "CAUTION, the axes are NOT independent: success rate is the denominator of the "
         "y-value, so part of the up-and-left sweep is arithmetic rather than physics. "
         "This is a cost-per-useful-outcome view, not independent\n"
         "evidence on top of the success-vs-energy chart — read the two together. "
         "PROVISIONAL: 24 episodes, ONE seed; the success axis carries at least the "
         "marked noise band. A 1,760-run sweep is in flight to replace it.",
         ha="center", va="top", fontsize=7.8, style="italic", color="#555")
fig.subplots_adjust(top=0.855, bottom=0.215)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for task, _ in TASKS:
    rows = []
    for arm, alab, _ in ARMS:
        c = CACHE.get(f"{task}_{arm}")
        if not c:
            continue
        ns = c["sr"] / 100.0 * c["nep"]
        rows.append((alab, c["sr"], float(np.sum(c["t2"])) / ns if ns else float("inf")))
    best = min((r for r in rows if np.isfinite(r[2])), key=lambda r: r[2])
    inf = [r[0] for r in rows if not np.isfinite(r[2])]
    print(f"  {task:7s} cheapest per success: {best[0]} = {best[2]:.0f} "
          f"({best[1]:.1f}% sr)" + (f"   |   infinite: {', '.join(inf)}" if inf else ""))
