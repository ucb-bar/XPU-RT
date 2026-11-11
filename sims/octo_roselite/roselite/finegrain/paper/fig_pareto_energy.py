#!/usr/bin/env python3
"""Success vs actuator energy: which schedules are Pareto-optimal?

Both axes come from the SAME rollouts. That matters here more than usual: the harness is
not run-to-run deterministic (~20% of byte-identical invocations diverge), so pairing a
success rate from one run with an energy figure from another would not describe any
episode that actually happened. `trace_eval.py` emits both, so each point is a real pair.

ENERGY is the copper-loss proxy INT sum_i tau_i^2 dt, summed over joints and integrated
over the whole episode -- the only term here that sees a stall or a collision, since a
servo pushing into a rigid constraint has large tau and ~zero omega. Ratios between arms
are meaningful; absolute joules are not (no motor constant). Per-episode MEDIAN, not mean:
the distribution is heavy-tailed (one eggplant outlier sits near 27,000 against medians of
48-115) and a mean would track that single episode.

UNCONDITIONED on success, deliberately -- conditioning keeps only the episodes an arm
happened to win, which on eggplant cpu685 is 1 in 24, and hides the flailing that is the
whole effect.

The frontier is drawn per task because the two axes have different units and different
baselines per embodiment; a pooled frontier would just rank the tasks.

CAVEAT, printed on the figure: 24 episodes, ONE seed per cell. The measured 20-seed noise
bands are +/-3.33 (egg) / 2.29 (spoon) / 2.08 (coke) points on SUCCESS alone, so the
success axis of any single point here carries at least that much uncertainty, and no
frontier membership within a band's width should be treated as established. The 1760-run
energy sweep now in flight replaces this with 10 seeds.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "fig_pareto_energy.png"
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
    """Maximal success at minimal energy: keep points nothing else dominates."""
    keep = []
    for i, (e, s, *_r) in enumerate(pts):
        if not any((e2 <= e and s2 >= s and (e2 < e or s2 > s))
                   for j, (e2, s2, *_x) in enumerate(pts) if j != i):
            keep.append(i)
    return sorted(keep, key=lambda i: pts[i][0])


fig, axes = plt.subplots(1, 4, figsize=(20.0, 6.6), dpi=150,
                         gridspec_kw={"wspace": 0.22})
assert len(axes) == len(TASKS)
for ax, (task, lab) in zip(axes, TASKS):
    pts = []
    for arm, alab, col in ARMS:
        c = CACHE.get(f"{task}_{arm}")
        if c is None:
            continue
        pts.append((float(np.median(c["t2"])), c["sr"], alab, col))
    fr = frontier(pts)

    fx = [pts[i][0] for i in fr]; fy = [pts[i][1] for i in fr]
    ax.step(fx, fy, where="post", color="#16161C", lw=1.6, alpha=0.55, zorder=2)
    for i, (e, s, alab, col) in enumerate(pts):
        on = i in fr
        ax.scatter(e, s, s=300 if on else 170, c=col, zorder=4,
                   edgecolor="#16161C" if on else "#9a9aa4",
                   linewidth=2.4 if on else 1.0, alpha=1.0 if on else 0.72)
        # near-coincident arms overprint at a fixed offset; alternate above/below
        dy = 13 if (i % 2 == 0) else -19
        ax.annotate(alab, (e, s), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=7.4,
                    fontweight="bold" if on else "normal",
                    color="#16161C" if on else "#6B6B78", zorder=5)
    ax.axhspan(0, 0, color="none")
    # the success axis is uncertain by at least the measured noise band
    lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
    ax.set_ylim(lo - 0.22 * (hi - lo) - BAND[task], hi + 0.30 * (hi - lo) + BAND[task])
    ax.errorbar(pts[0][0], lo - 0.12 * (hi - lo), yerr=BAND[task], color="#6B6B78",
                capsize=4, lw=1.4, zorder=3)
    ax.text(pts[0][0], lo - 0.12 * (hi - lo) - BAND[task] - 0.4,
            f"±{BAND[task]:.2f} noise", ha="center", va="top", fontsize=6.6,
            color="#6B6B78")
    ax.set_xlabel(r"actuator energy  $\int\sum_i \tau_i^2\,dt$  (median episode)",
                  fontsize=8.4)
    ax.set_title(f"{lab}\n{len(fr)} of {len(pts)} arms on the frontier", fontsize=9.6)
    ax.grid(True, alpha=0.25, zorder=0)
axes[0].set_ylabel("success rate (%)", fontsize=9)

fig.suptitle("Pareto frontier: success vs actuator energy, both from the SAME rollouts  "
             "—  bold = non-dominated, arrow of preference is up and left",
             fontsize=12.4, y=0.975)
fig.text(0.5, 0.118,
         "Energy is the copper-loss proxy INT sum tau^2 dt, per-episode MEDIAN, "
         "UNCONDITIONED on success; ratios between arms are meaningful, absolute joules "
         "are not. Both axes come from the same episodes, which matters because the\n"
         "harness is nondeterministic — pairing success from one run with energy from "
         "another would describe no episode that happened. PROVISIONAL: 24 episodes, ONE "
         "seed per point; the success axis carries at least the marked noise band, so "
         "frontier membership within a band's width is not established.\n"
         "A 1,760-run sweep (44 arms × 4 tasks × 10 seeds) is in flight to replace this.",
         ha="center", va="top", fontsize=7.8, style="italic", color="#555")
fig.subplots_adjust(top=0.860, bottom=0.215)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for task, _ in TASKS:
    pts = [(float(np.median(CACHE[f"{task}_{a}"]["t2"])), CACHE[f"{task}_{a}"]["sr"], al)
           for a, al, _ in ARMS if f"{task}_{a}" in CACHE]
    fr = frontier(pts)
    print(f"  {task:7s} frontier: " + ", ".join(f"{pts[i][2]}({pts[i][1]:.1f}%)" for i in fr))
