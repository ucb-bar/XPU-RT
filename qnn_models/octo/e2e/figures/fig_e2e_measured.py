#!/usr/bin/env python3
"""Every point MEASURED. No propagation, no duplicates, no empty cells.

The (period, window) heatmap can never be fully filled by measurement, for two reasons
that are properties of the schedule space rather than gaps in the sweep:

  25 cells are PROVEN INFEASIBLE -- no schedule exists, so there is nothing to run.
  27 cells are EXACT DUPLICATES  -- CP-SAT returns the same schedule for every window
                                    once the deadline stops binding, and job_grid.sh
                                    passes the simulator only --latency-ms and
                                    --issue-period-ms. Verified: all 27 are identical to
                                    their twin at the 0.1 ms precision actually passed.
                                    Running them would re-execute the same command.

So this figure drops the REQUESTED (period, window) coordinates and plots what the
scheduler ACHIEVED: cadence against observation age. In that space the 71 solved cells
collapse to 44 distinct operating points, every one of them simulated at 20 seeds x 24
episodes. Nothing here is copied or inferred.

The cost of this view is that it is a scatter, not a grid -- the achieved points are not
on a lattice. The gain is that a reader cannot mistake a duplicate for evidence, which is
the failure mode the hatched heatmap has to warn about in prose.
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "fig_e2e_measured.png"
FLOOR = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}
TASKS = [("egg", "eggplant in basket", "widowx, 40 ms grid"),
         ("spoon", "spoon on towel", "widowx, 40 ms grid"),
         ("coke", "pick coke can", "google, 333 ms grid"),
         ("drawer", "close drawer", "google, 333 ms grid")]

succ = json.load(open(DATA / "grid_e2e_success.json"))
warm = json.load(open(DATA / "grid_warm.json"))["cells"]

fig, axes = plt.subplots(1, 4, figsize=(21.5, 6.4), dpi=150,
                         gridspec_kw={"wspace": 0.16})
for ax, (task, lab, emb) in zip(axes, TASKS):
    xs, ys, cs = [], [], []
    for arm, v in succ[task].items():
        if v.get("source") != "measured":
            continue                      # copies are not evidence
        c = warm.get(arm[1:] if arm.startswith("g") else arm, {})
        if c.get("lat_med") is None:
            continue
        xs.append(c["cadence"]); ys.append(c["lat_med"]); cs.append(v["rate"])
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)
    assert len(xs) == 44, f"{task}: expected 44 measured points, got {len(xs)}"

    sc = ax.scatter(xs, ys, c=cs, s=260, cmap="Blues", edgecolor="#16161C",
                    linewidth=1.1, zorder=3,
                    vmin=cs.min(), vmax=cs.max())
    for x, y, v in zip(xs, ys, cs):
        ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=6.4, zorder=4,
                color="white" if v > cs.min() + 0.55 * (cs.max() - cs.min()) else "#16161C")
    ax.set_xlabel("cadence ACHIEVED (ms)", fontsize=9)
    ax.set_title(f"{lab}   [{emb}]\n{len(xs)} measured operating points\n"
                 f"range {cs.max()-cs.min():.1f} pts = "
                 f"{(cs.max()-cs.min())/FLOOR[task]:.1f} x noise band (±{FLOOR[task]:.2f})",
                 fontsize=9.0, linespacing=1.35)
    ax.grid(True, alpha=0.25, zorder=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.015)
    cb.ax.tick_params(labelsize=7)
    if task == TASKS[-1][0]:
        cb.set_label("success rate (%)", fontsize=7.6)
axes[0].set_ylabel("observation age ACHIEVED (ms)", fontsize=9)

fig.suptitle("EVERY POINT MEASURED — 44 distinct operating points × 4 tasks × 20 seeds "
             "× 24 episodes = 84,480 episodes", fontsize=12.6, y=0.980)
fig.text(0.5, 0.128,
         "Plotted in ACHIEVED coordinates, not requested (period, window). That plane "
         "cannot be filled by measurement: 25 of its cells are PROVEN INFEASIBLE (no "
         "schedule exists) and 27 are EXACT DUPLICATES —\n"
         "CP-SAT returns the same schedule once the deadline stops binding, and the "
         "simulator is given only latency and cadence, so those cells re-run an identical "
         "command. Here each point appears once, measured.\n"
         "Cadence runs left to right and age bottom to top, so the useful corner is the "
         "bottom left. Colour is per-panel: the tasks differ in baseline difficulty, so "
         "compare through the RANGE in each title, in that task's own noise band.",
         ha="center", va="top", fontsize=8.0, style="italic", color="#555")
fig.subplots_adjust(top=0.800, bottom=0.205)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
for task, _, _ in TASKS:
    r = [v["rate"] for v in succ[task].values() if v.get("source") == "measured"]
    print(f"  {task:6s} {len(r)} measured points  {min(r):.1f}-{max(r):.1f}%  "
          f"range {max(r)-min(r):.1f}")
