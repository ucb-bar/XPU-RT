#!/usr/bin/env python3
"""Schmoo of the schedule space: 12 release periods x 9 deadlines = 108 cells.

Built on the WARM CP-SAT grid (`grid_warm.json`, solver cpsat:warmbest). Supersedes the
greedy version (`fig_schmoo_greedy.py.bak`), which coloured cells by what `greedy`
achieves -- and greedy ignores `window_duration` entirely, so its 108 cells collapse to
only 12 distinct operating points, each repeated across all 9 windows. The warm CP-SAT
grid gives 39.

Colour   the observation age / cadence CP-SAT actually achieves in that cell. Only the
         71 SOLVED cells have a value; the rest are blank by construction, not missing
         data.
Glyph    what CP-SAT proves about the cell:
             OPTIMAL / FEASIBLE   a schedule exists
             INFEASIBLE           PROOF that none does
             UNKNOWN              undecided at this budget -- NOT a negative result
         The three-way distinction matters: an exit-code bisection that read UNKNOWN as
         INFEASIBLE produced a completely wrong frontier earlier in this work.
Box      the 44 distinct operating points being run in the wide E2E sweep.

Solved in TWO passes: 120 s per cell, then a DEEP pass at 900 s over the 21 cells the
first left UNKNOWN. That resolved 8 -- 5 solved at p=130, forming a latency ladder at
FIXED cadence 130 ms over 271-339 ms of age (the contrast the first sweep had least
power on), and 3 proven INFEASIBLE -- leaving 13 genuinely undecided at p=100-120.

CONSISTENCY CHECK. A larger window is a strictly looser deadline, so a cell solvable at
a tighter window cannot be infeasible at a looser one. The earlier COLD grid
(`grid_status.txt`) violates this at all 6 of its INFEASIBLE cells -- e.g. p=180 w=350
OPTIMAL but p=180 w=400 INFEASIBLE -- so those labels are artifacts, not proofs. This
warm grid has ZERO violations and its INFEASIBLE cells sit only at the tightest windows
(240-290), which is where infeasibility belongs. The script asserts this.

All timings are PREDICTED from the profiled durations, not board-measured. The one
anchor: g150_290 predicts age 281.3 ms vs 281.6 ms measured on the QRB5165 (-0.1%).
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "fig_schmoo.png"
PERIODS = [100, 110, 120, 130, 140, 150, 165, 180, 200, 220, 250, 283]
WINDOWS = [240, 260, 275, 290, 305, 320, 350, 400, 450]

warm = json.load(open(DATA / "grid_warm.json"))
cells = warm["cells"]

# the cells actually being run E2E, from the live joblists' arm names
running = set()
# arms.tsv sits beside this script in the repo, and under g5grid/ in the live
# sweep tree; accept either so the figure reproduces from a clean checkout.
arms = next((c for c in (DATA / "arms.tsv", HERE.parent / "g5grid" / "arms.tsv")
             if c.exists()), DATA / "arms.tsv")
if arms.exists():
    for ln in open(arms):
        m = re.match(r"g(\d+)_(\d+)\t", ln)
        if m:
            running.add((int(m.group(1)), int(m.group(2))))


def monotonicity_violations(get):
    """A cell solvable at a tighter window cannot be infeasible at a looser one."""
    bad = []
    for p in PERIODS:
        for i, w in enumerate(WINDOWS):
            if get(p, w) == "INFEASIBLE":
                for w2 in WINDOWS[:i]:
                    if get(p, w2) in ("OPTIMAL", "FEASIBLE"):
                        bad.append((p, w2, w))
    return bad


viol = monotonicity_violations(lambda p, w: cells.get(f"{p}_{w}", {}).get("status"))
assert not viol, f"warm grid is not monotone in window: {viol[:5]}"

lat = np.full((len(WINDOWS), len(PERIODS)), np.nan)
cad = np.full_like(lat, np.nan)
for i, w in enumerate(WINDOWS):
    for j, p in enumerate(PERIODS):
        c = cells.get(f"{p}_{w}", {})
        if c.get("lat_med") is not None:
            lat[i, j] = c["lat_med"]; cad[i, j] = c["cadence"]

MARK = {"INFEASIBLE": "✗", "OPTIMAL": "★", "FEASIBLE": "●", "UNKNOWN": "?"}
GCOL = {"INFEASIBLE": "#c0392b", "OPTIMAL": "#f1c40f", "FEASIBLE": "#ecf0f1",
        "UNKNOWN": "#95a5a6"}

fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.6), dpi=150,
                         gridspec_kw={"wspace": 0.14})
fig.subplots_adjust(top=0.86)
for ax, (M, lab, cm) in zip(axes, [(lat, "CP-SAT observation age (ms)", "viridis_r"),
                                   (cad, "CP-SAT achieved cadence (ms)", "plasma_r")]):
    ax.set_facecolor("#e8e8e8")          # unsolved cells read as absent, not as zero
    im = ax.imshow(M, cmap=cm, aspect="auto", origin="lower")
    for i, w in enumerate(WINDOWS):
        for j, p in enumerate(PERIODS):
            st = cells.get(f"{p}_{w}", {}).get("status", "")
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i - 0.17, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                        color="white" if v > np.nanpercentile(M, 55) else "black")
            ax.text(j, i + 0.27, MARK.get(st, ""), ha="center", va="center", fontsize=9,
                    color=GCOL.get(st, "#555"), fontweight="bold")
            if (p, w) in running:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False,
                                       edgecolor="#0b5394", lw=1.9, zorder=5))
    ax.set_xticks(range(len(PERIODS))); ax.set_xticklabels(PERIODS, fontsize=8)
    ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels(WINDOWS, fontsize=8)
    ax.set_xlabel("release period requested (ms)")
    ax.set_title(lab, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
axes[0].set_ylabel("deadline  window_duration (ms)")

h = [Patch(facecolor="none", edgecolor="none",
           label="★ OPTIMAL   ● FEASIBLE   ✗ INFEASIBLE (proof)   ? UNKNOWN (undecided)"),
     Patch(facecolor="none", edgecolor="#0b5394", lw=1.9,
           label=f"blue box = one of the {len(running)} operating points in the E2E sweep")]
# figure-level, above the axes: inside the panel it covered the w=450 row
fig.legend(handles=h, fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, 0.945),
           ncol=2, frameon=False, handlelength=1.6)

t = warm["tally"]
fig.suptitle("Schedule space: 108 cells, warm-started CP-SAT. Colour = achieved timing, "
             "glyph = proof status, box = simulated end-to-end.", fontsize=12.5, y=0.995)
fig.text(0.5, -0.13,
         f"CP-SAT outcomes: {t['OPTIMAL']} OPTIMAL, {t['FEASIBLE']} FEASIBLE, "
         f"{t['INFEASIBLE']} INFEASIBLE, {t['UNKNOWN']} UNKNOWN. "
         f"{warm['n_solved']} cells solve to a schedule; those collapse to "
         f"{warm['n_distinct_operating_points']} DISTINCT (cadence, age) operating points, "
         f"and all {warm['n_distinct_operating_points']} are being simulated end-to-end "
         "(3 tasks x 10 seeds = 1170 runs).\n"
         "SIMULATOR RESOLUTION LIMIT: the harness quantises latency onto its tick grid "
         "(40 ms widowx, 37.04 ms google), so some of the 39 are the SAME experiment by "
         "construction -- 39 design arms resolve to 33 distinct experiments on coke and "
         "28 on eggplant/spoon.\nThe redundant arms are exact replicates, so they add "
         "seeds rather than latency coverage (see g5grid/resolution_audit.py). The "
         "latency axis of this plane is finer than the simulator can represent; the "
         "cadence axis is not affected.\n"
         "UNKNOWN means the solver exhausted its budget and is NOT evidence of "
         "infeasibility. Grey cells have no schedule, so no timing to colour.\n"
         "Warm-starting is what makes this figure worth plotting: greedy ignores "
         "window_duration entirely, so its 108 cells collapse to just 12 distinct "
         "operating points -- one per period, repeated down every window.\n"
         "Timings are PREDICTED from profiled durations. Single board anchor: p150/w290 "
         "predicts 281.3 ms age vs 281.6 ms measured (-0.1%); that bounds the "
         "extrapolation near that cell only.",
         ha="center", fontsize=8.2, style="italic", color="#555")
fig.savefig(OUT, bbox_inches="tight")
print(f"[ok] {OUT}")
print(f"  tally={t}  solved={warm['n_solved']}  distinct={warm['n_distinct_operating_points']}"
      f"  running_cells={len(running)}  monotonicity_violations={len(viol)}")
