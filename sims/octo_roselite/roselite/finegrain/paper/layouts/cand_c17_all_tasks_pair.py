#!/usr/bin/env python3
"""C17 -- all four workloads: one overlay scatter, four mini planes beside it.

Left: success against energy, every workload overlaid. ENERGY IS NORMALISED PER
WORKLOAD to its own cheapest cell -- the raw integrals differ ~70x between
eggplant (4.3e5) and close_drawer (1.3e4), so on one absolute axis three of the
four series collapse to a vertical line. Each series starts at 1.0 and the axis
reads "how much more than this workload's cheapest schedule"; shapes become
comparable, magnitudes deliberately do not.

Right: the same 44 schedules per workload as four SMALL MULTIPLES of the plane,
drawn as CONTIGUOUS CELLS on the (period, window) grid -- the grid is categorical,
a request the scheduler was given, not a continuous quantity, so cells that abut
say "these are the choices" while scattered markers wrongly imply the space
between them was sampled. Four separate panels rather than one grid with
quartered cells: the quartered version fitted four scales into one grid but asked
the reader to segment every cell by eye before reading anything.

EACH PANEL KEEPS ITS OWN GRADIENT, in its own hue, over its own success range,
and prints that range on its own colourbar. A shared scale would flatten the two
google workloads, whose entire ranges (33-49%, 36-45%) are narrower than
eggplant's alone.

Palette validated all-pairs by palette_check.py: worst CVD separation dE 8.5
(deutan), worst normal-vision dE 16.8, all four inside the lightness band. Marker
shape repeats the workload in the overlay and the panel title repeats it on the
right, so neither half is colour-alone.

A STAR marks each workload's best schedule by success rate, on both halves: on
the overlay it sits at that schedule's (energy, success), on the mini plane at
its (period, window). Drawn white-filled with the workload's hue as the edge,
because the best cell is by construction the most saturated one and a hue-filled
star would disappear into it. Best is taken over MEASURED cells only.

DRAW ORDER is deliberate, not incidental: coke and drawer are laid down first and
the two widowx workloads over them. The google clouds are dense and nearly
vertical, and drawn last they bury the eggplant and spoon ladders that carry the
result.

Hatched cells are propagated, filled from their twin; a faint x or ? marks where
the schedule was infeasible or unsolved within the CP-SAT budget.
"""
from __future__ import annotations
import json, re, sys
import numpy as np
import matplotlib
matplotlib.rcParams["hatch.linewidth"] = 0.45
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import vocab as V
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
PAPER = V.HERE.parent
warm = json.load(open(PAPER / "grid_warm.json"))["cells"]
TWIN = {a: v["twin"] for a, v in
        json.load(open(PAPER / "grid_e2e_success.json"))["egg"].items()
        if v.get("source") == "propagated"}
T = L.table()
HUE = {"egg": "#1b7f5f", "spoon": "#2a78d6", "coke": "#c0392b", "drawer": "#7a3fa5"}
MARK = {"egg": "o", "spoon": "s", "coke": "^", "drawer": "D"}
SHORT = {"egg": "eggplant", "spoon": "spoon", "coke": "coke can", "drawer": "drawer"}
CMAP = {t: LinearSegmentedColormap.from_list(t, ["#f4f3ef", h]) for t, h in HUE.items()}

fig = plt.figure(figsize=(7.16, 3.30), dpi=170)
gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 0.79, 0.79], left=0.068,
                      right=0.972, top=0.862, bottom=0.128, wspace=0.52, hspace=0.52)
ax = fig.add_subplot(gs[:, 0])

# google first, widowx on top -- see docstring
ORDER = ["coke", "drawer", "spoon", "egg"]
BEST = {t: max(T[t], key=lambda k: T[t][k]["success"]) for t in ORDER}
for zi, task in enumerate(ORDER):
    arms = T[task]; ks = sorted(arms)
    e = np.array([arms[a]["energy"] for a in ks]); sv = np.array([arms[a]["success"] for a in ks])
    ax.scatter(e / e.min(), sv, s=15, marker=MARK[task], c=HUE[task],
               edgecolor="white", linewidth=0.30, zorder=4 + zi, label=SHORT[task])
    b = BEST[task]
    ax.scatter(arms[b]["energy"] / e.min(), arms[b]["success"], s=78, marker="*",
               c="white", edgecolor=HUE[task], linewidth=1.0, zorder=12 + zi)
ax.set_xlabel("energy / this workload's cheapest schedule", fontsize=6.6, labelpad=1.5)
ax.set_ylabel("success rate (%)", fontsize=6.8, labelpad=2.0)
ax.tick_params(labelsize=6.0)
ax.set_title("success vs energy, per-workload normalised", fontsize=7.2, pad=3.0)
ax.margins(0.10); ax.set_ylim(0, ax.get_ylim()[1])
V.tidy(ax, grid="both")
h, lb = ax.get_legend_handles_labels()
o = [lb.index(SHORT[t]) for t, _, _ in L.TASKS]        # egg, spoon, coke, drawer
ax.legend([h[i] for i in o], [lb[i] for i in o], fontsize=5.8, loc="lower left",
          handletextpad=0.25, borderpad=0.3, labelspacing=0.25, markerscale=0.95)

# ---- four mini planes, each on its own gradient ---------------------------
PERIODS = sorted({int(k.split("_")[0]) for k in warm})
WINDOWS = sorted({int(k.split("_")[1]) for k in warm})
for n, (task, _, _) in enumerate(L.TASKS):
    bx = fig.add_subplot(gs[n // 2, 1 + n % 2])
    arms = T[task]
    M = np.full((len(WINDOWS), len(PERIODS)), np.nan)
    PR = np.zeros_like(M, dtype=bool)
    for a, v in arms.items():
        p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
        M[WINDOWS.index(w), PERIODS.index(p)] = v["success"]
    lo, hi = float(np.nanmin(M)), float(np.nanmax(M))        # MEASURED cells only
    for a, twin in TWIN.items():
        p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
        if twin in arms:
            i, j = WINDOWS.index(w), PERIODS.index(p)
            M[i, j] = arms[twin]["success"]; PR[i, j] = True
    bx.set_facecolor(V.ABSENT)
    im = bx.imshow(M, cmap=CMAP[task], vmin=lo, vmax=hi, origin="lower", aspect="auto")
    for i in range(len(WINDOWS)):
        for j in range(len(PERIODS)):
            if PR[i, j]:
                bx.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                       linewidth=0.0, edgecolor="#4a4a4a"))
            elif not np.isfinite(M[i, j]):
                st = warm.get(f"{PERIODS[j]}_{WINDOWS[i]}", {}).get("status", "")
                bx.text(j, i, {"INFEASIBLE": "✗", "UNKNOWN": "?"}.get(st, ""),
                        ha="center", va="center", fontsize=3.4, color="#a8a7a1")
    # Every other tick: 12 periods and 9 windows will not fit a panel this size.
    bx.set_xticks(range(0, len(PERIODS), 2))
    bx.set_xticklabels([PERIODS[i] for i in range(0, len(PERIODS), 2)], fontsize=4.6, rotation=90)
    bx.set_yticks(range(0, len(WINDOWS), 2))
    bx.set_yticklabels([WINDOWS[i] for i in range(0, len(WINDOWS), 2)], fontsize=4.8)
    bx.tick_params(pad=1.0, length=1.6)
    bx.set_title(f"{SHORT[task]}   {lo:.0f}–{hi:.0f}%", fontsize=6.4, pad=2.0,
                 color=HUE[task], fontweight="bold")
    V.tidy(bx, grid="")
    if n // 2 == 1:
        bx.set_xlabel("period (ms)", fontsize=5.8, labelpad=0.8)
    if n % 2 == 0:
        bx.set_ylabel("window (ms)", fontsize=5.8, labelpad=1.2)
    b = BEST[task]
    bp, bw = (int(z) for z in re.match(r"g(\d+)_(\d+)$", b).groups())
    bx.scatter(PERIODS.index(bp), WINDOWS.index(bw), s=130, marker="*", c="white",
               edgecolor=HUE[task], linewidth=1.1, zorder=8)
    cb = fig.colorbar(im, ax=bx, fraction=0.048, pad=0.03)
    cb.set_ticks([lo, hi]); cb.set_ticklabels([f"{lo:.0f}", f"{hi:.0f}"])
    cb.ax.tick_params(labelsize=4.8, length=1.4)

fig.text(0.735, 0.962, "SUCCESS RATE on the schedule plane, per workload",
         ha="center", va="top", fontsize=7.4, fontweight="bold", color=V.INK)
fig.text(0.5, 0.006, "★ = that workload's best schedule by success  ·  each workload on its OWN "
                     "gradient  ·  hatched = propagated  ·  ✗ infeasible   ? unsolved",
         ha="center", va="bottom", fontsize=5.8, color=V.MUTED)
V.save(fig, "cand_c17_all_tasks_pair")
