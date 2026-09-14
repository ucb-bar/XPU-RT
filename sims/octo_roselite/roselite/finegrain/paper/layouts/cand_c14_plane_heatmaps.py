#!/usr/bin/env python3
"""C14 -- the three metrics as PLANE HEATMAPS, all four workloads, no highlight.

The same 44 operating points C13 scatters, laid back onto the grid the scheduler
actually searched: release period across, deadline window down. C13 answers "is
there a trade-off"; this answers "where in the schedule space does the metric
live", which is the question the CP-SAT grid was built to ask.

No Pareto rings, no annotation of a best cell -- the surface is the message.

Colour is a MAGNITUDE in every row, so each panel is a sequential single hue on
its OWN scale: tasks differ in baseline difficulty and a shared scale would paint
that difficulty as schedule sensitivity. One hue per metric, so an energy panel
can never be mistaken for a success panel. Cells with no feasible schedule are
neutral grey -- absent data, never a zero on the ramp.

Energy is a ratio to the cheapest cell in the same panel; the absolute
denominator is printed so the ratio can be undone. Mission time is conditioned on
success and therefore biased DOWNWARD for the arms that fail most.

44 arms x 4 tasks x 10 seeds x 24 episodes = 42,240 episodes.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import vocab as V
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
T = L.table()
plane = L.arm_plane()
PERIODS = sorted({int(re.match(r"g(\d+)_", a).group(1)) for a in plane})
WINDOWS = sorted({int(re.match(r"g\d+_(\d+)$", a).group(1)) for a in plane})
ROWS = [("success", "success (%)", "Blues", "{:.0f}"),
        ("mission", "mission time (s)", "Purples", "{:.1f}"),
        ("energy", "energy / cheapest cell", "Oranges", "{:.2f}")]

fig, axes = plt.subplots(len(ROWS), len(L.TASKS), figsize=(7.16, 4.95), dpi=170)
fig.subplots_adjust(left=0.077, right=0.985, top=0.910, bottom=0.092,
                    wspace=0.44, hspace=0.30)

for c, (task, tlab, emb) in enumerate(L.TASKS):
    arms = T[task]
    for r, (metric, lab, cmap, fmt) in enumerate(ROWS):
        ax = axes[r, c]
        M = np.full((len(WINDOWS), len(PERIODS)), np.nan)
        for a, v in arms.items():
            p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
            M[WINDOWS.index(w), PERIODS.index(p)] = v[metric]
        denom = np.nanmin(M)
        if metric == "energy":
            M = M / denom
        lo, hi = np.nanmin(M), np.nanmax(M)
        ax.set_facecolor(V.ABSENT)                    # absent != zero
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=lo, vmax=hi)
        # NO in-cell numbers. At this panel width a cell is ~9.5 pt across and a
        # 4-character value needs ~9 pt, so the columns touch and the surface --
        # the only thing this figure is for -- disappears under the digits. The
        # numbered version of the same data is paper/fig_plane3_progress.png.
        ax.set_xticks(range(len(PERIODS)))
        ax.set_xticklabels(PERIODS, fontsize=4.8, rotation=90)
        ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels(WINDOWS, fontsize=5.0)
        if r == len(ROWS) - 1:
            ax.set_xlabel("release period (ms)", fontsize=6.0, labelpad=1.0)
        if c == 0:
            ax.set_ylabel(f"{lab}\n\ndeadline window (ms)", fontsize=6.0, labelpad=1.5)
        ax.set_title((f"{tlab}\n" if r == 0 else "")
                     + (f"{lo:.0f}–{hi:.0f}%" if metric == "success"
                        else f"{lo:.1f}–{hi:.1f} s" if metric == "mission"
                        else f"spread {hi:.2f}×  (1.00 = {denom:.3g})"),
                     fontsize=6.4, pad=2.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.020)
        cb.ax.tick_params(labelsize=4.6)

fig.suptitle("The schedule plane, all four workloads — 44 operating points × 10 seeds",
             fontsize=8.4, y=0.988)
fig.text(0.5, 0.005, "grey = no feasible schedule at that (period, window)  ·  "
                     "each panel on its own scale  ·  mission time is conditioned on success",
         ha="center", va="bottom", fontsize=5.8, color=V.MUTED)
V.save(fig, "cand_c14_plane_heatmaps")
