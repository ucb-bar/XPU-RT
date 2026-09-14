#!/usr/bin/env python3
"""B4 -- PARALLEL COORDINATES over the 44 operating points.

Five axes: requested period, requested window, achieved period, achieved age,
makespan. One polyline per operating point, coloured by achieved period (the
sequential channel). Every axis is independently scaled and its range printed at
the ends, because a parallel-coordinates plot with unlabelled axes is decoration.

What it is for: seeing that the achieved period tracks the request while the
achieved age does NOT track the window -- the solver spends the window on
pipelining, not on lateness.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
cells, _, _ = V.grid_warm()
rows = []
for k, c in cells.items():
    if "lat_med" in c:
        rows.append([c["p"], c["w"], c["cadence"], c["lat_med"], c["makespan"]])
X = np.array(rows)
# collapse to distinct achieved points
_, idx = np.unique(np.round(X[:, 2:4], 1), axis=0, return_index=True)
X = X[np.sort(idx)]
NAMES = ["period\nrequested", "window\nrequested", "period\nachieved",
         "age\nachieved", "makespan"]
UNITS = "ms"

lo, hi = X.min(0), X.max(0)
Z = (X - lo) / (hi - lo)

fig = plt.figure(figsize=(7.16, 2.75))
ax = fig.add_axes([0.045, 0.145, 0.90, 0.70])
for i in range(len(Z)):
    ax.plot(range(5), Z[i], color=V.c_period(X[i, 2]), lw=0.9, alpha=0.85, zorder=3)
for j in range(5):
    ax.axvline(j, color=V.AXIS, lw=0.9, zorder=2)
    ax.text(j, 1.055, NAMES[j], ha="center", va="bottom", fontsize=7.2, color=V.INK)
    ax.text(j, -0.045, f"{lo[j]:.0f}", ha="center", va="top", fontsize=6.4, color=V.MUTED)
    ax.text(j, 1.012, f"{hi[j]:.0f}", ha="center", va="bottom", fontsize=6.4, color=V.MUTED)
ax.set_xlim(-0.35, 4.35)
ax.set_ylim(-0.02, 1.02)
ax.axis("off")
ax.text(-0.34, -0.045, UNITS, ha="left", va="top", fontsize=6.4, color=V.MUTED)

sm = plt.cm.ScalarMappable(norm=V.norm_period(X[:, 2].min(), X[:, 2].max()),
                           cmap=V.CMAP_PERIOD)
cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.012)
cb.set_label("achieved release period (ms)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0)
cb.outline.set_visible(False)
fig.text(0.045, 0.965, f"{len(X)} distinct operating points  ·  PREDICTED from the "
         f"schedule, all five axes in ms", fontsize=8.0, color=V.INK, fontweight="bold")
V.save(fig, "cand_b4_parcoords", f"n={len(X)}")
