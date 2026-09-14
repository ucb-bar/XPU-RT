#!/usr/bin/env python3
"""In-flight view of the 44-arm plane: success, mission time, actuator energy.

PROGRESS FIGURE, not a result. The sweep is seed-major, so all 176 (task, arm)
pairs are already present -- what is missing is seeds, i.e. width of the error
bar, not coverage of the plane. Cells carrying fewer than 3 seeds are hatched.

Layout deliberately matches fig_e2e_heatmap.py: the same period x window grid,
so the success row can be read directly against the earlier 20-seed surface.

Colour is a magnitude in every row, so every panel is a SEQUENTIAL single hue,
light -> dark, with its OWN scale -- tasks differ in baseline difficulty and a
shared scale would paint that difficulty as schedule sensitivity. One hue per
metric so a reader never mistakes an energy panel for a success panel.

Energy is shown as a RATIO to the cheapest arm in the same panel. There is no
ideal (0 ms) arm inside the plane to normalise against, and the absolute
integral is not calibrated to joules (ENERGY_AUDIT.md section 4). The absolute
denominator is printed in each panel title so the ratio can be undone.
"""
from __future__ import annotations
import re
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 0.6
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import plane3_lib as L

HERE = Path(__file__).parent
OUT = HERE / "fig_plane3_progress.png"
MIN_SEEDS = 3                      # below this the cell is drawn as provisional

# PROPAGATED CELLS. 27 of the grid's 71 solved cells return the SAME CP-SAT
# schedule as a cell that was measured -- past ~220 ms of release period the
# deadline window stops binding, so widening it changes nothing. Checked at the
# bit level, 26 of the 27 are identical to their twin in (latency, cadence) at
# full float precision, and the one that is not (g150_400, 0.042 ms) becomes
# identical under the 0.1 ms formatting the job scripts apply. trace_eval2.py
# receives nothing but those two numbers, so simulating them separately would
# re-sample harness nondeterminism under 27 new labels, not measure new points.
# They are filled from their twin and HATCHED: a copy of evidence, never new
# evidence. Hatch rather than a dimmed fill, because the fill is a value on the
# scale and dimming it would read as a lower value.
# Reasoning in g5grid/PROPAGATED_CELLS.md.
TWIN = {a: v["twin"] for a, v in
        json.load(open(HERE / "grid_e2e_success.json"))["egg"].items()
        if v.get("source") == "propagated"}

T = L.table()
plane = L.arm_plane()
# Axes are the FULL CP-SAT search grid, not just the periods and windows that
# happen to carry a measured arm. Deriving them from the measured set alone drops
# the window-240 row, and with it three propagated cells (g220_240, g250_240,
# g283_240) that do have values. The grid is the search space; the figure should
# show all of it, including the parts where nothing was feasible.
_g = json.load(open(HERE / "grid_warm.json"))["cells"]
PERIODS = sorted({int(k.split("_")[0]) for k in _g})
WINDOWS = sorted({int(k.split("_")[1]) for k in _g})

ROWS = [("success", "success rate (%)", "Blues", "{:.0f}"),
        ("mission", "completion time (s), successes only", "Purples", "{:.1f}"),
        ("energy", "actuator energy / cheapest arm in panel", "Oranges", "{:.2f}")]

fig, axes = plt.subplots(len(ROWS), len(L.TASKS), figsize=(21.0, 13.2), dpi=145,
                         gridspec_kw={"hspace": 0.30, "wspace": 0.20})
assert axes.shape == (len(ROWS), len(L.TASKS)), axes.shape

n_prov = 0
for c, (task, lab, emb) in enumerate(L.TASKS):
    arms = T[task]
    for r, (metric, cblab, cmap, fmt) in enumerate(ROWS):
        ax = axes[r, c]
        M = np.full((len(WINDOWS), len(PERIODS)), np.nan)
        PROV = np.zeros_like(M, dtype=bool)
        COPY = np.zeros_like(M, dtype=bool)
        for arm, v in arms.items():
            p, w = (int(x) for x in re.match(r"g(\d+)_(\d+)$", arm).groups())
            i, j = WINDOWS.index(w), PERIODS.index(p)
            M[i, j] = v[metric]
            PROV[i, j] = v["n_seeds"] < MIN_SEEDS
        meas_vals = M[np.isfinite(M)].copy()         # MEASURED cells only
        for arm, twin in TWIN.items():
            p, w = (int(x) for x in re.match(r"g(\d+)_(\d+)$", arm).groups())
            if twin in arms and p in PERIODS and w in WINDOWS:
                i, j = WINDOWS.index(w), PERIODS.index(p)
                M[i, j] = arms[twin][metric]
                COPY[i, j] = True
        if metric == "energy":                       # normalise inside the panel
            denom = np.nanmin(meas_vals)
            M = M / denom
            meas_vals = meas_vals / denom
        # The colour scale and the printed range are computed on MEASURED cells
        # only, so propagation can never move either. Asserted, not intended.
        lo, hi = float(meas_vals.min()), float(meas_vals.max())
        assert np.nanmin(M) >= lo - 1e-9 and np.nanmax(M) <= hi + 1e-9, \
            f"{task}/{metric}: a propagated cell falls outside the measured range"
        ax.set_facecolor("#e4e4e4")                  # absent != zero
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=lo, vmax=hi)
        for i in range(len(WINDOWS)):
            for j in range(len(PERIODS)):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "·", ha="center", va="center", fontsize=7, color="#999")
                    continue
                dark = v > lo + 0.55 * (hi - lo)
                if COPY[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", linewidth=0.0,
                                           edgecolor="#ffffff" if dark else "#5a5a5a"))
                if PROV[i, j]:
                    n_prov += 1
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", linewidth=0.35,
                                           edgecolor="#ffffff" if dark else "#5a5a5a"))
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.4,
                        color="white" if dark else "#1a1a1a")
        ax.set_xticks(range(len(PERIODS)))
        ax.set_xticklabels(PERIODS, fontsize=7.0, rotation=90)
        ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels(WINDOWS, fontsize=7.2)
        if r == len(ROWS) - 1:
            ax.set_xlabel("release period requested (ms)", fontsize=8.4)
        if c == 0:
            ax.set_ylabel(f"{cblab}\n\ndeadline  window_duration (ms)", fontsize=8.2)
        ns = sum(v["n_seeds"] for v in arms.values())
        if r == 0:
            ax.set_title(f"{lab}   [{emb}]\nsuccess  {lo:.0f}–{hi:.0f}%   "
                         f"({len(arms)} arms, {ns} seed-runs)", fontsize=9.2)
        elif metric == "energy":
            ax.set_title(f"energy spread {hi/lo:.2f}x   (1.00 = {denom:.3g} N²m²s)",
                         fontsize=8.8)
        else:
            ax.set_title(f"completion {lo:.1f}–{hi:.1f} s", fontsize=8.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.015)
        cb.ax.tick_params(labelsize=6.6)

# The provisional key earns its place only while provisional cells exist; once
# every arm clears MIN_SEEDS a legend entry with no instances is just clutter.
keys = [Patch(facecolor="#9ec5f4", edgecolor="#5a5a5a", hatch="////",
              label="propagated: same schedule as a measured twin"),
        Patch(facecolor="#e4e4e4", edgecolor="#999",
              label="no schedule at this (period, window)")]
if n_prov:
    keys.insert(0, Patch(facecolor="white", edgecolor="#5a5a5a", hatch="////",
                         label=f"provisional: fewer than {MIN_SEEDS} seeds so far"))
fig.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, 0.955),
           frameon=False, fontsize=8.6, ncol=len(keys))
tot = sum(v["n_seeds"] for t in T.values() for v in t.values())
neps = sum(v["n_ep"] for t in T.values() for v in t.values())
done = tot >= 1760
fig.suptitle(("44-arm schedule plane, COMPLETE — " if done else "44-arm schedule plane, sweep IN FLIGHT — ")
             + f"{tot}/1760 cells reduced, {neps:,} episodes",
             fontsize=13.0, y=0.985)
head = ("All 176 (task, arm) pairs at the full 10 seeds x 24 episodes."
        if done else
        "Every (task, arm) pair is already populated: the sweep walks the whole plane at one seed before "
        "advancing to the next, so what grows from here is the error bar, not the coverage.")
fig.text(0.5, 0.055,
         head + "\nMission time is conditioned on success and therefore biased DOWNWARD for arms that fail often; "
         "energy is unconditioned, so an arm that flails without finishing still pays for it.",
         ha="center", va="top", fontsize=8.4, style="italic", color="#555")
fig.savefig(OUT, dpi=145, bbox_inches="tight")
print(f"wrote {OUT}  ({n_prov} provisional cells of {4*3*44})")
