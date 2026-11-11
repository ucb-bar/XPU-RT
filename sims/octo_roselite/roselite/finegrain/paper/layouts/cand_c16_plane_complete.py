#!/usr/bin/env python3
"""C16 -- the COMPLETE schedule plane: every solved cell filled, provenance marked.

C14 leaves the long-period corner grey because plane3_lib carries only the 44
MEASURED operating points. Those cells are not missing evidence, and simulating
them would not add any: CP-SAT returns the SAME schedule for them as for a cell
already measured, because past ~220 ms the deadline window stops binding -- the
schedule already finishes inside it, so widening the window changes nothing.

The numbers settle it at the bit level, not just the physical one. The 27
propagated cells collapse to FIVE distinct simulator commands (g220_260 x8,
g250_260 x8, g283_260 x8, g200_400 x2, g150_350 x1), and comparing each cell's
(lat_med, cadence) against its twin's:

  26 of 27 are BIT-IDENTICAL at full float precision.
   1 of 27 (g150_400) differs by 0.042 ms of latency -- and the arm table the job
     scripts read formats latency to 0.1 ms, so 319.012 and 318.970 both become
     "319.0" and even that one is the same command as executed.

So 0 of 27 are bit-different as the pipeline actually runs them. trace_eval2.py
is handed nothing but those two numbers; re-running these cells would sample
harness nondeterminism 27 times under 27 different labels, not measure anything
new. (For scale, the physical argument is looser still: 0.042 ms against a 40 ms
simulator tick.)

So the plane is completed by PROPAGATION, drawn with a hatch. Hatch rather than a
dimmed colour: the fill is a value on the scale, and dimming it would read as a
lower value. A hatched cell is a COPY of measured evidence, never new evidence,
and every panel's colour scale and printed range is computed on MEASURED cells
only -- asserted below, not merely intended.

The grey cells carry TWO different claims and are marked apart, because they are
not the same statement:
  x  INFEASIBLE (25) -- no schedule exists at that (period, window). A fact.
  ?  UNKNOWN    (12) -- CP-SAT found none within its budget. An outcome of the
                        search, not a property of the plane; a longer budget
                        might solve them. They sit at SHORT periods, 100-130 ms.
Collapsing the two into one grey would claim the second is the first.
"""
from __future__ import annotations
import json, re, sys
import numpy as np
import matplotlib
matplotlib.rcParams["hatch.linewidth"] = 0.5
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import vocab as V
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
PAPER = V.HERE.parent
warm = json.load(open(PAPER / "grid_warm.json"))["cells"]
succ = json.load(open(PAPER / "grid_e2e_success.json"))
TWIN = {a: v["twin"] for a, v in succ["egg"].items() if v.get("source") == "propagated"}
T = L.table()
plane = L.arm_plane()
PERIODS = [100, 110, 120, 130, 140, 150, 165, 180, 200, 220, 250, 283]
WINDOWS = [240, 260, 275, 290, 305, 320, 350, 400, 450]
ROWS = [("success", "success (%)", "Blues"),
        ("mission", "mission time (s)", "Purples"),
        ("energy", "energy / cheapest cell", "Oranges")]

fig, axes = plt.subplots(len(ROWS), len(L.TASKS), figsize=(7.16, 5.05), dpi=170)
fig.subplots_adjust(left=0.079, right=0.983, top=0.900, bottom=0.105,
                    wspace=0.44, hspace=0.30)
n_meas = n_prop = n_grey = 0

for c, (task, tlab, emb) in enumerate(L.TASKS):
    arms = T[task]
    for r, (metric, lab, cmap) in enumerate(ROWS):
        ax = axes[r, c]
        M = np.full((len(WINDOWS), len(PERIODS)), np.nan)
        P = np.zeros_like(M, dtype=bool)
        meas = []
        for a, v in arms.items():                       # the 44 measured points
            p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
            M[WINDOWS.index(w), PERIODS.index(p)] = v[metric]
            meas.append(v[metric])
        for a, twin in TWIN.items():                    # copies of those points
            p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
            if twin in arms:
                i, j = WINDOWS.index(w), PERIODS.index(p)
                M[i, j] = arms[twin][metric]; P[i, j] = True
        lo, hi = min(meas), max(meas)
        assert np.nanmin(M) >= lo - 1e-9 and np.nanmax(M) <= hi + 1e-9, \
            f"{task}/{metric}: a propagated cell falls outside the measured range"
        denom = lo
        if metric == "energy":
            M, lo, hi = M / denom, 1.0, hi / denom
        ax.set_facecolor(V.ABSENT)
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=lo, vmax=hi)
        for i in range(len(WINDOWS)):
            for j in range(len(PERIODS)):
                if not np.isfinite(M[i, j]):
                    st = warm.get(f"{PERIODS[j]}_{WINDOWS[i]}", {}).get("status", "")
                    ax.text(j, i, {"INFEASIBLE": "\u2717", "UNKNOWN": "?"}.get(st, ""),
                            ha="center", va="center", fontsize=3.6, color="#8f8e88")
                if P[i, j]:
                    dark = M[i, j] > lo + 0.55 * (hi - lo)
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", linewidth=0.0,
                                           edgecolor="white" if dark else "#5a5a5a"))
        if r == 0 and c == 0:
            n_meas, n_prop = len(meas), int(P.sum())
            n_grey = M.size - n_meas - n_prop
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
        cb.ax.tick_params(labelsize=4.8)

fig.legend(handles=[Patch(facecolor="#9ec5f4", edgecolor="none", label=f"measured ({n_meas})"),
                    Patch(facecolor="#9ec5f4", edgecolor="#5a5a5a", hatch="////",
                          label=f"propagated from a twin ({n_prop})"),
                    Patch(facecolor=V.ABSENT, edgecolor="#999",
                          label="✗ infeasible (25)"),
                    Patch(facecolor=V.ABSENT, edgecolor="#999",
                          label="? unsolved in budget (12)")],
           loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=4, fontsize=6.0)
fig.suptitle("The complete schedule plane, all four workloads", fontsize=8.4, y=0.992)
fig.text(0.5, 0.006, "a propagated cell is the same simulator command as its twin — "
                     "26 of 27 bit-identical, all 27 identical at the 0.1 ms the job scripts use",
         ha="center", va="bottom", fontsize=5.8, color=V.MUTED)
V.save(fig, "cand_c16_plane_complete")
