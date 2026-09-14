#!/usr/bin/env python3
"""C15 -- eggplant: the success/energy relationship beside the plane it lives on.

Two views of ONE quantity. Left: what success costs in actuator energy across the
44 operating points. Right: where in the schedule space those points sit, release
period across and deadline window down.

ONE COLOUR MEANING, ONE COLOURBAR. Both panels encode SUCCESS RATE on the same
Blues ramp and the same limits, so the panels are linked by eye: find a dark
point on the left, find the dark cells on the right, and you have read the
operating point's schedule off the map. Colouring the scatter by release period
instead would put two different sequential blues in one figure, and a reader
would have to hold which is which.

The scatter's y position and its colour are the same variable. That redundancy is
the linkage and is deliberate -- it is what lets the two panels share a legend.

rho(success, energy) = -0.94: the tightest coupling in the whole sweep. Better
schedules are not buying success with energy, they are spending less of it.
"""
from __future__ import annotations
import re, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import vocab as V
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
TASK = "egg"
T = L.table()[TASK]
plane = L.arm_plane()
PERIODS = sorted({int(re.match(r"g(\d+)_", a).group(1)) for a in plane})
WINDOWS = sorted({int(re.match(r"g\d+_(\d+)$", a).group(1)) for a in plane})

arms = sorted(T)
S = np.array([T[a]["success"] for a in arms])
E = np.array([T[a]["energy"] for a in arms])
LO, HI = S.min(), S.max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.62), dpi=170,
                             gridspec_kw={"width_ratios": [1.0, 1.12]})
fig.subplots_adjust(left=0.072, right=0.885, top=0.858, bottom=0.175, wspace=0.30)

# ---- left: success against energy ----------------------------------------
sc = ax.scatter(E, S, c=S, cmap="Blues", vmin=LO, vmax=HI, s=34,
                edgecolor=V.INK2, linewidth=0.45, zorder=4)
ax.set_xlabel("actuator energy (N²m²s)", fontsize=6.8, labelpad=1.5)
ax.set_ylabel("success rate (%)", fontsize=6.8, labelpad=2.0)
ax.tick_params(labelsize=6.0)
ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
ax.xaxis.get_offset_text().set_fontsize(5.6)
ax.set_title(f"success vs energy   ρ = {spearmanr(E, S).statistic:+.2f}",
             fontsize=7.4, pad=3.0)
ax.margins(0.11)
V.tidy(ax, grid="both")
# Parked top-right: the cloud runs top-left to bottom-right, so that corner is
# the one nothing occupies.
ax.annotate("", xy=(0.790, 0.955), xytext=(0.950, 0.845),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=V.MUTED, lw=0.9))
ax.text(0.962, 0.832, "better", transform=ax.transAxes, fontsize=6.2,
        color=V.MUTED, ha="right", va="top", style="italic")

# ---- right: the same numbers on the schedule plane ------------------------
M = np.full((len(WINDOWS), len(PERIODS)), np.nan)
for a in arms:
    p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
    M[WINDOWS.index(w), PERIODS.index(p)] = T[a]["success"]
bx.set_facecolor(V.ABSENT)
im = bx.imshow(M, cmap="Blues", aspect="auto", origin="lower", vmin=LO, vmax=HI)
bx.set_xticks(range(len(PERIODS)))
bx.set_xticklabels(PERIODS, fontsize=5.4, rotation=90)
bx.set_yticks(range(len(WINDOWS))); bx.set_yticklabels(WINDOWS, fontsize=5.6)
bx.set_xlabel("release period (ms)", fontsize=6.8, labelpad=1.5)
bx.set_ylabel("deadline window (ms)", fontsize=6.8, labelpad=2.0)
bx.set_title("the same 44 points, on the schedule plane", fontsize=7.4, pad=3.0)

cb = fig.colorbar(im, ax=[ax, bx], fraction=0.026, pad=0.014)
cb.set_label("success rate (%) — one scale, both panels", fontsize=6.0)
cb.ax.tick_params(labelsize=5.8)
fig.suptitle("eggplant in basket  ·  44 operating points × 10 seeds × 24 episodes",
             fontsize=8.2, x=0.478, y=0.982)
fig.text(0.478, 0.008, "grey = no feasible schedule at that (period, window)",
         ha="center", va="bottom", fontsize=6.0, color=V.MUTED)
V.save(fig, "cand_c15_egg_pair")
