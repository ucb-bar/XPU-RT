#!/usr/bin/env python3
r"""C19 -- C17b with the energy axis CALIBRATED to joules.

Left: success against calibrated energy per episode, all four workloads, log
axis. Right: success on the plane, four small multiples -- unchanged from C17,
because success rate needs no calibration.

CALIBRATION IS GLOBAL, APPLIED PER EPISODE. Constants derived once, then used to
transform every episode so the per-arm value keeps the same estimator as the rest
of the sweep (median over 24 episodes, mean over seeds):

    E_J(episode) = c_qf * t2_qf_sub_arm  +  P_idle * duration_s

    widowx  c_qf = 0.827 W/(N.m)^2, P_idle = 4.61 W  -- DERIVED from ROBOTIS'
            published stall torque/current for the Dynamixel XM430-W350.
    google  c_qf = 0.0125 W/(N.m)^2, P_idle = 40 W   -- PROXY. Everyday Robots
            published nothing; these come from a Harmonic Drive FHA-14C-100 joint
            module and a Kinova/Franka whole-arm envelope.

WHY THE GOOGLE SERIES ARE DRAWN HOLLOW. Two reasons, both disqualifying for
quantitative use. Their constants are a proxy, and P_idle is ~92% of the result,
so the numbers are proxy-determined rather than measured. And calibration does
not preserve their ordering: Spearman between the raw tau^2 channel and the
calibrated energy is +0.26 on coke and -0.04 on drawer, against +0.92 and +0.77
on the two widowx scenes. Filled markers are calibrated from published data;
hollow markers are indicative only.

This is the quasi-static LOWER bound. The raw tau^2 proxy cannot be converted at
face value at all: the simulated PD torque runs ~76x the physical requirement, so
converting it literally gives 1.6 kW-84 kW against a 60 W supply. Full derivation
in calib/ENERGY_CALIBRATION.md.
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
# Global calibration constants, applied PER EPISODE. widowx is DERIVED from
# ROBOTIS' published stall torque/current; google is a PROXY (Everyday Robots
# published nothing) and its series are drawn hollow for that reason.
# Google constants are BALANCE-MATCHED to the WidowX, not independently sourced.
#
# The Everyday Robots arm has no published actuator data. calib/ used a Harmonic
# Drive FHA-14C-100 for c and a Kinova/Franka envelope for P_idle -- two proxies
# from different machines, and the pairing was incoherent: a Franka-class 40 W
# idle next to a 3 W actuation term, so idle was 93% of the result and the
# actuation signal was buried under an assumed constant.
#
# This study is about how end-to-end robot behaviour responds to scheduling, not
# about absolute watts for a robot nobody has specified. So c_google is set so the
# arm's dynamic/idle BALANCE matches the WidowX's measured 0.810, keeping the
# literature-anchored 40 W idle:
#
#     c_google = 0.0125 * (0.810 * 40) / 3.07 = 0.1321 W/(N.m)^2
#
# That gives 30-31 W of actuation against 40 W idle (43% actuation), beside the
# WidowX's 45% -- a plausible balance for a Franka-class arm.
#
# WHAT THIS CHANGES: absolute joules for the google tasks, nothing else. Scaling c
# scales every arm equally, so the across-arm ratios are EXACTLY invariant --
# drawer max/min actuation is 1.0269 under either constant, coke 1.2404. That
# drawer is flat and coke nearly so does not rest on the proxy at all; only the
# vertical scale does. The series stay hollow to mark that.
CAL = {"egg": (0.8267, 4.608), "spoon": (0.8267, 4.608),      # DERIVED
       "coke": (0.1321, 40.0), "drawer": (0.1321, 40.0)}      # BALANCE-MATCHED
PROXY = {"egg": False, "spoon": False, "coke": True, "drawer": True}


def energy_J(task, arm_cells):
    """Median over an arm's 24 episodes, then mean over seeds -- the estimator
    every other panel in this set uses."""
    c, pi = CAL[task]
    return float(np.mean([np.median([c * e["t2_qf_sub_arm"] + pi * e["duration_s"]
                                     for e in d["episodes"]])
                          for d in arm_cells.values()]))


HUE = {"egg": "#1b7f5f", "spoon": "#2a78d6", "coke": "#c0392b", "drawer": "#7a3fa5"}
MARK = {"egg": "o", "spoon": "s", "coke": "^", "drawer": "D"}
SHORT = {"egg": "eggplant", "spoon": "spoon", "coke": "coke can", "drawer": "drawer"}
CMAP = {t: LinearSegmentedColormap.from_list(t, ["#f4f3ef", h]) for t, h in HUE.items()}

fig = plt.figure(figsize=(7.16, 3.30), dpi=170)
gs = fig.add_gridspec(2, 3, width_ratios=[1.42, 0.79, 0.79], left=0.068,
                      right=0.972, top=0.862, bottom=0.108, wspace=0.52, hspace=0.52)
ax = fig.add_subplot(gs[:, 0])

# google first, widowx on top -- see docstring
ORDER = ["coke", "drawer", "spoon", "egg"]
BEST = {t: max(T[t], key=lambda k: T[t][k]["success"]) for t in ORDER}
RAW = L.load()
EJ = {t: {a: energy_J(t, RAW[(t, a)]) for a in T[t]} for t, _, _ in L.TASKS}
for zi, task in enumerate(ORDER):
    arms = T[task]; ks = sorted(arms)
    e = np.array([EJ[task][a] for a in ks]); sv = np.array([arms[a]["success"] for a in ks])
    hollow = PROXY[task]
    ax.scatter(e, sv, s=15, marker=MARK[task],
               c="none" if hollow else HUE[task], edgecolor=HUE[task],
               linewidth=0.7 if hollow else 0.30, zorder=4 + zi,
               label=SHORT[task] + (" (proxy)" if hollow else ""))
    b = BEST[task]
    ax.scatter(EJ[task][b], arms[b]["success"], s=78, marker="*",
               c="white", edgecolor=HUE[task], linewidth=1.0, zorder=12 + zi)
ax.set_xscale("log")
ax.set_xlabel("actuator energy per episode (J), log scale", fontsize=6.6, labelpad=1.5)
ax.set_ylabel("success rate (%)", fontsize=6.8, labelpad=2.0)
ax.tick_params(labelsize=6.0)
ax.set_title("success vs calibrated energy", fontsize=7.2, pad=3.0)
ax.margins(0.10); ax.set_ylim(0, ax.get_ylim()[1])
V.tidy(ax, grid="both")
h, lb = ax.get_legend_handles_labels()
o = [lb.index(SHORT[t] + (" (proxy)" if PROXY[t] else "")) for t, _, _ in L.TASKS]        # egg, spoon, coke, drawer
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

fig.text(0.735, 0.968, "SUCCESS RATE on the schedule plane, per workload",
         ha="center", va="top", fontsize=7.4, fontweight="bold", color=V.INK)
# Caption lives in cand_c19_caption.tex, not on the canvas.
V.save(fig, "cand_c19_all_tasks_calibrated")
