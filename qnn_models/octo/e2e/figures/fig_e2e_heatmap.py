#!/usr/bin/env python3
"""MEASURED end-to-end success across the schedule plane.

fig_schmoo.py colours the same plane by what the SCHEDULER predicts (observation age,
cadence). This one colours it by what the ROBOT actually achieved: 44 operating points x
3 tasks x 20 seeds x 24 episodes = 63,360 episodes.

Colour is a magnitude, so it is a SEQUENTIAL single hue, light -> dark (never a rainbow,
never diverging -- there is no meaningful midpoint in a success rate). Each panel carries
its OWN scale because the tasks have different baseline difficulty; comparing tasks is
done through the printed RANGE and the noise band, not through a shared colour, which
would conflate task difficulty with schedule sensitivity.

PROVENANCE. A filled cell is one of two things and the figure must not conflate them:
  MEASURED   -- simulated end to end (plain fill).
  PROPAGATED -- solved, but CP-SAT's schedule has the same (observation age, cadence) as
                a MEASURED cell.  job_grid.sh passes the simulator nothing but those two
                numbers, so the two cells are the same command; re-running one would
                measure harness nondeterminism, not a new operating point.  Drawn with a
                diagonal HATCH over the same fill.
Hatch, not desaturation: the fill is a value on the colour scale, so dimming it would read
as a lower success rate.  The hatch adds a mark without moving the value.

A propagated cell is a COPY of measured evidence, never new evidence.  Every statistic
here -- the per-panel colour scale, the range, the noise-band multiple -- is computed on
source=="measured" cells ONLY.  This is asserted below, not merely intended.

Cells with no schedule (INFEASIBLE, UNKNOWN) are neutral grey and labelled -- they are
absent data, not zero success.

The range annotation is the finding: egg and spoon vary ~2x the coke range across the same
plane, and coke's variation is within a few noise bands of flat.
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 0.65
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "fig_e2e_heatmap.png"
PERIODS = [100, 110, 120, 130, 140, 150, 165, 180, 200, 220, 250, 283]
WINDOWS = [240, 260, 275, 290, 305, 320, 350, 400, 450]
# measured 95% noise band on a 10-seed paired mean (g5grid/analyze_merged.py)
# 20-seed empirical bands (g5grid/analyze_merged.py). NOT the 10-seed values: the
# band must match the seed count the contrasts actually average, or every test is
# run against a null that is ~sqrt(2) too wide.
FLOOR = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}
TASKS = [("egg", "eggplant in basket", "widowx, 40 ms grid"),
         ("spoon", "spoon on towel", "widowx, 40 ms grid"),
         ("coke", "pick coke can", "google, 333 ms grid"),
         ("drawer", "close drawer", "google, 333 ms grid")]

succ = json.load(open(DATA / "grid_e2e_success.json"))
warm = json.load(open(DATA / "grid_warm.json"))["cells"]

fig, axes = plt.subplots(1, 4, figsize=(22.0, 7.6), dpi=150,
                         gridspec_kw={"wspace": 0.16})
stats = {}
assert len(axes) == len(TASKS), (len(axes), len(TASKS))  # zip() would silently drop a task
for ax, (task, lab, emb) in zip(axes, TASKS):
    M = np.full((len(WINDOWS), len(PERIODS)), np.nan)     # every filled cell
    P = np.zeros_like(M, dtype=bool)                      # propagated?
    meas = []
    for arm, v in succ[task].items():
        m = re.match(r"g(\d+)_(\d+)$", arm)
        if not m:
            continue
        p, w = int(m.group(1)), int(m.group(2))
        if p not in PERIODS or w not in WINDOWS:
            continue
        i, j = WINDOWS.index(w), PERIODS.index(p)
        M[i, j] = v["rate"]
        P[i, j] = v.get("source") == "propagated"
        if not P[i, j]:
            meas.append(v["rate"])
    # SCALE AND RANGE ON MEASURED CELLS ONLY -- propagation must not move either.
    lo, hi = min(meas), max(meas)
    assert np.nanmin(M) >= lo - 1e-9 and np.nanmax(M) <= hi + 1e-9, \
        f"{task}: a propagated cell falls outside the measured range"
    stats[task] = (len(meas), int(P.sum()), lo, hi)
    ax.set_facecolor("#e4e4e4")                 # absent != zero
    im = ax.imshow(M, cmap="Blues", aspect="auto", origin="lower", vmin=lo, vmax=hi)
    for i in range(len(WINDOWS)):
        for j in range(len(PERIODS)):
            v = M[i, j]
            if np.isfinite(v):
                dark = v > lo + 0.55 * (hi - lo)
                if P[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="////", linewidth=0.35,
                                           edgecolor="#ffffff" if dark else "#5a5a5a"))
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if dark else "#1a1a1a",
                        style="italic" if P[i, j] else "normal")
            else:
                st = warm.get(f"{PERIODS[j]}_{WINDOWS[i]}", {}).get("status", "")
                ax.text(j, i, {"INFEASIBLE": "✗", "UNKNOWN": "?"}.get(st, "·"),
                        ha="center", va="center", fontsize=7, color="#999")
    ax.set_xticks(range(len(PERIODS))); ax.set_xticklabels(PERIODS, fontsize=7.6)
    ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels(WINDOWS, fontsize=7.6)
    ax.set_xlabel("release period requested (ms)", fontsize=8.6)
    ax.set_title(f"{lab}   [{emb}]\nrange {hi-lo:.1f} pts  =  "
                 f"{(hi-lo)/FLOOR[task]:.1f} x its noise band (±{FLOOR[task]:.2f})",
                 fontsize=9.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.015)
    cb.ax.tick_params(labelsize=7)
    if task == TASKS[-1][0]:
        cb.set_label("success rate (%)", fontsize=7.6)
axes[0].set_ylabel("deadline  window_duration (ms)", fontsize=8.6)

n_meas, n_prop, _, _ = stats["egg"]
n_inf = sum(1 for c in warm.values() if c["status"] == "INFEASIBLE")
n_unk = sum(1 for c in warm.values() if c["status"] == "UNKNOWN")
assert n_meas + n_prop + n_inf + n_unk == len(PERIODS) * len(WINDOWS)

fig.legend(handles=[Patch(facecolor="#9ecae1", edgecolor="#6b6b6b", label="measured"),
                    Patch(facecolor="#9ecae1", edgecolor="#5a5a5a", hatch="////",
                          label="propagated:  same (age, cadence) as a measured cell — "
                                "copied, not re-run"),
                    Patch(facecolor="#e4e4e4", edgecolor="#bbb",
                          label="no schedule:  ✗ INFEASIBLE   ? UNKNOWN")],
           loc="upper center", bbox_to_anchor=(0.5, 0.917), frameon=False, fontsize=8.2,
           ncol=3, handlelength=1.7, columnspacing=1.9)
fig.suptitle("MEASURED end-to-end success across the schedule plane  —  "
             "63,360 episodes (44 operating points × 3 tasks × 20 seeds × 24)",
             fontsize=12.6, y=0.975)
fig.text(0.5, 0.171,
         "Colour is per-panel: the tasks differ in baseline difficulty, so a shared scale "
         "would conflate task difficulty with schedule sensitivity. Compare tasks through "
         "the RANGE in each title, expressed in that task's own measured noise band.\n"
         "The two widowx tasks (40 ms actuation, never saturated by these schedules) vary "
         f"{(stats['egg'][3]-stats['egg'][2])/FLOOR['egg']:.1f} and "
         f"{(stats['spoon'][3]-stats['spoon'][2])/FLOOR['spoon']:.1f} noise bands across "
         "the plane; coke (333 ms actuation, saturated by every schedule here) varies "
         f"{(stats['coke'][3]-stats['coke'][2])/FLOOR['coke']:.1f} — and its variation "
         "carries no trend:\n"
         "its per-seed cadence ladder is ρ = +0.07 [-0.22, +0.35], against ρ = -0.58 (egg) "
         "and -0.49 (spoon), both p < 0.0002.\n"
         "At n=20, FOUR of the 6 pre-specified TREND tests survive Benjamini-Hochberg "
         "(q=0.05): BOTH cadence ladders and BOTH latency ladders on the two widowx tasks.\n"
         "Neither coke test survives — both trend intervals span zero (cadence "
         "ρ = -0.20 [-0.52, +0.12], latency ρ = -0.13 [-0.33, +0.07]).\n"
         "So on an UNSATURATED actuator both axes matter, cadence about twice as strongly "
         "as latency; on a SATURATED one neither is resolvable.\n"
         f"PROVENANCE — {n_meas} cells MEASURED (plain), {n_prop} PROPAGATED (hatched, "
         "italic). A propagated cell is a SOLVED cell whose CP-SAT schedule carries the "
         "same (observation age, cadence) as a measured cell, matched by TIMING to ≤ 0.6 "
         "ms, never by name;\n"
         "job_grid.sh feeds the simulator only those two numbers, so re-running it would "
         "measure harness nondeterminism, not a new operating point. Propagated cells are "
         "COPIES, never evidence: the colour scale, both range figures and every test "
         "above use the "
         f"{n_meas} measured cells only.\n"
         f"The {n_inf + n_unk} empty cells are {n_inf} ✗ INFEASIBLE (proven impossible — "
         f"they stay empty) and {n_unk} ? UNKNOWN (CP-SAT budget exhausted at 3600 s, "
         "which is not a proof of infeasibility). Rates are means over the 20 DISTINCT "
         "seeds of a cell; duplicated fetch copies are counted once, not summed per host.",
         ha="center", va="top", fontsize=7.9, style="italic", color="#555")
fig.subplots_adjust(top=0.800, bottom=0.235)
fig.savefig(OUT, dpi=150)
print(f"[ok] {OUT}")
print(f"  plane: {n_meas} measured + {n_prop} propagated = {n_meas+n_prop} filled; "
      f"{n_inf} INFEASIBLE + {n_unk} UNKNOWN = {n_inf+n_unk} empty  (108 cells)")
for t, _, _ in TASKS:
    nm, np_, lo, hi = stats[t]
    print(f"  {t:6s} MEASURED n_cells={nm:2d}  {lo:.1f}-{hi:.1f}%  "
          f"range {hi-lo:.1f} = {(hi-lo)/FLOOR[t]:.1f} noise bands   (+{np_} propagated)")
