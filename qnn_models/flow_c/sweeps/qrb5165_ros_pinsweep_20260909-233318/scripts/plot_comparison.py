#!/usr/bin/env python3
"""ROS whole-network pinning vs XPU-RT scheduling, on the objective.

Supersedes `plots/ros_vs_xpurt_nonperiodic.png` for reporting. Same data,
three changes, each fixing something that made the first version misread:

  1. THE AXIS IS LOG. These are ratios. On a linear axis "2x slower" sits
     twice as far from 1.0 as "2x faster", so the eye reads the scheduler's
     wins as bigger than pinning's wins of identical magnitude. On log2 they
     are symmetric, which is the only honest way to draw a ratio.
  2. THE CLAIM IS SEPARATED FROM THE EXCLUSIONS. The first version sorted all
     42 cells into one column and greyed the 16 that are not part of the
     headline, which leaves the reader to do the exclusion themselves and
     invites reading a grey bar as a result. The 26 headline cells and the 16
     excluded ones are now different panels with their reasons named.
  3. THE MECHANISM IS ON THE FIGURE. The distribution is bimodal and the two
     tails have different causes; that is the finding, and it was previously
     only in the prose.

Colour is the reference palette's documented diverging pair (blue <-> red,
neutral grey midpoint), used unmodified. The skill's validator is a node
script and this host has no node, so inventing or re-stepping hexes here
would mean shipping unvalidated ones; the shipped pair is pre-validated.

    python3 plot_comparison.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)

# Reference palette: diverging pair + neutral midpoint, chrome and ink.
SURFACE = "#fcfcfb"
COOL, WARM = "#2a78d6", "#e34948"        # pinning faster / scheduler faster
NEUTRAL = "#f0efec"                       # diverging midpoint
EXCLUDED = "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def short(cell: str) -> str:
    return cell[len("networks_"):] if cell.startswith("networks_") else cell


def draw(ax, rows, band, *, excluded=False, title=""):
    """Horizontal diverging bars anchored at 1.0 on a log2 axis."""
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ratio = r["ratio"]
        color = EXCLUDED if excluded else (WARM if ratio > 1.0 else COOL)
        # Anchored at 1.0; on a log axis the bar spans [min(1,r), max(1,r)].
        lo, hi = min(1.0, ratio), max(1.0, ratio)
        ax.barh(yi, hi - lo, left=lo, height=0.62, color=color,
                edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"], fontsize=8.5)
    ax.set_xlim(0.45, 3.9)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    # The noise band, drawn over the bars: a bar that ends inside it is not a
    # result, and that has to be visible without consulting a table.
    ax.axvspan(1 / band, band, color=NEUTRAL, alpha=0.85, zorder=1)
    ax.axvline(1.0, color=BASELINE, lw=1.2, zorder=2)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)


def main() -> int:
    d = json.load(open(os.path.join(SWEEP, "results", "analysis.json")))
    band = 1.0 + d["noise_floor"]["np_pct"] / 100.0

    head, no_ap, uneq = [], [], []
    for c in d["cells"]:
        ratio = c.get("ros_over_xrt_np_best")
        if ratio is None:
            continue
        row = {"label": short(c["cell"]), "ratio": float(ratio),
               "inside": bool(c.get("inside_noise_np"))}
        if c.get("np_degenerate"):
            no_ap.append(row)
        elif not c.get("np_work_equal", True):
            uneq.append(row)
        else:
            head.append(row)
    for lst in (head, no_ap, uneq):
        lst.sort(key=lambda r: r["ratio"])

    # Read the headline counts from the record rather than recomputing them,
    # so the figure cannot drift from ANALYSIS.md. `faster` and `slower`
    # partition all 26 cells; `inside` is a SUBSET of those, not a third
    # bucket -- recomputing it as one is how the first draft got 10/8/8.
    h = d["headline"]
    faster, slower = h["np_ros_faster_cells"], h["np_xrt_faster_cells"]
    inside, median = h["np_inside_noise_cells"], h["np_ros_over_xrt_median"]
    assert faster + slower == len(head), (faster, slower, len(head))

    # Two exclusions, two sub-panels. Repeating the reason in all 16 tick
    # labels made them long enough to overrun the neighbouring panel, and a
    # reason that is constant within a group is a title, not a per-row string.
    fig = plt.figure(figsize=(14.6, 8.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          height_ratios=[len(no_ap), len(uneq)],
                          wspace=0.34, hspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 1])
    for a in (ax, bx, cx):
        a.set_facecolor(SURFACE)

    draw(ax, head, band,
         title=f"The comparison — {len(head)} cells both runtimes express")
    draw(bx, no_ap, band, excluded=True,
         title=f"No aperiodic network ({len(no_ap)})\n"
               f"the objective degenerates to the wall clock")
    draw(cx, uneq, band, excluded=True,
         title=f"Fewer aperiodic instances scheduled ({len(uneq)})\n"
               f"different work, so not a comparison")
    for a in (bx, cx):
        a.title.set_fontsize(9.0)
        a.title.set_color(INK2)

    arrow = dict(arrowstyle="-", lw=0.7, color=MUTED, alpha=0.65)
    # Blue tail is at the TOP (head[0], smallest ratio); its text goes in the
    # empty top-right quadrant. Red tail is at the BOTTOM; its text goes
    # bottom-left. Getting these two backwards is what the first render did.
    ax.annotate("every aperiodic network pinned to the DSP — the timed\n"
                "work gets the fast lane uninterrupted, where the\n"
                "scheduler pays a gate per entry",
                xy=(head[0]["ratio"], len(head) - 1),
                xytext=(0.44, 0.965), textcoords="axes fraction",
                fontsize=8.5, color=INK2, ha="left", va="top", arrowprops=arrow)
    ax.annotate("ViNT's encoder composes on\n"
                "{DSP, CPU}, its decoder on\n"
                "{CPU, GPU} — one lane for the\n"
                "whole network means CPU:\n"
                "121.99 ms against 14.2 ms\n"
                "of DSP encoder work",
                xy=(head[-1]["ratio"], 0),
                xytext=(0.015, 0.030), textcoords="axes fraction",
                fontsize=8, color=INK2, ha="left", va="bottom", arrowprops=arrow)

    ax.legend(handles=[
        Patch(facecolor=COOL, label="pinning faster"),
        Patch(facecolor=WARM, label="scheduler faster"),
        Patch(facecolor=NEUTRAL, label=f"±{d['noise_floor']['np_pct']:.2f}% noise floor"),
    ], fontsize=8.5, frameon=False, labelcolor=INK2, loc="center right",
        bbox_to_anchor=(1.0, 0.40), handletextpad=0.4, borderpad=0.2,
        labelspacing=0.3)

    fig.suptitle(
        "How long the non-periodic work takes while the periodic tasks' "
        "constraints hold", fontsize=13.5, color=INK, x=0.007, ha="left", y=0.988)
    fig.text(0.007, 0.930,
             f"ROS whole-network pinning ÷ measured XPU-RT, best legal placement "
             f"vs best measured solver, medians of 3 reps.\n"
             f"Median {median:.3f} — {faster} cells pinning faster, {slower} scheduler "
             f"faster; {inside} of the {len(head)} sit inside the noise band.",
             fontsize=9.5, color=INK2, ha="left", va="top")
    fig.text(0.007, 0.020,
             "Log axis: a 2× win and a 2× loss are equidistant from 1.0. Bars are "
             "anchored at 1.0; a bar ending inside the grey band is not a result.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.830, bottom=0.068)

    out = os.path.join(SWEEP, "plots", "ros_vs_xpurt_objective.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"     median {median:.4f}  |  {faster} pinning / {slower} scheduler "
          f"/ {inside} inside noise  |  {len(no_ap)}+{len(uneq)} excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
