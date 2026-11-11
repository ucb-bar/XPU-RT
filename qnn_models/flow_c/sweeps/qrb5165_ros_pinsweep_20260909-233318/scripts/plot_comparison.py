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

  4. THE OPPONENT IS NAMED. The first version scored pinning against
     "the best measured solver", which sounds like the strongest possible
     opponent and on 30 of 42 cells was really greedy -- phase 4 of the XPU-RT
     sweep tiered by board time and only ran all twelve solvers on twelve
     cells. `--against warmbest` (the default) scores against cpsat:warmbest,
     which is what that study recommends for the offline path;
     `--against best` reproduces the older reading. Each writes its own file
     and says on the figure which one it is.

    python3 plot_comparison.py [--against warmbest|best]
"""
from __future__ import annotations

import argparse
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


def draw(ax, rows, band, xlim, *, excluded=False, title=""):
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
    # Ticks and limits follow the DATA. The limits used to be hardcoded at
    # (0.45, 3.9), which silently CLIPPED any bar beyond them -- on the quad
    # panel both `vint` cells run past 3.9 and were drawn ending at the axis
    # edge, i.e. the figure understated the two largest results on it.
    ticks = [t for t in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
             if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8.5)
    ax.set_xlim(*xlim)
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


# (ratio field, inside-noise field, headline key prefix, output stem,
#  how the opponent is described on the figure)
#: How the opponent is described on the figure.
OPPONENTS = {
    "warmbest": "cpsat:warmbest — the XPU-RT sweep's own recommendation for "
                "the offline/build-time path",
    "best": "the best measured solver per cell — the most favourable reading "
            "for the scheduler",
}

#: (baseline, timing, opponent) -> the per-cell ratio field in analysis.json.
#: `isolation` is the PRIMARY baseline: each network on the lane it is fastest
#: on ALONE, chosen from a per-model benchmark with no knowledge of the
#: co-tenants -- what a ROS user actually deploys. `oracle` is the best of
#: every legal placement measured, which no user has; it is kept as an upper
#: bound on what pinning could reach with perfect knowledge, and the gap
#: between the two is itself a result.
RATIO = {
    ("isolation", "raw", "warmbest"): "iso_over_xrt_np_warmbest",
    ("isolation", "corrected", "warmbest"): "iso_over_xrt_np_warmbest_corrected",
    ("oracle", "raw", "warmbest"): "ros_over_xrt_np_warmbest",
    ("oracle", "corrected", "warmbest"): "ros_over_xrt_np_warmbest_corrected",
    ("isolation", "raw", "best"): "iso_over_xrt_np_best",
    ("isolation", "corrected", "best"): "iso_over_xrt_np_best_corrected",
    ("oracle", "raw", "best"): "ros_over_xrt_np_best",
    ("oracle", "corrected", "best"): "ros_over_xrt_np_best_corrected",
}

BASELINE_TEXT = {
    "isolation": "each network pinned to the lane it is fastest on IN "
                 "ISOLATION — no search, no knowledge of the co-tenants",
    "oracle": "the best of every legal placement measured — a placement "
              "oracle, an upper bound on pinning rather than a deployment",
}
TIMING_TEXT = {
    "raw": "measured from each runtime's t0",
    "corrected": "each runtime re-timed from its OWN first dispatch, so "
                 "neither is charged for the other's startup",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", default="warmbest", choices=sorted(OPPONENTS))
    ap.add_argument("--baseline", default="isolation",
                    choices=["isolation", "oracle"],
                    help="`isolation` (the default and the headline) pins each "
                         "network to the lane it is fastest on alone, which is "
                         "what a ROS user deploys; `oracle` takes the best of "
                         "every legal placement measured, which is an upper "
                         "bound nobody has.")
    ap.add_argument("--timing", default="corrected",
                    choices=["raw", "corrected"],
                    help="`corrected` (the default) re-times BOTH runtimes "
                         "from their own first dispatch; `raw` reproduces the "
                         "earlier reading, measured from each runtime's t0.")
    ap.add_argument("--config", default=None,
                    help="restrict to one lane config (e.g. quad). The config "
                         "axis is which lane SUBSET is available: hd=hta+dsp, "
                         "dc=dsp+cpu, cg=cpu+gpu, quad=all four. Only `quad` "
                         "describes a machine that physically exists -- a "
                         "QRB5165 always has all four backends -- so the other "
                         "three model hardware you cannot buy, and they distort "
                         "individual cells badly (in `hd`, mlp_control_sd is "
                         "forced onto the DSP at 523.6 us because there is no "
                         "CPU lane, though CPU runs it in 110.2 us).")
    a = ap.parse_args()
    # Bind the config now: `a` is rebound by the `for a in (ax, bx, cx)` axes
    # loop further down, so reading a.config after that point picks up an
    # AxesSubplot instead of the namespace.
    CONFIG = a.config
    BASELINE, TIMING, OPP = a.baseline, a.timing, a.against
    RKEY = RATIO[(BASELINE, TIMING, OPP)]
    INSIDE = "inside_noise_" + RKEY
    OPPONENT = OPPONENTS[OPP]
    STEM = ("ros_vs_xpurt_objective"
            + ("" if OPP == "best" else "_warmbest")
            + ("" if BASELINE == "oracle" else "_iso")
            + ("" if TIMING == "raw" else "_corrected"))

    d = json.load(open(os.path.join(SWEEP, "results", "analysis.json")))
    band = 1.0 + d["noise_floor"]["np_pct"] / 100.0

    head, no_ap, uneq = [], [], []
    for c in d["cells"]:
        if CONFIG and c.get("config") != CONFIG:
            continue
        ratio = c.get(RKEY)
        if ratio is None:
            continue
        row = {"label": short(c["cell"]), "ratio": float(ratio),
               "inside": bool(c.get(INSIDE)), "rec": c}
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
    # Read the counts from `results/analysis.json`'s own `readings` block
    # rather than recomputing them here, so the figure cannot drift from
    # ANALYSIS.md. `faster` and `slower` partition the compared cells;
    # `inside` is a SUBSET of those, not a third bucket.
    rec = d["readings"][CONFIG or "all"][BASELINE][OPP][TIMING]
    faster, slower = rec["pinning_faster"], rec["scheduler_faster"]
    inside, median = rec["inside_noise"], rec["median"]
    assert faster + slower == len(head) == rec["n_cells"], \
        (faster, slower, len(head), rec["n_cells"])

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

    allr = [r["ratio"] for r in head + no_ap + uneq]
    xlim = (min(0.45, min(allr) / 1.12), max(3.9, max(allr) * 1.12))
    draw(ax, head, band, xlim,
         title=f"The comparison — {len(head)} cells both runtimes express")
    draw(bx, no_ap, band, xlim, excluded=True,
         title=f"No aperiodic network ({len(no_ap)})\n"
               f"the objective degenerates to the wall clock")
    draw(cx, uneq, band, xlim, excluded=True,
         title=f"Fewer aperiodic instances scheduled ({len(uneq)})\n"
               f"different work, so not a comparison")
    for a in (bx, cx):
        a.title.set_fontsize(9.0)
        a.title.set_color(INK2)

    arrow = dict(arrowstyle="-", lw=0.7, color=MUTED, alpha=0.65)

    # The two tail annotations name a MECHANISM, and a mechanism claim has to
    # be checked against the cell it lands on -- these are hardcoded prose
    # pinned to whichever cell happens to be extreme, and the extreme cell
    # changes when the figure is filtered. Verify, or say nothing.
    LABEL = "iso_label" if BASELINE == "isolation" else "ros_np_best_label"

    def all_aperiodic_on_dsp(rec):
        aper = list((rec.get("np_work_declared") or {}).keys())
        lab = rec.get(LABEL) or ""
        return bool(aper) and all(f"{n}@dsp" in lab for n in aper)

    def is_vint(rec):
        return "vint@" in (rec.get(LABEL) or "")

    # The mechanism annotation describes a WIN. On the isolation baseline the
    # fastest cell can be a tie inside the noise band, and labelling a tie
    # "the timed work gets the fast lane uninterrupted" would be asserting a
    # result the bar does not show. Only annotate a cell that is outside the
    # band.
    if head and not head[0]["inside"] and all_aperiodic_on_dsp(head[0]["rec"]):
        ax.annotate("every aperiodic network pinned to the DSP — the timed\n"
                    "work gets the fast lane uninterrupted, where the\n"
                    "scheduler pays a gate per entry",
                    xy=(head[0]["ratio"], len(head) - 1),
                    xytext=(0.44, 0.965), textcoords="axes fraction",
                    fontsize=8.5, color=INK2, ha="left", va="top", arrowprops=arrow)
    if head and is_vint(head[-1]["rec"]):
        ax.annotate("ViNT's encoder composes on\n"
                    "{DSP, CPU}, its decoder on\n"
                    "{CPU, GPU} — one lane for the\n"
                    "whole network means CPU:\n"
                    "121.99 ms against 14.2 ms\n"
                    "of DSP encoder work",
                    xy=(head[-1]["ratio"], 0), annotation_clip=False,
                    xytext=(0.015, 0.030), textcoords="axes fraction",
                    fontsize=8, color=INK2, ha="left", va="bottom", arrowprops=arrow)

    ax.legend(handles=[
        Patch(facecolor=COOL, label="pinning faster"),
        Patch(facecolor=WARM, label="scheduler faster"),
        Patch(facecolor=NEUTRAL, label=f"±{d['noise_floor']['np_pct']:.2f}% noise floor"),
    ], fontsize=8.5, frameon=False, labelcolor=INK2, loc="center right",
        bbox_to_anchor=(1.0, 0.40), handletextpad=0.4, borderpad=0.2,
        labelspacing=0.3)

    scope = (f"  —  {CONFIG} only (all four backends available)"
             if CONFIG == "quad" else f"  —  {CONFIG} only" if CONFIG else "")
    fig.suptitle(
        "How long the non-periodic work takes while the periodic tasks' "
        f"constraints hold{scope}",
        fontsize=13.5, color=INK, x=0.007, ha="left", y=0.988)
    fig.text(0.007, 0.930,
             f"ROS whole-network pinning ÷ measured XPU-RT against {OPPONENT}, "
             f"medians of 3 reps.\n"
             f"Baseline: {BASELINE_TEXT[BASELINE]}.\n"
             f"Timing: {TIMING_TEXT[TIMING]}.\n"
             f"Median {median:.3f} — {faster} cell{'' if faster == 1 else 's'} "
             f"pinning faster, {slower} scheduler faster; {inside} of the "
             f"{len(head)} sit inside the noise band.",
             fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.007, 0.020,
             "Log axis: a 2× win and a 2× loss are equidistant from 1.0. Bars are "
             "anchored at 1.0; a bar ending inside the grey band is not a result."
             + ("   `quad` is the only config that describes a machine that exists: "
                "a QRB5165 always has all four backends."
                if CONFIG == "quad" else ""),
             fontsize=8.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.795, bottom=0.068)

    out = os.path.join(SWEEP, "plots",
                       STEM + (f"_{CONFIG}" if CONFIG else "") + ".png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"     median {median:.4f}  |  {faster} pinning / {slower} scheduler "
          f"/ {inside} inside noise  |  {len(no_ap)}+{len(uneq)} excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
