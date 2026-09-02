#!/usr/bin/env python3
"""The compiler<->scheduler loop, with the contract on every arrow.

A schematic, not a plot: nothing here is measured, and it says so. Its job is
to name the artefact that crosses each boundary, because the whole design is
that the two projects talk through FILES with contracts rather than through
code, and someone who knows which file to look at can debug any stage alone.

The verb table underneath is measured, and it is the part that goes stale:
each verb's state is what its round-trip has actually reached.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

figstyle.use()

#: (label, sublabel, owner). Laid out as a U: the top row runs left to right
#: from the IR to a schedule, the bottom row runs right to left from the
#: measured run back to a rewritten IR. The cycle closes at the left edge,
#: which is the point -- the output of the loop is an input to itself.
NODES = [
    ("ModelBlaster\nIR",   "graph.json",          "mb"),
    ("build +\nkernels",   "kernel_picks.json",   "mb"),
    ("board\nprofile",     "results.csv",         "mb"),
    ("XPU-RT\nschedule",   "scheduled_*.json",    "rt"),
    ("board\nrun",         "trace.csv",           "both"),
    ("advice",              "compile_advice.json", "rt"),
    ("hint",                "*_hints/v1",          "rt"),
    ("rewrite",             "graph.json'",         "mb"),
]

#: Top row left to right, then down the right edge, then bottom row right to
#: left, then up the left edge.
POS = [(0, 1), (1.85, 1), (3.70, 1), (5.55, 1),
       (5.55, 0), (3.70, 0), (1.85, 0), (0, 0)]

OWNER_COLOR = {"mb": figstyle.BLUE, "rt": figstyle.ORANGE,
               "both": figstyle.GREEN}
OWNER_LABEL = {"mb": "ModelBlaster", "rt": "XPU-RT",
               "both": "both (MB harness, XPU-RT schedule)"}

#: (from, to, what carries it)
EDGES = [
    (0, 1, "generate_kernels"),
    (1, 2, "profile_writer"),
    (2, 3, "profile_loader"),
    (3, 4, "harness_xpurt"),
    (4, 5, "emit_compile_advice"),
    (5, 6, "advice_to_*_hint"),
    (6, 7, "apply_*_hint"),
    (7, 0, "diff_dispatch_graph\n(gate)"),
]


#: verb, producer, bridge, consumer, and where its round-trip has ACTUALLY
#: reached. This is the part of the figure that goes stale, so it states an
#: outcome rather than a capability -- "rejected on measurement" is the loop
#: working, and reads as such only if the rejection is on the record.
VERBS = [
    ("fuse_with_successor", "overhead_advice", "advice_to_fusion_hint",
     "apply_fusion_hint", "board: rejected, 36% slower"),
    ("split", "blocking_advice", "advice_to_split_hint",
     "apply_split_hint", "board: rejected, +13.7% (DroNet)"),
    ("unfuse", "unfuse_advice", "advice_to_unfuse_hint",
     "apply_unfuse_hint", "host-verified; no board rung yet"),
    ("choose_implementation", "implementation_advice", "advice_to_kernel_choice",
     "--keep-reference-ops", "host-verified; no board rung yet"),
    ("shard", "shard_advice", "(none)", "MB_SHARD_FACTOR (build-level)",
     "cannot fire: needs multi-core profiles"),
]


def draw(ax):
    for (label, sub, owner), (x, y) in zip(NODES, POS):
        ax.add_patch(FancyBboxPatch(
            (x - 0.60, y - 0.24), 1.20, 0.48,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.8, edgecolor=OWNER_COLOR[owner],
            facecolor="white", zorder=3))
        ax.text(x, y + 0.06, label, ha="center", va="center", fontsize=5.5,
                zorder=4, linespacing=1.15)
        ax.text(x, y - 0.14, sub, ha="center", va="center", fontsize=4.2,
                style="italic", color="#666666", zorder=4)

    for i, j, contract in EDGES:
        (x0, y0), (x1, y1) = POS[i], POS[j]
        if y0 == y1:
            dx = 0.60 if x1 > x0 else -0.60
            a, b = (x0 + dx, y0), (x1 - dx, y1)
            # Top row labels sit above the arrow, bottom row below, so no
            # label ever lands on top of a box.
            off = 0.10 if y0 == 1 else -0.10
            ax.text((x0 + x1) / 2, y0 + off, contract, ha="center",
                    va="bottom" if off > 0 else "top", fontsize=4.3,
                    color="#444444", zorder=4)
        else:
            dy = 0.24 if y1 > y0 else -0.24
            a, b = (x0, y0 + dy), (x1, y1 - dy)
            # Vertical connectors run at the figure's edges; their labels go
            # OUTSIDE, away from the boxes.
            side = 0.70 if x0 > 2.5 else -0.70
            ax.text(x0 + side, 0.5, contract, ha="center", va="center",
                    fontsize=4.3, color="#444444", rotation=90,
                    linespacing=1.2, zorder=4)
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=5, linewidth=0.8,
            color="#444444", shrinkA=0, shrinkB=0, zorder=2))

    handles = [Patch(facecolor="white", edgecolor=OWNER_COLOR[k],
                     linewidth=0.8, label=OWNER_LABEL[k])
               for k in ("mb", "rt", "both")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5),
              frameon=False, ncol=1, handlelength=1.2, handletextpad=0.4,
              borderpad=0.1, labelspacing=0.35, fontsize=4.8)
    ax.set_xlim(-1.05, 6.60); ax.set_ylim(-0.45, 1.45)
    ax.axis("off")


def table(ax):
    ax.axis("off")
    cols = [0.0, 0.20, 0.38, 0.57, 0.78]
    head = ["verb", "producer", "advice -> hint", "consumer", "reached"]
    for x, h in zip(cols, head):
        ax.text(x, 1.0, h, fontsize=5, fontweight="bold", va="top")
    for k, row in enumerate(VERBS):
        y = 0.86 - k * 0.165
        closed = row[2] != "(none)"
        for x, cell in zip(cols, row):
            ax.text(x, y, cell, fontsize=4.6, va="top",
                    color="#000000" if closed else figstyle.C_DEADLINE,
                    family="DejaVu Sans")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.06)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="k1_loop_schematic")
    a = ap.parse_args()
    fig, axes = plt.subplots(
        2, 1, figsize=(figstyle.DOUBLE_COL, 74 * figstyle.MM),
        gridspec_kw={"height_ratios": [1.5, 1.0]})
    draw(axes[0])
    table(axes[1])
    figstyle.panel_label(axes[0], "a", x=0.0, y=0.96)
    figstyle.panel_label(axes[1], "b", x=0.0, y=1.10)
    fig.tight_layout(pad=0.3)
    print(f"wrote {figstyle.save(fig, a.stem, a.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
