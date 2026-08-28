#!/usr/bin/env python3
"""What a granularity change actually did, measured on the K1.

The B3 rung split DroNet's dispatch 0 -- a `conv2d_s8` with OC=32 -- into two
OC=16 tiles. That is the only granularity rewrite this project has executed on
hardware, and until now it had no figure: its directory was entirely JSON,
against a standing rule that every rung emits a Gantt.

Three panels, because the interesting part is not the headline:

  a  per-dispatch measured cost, before and after. One bar becomes two. This is
     the granularity change itself, not a summary of it.
  b  what the split cost and what it could buy. The tiles sum to MORE than the
     dispatch they replaced, because each re-reads the whole input; the only
     gain is the critical path, and only if the two land on different harts.
  c  the methodology trap. The same split measures -0.2% or +13.7% depending on
     which baseline it is compared against, and one of those baselines is
     stale. Panel c is why this file compares against gen_mb and not against
     the backup sitting next to the split profile.

Everything plotted is MEASURED. Nothing here is predicted or modelled.

Usage:
    python3 scripts/plot_granularity_evolution.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM

# Okabe-Ito: colourblind-safe, and it still reads in greyscale.
C_BASE = "#0072B2"   # blue    - baseline
C_SPLIT = "#D55E00"  # vermill - the split tiles
C_MUTE = "#999999"
C_WARN = "#CC79A7"   # purple  - the stale comparison

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6,
    "axes.labelsize": 6, "axes.titlesize": 7,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})

_B3 = os.path.join(REPO, "artifacts", "k1_run", "round_B3_dronet_split")

#: The three profiles this figure rests on. `stale` is deliberately included:
#: it is the one that must NOT be used for the comparison, and panel c shows
#: why. It sits inside the B3 directory, which is exactly what makes it a trap.
PROFILES = {
    "baseline": os.path.join(
        REPO, "gen", "profile_mb", "rvv_x60", "spacemit_x60", "dronet",
        "dronet.int8", "dronet_spacemit_x60_rvv_x60_dronet.int8", "topo_0",
        "results.csv"),
    "split": os.path.join(
        _B3, "split_profile", "dronet_spacemit_x60_rvv_x60_dronet.int8",
        "topo_0", "results.csv"),
    "stale": os.path.join(
        _B3, "baseline_profile_backup",
        "dronet_spacemit_x60_rvv_x60_dronet.int8", "topo_0", "results.csv"),
}


def load(path):
    """[(dispatch_id, op, ms, shape)], in dispatch order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append((int(r["dispatch_id"]), r["op"],
                    float(r["mean_time"]), r.get("shape", "")))
    return sorted(out, key=lambda t: t[0])


def _oc(shape: str) -> str:
    for kv in shape.split(";"):
        if kv.startswith("OC="):
            return kv[3:]
    return "?"


def panel_a(ax, base, split):
    """One segment becomes two. Laid out in TIME, not by dispatch index.

    An index-aligned bar chart would be wrong here and wrong in a way that
    looks like a result: the split renumbers every later dispatch, so baseline
    `i` sits against split `i+1` and all twenty of them appear to have changed
    cost. That is the `dispatch_id` join hazard this repo documents in
    `diff_dispatch_graph.py`, drawn instead of computed.

    So each row is a timeline and each dispatch a segment whose WIDTH is its
    measured cost. The tiling change reads as one segment becoming two, and the
    total-length difference is the +13.7% -- both without any join at all.
    """
    rows = [("baseline\n21 dispatches", base, C_BASE),
            ("split ×2\n22 dispatches", split, C_SPLIT)]
    for y, (label, prof, colour) in enumerate(rows):
        x = 0.0
        for k, (_did, op, ms, shape) in enumerate(prof):
            split_tile = (y == 1 and k < 2)
            ax.barh(y, ms, left=x, height=0.5,
                    color=C_SPLIT if split_tile else colour,
                    edgecolor="black" if split_tile else "white",
                    linewidth=0.7 if split_tile else 0.4,
                    hatch="///" if split_tile else None)
            if (y == 0 and k == 0) or split_tile:
                # Below the bar, not inside it: rotated text inside a 1.5 mm
                # segment clips, and the label is the point of the panel.
                ax.text(x + ms / 2, y - 0.34, f"OC={_oc(shape)}", ha="center",
                        va="top", fontsize=4.8, color=colour,
                        fontweight="bold")
            x += ms
        ax.text(x + 0.12, y, f"{x:.3f} ms", va="center", fontsize=5.5,
                fontweight="bold", color=colour)
    ax.annotate("", xy=(3.08, 1.32), xytext=(1.945, 0.68),
                arrowprops=dict(arrowstyle="->", lw=0.6, color="black",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.0, 1.55, "one conv2d_s8 OC=32 becomes two OC=16 tiles;\n"
                       "every later dispatch is unchanged work, renumbered",
            fontsize=5, ha="left")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("cumulative measured time (ms) — each segment is one dispatch")
    ax.set_title("a   DroNet's dispatch stream, before and after the split",
                 loc="left", fontweight="bold")
    ax.set_ylim(-0.85, 2.1)
    ax.set_xlim(0, 10.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_b(ax, base, split):
    unsplit = base[0][2]
    t0, t1 = split[0][2], split[1][2]
    bars = [("unsplit\nOC=32", unsplit, C_BASE),
            ("tiles, summed\n(total work)", t0 + t1, C_SPLIT),
            ("tiles, critical path\n(if on 2 harts)", max(t0, t1), C_MUTE)]
    ax.bar(range(3), [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    for i, (_lab, v, _c) in enumerate(bars):
        ax.text(i, v + 0.06, f"{v:.3f}", ha="center", fontsize=5)
    ax.text(1, (t0 + t1) / 2, f"{100 * ((t0 + t1) / unsplit - 1):+.0f}%",
            ha="center", va="center", fontsize=6, color="white",
            fontweight="bold")
    ax.text(2, max(t0, t1) / 2, f"{100 * (max(t0, t1) / unsplit - 1):+.0f}%",
            ha="center", va="center", fontsize=6, color="white",
            fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("measured time (ms)")
    ax.set_title("b   the split costs more work than it saves latency",
                 loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, (t0 + t1) * 1.25)


def panel_c(ax, totals):
    """Same split, two answers, depending on the baseline."""
    stale, good, sp = totals["stale"], totals["baseline"], totals["split"]
    ax.bar([0, 1], [stale, sp], width=0.55, color=[C_WARN, C_SPLIT])
    ax.bar([2.6, 3.6], [good, sp], width=0.55, color=[C_BASE, C_SPLIT])
    for x, v in [(0, stale), (1, sp), (2.6, good), (3.6, sp)]:
        ax.text(x, v + 0.12, f"{v:.3f}", ha="center", fontsize=5)
    ax.text(0.5, stale * 1.14, f"{100 * (sp / stale - 1):+.1f}%  spurious",
            ha="center", fontsize=5.5, color=C_WARN, fontweight="bold")
    ax.text(3.1, sp * 1.14, f"{100 * (sp / good - 1):+.1f}%  real",
            ha="center", fontsize=5.5, color=C_BASE, fontweight="bold")
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(["stale\nbaseline", "split", "correct\nbaseline",
                        "split"])
    ax.set_ylabel("DroNet total (ms)")
    ax.set_title("c   the same split, measured against two baselines",
                 loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(stale, sp) * 1.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=os.environ.get("XPURT_FIGURE_DIR")
                    or os.path.join(REPO, "out", "figures"))
    a = ap.parse_args()

    missing = [k for k, p in PROFILES.items() if not os.path.exists(p)]
    if missing:
        print("missing measured profiles, refusing to plot: "
              + ", ".join(f"{k} ({PROFILES[k]})" for k in missing),
              file=sys.stderr)
        return 2

    prof = {k: load(p) for k, p in PROFILES.items()}
    totals = {k: sum(r[2] for r in v) for k, v in prof.items()}

    os.makedirs(a.out_dir, exist_ok=True)
    fig = plt.figure(figsize=(DOUBLE_COL, 105 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.28)
    panel_a(fig.add_subplot(gs[0, :]), prof["baseline"], prof["split"])
    panel_b(fig.add_subplot(gs[1, 0]), prof["baseline"], prof["split"])
    panel_c(fig.add_subplot(gs[1, 1]), totals)

    for ext in ("png", "pdf"):
        out = os.path.join(a.out_dir, f"k1_granularity_b3.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {out}")
    plt.close(fig)

    print(f"\nbaseline {totals['baseline']:.4f} ms over {len(prof['baseline'])} "
          f"dispatches")
    print(f"split    {totals['split']:.4f} ms over {len(prof['split'])} "
          f"dispatches  "
          f"({100 * (totals['split'] / totals['baseline'] - 1):+.1f}%)")
    print(f"stale    {totals['stale']:.4f} ms -- NOT a valid comparison; it "
          f"predates the _zfh_zvfh march change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
