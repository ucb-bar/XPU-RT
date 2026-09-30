#!/usr/bin/env python3
"""Figures for the QRB5165 overhead ablation.

The ablation's question is not "how fast" but "does the recommendation
survive" -- a threshold question -- so the primary figure is a REGIME strip:
for each network, the band of call overheads where slicing still wins, and
where it flips to the monolith. The measured 0.37 ms fit is drawn on top,
which is the whole point: four of five recommendations flip within a factor
of ~4 of it.

  qrb5165_overhead_regimes.png  where each recommendation flips
  qrb5165_overhead_cost.png     what the winning set costs as overhead grows

The x axis is categorical (the swept grid points), not continuous: the sweep
includes 0 ms, which a log axis cannot show, and the points are not evenly
spaced. What matters is the ORDER and where the flip lands between them.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"        # validated pair: CVD dE 24.7, contrast pass
INK, MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"
MEASURED = 0.37


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default="docs/Qualcomm/experiments/overhead/overhead_ablation.jsonl")
    ap.add_argument("--out", default="docs/Qualcomm/experiments/overhead")
    ap.add_argument("--rate", type=float, default=0.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    rows = [r for r in rows if r["ns_per_byte"] == a.rate]
    calls = sorted({r["call_ms"] for r in rows})
    by = defaultdict(dict)
    for r in rows:
        by[r["network"]][r["call_ms"]] = r
    order = [n for n in ["vint", "yolov8n", "fused_full", "dronet", "mlp_control"]
             if n in by]

    # ---------- figure 1: regimes ----------
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ys = list(range(len(order)))[::-1]
    for y, net in zip(ys, order):
        for i, c in enumerate(calls):
            r = by[net].get(c)
            if not r:
                continue
            sliced = r["tiles"] > 1
            ax.barh(y, 0.92, left=i - 0.46, height=0.56,
                    color=BLUE if sliced else ORANGE,
                    edgecolor="#fcfcfb", linewidth=1.4, zorder=3)
            if sliced:
                ax.annotate(f"k={r['tiles']}", (i, y), ha="center", va="center",
                            fontsize=7.2, color="white", weight="bold", zorder=4)
        # flip point
        seq = [by[net][c]["tiles"] > 1 for c in calls if c in by[net]]
        flip = next((i for i in range(1, len(seq)) if seq[i - 1] and not seq[i]), None)
        if flip is not None:
            ax.annotate(f"flips at {calls[flip]:g} ms",
                        (flip - 0.5, y), xytext=(6, 17), textcoords="offset points",
                        fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
        elif all(seq):
            ax.annotate("never flips", (len(calls) - 0.5, y), xytext=(10, 0),
                        textcoords="offset points", va="center",
                        fontsize=8.5, color=BLUE, weight="bold")
        elif not any(seq):
            ax.annotate("never slices", (len(calls) - 0.5, y), xytext=(10, 0),
                        textcoords="offset points", va="center",
                        fontsize=8.5, color=ORANGE)

    mi = calls.index(MEASURED) if MEASURED in calls else None
    if mi is not None:
        ax.axvline(mi, color=INK, lw=1.6, ls="--", zorder=5)
        ax.annotate("measured: 0.37 ms + 5.4 ns/byte  (fit on an idle board)",
                    (mi, len(order) - 0.30), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5, color=INK, weight="bold")

    ax.set_yticks(ys); ax.set_yticklabels(order, fontsize=10, color=INK)
    ax.set_xticks(range(len(calls)))
    ax.set_xticklabels([f"{c:g}" for c in calls], fontsize=8.5, color=MUTED)
    ax.set_xlabel("per-tile call overhead (ms)", fontsize=9, color=MUTED)
    ax.set_xlim(-0.6, len(calls) + 1.4)
    ax.set_ylim(-0.7, len(order) - 0.05)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("QRB5165: how much overhead each slicing recommendation survives\n"
                 "blue = cutting still wins (k = tiles) · orange = the monolith wins",
                 fontsize=11.5, loc="left", color=INK, pad=16)
    fig.tight_layout()
    p1 = os.path.join(a.out, "qrb5165_overhead_regimes.png")
    fig.savefig(p1, dpi=160, facecolor="#fcfcfb"); print(f"  wrote {p1}")

    # ---------- figure 2: cost of the winner ----------
    fig2, ax2 = plt.subplots(figsize=(9.0, 4.2))
    placed: list[float] = []
    for net in order:
        xs = [i for i, c in enumerate(calls) if c in by[net]]
        vs = [by[net][calls[i]]["ms"] for i in xs]
        sliced = [by[net][calls[i]]["tiles"] > 1 for i in xs]
        # the line is neutral: it flips mid-course on three of five networks, so
        # colouring the whole line by its first point would state the wrong
        # winner for most of its length. The MARKERS carry the winner.
        ax2.plot(xs, vs, lw=1.8, color=GRID, zorder=2, solid_capstyle="round")
        ax2.scatter([x for x, s in zip(xs, sliced) if s],
                    [v for v, s in zip(vs, sliced) if s],
                    s=26, color=BLUE, zorder=4, edgecolors="white", lw=0.9)
        ax2.scatter([x for x, s in zip(xs, sliced) if not s],
                    [v for v, s in zip(vs, sliced) if not s],
                    s=26, color=ORANGE, zorder=4, edgecolors="white", lw=0.9)
        # label at the LEFT end: the curves converge on the right and three of
        # the five labels collided there, while at 0 ms they span 0.03-25 ms.
        # nudge apart labels whose start values are within ~15% of each other
        # (vint 23.2 vs yolov8n 25.2 sit on top of one another on a log axis)
        near = [v for v in placed if abs(v / vs[0] - 1) < 0.15]
        dy = 9 if len(near) % 2 else (-9 if near else 0)
        placed.append(vs[0])
        ax2.annotate(net, (xs[0], vs[0]), xytext=(-9, dy),
                     textcoords="offset points", va="center", ha="right",
                     fontsize=9, color=INK)
    if mi is not None:
        ax2.axvline(mi, color=INK, lw=1.4, ls="--", zorder=2)
        ax2.annotate("measured 0.37 ms", (mi, ax2.get_ylim()[1]),
                     xytext=(4, -12), textcoords="offset points",
                     fontsize=8.5, color=INK)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(calls)))
    ax2.set_xticklabels([f"{c:g}" for c in calls], fontsize=8.5, color=MUTED)
    ax2.set_xlabel("per-tile call overhead (ms)", fontsize=9, color=MUTED)
    ax2.set_ylabel("best achievable makespan (ms, log)", fontsize=9, color=MUTED)
    ax2.set_xlim(-1.9, len(calls) - 0.4)
    ax2.grid(axis="y", color=GRID, lw=0.7, alpha=0.7); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(GRID)
    ax2.tick_params(colors=MUTED, length=0)
    ax2.set_title("What the winning configuration costs as overhead grows\n"
                  "marker colour is the winner at that point: blue = sliced, orange = monolith",
                  fontsize=11.5, loc="left", color=INK, pad=14)
    fig2.tight_layout()
    p2 = os.path.join(a.out, "qrb5165_overhead_cost.png")
    fig2.savefig(p2, dpi=160, facecolor="#fcfcfb"); print(f"  wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
