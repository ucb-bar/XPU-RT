#!/usr/bin/env python3
"""Before/after figures for the QRB5165 optimizations: precision, slice, branch.

Reads the stage-ladder record and draws two views:

  qrb5165_before_after.png   what each network cost before any choice was
                             available, and what it costs after -- a dumbbell
                             per network on a log axis, since the five span
                             0.03 ms to 73 ms.
  qrb5165_knob_attribution.png
                             which knob removed which milliseconds. Bars are
                             ms SAVED, so a knob that costs time points the
                             other way and is visible as such.

Colors are the validated categorical slots 1-4 (blue/orange/aqua/yellow); the
palette was checked with the dataviz validator (adjacent-pair CVD dE 9.1
protan, normal-vision 22.9). Every bar is direct-labelled, which is also what
the validator's contrast WARN requires.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"
KNOB_COLOR = {"+backend": YELLOW, "+precision": BLUE,
              "+slice": ORANGE, "+branch": AQUA}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    nets = {}
    for r in rows:
        nets.setdefault(r["network"], []).append(r)
    for v in nets.values():
        v.sort(key=lambda r: r["stage"])
    return nets


def short(knob):
    return knob.split(" ")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default="docs/Qualcomm/experiments/stages/stage_ladder.jsonl")
    ap.add_argument("--out", default="docs/Qualcomm/experiments/stages")
    a = ap.parse_args()
    nets = load(a.jsonl)
    order = ["vint", "yolov8n", "fused_full", "dronet", "mlp_control"]
    order = [n for n in order if n in nets]

    # ---------- figure 1: before -> after ----------
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ys = list(range(len(order)))[::-1]
    for y, net in zip(ys, order):
        rs = [r for r in nets[net] if r["ms"] is not None]
        # "after" is the BEST configuration found, not the last stage tried:
        # fused_full's best is 0.452 ms at S3 and it then regresses to 0.503 at
        # S4, and no one would ship the regression. This also keeps this figure
        # consistent with the cumulative one, which is best-so-far by
        # construction.
        before = rs[0]
        after = min(rs[1:], key=lambda r: r["ms"]) if len(rs) > 1 else rs[0]
        b, af = before["ms"], after["ms"]
        ax.plot([b, af], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([b], [y], s=74, color=MUTED, zorder=3, edgecolors="white", lw=1.4)
        ax.scatter([af], [y], s=98, color=BLUE, zorder=4, edgecolors="white", lw=1.4)
        # When before and after are nearly the same value the two labels land on
        # top of each other on a log axis (mlp_control: 0.0295 vs 0.0255), so
        # push them to opposite sides of the pair instead of stacking them.
        tight = (max(b, af) / min(b, af)) < 1.6 if (b and af) else False
        if tight:
            ax.annotate(f"{b:,.3g} ms", (b, y), xytext=(9, 11),
                        textcoords="offset points", ha="left",
                        fontsize=8.5, color=MUTED)
            ax.annotate(f"{af:,.3g} ms", (af, y), xytext=(-9, -20),
                        textcoords="offset points", ha="right",
                        fontsize=9, color=INK, weight="bold")
        else:
            ax.annotate(f"{b:,.3g} ms", (b, y), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=MUTED)
            ax.annotate(f"{af:,.3g} ms", (af, y), xytext=(0, -20),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=INK, weight="bold")
        # name the DECISIVE knob -- the step with the largest gain -- not the
        # last stage. yolov8n ends at +slice but slicing HURT it (0.89x); its
        # win came from +backend. Labelling the final row would credit the
        # wrong knob on three of five networks.
        best_k, best_gain = None, 1.0
        for q in range(1, len(rs)):
            g = rs[q - 1]["ms"] / rs[q]["ms"] if rs[q]["ms"] else 1.0
            if g > best_gain:
                best_gain, best_k = g, short(rs[q]["knob"])
        if b and af:
            tag = f"{b/af:.2f}x" + (f"  mostly {best_k}" if best_k else "  (no knob helped)")
            ax.annotate(tag, (max(b, af), y), xytext=(14, 0),
                        textcoords="offset points", va="center",
                        fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("critical path / makespan (ms, log scale)", fontsize=9, color=MUTED)
    ax.set_xlim(0.015, 400)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("QRB5165: before and after the slicing/placement optimizations\n"
                 "grey = no choice available · blue = best configuration found",
                 fontsize=11.5, loc="left", color=INK, pad=14)
    fig.tight_layout()
    p1 = os.path.join(a.out, "qrb5165_before_after.png")
    fig.savefig(p1, dpi=160, facecolor="#fcfcfb")
    print(f"  wrote {p1}")

    # ---------- figure 2: cumulative speedup ----------
    # Each knob's segment is how much it added to the RUNNING speedup over the
    # baseline, so the bar can only grow. A knob that makes things worse -- and
    # two of them do, yolov8n's +slice and fused_full's +branch -- contributes
    # ZERO rather than a drop, because you would simply not adopt it. The
    # running value is therefore best-so-far, which is what a feedback loop
    # would actually keep.
    fig2, ax2 = plt.subplots(figsize=(9.6, 4.6))
    knobs = ["+backend", "+precision", "+slice", "+branch"]
    for j, net in enumerate(order):
        rs = [r for r in nets[net] if r["ms"] is not None]
        base = rs[0]["ms"]
        best = base
        bottom, segs = 1.0, []
        for r in rs[1:]:
            if r["ms"] < best:                 # only an improvement moves the bar
                best = r["ms"]
            cum = base / best
            segs.append((short(r["knob"]), cum - bottom))
            bottom = cum
        run = 1.0
        for k, inc in segs:
            if inc <= 1e-9:
                continue
            ax2.bar(j, inc, 0.56, bottom=run, color=KNOB_COLOR[k],
                    edgecolor="#fcfcfb", lw=1.4, zorder=3)
            if inc / bottom > 0.06:
                ax2.annotate(f"+{inc:.2f}x", (j, run + inc / 2), ha="center",
                             va="center", fontsize=8, color="white", weight="bold",
                             zorder=4)
            run += inc
        ax2.annotate(f"{run:.2f}x", (j, run), xytext=(0, 6),
                     textcoords="offset points", ha="center",
                     fontsize=10, color=INK, weight="bold")
        dropped = [k for k, inc in segs if inc <= 1e-9]
        if dropped:
            ax2.annotate("no gain: " + ", ".join(dropped), (j, run),
                         xytext=(0, 20), textcoords="offset points",
                         ha="center", fontsize=7.4, color=MUTED)
    ax2.axhline(1.0, color=INK, lw=1.0)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, fontsize=10, color=INK)
    ax2.set_ylabel("cumulative speedup over the no-choice baseline (x)",
                   fontsize=9, color=MUTED)
    ax2.set_ylim(0, 12.9)
    ax2.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(GRID)
    ax2.tick_params(colors=MUTED, length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KNOB_COLOR[k]) for k in knobs]
    # legend sits ABOVE the axes: dronet's bar reaches 11.5x and its caption
    # was landing inside an in-axes legend
    ax2.legend(handles, knobs, fontsize=8.5, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 1.005))
    ax2.set_title("Cumulative speedup as each optimization is added\n"
                  "segments stack in ladder order; a knob that does not help "
                  "contributes nothing rather than a drop",
                  fontsize=11.5, loc="left", color=INK, pad=34)
    fig2.tight_layout()
    p2 = os.path.join(a.out, "qrb5165_knob_attribution.png")
    fig2.savefig(p2, dpi=160, facecolor="#fcfcfb")
    print(f"  wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
