#!/usr/bin/env python3
"""Full experiment log + figures for the QRB5165 feedback-stage ladder.

`flowc_feedback_stages.py` prints the summary; this writes the record behind
it. For every stage of every network it logs the slice set used, the tile-level
cells that were available, the assignment chosen, and the cells that were
*rejected* and why -- a compose failure is as much a result as a timing, and
it is the reason a stage sometimes cannot move.

Outputs
  <out>/stage_ladder.md     human-readable log, one section per network
  <out>/stage_ladder.jsonl  one record per (network, stage), machine-readable
  <out>/stage_ladder.png    the ladder, per network
  <out>/stage_knobs.png     which knob moved which network
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qnn_models", "slicing_study"))

import analyze                      # noqa: E402
import flowc_feedback_stages as fs  # noqa: E402

STAGES = ["S0", "S1", "S2", "S3", "S4"]
KNOB = {"S0": "monolith", "S1": "+backend", "S2": "+precision",
        "S3": "+slice", "S4": "+branch"}


def cells_table(e: dict) -> list[dict]:
    out = []
    for t in sorted(x["index"] for x in e["tiles"]):
        row = {"tile": t, "cells": {}, "fails": {}}
        for (ti, b, p), v in sorted(e["cell"].items()):
            if ti == t:
                row["cells"][f"{b}@{p}"] = round(v / 1000.0, 3)
        for (ti, b, p), why in sorted(e.get("fails", {}).items()):
            if ti == t:
                row["fails"][f"{b}@{p}"] = (why or "")[:70]
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", default="vint,yolov8n,fused_full,dronet,mlp_control")
    ap.add_argument("--out", default="results/flowc_stages")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pooled = analyze.pool(analyze.load())
    nets = [n.strip() for n in a.networks.split(",") if n.strip()]
    log, records = [], []

    log.append("# QRB5165 feedback-stage ladder — full experiment log\n")
    log.append("Each stage adds exactly one degree of freedom, so the delta "
               "between two rows is attributable to that knob alone. Costs are "
               "pooled medians from `qnn_models/slicing_study/experiments.jsonl` "
               "(the measured sweep), not a re-run.\n")
    log.append("| stage | knob | freedom added |\n|---|---|---|")
    for s in STAGES:
        log.append(f"| {s} | `{KNOB[s]}` | "
                   f"{'the fallback, no choice at all' if s=='S0' else 'see below'} |")
    log.append("")

    for net in nets:
        rows = fs.stages(net, pooled)
        if not rows:
            continue
        base = next((r["ms"] for r in rows if r["stage"] == "S0"), None)
        log.append(f"\n## {net}\n")
        log.append("| stage | knob | ms | vs S0 | step | tiles | assignment |")
        log.append("|---|---|---:|---:|---:|---:|---|")
        prev = None
        for r in rows:
            ms = r["ms"]
            cum = f"{base/ms:.2f}x" if (base and ms) else "—"
            step = f"{prev/ms:.2f}x" if (prev and ms) else "—"
            log.append(f"| {r['stage']} | `{r['knob']}` | "
                       f"{('%.3f' % ms) if ms is not None else 'n/a'} | {cum} | {step} | "
                       f"{r['tiles']} | `{r['detail']}` |")
            if ms:
                prev = ms
            records.append({"network": net, **r})
        s4 = next((r for r in rows if r["stage"] == "S4"), None)
        if s4 and s4.get("serial_ms"):
            log.append(f"\nConcurrency is worth **{s4['serial_ms'] - s4['ms']:+.3f} ms** "
                       f"({s4['serial_ms']:.3f} serial → {s4['ms']:.3f} overlapped).\n")

        # the evidence behind the winning rows
        for r in rows:
            e = pooled.get(r["label"])
            if not e or r["stage"] not in ("S3", "S4"):
                continue
            log.append(f"\n### {net} {r['stage']} evidence — `{r['label']}`\n")
            log.append(f"cut: `{e['cut']}`  ·  sweeps: {e['sweeps']}\n")
            log.append("| tile | measured cells (ms) | rejected |")
            log.append("|---|---|---|")
            for c in cells_table(e):
                cells = ", ".join(f"`{k}`={v}" for k, v in c["cells"].items()) or "—"
                fails = ", ".join(f"`{k}`: {w}" for k, w in c["fails"].items()) or "—"
                log.append(f"| t{c['tile']} | {cells} | {fails} |")

    md = os.path.join(a.out, "stage_ladder.md")
    with open(md, "w") as f:
        f.write("\n".join(log) + "\n")
    jl = os.path.join(a.out, "stage_ladder.jsonl")
    with open(jl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {md}")
    print(f"  wrote {jl}  ({len(records)} records)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (no matplotlib: {exc}) — figures skipped")
        return 0

    by_net: dict[str, list] = {}
    for r in records:
        by_net.setdefault(r["network"], []).append(r)

    # fig 1: the ladder
    fig, axes = plt.subplots(1, len(by_net), figsize=(3.1 * len(by_net), 4.0),
                             sharey=False)
    if len(by_net) == 1:
        axes = [axes]
    for ax, (net, rs) in zip(axes, by_net.items()):
        xs = [r["stage"] for r in rs if r["ms"] is not None]
        ys = [r["ms"] for r in rs if r["ms"] is not None]
        ax.step(range(len(xs)), ys, where="mid", color="#1f77b4", lw=1.6)
        ax.scatter(range(len(xs)), ys, s=26, color="#1f77b4", zorder=3)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{y:.3g}", (i, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_title(net, fontsize=10, loc="left")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("critical path / makespan (ms, log)")
    fig.suptitle("QRB5165 feedback-stage ladder — one knob per step",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p1 = os.path.join(a.out, "stage_ladder.png")
    fig.savefig(p1, dpi=150)
    print(f"  wrote {p1}")

    # fig 2: which knob moved which network
    fig2, ax = plt.subplots(figsize=(7.6, 3.8))
    knobs = ["S1", "S2", "S3", "S4"]
    colors = {"S1": "#2ca02c", "S2": "#9467bd", "S3": "#ff7f0e", "S4": "#d62728"}
    width = 0.2
    for i, k in enumerate(knobs):
        gains, labels = [], []
        for net, rs in by_net.items():
            seq = [r for r in rs if r["ms"] is not None]
            idx = next((j for j, r in enumerate(seq) if r["stage"] == k), None)
            g = (seq[idx - 1]["ms"] / seq[idx]["ms"]) if (idx and idx > 0) else 1.0
            gains.append(g)
            labels.append(net)
        ax.bar([x + i * width for x in range(len(gains))], gains, width,
               label=f"{k} {KNOB[k]}", color=colors[k], edgecolor="black", lw=0.4)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks([x + 1.5 * width for x in range(len(by_net))])
    ax.set_xticklabels(list(by_net), fontsize=9)
    ax.set_ylabel("step speedup (x)")
    ax.set_title("Which knob pays, per network — 1.0 = no change, <1 = the knob hurts",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig2.tight_layout()
    p2 = os.path.join(a.out, "stage_knobs.png")
    fig2.savefig(p2, dpi=150)
    print(f"  wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
