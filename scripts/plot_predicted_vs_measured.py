#!/usr/bin/env python3
"""Predicted vs measured per-dispatch service time, and what a bad join looks like.

Panel **a** is the question the profile has to answer: the scheduler planned
each dispatch with a duration taken from a standalone board profile, then the
board ran them all concurrently. How close was it?

Panel **b** is the same trace and the same schedule joined on the trace's raw
`dispatch_id`. That column is a record SLOT, not the IR dispatch id -- it drifts
by the number of zero-cost ops before it (`k1_trace.ir_slot_map`) -- so 44 of
yolov8_nano's 90 dispatches land on an op of a different KIND. Every point in
b is a real measurement of a real dispatch. It is just not the dispatch its x
coordinate names.

b is here because it is what the figure looked like before the join was fixed,
and because "the prediction is bad" and "the labels are wrong" are
indistinguishable from a scatter alone. The methodology is what tells them
apart: the schedule carries the op kind and so does the trace, so the two files
can be asked whether they agree before anything is plotted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import figstyle  # noqa: E402
import k1_trace  # noqa: E402

figstyle.use()


def pairs(schedule, trace_rows):
    """`{model: [(predicted_ms, measured_ms, agrees), ...]}`."""
    by_key = {}
    for r in trace_rows:
        by_key.setdefault(r["dispatch_key"], r)
    out = defaultdict(list)
    for key, v in schedule["dispatches"].items():
        r = by_key.get(key)
        if r is None:
            continue
        pred = float(v["duration"])
        meas = float(r["run_us"]) / 1000.0
        if pred <= 0 or meas <= 0:
            continue
        agrees = f'_{r.get("op", "")}_' in v.get("module_name", "")
        model = (v.get("job_name") or "").rstrip("0123456789")
        out[model].append((pred, meas, agrees))
    return out


def panel(ax, data, title, mark_mismatch):
    lo, hi = 1e9, 0.0
    for model, pts in sorted(data.items()):
        for subset, alpha, edge in (
                ([p for p in pts if p[2] or not mark_mismatch], 0.75, "none"),
                ([p for p in pts if not p[2] and mark_mismatch], 0.9,
                 figstyle.C_DEADLINE)):
            if not subset:
                continue
            xs = [p[0] for p in subset]
            ys = [p[1] for p in subset]
            lo = min(lo, min(xs), min(ys))
            hi = max(hi, max(xs), max(ys))
            ax.scatter(xs, ys, s=3.0, alpha=alpha,
                       color=figstyle.model_color(model),
                       edgecolors=edge, linewidths=0.35,
                       label=model if edge == "none" else None)
    span = [lo * 0.7, hi * 1.4]
    ax.plot(span, span, ls="--", lw=0.6, color="#666666", zorder=0)
    ax.text(span[1], span[1], " perfect", fontsize=4.5, color="#666666",
            ha="right", va="top", rotation=45)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*span); ax.set_ylim(*span)
    ax.set_xlabel("predicted service time (ms)")
    ax.set_ylabel("measured on the K1 (ms)")
    ax.set_title(title, fontsize=6)
    figstyle.despine(ax)


def _median_rel(pts):
    errs = sorted((m - p) / p for p, m, _ in pts)
    return errs[len(errs) // 2] * 100 if errs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ir", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="k1_predicted_vs_measured")
    a = ap.parse_args()

    schedule = json.load(open(a.schedule))
    aligned = pairs(schedule, k1_trace.read(
        a.trace, k1_trace.slot_maps_from_irs(a.ir) if a.ir else None))
    raw = pairs(schedule, k1_trace.read(a.trace))

    fig, axes = plt.subplots(1, 2, figsize=(figstyle.DOUBLE_COL, 62 * figstyle.MM))
    flat_a = [p for pts in aligned.values() for p in pts]
    flat_r = [p for pts in raw.values() for p in pts]
    n_bad = sum(1 for p in flat_r if not p[2])
    panel(axes[0], aligned,
          f"a  joined on the IR dispatch id\n"
          f"median error {_median_rel(flat_a):+.1f}%  (n={len(flat_a)})", False)
    panel(axes[1], raw,
          f"b  joined on the trace's raw id\n"
          f"median error {_median_rel(flat_r):+.1f}%  "
          f"({n_bad} of {len(flat_r)} name a different op)", True)
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.3,
                   borderpad=0.2)
    axes[1].scatter([], [], s=3.0, color="none",
                    edgecolors=figstyle.C_DEADLINE, linewidths=0.35,
                    label="op kind disagrees")
    axes[1].legend(loc="upper left", frameon=False, handletextpad=0.3,
                   borderpad=0.2)
    fig.tight_layout(pad=0.4)
    png = figstyle.save(fig, a.stem, a.out_dir)
    print(f"wrote {png}")
    print(f"  aligned: median {_median_rel(flat_a):+.2f}%  n={len(flat_a)}")
    print(f"  raw    : median {_median_rel(flat_r):+.2f}%  n={len(flat_r)}  "
          f"{n_bad} mislabelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
