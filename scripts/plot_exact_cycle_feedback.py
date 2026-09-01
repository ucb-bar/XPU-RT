#!/usr/bin/env python3
"""Render the exact-cycle proof and repeated K1 corroboration figure."""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import figstyle  # noqa: E402
import plot_k1_evolution as gantt  # noqa: E402
import schedule_trace  # noqa: E402

figstyle.use()


def _json(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--board-result", required=True)
    ap.add_argument("--out", required=True, help="output stem")
    args = ap.parse_args()
    result = _json(args.result)
    board = _json(args.board_result)
    original = _json(os.path.join(_REPO, result["inputs"]["original_schedule"]["path"]))
    feedback = _json(os.path.join(_REPO, result["inputs"]["feedback_schedule"]["path"]))
    workload = _json(os.path.join(_REPO, result["inputs"]["original_workload"]["path"]))
    periods = {name: float(spec["period"])
               for name, spec in workload["networks"].items()}
    known = set(periods)
    colours = {name: figstyle.model_color(name) for name in known}
    cores = gantt.cores_from_schedule({
        **{f"o:{k}": v for k, v in original["dispatches"].items()},
        **{f"f:{k}": v for k, v in feedback["dispatches"].items()},
    })

    fig = plt.figure(figsize=(figstyle.DOUBLE_COL, 142 * figstyle.MM))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.15, 1.15, 0.90),
                          hspace=0.58, wspace=0.35)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    proof = result["proof"]
    panels = [
        ("A  Original implementation graph — "
         f"global optimum = {proof['original_response_floor_ms']:.3f} ms",
         original),
        ("B  After XPU-RT → ModelBlaster feedback — "
         f"global optimum = {proof['feedback_response_ms']:.3f} ms",
         feedback),
    ]
    for ax, (title, schedule) in zip(axes, panels):
        rows = schedule_trace.trace_rows_from_schedule(schedule)
        gantt.draw_gantt_axis(
            ax, rows, schedule["dispatches"], cores=cores, window_ms=100.0,
            colours=colours, periods=periods, deadline_model="mlp_control",
            known=known, impl_hatch=True, repeat_frame=True)
        ax.set_title(title, loc="left", fontsize=7, fontweight="bold")
        ax.set_ylabel("physical K1 cores")
    axes[1].set_xlabel("Time in the exact repeating cycle (ms)")
    axes[1].text(
        0.995, -0.26,
        "bar height = physical harts reserved by that dispatch",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=4.7,
        color="0.35")

    ax = fig.add_subplot(gs[2, 0])
    pred = [proof["original_response_floor_ms"], proof["feedback_response_ms"]]
    bars = ax.bar([0, 1], pred, width=0.58,
                  color=[figstyle.C_MUTED, figstyle.BLUE], edgecolor="white")
    for bar, value in zip(bars, pred):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18,
                f"{value:.3f} ms", ha="center", va="bottom", fontsize=6,
                fontweight="bold")
    ax.set_xticks([0, 1], ["original", "feedback"])
    ax.set_ylabel("worst critical response (ms)")
    ax.set_ylim(0, max(pred) * 1.24)
    ax.set_title("C  Certified global optima", loc="left", fontweight="bold")
    ax.text(0.52, 0.65, f"−{proof['improvement_pct']:.2f}%",
            transform=ax.transAxes, ha="center",
            color=figstyle.BLUE, fontweight="bold", fontsize=8)
    ax.text(0.02, 0.97, "feasible schedule = analytic lower bound",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.2,
            color="0.35")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[2, 1])
    samples = [
        board["original"]["aggregate"]["critical_response_ms"]["samples"],
        board["feedback"]["aggregate"]["critical_response_ms"]["samples"],
    ]
    bp = ax.boxplot(samples, positions=[0, 1], widths=0.52, patch_artist=True,
                    showfliers=False, medianprops={"color": "white", "lw": 1.2})
    for patch, colour in zip(bp["boxes"], [figstyle.C_MUTED, figstyle.BLUE]):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    for i, values in enumerate(samples):
        offsets = ([0.0] if len(values) == 1 else
                   [-0.14 + j * 0.28 / (len(values) - 1)
                    for j in range(len(values))])
        ax.scatter([i + offsets[j] for j in range(len(values))], values,
                   s=8, color="black", zorder=3, linewidths=0)
    ax.set_xticks([0, 1], ["original", "feedback"])
    ax.set_ylabel("measured critical response (ms)")
    runner = board["stdout_audits"]["original"][0]["observed_runner_policy"]
    policy = runner["policy"] if runner else "unreported policy"
    ax.set_title(
        f"D  {board['runs_per_phase']} complete K1 runs per phase ({policy})",
        loc="left", fontweight="bold")
    cmp = board["comparison"]
    all_samples = samples[0] + samples[1]
    ax.set_ylim(min(all_samples) - 0.45, max(all_samples) + 1.25)
    misses = (board["original"]["aggregate"]["total_deadline_misses"]
              + board["feedback"]["aggregate"]["total_deadline_misses"])
    separation = ("all feedback runs < original minimum"
                  if cmp["feedback_max_below_original_min"] else
                  "run ranges overlap")
    ax.text(
        0.5, 0.97,
        f"median −{cmp['median_critical_improvement_pct']:.2f}% · "
        f"exact p={cmp['exact_mann_whitney_less_p']:.2g} · {misses} misses\n"
        f"{separation}",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.4)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=colours[m], label=m) for m in sorted(known)]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                         label="IME linear/matmul"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Feedback expands the attainable schedule—not just the solver search\n"
        "Same 100 ms work: 5 MLP + 5 fused-control + 3 DroNet + 1 FFN jobs "
        "(178 dispatches); the cycle repeats exactly.",
        x=0.02, y=0.995, ha="left", fontsize=8.5, fontweight="bold")
    _save(fig, args.out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
