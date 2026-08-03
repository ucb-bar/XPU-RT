"""Figures for the freshness-validity evaluation.

    python -m benchmarks.freshness_eval.plot \
        --input results/freshness_eval --output figures/freshness_eval

Reads only the CSVs the sweep wrote, so figures are reproducible from the
artifacts alone and never from a live solve:

  aggregate.csv        one row per (policy, B, phi, seed)
  per_invocation.csv   one row per consumer invocation per edge
  intervals.csv        every invocation interval, incl. the soft interfering work

Three figures:

  1 deadline vs freshness validity against contention -- the headline. Does
    deadline success stay high while freshness validity falls?
  2 diagnostic timeline for the largest-divergence operating point, so the
    stale-but-on-time failure is visible as a mechanism rather than a rate.
  3 phi sensitivity heat map, so the result does not rest on one window.

Colour follows the reference palette in the dataviz skill. Categorical hues are
assigned in fixed order and never cycled; the oracle is drawn as a neutral
dashed upper bound rather than a fifth category, because it is not a deployable
policy. Every series also carries a distinct marker and the lines are directly
labelled, which is the secondary encoding that keeps identity from resting on
colour alone (two palette hues sit below 3:1 against the surface).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# --- palette (validated: node scripts/validate_palette.js, light surface) ---
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order
MARKERS = ["o", "s", "^", "D"]
NEUTRAL = "#4a4a47"

RATE_COLOR = {
    "deadline_success_rate": SERIES[0],
    "freshness_success_rate": SERIES[1],
    "output_valid_rate": SERIES[2],
}
RATE_MARKER = {
    "deadline_success_rate": MARKERS[0],
    "freshness_success_rate": MARKERS[1],
    "output_valid_rate": MARKERS[2],
}
RATE_LABEL = {
    "deadline_success_rate": "deadline-valid",
    "freshness_success_rate": "freshness-valid",
    "output_valid_rate": "output-valid (both)",
}

# Sequential single hue, light -> dark, from the palette's blue ramp.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

# Status colours, reserved -- never reused as a series.
STATUS_CRITICAL = "#d03b3b"
STATUS_WARNING = "#fab219"

ORACLE = "oracle"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _save(fig, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    return paths


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v: str) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# --- Plot 1: deadline vs freshness validity against contention -------------


def plot_validity_vs_contention(agg: List[Dict], out_dir: str, phi: float) -> List[str]:
    policies = [p for p in dict.fromkeys(r["policy"] for r in agg) if p != ORACLE]
    rows = [r for r in agg if abs(_f(r["freshness_window"]) - phi) < 1e-6]

    fig, axes = plt.subplots(
        1, len(policies), figsize=(3.5 * len(policies), 3.6), sharey=True
    )
    if len(policies) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    for ax, pol in zip(axes, policies):
        _style(ax)
        sel = [r for r in rows if r["policy"] == pol]
        bursts = sorted({int(_f(r["contention_level"])) for r in sel})
        for rate in ("deadline_success_rate", "freshness_success_rate", "output_valid_rate"):
            mean, lo, hi = [], [], []
            for b in bursts:
                vals = [
                    _f(r[rate]) for r in sel
                    if int(_f(r["contention_level"])) == b and _f(r[rate]) is not None
                ]
                if not vals:
                    mean.append(None); lo.append(None); hi.append(None); continue
                mean.append(sum(vals) / len(vals)); lo.append(min(vals)); hi.append(max(vals))
            xs = [b for b, m in zip(bursts, mean) if m is not None]
            ys = [m for m in mean if m is not None]
            if not xs:
                continue
            # Seed spread as a band; with deterministic policies it collapses to
            # the line, which is itself the useful signal.
            if any(h is not None and l is not None and h > l for l, h in zip(lo, hi)):
                ax.fill_between(
                    xs,
                    [l for l in lo if l is not None],
                    [h for h in hi if h is not None],
                    color=RATE_COLOR[rate], alpha=0.15, linewidth=0, zorder=1,
                )
            # output_valid = deadline AND freshness. When deadlines are never
            # missed the two coincide exactly, so draw output_valid narrower and
            # dashed on top: otherwise it hides the freshness line completely
            # and the figure looks like it is missing a series.
            is_combined = rate == "output_valid_rate"
            ax.plot(
                xs, ys, color=RATE_COLOR[rate],
                linewidth=1.6 if is_combined else 2.6,
                linestyle=(0, (4, 2)) if is_combined else "-",
                marker=RATE_MARKER[rate], markersize=7 if is_combined else 9,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=RATE_LABEL[rate], zorder=4 if is_combined else 3,
            )
            # Direct label on the last point: the relief the contrast WARN needs.
            # Nudge the combined series so a coincident pair stays legible.
            ax.annotate(
                f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(7, -9 if is_combined else 4), fontsize=8,
                color=INK_MUTED, va="center",
            )
        ax.set_title(pol.replace("_", " "), fontsize=10, color=INK)
        ax.set_xlabel("YOLO burst size B (instances/epoch)", fontsize=9, color=INK_MUTED)
        ax.set_xticks(bursts)
        ax.set_ylim(-0.04, 1.12)

    axes[0].set_ylabel("fraction of control outputs", fontsize=9, color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, 1.06), labelcolor=INK,
    )
    fig.suptitle(
        f"Local deadline success does not imply valid output   "
        f"(freshness window φ = {phi:.1f} ms)",
        fontsize=11, color=INK, y=1.16,
    )
    fig.tight_layout()
    return _save(fig, out_dir, "plot1_deadline_vs_freshness")


# --- Plot 2: diagnostic timeline -------------------------------------------


def plot_diagnostic_timeline(
    per_inv: List[Dict], intervals: List[Dict], out_dir: str,
    policy: str, burst: int, seed: int, phi: float, epoch_ms: float,
) -> List[str]:
    recs = [
        r for r in per_inv
        if r["policy"] == policy
        and int(_f(r["contention_level"])) == burst
        and int(_f(r["seed"])) == seed
        and abs(_f(r["freshness_window"]) - phi) < 1e-6
    ]
    ivs = [
        r for r in intervals
        if r["policy"] == policy
        and int(_f(r["contention_level"])) == burst
        and int(_f(r["seed"])) == seed
    ]
    if not recs or not ivs:
        return []

    tasks = list(dict.fromkeys(r["task"] for r in ivs))
    # Producer first, then soft work, then the consumer at the bottom.
    consumer = recs[0]["consumer_task"]
    producer = recs[0]["producer_task"]
    order = [producer] + [t for t in tasks if t not in (producer, consumer)] + [consumer]
    ypos = {t: len(order) - 1 - i for i, t in enumerate(order)}
    color_of = {t: SERIES[i % len(SERIES)] for i, t in enumerate(order)}

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    fig.patch.set_facecolor(SURFACE)
    _style(ax); _style(ax2)

    # execution bars. A minimum drawn width keeps sub-millisecond instances
    # visible on a 300 ms axis -- the controller is 0.546 ms and would otherwise
    # render as an invisible hairline, hiding the very task under study.
    span = max(_f(r["end_time"]) for r in ivs)
    min_w = span * 0.0018
    for r in ivs:
        t = r["task"]
        s, e = _f(r["start_time"]), _f(r["end_time"])
        ax.add_patch(Rectangle(
            (s, ypos[t] - 0.3), max(e - s, min_w), 0.6,
            facecolor=color_of[t], edgecolor=SURFACE, linewidth=0.8, zorder=3,
        ))

    # producer sample times, and the consumed-instance link
    for r in recs:
        cs, ce = _f(r["consumer_start_time"]), _f(r["consumer_end_time"])
        stale = r["freshness_valid"] in ("False", "false", False)
        no_prod = r["invalid_reason"] == "no_completed_producer"
        ps = _f(r["producer_sample_time"])
        if ps is not None:
            ax.plot(
                [ps, ce], [ypos[producer] - 0.42, ypos[consumer] + 0.32],
                color=STATUS_CRITICAL if stale else INK_MUTED,
                linewidth=1.4 if stale else 0.6,
                alpha=0.9 if stale else 0.35, zorder=2,
            )
            ax.plot([ps], [ypos[producer] - 0.42], marker="v", markersize=5,
                    color=INK_MUTED, zorder=4)
        if no_prod:
            ax.plot([cs], [ypos[consumer] - 0.5], marker="x", markersize=7,
                    color=STATUS_WARNING, markeredgewidth=2, zorder=5)
        elif stale:
            ax.plot([ce], [ypos[consumer] + 0.42], marker="v", markersize=6,
                    color=STATUS_CRITICAL, zorder=5)

    ax.axvline(epoch_ms, color=INK_MUTED, linestyle=":", linewidth=1.4)
    # Label at the bottom, clear of the legend in the upper right.
    ax.annotate("epoch end", (epoch_ms, -0.65), fontsize=8,
                color=INK_MUTED, ha="right", xytext=(-4, 0),
                textcoords="offset points")
    ax.set_yticks([ypos[t] for t in order])
    ax.set_yticklabels([t.replace("_", " ") for t in order], fontsize=9, color=INK)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_title(
        f"{policy.replace('_', ' ')}, B={burst}, φ={phi:.1f} ms   "
        f"— red links mark control outputs computed from a stale input; "
        f"every one of them met its own deadline",
        fontsize=10, color=INK, pad=28,
    )
    ax.legend(
        handles=[
            Patch(facecolor=color_of[t], label=t.replace("_", " ")) for t in order
        ] + [
            plt.Line2D([], [], color=STATUS_CRITICAL, linewidth=1.6,
                       label="stale input consumed"),
            plt.Line2D([], [], color=STATUS_WARNING, marker="x", linestyle="none",
                       markeredgewidth=2, label="no completed producer"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8, frameon=False,
        ncol=5, labelcolor=INK,
    )

    # input age against the window
    pts = [(_f(r["consumer_end_time"]), _f(r["input_age_at_output"])) for r in recs
           if _f(r["input_age_at_output"]) is not None]
    pts.sort()
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax2.plot(xs, ys, color=SERIES[0], linewidth=2.0, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                 label="input age at output")
        over = [(x, y) for x, y in pts if y > phi]
        if over:
            ax2.plot([p[0] for p in over], [p[1] for p in over], linestyle="none",
                     marker="o", markersize=7, color=STATUS_CRITICAL, zorder=4,
                     label="stale")
        ax2.axhline(phi, color=STATUS_CRITICAL, linestyle="--", linewidth=1.6,
                    zorder=2)
        ax2.annotate(f"φ = {phi:.1f} ms", (xs[0], phi), fontsize=8,
                     color=STATUS_CRITICAL, va="bottom",
                     xytext=(2, 3), textcoords="offset points")
    ax2.axvline(epoch_ms, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax2.set_xlabel("time (ms)", fontsize=9, color=INK_MUTED)
    ax2.set_ylabel("input age (ms)", fontsize=9, color=INK_MUTED)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)

    fig.tight_layout()
    return _save(fig, out_dir, "plot2_diagnostic_timeline")


# --- Plot 3: phi sensitivity ----------------------------------------------


def plot_phi_sensitivity(agg: List[Dict], out_dir: str) -> List[str]:
    policies = [p for p in dict.fromkeys(r["policy"] for r in agg) if p != ORACLE]
    phis = sorted({_f(r["freshness_window"]) for r in agg})
    bursts = sorted({int(_f(r["contention_level"])) for r in agg})

    fig, axes = plt.subplots(
        1, len(policies), figsize=(3.1 * len(policies) + 1.2, 3.4), sharey=True
    )
    if len(policies) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)

    im = None
    for ax, pol in zip(axes, policies):
        grid = []
        for phi in phis:
            row = []
            for b in bursts:
                vals = [
                    _f(r["output_valid_rate"]) for r in agg
                    if r["policy"] == pol
                    and int(_f(r["contention_level"])) == b
                    and abs(_f(r["freshness_window"]) - phi) < 1e-6
                    and _f(r["output_valid_rate"]) is not None
                ]
                row.append(sum(vals) / len(vals) if vals else float("nan"))
            grid.append(row)
        im = ax.imshow(grid, cmap=SEQ_BLUE, vmin=0.0, vmax=1.0,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(bursts))); ax.set_xticklabels(bursts, fontsize=9)
        ax.set_yticks(range(len(phis)))
        ax.set_yticklabels([f"{p:.0f}" for p in phis], fontsize=9)
        ax.set_xlabel("burst size B", fontsize=9, color=INK_MUTED)
        ax.set_title(pol.replace("_", " "), fontsize=10, color=INK)
        ax.tick_params(colors=INK_MUTED)
        # Cell values: the table view that the contrast WARN obliges.
        for i in range(len(phis)):
            for j in range(len(bursts)):
                v = grid[i][j]
                if v != v:
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if v > 0.55 else INK)

    axes[0].set_ylabel("freshness window φ (ms)", fontsize=9, color=INK_MUTED)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cb.set_label("output-valid rate", fontsize=9, color=INK_MUTED)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(
        "Output validity across contention and freshness window",
        fontsize=11, color=INK, y=1.06,
    )
    return _save(fig, out_dir, "plot3_phi_sensitivity")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/freshness_eval")
    ap.add_argument("--output", default="figures/freshness_eval")
    ap.add_argument("--phi", type=float, default=None,
                    help="window for plots 1 and 2 (default: the median swept phi)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    in_dir = args.input if os.path.isabs(args.input) else os.path.join(repo, args.input)
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(repo, args.output)

    agg = read_csv(os.path.join(in_dir, "aggregate.csv"))
    per_inv = read_csv(os.path.join(in_dir, "per_invocation.csv"))
    ivs_path = os.path.join(in_dir, "intervals.csv")
    intervals = read_csv(ivs_path) if os.path.exists(ivs_path) else []
    with open(os.path.join(in_dir, "manifest.json")) as f:
        manifest = json.load(f)
    epoch_ms = float(manifest.get("epoch_ms", 300.0))

    phis = sorted({_f(r["freshness_window"]) for r in agg})
    phi = args.phi if args.phi is not None else phis[len(phis) // 2]

    written: List[str] = []
    written += plot_validity_vs_contention(agg, out_dir, phi)

    # Plot 2 targets the largest divergence among deployable policies, so the
    # figure shows the mechanism where it is strongest rather than a cell picked
    # by hand.
    cand = [
        r for r in agg
        if r["policy"] != ORACLE and _f(r["divergence"]) is not None
    ]
    if cand and intervals:
        worst = max(cand, key=lambda r: _f(r["divergence"]))
        written += plot_diagnostic_timeline(
            per_inv, intervals, out_dir,
            policy=worst["policy"], burst=int(_f(worst["contention_level"])),
            seed=int(_f(worst["seed"])), phi=_f(worst["freshness_window"]),
            epoch_ms=epoch_ms,
        )
        print(f"plot 2 target: {worst['policy']} B={int(_f(worst['contention_level']))} "
              f"phi={_f(worst['freshness_window']):.1f} "
              f"divergence={_f(worst['divergence']):.3f}")
    elif not intervals:
        print(f"plot 2 skipped: no intervals.csv in {in_dir} (re-run the sweep)")

    written += plot_phi_sensitivity(agg, out_dir)

    print(f"\nwrote {len(written)} file(s) to {os.path.relpath(out_dir, repo)}/")
    for p in written:
        print(f"  {os.path.basename(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
