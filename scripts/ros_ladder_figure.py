#!/usr/bin/env python3
"""The ROS 2 effort ladder, drawn: what each unit of expert work buys the baseline.

The panel answers the reviewer who says the baseline is a strawman. Each row is a real deployment of
the same 3-network chain on the K1, one expert change above the row before it -- a worker pool for
YOLO, a wider pool, keep-last-1 queues, a control timer, hand-pinning, a second model instance -- and
each row carries what the board measured for it (harts doing work, control cadence, camera->goal
latency) next to how the flights that replay those measurements came out. XPU-RT is the bottom row
and the dashed reference in every column: it needs none of the rungs, because the schedule is solved.

Nothing here is a new measurement. The rungs, the hart counts and the board timing come from
`scripts/ros_effort_ladder.py` (which derives them from `measured_timing` and the per-core sampler in
results/codesign_feedback/ros_traced/summary.csv); the flights come from the campaign CSVs through
`flight_quarantine`. The outcome column is restricted to ONE cell -- course a, prop density 0.30,
static 2.4 m people placed along the aisle, the deployed gain, latency replayed as well as cadence --
at the (cruise, seed) cells every drawn arm flew, so the rungs differ by the deployment and nothing
else. That restriction is NOT defined here: `CELL`, `in_cell` and `matched_cells` are imported from
`scripts/ros_effort_ladder.py`, so the table and this figure score the flight column with one
definition and cannot report different numbers for the same rung. A rung with no flights in the cell
is drawn as "not flown", never as a zero, and the pooled tally over every campaign -- which mixes
courses, densities, gains, crossing people and cadence-only runs, and is not comparable across
rungs -- is kept in the sidecar beside it for provenance only.

    scripts/ros_ladder_figure.py [--out ...] [--dpi 300]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import figure_constants as FC   # noqa: E402
import measured_timing as MT   # noqa: E402
import ros_effort_ladder as LAD   # noqa: E402
from hil_envelope_panel import wilson   # noqa: E402

INK = "#22242a"
C_XPU = "#1f9e5a"
C_ROS = "#e2231a"
C_P3 = "#5c6bc0"        # the hand-pinned rungs, as the showdown figure colours them
C_ROS8 = "#f28c28"      # the two-instance rungs
GREY = "#8b8b8b"
N_HARTS = 8             # the K1's harts; the ceiling every row is drawn against

# The reference arm: the schedule the showdown figure flies, at the same 45 Hz camera as the rungs.
# The flight cell, the cell test and the tally all come from scripts/ros_effort_ladder.py, so the
# table and this figure score the flight column with ONE definition and cannot diverge.
XPU_TRACE = LAD.XPU_TRACE
CELL = LAD.CELL

# which colour a rung takes: the family its deployment belongs to
FAMILY = {"p3@45": C_P3, "p3_q1@45": C_P3, "vanilla4x2@45": C_ROS8, "vanilla4x2@36": C_ROS8,
          "vanilla4x2tm@45": C_ROS8}


# --------------------------------------------------------------------------------- XPU-RT's harts
def xpu_hart_duty(prefix):
    """Per-hart duty cycle of the reference schedule, from its own board traces.

    The ROS rows count harts at or above 20 % in the per-core sampler; this counts a hart's dispatch
    cycles over the traced span and applies the same threshold, so the columns mean the same thing.
    Dispatch cycles only, so it is a lower bound on what the hart actually did."""
    counts, duties, files = [], [], []
    for f in sorted(glob.glob(os.path.join(RES, "xpurt_long", f"trace_{prefix}[0-9]_other_run1.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        files.append(f)
        span = max(int(r["actual_end_cycles"]) for r in rows) - min(int(r["actual_start_cycles"]) for r in rows)
        busy = collections.Counter()
        for r in rows:
            h = int(r["worker_hart"])
            if h >= 0:
                busy[h] += int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])
        if span > 0:
            d = {h: 100.0 * c / span for h, c in busy.items()}
            duties.append(d); counts.append(sum(v >= 20 for v in d.values()))
    if not counts:
        raise FC.MissingMeasurement(f"no xpurt_long trace for {prefix}: XPU-RT's hart count cannot be drawn")
    return statistics.median(counts), len({h for d in duties for h in d}), duties, files


# --------------------------------------------------------------------------------- the rows
def build_rows():
    arms = MT.derive()["ros_arms"]
    harts = LAD.hart_counts()
    _k, _n, k_all, n_all = LAD.flight_tally()      # pooled counts, for the sidecar's provenance only
    cells, cruises, seeds, csvs = LAD.matched_cells()

    rows = []
    for key, label, trace in LAD.LADDER:
        a = arms.get(key)
        if not a or not a.get("gap_mean_pooled"):
            continue
        h = harts.get(key)
        rows.append(dict(arm=key, rung=label, trace=trace, family="ros",
                         cores=(statistics.median(h) if h else None), cores_reps=(sorted(h) if h else []),
                         ctrl_hz=round(1000.0 / a["gap_mean_pooled"], 1),
                         latency_ms=round(a["e2e_goal_med_pooled"], 1),
                         latency_field="e2e_goal_med_pooled",
                         colour=FAMILY.get(key, C_ROS),
                         pooled=[k_all.get(trace, 0), n_all.get(trace, 0)]))

    xh, xh_any, _duties, xfiles = xpu_hart_duty(FC.arm_for(XPU_TRACE).derive[1])
    rows.append(dict(arm="xpu_a_cpsat_hard", rung="XPU-RT · CP-SAT, the same chain, solved", trace=XPU_TRACE,
                     family="xpu", cores=xh, cores_reps=[], cores_any=xh_any,
                     ctrl_hz=round(1000.0 / FC.ctrl_gap_ms(XPU_TRACE), 1),
                     latency_ms=round(FC.lat_ms(XPU_TRACE), 1), latency_field="SOLVER_ARMS.chain_ms",
                     colour=C_XPU, pooled=[k_all.get(XPU_TRACE, 0), n_all.get(XPU_TRACE, 0)]))

    for r in rows:                                     # the outcome column: the matched cell only
        fl = cells.get(r["trace"], [])
        k = sum(f["outcome"] == "success" for f in fl)
        r["flights"] = [k, len(fl)]
        r["ci"] = [round(x, 4) for x in wilson(k, len(fl))[1:]] if fl else None
    return rows, cruises, seeds, csvs, xfiles


# --------------------------------------------------------------------------------- the drawing
def draw(rows, cruises, seeds, out, dpi):
    ros = [r for r in rows if r["family"] == "ros"]
    xpu = next(r for r in rows if r["family"] == "xpu")
    order = ros + [xpu]
    y = list(range(len(order)))[::-1]                  # rung 1 at the top, XPU-RT at the bottom
    yy = {id(r): y[i] for i, r in enumerate(order)}
    nflown = sum(1 for r in order if r["flights"][1])

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#b9b7b2",
                         "axes.linewidth": 0.9, "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(13.6, 7.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.2, 1.5], left=0.305, right=0.985,
                          top=0.795, bottom=0.235, wspace=0.30)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_ylim(-0.9, len(order) - 0.4)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", ls=":", lw=0.6, color="#d8d5cf", zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.axhline(0.55, color="#cfccc6", lw=1.0, zorder=1)     # XPU-RT sits below the rule
        ax.tick_params(labelsize=9.5)

    # rung labels, on the left of the first panel
    labels = []
    for i, r in enumerate(order):
        labels.append(r["rung"] if r["family"] == "xpu" else f"{i + 1}.  {r['rung']}")
    axes[0].set_yticklabels(labels, fontsize=9.3)
    for t, r in zip(axes[0].get_yticklabels(), order):
        t.set_color(r["colour"])
        if r["family"] == "xpu":
            t.set_fontweight("bold")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    def bars(ax, field, xmax, fmt, title, ref_note=None):
        for r in order:
            v = r[field]
            if v is None:
                continue
            ax.barh(yy[id(r)], v, height=0.62, color=r["colour"], alpha=0.30 if r["family"] != "xpu" else 0.42,
                    edgecolor=r["colour"], lw=1.1, zorder=3)
            ax.annotate(fmt(v), (v, yy[id(r)]), xytext=(4, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=9.2, weight="bold", color=r["colour"], zorder=6)
        ax.axvline(xpu[field], color=C_XPU, lw=1.5, ls=(0, (4, 2.5)), alpha=0.85, zorder=2)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=10.6, weight="bold", loc="left", pad=9)
        if ref_note:
            ax.annotate(ref_note, (xpu[field], len(order) - 0.55), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.2, weight="bold", color=C_XPU, annotation_clip=False)

    # --- 1. how much of the machine the deployment actually uses --------------------------------
    bars(axes[0], "cores", N_HARTS * 1.32, lambda v: f"{v:.0f}", "harts busy (≥20 %)", "XPU-RT")
    axes[0].axvline(N_HARTS, color=INK, lw=1.0, ls=(0, (1.5, 2)), alpha=0.5, zorder=2)
    axes[0].annotate("all 8 harts of the K1", (N_HARTS, len(order) - 4.6), xytext=(4, 0), rotation=90,
                     textcoords="offset points", ha="left", va="center", fontsize=7.6, color="0.42")
    axes[0].set_xticks(range(0, N_HARTS + 1, 2))

    # --- 2. the cadence the controller actually got ---------------------------------------------
    bars(axes[1], "ctrl_hz", 140, lambda v: f"{v:.0f}", "control rate (Hz)", "XPU-RT")

    # --- 3. how stale the goal the controller acts on is ----------------------------------------
    bars(axes[2], "latency_ms", 315, lambda v: f"{v:.0f}", "camera → goal (ms)", "XPU-RT")

    # --- 4. the flights ------------------------------------------------------------------------
    ax = axes[3]
    lo_x, hi_x = xpu["ci"]
    ax.axvspan(100 * lo_x, 100 * hi_x, color=C_XPU, alpha=0.10, zorder=1, lw=0)
    ax.axvline(100 * xpu["flights"][0] / xpu["flights"][1], color=C_XPU, lw=1.5, ls=(0, (4, 2.5)), alpha=0.85, zorder=2)
    for r in order:
        k, n = r["flights"]
        if not n:
            ax.annotate("not flown", (0.7, yy[id(r)]), fontsize=8.8, style="italic",
                        color=GREY, va="center", ha="left", zorder=6)
            continue
        p, lo, hi = wilson(k, n)
        ax.errorbar([100 * p], [yy[id(r)]], xerr=[[100 * (p - lo)], [100 * (hi - p)]], marker="o",
                    ms=7.5, ls="none", capsize=2.8, elinewidth=1.4, mew=1.2, mec="white",
                    color=r["colour"], ecolor=r["colour"], zorder=5)
        ax.annotate(f"{k}/{n}", (100 * hi, yy[id(r)]), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.8, weight="bold", color=r["colour"], zorder=6)
    ax.set_xlim(0, 33)
    ax.set_title("course completed  (Wilson 95 % CI)", fontsize=10.6, weight="bold", loc="left", pad=9)
    ax.annotate("XPU-RT", (100 * xpu["flights"][0] / xpu["flights"][1], len(order) - 0.55), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.2, weight="bold",
                color=C_XPU, annotation_clip=False)
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0f} %")

    # --- framing -------------------------------------------------------------------------------
    fig.text(0.006, 0.968, "The ROS 2 baseline is a ladder, not a strawman:",
             fontsize=15.0, weight="bold", ha="left", va="center", color=INK)
    fig.text(0.006, 0.928, "each rung is one more thing an engineer does — and they climb toward a schedule that was solved",
             fontsize=11.6, ha="left", va="center", color="0.22")
    fig.text(0.006, 0.878,
             "Same K1, same int8 kernels, same 45 Hz camera (rung 9 at 36 Hz). XPU-RT needs none of the rungs.",
             fontsize=9.8, ha="left", va="center", color="0.38")
    fig.legend(handles=[Line2D([0], [0], color=C_ROS, lw=6, alpha=0.45, label="ROS 2, default placement"),
                        Line2D([0], [0], color=C_P3, lw=6, alpha=0.45, label="hand-pinned"),
                        Line2D([0], [0], color=C_ROS8, lw=6, alpha=0.45, label="two model instances"),
                        Line2D([0], [0], color=C_XPU, lw=6, alpha=0.55, label="XPU-RT (reference, dashed)")],
               loc="upper right", bbox_to_anchor=(0.988, 0.995), ncol=2, frameon=False, fontsize=9.4,
               handlelength=1.6, columnspacing=1.6)
    notes = [
        "Board — harts: ≥20 % busy in the per-core sampler (ROS 2 rows, median over replicates); XPU-RT at ≥20 % dispatch duty in its own board trace "
        f"(8 harts carry work, {xpu['cores']:.0f} clear the threshold, and dispatch cycles alone are a lower bound).",
        "Control rate = 1 / mean control-output gap.   camera→goal = median camera-to-goal chain for the ROS 2 rows, camera-to-control chain for XPU-RT.",
        f"Flights — one cell only: course a, prop density 0.30, static 2.4 m people along the aisle, deployed gain 0.0055, cadence AND latency replayed, at the "
        f"{len(cruises)} cruise speeds ({cruises[0]:g}–{cruises[-1]:g} m/s) and {len(seeds)} seeds every drawn arm flew.",
        f"{nflown} of {len(order)} rows have flights in that cell; the rest are marked, never scored as zero. Rung 1's only flights replay cadence without "
        "latency, a different experiment. Pooled tallies over all campaigns: sidecar.",
    ]
    for i, t in enumerate(notes):
        fig.text(0.006, 0.145 - i * 0.032, t, fontsize=8.1, ha="left", va="center", color="0.32")

    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "refined", "ros_effort_ladder"))
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    rows, cruises, seeds, csvs, xfiles = build_rows()
    draw(rows, cruises, seeds, a.out, a.dpi)

    side = {"figure": os.path.basename(a.out),
            "cell": {**CELL, "latency_replayed": True, "cruise_speeds": cruises, "seeds": seeds},
            "rungs": [{"arm": r["arm"], "rung": r["rung"], "ctrl_trace": r["trace"],
                       "cores_busy_ge20pct": r["cores"], "cores_per_replicate": r["cores_reps"],
                       "cores_carrying_work": r.get("cores_any"),
                       "control_hz": r["ctrl_hz"], "camera_to_goal_ms": r["latency_ms"],
                       "latency_source": r["latency_field"],
                       "flights_matched": {"completed": r["flights"][0], "flown": r["flights"][1],
                                           "fraction": (round(r["flights"][0] / r["flights"][1], 4) if r["flights"][1] else None),
                                           "wilson95": r["ci"]},
                       "flights_pooled_all_campaigns": {"completed": r["pooled"][0], "flown": r["pooled"][1]}}
                      for r in rows],
            "notes": ["Rungs, hart counts, board timing and the flight tally all come from "
                      "scripts/ros_effort_ladder.py (measured_timing + results/codesign_feedback/ros_traced/"
                      "summary.csv); CELL, in_cell and matched_cells are imported, not reimplemented, so this "
                      "figure and the table print the same flight column by construction.",
                      "flights_matched is the drawn column, and is what ros_effort_ladder.py prints: one scene "
                      "cell, at the (cruise, seed) cells every drawn arm flew. flights_pooled_all_campaigns "
                      "mixes courses, densities, gains, people placed across the aisle (walk_cross) and "
                      "cadence-only campaigns; it is NOT comparable across rungs and is recorded only for "
                      "provenance.",
                      "walk_cross is part of the cell although these flights have walk_speed == 0: the flag "
                      "moves the people onto the aisle centre line rather than scattering them across it "
                      "(sims/isaaclab_tasks/warehouse_nav/mdp_obstacles.py), so a crossing run is a different "
                      "obstacle field. Excluding it is what took ros_vanilla4 from 6/216 to 5/180.",
                      "Rung 1 (vanilla@45) has 48 pooled flights, all from a cadence-only campaign that replays no "
                      "perception latency, so it has no flights in the matched cell and is drawn as not flown.",
                      "XPU-RT hart count is dispatch duty in its own board trace (a lower bound: dispatch cycles "
                      "only); the ROS rows are OS per-core busy fractions."],
            **FC.sidecar_common("ros_ladder_figure",
                                csvs + xfiles + [os.path.join(RES, "ros_traced", "summary.csv"),
                                                 os.path.join(RES, "flight_quarantine.csv")])}
    with open(a.out + "_metrics.json", "w") as f:
        json.dump(side, f, indent=1)
    print("wrote", a.out + "_metrics.json")

    w = max(len(r["rung"]) for r in rows) + 2
    print(f"\n{'rung':<{w}}{'harts':>6}{'ctrl Hz':>9}{'cam->goal':>11}{'matched flights':>18}")
    for r in rows:
        k, n = r["flights"]
        fl = f"{k}/{n} ({100 * k / n:.1f} %)" if n else "not flown"
        print(f"{r['rung']:<{w}}{(r['cores'] or 0):>6.0f}{r['ctrl_hz']:>9.1f}{r['latency_ms']:>11.1f}{fl:>18}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
