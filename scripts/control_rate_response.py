#!/usr/bin/env python3
"""How flight outcome responds to the rate at which commands actually reach the vehicle.

Every campaign on disk replays a *measured* control cadence into the same simulator, the same course
and the same vehicle. Pooling them by the cadence each arm achieved -- rather than by which runtime
produced it -- asks one question: does the command rate predict the outcome, whoever is scheduling?

It does. The ROS 2 arrangements climb monotonically with their own command rate and level off at the
value XPU-RT holds at every camera rate, because XPU-RT schedules control in its own slot and always
commands near 100 Hz. The architectural difference therefore decides the outcome exactly where it
decides whether the command rate falls under the floor, and nowhere else.

Two things are drawn, and they are not the same kind of evidence:

  * the pooled points, one per arm configuration, which differ in more than their rate (pinning,
    camera rate, goal hold) and are therefore observational;
  * the controlled sweep, one ROS arrangement (two YOLO nodes across both clusters) at three camera
    rates with the goal hold kept at one camera period throughout, where only the rate moves.

    scripts/control_rate_response.py [--out results/codesign_feedback/refined/control_rate_response]

Only flights that pass `flight_quarantine.flight_rows` are counted: a GPU-faulted batch is not data.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figure_constants as fc
from flight_quarantine import flight_rows

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the deployed controller gain. The gain-policy experiments (0.005, 0.01667, 0.01277) change the
# vehicle's response to a command, so pooling them with the deployed gain would confound the axis.
GAIN = "0.0055"
MIN_N = 36                      # an arm configuration needs this many surviving flights to be a point

# The controlled part of the figure: one ROS arrangement flown at several camera rates with the goal
# hold kept at exactly one camera period, so the only quantity that moves along the line is the rate
# at which commands reach the vehicle. Two are available and they are different kinds of evidence:
#
#   vanilla4x2  the unpinned two-YOLO-node arrangement over 30/36/40 Hz cameras. A narrow span --
#               33.3 down to 25.0 Hz -- because above 40 Hz this arrangement stops gaining.
#   cp3         the pinned six-core deployment the Tier A showdown drew, over 15/25/30/45 Hz
#               cameras (campaign_cp3_ladder). Wider span and it is the arm the paper's baseline
#               claims are about, so the sweep and the baseline are the same deployment rather than
#               a neighbouring one borrowed to make the point.
#
# Above a 45 Hz camera cp3 saturates at ~39 Hz control, so the ladder stops at 45 rather than flying
# cells that would land on top of each other; that ceiling is itself part of the story.
SWEEPS = {
    "vanilla4x2": ("unpinned ROS 2, camera 30\u219240 Hz",
                   [("ros_vanilla4x230.csv", "33.3", 30), ("ros_vanilla4x236.csv", "27.8", 36),
                    ("ros_vanilla4x240.csv", "25.0", 40)]),
    "cp3": ("pinned ROS 2, camera 15\u219245 Hz",
            [("ros_cp315.csv", "66.7", 15), ("ros_cp325.csv", "40.0", 25),
             ("ros_cp330.csv", "33.3", 30), ("ros_cp345.csv", "22.2", 45)]),
}
SWEEP = SWEEPS["vanilla4x2"][1]


def wilson(k, n, z=1.96):
    """Wilson score interval -- behaves at k=0 and k=n, where the normal approximation does not."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def collect():
    """Every arm configuration on disk, keyed by (trace, hold), with the cadence it actually replayed."""
    seen, groups = [], collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(REPO, "results/codesign_feedback/campaign_*/campaign.csv"))):
        try:
            rows = flight_rows(p)
        except Exception as e:                      # a malformed census is reported, never silently skipped
            print(f"   skipped {os.path.relpath(p, REPO)}: {e}", file=sys.stderr)
            continue
        if rows:
            seen.append(p)
        for r in rows:
            if r.get("moment_scale") != GAIN:
                continue
            groups[(r["ctrl_trace"], r["percep_hold_ms"])].append(r)

    pts = []
    for (trace, hold), v in groups.items():
        if len(v) < MIN_N:
            continue
        k = sum(1 for r in v if r["outcome"] == "success")
        g = [float(r["gates_passed"]) for r in v]
        # a greedy schedule is kept but held apart: it misses deadlines, so the rate it nominally
        # delivers is not a rate the chain can use, and pooling it would blur what the axis means
        kind = "ros"
        if trace.startswith("xpu"):
            kind = "greedy" if "greedy" in trace else "xpu"
        pts.append({
            "trace": trace, "hold_ms": float(hold), "n": len(v), "completed": k,
            "hz": sum(float(r["eff_cmd_hz"]) for r in v) / len(v),
            "mean_gates": sum(g) / len(g),
            "kind": kind,
        })
    return sorted(pts, key=lambda d: d["hz"]), seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results/codesign_feedback/refined/control_rate_response"))
    ap.add_argument("--sweep", default="vanilla4x2", choices=sorted(SWEEPS),
                    help="which controlled ladder to draw as the joined line (default: vanilla4x2, "
                         "the form already rendered; cp3 draws the pinned deployment's own ladder)")
    a = ap.parse_args()
    sweep_label, sweep_spec = SWEEPS[a.sweep]

    pts, sources = collect()
    ros = [d for d in pts if d["kind"] == "ros"]
    xpu = [d for d in pts if d["kind"] == "xpu"]
    greedy = [d for d in pts if d["kind"] == "greedy"]
    if not ros or not xpu:
        print("no points; is the results tree present?", file=sys.stderr)
        return 1

    sweep = []
    for trace, hold, cam in sweep_spec:
        m = [d for d in ros if d["trace"] == trace and abs(d["hold_ms"] - float(hold)) < 0.05]
        if m:
            sweep.append({**m[0], "camera_hz": cam})

    C_ROS, C_XPU, C_SWEEP, C_GRE = "#b3402c", "#1c4e91", "#d98218", "#6b6b6b"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # the band where panel B of the showdown measures success climbing out of the floor
    for x in (ax, bx):
        x.axvspan(25, 50, color="#8d8d8d", alpha=0.10, zorder=0, lw=0)
        x.text(37.5, x.get_ylim()[1], "", ha="center")

    # ---- left: mean gates ------------------------------------------------------------------
    ax.scatter([d["hz"] for d in ros], [d["mean_gates"] for d in ros], s=[min(150, 18 + d["n"] / 6) for d in ros],
               facecolor=C_ROS, edgecolor="white", lw=0.8, zorder=3, label="ROS 2 arrangements")
    ax.scatter([d["hz"] for d in xpu], [d["mean_gates"] for d in xpu], s=[min(150, 18 + d["n"] / 6) for d in xpu],
               marker="s", facecolor=C_XPU, edgecolor="white", lw=0.8, zorder=3, label="XPU-RT schedules")
    if len(sweep) > 1:
        ax.plot([d["hz"] for d in sweep], [d["mean_gates"] for d in sweep], "-o", color=C_SWEEP,
                lw=2.2, ms=7, mfc="white", mew=2.0, zorder=4,
                label=sweep_label)
        # labels sit above their own marker so the line stays readable however many points it has;
        # the leftmost is pushed inward because at the axis minimum a centred label clips the spine
        for i, d in enumerate(sweep):
            off, ha = ((6, 9), "left") if i == 0 else ((0, 9), "center")
            ax.annotate(f"{d['camera_hz']} Hz cam", (d["hz"], d["mean_gates"]), textcoords="offset points",
                        xytext=off, fontsize=7.5, color=C_SWEEP, ha=ha, va="bottom")

    if greedy:
        ax.scatter([d["hz"] for d in greedy], [d["mean_gates"] for d in greedy],
                   s=[min(150, 18 + d["n"] / 6) for d in greedy], marker="X", facecolor="none",
                   edgecolor=C_GRE, lw=1.6, zorder=3, label="XPU-RT, greedy (misses deadlines)")
        for d in greedy:
            ax.annotate(f"greedy {d['completed']}/{d['n']}", (d["hz"], d["mean_gates"]),
                        textcoords="offset points", xytext=(0, 9), fontsize=7.5, color=C_GRE,
                        ha="center", va="bottom")

    mx = sum(d["mean_gates"] for d in xpu) / len(xpu)
    ax.axhline(mx, color=C_XPU, ls=":", lw=1.3, zorder=2)
    ax.text(20.5, mx + 0.07, f"XPU-RT holds {mx:.2f} at every camera rate",
            color=C_XPU, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("control rate actually delivered to the vehicle (Hz)")
    ax.set_ylabel("mean gates cleared (of 4)")
    ax.set_title("Outcome tracks the command rate, whoever schedules it", fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 40, 50, 70, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-0.1, 2.6)
    ax.grid(alpha=0.25, lw=0.6)
    lo = min(d["hz"] for d in pts); hi = max(d["hz"] for d in pts)
    for x in (ax, bx):
        x.set_xlim(lo * 0.88, hi * 1.30)

    # ---- right: completion, with Wilson bands ----------------------------------------------
    for d in pts:
        lo, hi = wilson(d["completed"], d["n"])
        c = {"xpu": C_XPU, "ros": C_ROS, "greedy": C_GRE}[d["kind"]]
        bx.plot([d["hz"], d["hz"]], [lo * 100, hi * 100], color=c, lw=1.1, alpha=0.5, zorder=2)
    if greedy:
        bx.scatter([d["hz"] for d in greedy], [100 * d["completed"] / d["n"] for d in greedy],
                   s=[min(150, 18 + d["n"] / 6) for d in greedy], marker="X", facecolor="none",
                   edgecolor=C_GRE, lw=1.6, zorder=3)
    bx.scatter([d["hz"] for d in ros], [100 * d["completed"] / d["n"] for d in ros],
               s=[min(150, 18 + d["n"] / 6) for d in ros], facecolor=C_ROS, edgecolor="white", lw=0.8, zorder=3)
    bx.scatter([d["hz"] for d in xpu], [100 * d["completed"] / d["n"] for d in xpu],
               s=[min(150, 18 + d["n"] / 6) for d in xpu], marker="s", facecolor=C_XPU,
               edgecolor="white", lw=0.8, zorder=3)
    bx.set_xlabel("control rate actually delivered to the vehicle (Hz)")
    bx.set_ylabel("flights completing all 4 gates (%)")
    bx.set_title("Same axis, completion with 95 % Wilson bands", fontsize=10.5)
    bx.set_xscale("log")
    bx.set_xticks([20, 30, 40, 50, 70, 100])
    bx.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    bx.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.grid(alpha=0.25, lw=0.6)

    n_tot = sum(d["n"] for d in pts)
    fig.text(0.5, 0.075,
             f"{n_tot} quarantine-passing flights across {len(pts)} arm configurations, gain {GAIN}; "
             "marker area ∝ flights per point. Shaded 25–50 Hz is where the rate-injected envelope "
             "climbs out of the floor.",
             ha="center", fontsize=7.6, color="#444")
    fig.text(0.5, 0.043,
             "Pooled points differ in more than their rate, so they are observational; the joined line is "
             "the controlled sweep — one ROS arrangement, goal hold at one camera period, only the rate "
             "moving.",
             ha="center", fontsize=7.6, color="#444")
    fig.text(0.5, 0.011,
             "Greedy schedules are drawn apart: they miss deadlines, so the rate they nominally deliver "
             "never reaches the vehicle as usable commands. Rate alone does not rescue them.",
             ha="center", fontsize=7.6, color="#666")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.152), ncol=len(h),
               fontsize=8.2, frameon=False, handletextpad=0.5, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.205, 1, 1))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=200)
    fig.savefig(a.out + ".pdf")

    side = {"figure": a.out + ".png",
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
            **fc.sidecar_common("control_rate_response", sources),
            "gain": GAIN, "min_n": MIN_N, "flights_total": n_tot,
            "controlled_sweep_arm": a.sweep, "controlled_sweep_label": sweep_label,
            "xpu_mean_gates": round(mx, 4),
            "points": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for d in pts],
            "controlled_sweep": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                                 for d in sweep]}
    json.dump(side, open(a.out + "_metrics.json", "w"), indent=2)

    print(f"wrote {a.out}.png  ({len(pts)} arm configurations, {n_tot} flights)")
    print(f"  ROS 2 : {min(d['mean_gates'] for d in ros):.2f} → {max(d['mean_gates'] for d in ros):.2f} mean gates "
          f"over {min(d['hz'] for d in ros):.0f}–{max(d['hz'] for d in ros):.0f} Hz")
    print(f"  XPU-RT: {min(d["mean_gates"] for d in xpu):.2f} → {max(d['mean_gates'] for d in xpu):.2f} mean gates "
          f"over {min(d['hz'] for d in xpu):.0f}–{max(d['hz'] for d in xpu):.0f} Hz")
    for d in sweep:
        print(f"  sweep : camera {d['camera_hz']} Hz → control {d['hz']:.1f} Hz, "
              f"{d['completed']}/{d['n']}, mean {d['mean_gates']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
