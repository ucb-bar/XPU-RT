#!/usr/bin/env python3
"""YOLO per-frame service against the cores allocated to it, derived from the schedules.

The earlier `fig_cores_yolo` was drawn by `fig_fair_v6.py`, which held all thirty of its points
as literals (docs/Baselines/ros_baseline_tiers.md). This derives every point from the schedules under `schedules/`, and records in a sidecar where each row comes from -- because
the rows are not all the same kind of number, and the figure's claim depends on which kind each is.

Every row is the **median per-frame YOLO response** of its schedule: per YOLO instance, the last
dispatch's finish minus the first dispatch's start, then the median over the instances. One
definition, applied to every arm, so the three curves are comparable.

    greedy   scheduled_m_greedy_shard_K{4..8}_{predicted,board}_greedy_profiled.json
    CP-SAT   scheduled_fine_K{4,6,8}_D*  (AOT) and scheduled_cbp_K{4..8}_board_D*  (board)
    ROS      scheduled_ros_pin_{predicted,board}.json

Two properties of the rows follow from their schedules.

**The CP-SAT rows come from a deadline search, and CP-SAT schedules to its deadline.** The achieved
median equals the budget it was given to within 0.01 ms at every core count, so the level of the
CP-SAT curve is set by where the search stopped, not by how fast the solver can make the chain go.
It is a tightest-feasible-budget curve drawn on the same axis as two achieved-service curves. The
sidecar records the deadline beside the achieved value at every point so the two cannot be confused.

**The ROS arm has one schedule, not a sweep.** `scheduled_ros_pin_*` is a single 1-hart pin, which
is what the paper's caption means by core-independent. It is a Tier A schedule
(`policy: ros_pinning_periodic`; docs/Baselines/ros_baseline_tiers.md). The literal curve drew one
through five core counts that varies by 5 ms. `--ros flat` draws the one schedule pair at every width;
`--ros published` reproduces the earlier literal curve and marks the four unsourced points in the sidecar.

    scripts/cores_yolo_service.py [--ros flat|published] [--anchor MS] [--out PREFIX]

`--anchor` is the service that reads as 1.0x nominal cruise on the right axis. It is a choice, not a
measurement, and it decides which curves fall inside the band -- so it is required to appear in the
sidecar and is printed on the figure. 24.0 is what the published figure used; the paper's prose
calls the frame 22 ms; a third render used 24.5.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCH = os.path.join(REPO, "schedules")
K = [4, 5, 6, 7, 8]

# the deadline search behind each CP-SAT point: the tightest budget a solve accepted at that width.
# K5 and K7 were never solved for the AOT row; the published figure interpolated them and said so.
FINE = {4: "D19.0", 6: "D18.5", 8: "D18.25"}
CBP = {4: "D24p5", 5: "D24", 6: "D23p25", 7: "D23p25", 8: "D23"}
# the four ROS points with no schedule behind them, as the published figure drew them
ROS_PUBLISHED = {"sched": {4: 25.757, 5: 26.694, 6: 24.884, 7: 25.914},
                 "board": {4: 31.414, 5: 36.264, 6: 31.659, 7: 33.083}}

B, G, R = "#0072B2", "#009E73", "#D55E00"
FILL_OK, FILL_BAD = "#e7f1ec", "#f7eae4"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def median_response(rel):
    """The median per-frame YOLO response of a schedule, and how many instances it is over.

    Per YOLO instance: the last dispatch to finish, minus the first to start. One definition for
    every arm -- a sharded inference and a sequential one are then the same measurement.
    """
    p = os.path.join(REPO, rel)
    s = json.load(open(p))
    disp = s["dispatches"]
    disp = disp.values() if isinstance(disp, dict) else disp
    fin, start = collections.defaultdict(float), collections.defaultdict(lambda: float("inf"))
    for d in disp:
        job = str(d.get("job_name", ""))
        if "yolo" not in job.lower():
            continue
        m = re.match(r"(.*?)(\d+)$", job)
        i = int(m.group(2)) if m else 0
        fin[i] = max(fin[i], d["start_time"] + d["duration"])
        start[i] = min(start[i], d["start_time"])
    if not fin:
        raise SystemExit(f"{rel}: no YOLO dispatches")
    resp = sorted(fin[i] - start[i] for i in fin)
    return statistics.median(resp), len(resp), rel


def deadline_of(token):
    return float(token[1:].replace("p", "."))


def searched(k, cost):
    """Every deadline a solve was attempted at, for this width and cost model.

    The plotted point is the tightest of these. That is a statement about what was tried, not about
    what is feasible: at K7 and K8 of the board search only one deadline was ever attempted, so
    nothing here says a tighter one would have failed.
    """
    import glob
    pat = (f"scheduled_fine_K{k}_D*_cpsat_profiled.json" if cost == "AOT"
           else f"scheduled_cbp_K{k}_board_D*_cpsat_profiled.json")
    out = []
    for f in glob.glob(os.path.join(SCH, pat)):
        if f.endswith(("_metrics.json", "_report.json")):
            continue
        m = re.search(r"_D([0-9p.]+)_cpsat", os.path.basename(f))
        if m:
            out.append(float(m.group(1).replace("p", ".")))
    return sorted(out)


def rows(ros_mode):
    """Every drawn point, with where it came from. Returns (series, provenance)."""
    prov, series = [], {}

    def add(name, per_k):
        series[name] = [per_k[k] for k in K]

    # -- greedy: a real schedule at every width, in both cost models
    for name, cost in (("greedy_sched", "predicted"), ("greedy_board", "board")):
        vals = {}
        for k in K:
            rel = f"schedules/scheduled_m_greedy_shard_K{k}_{cost}_greedy_profiled.json"
            v, n, _ = median_response(rel)
            vals[k] = v
            prov.append({"series": name, "cores": k, "value_ms": round(v, 4), "kind": "achieved service",
                         "source": rel, "sha256": sha(os.path.join(REPO, rel)), "instances": n})
        add(name, vals)

    # -- CP-SAT: the tightest budget a solve accepted. The achieved median equals it, because the
    #    solver schedules to its deadline, so the curve traces the search and not a capability.
    for name, table, cost in (("cpsat_sched", FINE, "AOT"), ("cpsat_board", CBP, "board")):
        vals, known = {}, {}
        for k, tok in table.items():
            rel = (f"schedules/scheduled_fine_K{k}_{tok}_cpsat_profiled.json" if cost == "AOT"
                   else f"schedules/scheduled_cbp_K{k}_board_{tok}_cpsat_profiled.json")
            v, n, _ = median_response(rel)
            known[k] = v
            mt = os.path.join(REPO, rel.replace(".json", "_metrics.json"))
            miss = json.load(open(mt)).get("deadline_miss_count") if os.path.exists(mt) else None
            tried = searched(k, cost)
            prov.append({"series": name, "cores": k, "value_ms": round(v, 4),
                         "kind": "tightest deadline attempted and accepted; the schedule packs to it",
                         "deadline_ms": deadline_of(tok), "solver_deadline_miss_count": miss,
                         "deadlines_attempted": tried,
                         "tightest_attempted": min(tried) == deadline_of(tok) if tried else None,
                         "source": rel, "sha256": sha(os.path.join(REPO, rel)), "instances": n})
        for k in K:                       # K5 and K7 of the AOT row were never solved
            if k in known:
                vals[k] = known[k]
            else:
                lo = max(x for x in known if x < k); hi = min(x for x in known if x > k)
                vals[k] = known[lo] + (known[hi] - known[lo]) * (k - lo) / (hi - lo)
                prov.append({"series": name, "cores": k, "value_ms": round(vals[k], 4),
                             "kind": f"interpolated between K{lo} and K{hi}; no solve at this width",
                             "source": None})
        add(name, vals)

    # -- ROS: one 1-hart pin, in both cost models. There is no sweep.
    for name, rel in (("ros_sched", "schedules/scheduled_ros_pin_predicted.json"),
                      ("ros_board", "schedules/scheduled_ros_pin_board.json")):
        v, n, _ = median_response(rel)
        pub = ROS_PUBLISHED["sched" if name == "ros_sched" else "board"]
        vals = {}
        for k in K:
            if k == 8 or ros_mode == "flat":
                vals[k] = v
                prov.append({"series": name, "cores": k, "value_ms": round(v, 4),
                             "kind": "achieved service of the one 1-hart pin; it does not shard, so "
                                     "it is the same at every width",
                             "source": rel, "sha256": sha(os.path.join(REPO, rel)), "instances": n})
            else:
                vals[k] = pub[k]
                prov.append({"series": name, "cores": k, "value_ms": pub[k],
                             "kind": "as published; no schedule in this repository produces it",
                             "source": None})
        add(name, vals)
    return series, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ros", choices=("flat", "published"), default="flat",
                    help="flat: the one 1-hart pin measurement at every width (default). "
                         "published: reproduce the published curve, marking its four unsourced points")
    ap.add_argument("--anchor", type=float, default=24.0,
                    help="the service that reads as 1.0x nominal cruise (a choice, recorded and drawn)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    out = a.out or os.path.join(REPO, "results/codesign_feedback/refined",
                                f"cores_yolo_service_derived_{a.ros}")

    S, prov = rows(a.ros)
    DL = a.anchor

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 13.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    ax.axhspan(0, DL, color=FILL_OK, zorder=0)
    ax.axhspan(DL, 400, color=FILL_BAD, zorder=0)
    ax.axhline(DL, color="#111", lw=1.7, ls=(0, (5, 3)), zorder=4)
    ax.text(6.3, DL + 0.18, f"cruise anchor (1.0× = {DL:g} ms) · a chosen budget", fontsize=11,
            color="#166b3d", weight="bold", va="bottom", ha="center")

    def pair(sched, board, color, mk, label):
        ax.plot(K, sched, color=color, ls=(0, (2, 1.6)), lw=1.9, marker=mk, ms=6.5, mfc="white",
                mec=color, mew=1.5, alpha=0.85, zorder=6)
        ax.plot(K, board, color=color, ls="-", lw=2.8, marker=mk, ms=8.5, mfc=color, mec="white",
                mew=1.3, zorder=8, label=label)
    pair(S["cpsat_sched"], S["cpsat_board"], B, "D", "CP-SAT · tightest budget accepted")
    pair(S["greedy_sched"], S["greedy_board"], G, "o", "XPU-RT greedy + shard · achieved")
    pair(S["ros_sched"], S["ros_board"], R, "s", "ROS · 1-hart pin (no shard) · achieved")

    for arr, color in ((S["cpsat_board"], B), (S["greedy_board"], G), (S["ros_board"], R)):
        ax.annotate(f"{DL / arr[-1]:.2f}×", (8, arr[-1]), textcoords="offset points", xytext=(9, 0),
                    fontsize=12, weight="bold", color=color, ha="left", va="center")

    lo, hi = min(min(v) for v in S.values()), max(max(v) for v in S.values())
    ax.set_ylim(lo - 1.8, hi + 1.6); ax.set_xlim(3.7, 8.9); ax.set_xticks(K)
    ax.set_xlabel("cores allocated to YOLO", fontsize=15, labelpad=9)
    ax.set_ylabel("median per-frame YOLO response (ms)\n(lower = faster inference ↓)", fontsize=13)
    ax.tick_params(labelsize=12.5)
    axr = ax.secondary_yaxis("right", functions=(lambda s: DL / np.where(s <= 0, np.nan, s),
                                                 lambda v: DL / np.where(v <= 0, np.nan, v)))
    axr.set_ylabel("max sustainable cruise speed (× nominal)\n(higher = faster flight ↑)", fontsize=13)
    tk = [t for t in (1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5) if lo - 1.8 < DL / t < hi + 1.6]
    axr.set_yticks([DL / t for t in tk]); axr.set_yticklabels([f"{t:.1f}×" for t in tk], fontsize=12)

    style = [Line2D([0], [0], color="#444", ls="-", lw=2.8, label="board-calibrated cost"),
             Line2D([0], [0], color="#444", ls=(0, (2, 1.6)), lw=1.9, label="scheduled · AOT")]
    h, _ = ax.get_legend_handles_labels()
    sub = ("every point is the median per-frame response of a schedule in this repository"
           if a.ros == "flat" else
           "ROS K4–K7 are the published values; no schedule here produces them")
    leg = ax.legend(handles=[h[0], style[0], h[1], style[1], h[2]], loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), fontsize=12.5, ncol=3, title=sub,
                    title_fontsize=12, columnspacing=1.6, handletextpad=0.6, borderaxespad=0)
    leg._legend_box.align = "center"

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=a.dpi, bbox_inches="tight")

    side = {
        "figure": out + ".png",
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "cores_yolo_service",
        "statistic": "median per-frame YOLO response: per instance, last dispatch finish minus "
                     "first dispatch start; median over instances",
        "cruise_anchor_ms": DL,
        "cruise_anchor_note": "a chosen budget, not a measurement: it decides which curves fall "
                              "inside the band. The published figure used 24.0, a third render "
                              "24.5, and the paper's prose calls the frame 22 ms.",
        "ros_mode": a.ros,
        "cores": K,
        "series": {k: [round(x, 4) for x in v] for k, v in S.items()},
        "cruise_at_8_cores": {k: round(DL / v[-1], 4) for k, v in S.items() if k.endswith("board")},
        "provenance": prov,
        "unsourced_points": sum(1 for p in prov if p["source"] is None),
        "inputs": {p["source"]: p["sha256"] for p in prov if p["source"]},
    }
    json.dump(side, open(out + "_metrics.json", "w"), indent=1)
    print(f"wrote {out}.png  ({a.ros} ROS, anchor {DL:g} ms, "
          f"{side['unsourced_points']} of {len(prov)} points without a schedule)")
    for name in ("cpsat_board", "greedy_board", "ros_board"):
        print(f"  {name:<13} " + "  ".join(f"K{k} {v:6.2f}" for k, v in zip(K, S[name]))
              + f"   -> {DL / S[name][-1]:.2f}x at 8 cores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
