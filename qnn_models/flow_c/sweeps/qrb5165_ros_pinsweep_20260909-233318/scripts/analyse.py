#!/usr/bin/env python3
"""Turn `measured.json` into the tables and figures ANALYSIS.md quotes.

    tables   per-cell ROS-vs-XPU-RT, placement value, cost-model error,
             window feasibility, ranking check   -> results/analysis.json
    plots    plots/*.png

Nothing here re-runs hardware. Every number comes from `measured.json`, which
carries both sides: this baseline's runs and the XPU-RT sweep's own
`phase4_results.json` medians.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))

#: the XPU-RT sweep's own measured rep spread, quoted before it is used
NOISE_WALL_PCT = 7.62
NOISE_NP_PCT = 9.18


def load():
    with open(os.path.join(SWEEP, "measured.json")) as f:
        return json.load(f)


def by_cell(doc):
    out = collections.defaultdict(list)
    for r in doc["runs"].values():
        out[r["cell"]].append(r)
    return out


def rank_key(r):
    """The ranking rule: (networks starved to zero, then makespan)."""
    return (r["n_starved"], r["makespan_median_ms"])


def np_rank_key(r):
    return (r["n_starved"], r["np_median_ms"])


def cell_summary(doc):
    cells = by_cell(doc)
    xrt = doc["xpurt"]
    rows = []
    for cell, runs in sorted(cells.items()):
        runs = [r for r in runs if r["ok"]]
        if not runs:
            continue
        best = min(runs, key=rank_key)
        worst = max(runs, key=rank_key)
        np_best = min(runs, key=np_rank_key)
        x = xrt.get(cell, {})
        # a cell with no aperiodic network has no non-periodic objective; the
        # sweep10 runner's evaluate() degenerates to the all-operations
        # makespan there, and so does this
        plan_nets = runs[0]["per_net"]
        np_degen = all(v["period_ms"] for v in plan_nets.values())
        # the reference's 6.5: is the minimum-makespan placement the placement
        # you would actually ship? Here the analogue is window feasibility,
        # which is deliberately kept OUT of the ranking key.
        feas = min(runs, key=lambda r: (r["missed_instances"],
                                        r["makespan_median_ms"]))
        row = dict(
            cell=cell, family=runs[0]["family"], config=runs[0]["config"],
            verdict=runs[0]["verdict"], n_legal=runs[0]["n_legal"],
            measure_mode=runs[0]["measure_mode"], n_measured=len(runs),
            ros_best_id=best["assignment"], ros_best_label=best["label"],
            ros_best_ms=best["makespan_median_ms"],
            ros_best_spread_ms=best["makespan_spread_ms"],
            ros_best_spread_pct=pct(best["makespan_spread_ms"],
                                    best["makespan_median_ms"]),
            ros_worst_ms=worst["makespan_median_ms"],
            ros_placement_spread=ratio(worst["makespan_median_ms"],
                                       best["makespan_median_ms"]),
            ros_np_best_ms=np_best["np_median_ms"],
            ros_np_best_label=np_best["label"],
            ros_np_worst_ms=max(r["np_median_ms"] for r in runs),
            np_degenerate=np_degen,
            ros_best_missed=best["missed_instances"],
            feasible_first_id=feas["assignment"],
            feasible_first_ms=feas["makespan_median_ms"],
            feasible_first_missed=feas["missed_instances"],
            makespan_rule_disagrees=(feas["assignment"] != best["assignment"]),
            makespan_rule_cost=ratio(feas["makespan_median_ms"],
                                     best["makespan_median_ms"]),
            ros_best_missed_nets=best["missed_nets"],
            ros_any_starved=any(r["n_starved"] for r in runs),
            xrt_best_ms=x.get("best_makespan_ms"),
            xrt_best_solver=x.get("best_makespan_solver"),
            xrt_greedy_ms=x.get("greedy_makespan_ms"),
            xrt_np_best_ms=x.get("best_np_ms"),
            xrt_np_greedy_ms=x.get("greedy_np_ms"),
            xrt_n_solvers=x.get("n_solvers_measured"),
        )
        # ROS / XPU-RT. >1 means the scheduler wins.
        row["ros_over_xrt_best"] = ratio(row["ros_best_ms"], row["xrt_best_ms"])
        row["ros_over_xrt_greedy"] = ratio(row["ros_best_ms"], row["xrt_greedy_ms"])
        row["ros_over_xrt_np_best"] = ratio(row["ros_np_best_ms"],
                                            row["xrt_np_best_ms"])
        row["inside_noise_wall"] = inside(row["ros_over_xrt_best"], NOISE_WALL_PCT)
        row["inside_noise_np"] = inside(row["ros_over_xrt_np_best"], NOISE_NP_PCT)
        # did the cost model's ranking survive the board?
        pred_best = min(runs, key=lambda r: (r["rank"],))
        row["predicted_best_id"] = pred_best["assignment"]
        row["predicted_best_is_measured_best"] = (
            pred_best["assignment"] == best["assignment"])
        row["predicted_best_penalty"] = ratio(pred_best["makespan_median_ms"],
                                              best["makespan_median_ms"])
        pred_np_best = min(runs, key=lambda r: (r["np_rank"],))
        row["predicted_np_best_is_measured_np_best"] = (
            pred_np_best["assignment"] == np_best["assignment"])
        rows.append(row)
    return rows


def pct(a, b):
    return round(a / b * 100, 2) if a is not None and b else None


def ratio(a, b):
    return round(a / b, 4) if a is not None and b else None


def inside(r, noise_pct):
    if r is None:
        return None
    return abs(r - 1.0) * 100 <= noise_pct


def costmodel_error(doc):
    """measured / predicted per assignment, split on the structures that break it."""
    out = []
    for tag, r in doc["runs"].items():
        if not r["ok"] or not r["predicted_makespan_ms"]:
            continue
        shared = len(set(r["assign"].values())) < len(r["assign"])
        out.append(dict(tag=tag, cell=r["cell"], label=r["label"],
                        predicted=r["predicted_makespan_ms"],
                        measured=r["makespan_median_ms"],
                        err_pct=round((r["makespan_median_ms"]
                                       / r["predicted_makespan_ms"] - 1) * 100, 2),
                        spread_pct=pct(r["makespan_spread_ms"],
                                       r["makespan_median_ms"]),
                        contended=shared,
                        n_nets=len(r["assign"]),
                        has_edge="EDGE-WIRED" in (r["notes"] or [])))
    return out


def cmd_tables(args):
    doc = load()
    rows = cell_summary(doc)
    err = costmodel_error(doc)
    res = {"noise_floor": {"wall_pct": NOISE_WALL_PCT, "np_pct": NOISE_NP_PCT},
           "cells": rows, "costmodel": err}

    # aggregates
    comp = [r for r in rows if r["ros_over_xrt_best"]]
    res["headline"] = {
        "cells_compared": len(comp),
        "ros_over_xrt_best_median": round(statistics.median(
            [r["ros_over_xrt_best"] for r in comp]), 4) if comp else None,
        "ros_faster_cells": sum(1 for r in comp if r["ros_over_xrt_best"] < 1),
        "xrt_faster_cells": sum(1 for r in comp if r["ros_over_xrt_best"] > 1),
        "inside_noise_cells": sum(1 for r in comp if r["inside_noise_wall"]),
        "np_median": round(statistics.median(
            [r["ros_over_xrt_np_best"] for r in rows
             if r["ros_over_xrt_np_best"]]), 4),
        "np_inside_noise_cells": sum(1 for r in rows if r["inside_noise_np"]),
        "placement_spread_median": round(statistics.median(
            [r["ros_placement_spread"] for r in rows
             if r["ros_placement_spread"]]), 4),
        "placement_spread_max": max(
            (r["ros_placement_spread"] for r in rows
             if r["ros_placement_spread"]), default=None),
        "predicted_best_hit_rate": round(
            sum(1 for r in rows if r["predicted_best_is_measured_best"])
            / max(1, len([r for r in rows if r["n_measured"] > 1])), 4),
        "ros_rep_spread_median_pct": round(statistics.median(
            [pct(v["makespan_spread_ms"], v["makespan_median_ms"])
             for v in doc["runs"].values() if v["ok"]]), 2),
        "ros_rep_spread_max_pct": round(max(
            pct(v["makespan_spread_ms"], v["makespan_median_ms"])
            for v in doc["runs"].values() if v["ok"]), 2),
        "np_meaningful_cells": sum(1 for r in rows if not r["np_degenerate"]),
        "cells_where_makespan_rule_disagrees_with_feasibility":
            [r["cell"] for r in rows if r["makespan_rule_disagrees"]],
        "assignments_with_a_starved_network":
            [t for t, v in doc["runs"].items() if v["n_starved"]],
        "assignments_run": len(doc["runs"]),
        "assignments_ok": sum(1 for v in doc["runs"].values() if v["ok"]),
    }
    if err:
        res["headline"]["costmodel_median_abs_err_pct"] = round(
            statistics.median([abs(e["err_pct"]) for e in err]), 2)
        cont = [e for e in err if e["contended"]]
        solo = [e for e in err if not e["contended"]]
        res["headline"]["costmodel_median_abs_err_contended_pct"] = round(
            statistics.median([abs(e["err_pct"]) for e in cont]), 2) if cont else None
        res["headline"]["costmodel_median_abs_err_uncontended_pct"] = round(
            statistics.median([abs(e["err_pct"]) for e in solo]), 2) if solo else None

    p = os.path.join(SWEEP, "results", "analysis.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {p}\n")

    print(f'{"cell":32s} {"n":>4s}{"m":>4s} {"ROS best ms":>12s} {"sprd%":>6s} '
          f'{"XRT best ms":>12s} {"ROS/XRT":>8s} {"noise?":>7s} '
          f'{"place":>6s} {"miss":>5s}')
    for r in rows:
        print(f'{r["cell"][len("networks_"):]:32s} {r["n_legal"]:4d}'
              f'{r["n_measured"]:4d} {r["ros_best_ms"]:12.3f} '
              f'{(r["ros_best_spread_pct"] or 0):6.1f} '
              f'{(r["xrt_best_ms"] or 0):12.3f} '
              f'{(r["ros_over_xrt_best"] or 0):8.3f} '
              f'{"in" if r["inside_noise_wall"] else "OUT":>7s} '
              f'{(r["ros_placement_spread"] or 0):6.2f} '
              f'{r["ros_best_missed"]:5.1f}')
    h = res["headline"]
    print("\nheadline:", json.dumps(h, indent=1))
    return 0


def cmd_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    doc = load()
    with open(os.path.join(SWEEP, "results", "analysis.json")) as f:
        res = json.load(f)
    rows = [r for r in res["cells"] if r["ros_over_xrt_best"]]
    os.makedirs(os.path.join(SWEEP, "plots"), exist_ok=True)

    # ---- 1. per-cell ROS vs XPU-RT ------------------------------------
    rows_s = sorted(rows, key=lambda r: r["ros_over_xrt_best"])
    y = np.arange(len(rows_s))
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(rows_s) + 2.2))
    vals = [r["ros_over_xrt_best"] for r in rows_s]
    cols = ["#3b7dd8" if v < 1 else "#d1495b" for v in vals]
    ax.barh(y, vals, color=cols, height=0.7)
    ax.axvline(1.0, color="k", lw=1)
    ax.axvspan(1 - NOISE_WALL_PCT / 100, 1 + NOISE_WALL_PCT / 100,
               color="k", alpha=0.10, lw=0,
               label=f"XPU-RT rep-spread noise floor ±{NOISE_WALL_PCT}%")
    ax.set_yticks(y)
    ax.set_yticklabels([r["cell"][len("networks_"):] for r in rows_s], fontsize=7)
    ax.set_xlabel("ROS whole-model pinning makespan ÷ measured XPU-RT makespan\n"
                  "(<1: pinning is faster, >1: the scheduler is faster)")
    ax.set_title("ROS 2 whole-network pinning vs measured XPU-RT scheduling\n"
                 "best legal placement vs best measured solver, medians of 3 reps",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SWEEP, "plots", "ros_vs_xpurt.png"), dpi=150)
    plt.close(fig)

    # ---- 2. what the placement decision is worth ----------------------
    pr = [r for r in rows if r["ros_placement_spread"]]
    pr.sort(key=lambda r: -r["ros_placement_spread"])
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(pr) + 2.2))
    y = np.arange(len(pr))
    ax.barh(y, [r["ros_placement_spread"] for r in pr], color="#5b8c5a", height=0.7)
    ax.axvline(1.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["cell"][len("networks_"):]}  (n={r["n_legal"]},'
                        f' m={r["n_measured"]})' for r in pr], fontsize=7)
    ax.set_xlabel("worst measured legal placement ÷ best measured legal placement")
    ax.set_title("What choosing the lane is worth, measured\n"
                 "n = legal assignments, m = measured", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SWEEP, "plots", "placement_value.png"), dpi=150)
    plt.close(fig)

    # ---- 3. whole-model cost model vs the board ------------------------
    err = res["costmodel"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for flag, col, lab in ((True, "#d1495b", "two networks share a backend"),
                           (False, "#3b7dd8", "every network alone on its backend")):
        xs = [e["predicted"] for e in err if e["contended"] == flag]
        ys = [e["measured"] for e in err if e["contended"] == flag]
        ax.scatter(xs, ys, s=16, alpha=0.75, color=col, label=lab)
    lim = [min(e["predicted"] for e in err) * 0.8,
           max(e["predicted"] for e in err) * 1.25]
    ax.plot(lim, lim, "k-", lw=1, label="perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("whole-model cost-model makespan (ms)")
    ax.set_ylabel("measured makespan, median of 3 reps (ms)")
    ax.set_title("Where the whole-model cost model breaks", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(SWEEP, "plots", "costmodel.png"), dpi=150)
    plt.close(fig)

    # ---- 4. the non-periodic objective --------------------------------
    npr = [r for r in rows if r["ros_over_xrt_np_best"]]
    npr.sort(key=lambda r: r["ros_over_xrt_np_best"])
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(npr) + 2.2))
    y = np.arange(len(npr))
    vals = [r["ros_over_xrt_np_best"] for r in npr]
    ax.barh(y, vals, color=["#3b7dd8" if v < 1 else "#d1495b" for v in vals],
            height=0.7)
    ax.axvline(1.0, color="k", lw=1)
    ax.axvspan(1 - NOISE_NP_PCT / 100, 1 + NOISE_NP_PCT / 100, color="k",
               alpha=0.10, lw=0, label=f"noise floor ±{NOISE_NP_PCT}%")
    ax.set_yticks(y)
    ax.set_yticklabels([r["cell"][len("networks_"):] for r in npr], fontsize=7)
    ax.set_xlabel("ROS ÷ XPU-RT, NON-PERIODIC makespan\n"
                  "(the objective the sweep10 solver ranking is about)")
    ax.set_title("ROS pinning vs XPU-RT on the non-periodic objective", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SWEEP, "plots", "ros_vs_xpurt_nonperiodic.png"), dpi=150)
    plt.close(fig)
    print("wrote 4 figures to", os.path.join(SWEEP, "plots"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tables"); s.set_defaults(fn=cmd_tables)
    s = sub.add_parser("plots"); s.set_defaults(fn=cmd_plots)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
