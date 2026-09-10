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
import csv
import glob
import io
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))

#: the XPU-RT sweep's own measured rep spread, quoted before it is used
NOISE_WALL_PCT = 7.62
NOISE_NP_PCT = 9.18

XSWEEP = os.path.abspath(os.path.join(
    SWEEP, "..", "qrb5165_sched_algo_sweep10_20260908-210226"))
TOPLEVEL = os.path.abspath(os.path.join(SWEEP, "..", "..", "..", "..",
                                        "data", "toplevel", "s10port"))


def xpurt_np_instances():
    """{cell: {net: instances}} for the NON-PERIODIC networks, as XPU-RT
    actually executed them.

    The objective both sides are compared on is the makespan of the
    non-periodic work subject to the periodic tasks' constraints. Running
    FEWER periodic instances is a legitimately better solution and is not a
    mismatch. Running fewer NON-PERIODIC instances is a different workload,
    because that is the work being timed -- so it is checked here, per cell,
    from the XPU-RT sweep's own trace blocks.
    """
    out = {}
    for fn in sorted(os.listdir(TOPLEVEL)):
        cell = fn[:-len(".json")]
        d = json.load(open(os.path.join(TOPLEVEL, fn)))
        declared = {k: int(v.get("num_instances", 1))
                    for k, v in d["networks"].items() if not v.get("period")}
        if not declared:
            continue
        logs = sorted(glob.glob(os.path.join(
            XSWEEP, "runs", cell[len("networks_"):] + "__*", "rep1", "run.log")))
        got = None
        if logs:
            txt = open(logs[0], errors="replace").read()
            m = re.search(r"MODELBLASTER_XPURT_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
                          r"MODELBLASTER_XPURT_TRACE_END", txt, re.S)
            if m:
                rows = list(csv.DictReader(io.StringIO(m.group(1).strip())))
                got = {n: len({r["instance"] for r in rows if r["network"] == n})
                       for n in declared}
        out[cell] = {"declared": declared, "xpurt": got,
                     "equal": got == declared if got is not None else None}
    return out


def load():
    with open(os.path.join(SWEEP, "measured.json")) as f:
        return json.load(f)


def by_cell(doc, arm="main"):
    """Runs grouped by cell.

    The two arms live in one measured.json and must not be mixed: the main arm
    is the 42 sched_algo_sweep10 cells (`networks_*`), the 3net arm is the
    RoSE 3-network shapes (`3net_*`), and they have different comparators.
    """
    out = collections.defaultdict(list)
    for r in doc["runs"].values():
        is_main = r["cell"].startswith("networks_")
        if (arm == "main") != is_main:
            continue
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
    npwork = xpurt_np_instances()
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
            np_work_equal=(npwork.get(cell, {}).get("equal")
                           if not np_degen else None),
            np_work_declared=(npwork.get(cell, {}).get("declared")
                              if not np_degen else None),
            np_work_xpurt=(npwork.get(cell, {}).get("xpurt")
                           if not np_degen else None),
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
            # The study's OWN recommendation for the offline/build-time path,
            # and therefore the right opponent for a fixed-pinning baseline.
            # Phase 4 tiered by board time -- all 12 solvers on 12 cells,
            # winner+greedy on the other 30 -- so this was measured on 12 of 42
            # cells when the first version of this analysis ran, and on the
            # other 30 the "best measured solver" was really greedy or its
            # near-tie. That gap is now closed.
            xrt_np_warmbest_ms=(x.get("solvers", {}).get("cpsat:warmbest")
                                or {}).get("np_median_ms"),
            xrt_warmbest_ms=(x.get("solvers", {}).get("cpsat:warmbest")
                             or {}).get("makespan_median_ms"),
            xrt_has_warmbest=("cpsat:warmbest" in (x.get("solvers") or {})),
        )
        # ROS / XPU-RT. >1 means the scheduler wins.
        row["ros_over_xrt_best"] = ratio(row["ros_best_ms"], row["xrt_best_ms"])
        row["ros_over_xrt_greedy"] = ratio(row["ros_best_ms"], row["xrt_greedy_ms"])
        row["ros_over_xrt_np_best"] = ratio(row["ros_np_best_ms"],
                                            row["xrt_np_best_ms"])
        row["ros_over_xrt_np_warmbest"] = ratio(row["ros_np_best_ms"],
                                                row["xrt_np_warmbest_ms"])
        row["inside_noise_wall"] = inside(row["ros_over_xrt_best"], NOISE_WALL_PCT)
        row["inside_noise_np"] = inside(row["ros_over_xrt_np_best"], NOISE_NP_PCT)
        row["inside_noise_np_warmbest"] = inside(row["ros_over_xrt_np_warmbest"],
                                                 NOISE_NP_PCT)
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
    #
    # THE HEADLINE OBJECTIVE IS THE NON-PERIODIC MAKESPAN. Both runtimes are
    # measured on how long the non-periodic work takes while the periodic
    # tasks' constraints are honoured; a solution that finishes that work
    # having had to run fewer periodic instances is a better solution, not a
    # different workload. The all-operations wall clock is reported too, but
    # it is pinned by the last periodic RELEASE on most cells and is the
    # weaker of the two comparisons.
    #
    # A cell is only in the headline if the NON-PERIODIC work itself matches:
    # `saturation` declares yolov8_nano_se twice and XPU-RT scheduled it once,
    # so on those four cells the pinning baseline times twice the work.
    npc = [r for r in rows if r["ros_over_xrt_np_best"] and not r["np_degenerate"]
           and r["np_work_equal"]]
    npx = [r for r in rows if r["ros_over_xrt_np_best"] and not r["np_degenerate"]
           and r["np_work_equal"] is False]
    comp = [r for r in rows if r["ros_over_xrt_best"]]
    # Same 26 cells, scored against cpsat:warmbest specifically rather than
    # against whichever solver happened to be measured fastest. Both are
    # reported and each is labelled; neither is allowed to stand in for the
    # other.
    npw = [r for r in npc if r["ros_over_xrt_np_warmbest"]]
    res["headline"] = {
        "objective": "non-periodic makespan (primary); wall clock (secondary)",
        "np_opponent_primary": "cpsat:warmbest (the sweep's own recommendation "
                               "for the offline/build-time path)",
        "np_opponent_secondary": "best measured solver per cell (the most "
                                 "favourable reading for the scheduler)",
        "np_cells_compared": len(npc),
        "np_cells_with_warmbest": len(npw),
        "np_cells_without_warmbest": [r["cell"] for r in npc
                                      if not r["ros_over_xrt_np_warmbest"]],
        "np_warmbest_ros_over_xrt_median": round(statistics.median(
            [r["ros_over_xrt_np_warmbest"] for r in npw]), 4) if npw else None,
        "np_warmbest_ros_faster_cells": sum(
            1 for r in npw if r["ros_over_xrt_np_warmbest"] < 1),
        "np_warmbest_xrt_faster_cells": sum(
            1 for r in npw if r["ros_over_xrt_np_warmbest"] > 1),
        "np_warmbest_inside_noise_cells": sum(
            1 for r in npw if r["inside_noise_np_warmbest"]),
        "np_warmbest_worst_for_ros": max(
            npw, key=lambda r: r["ros_over_xrt_np_warmbest"])["cell"] if npw else None,
        "np_warmbest_worst_for_ros_ratio": max(
            (r["ros_over_xrt_np_warmbest"] for r in npw), default=None),
        "np_warmbest_best_for_ros": min(
            npw, key=lambda r: r["ros_over_xrt_np_warmbest"])["cell"] if npw else None,
        "np_warmbest_best_for_ros_ratio": min(
            (r["ros_over_xrt_np_warmbest"] for r in npw), default=None),
        "np_cells_excluded_unequal_work": [r["cell"] for r in npx],
        "np_cells_degenerate_no_aperiodic": [r["cell"] for r in rows
                                             if r["np_degenerate"]],
        "np_ros_over_xrt_median": round(statistics.median(
            [r["ros_over_xrt_np_best"] for r in npc]), 4) if npc else None,
        "np_ros_faster_cells": sum(1 for r in npc if r["ros_over_xrt_np_best"] < 1),
        "np_xrt_faster_cells": sum(1 for r in npc if r["ros_over_xrt_np_best"] > 1),
        "np_inside_noise_cells": sum(1 for r in npc if r["inside_noise_np"]),
        "np_worst_for_ros": max(npc, key=lambda r: r["ros_over_xrt_np_best"])["cell"]
                            if npc else None,
        "np_worst_for_ros_ratio": max(
            (r["ros_over_xrt_np_best"] for r in npc), default=None),
        "np_best_for_ros": min(npc, key=lambda r: r["ros_over_xrt_np_best"])["cell"]
                           if npc else None,
        "np_best_for_ros_ratio": min(
            (r["ros_over_xrt_np_best"] for r in npc), default=None),
        "wall_cells_compared": len(comp),
        "wall_ros_over_xrt_median": round(statistics.median(
            [r["ros_over_xrt_best"] for r in comp]), 4) if comp else None,
        "wall_ros_faster_cells": sum(1 for r in comp if r["ros_over_xrt_best"] < 1),
        "wall_xrt_faster_cells": sum(1 for r in comp if r["ros_over_xrt_best"] > 1),
        "wall_inside_noise_cells": sum(1 for r in comp if r["inside_noise_wall"]),
        "placement_spread_median": round(statistics.median(
            [r["ros_placement_spread"] for r in rows
             if r["ros_placement_spread"]]), 4),
        "placement_spread_max": max(
            (r["ros_placement_spread"] for r in rows
             if r["ros_placement_spread"]), default=None),
        "np_placement_spread_max": max(
            (ratio(r["ros_np_worst_ms"], r["ros_np_best_ms"]) for r in rows),
            default=None),
        "predicted_best_hit_rate": round(
            sum(1 for r in rows if r["predicted_best_is_measured_best"])
            / max(1, len([r for r in rows if r["n_measured"] > 1])), 4),
        "predicted_np_best_hit_rate": round(
            sum(1 for r in rows if r["predicted_np_best_is_measured_np_best"])
            / max(1, len([r for r in rows if r["n_measured"] > 1])), 4),
        "ros_rep_spread_median_pct": round(statistics.median(
            [pct(v["makespan_spread_ms"], v["makespan_median_ms"])
             for v in doc["runs"].values() if v["ok"]]), 2),
        "ros_rep_spread_max_pct": round(max(
            pct(v["makespan_spread_ms"], v["makespan_median_ms"])
            for v in doc["runs"].values() if v["ok"]), 2),
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

    print("PRIMARY: non-periodic makespan (the objective both runtimes are "
          "scored on)\n")
    print(f'{"cell":30s} {"m":>3s} {"ROS np":>9s} | {"warmbest":>9s} '
          f'{"ratio":>7s} {"noise":>6s} | {"best-of":>9s} {"ratio":>7s} '
          f'{"noise":>6s} {"solver":>16s}   note')
    for r in rows:
        note = ""
        if r["np_degenerate"]:
            note = "no aperiodic net -- np == wall"
        elif r["np_work_equal"] is False:
            note = (f'UNEQUAL np work: declared {r["np_work_declared"]}, '
                    f'XPU-RT ran {r["np_work_xpurt"]}')
        if not r.get("xrt_has_warmbest"):
            note = (note + "; " if note else "") + "no cpsat:warmbest measured"
        x = doc["xpurt"].get(r["cell"], {})
        f3 = lambda v, w: (f"{v:{w}.3f}" if v else "-".rjust(w))
        nz = lambda v, b: ("in" if b else "OUT").rjust(6) if v else "-".rjust(6)
        print(f'{r["cell"][len("networks_"):]:30s} {r["n_measured"]:3d} '
              f'{r["ros_np_best_ms"]:9.3f} | '
              f'{f3(r["xrt_np_warmbest_ms"], 9)} '
              f'{f3(r["ros_over_xrt_np_warmbest"], 7)} '
              f'{nz(r["xrt_np_warmbest_ms"], r["inside_noise_np_warmbest"])} | '
              f'{f3(r["xrt_np_best_ms"], 9)} '
              f'{f3(r["ros_over_xrt_np_best"], 7)} '
              f'{nz(r["xrt_np_best_ms"], r["inside_noise_np"])} '
              f'{str(x.get("best_np_solver") or "-"):>16s}   {note}')
    print("\nSECONDARY: all-operations wall clock (release-bound on most "
          "cells; periodic instance counts may legitimately differ)\n")
    print(f'{"cell":32s} {"ROS ms":>11s} {"sprd%":>6s} {"XRT ms":>11s} '
          f'{"ROS/XRT":>8s} {"noise?":>7s} {"miss/rep":>8s}')
    for r in rows:
        print(f'{r["cell"][len("networks_"):]:32s} {r["ros_best_ms"]:11.3f} '
              f'{(r["ros_best_spread_pct"] or 0):6.1f} '
              f'{(r["xrt_best_ms"] or 0):11.3f} '
              f'{(r["ros_over_xrt_best"] or 0):8.3f} '
              f'{"in" if r["inside_noise_wall"] else "OUT":>7s} '
              f'{r["ros_best_missed"]:8.1f}')
    h = res["headline"]
    print("\nheadline:", json.dumps(h, indent=1))
    return 0


def cmd_tables3net(args):
    """The 3net arm: RoSE's 3-network shapes, pinning vs XPU-RT on this board.

    Both sides ran here, on the same three lanes, in the same session. The
    comparison is the NON-PERIODIC makespan; XPU-RT's schedules trim periodic
    instances that fall after it, which is a better solution to the same
    problem, not a different workload. Two shapes carry no aperiodic network
    at all -- there the objective degenerates to the wall clock, and it is
    quoted only because both sides happened to execute identical entry counts.

    TWO OPPONENTS, BOTH REPORTED, NEITHER STANDING IN FOR THE OTHER -- exactly
    as §4 does for the main arm:

      primary    `cpsat:warmbest`, the sweep10 study's own recommendation for
                 the offline/build-time path. This is the number to quote,
                 because it is the one a user following that study would get.
      secondary  the best measured solver on that shape, an oracle over the
                 four measured here. The most favourable reading available to
                 the scheduler.

    The first version of this arm had neither: it scored against cold `cpsat`
    and, because the driver's dedupe key hashed a JSON field that does not
    exist, against a `cpsat` that had never actually been run -- every solver
    was recorded as a duplicate of `greedy`, so all three columns were greedy.
    """
    doc = load()
    xrt = json.load(open(os.path.join(SWEEP, "results", "xpurt3net.json")))["shapes"]
    plans = {}
    for fn in sorted(os.listdir(os.path.join(SWEEP, "plans3net"))):
        pl = json.load(open(os.path.join(SWEEP, "plans3net", fn)))
        plans[pl["cell"]] = pl
    runs = by_cell(doc, arm="3net")
    rows = []
    for cell, rr in sorted(runs.items()):
        rr = [r for r in rr if r["ok"]]
        if not rr:
            continue
        x = xrt.get(cell, {})
        best = min(rr, key=rank_key)
        np_best = min(rr, key=np_rank_key)
        aper = [n for n, v in plans[cell]["networks"].items()
                if v["period_ms"] is None]
        sv = x.get("solvers") or {}
        wb = sv.get("cpsat:warmbest") or {}
        rows.append(dict(
            shape=cell, sources=plans[cell]["sources"],
            n_legal=plans[cell]["n_legal"], n_measured=len(rr),
            np_degenerate=not aper,
            ros_np_ms=np_best["np_median_ms"],
            ros_np_spread_pct=pct(np_best["np_spread_ms"], np_best["np_median_ms"]),
            ros_np_label=np_best["label"],
            ros_np_worst_ms=max(r["np_median_ms"] for r in rr),
            ros_wall_ms=best["makespan_median_ms"],
            xrt_np_ms=x.get("best_np_ms"), xrt_wall_ms=x.get("best_makespan_ms"),
            xrt_solver=x.get("best_solver"),
            xrt_np_solver=x.get("best_np_solver"),
            xrt_np_instances=x.get("np_instances"),
            # the recommended solver, and how it was obtained: a shape whose
            # warmbest schedule was byte-identical to one already measured
            # carries `measured_via` and consumed no board time.
            xrt_np_warmbest_ms=wb.get("np_median_ms"),
            xrt_warmbest_ms=wb.get("median_ms"),
            xrt_warmbest_spread_pct=pct(wb.get("np_spread_ms"),
                                        wb.get("np_median_ms")),
            xrt_warmbest_via=wb.get("measured_via"),
            xrt_solvers_measured=sorted(sv),
            xrt_n_unique_schedules=x.get("n_unique_schedules"),
            ros_over_xrt_np=ratio(np_best["np_median_ms"], x.get("best_np_ms")),
            ros_over_xrt_np_warmbest=ratio(np_best["np_median_ms"],
                                           wb.get("np_median_ms")),
            ros_over_xrt_wall=ratio(best["makespan_median_ms"],
                                    x.get("best_makespan_ms")),
            placement_spread_np=ratio(max(r["np_median_ms"] for r in rr),
                                      np_best["np_median_ms"]),
        ))
        rows[-1]["inside_noise_np"] = inside(rows[-1]["ros_over_xrt_np"],
                                             NOISE_NP_PCT)
        rows[-1]["inside_noise_np_warmbest"] = inside(
            rows[-1]["ros_over_xrt_np_warmbest"], NOISE_NP_PCT)
    real = [r for r in rows if not r["np_degenerate"]]
    wbr = [r for r in real if r["ros_over_xrt_np_warmbest"]]
    head = {
        "np_opponent_primary": "cpsat:warmbest (the sweep10 study's own "
                               "recommendation for the offline/build-time path)",
        "np_opponent_secondary": "best measured solver per shape (an oracle "
                                 "over the solvers measured here)",
        "shapes_with_an_aperiodic_network": len(real),
        "shapes_with_warmbest": len(wbr),
        "warmbest_median": round(statistics.median(
            [r["ros_over_xrt_np_warmbest"] for r in wbr]), 4) if wbr else None,
        "warmbest_ros_faster": sum(1 for r in wbr
                                   if r["ros_over_xrt_np_warmbest"] < 1),
        "warmbest_xrt_faster": sum(1 for r in wbr
                                   if r["ros_over_xrt_np_warmbest"] > 1),
        "warmbest_inside_noise": [r["shape"] for r in wbr
                                  if r["inside_noise_np_warmbest"]],
        "warmbest_worst_for_ros": max(
            wbr, key=lambda r: r["ros_over_xrt_np_warmbest"])["shape"] if wbr else None,
        "warmbest_worst_for_ros_ratio": max(
            (r["ros_over_xrt_np_warmbest"] for r in wbr), default=None),
        "warmbest_best_for_ros": min(
            wbr, key=lambda r: r["ros_over_xrt_np_warmbest"])["shape"] if wbr else None,
        "warmbest_best_for_ros_ratio": min(
            (r["ros_over_xrt_np_warmbest"] for r in wbr), default=None),
        "best_median": round(statistics.median(
            [r["ros_over_xrt_np"] for r in real]), 4) if real else None,
        "best_ros_faster": sum(1 for r in real if r["ros_over_xrt_np"] < 1),
        "best_xrt_faster": sum(1 for r in real if r["ros_over_xrt_np"] > 1),
        "best_inside_noise": [r["shape"] for r in real if r["inside_noise_np"]],
        "shapes_deduped_onto_an_existing_run": {
            r["shape"]: r["xrt_warmbest_via"] for r in rows
            if r["xrt_warmbest_via"]},
        "np_degenerate_shapes": [r["shape"] for r in rows if r["np_degenerate"]],
    }
    out = {"_comment":
           "The 3net arm. ROS whole-model pinning vs XPU-RT scheduling on the "
           "same three lanes of the same board, both measured in this "
           "campaign. Primary objective: the non-periodic makespan, scored "
           "against `cpsat:warmbest` -- the solver the sweep10 study "
           "recommends -- with the best measured solver reported second as an "
           "oracle. XPU-RT trims periodic instances falling after the "
           "objective; that is a better solution to the same problem, so the "
           "wall clock is NOT the comparison except on the two shapes with no "
           "aperiodic network, where both sides executed identical entry "
           "counts.",
           "noise_floor_np_pct": NOISE_NP_PCT,
           "headline": head, "shapes": rows}
    p = os.path.join(SWEEP, "results", "analysis_3net.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {p}\n")
    print(f'noise floor on the objective: +/-{NOISE_NP_PCT}% '
          f'(the XPU-RT sweep\'s own measured rep spread)\n')
    print(f'{"shape":26s} {"n":>3s}{"m":>3s} {"ROS np":>9s} {"sprd%":>6s} | '
          f'{"warmbest":>9s} {"ratio":>7s} {"noise":>6s} | '
          f'{"best-of":>9s} {"ratio":>7s} {"noise":>6s} {"solver":>10s}  note')
    for r in rows:
        note = "no aperiodic net -- np == wall" if r["np_degenerate"] else ""
        if r["xrt_warmbest_via"]:
            note = (note + "; " if note else "") + \
                   f'warmbest == {r["xrt_warmbest_via"].split("__")[-1]} (dedupe)'
        f3 = lambda v, w: (f"{v:{w}.3f}" if v else "-".rjust(w))
        nz = lambda v, b: ("in" if b else "OUT").rjust(6) if v else "-".rjust(6)
        print(f'{r["shape"]:26s} {r["n_legal"]:3d}{r["n_measured"]:3d} '
              f'{r["ros_np_ms"]:9.3f} {(r["ros_np_spread_pct"] or 0):6.1f} | '
              f'{f3(r["xrt_np_warmbest_ms"], 9)} '
              f'{f3(r["ros_over_xrt_np_warmbest"], 7)} '
              f'{nz(r["xrt_np_warmbest_ms"], r["inside_noise_np_warmbest"])} | '
              f'{f3(r["xrt_np_ms"], 9)} {f3(r["ros_over_xrt_np"], 7)} '
              f'{nz(r["xrt_np_ms"], r["inside_noise_np"])} '
              f'{str(r["xrt_np_solver"] or "-"):>10s}  {note}')
    print("\nheadline:", json.dumps(head, indent=1))
    return 0


def cmd_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

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

    # ---- 4. the non-periodic objective, which is THE comparison --------
    npr = [r for r in res["cells"] if r["ros_over_xrt_np_best"]]
    npr.sort(key=lambda r: r["ros_over_xrt_np_best"])
    fig, ax = plt.subplots(figsize=(10, 0.30 * len(npr) + 2.4))
    y = np.arange(len(npr))
    vals, cols, labs = [], [], []
    for r in npr:
        vals.append(r["ros_over_xrt_np_best"])
        if r["np_work_equal"] is False:
            cols.append("#b0b0b0")
            labs.append(r["cell"][len("networks_"):] + "  (unequal np work)")
        elif r["np_degenerate"]:
            cols.append("#9aa5b1")
            labs.append(r["cell"][len("networks_"):] + "  (no aperiodic net)")
        else:
            cols.append("#3b7dd8" if vals[-1] < 1 else "#d1495b")
            labs.append(r["cell"][len("networks_"):])
    ax.barh(y, vals, color=cols, height=0.72)
    ax.axvline(1.0, color="k", lw=1)
    ax.axvspan(1 - NOISE_NP_PCT / 100, 1 + NOISE_NP_PCT / 100, color="k",
               alpha=0.10, lw=0, label=f"noise floor ±{NOISE_NP_PCT}%")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("ROS whole-model pinning ÷ measured XPU-RT, NON-PERIODIC "
                  "makespan\n(<1: pinning is faster; grey bars are not part of "
                  "the headline)")
    ax.set_title("The comparison: how long the non-periodic work takes while "
                 "the periodic tasks' constraints hold\n"
                 "best legal placement vs best measured solver, medians of 3 reps",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SWEEP, "plots", "ros_vs_xpurt_nonperiodic.png"), dpi=150)
    plt.close(fig)

    # ---- 5. the 3net arm ----------------------------------------------
    #
    # Same discipline as `plot_comparison.py`, for the same reasons: log2 ratio
    # axis so a 2x win and a 2x loss are equidistant from 1.0; the noise band
    # drawn OVER the bars so a bar that ends inside it cannot be read as a
    # result; the two shapes with no aperiodic network in their own panel
    # rather than greyed inside the ranking.
    #
    # BOTH OPPONENTS ON ONE FIGURE. The bar is `cpsat:warmbest` -- the solver
    # the sweep10 study recommends, and the number to quote. The open marker is
    # the best measured solver on that shape, an oracle over the four measured
    # here. Two files would let a reader quote whichever they saw first; one
    # figure with both marks makes the gap between them the visible thing,
    # because on this arm that gap IS the finding.
    p3 = os.path.join(SWEEP, "results", "analysis_3net.json")
    if os.path.exists(p3):
        d3 = json.load(open(p3))
        r3, h3 = d3["shapes"], d3["headline"]
        band = 1.0 + NOISE_NP_PCT / 100.0
        COOL, WARM = "#2a78d6", "#e34948"
        NEUTRAL, EXCLUDED = "#f0efec", "#c3c2b7"
        SURFACE = "#fcfcfb"
        INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
        GRID, BASELINE = "#e1e0d9", "#c3c2b7"

        real = sorted([r for r in r3 if not r["np_degenerate"]],
                      key=lambda r: r["ros_over_xrt_np_warmbest"] or 9)
        degen = sorted([r for r in r3 if r["np_degenerate"]],
                       key=lambda r: r["ros_over_xrt_np_warmbest"] or 9)

        def panel(ax, rows, excluded, title):
            y = np.arange(len(rows))[::-1]
            for yi, r in zip(y, rows):
                v = r["ros_over_xrt_np_warmbest"]
                col = EXCLUDED if excluded else (WARM if v > 1.0 else COOL)
                lo, hi = min(1.0, v), max(1.0, v)
                ax.barh(yi, hi - lo, left=lo, height=0.60, color=col,
                        edgecolor=SURFACE, linewidth=0.8, zorder=3)
                b = r["ros_over_xrt_np"]
                if b:
                    ax.plot([b], [yi], marker="D", ms=5.2, mfc="none",
                            mec=INK2, mew=1.2, zorder=5, linestyle="none")
            ax.set_yticks(y)
            ax.set_yticklabels([r["shape"][len("3net_"):] for r in rows],
                               fontsize=8.5, color=INK2)
            ax.set_xscale("log", base=2)
            # The main arm spans 0.52-2.45 and needs a wide axis; this arm
            # spans 0.865-1.025 and a wide one would compress every bar into a
            # smear at 1.0. The limits are kept RECIPROCAL (0.78 and 1/0.78)
            # so the log axis stays symmetric about 1.0 -- a win and a loss of
            # equal magnitude are still equidistant, which is the whole reason
            # the axis is log.
            ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.25])
            ax.set_xticklabels(["0.80", "0.90", "1.00", "1.10", "1.25"],
                               fontsize=8.5)
            ax.set_xlim(0.78, 1 / 0.78)
            ax.set_ylim(-0.75, len(rows) - 0.25)
            ax.axvspan(1 / band, band, color=NEUTRAL, alpha=0.85, zorder=1)
            ax.axvline(1.0, color=BASELINE, lw=1.2, zorder=2)
            ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(False)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(BASELINE)
            ax.tick_params(colors=MUTED, labelsize=8.5, length=2)
            ax.set_title(title, fontsize=10.0, color=INK, loc="left", pad=7)

        fig = plt.figure(figsize=(11.4, 5.9))
        fig.patch.set_facecolor(SURFACE)
        gs = fig.add_gridspec(2, 1, height_ratios=[len(real), max(len(degen), 1)],
                              hspace=0.42)
        ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
        for a in (ax, bx):
            a.set_facecolor(SURFACE)
        panel(ax, real, False,
              f"The comparison — {len(real)} shapes with an aperiodic network")
        panel(bx, degen, True,
              f"No aperiodic network ({len(degen)}) — the objective degenerates "
              f"to the wall clock")
        bx.title.set_fontsize(9.0); bx.title.set_color(INK2)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Patch(facecolor=COOL, label="pinning faster"),
            Patch(facecolor=WARM, label="scheduler faster"),
            Patch(facecolor=NEUTRAL, label=f"±{NOISE_NP_PCT}% noise floor"),
            Line2D([], [], marker="D", ms=5.2, mfc="none", mec=INK2, mew=1.2,
                   linestyle="none", label="vs best measured solver (oracle)"),
        ], fontsize=8.2, frameon=False, labelcolor=INK2, loc="center right",
            bbox_to_anchor=(1.0, 0.30), handletextpad=0.5, borderpad=0.2,
            labelspacing=0.32)

        fig.suptitle("3net arm — RoSE's 3-network shapes, both sides measured "
                     "here on the same three lanes",
                     fontsize=13.0, color=INK, x=0.007, ha="left", y=0.988)
        fig.text(0.007, 0.922,
                 f"ROS whole-network pinning ÷ measured XPU-RT on the "
                 f"non-periodic makespan, medians of 3 reps.\n"
                 f"Bars — against `cpsat:warmbest`, the sweep10 study's own "
                 f"recommendation: median {h3['warmbest_median']:.3f}, "
                 f"{h3['warmbest_ros_faster']} pinning / "
                 f"{h3['warmbest_xrt_faster']} scheduler faster, "
                 f"{len(h3['warmbest_inside_noise'])} of {len(real)} inside "
                 f"the band.\n"
                 f"Markers — against the best measured solver per shape, an "
                 f"oracle over the four measured here: median "
                 f"{h3['best_median']:.3f}.  Both are reported; neither "
                 f"stands in for the other.",
                 fontsize=9.0, color=INK2, ha="left", va="top", linespacing=1.5)
        fig.text(0.007, 0.020,
                 "Log axis, reciprocal limits: a win and a loss of equal "
                 "magnitude are equidistant from 1.0. Bars anchored at 1.0; a "
                 "bar ending inside the grey band is not a result.",
                 fontsize=8.5, color=MUTED, ha="left")
        fig.subplots_adjust(left=0.215, right=0.985, top=0.745, bottom=0.100)
        out = os.path.join(SWEEP, "plots", "ros_vs_xpurt_3net.png")
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"  -> {out}")
    print("wrote figures to", os.path.join(SWEEP, "plots"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tables"); s.set_defaults(fn=cmd_tables)
    s = sub.add_parser("tables3net"); s.set_defaults(fn=cmd_tables3net)
    s = sub.add_parser("plots"); s.set_defaults(fn=cmd_plots)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
