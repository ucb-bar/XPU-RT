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


#: Lane precedence for breaking an EXACT tie in the isolation cost. It never
#: fires in this matrix -- the smallest margin between a network's best lane
#: and its runner-up is 12.3 % (`yolov8_nano_sc`, DSP 3.461 ms against CPU
#: 3.887 ms), which is outside the +/-9.18 % band, so no selection here is a
#: coin flip and the rule is stated for completeness rather than used.
LANE_ORDER = ("dsp", "cpu", "hta", "gpu")


def model_costs():
    with open(os.path.join(SWEEP, "model_costs.json")) as f:
        return json.load(f)["networks"]


def isolation_best(costs, net, legal=None):
    """The lane a network is fastest on **by itself**, ignoring co-tenants.

    This is the placement rule of the PRIMARY baseline: what a ROS user
    actually does. Pick each network's best backend from a per-model
    benchmark, one network at a time, and deploy. No enumeration, no search,
    no scoring pass over the joint placement -- that is a placement oracle no
    real user has, and it is reported separately (§ the oracle column) as an
    upper bound on what pinning could reach with perfect knowledge.

    The cost is the whole-model cost from `model_costs.json`: the sum of that
    network's binding tiles on one lane, out of the same frozen
    `cost_model.json` the XPU-RT sweep solved against. That file already
    excludes the `cpu@int8` alias, so the CPU number here is the CPU number a
    per-model benchmark would report.

    `legal` restricts to the lanes the cell actually offers. GPU is in the
    cost table but is not a pinning candidate (SETUP.md 1); it never wins an
    unrestricted argmin anywhere in this zoo either, so excluding it changes
    no selection -- asserted in `iso_placement`, not assumed.
    """
    per = {b: v["ms"] for b, v in costs[net].items()}
    if legal is not None:
        per = {b: v for b, v in per.items() if b in legal}
    if not per:
        return None
    return min(sorted(per, key=lambda b: LANE_ORDER.index(b)), key=per.get)


def iso_placement(costs, plan):
    """The isolation-best assignment for one cell, and what it had to give up.

    Returns `(assign, fallbacks, gpu_wins)`. A `fallback` is a network whose
    unrestricted isolation-best lane is not legal in this cell -- because the
    config does not declare it, or because some tile of that network does not
    compose there -- so it gets its best LEGAL lane instead and is recorded.
    That is not a defect of the rule: a user on a two-lane config has the same
    problem and makes the same substitution.
    """
    legal = {n: v["legal_backends"] for n, v in plan["networks"].items()}
    assign, fallbacks, gpu_wins = {}, [], []
    for n in plan["networks"]:
        free = isolation_best(costs, n)
        if free == "gpu":
            gpu_wins.append(n)
        b = isolation_best(costs, n, legal[n])
        if b is None:
            return None, None, None
        if free != b:
            fallbacks.append({"network": n, "isolation_best": free,
                              "legal_best": b,
                              "reason": ("not declared by this config"
                                         if free in ("cpu", "dsp", "hta", "gpu")
                                         else "no composable binding")})
        assign[n] = b
    return assign, fallbacks, gpu_wins


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
    costs = model_costs()
    plans = {}
    for fn in sorted(os.listdir(os.path.join(SWEEP, "plans"))):
        pl = json.load(open(os.path.join(SWEEP, "plans", fn)))
        plans[pl["cell"]] = pl
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

        # ---- the PRIMARY baseline: isolation-best pinning -----------------
        # One placement per cell, chosen per network from a per-model
        # benchmark with no knowledge of the co-tenants. This is what a ROS
        # user deploys; `ros_np_best_ms` above is the best of every legal
        # placement measured, which is a placement ORACLE and is kept as the
        # secondary reading and as an upper bound on pinning.
        pl = plans.get(cell)
        iso_assign, fbacks, gpu_wins = iso_placement(costs, pl)
        iso = next((r for r in runs if r["assign"] == iso_assign), None)
        row["iso_assign"] = iso_assign
        row["iso_fallbacks"] = fbacks
        row["iso_gpu_would_win"] = gpu_wins
        row["iso_measured"] = iso is not None
        if iso is not None:
            row["iso_id"] = iso["assignment"]
            row["iso_label"] = iso["label"]
            row["iso_np_ms"] = iso["np_median_ms"]
            row["iso_np_spread_pct"] = pct(iso["np_spread_ms"],
                                           iso["np_median_ms"])
            row["iso_np_corrected_ms"] = iso.get("np_corrected_median_ms")
            row["iso_wall_ms"] = iso["makespan_median_ms"]
            row["iso_missed"] = iso["missed_instances"]
            row["iso_rank"] = iso["rank"]
            row["iso_np_rank"] = iso["np_rank"]
            # what the naive rule leaves on the table against the oracle
            row["iso_over_oracle_np"] = ratio(iso["np_median_ms"],
                                              row["ros_np_best_ms"])
            row["iso_is_oracle"] = (iso["assignment"] == np_best["assignment"])

        # ---- the start-barrier correction, both sides, same rule ----------
        row["ros_np_best_corrected_ms"] = np_best.get("np_corrected_median_ms")
        row["ros_offset_median_ms"] = np_best.get("offset_median_ms")
        wbrec = (x.get("solvers", {}).get("cpsat:warmbest") or {})
        row["xrt_np_warmbest_corrected_ms"] = wbrec.get("np_corrected_median_ms")
        row["xrt_warmbest_offset_median_ms"] = wbrec.get("offset_median_ms")
        row["xrt_warmbest_release_bound_reps"] = wbrec.get("release_bound_reps")
        row["xrt_np_best_corrected_ms"] = x.get("best_np_corrected_ms")
        row["xrt_np_best_corrected_solver"] = x.get("best_np_corrected_solver")
        row["iso_over_xrt_np_warmbest"] = ratio(
            row.get("iso_np_ms"), row["xrt_np_warmbest_ms"])
        row["iso_over_xrt_np_warmbest_corrected"] = ratio(
            row.get("iso_np_corrected_ms"), row["xrt_np_warmbest_corrected_ms"])
        row["ros_over_xrt_np_warmbest_corrected"] = ratio(
            row["ros_np_best_corrected_ms"], row["xrt_np_warmbest_corrected_ms"])
        row["iso_over_xrt_np_best"] = ratio(row.get("iso_np_ms"),
                                            row["xrt_np_best_ms"])
        row["iso_over_xrt_np_best_corrected"] = ratio(
            row.get("iso_np_corrected_ms"), row["xrt_np_best_corrected_ms"])
        row["ros_over_xrt_np_best_corrected"] = ratio(
            row["ros_np_best_corrected_ms"], row["xrt_np_best_corrected_ms"])
        for k in ("iso_over_xrt_np_warmbest",
                  "iso_over_xrt_np_warmbest_corrected",
                  "ros_over_xrt_np_warmbest_corrected",
                  "iso_over_xrt_np_best", "iso_over_xrt_np_best_corrected",
                  "ros_over_xrt_np_best_corrected",
                  # the two pre-existing readings, under the uniform name the
                  # `readings` block and the figures index by. The older
                  # `inside_noise_np*` names stay so nothing that reads them
                  # breaks.
                  "ros_over_xrt_np_warmbest", "ros_over_xrt_np_best"):
            row["inside_noise_" + k] = inside(row[k], NOISE_NP_PCT)
        # did the correction move this cell further than the noise floor?
        for a, b, name in (
                ("iso_over_xrt_np_warmbest",
                 "iso_over_xrt_np_warmbest_corrected", "iso_warmbest"),
                ("ros_over_xrt_np_warmbest",
                 "ros_over_xrt_np_warmbest_corrected", "oracle_warmbest")):
            ra, rb = row.get(a), row.get(b)
            row["correction_move_pct_" + name] = (
                round((rb / ra - 1) * 100, 2) if ra and rb else None)
            row["correction_flips_" + name] = (
                bool(ra and rb and ((ra > 1) != (rb > 1))))
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


RATIO_KEYS = {
    ("isolation", "raw", "warmbest"): "iso_over_xrt_np_warmbest",
    ("isolation", "corrected", "warmbest"): "iso_over_xrt_np_warmbest_corrected",
    ("oracle", "raw", "warmbest"): "ros_over_xrt_np_warmbest",
    ("oracle", "corrected", "warmbest"): "ros_over_xrt_np_warmbest_corrected",
    ("isolation", "raw", "best"): "iso_over_xrt_np_best",
    ("isolation", "corrected", "best"): "iso_over_xrt_np_best_corrected",
    ("oracle", "raw", "best"): "ros_over_xrt_np_best",
    ("oracle", "corrected", "best"): "ros_over_xrt_np_best_corrected",
}


def comparable(rows, scope):
    """The cells a ratio may be quoted on.

    Two exclusions, both pre-existing and both about the WORK being timed, not
    about the runtime: a cell with no aperiodic network has no non-periodic
    objective (it degenerates to the wall clock and is reported separately),
    and a cell where XPU-RT scheduled fewer instances of the APERIODIC network
    is timing different work. Fewer PERIODIC instances is a better solution to
    the same problem and is not an exclusion.
    """
    out = [r for r in rows if not r["np_degenerate"] and r["np_work_equal"]]
    if scope != "all":
        out = [r for r in out if r["config"] == scope]
    return out


def summarise(rows, key, inside_key):
    v = [(r["cell"], r[key]) for r in rows if r.get(key)]
    if not v:
        return None
    rr = [x for _, x in v]
    return {
        "n_cells": len(v),
        "median": round(statistics.median(rr), 4),
        "pinning_faster": sum(1 for x in rr if x < 1),
        "scheduler_faster": sum(1 for x in rr if x > 1),
        "inside_noise": sum(1 for r in rows
                            if r.get(key) and r.get(inside_key)),
        "worst_for_pinning": max(v, key=lambda t: t[1]),
        "best_for_pinning": min(v, key=lambda t: t[1]),
        "cells": {c: x for c, x in sorted(v, key=lambda t: t[1])},
    }


def readings(rows):
    out = {}
    for scope in ("quad", "all", "hd", "dc", "cg"):
        sub = comparable(rows, scope)
        out[scope] = {}
        for (base, timing, opp), key in RATIO_KEYS.items():
            out[scope].setdefault(base, {}).setdefault(opp, {})[timing] = \
                summarise(sub, key, "inside_noise_" + key)
        # what the naive rule leaves on the table against the oracle
        gaps = [r["iso_over_oracle_np"] for r in sub if r.get("iso_over_oracle_np")]
        out[scope]["iso_over_oracle"] = {
            "n_cells": len(gaps),
            "median": round(statistics.median(gaps), 4) if gaps else None,
            "max": max(gaps) if gaps else None,
            "cells_where_naive_is_already_optimal":
                [r["cell"] for r in sub if r.get("iso_is_oracle")],
        }
        out[scope]["cells"] = [r["cell"] for r in sub]
        # how far the correction moved each cell, and whether it flipped one
        for name in ("iso_warmbest", "oracle_warmbest"):
            mv = [(r["cell"], r["correction_move_pct_" + name]) for r in sub
                  if r.get("correction_move_pct_" + name) is not None]
            out[scope].setdefault("correction", {})[name] = {
                "median_move_pct": round(statistics.median(
                    [x for _, x in mv]), 2) if mv else None,
                "cells_moved_more_than_noise":
                    {c: x for c, x in sorted(mv, key=lambda t: t[1])
                     if abs(x) > NOISE_NP_PCT},
                "cells_that_changed_direction":
                    [r["cell"] for r in sub if r.get("correction_flips_" + name)],
            }
    return out


def offset_distribution(doc):
    """The per-run first-dispatch offset on both harnesses.

    This is the evidence the correction rests on and it is reported as a
    distribution, not as a single number, because the delay is INTERMITTENT:
    the same point measures 0.025 ms on one rep and 1.638 ms on the next, so
    it moves a median of 3 rather than shifting every number equally.
    """
    def dist(v, extra=None):
        v = sorted(v)
        if not v:
            return None
        q = lambda f: round(v[min(len(v) - 1, int(f * len(v)))], 4)
        d = {"n_runs": len(v), "median_ms": round(statistics.median(v), 4),
             "p75_ms": q(0.75), "p90_ms": q(0.90), "max_ms": round(v[-1], 4),
             "runs_over_1ms": sum(1 for x in v if x > 1.0)}
        if extra:
            d.update(extra)
        return d

    ros_main, ros_3net = [], []
    for r in doc["runs"].values():
        (ros_main if r["cell"].startswith("networks_") else ros_3net).extend(
            r.get("offset_reps_ms") or [])
    xrt, worst = [], []
    for cell, e in doc["xpurt"].items():
        for sv, v in e["solvers"].items():
            if v.get("measured_via"):
                continue          # same run as its canonical point; count once
            o = v.get("offset_reps_ms") or []
            xrt.extend(o)
            if o:
                worst.append((round(statistics.median(o), 4), cell, sv))
    worst.sort(reverse=True)
    return {
        "_comment": (
            "First dispatch of a run, in that run's own time base. On the "
            "XPU-RT side the delay sits in `gate_ms` and not in `dep_wait_ms`, "
            "so it is the run loop's start gate releasing late and not a "
            "dependency. Root cause is left as future work: the runtime does "
            "two iterations (FLOWC_ITERATIONS=2) and the trace is the second, "
            "so first-touch on the fastRPC path or SCHED_FIFO lane spin-up "
            "bleeding across the iteration boundary are the candidates."),
        "ros_main_arm": dist(ros_main),
        "ros_3net_arm": dist(ros_3net),
        "xpurt_main_arm": dist(xrt),
        "xpurt_worst_points": [{"median_offset_ms": m, "cell": c, "solver": sv}
                               for m, c, sv in worst[:10]],
        "release_bound_xpurt_points": [
            f"{c}::{sv}" for c, e in doc["xpurt"].items()
            for sv, v in e["solvers"].items() if v.get("release_bound_reps")],
    }


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
    # ---------------------------------------------------------------- the
    # SCOPED readings. Four axes, every combination reported, each labelled:
    #
    #   scope     `quad` (the headline) or every config. The config axis is
    #             which lane SUBSET is available, and only `quad` describes
    #             hardware that exists -- a QRB5165 always has all four
    #             backends. `hd`, `dc` and `cg` are a lane-scarcity
    #             sensitivity study and they distort individual cells badly.
    #   baseline  `isolation` (the primary: each network on the lane it is
    #             fastest on ALONE, which is what a ROS user deploys) or
    #             `oracle` (the best of every legal placement measured -- an
    #             upper bound on pinning that no user has).
    #   timing    `raw` or `corrected` for the start-barrier offset.
    #   opponent  `cpsat:warmbest`, the sweep's own recommendation, or the
    #             best measured solver per cell.
    res["readings"] = readings(rows)
    res["primary"] = {
        "scope": "quad", "baseline": "isolation", "timing": "corrected",
        "opponent": "cpsat:warmbest",
        "what": "ROS whole-network pinning, each network on the lane it is "
                "fastest on in isolation, divided by measured XPU-RT running "
                "cpsat:warmbest, both re-timed from their own first dispatch, "
                "on the quad cells with aperiodic work and matching "
                "non-periodic instance counts.",
    }
    res["offsets"] = offset_distribution(doc)
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

    print("PRIMARY: non-periodic makespan, ISOLATION-BEST pinning against "
          "cpsat:warmbest,\n         both re-timed from their own first "
          "dispatch. `oracle` is the best of\n         every legal placement "
          "measured -- an upper bound, not a deployment.\n")
    print(f'{"cell":26s} {"m":>3s} {"isoNP":>9s} {"corr":>9s} | '
          f'{"wb":>9s} {"corr":>9s} | {"iso/wb":>8s} {"raw":>7s} {"noise":>6s} '
          f'| {"oracle":>8s} {"iso/ora":>8s}   note')
    for r in rows:
        note = ""
        if r["np_degenerate"]:
            note = "no aperiodic net -- np == wall"
        elif r["np_work_equal"] is False:
            note = (f'UNEQUAL np work: declared {r["np_work_declared"]}, '
                    f'XPU-RT ran {r["np_work_xpurt"]}')
        if r["measure_mode"] != "full":
            note = (note + "; " if note else "") + \
                   f'SAMPLED {r["n_measured"]}/{r["n_legal"]} -- the oracle ' \
                   f'column is a best-of-sample, not a minimum'
        if not r.get("xrt_has_warmbest"):
            note = (note + "; " if note else "") + "no cpsat:warmbest measured"
        f3 = lambda v, w: (f"{v:{w}.3f}" if v else "-".rjust(w))
        f4 = lambda v, w: (f"{v:{w}.4f}" if v else "-".rjust(w))
        nz = lambda v, b: ("in" if b else "OUT").rjust(6) if v else "-".rjust(6)
        print(f'{r["cell"][len("networks_"):]:26s} {r["n_measured"]:3d} '
              f'{f3(r.get("iso_np_ms"), 9)} {f3(r.get("iso_np_corrected_ms"), 9)} | '
              f'{f3(r["xrt_np_warmbest_ms"], 9)} '
              f'{f3(r["xrt_np_warmbest_corrected_ms"], 9)} | '
              f'{f4(r["iso_over_xrt_np_warmbest_corrected"], 8)} '
              f'{f4(r["iso_over_xrt_np_warmbest"], 7)} '
              f'{nz(r["xrt_np_warmbest_corrected_ms"], r["inside_noise_iso_over_xrt_np_warmbest_corrected"])} | '
              f'{f4(r["ros_over_xrt_np_warmbest_corrected"], 8)} '
              f'{f4(r.get("iso_over_oracle_np"), 8)}   {note}')
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
    print("\nHEADLINE, four ways (scope x baseline x timing), all against "
          "cpsat:warmbest:\n")
    for scope in ("quad", "all"):
        for base in ("isolation", "oracle"):
            for t in ("corrected", "raw"):
                v = res["readings"][scope][base]["warmbest"][t]
                if not v:
                    continue
                star = "  <-- THE HEADLINE" if (
                    scope, base, t) == ("quad", "isolation", "corrected") else ""
                print(f'  {scope:5s} {base:9s} {t:9s} n={v["n_cells"]:2d} '
                      f'median={v["median"]:.4f}  '
                      f'{v["pinning_faster"]} pinning / {v["scheduler_faster"]} '
                      f'scheduler, {v["inside_noise"]} inside the band{star}')
        g = res["readings"][scope]["iso_over_oracle"]
        print(f'  {scope:5s} what the placement search is worth: median '
              f'{g["median"]}, max {g["max"]}, naive already optimal on '
              f'{len(g["cells_where_naive_is_already_optimal"])} of '
              f'{g["n_cells"]}\n')
    print("headline:", json.dumps(res["headline"], indent=1))
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
    sys.path.insert(0, HERE)
    import pin3net
    costs3, _prov3 = pin3net.base_costs()
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
        # ---- the same two changes the main arm gets ----------------------
        # (1) the PRIMARY baseline is isolation-best pinning: one placement
        #     per shape, each network on the lane it is fastest on ALONE.
        #     `np_best` above is the best of the full enumeration, which is a
        #     placement ORACLE and stays as the secondary reading.
        # (2) both sides are re-timed from their own first dispatch.
        r0 = rows[-1]
        iso_assign, fbacks, gpu_wins = iso_placement(costs3, plans[cell])
        iso = next((q for q in rr if q["assign"] == iso_assign), None)
        r0["iso_assign"] = iso_assign
        r0["iso_fallbacks"] = fbacks
        r0["iso_measured"] = iso is not None
        if iso is not None:
            r0["iso_id"] = iso["assignment"]
            r0["iso_label"] = iso["label"]
            r0["iso_np_ms"] = iso["np_median_ms"]
            r0["iso_np_corrected_ms"] = iso.get("np_corrected_median_ms")
            r0["iso_np_spread_pct"] = pct(iso["np_spread_ms"],
                                          iso["np_median_ms"])
            r0["iso_over_oracle_np"] = ratio(iso["np_median_ms"],
                                             np_best["np_median_ms"])
            r0["iso_is_oracle"] = iso["assignment"] == np_best["assignment"]
        r0["ros_np_corrected_ms"] = np_best.get("np_corrected_median_ms")
        r0["ros_offset_median_ms"] = np_best.get("offset_median_ms")
        r0["xrt_np_warmbest_corrected_ms"] = wb.get("np_corrected_median_ms")
        r0["xrt_warmbest_offset_median_ms"] = wb.get("offset_median_ms")
        r0["xrt_np_corrected_ms"] = x.get("best_np_corrected_ms")
        r0["xrt_np_corrected_solver"] = x.get("best_np_corrected_solver")
        r0["iso_over_xrt_np_warmbest"] = ratio(r0.get("iso_np_ms"),
                                               wb.get("np_median_ms"))
        r0["iso_over_xrt_np_warmbest_corrected"] = ratio(
            r0.get("iso_np_corrected_ms"), wb.get("np_corrected_median_ms"))
        r0["ros_over_xrt_np_warmbest_corrected"] = ratio(
            r0["ros_np_corrected_ms"], wb.get("np_corrected_median_ms"))
        r0["iso_over_xrt_np_corrected"] = ratio(
            r0.get("iso_np_corrected_ms"), x.get("best_np_corrected_ms"))
        r0["ros_over_xrt_np_corrected"] = ratio(
            r0["ros_np_corrected_ms"], x.get("best_np_corrected_ms"))
        for k in ("iso_over_xrt_np_warmbest",
                  "iso_over_xrt_np_warmbest_corrected",
                  "ros_over_xrt_np_warmbest_corrected",
                  "iso_over_xrt_np_corrected", "ros_over_xrt_np_corrected"):
            r0["inside_noise_" + k] = inside(r0[k], NOISE_NP_PCT)
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
    # The same four-axis reading the main arm gets: isolation-best (primary)
    # against the enumeration oracle (secondary), raw against corrected.
    def sm(key):
        v = [(r["shape"], r[key]) for r in real if r.get(key)]
        if not v:
            return None
        rr2 = [x for _, x in v]
        return {"n_shapes": len(v),
                "median": round(statistics.median(rr2), 4),
                "pinning_faster": sum(1 for x in rr2 if x < 1),
                "scheduler_faster": sum(1 for x in rr2 if x > 1),
                "outside_noise": sum(1 for r in real if r.get(key)
                                     and not r.get("inside_noise_" + key)),
                "shapes": {c: x for c, x in sorted(v, key=lambda t: t[1])}}
    head["readings"] = {
        "isolation": {"warmbest": {
            "raw": sm("iso_over_xrt_np_warmbest"),
            "corrected": sm("iso_over_xrt_np_warmbest_corrected")}},
        "oracle": {"warmbest": {
            "raw": {"n_shapes": len(wbr),
                    "median": head["warmbest_median"],
                    "pinning_faster": head["warmbest_ros_faster"],
                    "scheduler_faster": head["warmbest_xrt_faster"],
                    "outside_noise": len(wbr) - len(head["warmbest_inside_noise"])},
            "corrected": sm("ros_over_xrt_np_warmbest_corrected")}},
    }
    head["iso_over_oracle"] = {
        "median": round(statistics.median(
            [r["iso_over_oracle_np"] for r in real
             if r.get("iso_over_oracle_np")]), 4),
        "max": max(r["iso_over_oracle_np"] for r in real
                   if r.get("iso_over_oracle_np")),
        "shapes_where_naive_is_already_optimal":
            [r["shape"] for r in real if r.get("iso_is_oracle")],
    }
    head["shapes_where_the_isolation_placement_was_not_measured"] = [
        r["shape"] for r in rows if not r.get("iso_measured")]
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

        # Bars are the PRIMARY reading: isolation-best pinning, corrected,
        # against `cpsat:warmbest`. Markers are the same shape's ORACLE --
        # the best of the full enumeration -- so the gap between marker and
        # bar is what a naive user leaves on the table, which is a result in
        # its own right rather than a second file someone might quote instead.
        BAR = "iso_over_xrt_np_warmbest_corrected"
        MRK = "ros_over_xrt_np_warmbest_corrected"
        real = sorted([r for r in r3 if not r["np_degenerate"]],
                      key=lambda r: r[BAR] or 9)
        degen = sorted([r for r in r3 if r["np_degenerate"]],
                       key=lambda r: r[BAR] or 9)
        hb = h3["readings"]["isolation"]["warmbest"]["corrected"]
        ho = h3["readings"]["oracle"]["warmbest"]["corrected"]

        def panel(ax, rows, excluded, title):
            y = np.arange(len(rows))[::-1]
            for yi, r in zip(y, rows):
                v = r[BAR]
                col = EXCLUDED if excluded else (WARM if v > 1.0 else COOL)
                lo, hi = min(1.0, v), max(1.0, v)
                ax.barh(yi, hi - lo, left=lo, height=0.60, color=col,
                        edgecolor=SURFACE, linewidth=0.8, zorder=3)
                b = r[MRK]
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
            ax.set_xlim(0.75, 1 / 0.75)
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
                   linestyle="none",
                   label="best measured placement (oracle upper bound)"),
        ], fontsize=8.2, frameon=False, labelcolor=INK2, loc="center right",
            bbox_to_anchor=(1.0, 0.30), handletextpad=0.5, borderpad=0.2,
            labelspacing=0.32)

        fig.suptitle("3net arm — RoSE's 3-network shapes, both sides measured "
                     "here on the same three lanes",
                     fontsize=13.0, color=INK, x=0.007, ha="left", y=0.988)
        fig.text(0.007, 0.922,
                 f"ROS whole-network pinning ÷ measured XPU-RT running "
                 f"`cpsat:warmbest` on the non-periodic makespan, medians of "
                 f"3 reps, both re-timed from their own first dispatch.\n"
                 f"Bars — each network pinned to the lane it is fastest on IN "
                 f"ISOLATION, which is what a ROS user deploys: median "
                 f"{hb['median']:.3f}, {hb['pinning_faster']} pinning / "
                 f"{hb['scheduler_faster']} scheduler faster, "
                 f"{len(real) - hb['outside_noise']} of {len(real)} inside "
                 f"the band.\n"
                 f"Markers — the best of every legal placement measured, a "
                 f"placement oracle nobody has: median {ho['median']:.3f}.  "
                 f"The gap between marker and bar is what the naive rule "
                 f"leaves on the table.",
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
