#!/usr/bin/env python3
"""The static per-node pinning model, costed from the BOARD'S OWN ROS 2 runs.

Tier B of docs/Baselines/ros_baseline_tiers.md. The Tier A ROS 2 baseline is a POLICY MODEL: one node per network, pinned to a static partition, the node's graph
run sequentially, a periodic timer releasing each instance (`scripts/ros_pinning_generic.py`,
`ros_pinning_periodic.py`). That model's per-node cost inputs come from an XPU-RT schedule of the
same workload -- XPU-RT's measured per-dispatch durations, summed per network, divided by a
standalone OC-shard speedup table (`ros_pinning_generic.py:63`) -- so the baseline pays the same
per-op compute as XPU-RT and the comparison isolates placement policy. This module (Tier B)
separates the two contributions: how much of that baseline is the MODEL and how much is the INPUT?

This module answers it by changing exactly one thing. Same model, same recurrence, same partitions,
same periodic release -- and per-node costs read from the K1's own ROS 2 C++ runs under
`results/codesign_feedback/ros_traced/`, where `ros_mb_chain_traced.cpp` brackets each node's kernel
call with `rdtime()` and writes one `node_callback` row per callback (its:288 for perception, :323
for nav, :346/:357 for control). Those rows are the per-node cost the model wants, measured where
the model claims to place it.

It is deliberately NOT a replacement for `ros_pinning_generic.py`: that one re-lays an arbitrary
workload spec, this one predicts the two quantities the board reports for the deployed
camera->perception->nav->control chain (camera->goal latency and control cadence) so that every
prediction has a measurement standing next to it.

    scripts/ros_pinning_profiled.py --costs       # the profiled per-node cost table, with spread
    scripts/ros_pinning_profiled.py --table       # predicted vs measured, per arm and camera rate
    scripts/ros_pinning_profiled.py --json out.json

WHAT THE MODEL IS (unchanged from the Tier A one).  Each node owns a static partition. Frame k is
released at k*T. A node starts frame k at max(arrival_k, partition_free) and runs for its whole cost,
serially; there is no cross-node overlap inside a partition and no per-op sharding across partitions.
Nothing is charged for DDS, for executor wake-up, or for a queue. The two extension terms this module
reports SEPARATELY, never folded into the model column, are named in `predict()`.

WHAT THE INPUTS ARE.  Median over the warm callbacks of a run, pooled across that arm's replicates.
The median, not the mean: the mean of a callback distribution with a rare 35 ms tail is not the cost
of the callback. p95 is carried alongside so a reader can see the tail the median hides.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
TRACED = os.path.join(RES, "ros_traced")
HZ = 24e6                 # rdtime on the K1, NOT the core clock (ros_mb_chain_traced.cpp:67)
WARMUP_MS = 3000.0        # discarded from every ROS run on the host, the same rule and the same
                          # anchor pull_ros_traced.py:108-110 uses, so these rows are the rows the
                          # run's own summary.json was rolled up from

# The stages, in chain order. `camera` has no traced callback: ros_mb_chain_traced.cpp:424 stamps
# the camera timer's rdtime into released.csv and publishes, but never brackets the callback, so the
# camera's own cost is not measured directly and is DERIVED in `camera_cost()` instead.
STAGES = ("camera", "perception", "nav", "control")

# The per-node costs the Tier A baseline was GIVEN, so a reader sees every input in one table.
# Both variants are real and both are in the paper's lineage, so both are carried: `submitted` is
# what panel I of the Tier A showdown was drawn from, `recost` is the board-recost pair the
# schedule sidecar warehouse_showdown_board_metrics.json reports 49.76 ms from. Each is the
# per-instance duration sum of that network in its schedule, already divided by the width-4
# speedup ros_pinning_generic.py:63 applies. Neither has a camera node at all.
ASSUMED = {
    "submitted": {"camera": 0.0, "perception": 24.353374, "nav": 3.621874, "control": 0.083,
                  "src": "schedules/scheduled_ros_partition_deployed.json"},
    "recost":    {"camera": 0.0, "perception": 30.704700, "nav": 4.905000, "control": 0.081,
                  "src": "schedules/scheduled_ros_partition_deployed_matched_board.json"},
}
ASSUMED_MS = ASSUMED["recost"]      # the variant the 49.76 ms headline is computed from

# ros_pinning_generic.py:63, the whole-net speedup a wider perception partition is credited with.
PERCEPTION_SPEEDUP = {1: 1.00, 2: 55.870 / 50.384, 4: 55.870 / 52.294, 8: 55.870 / 39.386}


# --------------------------------------------------------------------------------------------
# 1. the profiled inputs
# --------------------------------------------------------------------------------------------
def _pct(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))]


def run_costs(tag):
    """Per-node callback cost of one ros_traced run, in ms, plus what the run itself measured.

    A `node_callback` row's duration is exactly the kernel call, bracketed by rdtime with nothing
    else inside it -- not the publish, not the DDS take. That is the same quantity the pinning model
    calls a node's cost, which is what makes this a like-for-like substitution.
    """
    d = os.path.join(TRACED, tag)
    man = json.load(open(os.path.join(d, "manifest.json")))
    warm = int(man.get("run_t0_ticks") or 0) + WARMUP_MS * HZ / 1e3
    per = collections.defaultdict(list)
    harts = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(os.path.join(d, "trace.csv"))):
        if r["op"] != "node_callback" or int(r["actual_start_cycles"]) < warm:
            continue
        per[r["name"]].append((int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / HZ * 1e3)
        harts[r["name"]][r["worker_hart"]] += 1
    rel = [int(x["t_release_ticks"]) for x in csv.DictReader(open(os.path.join(d, "released.csv")))
           if int(x["t_release_ticks"]) >= warm]
    gaps = [(b - a) / HZ * 1e3 for a, b in zip(rel, rel[1:])]
    s = json.load(open(os.path.join(d, "summary.json")))
    cost = {n: {"n": len(v), "med": statistics.median(v), "p95": _pct(v, 0.95),
                "min": min(v), "max": max(v), "harts": sorted(harts[n])} for n, v in per.items()}
    return {"tag": tag, "manifest": man, "summary": s, "cost": cost,
            "release_gap_med_ms": statistics.median(gaps) if gaps else None,
            "n_release": len(rel)}


def procs_of(man):
    """The per-process view, whether the run recorded one process or several (ros_arms_catalog.py:30)."""
    p = man.get("processes")
    return p if isinstance(p, dict) and p else {"(single process)": man}


def proc_with(man, node):
    """The process that runs `node`, or the only one."""
    ps = procs_of(man)
    return next((v for v in ps.values() if node in str(v.get("nodes", "")).split(",")), next(iter(ps.values())))


def _split(tag):
    m = re.match(r"^(\d+)_(.+)_r(\d+)$", tag)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else (None, None, None)


def runs(arms=None, rates=None):
    """Every ros_traced run of the named arms, newest-first-agnostic, sorted (arm, rate, replicate)."""
    out = []
    for d in sorted(glob.glob(os.path.join(TRACED, "*_r[0-9]"))):
        tag = os.path.basename(d)
        hz, arm, rep = _split(tag)
        if arm is None or not os.path.exists(os.path.join(d, "trace.csv")):
            continue
        if arms and arm not in arms:
            continue
        if rates and hz not in rates:
            continue
        out.append((arm, hz, rep, tag))
    return sorted(out)


def profile(arms, rates=None):
    """Pool the per-node costs over every run of `arms`, split by the node's partition WIDTH.

    A node's cost is a property of the kernel and the harts it was given, not of the camera rate, so
    pooling across rates is legitimate -- and the pooled spread is the evidence for that claim: if
    the cost moved with the rate the min/max here would show it. It is NOT pooled across widths:
    perception on a 4-hart pool and perception serial are two different costs and are kept apart.
    """
    acc = collections.defaultdict(list)
    tail = collections.defaultdict(list)
    prov = collections.defaultdict(set)
    for arm, hz, rep, tag in runs(arms, rates):
        r = run_costs(tag)
        pool = int(proc_with(r["manifest"], "perception").get("yolo_pool", 0) or 0)
        for node, c in r["cost"].items():
            key = (node, pool if node == "perception" else 0)
            acc[key].append(c["med"])
            tail[key].append(c["p95"])
            prov[key].add(tag)
    return {k: {"n_runs": len(v), "med": statistics.median(v), "min": min(v), "max": max(v),
                "p95": statistics.median(tail[k]), "runs": sorted(prov[k])} for k, v in acc.items()}


def camera_cost(arms=("cp3", "p3", "p8", "part8"), rates=(45, 60, 75, 90, 120)):
    """The camera callback's cost, DERIVED -- it is the one input that is not traced.

    Above the saturation rate the camera timer no longer paces the percep process: the timer and the
    perception subscription share one executor thread, so the camera can only fire once per pass of
    that thread and released.csv's inter-release gap IS the process's service period. Subtracting the
    perception callback from it leaves the camera callback plus the executor's own per-pass overhead.
    This is a residual, so it is an UPPER bound on the camera's compute; it is reported as one.
    """
    v = []
    for arm, hz, rep, tag in runs(arms, rates):
        r = run_costs(tag)
        if "perception" not in r["cost"] or r["release_gap_med_ms"] is None:
            continue
        if r["release_gap_med_ms"] > 1.05 * 1000.0 / hz:     # saturated: releases slower than asked
            v.append(r["release_gap_med_ms"] - r["cost"]["perception"]["med"])
    return {"n": len(v), "med": statistics.median(v), "min": min(v), "max": max(v)} if v else None


# --------------------------------------------------------------------------------------------
# 2. the model
# --------------------------------------------------------------------------------------------
def predict(cost, camera_hz, ctrl_mode, ctrl_hz, qos_depth, camera_colocated_with_perception=True,
            horizon_s=17.0):
    """The Tier A pinning model, evaluated on `cost`. Returns the model, then the two extras.

    The model, verbatim from ros_pinning_generic.py:84-96: frame k is released at k*T; a node starts
    it at max(arrival, its partition's free time) and runs its whole cost; nothing overlaps inside a
    partition. camera->goal is stamped on the board at the control node's callback ENTRY
    (ros_mb_chain_traced.cpp:344), BEFORE mlp_control runs, so the modelled chain is perception + nav
    and the control cost is deliberately not in it.

    Two terms the model does not have are computed separately and never added into `chain_model_ms`:

      queue_ms  -- when the camera timer and the perception subscription share one executor thread
                   and the process is saturated, the thread fires the timer once per pass, so exactly
                   one unread frame stands in front of perception at all times. Depth-1 QoS overwrites
                   that frame instead of queueing it, so the term is gated on qos_depth >= 2. The
                   `p3_q1` arm is the falsification test for this and it is in the table.
      dds        -- the three topic hops the model charges nothing for (frame, detections, goal;
                   in these arms the first is intra-process and the other two cross processes).
                   Reported as the RESIDUAL of measured minus (model + queue), not predicted here.
    """
    T = 1000.0 / camera_hz
    cp, cn, cc = cost["perception"], cost["nav"], cost.get("camera", 0.0)

    # the model's own chain: sum of the pinned nodes' costs, nothing else
    chain_model = cp + cn

    # the model's throughput: the slowest partition on the chain paces it. The camera shares the
    # perception partition in every arm in the table, so its cost is on that partition too.
    service = cp + (cc if camera_colocated_with_perception else 0.0)
    goal_period = max(T, service)
    saturated = service > T

    # the model, run out: with no drop rule and no self-clocking, a saturated node's response grows
    # by (service - T) every frame and never settles. Reported so the divergence is visible.
    frames = max(1, int(horizon_s * 1000.0 / T))
    drift_ms = max(0.0, service - T) * frames

    queue_ms = service if (saturated and qos_depth >= 2 and camera_colocated_with_perception) else 0.0

    if ctrl_mode == "chained":
        ctrl_pred_hz = 1000.0 / goal_period
    else:
        ctrl_pred_hz = min(float(ctrl_hz), 1000.0 / cost["control"]) if cost.get("control") else float(ctrl_hz)
    return {"chain_model_ms": chain_model, "chain_model_queue_ms": chain_model + queue_ms,
            "queue_ms": queue_ms, "service_ms": service, "goal_period_ms": goal_period,
            "saturated": saturated, "drift_ms": drift_ms, "frames_in_horizon": frames,
            "ctrl_pred_hz": ctrl_pred_hz}


# --------------------------------------------------------------------------------------------
# 3. predicted vs measured
# --------------------------------------------------------------------------------------------
PINNED = ("cp3", "p3", "p8", "part8", "p3_q1", "p3_c50", "cp3n4", "cp3n4_d")


def table(arms=PINNED, rates=None, costs=None):
    """One row per (arm, camera rate): the model's prediction and what the board did."""
    if costs is None:
        costs = pooled_inputs()
    rows = []
    byarm = collections.defaultdict(list)
    for arm, hz, rep, tag in runs(arms, rates):
        byarm[(arm, hz)].append(tag)
    for (arm, hz), tags in sorted(byarm.items()):
        rs = [run_costs(t) for t in tags]
        man0 = rs[0]["manifest"]
        percep = proc_with(man0, "perception")
        ctlp = proc_with(man0, "control")
        pool = int(percep.get("yolo_pool", 0) or 0)
        # a run that predates the --ctrl-mode option records no mode and ran the 100 Hz wall timer
        # (ros_mb_chain_traced.cpp:355); only an explicit "chained" fires control off the goal
        ctrl_mode = ctlp.get("ctrl_mode") or "timer"
        ctrl_hz = float(ctlp.get("ctrl_hz", 100) or 100)
        qos = int(percep.get("qos_depth", 10) or 10)
        colocated = "camera" in str(percep.get("nodes", "")).split(",")
        c = {"perception": costs[("perception", pool)]["med"], "nav": costs[("nav", 0)]["med"],
             "control": costs[("control", 0)]["med"], "camera": costs[("camera", 0)]["med"]}
        p = predict(c, hz, ctrl_mode, ctrl_hz, qos, colocated)
        meas_chain = statistics.median([r["summary"]["e2e_goal_med_ms"] for r in rs
                                        if r["summary"].get("e2e_goal_med_ms")])
        meas_gap = statistics.median([r["summary"]["gap_mean_ms"] for r in rs if r["summary"].get("gap_mean_ms")])
        av = {k: predict(dict(v), hz, ctrl_mode, ctrl_hz, qos, colocated) for k, v in ASSUMED.items()}
        a = av["recost"]
        rows.append({"arm": arm, "camera_hz": hz, "runs": tags, "pool": pool, "ctrl_mode": ctrl_mode,
                     "qos_depth": qos, "saturated": p["saturated"],
                     "chain_assumed_submitted_ms": av["submitted"]["chain_model_ms"],
                     "chain_assumed_ms": a["chain_model_ms"], "chain_model_ms": p["chain_model_ms"],
                     "chain_model_queue_ms": p["chain_model_queue_ms"], "chain_measured_ms": meas_chain,
                     "chain_residual_ms": meas_chain - p["chain_model_queue_ms"],
                     "ctrl_assumed_submitted_hz": av["submitted"]["ctrl_pred_hz"],
                     "ctrl_assumed_hz": a["ctrl_pred_hz"], "ctrl_pred_hz": p["ctrl_pred_hz"],
                     "ctrl_measured_hz": 1000.0 / meas_gap, "ctrl_measured_gap_ms": meas_gap,
                     "drift_ms": p["drift_ms"]})
    return rows


# The spec the Tier A showdown's ROS arm was laid out on, read off that schedule's own instance
# release times (schedules/scheduled_ros_partition_deployed_matched_board.json): yolo released every
# 22 ms for 5 instances, nav every 20 for 6, control every 10 for 12, against the 23 ms perception
# budget results/codesign_feedback/warehouse_showdown_board_metrics.json states.
SUBMITTED_SPEC = {"yolo_period_ms": 22.0, "yolo_instances": 5, "budget_ms": 23.0,
                  "reported_instance": 2, "reported_response_ms": 49.764795263999986,
                  "reported_xpu_response_ms": 22.999776799999992}


def submitted_spec_response(perception_ms, spec=SUBMITTED_SPEC):
    """The Tier A showdown's own metric -- release-to-output for each perception instance.

    The recurrence, and only the recurrence: start_k = max(k*T, free), free += cost. No recost pass,
    so these will not reproduce the schedule JSON's numbers to the digit (that file was board-recost
    AFTER placement, which inflated durations while keeping start times). What it reproduces is the
    SHAPE -- a node whose cost exceeds its period falls further behind every instance -- which is the
    whole of what the metric reports.
    """
    T, n = spec["yolo_period_ms"], spec["yolo_instances"]
    free, out = 0.0, []
    for k in range(n):
        start = max(k * T, free)
        free = start + perception_ms
        out.append(free - k * T)
    return out


def pooled_inputs():
    """Every profiled input the table needs, each from the widest evidence available for it."""
    p = profile(("cp3", "p3", "p8", "part8", "p3_q1", "p3_c50", "cp3n4", "cp3n4_d",
                 "spin", "cspin", "ship", "cship", "multi", "vanilla4", "vanilla", "vanilla8"))
    out = {k: v for k, v in p.items()}
    cam = camera_cost()
    out[("camera", 0)] = {"n_runs": cam["n"], "med": cam["med"], "min": cam["min"], "max": cam["max"],
                          "p95": cam["max"],
                          "runs": ["derived: released.csv gap - perception callback, saturated runs"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", action="store_true", help="the profiled per-node cost table")
    ap.add_argument("--table", action="store_true", help="predicted vs measured, per arm and rate")
    ap.add_argument("--arms", default=",".join(PINNED))
    ap.add_argument("--submitted-spec", action="store_true",
                    help="the Tier A showdown's own release-to-output metric, both ways")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not (a.costs or a.table or a.submitted_spec):
        a.costs = a.table = a.submitted_spec = True

    costs = pooled_inputs()
    if a.costs:
        print("profiled per-node costs, K1, ros_traced (median of each run's warm callbacks, "
              "pooled over runs)\n")
        print("%-12s %5s %5s %8s %8s %8s %8s | %8s %6s | %8s %6s"
              % ("node", "pool", "runs", "med ms", "min", "max", "p95",
                 "submitted", "/prof", "recost", "/prof"))
        for (node, pool), v in sorted(costs.items(), key=lambda kv: (STAGES.index(kv[0][0]), kv[0][1])):
            cells = []
            for key in ("submitted", "recost"):
                asm = ASSUMED[key].get(node)
                show = asm is not None and (node != "perception" or pool == 4)
                cells += [("%8.3f" % asm) if show else "       -",
                          ("%5.2fx" % (asm / v["med"])) if (show and v["med"]) else "     -"]
            print("%-12s %5s %5d %8.3f %8.3f %8.3f %8.3f | %s %s | %s %s"
                  % (node, pool or "-", v["n_runs"], v["med"], v["min"], v["max"], v["p95"], *cells))
        print("")
        for k, v in ASSUMED.items():
            print("  %-9s = %s" % (k, v["src"]))
        print("  camera    = derived residual (upper bound), not a traced callback")
        print("")
        print("  the width-4 credit both assumed sets carry is %.3fx (ros_pinning_generic.py:63); the "
              "board's own\n  pool buys %.3fx (perception serial %.3f -> pool-4 %.3f), so the "
              "speedup assumption is %.2fx\n  pessimistic while the width-1 number it divides is "
              "%.2fx optimistic -- two errors in opposite\n  directions that partly cancel."
              % (PERCEPTION_SPEEDUP[4],
                 costs[("perception", 0)]["med"] / costs[("perception", 4)]["med"],
                 costs[("perception", 0)]["med"], costs[("perception", 4)]["med"],
                 (costs[("perception", 0)]["med"] / costs[("perception", 4)]["med"]) / PERCEPTION_SPEEDUP[4],
                 costs[("perception", 0)]["med"] / (ASSUMED["recost"]["perception"] * PERCEPTION_SPEEDUP[4])))

    if a.table:
        rows = table(tuple(a.arms.split(",")), costs=costs)
        print("\ncamera->goal chain (ms) and control cadence (Hz): model on profiled inputs vs the board\n")
        print("%-9s %4s %4s %7s %5s | %6s %6s %6s %6s %6s %6s | %5s %5s %5s %5s | %8s"
              % ("arm", "hz", "qos", "ctrl", "sat", "subm", "recost", "prof", "+queue", "meas", "resid",
                 "sHz", "rHz", "pHz", "mHz", "drift"))
        for r in rows:
            print("%-9s %4d %4d %7s %5s | %6.2f %6.2f %6.2f %6.2f %6.2f %+6.2f | %5.1f %5.1f %5.1f %5.1f | %8.0f"
                  % (r["arm"], r["camera_hz"], r["qos_depth"], r["ctrl_mode"], "yes" if r["saturated"] else "no",
                     r["chain_assumed_submitted_ms"], r["chain_assumed_ms"], r["chain_model_ms"],
                     r["chain_model_queue_ms"], r["chain_measured_ms"], r["chain_residual_ms"],
                     r["ctrl_assumed_submitted_hz"], r["ctrl_assumed_hz"], r["ctrl_pred_hz"],
                     r["ctrl_measured_hz"], r["drift_ms"]))
        res = [r["chain_residual_ms"] for r in rows]
        print("\n  residual (measured - [model + queue]) over %d rows: median %+.2f ms, min %+.2f, max %+.2f"
              % (len(res), statistics.median(res), min(res), max(res)))
        print("  the residual is the three topic hops (frame, detections, goal) and their executor")
        print("  wake-ups; the model charges nothing for them, and this is the size of what it omits.")
        print("\n  drift = what the recurrence max(k*T, free) alone says the response reaches after a")
        print("  17 s warm run, with no frame-drop rule and no self-clocked camera. Where it is not 0")
        print("  the model does not converge at all, and the figure's single number is whatever horizon")
        print("  it was evaluated over. The board is flat instead, because the camera self-clocks and")
        print("  the keep-last queue drops.")

    if a.submitted_spec:
        sp = SUBMITTED_SPEC
        pm = costs[("perception", 4)]["med"]
        rs = submitted_spec_response(ASSUMED["submitted"]["perception"])
        ra, rp = submitted_spec_response(ASSUMED["recost"]["perception"]), submitted_spec_response(pm)
        print("\nthe Tier A showdown's own metric: perception release-to-output, %g ms period, "
              "%d instances\n" % (sp["yolo_period_ms"], sp["yolo_instances"]))
        print("  %-30s %s" % ("instance", "  ".join("%7d" % k for k in range(len(ra)))))
        print("  %-30s %s" % ("submitted %6.3f ms/frame" % ASSUMED["submitted"]["perception"],
                              "  ".join("%7.2f" % v for v in rs)))
        print("  %-30s %s" % ("recost    %6.3f ms/frame" % ASSUMED["recost"]["perception"],
                              "  ".join("%7.2f" % v for v in ra)))
        print("  %-30s %s" % ("profiled %6.3f ms/frame" % pm, "  ".join("%7.2f" % v for v in rp)))
        print("\n  the sidecar reports instance %d: %.2f ms with the recost input (this recurrence "
              "says %.2f;\n  the schedule JSON was board-recost after placement, which moves it), "
              "%.2f ms with the\n  profiled input. The XPU-RT arm it is set against is %.2f ms and the "
              "budget is %.1f ms:\n  BOTH inputs miss the budget, so the figure's qualitative claim "
              "survives, but the ratio\n  against XPU-RT goes from %.2fx to %.2fx."
              % (sp["reported_instance"], sp["reported_response_ms"], ra[sp["reported_instance"]],
                 rp[sp["reported_instance"]], sp["reported_xpu_response_ms"], sp["budget_ms"],
                 sp["reported_response_ms"] / sp["reported_xpu_response_ms"],
                 rp[sp["reported_instance"]] / sp["reported_xpu_response_ms"]))

    if a.json:
        out = {"profiled_costs": {f"{n}@pool{p}": v for (n, p), v in costs.items()},
               "assumed_costs_ms": ASSUMED,
               "rows": table(tuple(a.arms.split(",")), costs=costs),
               "submitted_spec": dict(SUBMITTED_SPEC,
                                      response_submitted_ms=submitted_spec_response(ASSUMED["submitted"]["perception"]),
                                      response_assumed_ms=submitted_spec_response(ASSUMED["recost"]["perception"]),
                                      response_profiled_ms=submitted_spec_response(costs[("perception", 4)]["med"]))}
        json.dump(out, open(a.json, "w"), indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
