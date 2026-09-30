#!/usr/bin/env python3
"""Split measured deadline misses into what a scheduler could fix and what it cannot.

WHY. A miss count says how many deadlines were missed, never
whether a better schedule existed. Two very different things produce one: a dispatch that
WAITED (queueing -- a placement decision, and a scheduler's job), and a network whose own
EXECUTION already exceeds its window (no placement helps; the window or the kernel has to
change). A combined count makes a workload look like one "where no lever helps" (w4 and w5
read that way) when part of its misses are placement decisions.

The attribution is by execution, not by network shape. `fused_full` in w5 is a
single-width network, but that is not the cause of w5's residual; summing the trace's own
cycles per instance shows where it comes from:

    net            window   inst0    median(rest)  cold/warm  over-window
    fused_full      5.0    12.055       4.466        2.70x        1/12
    ffn_block      10.0    11.416      10.336        1.10x        3/5
    dronet          7.0     8.307       6.104        1.36x        1/5
    mlp_control     5.0     0.090       0.080        1.13x        0/12

`ffn_block`'s WARM execution is 10.34 ms against its 10.0 ms window -- the network the
whole rung is built around does not fit, on the real board, at the width the loop chose.
The profile the ladder was sized from says 7.72 ms at 8 cores, so the board is 34% slower
than the number that declared the rung feasible, and `make_scaling_workloads.check()`
called it "in the band" on profile times alone. Three of five ffn instances are over
window by execution alone; no scheduler can recover them.

The other two execution-bound misses are COLD START: the first instance of a network runs
2.70x its warm median for `fused_full` and 1.36x for `dronet`. The AOT profile is a warm
steady-state measurement and models none of it.

So w5's 16 measured misses are 5 execution-bound (3 ffn window + 2 cold start) and 11
queueing. The headline count is unchanged; the attribution is what differs, and only the
queueing half is the loop's to fix.

Usage:
  scripts/attribute_board_misses.py --trace <run>_trace.csv --spec <workload>.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys

HZ = 24_000_000.0  # rdtime, see xpu-rt/k1_trace.K1_RDTIME_HZ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--spec", required=True, help="for each network's window")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    win = {k: float(v.get("window_duration"))
           for k, v in (spec.get("networks") or {}).items()
           if v.get("window_duration")}

    # Execution excludes queueing on purpose: summing a dispatch's own cycles answers
    # "could this instance ever fit?", which is the question a scheduler cannot change.
    exec_ms = collections.defaultdict(float)
    for r in csv.DictReader(open(a.trace)):
        try:
            s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            exec_ms[(r["network"], int(r["instance"]))] += (e - s) / HZ * 1e3

    rows, tot_exec_bound, tot_cold = [], 0, 0
    for net in sorted(win):
        xs = [exec_ms[(net, i)] for i in sorted(i for n, i in exec_ms if n == net)]
        if not xs:
            continue
        w = win[net]
        warm = statistics.median(xs[1:] or xs)
        over = [i for i, v in enumerate(xs) if v > w]
        # A first instance over window whose WARM median fits is cold start, not a
        # window that cannot be met; the distinction changes who owns the fix.
        cold = [i for i in over if i == 0 and warm <= w]
        rows.append({"network": net, "window_ms": w, "n_instances": len(xs),
                     "inst0_exec_ms": round(xs[0], 3),
                     "warm_median_exec_ms": round(warm, 3),
                     "cold_over_warm": round(xs[0] / warm, 2) if warm else None,
                     "instances_over_window_by_execution": len(over),
                     "of_which_cold_start": len(cold),
                     "warm_execution_fits_window": warm <= w})
        tot_exec_bound += len(over)
        tot_cold += len(cold)

    print(f"{'network':22s} {'window':>7s} {'inst0':>8s} {'warm':>8s} {'cold/warm':>9s} "
          f"{'over':>6s}  verdict")
    for r in rows:
        v = ("WARM EXEC OVER WINDOW -- unschedulable at this width"
             if not r["warm_execution_fits_window"]
             else (f"{r['of_which_cold_start']} cold-start only"
                   if r["instances_over_window_by_execution"] else "fits"))
        print(f"{r['network']:22s} {r['window_ms']:7.1f} {r['inst0_exec_ms']:8.3f} "
              f"{r['warm_median_exec_ms']:8.3f} {str(r['cold_over_warm'])+'x':>9s} "
              f"{r['instances_over_window_by_execution']:6d}  {v}")
    print(f"\nexecution-bound instances (no schedule can fix): {tot_exec_bound}"
          f"  of which cold start: {tot_cold}")
    print("everything else that misses is queueing, which IS the scheduler's to fix")

    if a.json_out:
        json.dump({"trace": a.trace, "spec": a.spec, "networks": rows,
                   "execution_bound_instances": tot_exec_bound,
                   "cold_start_instances": tot_cold,
                   "means": ("execution = sum of the instance's own dispatch cycles, "
                             "queueing excluded; a scheduler cannot shrink it")},
                  open(a.json_out, "w"), indent=1)
        print("wrote", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
