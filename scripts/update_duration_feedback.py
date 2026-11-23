#!/usr/bin/env python3
"""Fold one board run into the measured/predicted duration factors.

Inputs are what a ModelBlaster `run_xpurt_k3.sh` round leaves behind:

  --schedule  the scheduled_*.json that round ran
  --trace     <N>_trace.csv pulled back from the board
  --graph     <network>=<ModelBlaster build>/.../graph.json, once per network
              (maps the trace's op `name` to the IR dispatch id the schedule
              and the profiles are keyed by; the trace's own `dispatch_id` is
              the codegen table index, which skips zero-cost ops)
  --prev      the factor file this round was SOLVED WITH (omit for round 0,
              i.e. every factor 1.0)
  --out       new factor file

The trace's `predicted_duration_ms` is checked against the schedule entry it
joins to; a mismatch means the wrong schedule/trace pair and is fatal.

Example (K3, ns timestamps):

  python scripts/update_duration_feedback.py --round round0 \\
    --schedule schedules/scheduled_..._fbr0_greedy_profiled.json \\
    --trace ../ModelBlaster/build/k3_xpurt/_gen/<N>/<N>_trace.csv \\
    --graph mlp_control=../ModelBlaster/build/k3_xpurt/mlp_control/int8/graph.json \\
    --graph dronet=... --graph yolov8_nano=... \\
    --out factors_after_round0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xpu-rt"))

import duration_feedback as dfb  # noqa: E402


def name_to_dispatch(graph_path: str) -> dict[str, int]:
    g = json.load(open(graph_path))
    ops = g["ops"] if isinstance(g, dict) and "ops" in g else g
    out: dict[str, int] = {}
    for o in ops:
        if o.get("dispatch_id") is None:     # view/alias: never scheduled
            continue
        if o["name"] in out:
            raise SystemExit(f"{graph_path}: op name {o['name']!r} is not unique")
        out[o["name"]] = int(o["dispatch_id"])
    return out


def observations(schedule_path: str, trace_path: str, graphs: dict[str, str],
                 ticks_per_ms: float) -> tuple[list[dict], dict]:
    sched = json.load(open(schedule_path))["dispatches"]
    maps = {net: name_to_dispatch(p) for net, p in graphs.items()}
    obs: list[dict] = []
    stats = {"rows": 0, "zero_cost_skipped": 0, "used": 0}
    for r in csv.DictReader(open(trace_path)):
        stats["rows"] += 1
        net = r["network"]
        if net not in maps:
            raise SystemExit(f"trace network {net!r} has no --graph")
        if int(r["dispatch_id"]) < 0:          # zero-cost op: no kernel ran
            stats["zero_cost_skipped"] += 1
            continue
        did = maps[net].get(r["name"])
        if did is None:
            raise SystemExit(f"{net}: trace op {r['name']!r} not in its graph.json")
        key = f"{net}{r['instance']}_dispatch_{did}"
        ent = sched.get(key)
        if ent is None:
            raise SystemExit(f"trace row {key} not in schedule {schedule_path}")
        pred = float(r["predicted_duration_ms"])
        if abs(pred - float(ent["duration"])) > 1e-5 + 1e-4 * abs(pred):
            raise SystemExit(f"{key}: trace predicted {pred} ms but schedule says "
                             f"{ent['duration']} ms -- wrong schedule/trace pair?")
        meas = (int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / ticks_per_ms
        obs.append({"network": net, "dispatch": f"dispatch_{did}",
                    "kind": dfb.machine_kind(ent["hardware_target"]),
                    "measured_ms": meas, "predicted_ms": pred})
        stats["used"] += 1
    return obs, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--graph", action="append", required=True, metavar="NET=PATH")
    ap.add_argument("--prev", default=None, help="factor file the round was solved with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", required=True, help="label recorded in the file")
    ap.add_argument("--ticks-per-ms", type=float, default=1e6,
                    help="trace clock; default 1e6 (ns, hosted Linux K1/K3 walker)")
    a = ap.parse_args(argv)

    graphs = dict(g.split("=", 1) for g in a.graph)
    prev = dfb.load(a.prev)
    obs, st = observations(a.schedule, a.trace, graphs, a.ticks_per_ms)
    new = dfb.update(prev, obs, a.round)
    dfb.save(new, a.out)
    print(f"[duration-feedback] {a.round}: {st['used']} dispatch executions used "
          f"({st['zero_cost_skipped']} zero-cost skipped) -> {a.out}")
    print(f"[duration-feedback] factors: {dfb.summarize(new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
