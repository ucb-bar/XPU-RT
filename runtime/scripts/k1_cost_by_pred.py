#!/usr/bin/env python3
"""Measure what it costs a dispatch to read what the PREVIOUS one wrote, from
another hart.

WHY THIS AND NOT A CO-RUNNER SWEEP
----------------------------------
`k1_contention_mb.py` asked whether two dispatches running AT THE SAME TIME on
different harts slow each other down. On this board, at up to four co-runners,
the answer measured as a null -- the same-cluster and cross-cluster
distributions overlap completely (docs/k1_contention.md).

That is not the only cross-cluster cost, and it is not the one the rest of the
data points at. The multi-core sweep found DroNet SLOWER on eight harts than on
four (5.32 vs 5.25 ms) while yolov8_nano was not, which is a working-set
effect: harts 4-7 are a second L2 domain, so a dispatch whose input was
produced on the other cluster has to fetch it across. That is a property of the
PRODUCER-CONSUMER EDGE, not of concurrency, and it is what `cost_by_pred`
models -- `workload_factory` already reads a per-dispatch
`{"CPU_P->CPU_E": ms, ...}` map and the MILP already consumes it. Nothing has
ever measured it.

THE EXPERIMENT
--------------
DroNet is a chain: every dispatch consumes the previous one's output. Run it
three times, SERIALLY in all three (each dispatch starts when its predecessor
finishes, so nothing is concurrent and contention cannot be the explanation),
changing only which hart each dispatch lands on:

    same_hart       every dispatch on CPU_P#0        producer's cache is ours
    same_cluster    alternating CPU_P#0 / CPU_P#1    different L1, shared L2
    other_cluster   alternating CPU_P#0 / CPU_E#0    different L1 AND L2

`same_cluster - same_hart` is the cost of crossing an L1. `other_cluster -
same_hart` is the cost of crossing the L2 domain. The walker runs one worker
per (core_kind, hart) inside ONE process, so the buffers are genuinely shared
memory and the consumer genuinely has to fetch what the producer wrote.

PAIRING, because the last measurement taught this the hard way. Two solo runs
of DroNet twenty minutes apart differed by 2.6% with nothing else on the board,
and the effects here are expected to be of that order. The three variants are
run INTERLEAVED (A B C A B C ...) rather than in blocks, so drift is spread
across arms instead of being confounded with them.

    k1_cost_by_pred.py --schedule schedules/scheduled_dronet_solo_greedy_profiled.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The three placements, as (name, hart lane cycle). A dispatch at chain index
#: i goes to lanes[i % len(lanes)].
PLACEMENTS = {
    "same_hart": ["CPU_P#0"],
    "same_cluster": ["CPU_P#0", "CPU_P#1"],
    "other_cluster": ["CPU_P#0", "CPU_E#0"],
}

#: Every machine name the emitted `cost_by_pred` map covers. The K1 solve
#: models the two clusters as two pools of four, so these are the eight names
#: `workload_factory` will look up.
HARTS = [f"CPU_P#{i}" for i in range(4)] + [f"CPU_E#{i}" for i in range(4)]


def chain_order(dispatches: Dict[str, dict]) -> List[str]:
    """Dispatch keys in dependency order.

    Sorting by `start_time` would work for the schedule this is derived from
    and would silently produce a WRONG order for any schedule whose dispatches
    were not already serial. The dependency graph is the thing that actually
    constrains the order, so use it.
    """
    remaining = dict(dispatches)
    done: List[str] = []
    placed = set()
    while remaining:
        ready = [k for k, v in remaining.items()
                 if all(d in placed or d not in dispatches
                        for d in (v.get("dependencies") or []))]
        if not ready:
            raise SystemExit("dependency cycle in the schedule; cannot order "
                             "the chain")
        ready.sort(key=lambda k: (float(remaining[k].get("start_time", 0.0)), k))
        for k in ready:
            done.append(k)
            placed.add(k)
            del remaining[k]
    return done


def serialise(schedule: dict, lanes: List[str]) -> dict:
    """One dispatch at a time, cycling through `lanes`.

    SERIAL IS THE CONTROL. If two dispatches could overlap, a difference
    between placements would be explainable as contention, which is a
    different mechanism with its own (null) measurement. One at a time makes
    the producer-consumer edge the only thing that changed.
    """
    out = copy.deepcopy(schedule)
    disp = out["dispatches"]
    t = 0.0
    for i, key in enumerate(chain_order(disp)):
        d = disp[key]
        d["hardware_target"] = lanes[i % len(lanes)]
        d["start_time"] = t
        t += float(d.get("duration", 0.0))
    return out


def run_board(schedule_path: str, models: str, backends: str,
              timeout: float) -> str:
    cmd = ["bash", os.path.join(REPO, "ModelBlaster", "scripts",
                                "run_xpurt_k1.sh"),
           "--schedule", schedule_path, "--models", models,
           "--backends", backends]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=os.path.join(REPO, "ModelBlaster"))
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:])
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"board run failed for {schedule_path}")
    return p.stdout


#: rdtime on the K1, fixed by the hardware. The walker's trace records
#: `actual_*_cycles`, which on Linux are rdtime TICKS and not core cycles --
#: the field name is kept for protocol compatibility with the Zephyr harness.
#: Converting with the core clock would be wrong by more than an order of
#: magnitude; harness_linux/src/main.c says so at the point it prints them.
CLOCK_MHZ = 24.0


def durations_from_trace(path: str) -> Dict[str, float]:
    """`{network:instance:dispatch_id -> ms}` from the walker's trace CSV.

    Keyed on the triple rather than on `dispatch_id` alone: a schedule with
    more than one instance of a network repeats the ids, and silently keeping
    the last one would make a 21-dispatch comparison out of a 42-dispatch run.
    """
    import csv
    out: Dict[str, float] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                key = f"{r['network']}:{r['instance']}:{r['dispatch_id']}"
                ticks = float(r["actual_end_cycles"]) - float(
                    r["actual_start_cycles"])
            except (KeyError, ValueError):
                continue
            out[key] = ticks / (CLOCK_MHZ * 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True,
                    help="a solved schedule to use as the template; only "
                         "hardware_target and start_time are rewritten")
    ap.add_argument("--models", default="dronet")
    ap.add_argument("--backends", default="rvv_x60,rvv_x60",
                    help="one backend per CORE KIND, in order. The K1 solve "
                         "declares two kinds (rvv, rvv_c1) because the two "
                         "clusters are separate pools, and both are compiled "
                         "with the same rvv_x60 backend -- so this is a "
                         "two-element list of the same name, not one name.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="interleaved rounds; drift is spread across arms "
                         "rather than confounded with them")
    ap.add_argument("--trace", default=None,
                    help="where run_xpurt_k1.sh leaves its trace CSV")
    ap.add_argument("--out", default=os.path.join(
        REPO, "artifacts", "k1_run", "cost_by_pred.json"))
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    base = json.load(open(a.schedule))
    work = a.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "k1_cost_by_pred")
    os.makedirs(work, exist_ok=True)

    paths = {}
    for name, lanes in PLACEMENTS.items():
        p = os.path.join(work, f"scheduled_cbp_{name}.json")
        json.dump(serialise(base, lanes), open(p, "w"), indent=1)
        paths[name] = p
        print(f"{name:<15} {len(lanes)} lane(s): {', '.join(lanes)} -> {p}")

    samples: Dict[str, List[Dict[str, float]]] = {k: [] for k in PLACEMENTS}
    for r in range(a.repeats):
        for name in PLACEMENTS:                    # INTERLEAVED, not blocked
            run_board(paths[name], a.models, a.backends, a.timeout)
            # run_xpurt_k1.sh files the trace under the schedule's own stem,
            # so the three variants cannot overwrite each other's results --
            # which is why they are given distinct filenames above.
            stem = os.path.splitext(os.path.basename(paths[name]))[0]
            trace = a.trace or os.path.join(
                REPO, "ModelBlaster", "build", "k1_xpurt", "_gen", stem,
                f"{stem}_trace.csv")
            if not os.path.exists(trace):
                raise SystemExit(f"no trace at {trace}; pass --trace")
            d = durations_from_trace(trace)
            if not d:
                raise SystemExit(f"{name}: empty trace")
            samples[name].append(d)
            print(f"  round {r}: {name:<15} "
                  f"{sum(d.values()):.3f} ms over {len(d)} dispatches")

    totals = {k: [sum(s.values()) for s in v] for k, v in samples.items()}
    ref = statistics.median(totals["same_hart"])
    print()
    result = {}
    for name in PLACEMENTS:
        med = statistics.median(totals[name])
        spread = max(totals[name]) - min(totals[name])
        result[name] = {
            "total_ms_samples": [round(x, 4) for x in totals[name]],
            "median_total_ms": round(med, 4),
            "ratio_vs_same_hart": round(med / ref, 6) if ref else 1.0,
            "spread_ms": round(spread, 4),
            "spread_pct_of_median": round(100.0 * spread / med, 2) if med else 0.0,
        }
        print(f"{name:<15} median {med:8.3f} ms  "
              f"{med/ref:6.3f}x vs same_hart   "
              f"spread {100.0*spread/med:4.1f}%")

    # SEPARATION, not just an ordering of medians. Three arms whose medians
    # differ by less than their sample ranges are one arm reported three
    # times, and the contention sweep is the cautionary case: its medians were
    # also ordered, and its distributions overlapped completely.
    def rng(name):
        return min(totals[name]), max(totals[name])
    seps = {}
    for lo_name, hi_name in (("same_hart", "same_cluster"),
                             ("same_cluster", "other_cluster"),
                             ("same_hart", "other_cluster")):
        lo_hi = rng(lo_name)[1]
        hi_lo = rng(hi_name)[0]
        seps[f"{lo_name}<{hi_name}"] = {
            "disjoint": bool(lo_hi < hi_lo),
            "gap_ms": round(hi_lo - lo_hi, 4),
        }
        print(f"  {lo_name:<14} max {lo_hi:7.3f}  <  "
              f"{hi_name:<14} min {hi_lo:7.3f}   "
              f"{'DISJOINT' if lo_hi < hi_lo else 'OVERLAP'}")

    # The per-dispatch map, in the form `workload_factory` already reads:
    # {"<pred machine>-><this machine>": ms}. The measurement gives three
    # classes of edge, so every hart pair of the same class gets the same
    # cost -- this is a MODEL fitted to three measured classes, not 64
    # independent measurements, and the artifact says so in `derivation`.
    per_disp_med = {
        name: {k: statistics.median([s[k] for s in samples[name] if k in s])
               for k in samples[name][0]}
        for name in PLACEMENTS}
    cost_by_pred = {}
    for k, base_ms in per_disp_med["same_hart"].items():
        same_c = per_disp_med["same_cluster"].get(k, base_ms)
        other_c = per_disp_med["other_cluster"].get(k, base_ms)
        entry = {}
        for pred in HARTS:
            for cur in HARTS:
                if pred == cur:
                    v = base_ms
                elif pred.split("#")[0] == cur.split("#")[0]:
                    v = same_c
                else:
                    v = other_c
                entry[f"{pred}->{cur}"] = round(v, 6)
        cost_by_pred[k] = entry

    out = {"schema": "xpurt.cost_by_pred/v1",
           "template_schedule": os.path.abspath(a.schedule),
           "models": a.models, "backends": a.backends,
           "repeats": a.repeats,
           "placements": {k: v for k, v in PLACEMENTS.items()},
           "results": result,
           "separation": seps,
           "derivation": (
               "Three measured edge classes -- predecessor on the same hart, "
               "on another hart of the same cluster, and on the other cluster "
               "-- expanded to every (pred, cur) hart pair of the same class. "
               "It is a model fitted to three measurements, not 64 "
               "independent ones. The per-dispatch values are medians over "
               "the repeats of the arm that class was measured in."),
           "per_dispatch_ms": {
               name: {k: round(v, 6) for k, v in d.items()}
               for name, d in per_disp_med.items()},
           "cost_by_pred": cost_by_pred}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
