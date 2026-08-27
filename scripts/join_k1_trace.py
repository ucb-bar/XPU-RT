#!/usr/bin/env python3
"""Join a predicted XPU-RT schedule against a measured K1 trace.

Joins on the stable dispatch key (`<job><instance>_dispatch_<id>`), which is the
same string the scheduler emits and the runner echoes -- never on array
position, which silently reorders.

Reports three things that answer different questions:

  service time     run_us vs planned duration. Is the *profile* right?
  queueing delay   queue_delay_us. Time waiting, not computing.
  lateness         completion vs the periodic window.

The split matters because a deadline miss caused by 9 ms of compute and one
caused by 8 ms of waiting call for opposite fixes -- recompile the kernel versus
move it earlier. Reporting only end-to-end latency cannot tell them apart.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict


def load_trace(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["dispatch_key"]].append(r)
    return rows


def base_and_instance(job: str):
    i = len(job)
    while i > 0 and job[i - 1].isdigit():
        i -= 1
    return job[:i], int(job[i:] or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-json")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    disp = sched["dispatches"]
    periods = (sched.get("metadata") or {}).get("periodic_networks") or {}
    trace = load_trace(a.trace)

    joined, missing = [], []
    for key, v in disp.items():
        rs = trace.get(key)
        if not rs:
            missing.append(key)
            continue
        r = rs[0]
        pred_ms = float(v["duration"])
        run_ms = float(r["run_us"]) / 1000.0
        joined.append({
            "key": key, "job": v.get("job_name", ""), "id": v.get("id"),
            "target": r.get("target", ""),
            "pred_dur_ms": pred_ms, "actual_run_ms": run_ms,
            "abs_err_ms": run_ms - pred_ms,
            "rel_err": (run_ms - pred_ms) / pred_ms if pred_ms > 0 else float("nan"),
            "pred_start_ms": float(v["start_time"]),
            "actual_start_ms": float(r["start_us"]) / 1000.0,
            "queue_ms": float(r["queue_delay_us"]) / 1000.0,
            "end_ms": float(r["end_us"]) / 1000.0,
        })

    if not joined:
        print("no dispatches joined -- key mismatch between schedule and trace",
              file=sys.stderr)
        return 1

    rel = [j["rel_err"] for j in joined if j["pred_dur_ms"] > 0]
    absr = [abs(j["abs_err_ms"]) for j in joined]
    print(f"joined {len(joined)} dispatches ({len(missing)} in schedule but not trace)\n")
    print("=== service-time prediction error (actual run vs predicted duration) ===")
    print(f"  median relative error : {statistics.median(rel)*100:+.1f}%")
    print(f"  mean   relative error : {statistics.fmean(rel)*100:+.1f}%")
    print(f"  median |abs| error    : {statistics.median(absr)*1000:.1f} us")
    print(f"  max    |abs| error    : {max(absr)*1000:.1f} us")

    # weight by predicted duration: the big dispatches are what the schedule turns on
    big = [j for j in joined if j["pred_dur_ms"] >= 1.0]
    if big:
        rb = [j["rel_err"] for j in big]
        print(f"\n  restricted to dispatches >=1ms (n={len(big)}):")
        print(f"    median relative error : {statistics.median(rb)*100:+.1f}%")

    print("\n=== where the time actually goes ===")
    tot_run = sum(j["actual_run_ms"] for j in joined)
    tot_q = sum(j["queue_ms"] for j in joined)
    print(f"  total service (run)   : {tot_run:9.2f} ms")
    print(f"  total queueing        : {tot_q:9.2f} ms")
    if tot_run + tot_q > 0:
        print(f"  queueing share        : {100*tot_q/(tot_run+tot_q):9.1f}%")

    print(f"\n=== worst {a.top} service-time mispredictions (by |abs|) ===")
    print(f"{'dispatch':<26}{'pred ms':>9}{'actual ms':>10}{'err':>9}{'rel':>9}")
    for j in sorted(joined, key=lambda x: -abs(x["abs_err_ms"]))[:a.top]:
        print(f"{j['key']:<26}{j['pred_dur_ms']:9.3f}{j['actual_run_ms']:10.3f}"
              f"{j['abs_err_ms']:+9.3f}{j['rel_err']*100:+8.1f}%")

    # per-instance lateness against the periodic window
    inst = defaultdict(lambda: [1e18, -1e18])
    for j in joined:
        b = inst[j["job"]]
        b[0] = min(b[0], j["actual_start_ms"])
        b[1] = max(b[1], j["end_ms"])
    late = []
    for job, (st, en) in inst.items():
        b, i = base_and_instance(job)
        T = periods.get(b)
        if T is None:
            continue
        rel_t = i * T
        late.append((job, en - (rel_t + T)))
    if late:
        misses = [l for _, l in late if l > 0]
        print(f"\n=== measured deadline outcome ({len(late)} periodic instances) ===")
        print(f"  instances missing their window : {len(misses)} / {len(late)}")
        if misses:
            print(f"  worst lateness                 : {max(misses):.2f} ms")
            print(f"  median lateness (missers)      : {statistics.median(misses):.2f} ms")

    if a.out_json:
        json.dump({"joined": joined, "missing": missing}, open(a.out_json, "w"),
                  indent=1)
        print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
