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
import re
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


_OP_TAIL = re.compile(r"\$async_dispatch_\d+_([a-zA-Z_][a-zA-Z0-9_]*?)(?:_\d+(?:x\d+)*)?(?:_[a-z0-9]+(?:x[a-z0-9]+)*)?$")


def _op_of(module_name: str) -> str:
    """'conv' from '...$async_dispatch_1_conv_32x56x56x3x3x3_i8xi8xi32'."""
    m = _OP_TAIL.search(module_name or "")
    return m.group(1) if m else ""


def _impl_from_vmfb(path: str) -> str:
    """'RVV' from '.../gen/vmfb/dronet/spacemit_x60/RVV/dronet.q.int8/...'.

    Positional rather than pattern-matched on names, so a new implementation
    label (RVV_split, IME_ukernel, ...) needs no change here.
    """
    parts = (path or "").split("/")
    try:
        i = parts.index("vmfb")
    except ValueError:
        return ""
    # vmfb / <model> / <target> / <impl>
    return parts[i + 3] if len(parts) > i + 3 else ""


def _err_table(title: str, groups: dict) -> None:
    """median/mean relative error and count, per group, worst median first."""
    if not groups:
        return
    print(f"\n=== {title} ===")
    print(f"{'group':<16}{'n':>5}{'median rel':>12}{'mean rel':>11}"
          f"{'median |abs|':>14}")
    rows = []
    for g, js in groups.items():
        rel = [j["rel_err"] for j in js if j["pred_dur_ms"] > 0]
        if not rel:
            continue
        absr = [abs(j["abs_err_ms"]) for j in js]
        rows.append((statistics.median(rel), g, len(js),
                     statistics.fmean(rel), statistics.median(absr)))
    for med, g, n, mean, mabs in sorted(rows, reverse=True):
        print(f"{g:<16}{n:>5}{med*100:>11.1f}%{mean*100:>10.1f}%"
              f"{mabs*1000:>13.1f}us")


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

    joined, missing, multi_iter_keys = [], [], []
    for key, v in disp.items():
        rs = trace.get(key)
        if not rs:
            missing.append(key)
            continue
        if len(rs) > 1:
            multi_iter_keys.append(key)
        r = rs[0]
        pred_ms = float(v["duration"])
        run_ms = float(r["run_us"]) / 1000.0
        joined.append({
            "key": key, "job": v.get("job_name", ""), "id": v.get("id"),
            "target": r.get("target", ""),
            # `target` is only CPU_P/CPU_E, so it cannot answer "which
            # implementation was this?". The trace's vmfb_path can: its layout
            # is gen/vmfb/<model>/<target>/<impl>/..., so the implementation is
            # already in every trace ever taken, just never read.
            "impl": _impl_from_vmfb(r.get("vmfb_path", ""))
                    or (v.get("implementation") or ""),
            "op": _op_of(r.get("module_name", "")),
            "cores": r.get("cores", ""),
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

    # The aggregate above hides a large spread. Grouping is what turns "+14%
    # median error" into an actionable statement about which op types the cost
    # model is wrong about.
    from collections import defaultdict
    by_op, by_impl = defaultdict(list), defaultdict(list)
    for j in joined:
        by_op[j["op"] or "(unparsed)"].append(j)
        by_impl[j["impl"] or "(unknown)"].append(j)
    _err_table("service-time prediction error by op type", by_op)
    _err_table("service-time prediction error by implementation", by_impl)

    if multi_iter_keys:
        print(f"\nNOTE: {len(multi_iter_keys)} dispatch key(s) had more than "
              f"one trace row (multiple graph iterations); only iteration 0 is "
              f"joined. Re-run with graph_iters=1 for a clean comparison, or "
              f"extend this script to join per (key, graph_iter).")

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
