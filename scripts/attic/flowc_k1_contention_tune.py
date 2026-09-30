#!/usr/bin/env python3
"""Tune a contention correction against a CONCURRENT multi-model workload.

The QRB5165 traces cannot answer this: all 440 of their dispatches ran solo,
because the smolVLA workloads are serial chains. The K1 exact-cycle runs can --
they are four networks (mlp_control, dronet, fused_full, ffn_block) across
eight harts, 178 dispatches per run, 20 runs, with per-dispatch
`actual_start_cycles`/`actual_end_cycles` at a 24 MHz timer. That is a real
concurrent multi-model measurement, already on disk.

The question is the one `contention_model.py` left open. Its own docstring
reports that a PAIRED co-runner measurement on K1 came back below the
measurement's resolution -- 0.999, 1.012, 1.010, 1.051 same-cluster against
1.061, 0.995, 1.002, 1.004 cross-cluster, two distributions straddling 1.0 --
and concludes no model should be installed. That was a microbenchmark. This
asks the same question of whole scheduled runs: does a dispatch that overlaps
other dispatches take measurably longer than one that does not?

Method. For each dispatch compute actual duration from the cycle counters, the
number of OTHER dispatches in flight during it, and the ratio to the
schedule's own prediction. Then fit a correction keyed on co-runner count and
score it out-of-sample, holding out whole runs -- never dispatches -- so a
run's own rows cannot leak into the model that scores it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

TICKS_PER_MS = 24_000.0
MIN_PRED_MS = 0.05


def read_run(path: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            pred = float(r["predicted_duration_ms"])
            s = int(r["actual_start_cycles"]) / TICKS_PER_MS
            e = int(r["actual_end_cycles"]) / TICKS_PER_MS
        except (KeyError, ValueError):
            continue
        if e <= s or pred <= 0:
            continue
        rows.append({"run": os.path.basename(path).replace("_trace.csv", ""),
                     "network": r.get("network", ""), "hart": r.get("hart", ""),
                     "kind": r.get("core_kind", ""),
                     "start": s, "end": e, "act": e - s, "pred": pred,
                     "ratio": (e - s) / pred, "usable": pred >= MIN_PRED_MS})
    # co-runners: other dispatches in flight during this one, on ANY hart
    for i, a in enumerate(rows):
        rows[i]["co"] = sum(1 for j, b in enumerate(rows)
                            if j != i and b["start"] < a["end"] and b["end"] > a["start"])
    for r in rows:
        r["bucket"] = "solo" if r["co"] == 0 else ("1-2" if r["co"] <= 2 else "3+")
    return rows


def fit(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r["usable"]:
            by[key(r)].append(r["ratio"])
    return {k: {"n": len(v), "factor": round(st.median(v), 4)}
            for k, v in by.items() if v}


def score(rows, model, key):
    le = []
    for r in rows:
        if not r["usable"]:
            continue
        f = (model.get(key(r)) or {}).get("factor", 1.0)
        le.append(abs(math.log(r["act"] / (r["pred"] * f))))
    return round(st.median(le), 4) if le else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces",
                    default="results/k1_feedback_exact/board_runs_rt_observed/*_trace.csv")
    ap.add_argument("--out", default="results/flowc_contention/k1_tune.json")
    a = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(a.traces)):
        r = read_run(p)
        if r:
            runs[r[0]["run"]] = r
    allrows = [r for v in runs.values() for r in v]
    if not allrows:
        print("  no usable rows"); return 1

    dist = defaultdict(int)
    for r in allrows:
        dist[r["bucket"]] += 1
    print(f"  {len(runs)} runs, {len(allrows)} dispatches")
    print(f"  co-runner distribution: " +
          ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    conc = sum(1 for r in allrows if r["co"] > 0) / len(allrows)
    print(f"  {conc*100:.1f}% of dispatches overlapped at least one other "
          f"— THIS is the concurrency the QRB5165 traces lacked\n")

    # is there a contention signal at all?
    print(f"  {'bucket':8s} {'n':>5s} {'median actual/predicted':>24s}")
    for b in ("solo", "1-2", "3+"):
        v = [r["ratio"] for r in allrows if r["bucket"] == b and r["usable"]]
        if v:
            print(f"  {b:8s} {len(v):5d} {st.median(v):24.4f}")

    keys = {
        "none":        lambda r: "all",
        "kind":        lambda r: r["kind"],
        "kind+co":     lambda r: f"{r['kind']}|{r['bucket']}",
    }
    print(f"\n  leave-one-RUN-out (whole runs held out, never dispatches)\n")
    print(f"  {'model':12s} {'held-out logerr':>16s}   vs uncorrected")
    base = None
    results = {}
    for name, key in keys.items():
        errs = []
        for held, rows in runs.items():
            train = [r for k, v in runs.items() if k != held for r in v]
            m = fit(train, key) if name != "none" else {}
            e = score(rows, m, key)
            if e is not None:
                errs.append(e)
        med = round(st.median(errs), 4)
        if name == "none":
            base = med
        rel = "" if name == "none" else f"   {(1-med/base)*100:+.1f}%"
        print(f"  {name:12s} {med:16.4f}{rel}")
        results[name] = med

    verdict = ("co-runner conditioning HELPS"
               if results["kind+co"] < results["kind"] - 1e-4
               else "co-runner conditioning does NOT help beyond per-kind")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"runs": len(runs), "dispatches": len(allrows),
               "concurrent_fraction": round(conc, 4),
               "bucket_counts": dict(dist),
               "bucket_medians": {b: (st.median([r["ratio"] for r in allrows
                                     if r["bucket"] == b and r["usable"]])
                                  if any(r["bucket"] == b for r in allrows) else None)
                                  for b in ("solo", "1-2", "3+")},
               "leave_one_run_out_logerr": results, "verdict": verdict},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                a.out), "w"), indent=1)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
