#!/usr/bin/env python3
"""Why giving the ROS 2 perception node all eight harts buys nothing: the board's own evidence.

`docs/Baselines/ros_arm_ranking.md` §3.2 measures `vanilla4` -> `vanilla8` (the same perception node, its YOLO
worker pool widened from 4 to 8 harts) at 45 Hz: no change in latency or cadence. This prints the
three board observations that explain it, from the traced runs themselves:

  * how many distinct harts each pooled YOLO dispatch actually ran on (the traced node records every
    shard's hart, `ros_mb_chain_traced.cpp` emit_detail / pool_shard rows);
  * the perception callback's time per frame;
  * the per-core busy % from the independent sampler (`cpu.csv`), after the 3 s warm-up the run's
    own summary uses.

The deployed YOLO kernels exist in a 1-wide and a 4-wide build (`docs/K1/nav_sharding.md`), so a
pooled op has four shards to hand out whatever the pool's size; workers beyond the fourth find none.

    scripts/ros_pool_width_evidence.py [--tags 45_vanilla4_r1,45_vanilla8_r1,...]
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACED = os.path.join(REPO, "results", "codesign_feedback", "ros_traced")
HZ = 24e6           # rdtime on the K1
WARMUP_MS = 3000.0  # the same warm-up pull_ros_traced.py drops


def evidence(tag):
    per = collections.defaultdict(set)
    cb = []
    with open(os.path.join(TRACED, tag, "trace.csv")) as f:
        for r in csv.DictReader(f):
            if not r["network"].startswith("yolo"):
                continue
            if r["op"] == "node_callback":
                cb.append((int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / HZ * 1e3)
            else:
                per[(r["instance"], r["dispatch_id"])].add(r["worker_hart"])
    spread = collections.Counter(len(h) for h in per.values())
    busy = collections.defaultdict(list)
    with open(os.path.join(TRACED, tag, "cpu.csv")) as f:
        for r in csv.DictReader(f):
            if float(r["t_ms"]) > WARMUP_MS:
                busy[int(r["cpu"])].append(float(r["busy_pct"]))
    busy = {c: round(statistics.mean(v)) for c, v in sorted(busy.items())}
    return {"tag": tag, "harts_per_dispatch": dict(sorted(spread.items())),
            "callback_median_ms": round(statistics.median(cb), 2), "frames": len(cb),
            "busy_pct_per_core": busy, "cores_ge_20pct": sum(v >= 20 for v in busy.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="45_vanilla4_r1,45_vanilla4_r2,45_vanilla8_r1,45_vanilla8_r2,45_vanilla8_r3")
    a = ap.parse_args()
    for tag in a.tags.split(","):
        e = evidence(tag)
        print(f"{tag:16s} YOLO dispatches by harts used {e['harts_per_dispatch']}  "
              f"callback {e['callback_median_ms']} ms/frame over {e['frames']}  "
              f"busy {e['busy_pct_per_core']}  ({e['cores_ge_20pct']} cores >= 20 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
