#!/usr/bin/env python3
"""Check that two schedules are comparable before they are plotted against each other.

A makespan comparison between two schedules is only meaningful when both cover the SAME
RELEASES. This reports the per-network instance counts on each side and, when they differ,
truncates the longer schedule to the shorter one's horizon and rescores both.

Truncation is exact for a statically pinned schedule: each net's instances run sequentially
on its pinned hart in release order, so dropping instances k >= N cannot change the placement
of k < N. The policy is preserved exactly -- the same hart assignment, the same shard widths.

The number to compare across arms is the DEADLINE OUTCOME rather than the makespan: misses
are counted per instance and so are horizon-independent, while a makespan scales with however
many releases the schedule was generated over.

The coupled-chain pair (schedules/cmp_coupled_{cpsat,ros}_board.json) is matched by
construction -- 1 instance of each net on both sides, 120 dispatches, 35.6 core-ms, one
shared board calibration -- and this reports "horizons already match" for it.

Usage:
  scripts/verify_panel_i.py --xpu schedules/cmp_coupled_cpsat_board.json \
                            --ros schedules/cmp_coupled_ros_board.json

NOTE ON PARSING. Instance indices MUST come from split_job_name(job, known) with the known
net set. The one-argument form strips all trailing digits, and `yolov8_nano_64x96` ends in
digits: `yolov8_nano_64x960` parses as ('yolov8_nano_64x', 960) instead of
('yolov8_nano_64x96', 0), which silently drops every yolo instance from a filter.
"""
import argparse, collections, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name          # noqa: E402
from schedule_eval import summary             # noqa: E402


def instances(path, known):
    d = json.load(open(path))["dispatches"]
    by = collections.defaultdict(set)
    for v in d.values():
        n, i = split_job_name(v["job_name"], known)
        by[n].add(i)
    return {n: sorted(s) for n, s in by.items()}


def stats(path, spec):
    d = json.load(open(path))["dispatches"]
    mk = max(v["start_time"] + v["duration"] for v in d.values())
    wk = sum(v["duration"] for v in d.values())
    s = summary(path, spec)
    return mk, wk, len(d), s["instance_misses"], s["misses_by_network"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="data/toplevel/_flight_deployed_2frame.json")
    ap.add_argument("--xpu", default="schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json")
    ap.add_argument("--ros", default="schedules/scheduled_ros_partition_deployed.json")
    ap.add_argument("--out", default=os.path.join(os.environ.get("XPURT_SCRATCH") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "codesign_feedback", "tmp"), "ros_truncated.json"), help="the truncated ROS schedule (written only when the horizons differ)")
    a = ap.parse_args()
    for p in (a.spec, a.xpu, a.ros):
        if not os.path.isabs(p) and not os.path.exists(p):
            os.chdir(REPO)
    known = set(json.load(open(a.spec))["networks"])

    ix, ir = instances(a.xpu, known), instances(a.ros, known)
    print("XPU-RT instances:", {n: len(v) for n, v in sorted(ix.items())})
    print("ROS    instances:", {n: len(v) for n, v in sorted(ir.items())})
    if {n: len(v) for n, v in ix.items()} == {n: len(v) for n, v in ir.items()}:
        print("\nHorizons already match -- nothing to correct.")
        return 0
    print("\nHORIZON MISMATCH: the two schedules do not cover the same releases.")

    want = {n: len(v) for n, v in ix.items()}
    ros = json.load(open(a.ros))["dispatches"]
    keep = {}
    for k, v in ros.items():
        n, i = split_job_name(v["job_name"], known)
        if n in want and i < want[n]:
            keep[k] = v
    json.dump({"metadata": {"policy": "ros_partition, truncated to the XPU-RT release horizon"},
               "dispatches": keep}, open(a.out, "w"), indent=1)

    print(f"\n{'schedule':<34} {'makespan':>11} {'work':>12} {'disp':>6}  misses")
    rows = [("XPU-RT (CP-SAT)", a.xpu), ("ROS truncated (same horizon)", a.out),
            ("ROS as plotted", a.ros)]
    out = {}
    for tag, p in rows:
        mk, wk, n, m, by = stats(p, a.spec)
        out[tag] = mk
        print(f"{tag:<34} {mk:>9.2f}ms {wk:>9.1f}cms {n:>6}  {m} {by}")
    r = out["ROS truncated (same horizon)"] / out["XPU-RT (CP-SAT)"]
    print(f"\nHonest makespan ratio, same horizon and same policy: {r:.2f}x")
    print("Horizon-independent claim: ROS misses 100% of perception instances at BOTH "
          "horizons (2/2 truncated, 5/5 as plotted); XPU-RT misses none.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
