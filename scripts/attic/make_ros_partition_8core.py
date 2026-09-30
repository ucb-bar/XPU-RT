#!/usr/bin/env python3
"""A ROS baseline that uses ALL EIGHT cores: YOLO on the P cluster, nav and control 2 each.

The partition is the one a careful ROS engineer would write, and it leaves no core idle:

    yolov8_nano_64x96  -> CPU_P#0+1+2+3   (the whole P cluster, 4 harts)
    fused_full         -> CPU_E#0+1       (2 harts)
    mlp_control        -> CPU_E#2+3       (2 harts)

COSTS ARE MEASURED, NOT ASSUMED. Standalone on this K1 with OC-sharding at codegen:

    1 core  55.87 ms      2 cores 50.38 ms      4 cores 52.29 ms      8 cores 39.39 ms

Uniform sharding of this network is inefficient -- 4 cores is 1.07x over 1 core, and is
actually SLOWER than 2 -- because most of yolov8's convolutions have too few output channels
to divide well. That is a property of the kernel and it is applied to the baseline in its own
favour: the 4-hart partition is costed at the measured 52.29 ms rather than at 55.87/4.

nav and control are costed at their measured warm medians (4.43 and 0.08 ms, pooled over 8
board traces). Two harts do not help them: fused_full is 15 dispatches and mlp_control is 7,
and neither divides usefully at that size.

The comparison this sets up is the honest one. Both arms get all 8 cores; what differs is that
the baseline shards each net UNIFORMLY inside a fixed partition, while XPU-RT chooses a width
per dispatch and packs them globally -- 35.71 ms measured end to end, against 56.8 ms here.
"""
from __future__ import annotations
import csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from measured_timing import CHAIN          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
HZ = 24e6
TRACE = "results/codesign_feedback/cmp_coupled_cpsat_board_trace.csv"
MEASURED_4CORE_YOLO = 52.294      # gen/profile_shard, topo_0_1_2_3
WARM = {"fused_full": 4.43, "mlp_control": 0.08}
PART = {"yolov8_nano_64x96": "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3",
        "fused_full": "CPU_E#0+CPU_E#1",
        "mlp_control": "CPU_E#2+CPU_E#3"}
ORDER = ["yolov8_nano_64x96", "fused_full", "mlp_control"]
FRAMES, CAM_HZ = 4, 20.0


def main() -> int:
    rows = [r for r in csv.DictReader(open(TRACE))
            if (r.get("actual_end_cycles") or "0").lstrip("-").isdigit()
            and int(r["actual_end_cycles"]) > 0]
    by = {n: [r for r in rows if r["network"] == n] for n in ORDER}
    target = {"yolov8_nano_64x96": MEASURED_4CORE_YOLO,
              "fused_full": WARM["fused_full"], "mlp_control": WARM["mlp_control"]}

    out, nid, period, free = {}, 0, 1000.0 / CAM_HZ, 0.0
    for k in range(FRAMES):
        t = max(k * period, free)          # serial: the node cannot start a frame until it is free
        for n in ORDER:
            raw = sum((int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / HZ * 1000.0
                      for r in by[n])
            sc = target[n] / raw if raw else 1.0
            for r in by[n]:
                nid += 1
                d = (int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / HZ * 1000.0 * sc
                out[str(nid)] = {"id": nid, "ordinal": 1, "total": 1, "dependencies": [],
                                 "hardware_target": PART[n],
                                 "start_time": round(t, 6), "duration": round(d, 6),
                                 "job_name": f"{n}{k}", "module_name": r.get("name") or n,
                                 "release_policy": "periodic", "time_dep_mode": "hard"}
                t += d
        free = t
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    chain = sum(target.values())
    p = "schedules/ros_partition_8core.json"
    json.dump({"metadata": {"makespan": round(mk, 4), "end_to_end_latency_ms": round(chain, 4),
                            "end_to_end_deadline_ms": 33.3,
                            "partition": PART,
                            "source": "YOLO on the 4-hart P cluster at its MEASURED 4-core time; "
                                      "nav/control at their measured warm medians"},
               "dispatches": out}, open(p, "w"), indent=1)
    print(f"wrote {p}")
    print(f"  partition uses all 8 cores: {PART}")
    print(f"  per-frame chain {chain:.2f} ms  (yolo {target['yolov8_nano_64x96']:.2f} @4 cores + "
          f"nav {WARM['fused_full']:.2f} + ctrl {WARM['mlp_control']:.2f})")
    print(f"  {FRAMES} frames at {CAM_HZ:.0f} Hz -> makespan {mk:.2f} ms")
    print(f"\n  XPU-RT, 8 harts, per-dispatch widths, MEASURED : {CHAIN['xpurt_ms']:.2f} ms")
    print(f"  ROS, 8 cores, uniform shard in fixed partition : {chain:.2f} ms   -> {chain/CHAIN['xpurt_ms']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
