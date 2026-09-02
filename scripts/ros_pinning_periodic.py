#!/usr/bin/env python3
"""ROS-node (per-net pinning) schedule with PERIODIC releases, for the annotated Gantt.

Each network is one ROS node pinned to ONE cluster-0 hart; a periodic timer releases each
instance at k*period; the node runs its whole dispatch graph SEQUENTIALLY on that one hart (no
per-op cross-hart sharding). YOLO cannot be sharded, so it runs serial on a single hart and
overruns its release period -> the machine's other harts sit idle. Reuses the REAL measured
per-dispatch durations from an XPU-RT schedule of the same workload.
"""
import argparse, json, os, re, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_HART = {"mlp_control": "CPU_P#0", "fused_full": "CPU_P#1", "yolov8_nano_64x96": "CPU_P#2"}
NET_PERIOD = {"mlp_control": 10.0, "fused_full": 20.0, "yolov8_nano_64x96": 22.0}
def net_of(j):
    for n in NET_HART:
        if (j or "").startswith(n): return n
    return re.sub(r"\d+$", "", j or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = json.load(open(os.path.join(REPO, a.src)))
    items = list(src["dispatches"].values())
    # group dispatches by instance (job_name), preserve order
    by_job = collections.OrderedDict()
    for x in sorted(items, key=lambda z: (z.get("job_name", ""), z.get("ordinal", z.get("id", 0)))):
        by_job.setdefault(x["job_name"], []).append(x)
    hart_free = collections.defaultdict(float)
    out = {}
    nid = 0
    for job, ds in by_job.items():
        net = net_of(job); hart = NET_HART[net]; per = NET_PERIOD[net]
        suf = job[len(net):]; k = int(suf) if suf.isdigit() else 0
        t = max(k * per, hart_free[hart])
        for x in ds:
            nid += 1
            dur = x["duration"]
            out[str(nid)] = {"id": nid, "ordinal": x.get("ordinal", 1), "total": x.get("total", 1),
                             "dependencies": [], "hardware_target": hart, "start_time": round(t, 6),
                             "duration": dur, "job_name": job, "module_name": x.get("module_name", job),
                             "release_policy": "periodic", "time_dep_mode": "hard"}
            t += dur
        hart_free[hart] = t
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    res = {"metadata": {"makespan": mk, "policy": "ros_pinning_periodic"}, "dispatches": out}
    json.dump(res, open(os.path.join(REPO, a.out), "w"))
    print("wrote %s: %d dispatches, makespan %.2f ms" % (a.out, len(out), mk))

if __name__ == "__main__":
    main()
