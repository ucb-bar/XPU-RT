#!/usr/bin/env python3
"""Generalized ROS per-node-pinning baseline (periodic releases), for arbitrary specs.

Same model as ros_pinning_periodic.py, but reads the net->hart pinning and per-net periods
from the SPEC (and an optional --perception-hart list to CO-LOCATE the vision pipeline on a
single ROS perception executor). Each net is one ROS node; a periodic timer releases each
instance at k*period; the node runs its whole dispatch graph SEQUENTIALLY on its pinned
hart(s) (no per-op cross-hart sharding). Reuses the REAL measured per-dispatch durations
from an XPU-RT schedule of the same workload, so ROS pays the SAME per-op costs as XPU-RT --
only the serialization policy differs.
"""
import argparse, json, os, re, collections, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="XPU-RT schedule JSON (source of real per-dispatch durations)")
    ap.add_argument("--spec", required=True, help="networks spec JSON (periods + net list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perception-nets", default="yolov8_nano_64x96,ffn_block,attn_block",
                    help="comma-list of nets co-located on ONE perception executor hart")
    ap.add_argument("--control-hart-base", type=int, default=0)
    ap.add_argument("--perception-speedup", type=float, default=None,
                    help="whole-net speedup of the perception kernel at --perception-width. "
                         "Default is the value MEASURED standalone on this K1 for the width "
                         "given (scripts/ros_fair_speed_sweep provenance; 1c 55.87, 2c 50.38, "
                         "4c 52.29, 8c 39.39 ms). Durations are divided by it, so a wider "
                         "partition costs the baseline less time as it should")
    ap.add_argument("--perception-width", type=int, default=1,
                    help="harts in the perception node's partition. 1 = the original "
                         "single-hart executor; >1 lets the node use a multi-threaded kernel "
                         "inside its static partition, which is the fairer baseline")
    a = ap.parse_args()

    spec = json.load(open(a.spec))["networks"]
    known = set(spec)
    period = {n: float(v.get("period", 0) or 0) for n, v in spec.items()}
    perception = [n for n in a.perception_nets.split(",") if n in spec]

    # Assign one PARTITION per node. A partition may be more than one hart: a ROS node is a
    # process, and nothing stops the kernel inside it from being multi-threaded, so confining
    # the perception node to a single core would understate the baseline rather than model it.
    # What the ROS policy actually costs is that partitions are STATIC and a node's work is
    # serial ACROSS instances -- not that a node gets one core.
    #
    # --perception-width therefore gives the perception executor a core set, and the dispatch
    # is costed at that width from the same profile the other arm reads. Width 1 is the
    # single-hart model, kept so runs made with it still reproduce.
    net_hart = {}
    h = a.control_hart_base
    for n in spec:
        if n in perception:
            continue
        net_hart[n] = f"CPU_P#{h}"; h += 1
    pw = max(1, int(a.perception_width))
    # MEASURED standalone on this board with OC-sharding at codegen (gen/profile_shard).
    # Sharding this network is inefficient -- 1.42x at eight cores, and four cores is SLOWER
    # than two -- so a wider partition helps the baseline much less than core count suggests.
    # That is a property of the kernel, and it is applied to the baseline in its own favour.
    _MEASURED = {1: 1.00, 2: 55.870 / 50.384, 4: 55.870 / 52.294, 8: 55.870 / 39.386}
    sp = a.perception_speedup if a.perception_speedup else _MEASURED.get(pw, 1.0)
    perc_hart = "+".join(f"CPU_P#{h + i}" for i in range(pw))
    for n in perception:
        net_hart[n] = perc_hart

    src = json.load(open(a.src if os.path.isabs(a.src) else os.path.join(REPO, a.src)))
    items = list(src["dispatches"].values())
    # Sort by (net, numeric instance, dispatch ordinal) so instance 10 does NOT string-sort
    # before instance 2 (which would corrupt the serial per-hart release timing).
    def _key(z):
        net, inst = split_job_name(z.get("job_name", ""), known)
        return (net, int(inst), z.get("ordinal", z.get("id", 0)))
    by_job = collections.OrderedDict()
    for x in sorted(items, key=_key):
        by_job.setdefault(x["job_name"], []).append(x)

    hart_free = collections.defaultdict(float)
    out = {}
    nid = 0
    for job, ds in by_job.items():
        net, inst = split_job_name(job, known)
        hart = net_hart[net]; per = period[net]
        t = max(inst * per, hart_free[hart])
        for x in ds:
            nid += 1
            out[str(nid)] = {"id": nid, "ordinal": x.get("ordinal", 1), "total": x.get("total", 1),
                             "dependencies": [], "hardware_target": hart,
                             "start_time": round(t, 6),
                             "duration": (x["duration"] / sp) if net in perception else x["duration"],
                             "job_name": job, "module_name": x.get("module_name", job),
                             "release_policy": "periodic", "time_dep_mode": "hard"}
            t += (x["duration"] / sp) if net in perception else x["duration"]
        hart_free[hart] = t
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    res = {"metadata": {"makespan": mk, "policy": "ros_pinning_generic",
                        "perception_executor_hart": perc_hart,
                        "perception_width": pw, "perception_speedup_applied": round(sp, 4),
                        "perception_speedup_source": ("--perception-speedup" if a.perception_speedup
                                                      else "measured standalone on this K1"),
                        "perception_nets": perception, "net_hart": net_hart},
           "dispatches": out}
    json.dump(res, open(a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out), "w"), indent=1)
    print("wrote %s: %d dispatches, makespan %.2f ms | perception hart %s <- %s"
          % (a.out, len(out), mk, perc_hart, perception))


if __name__ == "__main__":
    main()
