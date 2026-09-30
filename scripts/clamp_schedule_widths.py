#!/usr/bin/env python3
"""Make a solved schedule buildable: every packed-weight (convolution) dispatch gets one shard
width across all its periodic instances, and that width divides the dispatch's output channels.

    scripts/clamp_schedule_widths.py <schedule.json> <network>:<ir graph.json> [...] --out <json>

Why a post-pass: the generated model materialises one packed weight layout per convolution, so
`schedule_shards` refuses a schedule whose instances disagree on a width, or whose width does not
divide OC. A solver that lands there has produced a placement the board cannot execute. Widths
are only ever lowered here -- to the largest divisor of OC not above the scheduled width, and to
the minimum across instances -- so nothing is given more parallelism than the solver asked for;
the dropped harts are simply released. Timing fields are left as the solver wrote them; the board
trace, not the schedule, is what gets measured.
"""
from __future__ import annotations
import glob
import argparse, collections, json, re, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
from job_names import split_job_name   # noqa: E402

PACKED = {"conv2d_s8", "conv2d_batchnorm2d_silu_s8", "conv2d_batchnorm2d_s8", "depthwise_conv2d_s8"}


def oc_of(op):
    for k in ("shape", "attrs", "output_shape"):
        v = op.get(k)
        if isinstance(v, dict) and v.get("OC"):
            return int(v["OC"])
        if isinstance(v, str) and "OC=" in v:
            return int(v.split("OC=")[1].split(";")[0])
    for sub in (op.get("params") or {}, op.get("dispatch") or {}):
        if isinstance(sub, dict) and (sub.get("shape") or {}).get("OC"):
            return int(sub["shape"]["OC"])
    # A FUSED dispatch keeps the convolution's shape one level down. `conv2d_batchnorm2d_silu_s8`
    # carries no shape of its own -- the numbers belong to the `conv2d_s8` it fused -- so reading
    # only the top level returns nothing for 57 of this network's 63 packed convolutions, and the
    # rule below then narrows each of them to one hart for want of an output-channel count rather
    # than because the layout forbids the width.
    for sub in op.get("sub_ops") or []:
        if isinstance(sub, dict) and (sub.get("shape") or {}).get("OC"):
            return int(sub["shape"]["OC"])
    return 0


def oc_from_module_name(name: str) -> int:
    """`...xOC16xOH32...` -> 16. The solver writes the dispatch's shape into the module name, so
    this is a second source for the same number when the IR record does not carry it."""
    m = re.search(r"[x_]OC(\d+)[x_]", name or "")
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule"); ap.add_argument("irs", nargs="+", help="network:graph.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sched = json.load(open(a.schedule))
    irs = {}
    for spec in a.irs:
        net, path = spec.split(":", 1)
        irs[net] = {int(op["dispatch_id"]): op for op in json.load(open(path)).get("ops", []) if op.get("dispatch_id") is not None}
    known = set(irs)
    disp = sched["dispatches"]
    # pass 1: the width each (network, dispatch) is allowed
    widths = collections.defaultdict(set)
    modnames = {}
    for v in disp.values():
        net, _ = split_job_name(v["job_name"], known)
        if net not in irs:
            continue
        widths[(net, int(v["id"]))].add(len([h for h in v["hardware_target"].split("+") if h.strip()]))
        modnames.setdefault((net, int(v["id"])), v.get("module_name", ""))
    allowed = {}
    no_oc = []
    for (net, did), ws in widths.items():
        op = irs[net].get(did, {})
        w = min(ws)
        if op.get("op") in PACKED:
            oc = oc_of(op) or oc_from_module_name(modnames.get((net, did), ""))
            if oc <= 0 and w > 1:
                no_oc.append((net, did))
            while w > 1 and (oc <= 0 or oc % w):
                w -= 1
        allowed[(net, did)] = w
    if no_oc:
        print(f"  {len(no_oc)} packed dispatches narrowed for want of an output-channel count: "
              f"{no_oc[:6]}{' ...' if len(no_oc) > 6 else ''}")
    # pass 2: rewrite the targets, keeping the first w harts the solver reserved
    changed = 0
    for v in disp.values():
        net, _ = split_job_name(v["job_name"], known)
        if net not in irs:
            continue
        harts = [h for h in v["hardware_target"].split("+") if h.strip()]
        w = allowed[(net, int(v["id"]))]
        if len(harts) != w:
            v["hardware_target"] = "+".join(harts[:w]); changed += 1
    # the executed table must say which solver produced it: the solve writes that into its
    # *_report.json sidecar, next to the schedule it was copied from
    rep = None
    for cand in (a.schedule.replace(".json", "_report.json"),
                 *(g for g in glob.glob(os.path.join(os.path.dirname(a.schedule), "scheduled_*_report.json"))
                   if json.load(open(g)).get("solve_hash") == sched.get("metadata", {}).get("solve_hash"))):
        if os.path.exists(cand):
            rep = json.load(open(cand)); break
    if not rep:   # the copies are named fig_<tag>_<solver>.json by scripts/solve_stage2*.sh
        base = os.path.basename(a.schedule)
        rep = {"solver_name": "greedy_periodic" if "greedy" in base else ("cpsat" if "cpsat" in base else None), "solver_status": None, "solve_wall_s": None}
    if rep:
        sched.setdefault("metadata", {})["solver"] = rep.get("solver_name")
        sched["metadata"]["solver_status"] = rep.get("solver_status"); sched["metadata"]["solve_wall_s"] = rep.get("solve_wall_s")
    sched.setdefault("metadata", {})["solved_from"] = os.path.basename(a.schedule)
    sched.setdefault("metadata", {})["width_clamp"] = {"changed_dispatches": changed,
                                                        "rule": "min width across instances; conv widths must divide OC; never raised"}
    json.dump(sched, open(a.out, "w"), indent=1)
    hist = collections.Counter(v["hardware_target"].count("+") + 1 for v in disp.values())
    print(f"{a.out}: {changed} dispatch targets narrowed; width histogram {dict(sorted(hist.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
