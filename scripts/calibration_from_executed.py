#!/usr/bin/env python3
"""Per-dispatch board calibration fitted from the executed traces of one table, joined through the IR.

The runner stamps each trace record with its SLOT in the kernel-call array, not the IR dispatch id
(zero-cost ops such as chunk/split/slice emit no record), so on a network with zero-cost ops the
trace's `dispatch_id` drifts from the schedule's. This joins every record to the schedule through
`k1_trace.ir_slot_map` (the IR's own numbering), takes the predicted duration from the schedule the
trace executed, and measures each dispatch's SERVICE time on its hart: its execution (rdtime ticks)
plus the idle that followed it before the hart's next dispatch beyond what the plan itself left there
(dispatch launch, cross-hart dependency waits — the runtime overhead the isolated profile cannot see;
on the K1 dispatches execute faster than the profile predicts while this overhead slides the timeline
behind the plan). The per-dispatch multiplier is the median of service/predicted over the warm
instances of every run given; `--execution-only` measures execution alone instead. The op tier
(fallback for dispatches without a key) is the median over all well-measured dispatches of that op
kind; the aggregate is the ratio of sums. Output schema: k1_board_calibration/v2, the table
`--board-calibration` reads.

    scripts/calibration_from_executed.py --schedule schedules/fig_<tag>_cpsat_hard_clamped.json \\
        --trace-glob 'results/codesign_feedback/xpurt_long/trace_<tag>cpsat_hardr*_other_run1.csv' --out <cal.json>
"""
from __future__ import annotations
import argparse, collections, csv, datetime, glob, json, os, re, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from k1_trace import ir_slot_map, K1_RDTIME_HZ   # noqa: E402
from job_names import split_job_name             # noqa: E402
IRS = {"yolov8_nano_64x96": "ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/graph.json", "fused_full": "ModelBlaster/build/k1_xpurt/fused_full/int8/graph.json",
       "mlp_control": "ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json", "ffn_block": "ModelBlaster/build/k1_xpurt/ffn_block/int8/graph.json", "dronet": "ModelBlaster/build/k1_xpurt/dronet/int8/graph.json"}


def op_of(module_name):
    m = re.search(r"_x60_(.+?)_[A-Z0-9]+x[A-Z0-9]", module_name or "") or re.search(r"_x60_([a-z0-9_]+)$", module_name or "")
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True); ap.add_argument("--trace", action="append", default=[]); ap.add_argument("--trace-glob", action="append", default=[])
    ap.add_argument("--ir", action="append", default=[], help="net=graph.json (defaults for the deployed nets)")
    ap.add_argument("--min-pred-ms", type=float, default=0.05); ap.add_argument("--skip-instances", type=int, default=1, help="cold instances dropped per run")
    ap.add_argument("--workload", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--execution-only", action="store_true", help="execution time alone (no runtime overhead)")
    ap.add_argument("--stat", choices=["median", "mean"], default="median", help="per-dispatch statistic: median of the ratios, or the ratio of sums (mean)")
    a = ap.parse_args()
    irs = dict(IRS); irs.update(dict(x.split("=", 1) for x in a.ir))
    sched = json.load(open(a.schedule)); nets = set(sched.get("metadata", {}).get("periodic_networks", {}) or [])
    pred, opk = {}, {}
    for v in sched["dispatches"].values():
        net, inst = split_job_name(v["job_name"], nets or set(irs))
        pred[(net, int(inst or 0), int(v["id"]))] = float(v["duration"]); opk[(net, int(v["id"]))] = op_of(v.get("module_name", ""))
    slotmap = {}
    for net in {k[0] for k in pred}:
        p = os.path.join(REPO, irs.get(net, "")) if net in irs else ""
        if p and os.path.exists(p):
            slotmap[net] = ir_slot_map(json.load(open(p)))
    traces = list(a.trace) + [t for g in a.trace_glob for t in sorted(glob.glob(g))]
    plan_start = {}
    for v in sched["dispatches"].values():
        net, inst = split_job_name(v["job_name"], nets or set(irs)); plan_start[(net, int(inst or 0), int(v["id"]))] = float(v["start_time"])
    samples = collections.defaultdict(list); tot_a = tot_p = 0.0; unmapped = collections.Counter()
    for t in traces:
        recs = []
        for r in csv.DictReader(open(t)):
            try:
                net = r["network"]; inst = int(r["instance"]); slot = int(r["dispatch_id"]); s0 = int(r["actual_start_cycles"]); e0 = int(r["actual_end_cycles"]); hart = int(r["worker_hart"])
            except (KeyError, ValueError, TypeError):
                continue
            if slot < 0 or e0 <= s0:
                continue
            did = slotmap.get(net, {}).get(slot, slot if net not in slotmap else None)
            if did is None:
                unmapped[net] += 1; continue
            recs.append((hart, s0 / K1_RDTIME_HZ * 1e3, e0 / K1_RDTIME_HZ * 1e3, net, inst, did))
        byh = collections.defaultdict(list)
        for rec in recs:
            byh[rec[0]].append(rec)
        for hart, L in byh.items():
            L.sort(key=lambda x: x[1])
            for i, (h, s0, e0, net, inst, did) in enumerate(L):
                p = pred.get((net, inst, did))
                if p is None or inst < a.skip_instances or p < a.min_pred_ms:
                    continue
                service = e0 - s0
                if not a.execution_only and i + 1 < len(L):
                    nxt = L[i + 1]; ps = plan_start.get((net, inst, did)); nps = plan_start.get((nxt[3], nxt[4], nxt[5]))
                    gap_exec = max(0.0, nxt[1] - e0)
                    gap_plan = max(0.0, nps - (ps + p)) if (ps is not None and nps is not None) else 0.0
                    service += max(0.0, gap_exec - gap_plan)        # the overhead the plan did not leave room for
                samples[(net, did)].append(service / p); tot_a += service; tot_p += p
    sums = collections.defaultdict(lambda: [0.0, 0.0])
    agg_fn = statistics.median if a.stat == "median" else statistics.mean
    per_disp = {f"{net}/{did}": round(agg_fn(v), 4) for (net, did), v in samples.items() if len(v) >= 3}
    by_op = collections.defaultdict(list)
    for (net, did), v in samples.items():
        if opk.get((net, did)):
            by_op[opk[(net, did)]].extend(v)
    per_op = {op: round(agg_fn(v), 4) for op, v in by_op.items() if len(v) >= 10}
    per_net = {}
    for net in {k[0] for k in samples}:
        v = [x for (n, d), vv in samples.items() if n == net for x in vv]
        per_net[net] = {"keys": sum(1 for (n, d) in samples if n == net), "median": round(statistics.median(v), 3), "n_samples": len(v)}
    out = {"schema": "k1_board_calibration/v2", "source": f"{len(traces)} executed board trace(s), real SpaceMiT K1, joined to the schedule through the IR slot map",
           "workload": a.workload or os.path.basename(a.schedule), "schedule": os.path.relpath(a.schedule, REPO), "traces": [os.path.relpath(t, REPO) for t in traces],
           "generated_by": "scripts/calibration_from_executed.py", "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "measures": "execution only" if a.execution_only else "service time = execution + the unplanned idle that followed the dispatch on its hart (runtime overhead: launch, cross-hart waits)",
           "statistic": f"{a.stat} of service/predicted per dispatch over warm instances (first {a.skip_instances} dropped), predicted >= {a.min_pred_ms} ms; op tier = median over its dispatches; aggregate = sum(actual)/sum(predicted)",
           "aggregate_multiplier": round(tot_a / tot_p, 4) if tot_p else 1.0, "per_network": per_net, "unmapped_records": dict(unmapped),
           "primary_key": "network/dispatch_id (IR numbering; exact, measured)", "per_dispatch_multiplier": per_disp,
           "fallback_key": "op (median over the measured dispatches of that op kind)", "per_op_multiplier": per_op}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: aggregate {out['aggregate_multiplier']}, per network {per_net}, unmapped {dict(unmapped)}, op tier {per_op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
