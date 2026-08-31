#!/usr/bin/env python3
"""Build the periodic-only addendum to sweep qrb5165_20260829-200620.

The sweep's generator rejects any draw with zero non-periodic work, on the
stated grounds that there is "nothing to pack against the periodic load".
That is right for Q2/Q3, which are about how non-periodic vision work packs
against a periodic base. It is exactly wrong for Q1.

Q1 asks whether fused_full beats dronet in the mid-size periodic slot. In all
12 accepted points a yolov8n_head finishes last, so the mid-size slot cannot
move the makespan by more than second order -- which is why Q1 came back weak
(3 of 5 pairs, only one clearing the 4.4% noise floor). The six rejected draws
are the only ones in the taskset where the mid-size periodic model IS the
critical path, so they are the sharpest available test of Q1, and the
predicate throws away precisely them.

They remain matched pairs: baseline_seedN and fused_seedN differ only in the
mid-size slot, same as the accepted pairs.

This is an ADDENDUM, not a re-run. The sweep's pre-registered record --
generated.json, results.json, provenance.json, state.json -- is not touched;
the six points keep status REJECTED there. Everything here lands in its own
tree via drive.py's SWEEP_OUT, and reuses the sweep's frozen cost_model.json
so the two are directly comparable.
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "addendum_periodic_only")
sys.path.insert(0, HERE)
from sweep_unbounded_nonperiodic import flow_c_spec       # noqa: E402

for d in ("workloads", "logs", "schedules", "runtimes", "runs"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

gen = json.load(open(os.path.join(HERE, "generated.json")))
rows = []
for r in gen:
    if r["status"] != "REJECTED":
        continue
    pt = r["point"]
    src = os.path.join(HERE, r["workload"])            # workloads/rejected/<pt>.json
    cfg = json.load(open(src))
    shutil.copy2(src, os.path.join(OUT, "workloads", f"{pt}.json"))
    with open(os.path.join(OUT, "workloads", f"{pt}.flowc.json"), "w") as f:
        json.dump(flow_c_spec(pt, cfg), f, indent=2)
    row = dict(r)
    row["status"] = "ok"
    row["reinstated_from"] = {
        "record": "generated.json",
        "original_status": "REJECTED",
        "original_problems": r["problems"],
        "rationale": ("periodic-only draw: the mid-size periodic model is the "
                      "critical path, which is the condition Q1 needs and the "
                      "12 accepted points never provide"),
    }
    row["workload"] = f"workloads/{pt}.json"
    row["flowc_spec"] = f"workloads/{pt}.flowc.json"
    rows.append(row)

with open(os.path.join(OUT, "generated.json"), "w") as f:
    json.dump(rows, f, indent=1)
print(f"addendum: {len(rows)} points -> {OUT}")
for r in rows:
    nets = ", ".join(f"{k}x{v['num_instances']}" for k, v in r["networks"].items())
    print(f"  {r['point']:<18} {nets}")
