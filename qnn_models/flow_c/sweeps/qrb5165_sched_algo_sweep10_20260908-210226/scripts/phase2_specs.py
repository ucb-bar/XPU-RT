#!/usr/bin/env python3
"""Phase 2b — the Flow C side of each workload cell, and the shared artifacts.

Two halves of this flow read different files: `run_xpurt_schedule.py` /
`sweep10_runner.py` read the TASKSET (data/toplevel/s10port/networks_*.json,
written by mk_workloads_qrb5165.py), and `flow_c.py` reads which binding
manifest each network name resolves to. So each cell gets a Flow C spec here,
with the config's own slot -> kind map.

`artifacts` writes, per (network, backend), the two files xpu-rt's
profile_loader and workload_factory actually read:

    gen/qnn_vmfb/<net>/qrb5165_flowc/<HW>/<net>.int8/<net>.int8_dispatch_graph.json
    gen/profile/<HW>/qrb5165_flowc/<net>/<net>.int8/.../topo_0/results.csv

They are emitted from THIS SWEEP'S FROZEN cost_model.json, and are re-emitted
before the solves rather than trusted, because the shared gen/ tree is written
by every other tenant of this repo too. One pass over a spec that names all
four backends covers every config, since the files are per (network, backend)
and a config just selects which of them the taskset points at.

    python3 phase2_specs.py [--write] [--artifacts]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
sys.path.insert(0, HERE)
from mk_workloads_qrb5165 import CONFIGS, HW_LABEL, ARM, bindings_rel  # noqa: E402

SPECS = os.path.join(SWEEP, "specs")
REL = os.path.relpath(SWEEP, FLOWC)
COST_REL = os.path.join(REL, "cost_model.json")

#: every backend any config uses, plus hta so the profile tree is complete and
#: an hta placement (if one ever composed) would have a real cell rather than
#: an exclusion cost.
ALL_SLOTS = {"CPU_P": "dsp", "CPU_E": "cpu", "CPU_X": "gpu", "CPU_G": "hta"}


def spec_for(workload, doc, cfg):
    nets = []
    for name, e in doc["networks"].items():
        n = {"name": name, "bindings": bindings_rel(name)}
        if e.get("period") is not None:
            n["period"] = e["period"]
        nets.append(n)
    return {
        "name": workload,
        "_comment": doc["_comment"],
        "target": "qrb5165_flowc",
        "board": "qrb5165_v66",
        "registry": "cores/qrb5165_qnn.json",
        "measurements": COST_REL,
        "slots": dict(CONFIGS[cfg]),
        "ctx_dir": "/root/qnn_runtime_ctx",
        "networks": nets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--artifacts", action="store_true")
    a = ap.parse_args()
    tdir = os.path.join(REPO, "data", "toplevel", ARM)
    if not os.path.isdir(tdir):
        sys.exit(f"no tasksets at {tdir} — run mk_workloads_qrb5165.py --emit first")
    os.makedirs(SPECS, exist_ok=True)
    n = 0
    all_nets = set()
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".json"):
            continue
        workload = fn[:-5]
        cfg = workload.rsplit("_", 1)[1]
        if cfg not in CONFIGS:
            print(f"  {workload}: unknown config {cfg!r} — skipped")
            continue
        doc = json.load(open(os.path.join(tdir, fn)))
        sp = spec_for(workload, doc, cfg)
        all_nets |= set(doc["networks"])
        if a.write:
            with open(os.path.join(SPECS, f"{workload}.flowc.json"), "w") as f:
                json.dump(sp, f, indent=2)
            n += 1
    print(f"  {n} Flow C spec(s) -> {SPECS}" if a.write
          else "  (dry run — pass --write)")
    print(f"  {len(all_nets)} distinct networks: {', '.join(sorted(all_nets))}")

    # the artifacts spec: every network, every backend, no periods (this spec is
    # never scheduled -- it exists only to emit the per-(net, backend) files)
    art = {
        "name": "s10port_artifacts",
        "_comment": ("Artifacts-only spec for the sched_algo_sweep10 QRB5165 "
                     "port: names every network and every backend so one "
                     "`flow_c.py artifacts` pass emits the dispatch graphs and "
                     "profile CSVs all four lane configs read. The "
                     "data/toplevel/networks_s10port_artifacts.json it also "
                     "writes is a by-product and is NOT the taskset any cell "
                     "is solved from -- those come from "
                     "scripts/mk_workloads_qrb5165.py."),
        "target": "qrb5165_flowc",
        "board": "qrb5165_v66",
        "registry": "cores/qrb5165_qnn.json",
        "measurements": COST_REL,
        "slots": dict(ALL_SLOTS),
        "ctx_dir": "/root/qnn_runtime_ctx",
        "networks": [{"name": x, "bindings": bindings_rel(x)}
                     for x in sorted(all_nets)],
    }
    ap_path = os.path.join(SPECS, "s10port_artifacts.flowc.json")
    if a.write:
        with open(ap_path, "w") as f:
            json.dump(art, f, indent=2)
        print(f"  artifacts spec -> {os.path.relpath(ap_path, SWEEP)}")
    if a.artifacts:
        log = os.path.join(SWEEP, "logs", "phase2_artifacts.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        p = subprocess.run([sys.executable, "flow_c.py", "artifacts",
                            "--workload", ap_path],
                           cwd=FLOWC, capture_output=True, text=True)
        with open(log, "w") as f:
            f.write((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
        print(f"  flow_c.py artifacts rc={p.returncode}  (log {log})")
        for line in (p.stdout or "").splitlines():
            if "excluded:" in line or "profile CSV" in line:
                print("    " + line.strip())
        if p.returncode != 0:
            print("    stderr tail: " + (p.stderr or "")[-600:])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
