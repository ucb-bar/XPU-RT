#!/usr/bin/env python3
"""Flow C driver — PyTorch in, a scheduled QNN runtime out.

    python3 flow_c.py ir         # modelblaster extract_graph (+ ONNX export)
    python3 flow_c.py artifacts  # dispatch graphs + profile CSVs + workload spec
    python3 flow_c.py schedule   # xpu-rt scheduler
    python3 flow_c.py runtime    # modelblaster ingest -> QNN runtime sources
    python3 flow_c.py stage      # put context binaries where the board expects
    python3 flow_c.py run        # build on the board, run, capture the trace
    python3 flow_c.py all

Every stage is re-enterable and writes to disk; run one, inspect, run the
next.  `--workload` selects the spec under workloads/ (default:
dronet_mlp_yolo.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from flowc import (artifacts, bindings as bmod, emit_runtime, ir as irmod, mb,  # noqa: E402
                   plots as plotmod, schedule as smod)

REPO = mb.repo_root()


def _p(*parts) -> str:
    return os.path.join(HERE, *parts)


def load_workload(name: str) -> dict:
    path = name if os.path.isabs(name) else _p("workloads", name)
    if not path.endswith(".json"):
        path += ".json"
    with open(path) as f:
        wl = json.load(f)
    wl["_path"] = path
    return wl


def resolve(wl: dict) -> tuple[dict, dict, dict]:
    """-> (irs, binding sets, registry) for every network in the workload."""
    reg = mb.core_registry().load(_p(wl["registry"]))
    irs, bsets = {}, {}
    for net in wl["networks"]:
        doc_path = _p(net["bindings"])
        with open(doc_path) as f:
            doc = json.load(f)
        spec = doc["ir"].get("graph_json")
        spec = f"graph_json:{os.path.join(REPO, 'qnn_models', spec)}" if spec else doc["ir"]["source"]
        ir = irmod.load(spec, work_dir=_p("gen", "ir"), quant=doc["ir"].get("quant", "int8"))
        ir["name"] = net["name"]
        irs[net["name"]] = ir
        bsets[net["name"]] = bmod.load(doc_path, ir)
    return irs, bsets, reg


def stage_ir(wl: dict, args) -> None:
    irs, bsets, reg = resolve(wl)
    for name, ir in irs.items():
        bset = bsets[name]
        feas = bmod.check(bset, ir, reg)
        print(f"\n{name}: {len(ir['ops'])} IR ops from {bset.ir_spec or 'graph_json'}, "
              f"{len(bset.bindings)} binding(s)")
        for f in feas:
            ok = ", ".join(f.kinds_ok()) or "none"
            blocked = {k: v for k, v in f.allowed.items() if v}
            print(f"  {f.binding.name:22} ir ops {f.binding.first:>3}..{f.binding.last:<3} "
                  f"({f.binding.last - f.binding.first + 1:>3})  runs on: {ok}")
            for k, v in blocked.items():
                print(f"      {k:4} blocked by {', '.join(v)}")
        for note in bmod.reconcile(feas):
            print(f"  note: {note}")
    # ONNX export from the same module, for the QNN converter side.
    if args.export_onnx:
        for net in wl["networks"]:
            src = bsets[net["name"]].ir_spec
            if not src.startswith("pytorch:"):
                print(f"{net['name']}: graph_json-sourced, no ONNX export")
                continue
            out = _p("gen", "onnx", f"{net['name']}.onnx")
            irmod.onnx_from_pytorch(src.split(":", 1)[1], out,
                                    input_name=net.get("onnx_input_name", "input"))


def stage_artifacts(wl: dict, args) -> None:
    irs, bsets, reg = resolve(wl)
    qnn_cores = smod.load_qnn_cores(_p(wl["registry"]))
    kind_to_hw = {k: c.label for k, c in qnn_cores.items()}
    slot_to_hw = {slot: kind_to_hw[kind] for slot, kind in wl["slots"].items()}
    used_kinds = {kind: kind_to_hw[kind] for kind in wl["slots"].values()}
    with open(_p(wl["measurements"])) as f:
        meas = json.load(f)

    periodic = {n["name"] for n in wl["networks"] if n.get("period")}
    horizon_ms = artifacts.measured_horizon_ms(bsets, meas, periodic)

    gen_vmfb = os.path.join(REPO, "gen", "qnn_vmfb")
    gen_prof = os.path.join(REPO, "gen", "profile")
    nets_for_spec, all_warnings = [], []
    for net in wl["networks"]:
        name = net["name"]
        bset, ir = bsets[name], irs[name]
        feas = bmod.check(bset, ir, reg)
        graphs = artifacts.emit_dispatch_graphs(bset, ir, gen_vmfb, wl["target"],
                                                sorted(set(used_kinds.values())))
        paths, warnings = artifacts.emit_profiles(bset, feas, meas, gen_prof,
                                                  wl["target"], used_kinds)
        all_warnings += warnings
        print(f"{name}: {len(graphs)} dispatch graph(s), {len(paths)} profile CSV(s)")
        rel = os.path.relpath(artifacts.mb.emit_dispatch_graph().output_path(
            gen_vmfb, name, wl["target"], sorted(used_kinds.values())[0], bset.quant), REPO)
        n_inst = (math.ceil(horizon_ms / net["period"]) if net.get("period") else None)
        nets_for_spec.append({"name": name, "dispatch_deps_path": rel,
                              "period": net.get("period"),
                              "num_instances": n_inst})
    for w in all_warnings:
        print(f"  excluded: {w}")
    print(f"measured horizon (non-periodic worst case): {horizon_ms:.2f} ms -> "
          + ", ".join(f"{n['name']}x{n['num_instances']}"
                      for n in nets_for_spec if n.get("num_instances")))

    spec = artifacts.emit_workload_spec(
        nets_for_spec,
        os.path.join(REPO, "data", "toplevel", f"networks_{wl['name']}.json"),
        wl["target"], slot_to_hw,
        comment=wl.get("_comment", ""))
    print(f"workload spec: {os.path.relpath(spec, REPO)}")


def schedule_path(wl: dict) -> str:
    return os.path.join(REPO, "schedules",
                        f"scheduled_networks_{wl['name']}_{wl.get('solver_tag','profiled')}.json")


def stage_schedule(wl: dict, args) -> None:
    spec = os.path.join(REPO, "data", "toplevel", f"networks_{wl['name']}.json")
    cmd = [sys.executable, "scripts/run_xpurt_schedule.py",
           "--networks-json", os.path.relpath(spec, REPO),
           "--solver", args.solver, "--profiled"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def _tag(args) -> str:
    return f"_{args.tag}" if args.tag else ""


def stage_runtime(wl: dict, args) -> None:
    irs, bsets, _ = resolve(wl)
    sched = args.schedule or _find_schedule(wl, args.solver)
    entries = smod.ingest(sched, bsets, irs, _p(wl["registry"]), wl["slots"])
    print(smod.summarize(entries))
    out_dir = args.out_dir or _p("gen", "runtime", wl["name"] + _tag(args))
    info = emit_runtime.emit(entries, out_dir, wl.get("ctx_dir", "/root/qnn_runtime_ctx"),
                             lane_mode=args.lane_mode)
    print(f"\nemitted {out_dir}/dispatch_table.h + runtime_main.cpp")
    print(f"  lane mode {info['lane_mode']}: " + ", ".join(info["lanes"]))


def _find_schedule(wl: dict, solver: str) -> str:
    import glob
    pats = [os.path.join(REPO, "schedules", f"scheduled_networks_{wl['name']}*{solver}*.json"),
            os.path.join(REPO, "schedules", f"scheduled_networks_{wl['name']}*.json")]
    for p in pats:
        hits = sorted(glob.glob(p), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no schedule for {wl['name']} — run `flow_c.py schedule` first")


def stage_stage(wl: dict, args) -> None:
    """Symlink/copy the context binaries the emitted table names into the board's ctx dir."""
    irs, bsets, _ = resolve(wl)
    wanted = {}
    for bset in bsets.values():
        for b in bset.bindings:
            for kind, bb in b.backends.items():
                wanted[bb.ctx] = bb.graph
    host = args.board
    ctx_dir = wl.get("ctx_dir", "/root/qnn_runtime_ctx")
    src_dir = args.ctx_source
    script = ["set -e", f"mkdir -p {ctx_dir}"]
    for ctx in sorted(wanted):
        script.append(f'if [ ! -e {ctx_dir}/{ctx} ] && [ -e {src_dir}/{ctx} ]; then '
                      f'ln -sf {src_dir}/{ctx} {ctx_dir}/{ctx}; fi')
    script.append(f'ls -l {ctx_dir} | grep -E "' + "|".join(sorted(wanted)) + '" || true')
    out = subprocess.run(["ssh", host, "bash -s"], input="\n".join(script),
                         text=True, capture_output=True)
    print(out.stdout or out.stderr)


def stage_run(wl: dict, args) -> None:
    out_dir = args.out_dir or _p("gen", "runtime", wl["name"] + _tag(args))
    env = dict(os.environ)
    if args.log_dir:
        env["LOG_DIR"] = args.log_dir
    cmd = ["bash", "qnn_models/runtime/deploy_and_run.sh",
           os.path.relpath(out_dir, REPO), args.board,
           args.board_dir + _tag(args)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=False, cwd=REPO, env=env)


def stage_plots(wl: dict, args) -> None:
    # deploy_and_run.sh defaults LOG_DIR to runs/<gen dir basename>.
    log_dir = args.log_dir or os.path.join(REPO, "runs", wl["name"] + _tag(args))
    log = os.path.join(log_dir, "run.log")
    if not os.path.exists(log):
        raise FileNotFoundError(f"no run log at {log} — run `flow_c.py run` first")
    out_dir = os.path.join(REPO, "plots")
    res = plotmod.render(
        log,
        out_full=os.path.join(out_dir, f"{wl['name']}{_tag(args)}_predicted_vs_actual.png"),
        out_zoom=os.path.join(out_dir, f"{wl['name']}{_tag(args)}_predicted_vs_actual_zoom.png"),
        csv_path=os.path.join(log_dir, "trace.csv"),
        source=f"{wl.get('board', 'board')} (QNN lanes)")
    print(res["summary"])
    print(f"\npredicted vs actual : {os.path.relpath(res['full'], REPO)}")
    if "zoom" in res:
        print(f"  zoom (first {res['zoom_ms']:.0f} ms): {os.path.relpath(res['zoom'], REPO)}")
    print(f"  flat trace CSV     : {os.path.relpath(os.path.join(log_dir, 'trace.csv'), REPO)}")
    # The scheduler drew the predicted-only timeline when it solved.
    import glob
    # The scheduler names its own timeline after the solver it ran.
    sched_png = os.path.join(out_dir, f"networks_{wl['name']}_profiled.png") if args.tag \
        else os.path.join(out_dir, f"networks_{wl['name']}_{args.solver}_profiled.png")
    if not os.path.exists(sched_png):
        hits = sorted(glob.glob(os.path.join(out_dir, f"networks_{wl['name']}*profiled.png")),
                      key=os.path.getmtime, reverse=True)
        sched_png = hits[0] if hits else ""
    if sched_png:
        print(f"  scheduler timeline : {os.path.relpath(sched_png, REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ir", "artifacts", "schedule", "runtime",
                                      "stage", "run", "plots", "all"])
    ap.add_argument("--workload", default="dronet_mlp_yolo.json")
    ap.add_argument("--solver", default="greedy_periodic")
    ap.add_argument("--schedule", default=None, help="explicit schedules/*.json")
    ap.add_argument("--lane-mode", default="kind", choices=["kind", "kind-network"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for the runtime dir, log dir and "
                                              "plot names — lets two solvers coexist")
    ap.add_argument("--board", default=os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201"))
    ap.add_argument("--board-dir", default="/root/flowc_runtime")
    ap.add_argument("--ctx-source", default="/root/repro_perlane",
                    help="board-side directory holding the context binaries to link")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--export-onnx", action="store_true",
                    help="also export ONNX from the same PyTorch module (ir stage)")
    args = ap.parse_args()

    wl = load_workload(args.workload)
    stages = {"ir": stage_ir, "artifacts": stage_artifacts, "schedule": stage_schedule,
              "runtime": stage_runtime, "stage": stage_stage, "run": stage_run,
              "plots": stage_plots}
    order = ["ir", "artifacts", "schedule", "runtime", "stage", "run", "plots"] \
        if args.stage == "all" else [args.stage]
    for s in order:
        print(f"\n=== {s} " + "=" * (60 - len(s)))
        stages[s](wl, args)


if __name__ == "__main__":
    main()
