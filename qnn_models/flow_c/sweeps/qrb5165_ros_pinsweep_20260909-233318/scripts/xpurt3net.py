#!/usr/bin/env python3
"""The XPU-RT side of the `3net` arm, so that arm has something to be compared
against.

The 42-cell main arm compares against XPU-RT numbers that already existed
(`sweeps/qrb5165_sched_algo_sweep10_20260908-210226/results/phase4_results.json`).
The 3net shapes have no such measurement on this board, so this emits them,
schedules them and runs them the same way that sweep did:

    emit      data/toplevel/rospin3net/networks_<shape>.json  (the taskset)
              specs3net/<shape>.flowc.json                    (the bindings)
    solve     scripts/run_xpurt_schedule.py --profiled, one schedule per solver
    runtime   flow_c.py runtime --schedule ...
    run       flow_c.py run --tuned, 3 reps, median with spread

The machine given to the scheduler is the SAME three lanes the pinning
baseline may choose from -- hta + dsp + cpu, one of each, no GPU. Giving the
scheduler a lane the baseline is denied would not be a comparison.

`flow_c.py artifacts` is deliberately NOT run: it rewrites the shared
`gen/profile/` tree from `measurements/qrb5165_v66.json`, which other work in
this repo depends on. The profile CSVs these four networks need are already
there, and the pinning side reads the same ones.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

sys.path.insert(0, HERE)
import pin3net  # noqa: E402

BOARD = os.environ.get("BOARD", "root@qrb5165")
TOPLEVEL = os.path.join(REPO, "data", "toplevel", "rospin3net")
SPECS = os.path.join(SWEEP, "specs3net")
RUNTIMES = os.path.join(SWEEP, "runtimes3net")
RUNLOGS = os.path.join(SWEEP, "logs", "xpurt3net")
STATE = os.path.join(SWEEP, "results", "xpurt3net_state.json")

SOLVERS = ["greedy", "heft_edf", "cpsat"]
REPS = 3

BINDINGS = {"dronet": "bindings/dronet.json",
            "mlp_control": "bindings/mlp_control.json",
            "yolov8n": "bindings/yolov8n.json",
            "fused_full": "bindings/fused_full.json"}
ONNX_INPUT = {"dronet": "input", "mlp_control": "obs"}


def emit(shape):
    name = shape["name"]
    nets = {}
    for n, v in shape["networks"].items():
        e = {"id": v["id"], "identifier": n,
             "dispatch_deps_path":
                 f"gen/qnn_vmfb/{n}/qrb5165_flowc/CPU/{n}.int8/"
                 f"{n}.int8_dispatch_graph.json",
             "num_instances": v["num_instances"]}
        if v.get("period"):
            e["period"] = v["period"]
            e["window_duration"] = v["window_duration"]
        nets[n] = e
    doc = {
        "_comment": (
            f"RoSE 3net workload shape '{name}' on the QRB5165's three pinning "
            f"lanes. Ported from "
            f"{', '.join(shape['sources'])} by "
            f"qnn_models/flow_c/sweeps/<this sweep>/scripts/xpurt3net.py: the "
            f"networks, periods, windows and instance counts are verbatim; the "
            f"FireSim machine (gemmini_q31 / V256D128_rvv) is replaced by this "
            f"board's hta + dsp + cpu, one of each, and yolov8_nano by the "
            f"640x640 yolov8n this board actually has. Absolute latencies are "
            f"NOT comparable with the FireSim originals. The lane set is "
            f"exactly the one the ROS pinning baseline may choose from, so the "
            f"two sides see the same machine."),
        "hardware": {
            "machines": {"cpu_p": 1, "cpu_e": 1, "cpu_x": 1},
            "profile_hw": {"cpu_p": "HTA", "cpu_e": "DSP", "cpu_x": "CPU"},
            "profile": {"target": "qrb5165_flowc", "topo_tag": "topo_0",
                        "topo_tag_override": False, "gen_root": "gen"},
            "p_core_speedup": 1.0},
        "scheduler": {"random_seed": 42, "solver_verbosity": 2,
                      "time_limit": 120, "use_profiled": True,
                      "prune_periodic": True,
                      "restrict_makespan_to_nonperiodic": False},
        "networks": nets,
    }
    if shape["edges"]:
        doc["edges"] = shape["edges"]
    os.makedirs(TOPLEVEL, exist_ok=True)
    p = os.path.join(TOPLEVEL, f"networks_{name}.json")
    with open(p, "w") as f:
        json.dump(doc, f, indent=1)

    spec = {
        "name": name,
        "_comment": doc["_comment"],
        "target": "qrb5165_flowc", "board": "qrb5165_v66",
        "registry": "cores/qrb5165_qnn.json",
        "measurements": "measurements/qrb5165_v66.json",
        "slots": {"CPU_P": "hta", "CPU_E": "dsp", "CPU_X": "cpu"},
        "ctx_dir": "/root/qnn_runtime_ctx",
        "networks": [
            dict({"name": n, "bindings": BINDINGS[n]},
                 **({"period": v["period"]} if v.get("period") else {}),
                 **({"onnx_input_name": ONNX_INPUT[n]} if n in ONNX_INPUT else {}))
            for n, v in shape["networks"].items()],
    }
    os.makedirs(SPECS, exist_ok=True)
    sp = os.path.join(SPECS, f"{name}.flowc.json")
    with open(sp, "w") as f:
        json.dump(spec, f, indent=1)
    return p, sp


def sched_path(name, solver):
    tag = "" if solver in ("milp", "milp_native") else f"_{solver}"
    return os.path.join(REPO, "schedules",
                        f"scheduled_networks_{name}{tag}_profiled.json")


def cmd_emit(args):
    for s in pin3net.shapes():
        p, sp = emit(s)
        print(f'  {s["name"]:30s} -> {os.path.relpath(p, REPO)}')
    return 0


def cmd_solve(args):
    for s in pin3net.shapes():
        name = s["name"]
        rel = os.path.relpath(os.path.join(TOPLEVEL, f"networks_{name}.json"), REPO)
        for solver in SOLVERS:
            out = sched_path(name, solver)
            if os.path.exists(out) and not args.force:
                print(f"  [skip] {name} {solver}")
                continue
            cmd = [sys.executable, "scripts/run_xpurt_schedule.py",
                   "--networks-json", rel, "--solver", solver, "--profiled"]
            env = dict(os.environ)
            # CP-SAT runs out of process; the repo keeps an ortools venv at
            # .cpsat-venv, which is what the sched_algo_sweep10 port used too.
            venv = os.path.join(REPO, ".cpsat-venv", "bin", "python")
            if os.path.exists(venv):
                env.setdefault("XPURT_CPSAT_PYTHON", venv)
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               timeout=900, env=env)
            ok = os.path.exists(out)
            print(f'  [{"ok" if ok else "FAIL"}] {name:30s} {solver:10s}'
                  + ("" if ok else " " + r.stdout[-300:] + r.stderr[-300:]))
    return 0


SUMMARY = re.compile(r"\[summary\] (\d+)/(\d+) entries executed, wall=([\d.]+) ms "
                     r"\(predicted makespan ([\d.]+) ms")


def np_makespan_from_log(path, aperiodic):
    """The NON-PERIODIC makespan from the run's trace block.

    That is the objective both runtimes are scored on: how long the
    non-periodic work takes while the periodic tasks' constraints are
    honoured. It is extracted the same way `drive.py` extracts it for the
    sched_algo_sweep10 port -- max end over the rows of the aperiodic
    networks -- so the two arms are computed identically.
    """
    if not os.path.exists(path):
        return None, None
    txt = open(path, errors="replace").read()
    m = re.search(r"MODELBLASTER_XPURT_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
                  r"MODELBLASTER_XPURT_TRACE_END", txt, re.S)
    if not m:
        return None, None
    rows = list(csv.DictReader(io.StringIO(m.group(1).strip())))
    unit = {"us": 1e-3, "ms": 1.0, "ns": 1e-6}
    ends, np_ends, counts = [], [], collections.Counter()
    for r in rows:
        try:
            u = unit.get((r.get("time_unit") or "us").strip(), 1e-3)
            en = float(r["actual_end_cycles"]) * u
        except (TypeError, ValueError, KeyError):
            continue
        ends.append(en)
        n = (r.get("network") or "").strip()
        counts[n] += 1
        if n in aperiodic:
            np_ends.append(en)
    got = {n: len({r["instance"] for r in rows
                   if (r.get("network") or "").strip() == n}) for n in aperiodic}
    return (round(max(np_ends), 4) if np_ends
            else (round(max(ends), 4) if ends else None)), got


def cmd_run(args):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    os.makedirs(RUNLOGS, exist_ok=True)
    seen_hash = {}
    for s in pin3net.shapes():
        name = s["name"]
        spec = os.path.join(SPECS, f"{name}.flowc.json")
        for solver in SOLVERS:
            sp = sched_path(name, solver)
            if not os.path.exists(sp):
                continue
            key = f"{name}__{solver}"
            # dedupe identical schedules across solvers, as the XPU-RT sweep does
            # sha256, not hash(): PYTHONHASHSEED makes str hashing
            # non-reproducible across processes, and the dedupe decision has to
            # be the same on a re-run.
            h = hashlib.sha256(json.dumps(
                json.load(open(sp)).get("schedule"),
                sort_keys=True).encode()).hexdigest()
            dup = seen_hash.get((name, h))
            if dup:
                st[key] = {"duplicate_of": dup}
                print(f"  [dup] {key} == {dup}")
                continue
            seen_hash[(name, h)] = key
            if st.get(key, {}).get("ok") and not args.force:
                print(f"  [skip] {key}")
                continue
            out_dir = os.path.join(RUNTIMES, key)
            r = subprocess.run(
                [sys.executable, "flow_c.py", "runtime", "--workload", spec,
                 "--schedule", sp, "--out-dir", out_dir],
                cwd=FLOWC, capture_output=True, text=True, timeout=1800)
            if not os.path.exists(os.path.join(out_dir, "runtime_main.cpp")):
                print(f"  [FAIL runtime] {key}: {r.stdout[-400:]}{r.stderr[-400:]}")
                st[key] = {"ok": False, "stage": "runtime"}
                continue
            aperiodic = {n for n, v in s["networks"].items()
                         if not v.get("period")}
            walls, nps, npcount = [], [], None
            for rep in range(1, REPS + 1):
                ld = os.path.join(RUNLOGS, key, f"rep{rep}")
                os.makedirs(ld, exist_ok=True)
                t0 = time.time()
                subprocess.run(
                    [sys.executable, "flow_c.py", "run", "--workload", spec,
                     "--out-dir", out_dir, "--tag", key, "--tuned",
                     "--board", BOARD, "--board-dir", "/root/flowc_rospin3net",
                     "--log-dir", ld],
                    cwd=FLOWC, capture_output=True, text=True, timeout=1800)
                log = os.path.join(ld, "run.log")
                m = None
                if os.path.exists(log):
                    for m in SUMMARY.finditer(open(log, errors="replace").read()):
                        pass
                if m and int(m.group(1)) == int(m.group(2)):
                    walls.append(float(m.group(3)))
                    npm, npc = np_makespan_from_log(log, aperiodic)
                    if npm is not None:
                        nps.append(npm)
                        npcount = npc
                print(f"    {key} rep{rep} "
                      f"{'%.3f' % walls[-1] if walls else 'FAIL'} ms "
                      f"({time.time() - t0:.0f}s)")
            st[key] = dict(ok=len(walls) == REPS, reps_ms=walls,
                           median_ms=round(statistics.median(walls), 4) if walls else None,
                           spread_ms=round(max(walls) - min(walls), 4) if walls else None,
                           np_reps_ms=nps,
                           np_median_ms=round(statistics.median(nps), 4) if nps else None,
                           np_spread_ms=round(max(nps) - min(nps), 4) if nps else None,
                           np_instances=npcount,
                           predicted_ms=float(m.group(4)) if m else None)
            with open(STATE, "w") as f:
                json.dump(st, f, indent=1)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=1)
    return 0


def cmd_rescan(args):
    """Recompute the non-periodic makespan from run logs already on disk."""
    st = json.load(open(STATE))
    shapes = {s["name"]: s for s in pin3net.shapes()}
    for key, rec in st.items():
        if rec.get("duplicate_of") or not rec.get("ok"):
            continue
        name = key.split("__")[0]
        aperiodic = {n for n, v in shapes[name]["networks"].items()
                     if not v.get("period")}
        nps, npc = [], None
        for rep in range(1, REPS + 1):
            log = os.path.join(RUNLOGS, key, f"rep{rep}", "run.log")
            npm, c = np_makespan_from_log(log, aperiodic)
            if npm is not None:
                nps.append(npm)
                npc = c
        rec["np_reps_ms"] = nps
        rec["np_median_ms"] = round(statistics.median(nps), 4) if nps else None
        rec["np_spread_ms"] = round(max(nps) - min(nps), 4) if nps else None
        rec["np_instances"] = npc
        print(f'  {key:40s} np={rec["np_median_ms"]}  {npc}')
    with open(STATE, "w") as f:
        json.dump(st, f, indent=1)
    return 0


def cmd_collect(args):
    st = json.load(open(STATE))
    out = {}
    for key, rec in st.items():
        if rec.get("duplicate_of"):
            src = st.get(rec["duplicate_of"], {})
            rec = dict(src, measured_via=rec["duplicate_of"])
        name, solver = key.split("__")
        if not rec.get("ok"):
            continue
        e = out.setdefault(name, {"solvers": {}})
        e["solvers"][solver] = rec
    for name, e in out.items():
        s = e["solvers"]
        e["best_makespan_ms"] = min(v["median_ms"] for v in s.values())
        e["best_solver"] = min(s, key=lambda k: s[k]["median_ms"])
        e["greedy_makespan_ms"] = s.get("greedy", {}).get("median_ms")
        nps = {k: v["np_median_ms"] for k, v in s.items() if v.get("np_median_ms")}
        if nps:
            e["best_np_ms"] = min(nps.values())
            e["best_np_solver"] = min(nps, key=nps.get)
        e["np_instances"] = next((v.get("np_instances") for v in s.values()
                                  if v.get("np_instances")), None)
        e["n_solvers_measured"] = len(s)
    p = os.path.join(SWEEP, "results", "xpurt3net.json")
    with open(p, "w") as f:
        json.dump({"_comment":
                   "Measured XPU-RT makespans for the 3net shapes on the same "
                   "three lanes the pinning baseline gets. Medians of 3 reps "
                   "with spreads; `flow_c.py run --tuned`, the same mode the "
                   "sched_algo_sweep10 port used.",
                   "board": BOARD, "reps": REPS, "solvers": SOLVERS,
                   "shapes": out}, f, indent=1)
    print(f"wrote {p}: {len(out)} shapes")
    for n, e in sorted(out.items()):
        print(f'  {n:30s} best={e["best_makespan_ms"]:9.3f} '
              f'({e["best_solver"]}) n_solvers={e["n_solvers_measured"]}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("emit", cmd_emit), ("solve", cmd_solve),
                     ("run", cmd_run), ("rescan", cmd_rescan),
                     ("collect", cmd_collect)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--force", action="store_true")
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
