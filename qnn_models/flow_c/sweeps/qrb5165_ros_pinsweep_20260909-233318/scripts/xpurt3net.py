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

Two things changed after the first pass, both because the main arm's
`cpsat:warmbest` campaign exposed them:

  * THE SOLVER SET IS A FLAG, NOT A CONSTANT (`--solvers`). `results/xpurt3net.json`
    records the list that actually ran, so the recorded `solvers` field stays an
    accurate description of the measurement rather than of the file's current
    source.
  * THE DEDUPE KEY WAS WRONG AND HID REAL COVERAGE. It hashed
    `json.load(sched).get("schedule")`, and `postprocessing.output_scheduled_json`
    emits no `"schedule"` key -- it emits `dot_file` / `dispatches` / `metadata`.
    Every schedule therefore hashed to sha256("null"), every solver deduped onto
    whichever ran first, and the note that "all three solvers emitted identical
    schedules" was an artefact of that constant, not a measurement. They do not:
    on 7 of the 9 shapes the three disagree. The key is now the same one
    `sched_algo_sweep10`'s Phase 4 uses -- op -> (combination, start, duration),
    i.e. everything the codegen reads -- so two solvers share board time only
    when they really produced the same table.

Solver routing. `scripts/run_xpurt_schedule.py` exposes ten solvers and
`cpsat:warmbest` is not one of them; it lives in the standalone solver tree that
`sched_algo_sweep10`'s `fpga/emit_schedule.py` reaches. Solvers that script can
express keep going through it (the schedules already measured came from there);
the rest go through `emit_schedule.py --prune-periodic`, which builds the
workload the same way. That the two paths agree is CHECKED, not assumed:
`xpurt3net.py verify` re-emits every `run_xpurt_schedule.py` schedule through
`emit_schedule.py` and requires a byte-identical dispatch hash.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
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
sys.path.insert(0, HERE)
import offsets  # noqa: E402
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

sys.path.insert(0, HERE)
import pin3net  # noqa: E402

BOARD = os.environ.get("BOARD", "root@10.44.120.201")
TOPLEVEL = os.path.join(REPO, "data", "toplevel", "rospin3net")
SPECS = os.path.join(SWEEP, "specs3net")
RUNTIMES = os.path.join(SWEEP, "runtimes3net")
RUNLOGS = os.path.join(SWEEP, "logs", "xpurt3net")
STATE = os.path.join(SWEEP, "results", "xpurt3net_state.json")

#: The set as measured: the first pass's three plus `cpsat:warmbest`, which is
#: what §4 scores the MAIN arm against and therefore the only opponent that
#: makes the two arms comparable. ORDER MATTERS -- the first solver to produce a
#: given table becomes the canonical run the others dedupe onto, so greedy stays
#: first and a re-run reuses the schedules already on record instead of
#: re-measuring them. `--solvers` overrides the list; whatever ran is what
#: `results/xpurt3net.json` records.
DEFAULT_SOLVERS = ["greedy", "heft_edf", "cpsat", "cpsat:warmbest"]

#: solvers `scripts/run_xpurt_schedule.py` can express. Anything else is
#: emitted through sched_algo_sweep10's `fpga/emit_schedule.py`, which reaches
#: the standalone solver tree (`metaheuristics.py`, `cpsat_scheduler.py`).
RXS_SOLVERS = {"milp", "milp_native", "greedy", "greedy_periodic",
               "greedy_reserved", "decomposed", "heft", "heft_edf", "pso",
               "sa", "cpsat", "auto"}

XSWEEP = os.path.abspath(os.path.join(
    SWEEP, "..", "qrb5165_sched_algo_sweep10_20260908-210226"))
EMITTER = os.path.join(XSWEEP, "fpga", "emit_schedule.py")
UNDECLARED = os.path.join(SWEEP, "results", "undeclared_3net.json")

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


def solver_tag(solver):
    """Filesystem rendering of a solver name. `:` is not a path character in
    the sweep10 tags either -- `cpsat:warmbest` is `cpsat-warmbest` there, and
    the same rendering is used here so the two arms' artefacts read alike."""
    return solver.replace(":", "-")


def sched_path(name, solver):
    tag = "" if solver in ("milp", "milp_native") else f"_{solver_tag(solver)}"
    return os.path.join(REPO, "schedules",
                        f"scheduled_networks_{name}{tag}_profiled.json")


def lock_wait_s(timeout=900):
    """How long acquiring the board lock takes right now.

    `flow_c.py run` takes `/tmp/qnn_board.lock` itself, inside
    `qnn_models/runtime/deploy_and_run.sh`, so this script must NOT wrap it in
    an outer flock -- that would make the inner one wait out its own 900 s.
    The lock is instead probed with the same flock on the same path
    immediately before each rep, so a slow run can be attributed to another
    tenant rather than to the workload. Verbatim `sched_algo_sweep10`'s
    `drive.py::lock_wait_s`, so the two arms' recorded waits mean the same
    thing.
    """
    t0 = time.time()
    r = subprocess.run(["timeout", "-s", "KILL", str(timeout + 30), "ssh", "-n",
                        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
                        f"flock -w {timeout} /tmp/qnn_board.lock -c 'true'"],
                       capture_output=True, text=True)
    return round(time.time() - t0, 2), r.returncode


def run_key(key):
    """Filesystem name for a state key. The state key keeps the solver's real
    name (`3net_mlp2__cpsat:warmbest`); the directory it writes to renders the
    `:` as `-`, the same way the sweep10 run tags do."""
    return key.replace(":", "-")


def sched_hash(path):
    """The dedupe key: op -> (combination, start, duration).

    Verbatim `fpga/emit_schedule.py::_sched_hash`, which is what Phase 4 of the
    XPU-RT sweep deduped on, so the two arms spend board time under the same
    rule. NOT the objective -- two different assignments can share a makespan,
    and collapsing those would silently drop coverage.

    The first version of this function hashed `doc.get("schedule")`, a key
    `output_scheduled_json` does not emit, so every schedule hashed to
    sha256("null") and every solver after the first was recorded as a duplicate
    of it. See the module docstring.
    """
    doc = json.load(open(path))
    disp = doc["dispatches"]
    items = (sorted(disp.items()) if isinstance(disp, dict)
             else sorted((str(i), d) for i, d in enumerate(disp)))
    h = hashlib.sha256()
    for k, d in items:
        h.update(k.encode())
        h.update(b"\0")
        h.update(json.dumps(d, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def solve_env():
    env = dict(os.environ)
    env.update(XPURT_CODE_ROOT=REPO, XPURT_DATA_ROOT=REPO)
    venv = os.path.join(REPO, ".cpsat-venv", "bin", "python")
    if os.path.exists(venv):
        env.setdefault("XPURT_CPSAT_PYTHON", venv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"
    return env


def solvers_of(args):
    if getattr(args, "solvers", None):
        return [x.strip() for x in args.solvers.split(",") if x.strip()]
    return list(DEFAULT_SOLVERS)


def cmd_emit(args):
    for s in pin3net.shapes():
        p, sp = emit(s)
        print(f'  {s["name"]:30s} -> {os.path.relpath(p, REPO)}')
    return 0


def cmd_undeclared(args):
    """Which (tile, lane) cells the solver can see but the bindings forbid.

    The main arm's §13 found CP-SAT proving an unbuildable OPTIMAL because
    `build_cost_model.py`'s exclusion lands at INGEST while the solver reads
    `gen/profile/` directly. That asymmetry is a property of the pipeline, not
    of those two cells, so it is checked here rather than assumed away: a cell
    is undeclared when the profile tree carries a number for
    `<net>/<tile>@<lane>` and `bindings/<net>.json` declares no context for that
    tile on that lane.

    Writes the same `dropped_undeclared_cells` shape `emit_schedule.py
    --mask-undeclared` reads, so the mask is fed by a measurement of this arm's
    own inputs rather than by the main arm's cost model, which covers a
    different network set entirely.
    """
    lanes, nets = set(), set()
    for sh in pin3net.shapes():
        doc = json.load(open(os.path.join(TOPLEVEL, f'networks_{sh["name"]}.json')))
        lanes |= {v.upper() for v in doc["hardware"]["profile_hw"].values()}
        nets |= set(sh["networks"])
    dropped, seen = [], []
    for net in sorted(nets):
        man = os.path.join(FLOWC, "bindings", f"{net}.json")
        if not os.path.exists(man):
            continue
        binds = json.load(open(man))["bindings"]
        tiles = [b["name"] for b in binds]
        decl = {b["name"]: set(b.get("backends") or {}) for b in binds}
        for hw in sorted(lanes):
            csvs = glob.glob(os.path.join(
                REPO, "gen", "profile", hw, "qrb5165_flowc", net,
                "*", "*", "topo_0", "results.csv"))
            if not csvs:
                continue
            ids = {int(r["dispatch_id"])
                   for r in csv.DictReader(open(csvs[0]))
                   if (r.get("dispatch_id") or "").strip().isdigit()}
            for k, tile in enumerate(tiles):
                if k not in ids:
                    continue
                cell = f"{net}/{tile}@{hw.lower()}"
                seen.append(cell)
                if hw.lower() not in decl[tile]:
                    dropped.append(cell)
    out = {"_comment":
           "Cells the 3net shapes' solver can reach in gen/profile/ but the "
           "binding manifests cannot execute. Fed to `emit_schedule.py "
           "--mask-undeclared`, which masks them to +inf BEFORE the search "
           "instead of letting `flowc/schedule.py::ingest` reject the answer "
           "after it. Derived from bindings/<net>.json and the profile tree by "
           "`xpurt3net.py undeclared`; the main arm's cost_model.json covers a "
           "different network set and says nothing about these.",
           "lanes": sorted(lanes), "networks": sorted(nets),
           "profiled_cells": sorted(seen),
           "dropped_undeclared_cells": sorted(dropped)}
    os.makedirs(os.path.dirname(UNDECLARED), exist_ok=True)
    with open(UNDECLARED, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {UNDECLARED}")
    print(f"  {len(seen)} profiled (tile, lane) cells, "
          f"{len(dropped)} undeclared:")
    for c in dropped:
        print(f"    {c}")
    return 0


def emit_via_sweep10(name, solver, out, mask, timeout=1800):
    """One schedule from sched_algo_sweep10's emitter.

    `--prune-periodic` because every 3net spec carries
    `scheduler.prune_periodic: true` and `run_xpurt_schedule.py` -- the path the
    other solvers took -- applies that trim itself. Without it the two paths
    would emit different tables for the same placement and nothing would ever
    dedupe.
    """
    cmd = [sys.executable, EMITTER, "--arm", "rospin3net",
           "--name", f"networks_{name}", "--solver", solver,
           "--out", out, "--meta-out", out[:-len(".json")] + ".meta.json",
           "--prune-periodic", "--cpsat-time", "60", "--cpsat-workers", "8"]
    if mask:
        cmd += ["--mask-undeclared", mask]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, env=solve_env())


def cmd_solve(args):
    mask = UNDECLARED if (args.mask_undeclared and os.path.exists(UNDECLARED)) \
        else None
    if args.mask_undeclared and not mask:
        print(f"  [warn] --mask-undeclared but {UNDECLARED} is absent; "
              f"run `xpurt3net.py undeclared` first")
    for s in pin3net.shapes():
        name = s["name"]
        rel = os.path.relpath(os.path.join(TOPLEVEL, f"networks_{name}.json"), REPO)
        for solver in solvers_of(args):
            out = sched_path(name, solver)
            if os.path.exists(out) and not args.force:
                print(f"  [skip] {name} {solver}")
                continue
            if solver in RXS_SOLVERS:
                cmd = [sys.executable, "scripts/run_xpurt_schedule.py",
                       "--networks-json", rel, "--solver", solver, "--profiled"]
                # CP-SAT runs out of process; the repo keeps an ortools venv at
                # .cpsat-venv, which is what the sched_algo_sweep10 port used
                # too. `run_xpurt_schedule.py` has no --mask-undeclared, so a
                # masked solve of one of these has to go the emitter route.
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                   timeout=900, env=solve_env())
            else:
                r = emit_via_sweep10(name, solver, out, mask)
            ok = os.path.exists(out)
            print(f'  [{"ok" if ok else "FAIL"}] {name:30s} {solver:14s}'
                  + (f"  {sched_hash(out)[:12]}" if ok
                     else " " + r.stdout[-300:] + r.stderr[-300:]))
    return 0


def cmd_verify(args):
    """Are the two solve paths the same experiment?

    `cpsat:warmbest` can only come from `emit_schedule.py`, while every schedule
    already measured here came from `run_xpurt_schedule.py`. Comparing a
    warmbest schedule from one builder against a greedy schedule from another
    would be a harness difference dressed up as a solver difference. So every
    schedule that CAN be produced both ways is produced both ways and the
    dispatch hashes have to match; a mismatch is a result, not a warning.

    Also re-emits each solver WITH and WITHOUT the undeclared mask: the mask is
    a constraint, and on this arm it must either change nothing or change
    something the bindings cannot build. Either way it is recorded.
    """
    mask = UNDECLARED if os.path.exists(UNDECLARED) else None
    rows, bad = [], 0
    tmp = os.path.join(SWEEP, "results", "verify3net_tmp")
    os.makedirs(tmp, exist_ok=True)
    for sh in pin3net.shapes():
        name = sh["name"]
        for solver in solvers_of(args):
            have = sched_path(name, solver)
            row = {"shape": name, "solver": solver,
                   "path": os.path.relpath(have, REPO) if os.path.exists(have) else None,
                   "hash": sched_hash(have) if os.path.exists(have) else None}
            for label, m in (("masked", mask), ("unmasked", None)):
                o = os.path.join(tmp, f"{name}__{solver_tag(solver)}__{label}.json")
                r = emit_via_sweep10(name, solver, o, m)
                row[f"emit_{label}"] = sched_hash(o) if os.path.exists(o) else None
                if not os.path.exists(o):
                    row[f"emit_{label}_err"] = (r.stdout[-200:] + r.stderr[-200:])
                mp = o[:-len(".json")] + ".meta.json"
                if os.path.exists(mp):
                    mm = json.load(open(mp))
                    row[f"emit_{label}_masked_cells"] = mm.get("masked_cells")
                    row[f"emit_{label}_objective"] = mm.get("objective")
            row["paths_agree"] = (row["hash"] is not None
                                  and row["hash"] == row["emit_masked"])
            row["mask_changes_schedule"] = (row["emit_masked"] != row["emit_unmasked"])
            if solver in RXS_SOLVERS and not row["paths_agree"]:
                bad += 1
            rows.append(row)
            print(f'  {name:28s} {solver:14s} '
                  f'disk={str(row["hash"])[:8]:>8s} '
                  f'emit={str(row["emit_masked"])[:8]:>8s} '
                  f'{"AGREE" if row["paths_agree"] else ("n/a" if solver not in RXS_SOLVERS else "MISMATCH")}'
                  f'  mask_moves={row["mask_changes_schedule"]}')
    p = os.path.join(SWEEP, "results", "verify3net.json")
    with open(p, "w") as f:
        json.dump({"_comment":
                   "Path-equivalence check for the 3net arm. Every solver that "
                   "`run_xpurt_schedule.py` can express is re-emitted through "
                   "sched_algo_sweep10's `fpga/emit_schedule.py "
                   "--prune-periodic`, and the dispatch hashes must match -- "
                   "otherwise `cpsat:warmbest`, which only the latter can "
                   "produce, would be a different experiment from the "
                   "schedules it is compared against. `mask_changes_schedule` "
                   "reports whether masking the undeclared cells moved the "
                   "answer.",
                   "solvers": solvers_of(args), "mismatches": bad,
                   "rows": rows}, f, indent=1)
    print(f"\nwrote {p}: {bad} mismatch(es) among the "
          f"run_xpurt_schedule.py-expressible solvers")
    return 1 if bad else 0


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
    solvers = solvers_of(args)
    for s in pin3net.shapes():
        name = s["name"]
        spec = os.path.join(SPECS, f"{name}.flowc.json")
        for solver in solvers:
            sp = sched_path(name, solver)
            if not os.path.exists(sp):
                continue
            key = f"{name}__{solver}"
            # Dedupe identical schedules across solvers, as the XPU-RT sweep
            # does, on the op -> (combination, start, duration) map -- see
            # `sched_hash`. sha256, not hash(): PYTHONHASHSEED makes str
            # hashing non-reproducible across processes, and the dedupe
            # decision has to be the same on a re-run.
            #
            # Solver ORDER decides which key becomes canonical, so it is the
            # caller's list order and it is stable: with greedy first, a shape
            # whose new solver reproduces greedy's table keeps greedy's
            # existing measurement instead of re-running it.
            h = sched_hash(sp)
            dup = seen_hash.get((name, h))
            if dup:
                st[key] = {"duplicate_of": dup, "sched_hash": h}
                print(f"  [dup] {key} == {dup}")
                continue
            seen_hash[(name, h)] = key
            if st.get(key, {}).get("ok") and not args.force:
                st[key]["sched_hash"] = h
                print(f"  [skip] {key}")
                continue
            out_dir = os.path.join(RUNTIMES, run_key(key))
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
            walls, nps, npcount, locks = [], [], None, []
            for rep in range(1, REPS + 1):
                ld = os.path.join(RUNLOGS, run_key(key), f"rep{rep}")
                os.makedirs(ld, exist_ok=True)
                lw, lrc = lock_wait_s()
                locks.append(lw)
                t0 = time.time()
                subprocess.run(
                    [sys.executable, "flow_c.py", "run", "--workload", spec,
                     "--out-dir", out_dir, "--tag", run_key(key), "--tuned",
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
                      f"({time.time() - t0:.0f}s, lock {lw:.2f}s rc={lrc})")
            st[key] = dict(ok=len(walls) == REPS, sched_hash=h, reps_ms=walls,
                           median_ms=round(statistics.median(walls), 4) if walls else None,
                           spread_ms=round(max(walls) - min(walls), 4) if walls else None,
                           np_reps_ms=nps,
                           np_median_ms=round(statistics.median(nps), 4) if nps else None,
                           np_spread_ms=round(max(nps) - min(nps), 4) if nps else None,
                           np_instances=npcount, lock_wait_s=locks,
                           predicted_ms=float(m.group(4)) if m else None)
            with open(STATE, "w") as f:
                json.dump(st, f, indent=1)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=1)
    return 0


def cmd_rescan(args):
    """Recompute the non-periodic makespan from run logs already on disk.

    Also derives the START-BARRIER CORRECTION on this arm, by the same rule
    the main arm uses (scripts/offsets.py): per rep, the end of the last
    aperiodic operation minus the start of that run's first dispatch. Needs no
    board time -- the traces are already on disk.
    """
    st = json.load(open(STATE))
    shapes = {s["name"]: s for s in pin3net.shapes()}
    for key, rec in st.items():
        if rec.get("duplicate_of") or not rec.get("ok"):
            continue
        name = key.rsplit("__", 1)[0]
        aperiodic = {n for n, v in shapes[name]["networks"].items()
                     if not v.get("period")}
        nps, npc, oreps = [], None, []
        for rep in range(1, REPS + 1):
            log = os.path.join(RUNLOGS, run_key(key), f"rep{rep}", "run.log")
            npm, c = np_makespan_from_log(log, aperiodic)
            if npm is not None:
                nps.append(npm)
                npc = c
            o = offsets.xpurt_rep(log, aperiodic)
            if o:
                oreps.append(o)
        rec["np_reps_ms"] = nps
        rec["np_median_ms"] = round(statistics.median(nps), 4) if nps else None
        rec["np_spread_ms"] = round(max(nps) - min(nps), 4) if nps else None
        rec["np_instances"] = npc
        c_med, c_spr, c_v = offsets.summarize(oreps, "corrected_ms")
        o_med, o_spr, o_v = offsets.summarize(oreps, "offset_ms")
        rec["np_corrected_median_ms"] = c_med
        rec["np_corrected_spread_ms"] = c_spr
        rec["np_corrected_reps_ms"] = c_v
        rec["offset_median_ms"] = o_med
        rec["offset_reps_ms"] = o_v
        rec["release_bound_reps"] = sum(1 for r in oreps if r["release_bound"])
        print(f'  {key:40s} np={rec["np_median_ms"]} '
              f'corrected={rec["np_corrected_median_ms"]} '
              f'offset={rec["offset_median_ms"]}  {npc}')
    with open(STATE, "w") as f:
        json.dump(st, f, indent=1)
    return 0


def resolve_dup(st, key, seen=()):
    """`duplicate_of` CHAINS. Following a single hop lands on another duplicate
    and loses the measurement -- the same trap `plot_gantt_compare.py` names on
    the main arm, where cpsat:warmbest -> best-of-fast -> cpsat-warm ->
    greedy_periodic and only the last is a real run."""
    rec = st.get(key)
    if rec is None or key in seen:
        return key, None
    if not rec.get("duplicate_of"):
        return key, rec
    return resolve_dup(st, rec["duplicate_of"], seen + (key,))


def cmd_collect(args):
    st = json.load(open(STATE))
    out = {}
    for key, rec in st.items():
        via = None
        if rec.get("duplicate_of"):
            src_key, src = resolve_dup(st, key)
            if src is None:
                continue
            rec, via = dict(src, measured_via=src_key), src_key
        name, solver = key.rsplit("__", 1)
        if not rec.get("ok"):
            continue
        e = out.setdefault(name, {"solvers": {}})
        e["solvers"][solver] = rec
    for name, e in out.items():
        s = e["solvers"]
        e["n_unique_schedules"] = len({v.get("sched_hash") for v in s.values()
                                       if v.get("sched_hash")})
        e["warmbest_makespan_ms"] = (s.get("cpsat:warmbest") or {}).get("median_ms")
        e["warmbest_np_ms"] = (s.get("cpsat:warmbest") or {}).get("np_median_ms")
        e["warmbest_measured_via"] = (s.get("cpsat:warmbest") or {}).get("measured_via")
        e["best_makespan_ms"] = min(v["median_ms"] for v in s.values())
        e["best_solver"] = min(s, key=lambda k: s[k]["median_ms"])
        e["greedy_makespan_ms"] = s.get("greedy", {}).get("median_ms")
        nps = {k: v["np_median_ms"] for k, v in s.items() if v.get("np_median_ms")}
        if nps:
            e["best_np_ms"] = min(nps.values())
            e["best_np_solver"] = min(nps, key=nps.get)
        e["warmbest_np_corrected_ms"] = (
            s.get("cpsat:warmbest") or {}).get("np_corrected_median_ms")
        cnp = {k: v["np_corrected_median_ms"] for k, v in s.items()
               if v.get("np_corrected_median_ms")}
        if cnp:
            e["best_np_corrected_ms"] = min(cnp.values())
            e["best_np_corrected_solver"] = min(cnp, key=cnp.get)
        e["np_instances"] = next((v.get("np_instances") for v in s.values()
                                  if v.get("np_instances")), None)
        e["n_solvers_measured"] = len(s)
    p = os.path.join(SWEEP, "results", "xpurt3net.json")
    with open(p, "w") as f:
        json.dump({"_comment":
                   "Measured XPU-RT makespans for the 3net shapes on the same "
                   "three lanes the pinning baseline gets. Medians of 3 reps "
                   "with spreads; `flow_c.py run --tuned`, the same mode the "
                   "sched_algo_sweep10 port used. `solvers` is the list that "
                   "ACTUALLY ran, not a constant in the driver. A solver whose "
                   "schedule is byte-identical to one already measured for that "
                   "shape carries `measured_via` and consumed no board time; "
                   "the dedupe key is the op -> (combination, start, duration) "
                   "map, the same one sched_algo_sweep10 Phase 4 uses.",
                   "board": BOARD, "reps": REPS, "solvers": solvers_of(args),
                   "shapes": out}, f, indent=1)
    print(f"wrote {p}: {len(out)} shapes")
    for n, e in sorted(out.items()):
        wb = e.get("warmbest_np_ms")
        print(f'  {n:30s} best_np={(e.get("best_np_ms") or 0):8.3f} '
              f'({e.get("best_np_solver")}) warmbest_np='
              f'{("%8.3f" % wb) if wb else "       -"} '
              f'n_solvers={e["n_solvers_measured"]} '
              f'n_unique={e["n_unique_schedules"]}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("emit", cmd_emit), ("solve", cmd_solve),
                     ("run", cmd_run), ("rescan", cmd_rescan),
                     ("collect", cmd_collect), ("undeclared", cmd_undeclared),
                     ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--force", action="store_true")
        if name != "undeclared":
            s.add_argument(
                "--solvers", default=None,
                help="comma-separated solver list; default "
                     + ",".join(DEFAULT_SOLVERS) + ". Recorded verbatim in "
                     "results/xpurt3net.json so that file describes the run.")
        if name == "solve":
            s.add_argument("--mask-undeclared", action="store_true",
                           help="mask results/undeclared_3net.json's cells to "
                                "+inf BEFORE the search (emit_schedule.py "
                                "route only)")
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
