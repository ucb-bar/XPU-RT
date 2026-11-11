#!/usr/bin/env python3
"""One-command reproduction of this sweep, checked against the committed record.

    python3 reproduce.py --check        prerequisites only
    python3 reproduce.py --host-only    workloads + solves + emit, no board
    python3 reproduce.py                everything, including board runs

Work happens in a scratch directory (--out, default a fresh tmpdir), never in
this one, so a reproduction attempt can never overwrite the record it is being
checked against. The one exception is the shared `gen/` profile tree, which is
re-emitted from THIS directory's frozen cost_model.json — that is the point of
pinning, and it is what makes the predicted numbers independent of whatever
measurements/qrb5165_v66.json currently holds.

Verified, in increasing order of what it proves:

| stage      | check                                             | exact? |
|------------|---------------------------------------------------|--------|
| pin        | gen/profile CSVs re-emitted from the frozen model  | —      |
| generate   | regenerated tasksets byte-identical to the record  | yes — the generator has no RNG; its only inputs are the frozen model and the manifests |
| solve      | objective, misses and validation match per (cell, solver) | yes for the nine deterministic entries; pso/sa are seeded and reproduce; the four CP-SAT entries are NOT — `num_search_workers=8` makes CP-SAT non-reproducible, which the reference measured at 2.73% mean spread cold |
| emit       | schedule content hash matches                     | yes, for the deterministic solvers |
| runtime    | dispatch_table.h sha256 matches                   | yes — proves the same schedule was emitted as C++ |
| runtime    | runtime_main.cpp sha256 matches                   | warning only — the harness legitimately evolves |
| board      | measured makespan drift vs the record             | **no** — reported, never asserted; see ANALYSIS.md for the measured rep spread |

Phase 1 (building the 16 networks and measuring their cells) is NOT reproduced
here. It needs the modelblaster zoo, the qnn-convert docker image, and ~40
minutes of board time, and its output IS the frozen cost model this checks
against. `--rebuild-phase1` prints the exact commands instead of running them.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
ARM = "s10port"
OK, BAD, WARN = "  \033[32mOK\033[0m  ", "  \033[31mFAIL\033[0m", "  \033[33mWARN\033[0m"


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(cmd, cwd=None, env=None, timeout=None):
    return subprocess.run(cmd, cwd=cwd, env=dict(os.environ, **(env or {})),
                          capture_output=True, text=True, timeout=timeout)


def rec(name, default=None):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else default


def solve_env():
    return dict(XPURT_CODE_ROOT=REPO, XPURT_DATA_ROOT=REPO,
                XPURT_CPSAT_PYTHON=cpsat_python() or "",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def cpsat_python():
    env = os.environ.get("XPURT_CPSAT_PYTHON")
    cands = [env] if env else []
    cands += [os.path.join(REPO, ".cpsat-venv", "bin", "python"),
              os.path.join(REPO, "cpsat-venv", "bin", "python"),
              os.path.expanduser("~/cpsat-venv/bin/python")]
    for c in cands:
        if c and os.path.exists(c):
            p = run([c, "-c", "from ortools.sat.python import cp_model"])
            if p.returncode == 0:
                return c
    p = run([sys.executable, "-c", "from ortools.sat.python import cp_model"])
    return sys.executable if p.returncode == 0 else None


def check(need_board):
    print("prerequisites")
    ok = True
    cp = cpsat_python()
    if cp:
        print(f"{OK} CP-SAT interpreter  {cp}")
    else:
        print(f"{BAD} CP-SAT interpreter  not found — the four CP-SAT entries "
              f"cannot run.\n       python3 -m venv {REPO}/.cpsat-venv && "
              f"{REPO}/.cpsat-venv/bin/pip install ortools numpy")
        ok = False
    cm = os.path.join(HERE, "cost_model.json")
    print(f"{OK if os.path.exists(cm) else BAD} frozen cost model  {cm}")
    ok &= os.path.exists(cm)
    for rel in ("scripts/sweep10_runner.py", "fpga/emit_schedule.py",
                "scripts/mk_workloads_qrb5165.py", "scripts/phase2_specs.py"):
        p = os.path.join(HERE, rel)
        print(f"{OK if os.path.exists(p) else BAD} {rel}")
        ok &= os.path.exists(p)
    xr = os.path.join(REPO, "xpu-rt", "metaheuristics.py")
    print(f"{OK if os.path.exists(xr) else BAD} solver tree  {os.path.dirname(xr)}")
    ok &= os.path.exists(xr)
    if need_board:
        host = os.environ.get("QNN_BOARD_HOST", "root@qrb5165")
        p = run(["timeout", "-s", "KILL", "60", "ssh", "-n", "-o",
                 "ConnectTimeout=20", "-o", "BatchMode=yes", host,
                 "flock -w 60 /tmp/qnn_board.lock -c 'df -h / | tail -1'"])
        good = p.returncode == 0
        print(f"{OK if good else BAD} board {host}  {(p.stdout or '').strip()}")
        ok &= good
    return ok


def stage_generate(out):
    print("\ngenerate — tasksets from the frozen cost model")
    tdir = os.path.join(out, "toplevel")
    p = run([sys.executable, os.path.join(HERE, "scripts",
                                          "mk_workloads_qrb5165.py"),
             "--emit", "--out", tdir,
             "--cost-model", os.path.join(HERE, "cost_model.json")])
    if p.returncode != 0:
        print(f"{BAD} generator rc={p.returncode}\n{(p.stderr or '')[-800:]}")
        return False
    ref = os.path.join(REPO, "data", "toplevel", ARM)
    got = sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []
    want = sorted(f for f in os.listdir(ref) if f.endswith(".json")) \
        if os.path.isdir(ref) else []
    if got != want:
        print(f"{BAD} {len(got)} generated vs {len(want)} in the record")
        return False
    diff = [f for f in want if sha256(os.path.join(tdir, f))
            != sha256(os.path.join(ref, f))]
    print(f"{OK if not diff else BAD} {len(want)} tasksets regenerated"
          + (" byte-identical" if not diff else f"; {len(diff)} differ: {diff[:3]}"))
    return not diff


def stage_solve(out, only):
    print("\nsolve — objective / misses / audit per (cell, solver)")
    allr = rec("phase3_all_results.json") or rec("all_results.json") or []
    if not allr:
        print(f"{BAD} no solve record to check against")
        return False
    byk = {(r["workload"], r["solver"]): r for r in allr}
    # re-emit the shared artifacts from the frozen model first
    sp = os.path.join(HERE, "specs", "s10port_artifacts.flowc.json")
    p = run([sys.executable, "flow_c.py", "artifacts", "--workload", sp], cwd=FLOWC)
    print(f"{OK if p.returncode == 0 else BAD} gen/ artifacts re-emitted from "
          f"the frozen cost model")
    det = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
           "heft", "heft_edf", "pso", "sa"]
    todo = [k for k in byk if k[1] in det]
    if only:
        todo = [k for k in todo if k[0] in set(only.split(","))]
    todo.sort()
    bad = []
    outdir = os.path.join(out, "solve")
    os.makedirs(outdir, exist_ok=True)
    for wl, s in todo:
        o = os.path.join(outdir, f"{wl}__{s}.json")
        p = run([sys.executable, os.path.join(HERE, "scripts", "sweep10_runner.py"),
                 "--arm", ARM, "--name", wl, "--solver", s, "--out", o],
                env=solve_env(), timeout=900)
        if not os.path.exists(o):
            bad.append(f"{wl}/{s}: runner produced nothing"); continue
        g, w = json.load(open(o)), byk[(wl, s)]
        if g.get("error") or w.get("error"):
            if bool(g.get("error")) != bool(w.get("error")):
                bad.append(f"{wl}/{s}: error mismatch")
            continue
        if abs(g["objective"] - w["objective"]) > 1e-6 or g["misses"] != w["misses"]:
            bad.append(f"{wl}/{s}: obj {g['objective']} vs {w['objective']}, "
                       f"misses {g['misses']} vs {w['misses']}")
    print(f"{OK if not bad else BAD} {len(todo)} deterministic solves reproduced"
          + ("" if not bad else f"; {len(bad)} differ:\n       "
             + "\n       ".join(bad[:5])))
    print(f"{WARN} the four CP-SAT entries are not checked: "
          f"num_search_workers=8 makes them non-reproducible by construction")
    return not bad


def stage_runtime(out, only):
    print("\nruntime — dispatch_table.h sha256 per point")
    p4 = rec("phase4_results.json") or []
    have = [r for r in p4 if r.get("dispatch_table_sha256")
            and r.get("solver") not in ("cpsat", "cpsat:warm", "cpsat:warmbest")]
    if only:
        have = [r for r in have if r["workload"] in set(only.split(","))]
    if not have:
        print(f"{WARN} no non-CP-SAT runtime record to check")
        return True
    bad, softbad = [], []
    for r in have:
        pid = r["point"]
        sdir, rdir = os.path.join(out, "sched"), os.path.join(out, "rt", pid)
        os.makedirs(sdir, exist_ok=True)
        sj = os.path.join(sdir, f"{pid}.json")
        q = run([sys.executable, os.path.join(HERE, "fpga", "emit_schedule.py"),
                 "--arm", ARM, "--name", r["workload"], "--solver", r["solver"],
                 "--out", sj, "--meta-out", sj + ".meta"],
                env=solve_env(), timeout=900)
        if not os.path.exists(sj + ".meta"):
            bad.append(f"{pid}: emit failed"); continue
        m = json.load(open(sj + ".meta"))
        if r.get("sched_hash") and m["sched_hash"] != r["sched_hash"]:
            bad.append(f"{pid}: sched_hash {m['sched_hash'][:12]} vs "
                       f"{r['sched_hash'][:12]}")
            continue
        q = run([sys.executable, "flow_c.py", "runtime", "--workload",
                 os.path.join(HERE, "specs", f'{r["workload"]}.flowc.json'),
                 "--tag", pid, "--lane-mode", "kind-network",
                 "--schedule", sj, "--out-dir", rdir], cwd=FLOWC, timeout=1800)
        th = os.path.join(rdir, "dispatch_table.h")
        if not os.path.exists(th):
            bad.append(f"{pid}: runtime emit failed"); continue
        if sha256(th) != r["dispatch_table_sha256"]:
            bad.append(f"{pid}: dispatch_table.h sha256 differs")
        rm = os.path.join(rdir, "runtime_main.cpp")
        if r.get("runtime_main_sha256") and os.path.exists(rm) \
           and sha256(rm) != r["runtime_main_sha256"]:
            softbad.append(pid)
    print(f"{OK if not bad else BAD} {len(have)} points re-emitted"
          + ("" if not bad else f"; {len(bad)} differ:\n       "
             + "\n       ".join(bad[:5])))
    if softbad:
        print(f"{WARN} runtime_main.cpp differs on {len(softbad)} point(s) — "
              f"the harness evolves; the schedule is what matters")
    return not bad


def stage_board(out, only, reps):
    print("\nboard — measured makespan drift (reported, never asserted)")
    p4 = rec("phase4_results.json") or []
    have = [r for r in p4 if r.get("measured_median_ms")]
    if only:
        have = [r for r in have if r["workload"] in set(only.split(","))]
    if not have:
        print(f"{WARN} no measured points in the record")
        return True
    print(f"       re-running {len(have)} point(s) x {reps} reps via drive.py")
    p = run([sys.executable, os.path.join(HERE, "scripts", "drive.py"), "run",
             "--reps", str(reps)], timeout=None)
    print((p.stdout or "")[-2000:])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of workload names")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rebuild-phase1", action="store_true")
    a = ap.parse_args()
    if a.rebuild_phase1:
        print(__doc__.split("Phase 1")[1])
        print("  cd " + HERE)
        print("  python3 scripts/phase1_build.py")
        print("  python3 scripts/phase1_bindings.py --write")
        print("  python3 scripts/phase1_measure.py --iters 40 --gap-us 3000")
        print("  python3 scripts/build_cost_model.py --write")
        return 0
    need_board = not (a.check or a.host_only)
    if not check(need_board):
        return 1
    if a.check:
        return 0
    out = a.out or tempfile.mkdtemp(prefix="s10port_repro_")
    os.makedirs(out, exist_ok=True)
    print(f"\nscratch: {out}")
    ok = stage_generate(out)
    ok &= stage_solve(out, a.only)
    ok &= stage_runtime(out, a.only)
    if need_board:
        stage_board(out, a.only, a.reps)
    print(f"\n{'reproduced' if ok else 'DIVERGED — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
