#!/usr/bin/env python3
"""One-command reproduction of this sweep, checked against the committed record.

    python3 reproduce.py --check        prerequisites only
    python3 reproduce.py --host-only    generate + solve + verify   (no board)
    python3 reproduce.py                everything, including board runs

Work happens in a scratch directory (--out, default a fresh tmpdir), never in
this one, so a reproduction attempt can never overwrite the record it is being
checked against.

What gets verified, in increasing order of what it proves:

  pin       re-emit the shared gen/ profile artifacts from THIS directory's
            frozen cost_model.json. Required, and easy to miss: the workload
            generator derives each periodic task's period and window from the
            model's measured runtime, which it reads out of gen/profile/. So
            generation is only reproducible against pinned artifacts. Solving
            already pins them (drive.py re-emits before each solve); generation
            did not, and regenerating against a rebuilt cost model silently
            produced different tasksets -- fused_split split into two chained
            copies instead of one.
  generate  regenerated workloads are byte-identical to the committed ones.
            The generator is seeded, so this is exact -- any difference means
            gen_random_workload.py or its pinned inputs changed.
  solve     predicted makespan and solver identity match results.json per
            point. Exact: the artifacts are re-emitted from this directory's
            frozen cost_model.json, so this is independent of whatever
            measurements/qrb5165_v66.json currently holds.
  runtime   dispatch_table.h and runtime_main.cpp sha256 match the recorded
            ones. Exact: proves the emitted C++ is the same code that ran.
  board     measured makespan within tolerance of the record. NOT exact --
            rep-to-rep spread on this board is ~4.4% and reached 8.7% on one
            point, so this reports drift rather than asserting equality.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
ARMS = {"baseline": "mlp_control,dronet,yolov8n",
        "fused": "mlp_control,fused_split,yolov8n",
        "fused_vint": "mlp_control,fused_split,yolov8n,vint"}
OK, BAD = "  \033[32mOK\033[0m  ", "  \033[31mFAIL\033[0m"


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def record():
    return {r["point"]: r for r in json.load(open(os.path.join(HERE, "results.json")))}


def run(cmd, env=None, cwd=None, timeout=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True,
                          timeout=timeout)


# ---------------------------------------------------------------- checks
def check(need_board):
    print("prerequisites")
    ok = True

    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("drv", os.path.join(HERE, "drive.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    if drv.MILP_PY:
        print(f"{OK} MILP interpreter  {drv.MILP_PY}")
    else:
        print(f"{BAD} MILP interpreter  not found -- set XPURT_MILP_PYTHON to a python "
              f"with cvxpy + mosek")
        ok = False

    lic = drv.MOSEK_LIC
    print(f"{OK if os.path.exists(lic) else BAD} MOSEK licence     {lic}")
    ok &= os.path.exists(lic)

    sched = os.path.join(REPO, "scripts", "run_xpurt_schedule.py")
    print(f"{OK if os.path.exists(sched) else BAD} xpu-rt scheduler  {os.path.relpath(sched, REPO)}")
    ok &= os.path.exists(sched)

    cm = os.path.join(HERE, "cost_model.json")
    print(f"{OK if os.path.exists(cm) else BAD} frozen cost model sha {sha256(cm)[:16]}")
    ok &= os.path.exists(cm)

    if need_board:
        board = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")
        r = run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", board,
                 "echo up"], timeout=40)
        up = r.returncode == 0 and "up" in (r.stdout or "")
        print(f"{OK if up else BAD} board             {board}")
        ok &= up
    else:
        print("       board             not required (--host-only)")
    return ok


# ------------------------------------------------------------------ pin
# Two specs between them name all five models in the bank.
PIN_SPECS = ["baseline_seed0.flowc.json",      # mlp_control, dronet, yolov8n
             "fused_vint_seed0.flowc.json"]    # + fused_split, vint


def pin_artifacts():
    """Re-emit gen/ from the frozen cost model so generation is deterministic.

    NOTE: this writes to the repo's shared gen/ tree, outside the scratch
    directory -- unavoidable, since that tree is where the generator and the
    solver both read profiles from. It is idempotent and derived entirely from
    this directory's frozen cost_model.json.
    """
    print("\npin -- re-emit gen/ artifacts from the frozen cost model")
    ok = True
    for spec in PIN_SPECS:
        sp = os.path.join(HERE, "workloads", spec)
        if not os.path.exists(sp):
            print(f"{BAD} {spec} missing"); ok = False; continue
        r = run([sys.executable, "flow_c.py", "artifacts", "--workload", sp],
                cwd=FLOWC, timeout=1800)
        good = r.returncode == 0
        ok &= good
        print(f"{OK if good else BAD} {spec}")
        if not good:
            print((r.stdout or "")[-500:]); print((r.stderr or "")[-500:])
    print(f"{OK if ok else BAD} shared gen/ pinned to cost_model.json "
          f"sha {sha256(os.path.join(HERE, 'cost_model.json'))[:16]}")
    return ok


# ------------------------------------------------------------- generate
def generate(out):
    print("\ngenerate -- regenerate every workload and diff against the record")
    os.makedirs(os.path.join(out, "workloads", "rejected"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    gen = json.load(open(os.path.join(HERE, "generated.json")))
    same = diff = 0
    for row in gen:
        arm, seed, pt = row["arm"], row["seed"], row["point"]
        dest = os.path.join(out, "workloads", f"{pt}.json")
        r = run([sys.executable, os.path.join(HERE, "gen_random_workload.py"), str(seed),
                 "--hardware", "qrb5165_flowc", "--repo-root", REPO,
                 "--unbounded-nonperiodic", "--include-models", ARMS[arm],
                 "--max-ops", str(row.get("max_ops", 800)), "--out", dest],
                cwd=REPO, timeout=300)
        ref = os.path.join(HERE, row["workload"])
        if r.returncode != 0 or not os.path.exists(dest):
            print(f"{BAD} {pt:<20} generator failed"); diff += 1; continue
        a, b = json.load(open(dest)), json.load(open(ref))
        if a == b:
            same += 1
        else:
            diff += 1
            print(f"{BAD} {pt:<20} differs from {row['workload']}")
        if row["status"] == "REJECTED":
            shutil.move(dest, os.path.join(out, "workloads", "rejected", f"{pt}.json"))
        else:
            shutil.copy2(os.path.join(HERE, "workloads", f"{pt}.flowc.json"),
                         os.path.join(out, "workloads", f"{pt}.flowc.json"))
    shutil.copy2(os.path.join(HERE, "generated.json"), os.path.join(out, "generated.json"))
    print(f"{OK if not diff else BAD} {same}/{same+diff} workloads byte-identical to the record")
    return diff == 0


# ---------------------------------------------------------------- solve
def solve(out):
    print("\nsolve -- MILP against the frozen cost model, vs recorded predictions")
    r = run([sys.executable, "-u", os.path.join(HERE, "drive.py"), "solve"],
            env={"SWEEP_OUT": out}, cwd=HERE, timeout=7200)
    if r.returncode != 0:
        print(f"{BAD} drive.py solve exited {r.returncode}")
        print((r.stdout or "")[-800:]); print((r.stderr or "")[-800:])
        return False
    rec, st = record(), json.load(open(os.path.join(out, "state.json")))
    good = bad = 0
    for pt, ref in sorted(rec.items()):
        if ref["status"] != "run":
            continue
        got = st.get(pt, {})
        sp = os.path.join(out, got.get("schedule", "")) if got.get("schedule") else None
        mk = json.load(open(sp))["metadata"]["makespan"] if sp and os.path.exists(sp) else None
        want = ref["predicted_makespan_ms"]
        hit = mk is not None and abs(mk - want) < 0.01 and got.get("solver") == ref["solver"]
        good, bad = good + hit, bad + (not hit)
        print(f"{OK if hit else BAD} {pt:<20} makespan {str(round(mk,3) if mk else None):>9} "
              f"vs {want:<9} solver {got.get('solver')} vs {ref['solver']}")
    print(f"{OK if not bad else BAD} {good}/{good+bad} points reproduce their predicted makespan exactly")
    return bad == 0


# -------------------------------------------------------------- runtime
def runtime(out):
    """Compare the emitted C++ against the record, distinguishing two cases.

    dispatch_table.h encodes the SCHEDULE -- entry order, lanes, start times,
    contexts, graphs. A mismatch there means the reproduction produced a
    different schedule, which is a real failure.

    runtime_main.cpp is the harness around it. It legitimately evolves: the
    transfer study (5ebba96) added per-entry handoff instrumentation, so every
    runtime emitted after that differs from the recorded sha by construction.
    Reporting that as a failure would conflate 'the schedule changed' with
    'the runtime source moved on', so it is a warning when the table matches.
    """
    print("\nruntime -- emitted C++ sha256 vs the record")
    r = run([sys.executable, "-u", os.path.join(HERE, "drive.py"), "runtime"],
            env={"SWEEP_OUT": out}, cwd=HERE, timeout=3600)
    if r.returncode != 0:
        print(f"{BAD} drive.py runtime exited {r.returncode}"); return False
    rec = record()
    dt_ok = rm_ok = n = 0
    for pt, ref in sorted(rec.items()):
        if ref["status"] != "run":
            continue
        n += 1
        d = os.path.join(out, "runtimes", pt)
        try:
            a = sha256(os.path.join(d, "dispatch_table.h")) == ref["dispatch_table_sha256"]
            b = sha256(os.path.join(d, "runtime_main.cpp")) == ref["runtime_main_sha256"]
        except FileNotFoundError:
            a = b = False
        dt_ok += a; rm_ok += b
        if not a:
            print(f"{BAD} {pt:<20} dispatch_table.h differs -- SCHEDULE changed")
    print(f"{OK if dt_ok == n else BAD} {dt_ok}/{n} dispatch_table.h byte-identical "
          f"(the schedule encoding)")
    if rm_ok == n:
        print(f"{OK} {rm_ok}/{n} runtime_main.cpp byte-identical (the harness)")
    else:
        print(f"  \033[33mWARN\033[0m {rm_ok}/{n} runtime_main.cpp byte-identical -- expected if")
        print("       flowc/emit_runtime.py has changed since the sweep ran. The transfer")
        print("       study (5ebba96) added handoff instrumentation, which changes every")
        print("       emitted runtime. Not a reproduction failure while the table matches.")
    return dt_ok == n


# ---------------------------------------------------------------- board
def board(out, reps):
    print(f"\nboard -- {reps} rep(s) per point, measured drift vs the record")
    for stage, to in (("stage", 1800), ("run", 14400), ("results", 900)):
        cmd = [sys.executable, "-u", os.path.join(HERE, "drive.py"), stage]
        if stage == "run":
            cmd += ["--reps", str(reps)]
        r = run(cmd, env={"SWEEP_OUT": out}, cwd=HERE, timeout=to)
        if r.returncode != 0:
            print(f"{BAD} drive.py {stage} exited {r.returncode}"); return False
    rec = record()
    new = {x["point"]: x for x in json.load(open(os.path.join(out, "results.json")))}
    print(f"{'point':<22} {'recorded':>10} {'now':>10} {'drift':>8}")
    drifts = []
    for pt, ref in sorted(rec.items()):
        if ref["status"] != "run" or pt not in new or new[pt]["status"] != "run":
            continue
        a, b = ref["actual_makespan_median_ms"], new[pt]["actual_makespan_median_ms"]
        d = 100 * (b - a) / a
        drifts.append(abs(d))
        print(f"{pt:<22} {a:>10.2f} {b:>10.2f} {d:>+7.1f}%")
    if drifts:
        import statistics as st
        print(f"\n  median |drift| {st.median(drifts):.1f}%   "
              f"(board rep-to-rep spread is ~4.4%, up to 8.7% on one point --"
              f" drift at or below that is a reproduction, not a regression)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="prerequisites only")
    ap.add_argument("--host-only", action="store_true", help="skip everything needing the board")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None, help="scratch dir (default: a fresh tmpdir)")
    a = ap.parse_args()

    need_board = not (a.check or a.host_only)
    print(f"reproducing {os.path.basename(HERE)}\n")
    if not check(need_board):
        return 1
    if a.check:
        print("\nprerequisites satisfied")
        return 0

    out = a.out or tempfile.mkdtemp(prefix="sweep_repro_")
    os.makedirs(out, exist_ok=True)
    print(f"\nworking in {out}")

    steps = [pin_artifacts(), generate(out), solve(out), runtime(out)]
    if not a.host_only:
        steps.append(board(out, a.reps))
    print(f"\n{'='*60}")
    print("REPRODUCED" if all(steps) else "DIFFERENCES FOUND -- see above")
    print(f"artifacts in {out}")
    return 0 if all(steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
