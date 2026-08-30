#!/usr/bin/env python3
"""Sweep driver: everything after workload generation.

Phase 1b  solve      artifacts (per point) + xpu-rt solve, MILP with a 300 s
                     limit falling back to greedy_periodic. Host only.
Phase 2   runtime    emit dispatch_table.h + runtime_main.cpp per point,
                     sha256 them, and run predicate 7 (no capability-excluded
                     cell in the chosen placement). Host only.
Phase 3   stage      link the context binaries on the board, then check
                     predicate 6 (every (tile, backend) the schedule selects
                     has a context staged).
Phase 3   run        3 reps per point, serially, `--tuned`.
          results    assemble results.json.

Every board interaction this script makes itself goes behind
  ssh root@... "flock -w 900 /tmp/qnn_board.lock -c '...'"
wrapped in `timeout -s KILL`. `flow_c.py run` takes the same lock itself for
the run step (deploy_and_run.sh), so it is invoked without an outer lock —
taking one here would make its inner flock wait out its own 900 s.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))          # qnn_models/flow_c
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))          # XPU-RT
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")
MILP_PY = ("/tmp/claude-1172/-scratch2-dima-misc-sw-XPU-RT/"
           "3882980f-b60e-47b9-be3f-7fe85ec2bebb/scratchpad/milpenv/bin/python")
MOSEK_LIC = os.path.expanduser("~/mosek/mosek.lic")
MILP_TIME_LIMIT = 300

sys.path.insert(0, HERE)
from sweep_unbounded_nonperiodic import BINDINGS, model_of        # noqa: E402


def sh(cmd, cwd=None, env=None, timeout=None, log=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=timeout)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            f.write("+ " + " ".join(cmd) + f"\n(cwd={cwd})\n\n"
                    + p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p, time.time() - t0


def board(script: str, timeout=600):
    """One board interaction, serialised behind the shared board lock."""
    quoted = script.replace("'", "'\\''")
    cmd = ["timeout", "-s", "KILL", str(timeout + 60), "ssh",
           "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
           f"flock -w 900 /tmp/qnn_board.lock -c '{quoted}'"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)


def lock_wait_s(timeout=900):
    """How long it takes to acquire the board lock right now, in seconds.

    An honest proxy for the wait the run that follows will see: the same
    flock on the same path, taken immediately before it.
    """
    t0 = time.time()
    r = subprocess.run(["timeout", "-s", "KILL", str(timeout + 30), "ssh",
                        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
                        f"flock -w {timeout} /tmp/qnn_board.lock -c 'true'"],
                       capture_output=True, text=True)
    return round(time.time() - t0, 2), r.returncode


def points(only=None):
    rows = json.load(open(os.path.join(HERE, "generated.json")))
    out = [r for r in rows if r["status"] == "ok"]
    if only:
        want = set(only.split(","))
        out = [r for r in out if r["point"] in want]
    return out


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_state():
    p = os.path.join(HERE, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(st):
    with open(os.path.join(HERE, "state.json"), "w") as f:
        json.dump(st, f, indent=1)


# --------------------------------------------------------------------------
# Phase 1b — artifacts + solve
# --------------------------------------------------------------------------

def cmd_solve(args):
    st = load_state()
    cost = json.load(open(os.path.join(HERE, "cost_model.json")))
    for r in points(args.only):
        pt = r["point"]
        spec = os.path.join(HERE, "workloads", f"{pt}.flowc.json")
        rec = st.setdefault(pt, {})
        if rec.get("status") == "solved" and not args.force:
            print(f"  {pt:<22} already solved by {rec.get('solver')} "
                  f"({rec.get('solve_s')}s) — skipping")
            continue
        # (a) artifacts — dispatch graphs + profile CSVs, from the frozen
        #     cost model. Re-emitted immediately before each solve so the
        #     CSVs the solver reads are the ones this sweep froze, even if
        #     another tenant re-emitted the shared tree in between.
        p, dt = sh([sys.executable, "flow_c.py", "artifacts", "--workload", spec],
                   cwd=FLOWC, log=os.path.join(HERE, "logs", f"artifacts_{pt}.log"))
        rec["artifacts_rc"] = p.returncode
        rec["artifacts_s"] = round(dt, 1)
        if p.returncode != 0:
            rec["status"] = "artifacts_failed"
            print(f"  {pt:<22} ARTIFACTS FAILED rc={p.returncode}")
            save_state(st); continue

        # (b) solve. MILP first (SETUP: optimal at 66-86 ops), greedy_periodic
        #     on non-zero rc or on the 300 s limit.
        wl = os.path.join(HERE, "workloads", f"{pt}.json")
        env = dict(os.environ, MOSEKLM_LICENSE_FILE=MOSEK_LIC)
        solver, sched_json, sres, dt = None, None, None, 0.0
        if os.path.exists(MILP_PY) and not args.greedy_only:
            try:
                sres, dt = sh([MILP_PY, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
                               "--networks-json", wl, "--solver", "milp",
                               "--time-limit", str(MILP_TIME_LIMIT), "--profiled"],
                              cwd=HERE, env=env, timeout=MILP_TIME_LIMIT + 120,
                              log=os.path.join(HERE, "logs", f"sched_{pt}_milp.log"))
            except subprocess.TimeoutExpired:
                sres, dt = None, MILP_TIME_LIMIT + 120
                rec["milp_timed_out_s"] = MILP_TIME_LIMIT + 120
            cand = os.path.join(HERE, "schedules", f"scheduled_{pt}_profiled.json")
            if sres is not None and sres.returncode == 0 and os.path.exists(cand):
                solver, sched_json = "milp", cand
        if solver is None:
            sres, dt = sh([sys.executable, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
                           "--networks-json", wl, "--solver", "greedy_periodic",
                           "--profiled"],
                          cwd=HERE, log=os.path.join(HERE, "logs", f"sched_{pt}_greedy.log"))
            cand = os.path.join(HERE, "schedules",
                                f"scheduled_{pt}_greedy_periodic_profiled.json")
            if sres.returncode == 0 and os.path.exists(cand):
                solver, sched_json = "greedy_periodic", cand
        rec["solver"] = solver
        rec["solve_s"] = round(dt, 1)
        rec["sched_rc"] = None if sres is None else sres.returncode
        if solver is None:
            rec["status"] = "solve_failed"
            print(f"  {pt:<22} SOLVE FAILED")
            save_state(st); continue
        rec["schedule"] = os.path.relpath(sched_json, HERE)
        out = (sres.stdout or "")
        mk = [l.strip() for l in out.splitlines() if "makespan" in l.lower()]
        rec["makespan_line"] = mk[-1] if mk else None
        line = rec["makespan_line"] or ""
        m = re.search(r"Makespan \(non-periodic\):\s*([\d.]+)\s*ms", line)
        if m:
            rec["sched_makespan_nonperiodic_ms"] = float(m.group(1))
        m = re.search(r"all operations:\s*([\d.]+)\s*ms", line)
        if m:
            rec["sched_makespan_all_ms"] = float(m.group(1))
        rec["status"] = "solved"
        print(f"  {pt:<22} {solver:<16} {dt:6.1f}s  {rec.get('makespan_line','')}")
        save_state(st)
    return 0


# --------------------------------------------------------------------------
# Phase 2 — runtime emission + predicate 7
# --------------------------------------------------------------------------

TABLE_ROW = re.compile(r'^\s*\{\s*(\d+),\s*"([^"]*)",\s*(\d+),\s*(\d+),\s*"([^"]*)",'
                       r'\s*"([^"]*)",\s*"([^"]*)",\s*(-?\d+),\s*([\d.eE+-]+),'
                       r'\s*([\d.eE+-]+),\s*(\d+),\s*(\w+|NULL),\s*(-?\d+),'
                       r'\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"')


def parse_table(path):
    rows = []
    for line in open(path):
        m = TABLE_ROW.match(line)
        if m:
            rows.append(dict(entry_id=int(m.group(1)), network=m.group(2),
                             instance=int(m.group(3)), name=m.group(6),
                             kind=m.group(7), hart=int(m.group(8)),
                             start_ms=float(m.group(9)), dur_ms=float(m.group(10)),
                             backend=m.group(14), ctx=m.group(16), graph=m.group(17)))
    return rows


def cmd_runtime(args):
    st = load_state()
    cost = json.load(open(os.path.join(HERE, "cost_model.json")))
    cells = cost["cells"]
    manifest_net = {}
    for model, rel in BINDINGS.items():
        manifest_net[model] = json.load(open(os.path.join(FLOWC, rel)))["network"]
    for r in points(args.only):
        pt = r["point"]
        rec = st.setdefault(pt, {})
        if rec.get("status") not in ("solved", "emitted", "staged", "run"):
            print(f"  {pt:<22} skipped (status {rec.get('status')})"); continue
        spec = os.path.join(HERE, "workloads", f"{pt}.flowc.json")
        out_dir = os.path.join(HERE, "runtimes", pt)
        shutil.rmtree(out_dir, ignore_errors=True)
        p, dt = sh([sys.executable, "flow_c.py", "runtime", "--workload", spec,
                    "--tag", pt, "--lane-mode", "kind-network",
                    "--schedule", os.path.join(HERE, rec["schedule"]),
                    "--out-dir", out_dir],
                   cwd=FLOWC, log=os.path.join(HERE, "logs", f"runtime_{pt}.log"))
        rec["runtime_rc"] = p.returncode
        if p.returncode != 0 or not os.path.exists(os.path.join(out_dir, "dispatch_table.h")):
            rec["status"] = "runtime_failed"
            print(f"  {pt:<22} RUNTIME EMIT FAILED rc={p.returncode}")
            save_state(st); continue
        rec["dispatch_table_sha256"] = sha256(os.path.join(out_dir, "dispatch_table.h"))
        rec["runtime_main_sha256"] = sha256(os.path.join(out_dir, "runtime_main.cpp"))
        table = parse_table(os.path.join(out_dir, "dispatch_table.h"))
        rec["n_entries"] = len(table)
        rec["table_predicted_makespan_ms"] = round(
            max(t["start_ms"] + t["dur_ms"] for t in table), 3) if table else 0.0
        # lane placement: how the schedule spread the work
        lanes = {}
        for t in table:
            lanes.setdefault(f"{t['network']}/{t['name']}", {}).setdefault(t["kind"], 0)
            lanes[f"{t['network']}/{t['name']}"][t["kind"]] += 1
        rec["placement"] = lanes
        rec["lane_entry_counts"] = {k: sum(1 for t in table if t["kind"] == k)
                                    for k in sorted({t["kind"] for t in table})}
        rec["contexts"] = sorted({t["ctx"] for t in table})
        # predicate 7 — no capability-excluded sentinel in the chosen placement
        bad = []
        for t in table:
            cell = f"{manifest_net[model_of(t['network'])]}/{t['name']}"
            v = (cells.get(cell) or {}).get(t["kind"])
            if v is None:
                bad.append(f"{cell}@{t['kind']} has no measured cell (exclusion cost)")
        rec["predicate7_excluded_placements"] = sorted(set(bad))
        if bad:
            rec["status"] = "predicate7_failed"
            print(f"  {pt:<22} PREDICATE 7 FAILED: {sorted(set(bad))[:2]}")
            save_state(st); continue
        rec["status"] = "emitted"
        print(f"  {pt:<22} {rec['n_entries']:>3} entries  predicted "
              f"{rec['table_predicted_makespan_ms']:8.3f} ms  lanes "
              f"{rec['lane_entry_counts']}  sha {rec['dispatch_table_sha256'][:12]}")
        save_state(st)
    return 0


# --------------------------------------------------------------------------
# Phase 3 — stage (predicate 6) and run
# --------------------------------------------------------------------------

def cmd_stage(args):
    """Link the context binaries, then check predicate 6.

    `flow_c.py stage` links every context every binding manifest in the spec
    names, so one call per ARM covers every point in that arm — the network
    set, not the taskset, decides what is linked. Called once per arm rather
    than once per point to keep the number of board round trips down; the
    per-point record is the predicate-6 probe below, which is what actually
    gates the run.

    That one call is the only board interaction in this sweep not taken
    behind /tmp/qnn_board.lock: `flow_c.py stage` opens its own ssh and this
    driver cannot wrap it. It creates symlinks and runs no compute, so it
    cannot perturb another tenant's measurement. Everything that executes on
    the board — the probe here, and deploy_and_run.sh's run step — is locked.
    """
    st = load_state()
    pts = [r for r in points(args.only) if st.get(r["point"], {}).get("status")
           in ("emitted", "staged", "run")]
    if not pts:
        print("nothing to stage"); return 0
    staged_arms = set()
    for r in pts:
        pt, arm = r["point"], r["arm"]
        if arm not in staged_arms:
            spec = os.path.join(HERE, "workloads", f"{pt}.flowc.json")
            p, dt = sh([sys.executable, "flow_c.py", "stage", "--workload", spec,
                        "--board", BOARD],
                       cwd=FLOWC, log=os.path.join(HERE, "logs", f"stage_{arm}.log"))
            staged_arms.add(arm)
            print(f"  [stage] arm {arm}: flow_c.py stage rc={p.returncode} "
                  f"(from {pt})")
            st.setdefault(pt, {})["stage_rc"] = p.returncode
        # predicate 6 — every context the emitted table names is on the board
        want = st[pt]["contexts"]
        probe = "; ".join(f'[ -e /root/qnn_runtime_ctx/{c} ] && echo "OK {c}" || echo "MISSING {c}"'
                          for c in want)
        b = board(probe, timeout=120)
        missing = [l.split()[1] for l in b.stdout.splitlines() if l.startswith("MISSING")]
        st[pt]["predicate6_missing_contexts"] = missing
        if missing:
            st[pt]["status"] = "predicate6_failed"
            print(f"  {pt:<22} PREDICATE 6 FAILED: {missing}")
        else:
            st[pt]["status"] = "staged"
            print(f"  {pt:<22} {len(want)} context(s) staged")
        save_state(st)
    return 0


def cmd_run(args):
    st = load_state()
    for r in points(args.only):
        pt = r["point"]
        rec = st.get(pt, {})
        if rec.get("status") not in ("staged", "run"):
            print(f"  {pt:<22} skipped (status {rec.get('status')})"); continue
        rec.setdefault("reps", {})
        out_dir = os.path.join(HERE, "runtimes", pt)
        for rep in range(1, args.reps + 1):
            key = f"rep{rep}"
            if key in rec["reps"] and rec["reps"][key].get("ok") and not args.force:
                continue
            log_dir = os.path.join(HERE, "runs", pt, key)
            os.makedirs(log_dir, exist_ok=True)
            wait, wrc = lock_wait_s()
            t0 = time.time()
            p, dt = sh([sys.executable, "flow_c.py", "run", "--workload",
                        os.path.join(HERE, "workloads", f"{pt}.flowc.json"),
                        "--tag", pt, "--tuned", "--out-dir", out_dir,
                        "--log-dir", log_dir, "--board", BOARD,
                        "--board-dir", "/root/flowc_sweepc"],
                       cwd=FLOWC, timeout=1800,
                       log=os.path.join(log_dir, "driver.log"))
            info = analyse_run(os.path.join(log_dir, "run.log"))
            info["lock_wait_s"] = wait
            info["lock_probe_rc"] = wrc
            info["wall_s"] = round(dt, 1)
            info["flow_c_rc"] = p.returncode
            rec["reps"][key] = info
            print(f"  {pt:<22} {key}  wall {info.get('wall_ms')}  "
                  f"{info.get('entries_ran')}/{info.get('entries_total')} entries  "
                  f"lock wait {wait}s  ok={info.get('ok')}")
            save_state(st)
            # plots stage: writes trace.csv beside run.log (contract order)
            sh([sys.executable, "flow_c.py", "plots", "--workload",
                os.path.join(HERE, "workloads", f"{pt}.flowc.json"),
                "--tag", pt, "--log-dir", log_dir],
               cwd=FLOWC, log=os.path.join(log_dir, "plots.log"))
        if all(v.get("ok") for v in rec["reps"].values()):
            rec["status"] = "run"
        save_state(st)
    return 0


SUMMARY = re.compile(r"\[summary\] (\d+)/(\d+) entries executed, wall=([\d.]+) ms "
                     r"\(predicted makespan ([\d.]+) ms, ratio ([\d.]+)x\)")


def analyse_run(log_path):
    out = {"ok": False}
    if not os.path.exists(log_path):
        out["error"] = "no run.log"
        return out
    text = open(log_path, errors="replace").read()
    m = SUMMARY.search(text)
    if m:
        out.update(entries_ran=int(m.group(1)), entries_total=int(m.group(2)),
                   wall_ms=float(m.group(3)), predicted_ms=float(m.group(4)),
                   ratio=float(m.group(5)))
    out["bringup_lines"] = [l.strip() for l in text.splitlines()
                            if l.startswith("[bringup]")]
    out["bringup_missing"] = [l for l in out["bringup_lines"] if "MISSING" in l]
    out["iterations"] = [l.strip() for l in text.splitlines() if "iteration" in l and "wall=" in l]
    if "MODELBLASTER_XPURT_TRACE_BEGIN" in text:
        block = text.split("MODELBLASTER_XPURT_TRACE_BEGIN ===")[1] \
                    .split("=== MODELBLASTER_XPURT_TRACE_END")[0].strip()
        rows = list(csv.DictReader(io.StringIO(block)))
        per = {}
        ctxs = set()
        for r in rows:
            if not r.get("actual_end_cycles"):
                continue
            try:
                dur = (int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])) / 1000.0
            except (TypeError, ValueError):
                continue
            key = f"{r['network']}/{r['name']}@{r['core_kind']}"
            per.setdefault(key, {"pred_ms": float(r["predicted_duration_ms"]),
                                 "actual": []})["actual"].append(dur)
            ctxs.add((r["network"], r["name"], r["core_kind"], r["ctx"]))
        for k, v in per.items():
            v["n"] = len(v["actual"])
            v["actual_p50_ms"] = round(statistics.median(v["actual"]), 4)
            v["actual_min_ms"] = round(min(v["actual"]), 4)
            v["actual_max_ms"] = round(max(v["actual"]), 4)
            v["ratio_p50"] = round(v["actual_p50_ms"] / v["pred_ms"], 3) if v["pred_ms"] else None
            del v["actual"]
        out["per_tile"] = per
        out["trace_rows"] = len(rows)
        out["trace_ctx_pairs"] = sorted("/".join(c) for c in ctxs)
    out["ok"] = bool(m) and out.get("entries_ran") == out.get("entries_total") \
        and not out.get("bringup_missing")
    return out


# --------------------------------------------------------------------------
# results.json
# --------------------------------------------------------------------------

def solver_status(pt, solver):
    """cvxpy's own status line for a MILP point.

    MOSEK returns `optimal` when it proved optimality inside the 300 s limit
    and `optimal_inaccurate` when the limit cut it off and it handed back the
    incumbent. Those are different claims and the sweep records which one it
    got, because SETUP.md's solver plan ("MOSEK is optimal at 66-86 ops")
    only holds for the first.
    """
    log = os.path.join(HERE, "logs", f"sched_{pt}_{'milp' if solver == 'milp' else 'greedy'}.log")
    if not os.path.exists(log):
        return None
    for line in open(log, errors="replace"):
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return None


def cmd_results(args):
    st = load_state()
    gen = json.load(open(os.path.join(HERE, "generated.json")))
    cost_sha = sha256(os.path.join(HERE, "cost_model.json"))
    ppath = os.path.join(HERE, "provenance.json")
    prov = json.load(open(ppath)) if os.path.exists(ppath) else {}
    rows = []
    for g in gen:
        pt = g["point"]
        rec = st.get(pt, {})
        row = {
            "point": pt, "arm": g["arm"], "seed": g["seed"],
            "status": g["status"] if g["status"] != "ok" else rec.get("status", "not_run"),
            "validation_problems": g.get("problems", []),
            "max_ops": g.get("max_ops"),
            "op_count": g.get("op_count"),
            "horizon_ms": g.get("horizon_ms"),
            "hyperperiod_ms": g.get("hyperperiod_ms"),
            "periodic": g.get("periodic"), "nonperiodic": g.get("nonperiodic"),
            "networks": g.get("networks"),
            "workload": g.get("workload"),
        }
        if g["status"] != "ok":
            rows.append(row); continue
        row.update({
            "flowc_spec": g.get("flowc_spec"),
            "solver": rec.get("solver"),
            "solver_status": solver_status(pt, rec.get("solver")),
            "milp_timed_out_s": rec.get("milp_timed_out_s"),
            "solve_s": rec.get("solve_s"),
            "schedule": rec.get("schedule"),
            "sched_makespan_nonperiodic_ms": rec.get("sched_makespan_nonperiodic_ms"),
            "sched_makespan_all_ms": rec.get("sched_makespan_all_ms"),
            "predicted_makespan_ms": rec.get("table_predicted_makespan_ms"),
            "n_entries": rec.get("n_entries"),
            "dispatch_table_sha256": rec.get("dispatch_table_sha256"),
            "runtime_main_sha256": rec.get("runtime_main_sha256"),
            "lane_entry_counts": rec.get("lane_entry_counts"),
            "lane_placement": rec.get("placement"),
            "contexts": rec.get("contexts"),
            "provenance": {
                "cost_model_sha256": cost_sha,
                "predicate6_missing_contexts": rec.get("predicate6_missing_contexts"),
                "predicate7_excluded_placements": rec.get("predicate7_excluded_placements"),
                "table_vs_manifest_mismatches": prov.get(pt, {}).get(
                    "table_vs_manifest_mismatches"),
                "trace_vs_table": {k: {"entries_with_timings": v.get("entries_with_timings"),
                                       "entries_expected": v.get("entries_expected"),
                                       "all_entries_ran": v.get("all_entries_ran"),
                                       "n_mismatches": v.get("n_mismatches")}
                                   for k, v in (prov.get(pt, {}).get("reps") or {}).items()},
                "bringup_lines_unavailable": (
                    "deploy_and_run.sh's `exec {lockfd}> /tmp/qnn_board.lock 2>/dev/null` "
                    "redirects the board-side shell's stderr for the rest of the script, so "
                    "no [bringup] line reaches run.log. Replaced by predicate 6 (contexts "
                    "verified staged before the run), the N/N entry count (the runtime skips "
                    "entries whose context is missing), and trace_vs_table above, which "
                    "checks the ctx actually used per entry. See verify_provenance.py."),
            },
        })
        reps = rec.get("reps", {})
        walls, ratios = [], []
        rep_out = {}
        for k in sorted(reps):
            v = reps[k]
            rep_out[k] = {
                "wall_ms": v.get("wall_ms"), "ratio": v.get("ratio"),
                "entries": f"{v.get('entries_ran')}/{v.get('entries_total')}",
                "lock_wait_s": v.get("lock_wait_s"),
                "run_wall_s": v.get("wall_s"),
                "iterations": v.get("iterations"),
                "ok": v.get("ok"),
            }
            if v.get("wall_ms") is not None:
                walls.append(v["wall_ms"]); ratios.append(v["ratio"])
        row["reps"] = rep_out
        row["governor"] = "performance during run (flow_c.py run --tuned), restored to schedutil after each rep"
        row["warmup"] = "FLOWC_ITERATIONS=2; the reported trace is walk 2"
        if walls:
            row["actual_makespan_ms"] = walls
            row["actual_makespan_median_ms"] = round(statistics.median(walls), 3)
            row["actual_makespan_spread_ms"] = round(max(walls) - min(walls), 3)
            row["ratio_per_rep"] = ratios
            row["ratio_median"] = round(statistics.median(ratios), 3)
            row["lock_wait_s"] = [reps[k].get("lock_wait_s") for k in sorted(reps)]
            row["lock_wait_exceeds_run"] = [
                bool(reps[k].get("lock_wait_s", 0) * 1000.0 > (reps[k].get("wall_ms") or 0))
                for k in sorted(reps)]
            # per-tile ratios, pooled across reps (median of the per-rep medians)
            keys = set()
            for k in reps:
                keys |= set((reps[k].get("per_tile") or {}))
            tiles = {}
            for key in sorted(keys):
                pr, ac = [], []
                for k in reps:
                    t = (reps[k].get("per_tile") or {}).get(key)
                    if t:
                        pr.append(t["pred_ms"]); ac.append(t["actual_p50_ms"])
                if ac:
                    tiles[key] = {
                        "predicted_ms": round(statistics.median(pr), 4),
                        "actual_p50_ms": round(statistics.median(ac), 4),
                        "ratio": round(statistics.median(ac) / statistics.median(pr), 3)
                        if statistics.median(pr) else None,
                        "reps": len(ac),
                    }
            row["per_tile"] = tiles
        rows.append(row)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ok = sum(1 for r in rows if r["status"] == "run")
    print(f"wrote results.json: {ok} points run, "
          f"{sum(1 for r in rows if r['status'] == 'REJECTED')} rejected, "
          f"{len(rows)} total")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["solve", "runtime", "stage", "run", "results"])
    ap.add_argument("--only", default=None, help="comma list of point names")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--greedy-only", action="store_true")
    args = ap.parse_args()
    return {"solve": cmd_solve, "runtime": cmd_runtime, "stage": cmd_stage,
            "run": cmd_run, "results": cmd_results}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
