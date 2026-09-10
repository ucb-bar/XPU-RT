#!/usr/bin/env python3
"""Phase 4 driver: take solved schedules onto the board and measure them.

This is the step the reference experiment never took. `sched_algo_sweep10`'s
own README says it plainly -- "Every number here is a predicted makespan from
the xpu-rt cost model. No FPGA." -- and its `fpga/schedules/`, `fpga/logs/`
and `elf/` trees are empty. Its headline claim (pso +9.10%, cpsat:warmbest
+9.75%, decomposed -4.92% vs greedy) has never been checked against hardware.

    emit       fpga/emit_schedule.py per (cell, solver): the solver's (t, alpha)
               straight into postprocessing.output_scheduled_json. Dedupes by
               `_sched_hash` -- the op -> (combination, start, duration) map --
               so two solvers that produced the same assignment consume board
               time once. Host only.
    runtime    flow_c.py runtime per unique schedule -> dispatch_table.h +
               runtime_main.cpp, sha256'd, plus predicate 7 (no
               capability-excluded cell in the chosen placement). Host only.
    stage      predicate 6: every context the emitted table names is on the
               board. One locked probe per point.
    run        N reps per point, serially, `--tuned`, medians reported.
    results    assemble results.json.

Board discipline. Every board interaction this script makes itself goes behind
`ssh -n root@... "flock -w 900 /tmp/qnn_board.lock -c '...'"` wrapped in
`timeout -s KILL`. `flow_c.py run` takes the same lock itself inside
deploy_and_run.sh, so it is invoked WITHOUT an outer lock -- taking one here
would make its inner flock wait out its own 900 s. Lock wait is probed and
recorded immediately before each rep.
"""
from __future__ import annotations

import argparse, csv, glob, hashlib, io, json, os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")
ARM = "s10port"
CPSAT_PY = os.environ.get("XPURT_CPSAT_PYTHON",
                          os.path.join(REPO, ".cpsat-venv", "bin", "python"))
COST_MODEL = os.path.join(SWEEP, "cost_model.json")
STATE = os.path.join(SWEEP, "results", "phase4_state.json")
PLAN = os.path.join(SWEEP, "results", "phase4_plan.json")

SOLVERS = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
           "heft", "heft_edf", "pso", "sa", "cpsat", "cpsat:warm",
           "best-of-fast", "cpsat:warmbest"]


def sh(cmd, log=None, timeout=None, env=None, cwd=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        dec = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
        if log:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "w") as f:
                f.write("+ " + " ".join(cmd) + f"\n!! TIMED OUT after {timeout}s\n"
                        + dec(e.stdout) + "\n--- stderr ---\n" + dec(e.stderr))
        raise
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            f.write("+ " + " ".join(cmd) + f"\n(cwd={cwd})\n" + (p.stdout or "")
                    + "\n--- stderr ---\n" + (p.stderr or ""))
    return p, round(time.time() - t0, 2)


def board(script, timeout=600):
    q = script.replace("'", "'\\''")
    return subprocess.run(
        ["timeout", "-s", "KILL", str(timeout + 60), "ssh", "-n",
         "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
         f"flock -w 900 /tmp/qnn_board.lock -c '{q}'"],
        capture_output=True, text=True, timeout=timeout + 120)


def lock_wait_s(timeout=900):
    """How long acquiring the board lock takes right now: the same flock on the
    same path, taken immediately before the run that follows."""
    t0 = time.time()
    r = subprocess.run(["timeout", "-s", "KILL", str(timeout + 30), "ssh", "-n",
                        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
                        f"flock -w {timeout} /tmp/qnn_board.lock -c 'true'"],
                       capture_output=True, text=True)
    return round(time.time() - t0, 2), r.returncode


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(p, default):
    return json.load(open(p)) if os.path.exists(p) else default


def save(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1)


def solve_env():
    env = dict(os.environ)
    env.update(XPURT_CODE_ROOT=REPO, XPURT_DATA_ROOT=REPO,
               XPURT_CPSAT_PYTHON=CPSAT_PY)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"
    return env


# --------------------------------------------------------------------------
# the run plan
# --------------------------------------------------------------------------

def cmd_plan(args):
    """Board time is the binding constraint, so the plan is explicitly tiered.

    tier A  ALL 12 solvers on a representative subset of cells. This is what
            actually tests the reference's ranking claim, so it comes first.
    tier B  winner + greedy for EVERY cell (coverage), winner picked by
            fpga/pick_winners.py's feasible-first rule.
    tier C  whatever else fits.
    """
    res = json.load(open(os.path.join(SWEEP, "results", "phase3_all_results.json")))
    by_cell = {}
    for r in res:
        by_cell.setdefault((r["arm"], r["workload"]), {})[r["solver"]] = r
    subset = [x.strip() for x in args.tier_a.split(",") if x.strip()]
    plan = []
    for wl in subset:
        key = (ARM, wl)
        if key not in by_cell:
            print(f"  tier A: no solve results for {wl} — skipping")
            continue
        for s in SOLVERS:
            if s in by_cell[key]:
                plan.append(dict(tier="A", arm=ARM, workload=wl, solver=s))
    win = json.load(open(os.path.join(SWEEP, "results", "winners.json")))
    for w in win:
        for s in ("greedy", w["winner"]):
            if s is None:
                continue
            row = dict(tier="B", arm=ARM, workload=w["workload"], solver=s)
            if not any(p["workload"] == row["workload"] and p["solver"] == s
                       for p in plan):
                plan.append(row)
    save(PLAN, plan)
    n_a = sum(1 for p in plan if p["tier"] == "A")
    print(f"  {len(plan)} planned points: tier A {n_a}, tier B {len(plan)-n_a}"
          f"  -> {PLAN}")
    return 0


def cmd_plan_extend(args):
    """Append points to an EXISTING plan without rewriting it.

    Phase 4 was tiered by board time: tier A ran all 12 solvers on 12 cells,
    tier B only winner+greedy on the other 30. That left `cpsat:warmbest` --
    this study's own recommendation for the offline/build-time path -- measured
    on 12 cells out of 42, so any comparison against "XPU-RT" on the other 30
    was really a comparison against greedy. This closes that gap.

    Append-only on purpose: `cmd_plan` regenerates the plan from scratch, and
    rerunning it would renumber tiers under 199 already-measured points.
    """
    res = json.load(open(os.path.join(SWEEP, "results", "all_results.json")))
    cells = sorted({r["workload"] for r in res})
    plan = load(PLAN, [])
    have = {(p["workload"], p["solver"]) for p in plan}
    solvers = [x.strip() for x in args.solvers.split(",") if x.strip()]
    added = []
    for wl in cells:
        for s in solvers:
            if (wl, s) in have:
                continue
            plan.append(dict(tier=args.tier_label, arm=ARM, workload=wl, solver=s))
            added.append(f"{wl}::{s}")
    save(PLAN, plan)
    print(f"  +{len(added)} point(s) at tier {args.tier_label}; plan now {len(plan)}")
    for a in added:
        print(f"    {a}")
    return 0


def points(args):
    plan = load(PLAN, [])
    if args.tier:
        want = set(args.tier.split(","))
        plan = [p for p in plan if p["tier"] in want]
    if args.only:
        want = set(args.only.split(","))
        plan = [p for p in plan if p["workload"] in want]
    return plan


def point_id(p):
    return f'{p["workload"].replace("networks_", "")}__{p["solver"].replace(":", "-")}'


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def _emit_one(args, p, sdir, emitter):
    """Run the emitter for one point. Pure subprocess work, no state writes --
    the caller serialises those, so this is safe to run in a pool."""
    pid = point_id(p)
    out = os.path.join(sdir, f"scheduled_{pid}.json")
    meta = os.path.join(sdir, f"scheduled_{pid}.meta.json")
    cmd = [sys.executable, emitter, "--arm", p["arm"], "--name", p["workload"],
           "--solver", p["solver"], "--out", out, "--meta-out", meta,
           "--cpsat-time", str(args.cpsat_time),
           "--cpsat-workers", str(args.cpsat_workers)]
    if getattr(args, "mask_undeclared", False):
        cmd += ["--mask-undeclared", COST_MODEL]
    try:
        q, dt = sh(cmd, log=os.path.join(SWEEP, "logs", "emit", pid + ".log"),
                   timeout=args.cpsat_time + 900, env=solve_env())
    except subprocess.TimeoutExpired:
        return pid, None, None, out, meta
    return pid, q, dt, out, meta


def cmd_emit(args):
    st = load(STATE, {})
    emitter = os.path.join(SWEEP, "fpga", "emit_schedule.py")
    sdir = os.path.join(SWEEP, "schedules")
    os.makedirs(sdir, exist_ok=True)
    seen_hash = {pid: r["sched_hash"] for pid, r in st.items()
                 if r.get("sched_hash")}
    todo = []
    for p in points(args):
        pid = point_id(p)
        rec = st.setdefault(pid, dict(p))
        if rec.get("sched_hash") and not args.force:
            print(f"  {pid:<46} already emitted ({rec['sched_hash'][:12]})")
            continue
        todo.append(p)
    # Two pools, for the same reason `sweep10_dispatch.py` and `fpga/emit_all.py`
    # have two: CP-SAT asks for `--cpsat-workers` search threads inside its own
    # process, every other solver is single-threaded. One pool at either width
    # starves the heuristics or oversubscribes the box 8x.
    results = []
    if todo:
        import concurrent.futures as cf
        cheap = [p for p in todo if not p["solver"].startswith("cpsat")]
        heavy = [p for p in todo if p["solver"].startswith("cpsat")]
        with cf.ThreadPoolExecutor(max(1, args.cheap_workers)) as ex1, \
             cf.ThreadPoolExecutor(max(1, args.cpsat_parallel)) as ex2:
            futs = [ex1.submit(_emit_one, args, p, sdir, emitter) for p in cheap] \
                 + [ex2.submit(_emit_one, args, p, sdir, emitter) for p in heavy]
            for f in cf.as_completed(futs):
                results.append(f.result())
    # State is written here, serially and in a deterministic order, so the
    # dedupe identity does not depend on which future finished first.
    by_pid = {r[0]: r[1:] for r in results}
    for p in todo:
        pid = point_id(p)
        rec = st.setdefault(pid, dict(p))
        q, dt, out, meta = by_pid[pid]
        if q is None:
            rec["status"] = "emit_timeout"; save(STATE, st)
            print(f"  {pid:<46} EMIT TIMEOUT"); continue
        if q.returncode != 0 or not os.path.exists(meta):
            rec["status"] = "emit_failed"
            rec["emit_stderr"] = (q.stderr or "")[-600:]
            save(STATE, st)
            print(f"  {pid:<46} EMIT FAILED rc={q.returncode}")
            continue
        m = json.load(open(meta))
        rec.update({k: m[k] for k in
                    ("ops", "periodic_ops", "combos", "wall_s", "objective",
                     "all_ops", "misses", "validation", "dispatches",
                     "json_makespan", "sched_hash", "picked")})
        if m.get("masked_cells"):
            rec["masked_cells"] = m["masked_cells"]
        rec["schedule"] = os.path.relpath(out, SWEEP)
        rec["emit_s"] = dt
        # dedupe: identical assignment -> reuse the earlier point's runtime
        dup = [q for q, h in seen_hash.items()
               if h == rec["sched_hash"] and q != pid]
        rec["duplicate_of"] = sorted(dup)[0] if dup else None
        seen_hash[pid] = rec["sched_hash"]
        rec["status"] = "emitted"
        save(STATE, st)
        print(f"  {pid:<46} obj {rec['objective']:>10.3f}  misses "
              f"{rec['misses']:>3}  {rec['dispatches']:>4} disp  "
              f"hash {rec['sched_hash'][:12]}"
              + (f"  == {rec['duplicate_of']}" if rec["duplicate_of"] else ""))
    uniq = len({r["sched_hash"] for r in st.values() if r.get("sched_hash")})
    print(f"\n  {len(st)} points, {uniq} unique schedules by content hash")
    return 0


# --------------------------------------------------------------------------
# runtime  (+ predicate 7)
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
                             backend=m.group(14), ctx=m.group(16),
                             graph=m.group(17)))
    return rows


def flowc_spec_path(workload):
    return os.path.join(SWEEP, "specs", f"{workload}.flowc.json")


def cmd_runtime(args):
    st = load(STATE, {})
    cells = json.load(open(COST_MODEL))["cells"]
    for p in points(args):
        pid = point_id(p)
        rec = st.get(pid)
        if not rec or rec.get("status") not in ("emitted", "runtime", "staged", "run"):
            print(f"  {pid:<46} skipped (status {rec and rec.get('status')})")
            continue
        if rec.get("duplicate_of") and not args.no_dedupe:
            rec["status"] = "duplicate"
            save(STATE, st)
            print(f"  {pid:<46} duplicate of {rec['duplicate_of']} — no board time")
            continue
        if rec.get("dispatch_table_sha256") and not args.force:
            print(f"  {pid:<46} already emitted runtime "
                  f"({rec['dispatch_table_sha256'][:12]})")
            continue
        spec = flowc_spec_path(p["workload"])
        out_dir = os.path.join(SWEEP, "runtimes", pid)
        shutil.rmtree(out_dir, ignore_errors=True)
        q, dt = sh([sys.executable, "flow_c.py", "runtime", "--workload", spec,
                    "--tag", pid, "--lane-mode", "kind-network",
                    "--schedule", os.path.join(SWEEP, rec["schedule"]),
                    "--out-dir", out_dir],
                   log=os.path.join(SWEEP, "logs", "runtime", pid + ".log"),
                   cwd=FLOWC, timeout=1800)
        rec["runtime_rc"] = q.returncode
        table_h = os.path.join(out_dir, "dispatch_table.h")
        if q.returncode != 0 or not os.path.exists(table_h):
            rec["status"] = "runtime_failed"
            rec["runtime_stderr"] = (q.stderr or "")[-600:]
            save(STATE, st)
            print(f"  {pid:<46} RUNTIME EMIT FAILED rc={q.returncode}")
            continue
        rec["dispatch_table_sha256"] = sha256(table_h)
        rec["runtime_main_sha256"] = sha256(os.path.join(out_dir, "runtime_main.cpp"))
        table = parse_table(table_h)
        rec["n_entries"] = len(table)
        rec["table_predicted_makespan_ms"] = round(
            max(t["start_ms"] + t["dur_ms"] for t in table), 3) if table else 0.0
        rec["lane_entry_counts"] = {k: sum(1 for t in table if t["kind"] == k)
                                    for k in sorted({t["kind"] for t in table})}
        rec["contexts"] = sorted({t["ctx"] for t in table})
        # predicate 7 -- no capability-excluded sentinel in the chosen placement
        bad = []
        for t in table:
            net = t["network"]
            # instance copies are named <net>; the tile name carries the cell
            key = f'{net}/{t["name"]}'
            if (cells.get(key) or {}).get(t["kind"]) is None:
                bad.append(f'{key}@{t["kind"]} has no measured cell')
        rec["predicate7_excluded_placements"] = sorted(set(bad))
        if bad:
            rec["status"] = "predicate7_failed"
            save(STATE, st)
            print(f"  {pid:<46} PREDICATE 7 FAILED: {sorted(set(bad))[:2]}")
            continue
        rec["status"] = "runtime"
        save(STATE, st)
        print(f"  {pid:<46} {rec['n_entries']:>4} entries  predicted "
              f"{rec['table_predicted_makespan_ms']:9.3f} ms  "
              f"{rec['lane_entry_counts']}  sha {rec['dispatch_table_sha256'][:12]}")
    return 0


# --------------------------------------------------------------------------
# stage  (predicate 6)
# --------------------------------------------------------------------------

def cmd_stage(args):
    st = load(STATE, {})
    for p in points(args):
        pid = point_id(p)
        rec = st.get(pid)
        if not rec or rec.get("status") not in ("runtime", "staged", "run"):
            continue
        want = rec["contexts"]
        probe = "; ".join(
            f'[ -e /root/qnn_runtime_ctx/{c} ] && echo "OK {c}" || echo "MISSING {c}"'
            for c in want)
        b = board(probe + "; df -h / | tail -1", timeout=180)
        missing = [l.split()[1] for l in b.stdout.splitlines()
                   if l.startswith("MISSING")]
        rec["predicate6_missing_contexts"] = missing
        for l in b.stdout.splitlines():
            if l.strip().startswith("/dev/root"):
                rec["board_df"] = l.strip()
        rec["status"] = "predicate6_failed" if missing else "staged"
        save(STATE, st)
        print(f"  {pid:<46} " + (f"PREDICATE 6 FAILED: {missing}" if missing
                                 else f"{len(want)} context(s) present"))
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

SUMMARY = re.compile(r"\[summary\] (\d+)/(\d+) entries executed, wall=([\d.]+) ms "
                     r"\(predicted makespan ([\d.]+) ms, ratio ([\d.]+)x\)")


def periodic_networks(workload):
    """Which networks in this taskset carry a period.

    Needed because the quantity the reference RANKS solvers on is NOT the wall
    clock. `sweep10_runner` scores with `evaluate(ctx, t, alpha, True)`, whose
    objective is the makespan over NON-PERIODIC operations only: periodic ops
    have their own windows and are not what is being minimised. The board's
    wall clock is the all-operations makespan, which is usually pinned by the
    last periodic instance and is therefore nearly identical across solvers.
    Measuring only the wall clock would make every solver look the same.
    So both are extracted, and the ranking is checked against the one the
    ranking is about.
    """
    p = os.path.join(REPO, "data", "toplevel", ARM, workload + ".json")
    if not os.path.exists(p):
        return set(), set()
    nets = json.load(open(p))["networks"]
    per = {k for k, v in nets.items() if v.get("period") is not None}
    return per, set(nets) - per


def analyse_run(log_path, periodic=(), nonperiodic=()):
    out = {"ok": False}
    if not os.path.exists(log_path):
        out["error"] = "no run.log"
        return out
    txt = open(log_path, errors="replace").read()
    m = None
    for m in SUMMARY.finditer(txt):
        pass
    if m:
        out.update(entries_ran=int(m.group(1)), entries_total=int(m.group(2)),
                   wall_ms=float(m.group(3)),
                   predicted_ms=float(m.group(4)), ratio=float(m.group(5)),
                   ok=int(m.group(1)) == int(m.group(2)))
    out["bringups"] = len(re.findall(r"\[bringup\]", txt))
    out["skipped"] = len(re.findall(r"\[skip\]|skipped entry", txt))
    # per-tile in-situ durations from the embedded trace block
    # The markers are printed as `=== MODELBLASTER_XPURT_TRACE_BEGIN ===`, so
    # the capture must start AFTER that whole line -- otherwise csv.DictReader
    # takes " ===" as the one and only field name and every row parses to
    # nothing, silently.
    tr = re.search(r"MODELBLASTER_XPURT_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
                   r"MODELBLASTER_XPURT_TRACE_END", txt, re.S)
    if tr:
        rows = list(csv.DictReader(io.StringIO(tr.group(1).strip())))
        out["trace_rows"] = len(rows)
        per, ends, np_ends = {}, [], []
        unit_ms = {"us": 1e-3, "ms": 1.0, "ns": 1e-6}
        for r in rows:
            try:
                u = unit_ms.get((r.get("time_unit") or "us").strip(), 1e-3)
                st = float(r["actual_start_cycles"]) * u
                en = float(r["actual_end_cycles"]) * u
            except (TypeError, ValueError, KeyError):
                continue
            ends.append(en)
            net = (r.get("network") or "").strip()
            if net in nonperiodic:
                np_ends.append(en)
            key = f'{net}/{(r.get("name") or "").strip()}@{(r.get("core_kind") or "").strip()}'
            per.setdefault(key, []).append(round(en - st, 4))
        if ends:
            out["measured_all_ms"] = round(max(ends), 4)
        if np_ends:
            out["measured_nonperiodic_ms"] = round(max(np_ends), 4)
        elif ends:
            # every op is periodic: the objective degenerates to the
            # all-operations makespan, exactly as evaluate() does
            out["measured_nonperiodic_ms"] = round(max(ends), 4)
            out["nonperiodic_degenerate"] = True
        out["per_tile"] = {k: dict(n=len(v), p50=round(statistics.median(v), 4),
                                   max=round(max(v), 4))
                           for k, v in per.items() if v}
    return out


def cmd_run(args):
    st = load(STATE, {})
    for p in points(args):
        pid = point_id(p)
        rec = st.get(pid)
        if not rec or rec.get("status") not in ("staged", "run"):
            continue
        rec.setdefault("reps", {})
        out_dir = os.path.join(SWEEP, "runtimes", pid)
        for rep in range(1, args.reps + 1):
            key = f"rep{rep}"
            if key in rec["reps"] and rec["reps"][key].get("ok") and not args.force:
                continue
            log_dir = os.path.join(SWEEP, "runs", pid, key)
            os.makedirs(log_dir, exist_ok=True)
            wait, wrc = lock_wait_s()
            try:
                q, dt = sh([sys.executable, "flow_c.py", "run", "--workload",
                            flowc_spec_path(p["workload"]), "--tag", pid,
                            "--tuned", "--out-dir", out_dir, "--log-dir", log_dir,
                            "--board", BOARD, "--board-dir", "/root/flowc_s10run"],
                           log=os.path.join(log_dir, "driver.log"),
                           cwd=FLOWC, timeout=args.run_timeout)
                rc = q.returncode
            except subprocess.TimeoutExpired:
                rc, dt = -9, args.run_timeout
            per_nets, np_nets = periodic_networks(p["workload"])
            info = analyse_run(os.path.join(log_dir, "run.log"),
                               per_nets, np_nets)
            info.update(lock_wait_s=wait, lock_probe_rc=wrc, wall_s=dt,
                        flow_c_rc=rc)
            rec["reps"][key] = info
            print(f"  {pid:<46} {key}  wall {info.get('wall_ms')}  "
                  f"{info.get('entries_ran')}/{info.get('entries_total')} entries  "
                  f"lock {wait}s  ok={info.get('ok')}")
            save(STATE, st)
        walls = [v["wall_ms"] for v in rec["reps"].values() if v.get("wall_ms")]
        if walls:
            rec["measured_median_ms"] = round(statistics.median(walls), 3)
            rec["measured_spread_ms"] = round(max(walls) - min(walls), 3)
            rec["measured_reps_ms"] = walls
        nps = [v["measured_nonperiodic_ms"] for v in rec["reps"].values()
               if v.get("measured_nonperiodic_ms")]
        if nps:
            rec["measured_np_median_ms"] = round(statistics.median(nps), 3)
            rec["measured_np_spread_ms"] = round(max(nps) - min(nps), 3)
            rec["measured_np_reps_ms"] = nps
        if rec["reps"] and all(v.get("ok") for v in rec["reps"].values()):
            rec["status"] = "run"
        save(STATE, st)
        # Each point gets its own board dir with a built binary. Across ~100
        # points that is hundreds of MB, and cDSP crash dumps have filled this
        # board's / to 94% before. Drop the point's tree once its reps are in,
        # and record what df says so a slow leak is visible in the record
        # rather than discovered at hour six.
        b = board(f"rm -rf /root/flowc_s10run_{pid}; df -h / | tail -1", timeout=120)
        for l in (b.stdout or "").splitlines():
            if l.strip().startswith("/dev/root"):
                rec["board_df_after"] = l.strip()
        save(STATE, st)
    return 0


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

def cmd_results(args):
    st = load(STATE, {})
    plan = {point_id(p): p for p in load(PLAN, [])}
    # Attribute a representative's measurement back to every point that
    # deduped onto it. The dedupe key is the schedule's CONTENT hash -- the
    # op -> (combination, start, duration) map -- so two points sharing it
    # emit the same dispatch table and execute the same run. Not propagating
    # would leave a solver with a measured cell only where its schedule
    # happened to be unique, which is exactly the cells where it disagrees
    # with the others, and would bias every per-solver aggregate.
    by_hash = {}
    for pid, rec in st.items():
        if rec.get("measured_median_ms") and rec.get("sched_hash"):
            by_hash.setdefault(rec["sched_hash"], pid)
    n_prop = 0
    for pid, rec in st.items():
        if rec.get("measured_median_ms") or not rec.get("sched_hash"):
            continue
        src = by_hash.get(rec["sched_hash"])
        if not src or src == pid:
            continue
        r = st[src]
        for k in ("measured_median_ms", "measured_spread_ms", "measured_reps_ms",
                  "measured_np_median_ms", "measured_np_spread_ms",
                  "measured_np_reps_ms", "n_entries",
                  "table_predicted_makespan_ms", "lane_entry_counts", "reps"):
            if k in r:
                rec[k] = r[k]
        rec["measured_via"] = src
        n_prop += 1
    if n_prop:
        print(f"  attributed {n_prop} deduped point(s) to their representative")
    rows = []
    for pid, rec in sorted(st.items()):
        r = dict(point=pid, tier=rec.get("tier"), arm=rec.get("arm"),
                 workload=rec.get("workload"), solver=rec.get("solver"),
                 status=rec.get("status"))
        for k in ("ops", "periodic_ops", "objective", "misses", "dispatches",
                  "sched_hash", "duplicate_of", "picked", "wall_s",
                  "n_entries", "table_predicted_makespan_ms",
                  "lane_entry_counts", "dispatch_table_sha256",
                  "predicate6_missing_contexts",
                  "predicate7_excluded_placements", "measured_median_ms",
                  "measured_spread_ms", "measured_reps_ms",
                  "measured_np_median_ms", "measured_np_spread_ms",
                  "measured_np_reps_ms", "all_ops", "validation",
                  "board_df", "measured_via"):
            if k in rec:
                r[k] = rec[k]
        if rec.get("reps"):
            r["reps"] = rec["reps"]
        if r.get("measured_median_ms") and r.get("table_predicted_makespan_ms"):
            r["measured_over_predicted"] = round(
                r["measured_median_ms"] / r["table_predicted_makespan_ms"], 4)
        # the quantity the reference actually ranks on
        if r.get("measured_np_median_ms") and r.get("objective"):
            r["np_measured_over_predicted"] = round(
                r["measured_np_median_ms"] / r["objective"], 4)
        rows.append(r)
    out = os.path.join(SWEEP, "results", "phase4_results.json")
    save(out, rows)
    ran = [r for r in rows if r.get("measured_median_ms")]
    print(f"  {len(rows)} points, {len(ran)} with a measured median -> {out}")
    if ran:
        ratios = [r["measured_over_predicted"] for r in ran]
        spreads = [r["measured_spread_ms"] / r["measured_median_ms"] * 100
                   for r in ran if r.get("measured_median_ms")]
        print(f"  measured/predicted: median {statistics.median(ratios):.3f}x  "
              f"min {min(ratios):.3f}  max {max(ratios):.3f}")
        if spreads:
            print(f"  rep spread as % of median: median "
                  f"{statistics.median(spreads):.2f}%  max {max(spreads):.2f}%")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("plan-extend", cmd_plan_extend),
                     ("emit", cmd_emit),
                     ("runtime", cmd_runtime), ("stage", cmd_stage),
                     ("run", cmd_run), ("results", cmd_results)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--only", default=None)
        s.add_argument("--tier", default=None)
        s.add_argument("--force", action="store_true")
        if name == "plan":
            s.add_argument("--tier-a", default="")
        if name == "plan-extend":
            s.add_argument("--solvers", required=True)
            s.add_argument("--tier-label", default="D")
        if name == "emit":
            s.add_argument("--cpsat-time", type=float, default=60.0)
            s.add_argument("--cpsat-workers", type=int, default=8)
            s.add_argument("--cheap-workers", type=int, default=12)
            s.add_argument("--cpsat-parallel", type=int, default=5)
            s.add_argument("--mask-undeclared", action="store_true")
        if name == "runtime":
            s.add_argument("--no-dedupe", action="store_true")
        if name == "run":
            s.add_argument("--reps", type=int, default=3)
            s.add_argument("--run-timeout", type=int, default=1800)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
