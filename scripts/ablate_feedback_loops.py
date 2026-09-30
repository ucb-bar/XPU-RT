#!/usr/bin/env python3
"""Ablate the two feedback loops separately: inner (offline search) and outer (measured).

THE CLAIM THIS IS FOR. The channel closes at two distances:

  INNER = AOT co-design with ModelBlaster. Everything decided before deployment against
  the ISOLATED profile database -- graph rewrites (fuse/unfuse/split/shard) and the
  per-dispatch implementation choice (RVV vs the matrix engine), each costed by
  per-dispatch profiling in isolation. The board may be used here, but only as a
  PROFILER.

  OUTER = HIL, the board and the runtime. Costs observed while the whole schedule runs
  in real time -- the per-op inflation a dispatch actually suffers in situ, which the
  isolated profile does not predict -- returned as measured multipliers and re-solved
  against. The board here is the RUNTIME, not a profiler.

Showing one workload where the pair helps does not separate them, and does not say
whether either alone would have done. This runs the 2x2, and runs it per SOLVER, so the
same figure answers the other question a reader has: does the exact solver earn its
cost against the greedy heuristic, in each cell?

    cell   inner (levers)   outer (solve on measured costs)
    ----   --------------   -------------------------------
    A      no               no                (the naive deployment)
    B      YES              no                (offline co-design, deployed blind)
    C      no               YES               (measure-and-re-solve, no co-design)
    D      YES              YES               (both)

EVERY CELL IS SCORED ON BOARD COSTS, because that is what silicon does. The cells differ
in what the SCHEDULER KNEW, not in how they are judged:

  * A and B are solved against predicted costs and then RE-COST on the measured board
    multipliers with their assignment held fixed -- exactly what deploying them means.
  * C and D are solved with --board-calibration, so their own costs are already the
    board's; re-costing them again would apply the multiplier twice.

Scoring is instance-level (`xpu-rt/schedule_eval.py`): an instance misses when its last
dispatch ends past `inst*period + window`. That is the number the figures use.

Usage:
  scripts/ablate_feedback_loops.py --workloads data/toplevel/a.json ... \\
      --calibration results/codesign_feedback/k1_board_calibration.json \\
      --solver greedy --out-dir results/loop_ablation
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)

import schedule_eval  # noqa: E402

# The 25 K1 specs are not 25 experiments. Two files are byte-identical
# (_evo_og.json == networks_k1_mb_3model_4hz_yolo_ctrl.json), and every pair differing
# only in machine_combination_mode / enable_impls / topo_tag_override returned
# bit-identical loop results -- the loop sets those per lever, so the spec field has no
# observable effect. Counting them separately inflates n about fourfold. The family is
# the unit; the representative is the member the loop actually ran first.
FAMILIES = {
    "sensor_rich": ["networks_k1_sensor_sharded_rich",
                    "_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0",
                    "_evo_sen_og",
                    "_4w_networks_k1_sensor_tight_vision_head",
                    "_4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0",
                    "networks_k1_deployed_rich_vithead",
                    "networks_k1_deployed_rich_vithead_noime"],
    "flight_deployed": ["networks_k1_flight_deployed",
                        "networks_k1_flight_deployed_noshard",
                        "networks_k1_flight_deployed_singletons",
                        "_flight_deployed_matched_board",
                        "_flight_deployed_2frame"],
    "mb_3model": ["networks_k1_mb_3model_4hz",
                  "networks_k1_mb_3model_4hz_yolo_ctrl",
                  "_evo_og",
                  "networks_k1_mb_3model_12hz"],
    "mlp_dronet": ["networks_k1_mb", "networks_k1_mb_B1"],
    "tri": ["networks_k1_tri_exact_100ms", "networks_k1_tri_exact_100ms_feedback",
            "networks_k1_tri_small", "networks_k1_tri_small_feedback"],
    "ffn_ime": ["networks_k1_ffn_ime"],
    "multicore_shard": ["networks_k1_multicore_shard"],
}
# Broken: dispatch graphs point at a dead gen/vmfb/.../RVV/ tree.
UNRUNNABLE = {"networks_k1_mlp_dronet"}


def family_of(stem: str) -> str:
    for fam, members in FAMILIES.items():
        if stem in members:
            return fam
    return "unclassified"


CELLS = (("A", False, False, "neither"),
         ("B", True, False, "inner only (levers, deployed blind)"),
         ("C", False, True, "outer only (measured re-solve, no levers)"),
         ("D", True, True, "both"))


# Compaction is a post-pass that left-shifts a schedule. It is idempotent on tight
# solvers and slack-eliminating on list schedulers, so leaving it on systematically
# closes the gap between them -- it changes WHICH ARM WINS, not just the numbers. And
# the greedy family bypasses the registry entirely, so it never gets the post-pass at
# all: an asymmetry no manifest would show. Forced off, and recorded.
SOLVE_ENV = {"XPURT_NO_COMPACT": "1", "XPURT_CPSAT_WORKERS":
             os.environ.get("XPURT_CPSAT_WORKERS", "4")}

# A missing solver and a broken solve read very differently to whoever reads the table.
UNAVAILABLE_MARKERS = ("no module named", "modulenotfounderror", "license",
                       "msk_res_err_license", "is not installed", "you need to install")


def classify_failure(text: str) -> str:
    low = (text or "").lower()
    if "timeout" in low or "timed out" in low:
        return "timeout"
    return "unavailable" if any(m in low for m in UNAVAILABLE_MARKERS) else "error"


def net_core_times(net: str) -> dict:
    """`{n_cores: whole-net ms}` measured on the board, from the committed profiles."""
    out = {}
    for c in glob.glob(os.path.join(
            REPO, f"gen/profile_mb/*/spacemit_x60/{net}/*/*/topo_*/results.csv")):
        width = len(os.path.basename(os.path.dirname(c)).split("_")) - 1
        try:
            t = sum(float(r["mean_time"] or 0)
                    for r in csv.DictReader(open(c)))
        except Exception:
            continue
        if t > 0:
            out[width] = min(out.get(width, 9e9), t)
    return out


def feasibility(spec_obj: dict) -> dict:
    """Which of this workload's nets can meet their window AT ALL, and which cannot.

    WHY THIS IS A FIRST-CLASS OUTPUT. Scoring the loop on a deadline no schedule can
    meet measures the wrong thing. On the existing corpus 44 of the 50 residual misses
    were of this kind: yolov8_nano_64x96 needs 23.95 ms at 8 cores against a 22 ms
    window and its scaling has saturated (4->8 cores buys 1.6%), so no lever, solver or
    loop can clear it. Reporting that as a loop failure is a category error -- it is a
    COMPILER gap (yolo's fused convs do not parallelise past ~2x), and the honest claim
    is "the loop clears what is clearable".

    A net is infeasible-by-construction when its FASTEST measured implementation, across
    every profiled core width, still exceeds its window.
    """
    out = {}
    for net, info in (spec_obj.get("networks") or {}).items():
        window = float(info.get("window_duration") or 0)
        t = net_core_times(net)
        if not t or window <= 0:
            out[net] = {"verdict": "unknown", "window_ms": window or None}
            continue
        best_w = min(t, key=lambda w: t[w])
        best = t[best_w]
        out[net] = {
            "window_ms": round(window, 3),
            "best_ms": round(best, 3), "best_cores": best_w,
            "singleton_ms": round(t.get(1, best), 3),
            "verdict": "infeasible" if best > window else "achievable",
            "needs_ratio": round(best / window, 3) if window else None,
        }
    return out


def split_misses(cell: dict, feas: dict) -> dict:
    """Attribute a cell's misses to nets that COULD have met their deadline and nets
    that could not, so the loop's scorecard is not charged for the impossible."""
    ach = inf = unk = 0
    for net, n in (cell.get("misses_by_network") or {}).items():
        v = (feas.get(net) or {}).get("verdict")
        if v == "infeasible":
            inf += n
        elif v == "achievable":
            ach += n
        else:
            unk += n
    return {"achievable_misses": ach, "infeasible_misses": inf,
            "unknown_misses": unk}


def comparability_of(cells: dict) -> dict:
    """Whether the cells of one workload scheduled the SAME amount of work.

    Extracted so it can be pinned by a test. Two cells with different per-model
    instance counts are two amounts of work, not two schedules -- ranking them ranks
    different experiments, which is the trap `compare_candidates.py` refuses on and
    `feedback_benchmark` enforces per phase. A cell that never produced a schedule
    contributes no count and cannot make the set incomparable on its own.
    """
    counts = {n: (c or {}).get("instances")
              for n, c in (cells or {}).items()
              if isinstance(c, dict) and c.get("instances")}
    distinct = {json.dumps(v, sort_keys=True) for v in counts.values()}
    if len(distinct) > 1:
        return {"status": "REFUSED", "per_cell": counts,
                "why": ("cells scheduled different instance counts; that is two "
                        "amounts of work, not two schedules")}
    return {"status": "ok", "per_cell": counts}


def instances_per_model(sched_path, spec_obj):
    """`{model: n_instances}` -- two arms that scheduled different amounts of work are
    not two schedules, and ranking them ranks different experiments."""
    try:
        sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
        import schedule_scoring
        return schedule_scoring.instances_per_model(
            json.load(open(sched_path)), set((spec_obj.get("networks") or {}).keys()))
    except Exception:
        return None


def sh(cmd, timeout=None, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, env=e)


def solve(spec_path, solver, time_limit, calibration, tag, log, cpsat_limit=None):
    """Solve one spec, optionally with the measured board multipliers.

    Returns `(schedule_path, info)` where info carries the solver's own verdict --
    `solver_status` ("optimal"/"feasible"/...), `solve_wall_s` -- because the repeat
    policy keys on it: a PROVEN OPTIMAL objective is unique and needs no repeats, while
    a budget-truncated FEASIBLE one can move between runs.
    """
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
           "--profiled", "--max-periodic-iters", "1"]
    if solver == "cpsat":
        cmd += ["--solver", "milp", "--scheduler", "cpsat"]
        sfx = "cpsat_profiled"
    elif solver == "mosek":
        cmd += ["--solver", "milp", "--scheduler", "mosek"]
        sfx = "mosek_profiled"
    else:
        cmd += ["--solver", "greedy"]
        sfx = "greedy_profiled"
    if time_limit:
        cmd += ["--time-limit", str(time_limit)]
    if solver == "cpsat":
        # --time-limit is MILP-only; without this CP-SAT silently runs at its own
        # 300 s default while the other arms get the budget that was asked for.
        cmd += ["--cpsat-time-limit", str(float(cpsat_limit or time_limit or 300.0))]
    if calibration:
        cmd += ["--board-calibration", calibration]
    # THE CELL SOLVES MUST OBEY THE SAME CODEGEN CONTRACT AS THE INNER SEARCH. Cells C
    # and D re-solve a spec here rather than through run_codesign_loop, so without this
    # a cell could be scored on a schedule the compiler cannot build while the inner
    # search that produced its spec was held to the contract -- two different rulesets
    # inside one row. CP-SAT is the arm that can be constrained rather than merely
    # checked; greedy has no combination-selection variable to couple, and that
    # asymmetry is reported rather than papered over.
    env = dict(SOLVE_ENV)
    try:
        _mode = ((json.load(open(spec_path)).get("scheduler") or {})
                 .get("machine_combination_mode"))
        if solver == "cpsat" and _mode == "shard":
            env["XPURT_UNIFORM_PACKED_WIDTH"] = "1"
    except Exception:
        pass
    r = sh(cmd, env=env)
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}.json")
    if not os.path.exists(sched):
        detail = ((r.stderr or "") + (r.stdout or ""))[-300:]
        log(f"    {tag}: SOLVE FAILED — {detail}")
        return None, {"status": classify_failure(detail), "detail": detail}
    info = {"status": "ok"}
    rep_path = sched.replace(".json", "_report.json")
    if os.path.exists(rep_path):
        try:
            rj = json.load(open(rep_path))
            info.update(solver_status=rj.get("solver_status"),
                        solve_wall_s=rj.get("solve_wall_s"),
                        n_operations=rj.get("n_operations"))
        except Exception:
            pass
    return sched, info


def recost(sched, spec_path, calibration, out, log):
    """Re-cost a schedule under measured board costs with its assignment held fixed.

    This is what deploying a predicted-cost schedule means: the placement is already
    decided, and every dispatch takes what the board makes it take.
    """
    r = sh([PY, "scripts/recost_schedule_on_board.py", "--schedule", sched,
            "--spec", spec_path, "--calibration", calibration, "--out", out])
    if r.returncode != 0 or not os.path.exists(out):
        log(f"    recost failed: {((r.stderr or '') + (r.stdout or ''))[-300:]}")
        return None
    return out


def inner_search(workload, args, solver, out_dir, log):
    """Run the AOT search (levers, and with --rewrite-arm the ModelBlaster rewrites);
    return the spec it converged on."""
    stem = os.path.splitext(os.path.basename(workload))[0]
    loop_out = os.path.join(out_dir, "inner", f"{stem}__{solver}")
    cmd = [PY, "scripts/run_codesign_loop.py", "--workload", workload,
           "--solver", ("cpsat" if solver in ("cpsat", "mosek") else "greedy"),
           "--max-rounds", str(args.max_rounds),
           "--objective", args.objective, "--out-dir", loop_out]
    if args.time_limit:
        # run_codesign_loop's --time-limit is an int (CP-SAT seconds); passing 20.0
        # makes argparse reject it, and the whole arm then reports as "no report".
        cmd += ["--time-limit", str(int(args.time_limit))]
    if args.replay:
        cmd += ["--replay"]
    if args.rewrite_arm and args.ir:
        cmd += ["--rewrite-arm", "--runner", args.runner]
        for spec in args.ir:
            cmd += ["--ir", spec]
    # SAME ENV AS THE CELLS. The inner search solves too, and a search run under
    # different switches than the cells it feeds is a different experiment -- it was
    # inheriting the ambient environment while only the cell solves got SOLVE_ENV.
    # Harmless while XPURT_COMPACT is unset (compaction defaults off), but the point of
    # forcing a switch is that it does not depend on what the shell happened to hold.
    r = sh(cmd, timeout=args.timeout, env=SOLVE_ENV)
    rep_paths = glob.glob(os.path.join(loop_out, "*", "loop_report.json"))
    if not rep_paths:
        log(f"    inner search produced no report "
            f"({((r.stderr or '') + (r.stdout or ''))[-200:]})")
        return None, []
    rep = json.load(open(rep_paths[0]))
    levers = rep.get("levers_applied") or []
    # The loop writes every candidate spec it solved; the accepted one for the last
    # round is the converged spec. Prefer the recorded path, fall back to the newest.
    specs = sorted(glob.glob(os.path.join(loop_out, "*", "specs", "*.json")),
                   key=os.path.getmtime)
    final = None
    if levers:
        want = f"_{levers[-1]}.json"
        for p in reversed(specs):
            if p.endswith(want):
                final = p
                break
    if final is None:
        final = next((p for p in specs if p.endswith("_r0_baseline.json")), None)
    return final, levers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", required=True)
    ap.add_argument("--calibration",
                    default="results/codesign_feedback/k1_board_calibration.json")
    ap.add_argument("--solvers", default="cpsat,greedy",
                    help="comma-separated solver arms, e.g. cpsat,mosek,greedy. The "
                         "exact solvers come first so the figure reads as 'and here is "
                         "greedy', not the reverse. MOSEK is included only where it "
                         "converges; where it does not, that is reported, not hidden.")
    ap.add_argument("--rewrite-arm", action="store_true",
                    help="the FULL inner loop: let the offline search also propose "
                         "ModelBlaster graph rewrites, not only scheduling levers")
    ap.add_argument("--ir", action="append", default=[], metavar="NET=PATH",
                    help="IRs for --rewrite-arm, passed through to the loop")
    ap.add_argument("--runner", choices=["board", "none"], default="board")
    ap.add_argument("--cpsat-time-limit", type=float, default=900.0,
                    help="seconds CP-SAT may search per solve (default 900). Its cost "
                         "is set by this, not by the instance.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats for a solve whose result can MOVE. A CP-SAT solve "
                         "that returns OPTIMAL has a unique objective and is run once; "
                         "one that returns FEASIBLE is budget-truncated and is repeated "
                         "this many times, reported as median with the spread. Greedy "
                         "is deterministic and always runs once.")
    ap.add_argument("--one-per-family", action="store_true",
                    help="run only the first listed member of each family. The 25 K1 "
                         "specs contain byte-identical duplicates and variants that "
                         "return bit-identical results; the family is the real unit.")
    ap.add_argument("--objective", default="auto")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--max-rounds", type=int, default=3)
    # DEFAULT OFF, and this is a fairness fix rather than a preference. --replay makes
    # run_codesign_loop pin XPURT_CPSAT_WORKERS=1 for bit-exact reruns, while the CELL
    # solves here run at SOLVE_ENV's 4 workers. That put two different CP-SAT
    # configurations inside ONE row: cell B (inner only) came from a 1-worker search and
    # cells C/D from 4-worker re-solves, so inner-vs-outer was partly a comparison of
    # worker counts. The gap is not subtle -- on w3's shard spec at the same 300 s
    # budget, 1 worker gave 86.4 ms / 72 dispatch misses where 4 gave 72.3 ms / 4 -- so
    # with replay on, the inner arm was handicapped into rejecting the very lever that
    # halves misses, and the figure would have reported that as the inner loop failing.
    #
    # The repo's position is already written down: the CP-SAT experiment is reproducible,
    # its result is not, and the published numbers use more than one worker. Repeats with
    # a reported spread are how that is handled here.
    ap.add_argument("--replay", action="store_true", default=False,
                    help="pin the inner search to 1 CP-SAT worker for bit-exact reruns. "
                         "Off by default: the cell solves use XPURT_CPSAT_WORKERS (4), "
                         "and mixing the two inside one row confounds inner-vs-outer "
                         "with worker count.")
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out-dir", default="results/loop_ablation")
    a = ap.parse_args()

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cal = a.calibration if os.path.isabs(a.calibration) else os.path.join(
        REPO, a.calibration)
    if not os.path.exists(cal):
        print(f"no calibration at {cal}", file=sys.stderr)
        return 2
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    solvers = [x.strip() for x in a.solvers.split(",") if x.strip()]
    wl = []
    seen_fam = set()
    for w in a.workloads:
        st = os.path.splitext(os.path.basename(w))[0]
        if st in UNRUNNABLE:
            log(f"skipping {st}: its dispatch graphs point at a tree that no longer "
                f"exists; it cannot be solved by any arm")
            continue
        fam = family_of(st)
        if a.one_per_family and fam in seen_fam:
            continue
        seen_fam.add(fam)
        wl.append(w)
    a.workloads = wl
    rows = []
    for i, w in enumerate(a.workloads, 1):
        stem = os.path.splitext(os.path.basename(w))[0]
        log(f"[{i}/{len(a.workloads)}] {stem}")
        base_spec = w if os.path.isabs(w) else os.path.join(REPO, w)
        per_solver = {}
        for solver in solvers:
            t0 = time.time()
            log(f"  -- {solver} --")
            opt_spec, levers = inner_search(w, a, solver, out_dir, log)
            if opt_spec is None:
                per_solver[solver] = {"status": "inner_search_failed"}
                log(f"    {solver}: inner search failed — this arm contributes NO "
                    f"cells; it is counted as a failure, not as zeros")
                continue
            log(f"    inner (AOT) search: levers {levers or '(none)'}")
            spec_for = {False: base_spec, True: opt_spec}
            feas = feasibility(json.load(open(base_spec)))
            cells = {}
            for name, inner, outer, label in CELLS:
                sp = spec_for[inner]
                cell = {"label": label, "inner": inner, "outer": outer,
                        "spec": os.path.relpath(sp, REPO)}
                if inner and not levers:
                    # No lever was accepted, so B and D ARE A and C. Say so rather than
                    # presenting the same schedule twice as if it were evidence.
                    cell["note"] = ("no lever accepted; identical to the inner-off "
                                    "cell by construction")
                sched, info = solve(sp, solver, a.time_limit,
                                    cal if outer else None, f"{solver}/{name}", log,
                                    cpsat_limit=a.cpsat_time_limit)
                if sched is None:
                    cell.update(status=info.get("status", "solve_failed"),
                                detail=info.get("detail", "")[:300])
                    cells[name] = cell
                    continue
                cell["solver_status"] = info.get("solver_status")
                cell["solve_wall_s"] = info.get("solve_wall_s")
                # REPEAT ONLY WHERE THE RESULT CAN MOVE. A proven-optimal objective is
                # unique; a truncated one is whatever incumbent the wall clock caught.
                needs_repeat = (solver == "cpsat"
                                and str(info.get("solver_status", "")).lower()
                                not in ("optimal", "proven_optimal"))
                cell["repeated"] = bool(needs_repeat)
                spec_obj = json.load(open(sp))
                if outer:
                    scored = sched
                    cell["scored_how"] = "solved on measured costs"
                else:
                    out = os.path.join(out_dir,
                                       f"{stem}__{solver}_{name}_recost.json")
                    scored = recost(sched, sp, cal, out, log)
                    cell["scored_how"] = ("solved on predicted costs, then re-cost "
                                          "on measured with the assignment fixed")
                    if scored is None:
                        cell["status"] = "recost_failed"
                        cells[name] = cell
                        continue
                cell.update(schedule=os.path.relpath(scored, REPO),
                            **schedule_eval.summary(scored, spec_obj))
                # Comparability: every cell of a workload must schedule the same amount
                # of work, or the comparison is between two workloads.
                cell["instances"] = instances_per_model(scored, spec_obj)
                cell.update(split_misses(cell, feas))
                if needs_repeat and a.repeats > 1:
                    misses = [cell["instance_misses"]]
                    lates = [cell["worst_lateness_ms"]]
                    for k in range(2, a.repeats + 1):
                        s2, i2 = solve(sp, solver, a.time_limit,
                                       cal if outer else None,
                                       f"{solver}/{name}#{k}", log,
                                       cpsat_limit=a.cpsat_time_limit)
                        if s2 is None:
                            continue
                        if outer:
                            sc2 = s2
                        else:
                            o2 = os.path.join(out_dir,
                                              f"{stem}__{solver}_{name}_r{k}_recost.json")
                            sc2 = recost(s2, sp, cal, o2, log)
                            if sc2 is None:
                                continue
                        su = schedule_eval.summary(sc2, spec_obj)
                        misses.append(su["instance_misses"])
                        lates.append(su["worst_lateness_ms"])
                    ok_m = [x for x in misses if isinstance(x, (int, float))]
                    ok_l = [x for x in lates if isinstance(x, (int, float))]
                    if ok_m:
                        cell["instance_misses"] = statistics.median(ok_m)
                        cell["instance_misses_runs"] = ok_m
                        cell["instance_misses_spread"] = max(ok_m) - min(ok_m)
                    if ok_l:
                        cell["worst_lateness_ms"] = statistics.median(ok_l)
                        cell["worst_lateness_runs"] = [round(x, 3) for x in ok_l]
                        cell["worst_lateness_spread_ms"] = round(max(ok_l) - min(ok_l), 3)
                    log(f"    {name} repeated x{len(ok_m)} (status "
                        f"{cell.get('solver_status')}): misses {ok_m}, "
                        f"spread {cell.get('worst_lateness_spread_ms')} ms")
                cells[name] = cell
                log(f"    {name} {label:<42} misses={str(cell['instance_misses']):<4}"
                    f"worst_late={(cell['worst_lateness_ms'] or 0):8.3f} ms  "
                    f"makespan={(cell['makespan_ms'] or 0):7.2f} ms")
            per_solver[solver] = {"status": "ok", "levers": levers, "cells": cells,
                                  "seconds": round(time.time() - t0, 1)}
        # Comparability across the cells of one workload: same amount of work, or the
        # comparison is between two workloads rather than two schedulers.
        for solver, blob in per_solver.items():
            verdict = comparability_of(blob.get("cells") or {})
            blob["comparability"] = verdict
            if verdict["status"] == "REFUSED":
                log(f"    !! {solver}: COMPARABILITY REFUSED — {verdict['per_cell']}")
        rows.append({"workload": w, "family": family_of(stem),
                     "feasibility": feasibility(json.load(open(base_spec))),
                     "solvers": per_solver})

    # ---- aggregate -----------------------------------------------------------
    def cell_vals(solver, cell, key):
        out = []
        for r in rows:
            c = (((r["solvers"].get(solver) or {}).get("cells") or {}).get(cell) or {})
            v = c.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
        return out

    agg = {}
    for solver in solvers:
        agg[solver] = {}
        for name, inner, outer, label in CELLS:
            m = cell_vals(solver, name, "instance_misses")
            wl = cell_vals(solver, name, "worst_lateness_ms")
            mk = cell_vals(solver, name, "makespan_ms")
            agg[solver][name] = {
                "label": label, "n": len(m),
                "workloads_with_zero_misses": sum(1 for x in m if x == 0),
                "total_instance_misses": sum(m),
                "median_instance_misses": statistics.median(m) if m else None,
                "median_worst_lateness_ms": statistics.median(wl) if wl else None,
                "median_makespan_ms": statistics.median(mk) if mk else None,
            }

    # Where only the PAIR clears the misses -- the claim neither loop alone supports.
    only_both = {}
    for solver in solvers:
        names = []
        for r in rows:
            c = ((r["solvers"].get(solver) or {}).get("cells") or {})
            try:
                if (c["D"]["instance_misses"] == 0 and c["B"]["instance_misses"] > 0
                        and c["C"]["instance_misses"] > 0):
                    names.append(os.path.basename(r["workload"]))
            except (KeyError, TypeError):
                continue
        only_both[solver] = names

    # And where the exact solver beats greedy in the SAME cell.
    beats_greedy = {}
    if "greedy" in solvers:
        for solver in [s2 for s2 in solvers if s2 != "greedy"]:
            per_cell = {}
            for name, _i, _o, _l in CELLS:
                wins = ties = losses = 0
                for r in rows:
                    g = (((r["solvers"].get("greedy") or {}).get("cells") or {})
                         .get(name) or {})
                    e = (((r["solvers"].get(solver) or {}).get("cells") or {})
                         .get(name) or {})
                    gm, em = g.get("instance_misses"), e.get("instance_misses")
                    if not isinstance(gm, int) or not isinstance(em, int):
                        continue
                    if em < gm:
                        wins += 1
                    elif em == gm:
                        gl = g.get("worst_lateness_ms")
                        el = e.get("worst_lateness_ms")
                        if isinstance(gl, float) and isinstance(el, float):
                            if el < gl - 1e-9:
                                wins += 1
                            elif el > gl + 1e-9:
                                losses += 1
                            else:
                                ties += 1
                        else:
                            ties += 1
                    else:
                        losses += 1
                per_cell[name] = {"wins": wins, "ties": ties, "losses": losses}
            beats_greedy[solver] = per_cell

    # PRE-REGISTERED STRATUM: a workload is "at stake" when the naive deployment (cell
    # A) actually misses on board costs. Declared here rather than chosen after looking
    # at the results, because a stratum picked afterwards is indistinguishable from
    # cherry-picking. Workloads with nothing at stake are reported, not dropped.
    strata = {}
    for r in rows:
        a_cells = [((r["solvers"].get(sv) or {}).get("cells") or {}).get("A") or {}
                   for sv in solvers]
        misses = [c.get("instance_misses") for c in a_cells
                  if isinstance(c.get("instance_misses"), (int, float))]
        strata[os.path.basename(r["workload"])] = (
            "at_stake" if misses and max(misses) > 0 else "nothing_at_stake"
            if misses else "unknown")
    at_stake = [k for k, v in strata.items() if v == "at_stake"]

    def agg_over(subset_names, solver, cell, key):
        out = []
        for r in rows:
            if os.path.basename(r["workload"]) not in subset_names:
                continue
            c = (((r["solvers"].get(solver) or {}).get("cells") or {}).get(cell) or {})
            v = c.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
        return out

    # The loop's REAL scorecard: misses it could have cleared. An infeasible miss is a
    # compiler gap (an op that does not parallelise far enough), not a scheduling one.
    agg_achievable = {}
    for solver in solvers:
        agg_achievable[solver] = {}
        for name, inner, outer, label in CELLS:
            ach = agg_over(at_stake, solver, name, "achievable_misses")
            inf = agg_over(at_stake, solver, name, "infeasible_misses")
            agg_achievable[solver][name] = {
                "label": label, "n": len(ach),
                "total_achievable_misses": sum(ach),
                "total_infeasible_misses": sum(inf),
                "workloads_with_zero_achievable_misses": sum(1 for x in ach if x == 0),
            }

    agg_at_stake = {}
    for solver in solvers:
        agg_at_stake[solver] = {}
        for name, inner, outer, label in CELLS:
            m = agg_over(at_stake, solver, name, "instance_misses")
            wl2 = agg_over(at_stake, solver, name, "worst_lateness_ms")
            agg_at_stake[solver][name] = {
                "label": label, "n": len(m),
                "workloads_with_zero_misses": sum(1 for x in m if x == 0),
                "total_instance_misses": sum(m),
                "median_instance_misses": statistics.median(m) if m else None,
                "median_worst_lateness_ms": statistics.median(wl2) if wl2 else None,
            }

    summary = {
        "schema": "loop_ablation/v3",
        "population": {
            "families": {f: FAMILIES[f] for f in FAMILIES},
            "unrunnable": sorted(UNRUNNABLE),
            "one_per_family": bool(a.one_per_family),
            "strata": strata,
            "at_stake": at_stake,
            "note": ("the 25 K1 specs contain byte-identical duplicates and variants "
                     "that return bit-identical results; the family is the unit"),
        },
        "aggregate_at_stake_only": agg_at_stake,
        "aggregate_achievable_only": agg_achievable,
        "achievable_means": ("a miss is ACHIEVABLE when the net's fastest measured "
                             "implementation fits its window; otherwise no schedule "
                             "can meet that deadline and the miss is a compiler gap, "
                             "not a scheduling one"),
        "cpsat_time_limit_s": a.cpsat_time_limit,
        "repeats_policy": ("repeat only a CP-SAT solve that did not prove optimality; "
                           "greedy is deterministic"),
        "solve_env": SOLVE_ENV,
        "cpsat_workers": {
            "cells": SOLVE_ENV.get("XPURT_CPSAT_WORKERS"),
            "inner_search": ("1 (--replay pins it)" if a.replay
                             else SOLVE_ENV.get("XPURT_CPSAT_WORKERS")),
            "why_recorded": ("these were once different -- 1 for the inner search under "
                             "--replay, 4 for the cells -- which made cell B and cells "
                             "C/D two different CP-SAT configurations inside one row"),
        },
        "codegen_contract": ("cpsat shard solves add XPURT_UNIFORM_PACKED_WIDTH=1 so a "
                             "packed-weight dispatch takes one width across its "
                             "instances; greedy cannot be constrained that way and its "
                             "unbuildable candidates are rejected by the inner search "
                             "instead"),
        "inner_means": ("AOT co-design with ModelBlaster: graph rewrites and the "
                        "per-dispatch implementation choice, decided offline against "
                        "isolated per-dispatch profiles"),
        "outer_means": ("HIL: measured multipliers from real-time runs of the whole "
                        "schedule, returned and re-solved against"),
        "cells": {n: l for n, _i, _o, l in CELLS},
        "scoring": ("instance-level misses on BOARD costs for every cell; A/B are "
                    "solved on predicted costs then re-cost with the assignment fixed, "
                    "C/D are solved with --board-calibration"),
        "solvers": solvers, "rewrite_arm": bool(a.rewrite_arm and a.ir),
        "calibration": os.path.relpath(cal, REPO),
        "n_workloads": len(rows),
        "aggregate": agg,
        "workloads_only_both_clears": only_both,
        "exact_vs_greedy_per_cell": beats_greedy,
        "runs": rows,
    }
    json.dump(summary, open(os.path.join(out_dir, "ablation_summary.json"), "w"),
              indent=1)
    open(os.path.join(out_dir, "ablation.log"), "w").write("\n".join(lines) + "\n")

    log(f"\n=== aggregate over {len(rows)} workload(s), scored on board costs ===")
    for solver in solvers:
        n_ok = sum(1 for r in rows
                   if (r["solvers"].get(solver) or {}).get("status") == "ok")
        log(f"\n[{solver}]  {n_ok}/{len(rows)} workload(s) produced cells")
        log(f"  {'cell':<5}{'inner':<7}{'outer':<7}{'0-miss':<9}{'tot miss':<10}"
            f"{'med worst late':<16}{'med makespan':<13}")
        for name, inner, outer, label in CELLS:
            g = agg[solver][name]
            log(f"  {name:<5}{str(inner):<7}{str(outer):<7}"
                f"{str(g['workloads_with_zero_misses']) + '/' + str(g['n']):<9}"
                f"{g['total_instance_misses']:<10}"
                f"{(g['median_worst_lateness_ms'] or 0):<16.3f}"
                f"{(g['median_makespan_ms'] or 0):<13.2f}")
        if only_both.get(solver):
            log(f"  only the PAIR clears: {only_both[solver]}")
    for solver, per_cell in beats_greedy.items():
        log(f"\n[{solver} vs greedy, same cell, same workload]")
        for name, _i, _o, _l in CELLS:
            c = per_cell[name]
            log(f"  {name}: {c['wins']} win / {c['ties']} tie / {c['losses']} loss")
    log(f"\n=== AT-STAKE ONLY ({len(at_stake)} workload(s) whose cell A misses) ===")
    for solver in solvers:
        log(f"\n[{solver}] at-stake")
        log(f"  {'cell':<5}{'inner':<7}{'outer':<7}{'0-miss':<9}{'tot miss':<10}"
            f"{'med worst late':<16}")
        for name, inner, outer, label in CELLS:
            g = agg_at_stake[solver][name]
            log(f"  {name:<5}{str(inner):<7}{str(outer):<7}"
                f"{str(g['workloads_with_zero_misses']) + '/' + str(g['n']):<9}"
                f"{g['total_instance_misses']:<10}"
                f"{(g['median_worst_lateness_ms'] or 0):<16.3f}")
    log(f"\n=== THE LOOP'S REAL SCORECARD (achievable misses only) ===")
    for solver in solvers:
        log(f"\n[{solver}] achievable-only")
        log(f"  {'cell':<5}{'inner':<7}{'outer':<7}{'0-achv':<9}"
            f"{'achievable':<12}{'infeasible':<12}")
        for name, inner, outer, label in CELLS:
            g = agg_achievable[solver][name]
            log(f"  {name:<5}{str(inner):<7}{str(outer):<7}"
                f"{str(g['workloads_with_zero_achievable_misses']) + '/' + str(g['n']):<9}"
                f"{g['total_achievable_misses']:<12}{g['total_infeasible_misses']:<12}")
    log(f"\nnothing at stake: "
        f"{[k for k, v in strata.items() if v == 'nothing_at_stake']}")
    log(f"\nsummary: {os.path.relpath(os.path.join(out_dir, 'ablation_summary.json'), REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
