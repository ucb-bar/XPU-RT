#!/usr/bin/env python3
"""Fully-automatic ModelBlaster<->XPU-RT co-design feedback loop.

ONE command. Starts from a workload spec at a clean baseline (no levers), then each
round proposes every not-yet-applied lever as a candidate, SOLVES each candidate with
the real profiled scheduler, and ACCEPTS on the project's own acceptance rule —
`candidate_objective.accept()`, nine lexicographic terms with hard deadline misses
FIRST and makespan SEVENTH, each with its own noise tolerance. Among the candidates
that rule accepts, the declared objective picks which one to take. Iterates until no
lever is accepted (converged). The loop's intelligence is the advisor/cost-model; this
driver applies, measures, and accepts — honestly (a lever that does not help is
rejected, with the deciding term recorded).

The two-term rule (`misses_not_worse AND objective_delta > 0.05 ms`) is available behind
`--accept-rule legacy` only to reproduce pre-hardening runs. It can disagree with the
nine-term rule: results/codesign_loop/_ime_yolo_c0only/loop_report.json records an ACCEPT
of the ime lever with 93 deadline misses on both sides, won on makespan — term 7 of 9.

Levers:
  * ime   — expose the K1 IME matrix engine as a per-dispatch alternative. For every
            CONV-bearing net that lacks an ime_x60 profile, auto-build one from the
            measured conv speedups (scripts/make_ime_profile.py, which scales mean_time
            — the column the loader reads), then set scheduler.enable_impls=true. The
            scheduler picks IME per dispatch only where measured faster.
  * shard — expose 2/4/8-hart implementations (scheduler.machine_combination_mode=shard,
            topo_tag_override=false).
  * fuse  — the roofline decision-aid (xpu-rt/data/fusion_benefit.csv) is consulted and
            REPORTED, not applied: our stack is compute-bound, so fusion is a scheduling
            (dispatch-collapse) move, not a cycle win — the loop honestly does not credit
            a makespan gain it cannot measure.

Usage:
  scripts/run_codesign_loop.py --workload data/toplevel/<spec>.json [--max-rounds 4]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Interpreter used for the child scheduler runs: $XPURT_PY, else the repo-root venv, else this interpreter.
_venv_py = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv_py if os.path.exists(_venv_py) else sys.executable)
EPS = 0.05  # ms; a smaller "improvement" is noise, not a win

sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
try:
    import exact_cycle  # objective-aware acceptance for exact_cycle workloads
except Exception:
    exact_cycle = None
try:
    import rewrite_arm
except Exception:  # pragma: no cover - the scheduling-lever loop works without it
    rewrite_arm = None
try:
    import codegen_contract
except Exception:  # pragma: no cover - the loop still runs, it just cannot gate
    codegen_contract = None
try:
    import candidate_objective as objective
    import schedule_scoring
    import workload_spec
except Exception:  # pragma: no cover - the legacy rule still works without them
    objective = schedule_scoring = workload_spec = None


def critical_and_heavy(spec: dict, critical_arg=None, heavy_arg=None):
    """Which nets carry the hard-deadline term, and which one is the heavy perception net.

    Default: EVERY deadline-bearing net is critical. That keeps term 1 exactly as strict as
    this loop's previous instance-miss guard while adding the eight terms the guard ignored,
    so switching to the real rule can only ever reject more, never less. The heavy net
    defaults to the longest-cadence one -- the perception head, whose max latency and
    throughput are terms 5 and 6."""
    nets = spec.get("networks") or {}
    if critical_arg:
        crit = tuple(m.strip() for m in str(critical_arg).split(",") if m.strip())
    else:
        crit = tuple(n for n, v in nets.items()
                     if float(v.get("window_duration") or 0) > 0)
    heavy = heavy_arg
    if heavy is None and nets:
        heavy = max(nets, key=lambda n: float(nets[n].get("period") or 0))
    return crit, heavy


def outcome_of(label, sched_path, spec, critical, heavy):
    """A CandidateOutcome for one solved schedule, built exactly as compare_candidates.py
    builds it -- the project's stated acceptance inputs, not a single scalar."""
    if schedule_scoring is None:
        return None
    sched = json.load(open(sched_path))
    windows, known = workload_spec.windows_and_names(spec)
    periods = workload_spec.periods_ms(spec)
    _, out, _ = schedule_scoring.score(label, sched, windows, critical, heavy,
                                       known, periods)
    return out


def objective_verdict(cand_out, base_out, cand_sched_path, base_sched_path, spec):
    """(ok, why) from `candidate_objective.accept()` -- nine lexicographic terms, hard
    deadline misses FIRST and makespan SEVENTH, each with its own noise tolerance.

    WHY. A two-term rule of the driver's own,
    `misses_not_worse AND objective_delta > 0.05ms`, is not the project's rule and
    disagrees with it: results/codesign_loop/_ime_yolo_c0only/loop_report.json records an
    ACCEPT of the ime lever with 93 deadline misses on both sides, won on makespan -- the
    term the real rule ranks seventh. A tie is a rejection, and unequal instance counts are
    refused rather than judged, because two different amounts of work are not two graphs."""
    if objective is None or cand_out is None or base_out is None:
        return None, "objective rule unavailable"
    known = set((spec.get("networks") or {}).keys())
    bi = schedule_scoring.instances_per_model(json.load(open(base_sched_path)), known)
    ci = schedule_scoring.instances_per_model(json.load(open(cand_sched_path)), known)
    if bi != ci:
        return False, (f"refused -- instance counts differ ({bi} vs {ci}); that is two "
                       f"amounts of work, not two graphs")
    # DETERMINISTIC comparison: both sides are analytic evaluations of solved schedules
    # against fixed costs, not repeated executions, so the miss counts carry no jitter
    # for a tolerance to absorb. With the measurement tolerance the search converged a
    # lever short on w4 and b4 -- 5 -> 3 misses called "indistinguishable" because 8% of
    # 34 instances is 2.72. See candidate_objective.DETERMINISTIC_TOLERANCES.
    return objective.accept(cand_out, base_out,
                            tol=objective.DETERMINISTIC_TOLERANCES)


def buildable(sched_path, log, label):
    """`(ok, why)` -- is this candidate's schedule something ModelBlaster can BUILD?

    A LEVER THAT WINS ON PAPER AND CANNOT BE COMPILED IS NOT A WIN. `shard` mode lets
    every periodic instance of a dispatch pick its own core width, which for a packed
    convolution is unbuildable -- one generated model cannot carry two weight layouts
    for one dispatch. Without this gate the loop happily accepted such a candidate,
    reported the improvement, and the board build then died at stage 1 of 5 with an
    error from inside a shell script. The contract
    (`ModelBlaster/cores/codegen_contract.json`) is what makes the constraint visible on
    this side, and checking it costs milliseconds.

    A missing contract does NOT fail a candidate: refusing everything because the
    submodule is absent would be worse than not checking. It is reported once, so the
    run says which mode it was in.
    """
    if codegen_contract is None or not sched_path:
        return True, "codegen contract unavailable -- NOT gated"
    try:
        vs = codegen_contract.violations(sched_path)
    except codegen_contract.ContractUnavailable as e:
        return True, f"codegen contract unavailable ({e}) -- NOT gated"
    except Exception as e:  # a checker bug must not silently reject every candidate
        return True, f"codegen contract check errored ({type(e).__name__}) -- NOT gated"
    refuse = [v for v in vs if v.get("severity") == "refuse"]
    if not refuse:
        return True, "buildable"
    first = codegen_contract.describe(refuse[0])
    return False, (f"NOT BUILDABLE ({len(refuse)} contract violation"
                   f"{'s' if len(refuse) > 1 else ''}): {first}")


def objective_of(spec: dict) -> str:
    """Which metric this workload is really optimizing. A tri-exact-style workload
    declares `exact_cycle_worst_response`, whose win is WORST CRITICAL RESPONSE, not
    makespan — so a makespan-only loop would (correctly, but uselessly) credit its
    shard/IME levers nothing. Accept on the objective the user actually declared."""
    om = spec.get("scheduler", {}).get("objective_mode") or spec.get("objective_mode")
    return "worst_response" if om == "exact_cycle_worst_response" else "makespan"


def worst_response_ms(sched_path: str, spec: dict):
    """Worst critical response (ms) of a schedule, via the exact-cycle assessor.
    None if unavailable — the caller then falls back to makespan."""
    if exact_cycle is None or not sched_path or not os.path.exists(sched_path):
        return None
    try:
        sch = json.load(open(sched_path))
        crit = (spec.get("critical_models") or spec.get("scheduler", {}).get("critical_models") or [])
        heavy = spec.get("heavy_model") or spec.get("scheduler", {}).get("heavy_model")
        obj = exact_cycle.assess_schedule(sch, spec, crit, heavy)
        return float(obj["objective"]["worst_critical_response_ms"])
    except Exception:
        return None


def _run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=e)


def _rvv_profile(net, variant, hw="rvv_x60"):
    return os.path.join(
        REPO, f"gen/profile_mb/{hw}/spacemit_x60/{net}/{net}.{variant}/"
        f"{net}_spacemit_x60_{hw}_{net}.{variant}/topo_0/results.csv")


def _net_variant(deps: str, key: str):
    m = re.search(r"/vmfb/([^/]+)/", deps)
    net = m.group(1) if m else key
    tail = deps.split("/")[-1].replace("_dispatch_graph.json", "")
    variant = tail[len(net) + 1:] if tail.startswith(net + ".") else "int8"
    return net, variant


def _is_conv_net(net, variant):
    p = _rvv_profile(net, variant)
    if not os.path.exists(p):
        return False
    return any(r.get("op", "").startswith("conv2d") for r in csv.DictReader(open(p)))


def solve(spec_path, solver="greedy", board_cal=None, time_limit=None):
    """Run the profiled scheduler; return (makespan_ms, op_miss, sched_json, err).

    solver     — "greedy" (fast, default) or "cpsat" (CP-SAT MILP; needs an ortools
                 interpreter — run the driver under one, and bound XPURT_CPSAT_WORKERS).
    board_cal  — None/False: isolated profile costs (the AOT view). True or a path:
                 pass --board-calibration so profile_loader scales every op by the
                 measured K1 actual/predicted ratio (the run-honest view).
    time_limit — per-solve CP-SAT seconds (ignored by greedy)."""
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
           "--profiled", "--max-periodic-iters", "1"]
    if solver == "cpsat":
        cmd += ["--solver", "milp", "--scheduler", "cpsat"]
        sfx = "cpsat_profiled"
    else:
        cmd += ["--solver", "greedy"]
        sfx = "greedy_profiled"
    if time_limit is not None:
        # THE BUDGET TRAP. --time-limit is MILP-only; CP-SAT reads --cpsat-time-limit
        # and otherwise runs at scheduler.cpsat_time_limit (300 s default). Passing only
        # --time-limit to a cpsat arm therefore gives it 300 s while every other arm
        # gets what was asked, which makes any budget-matched comparison a fiction.
        cmd += ["--time-limit", str(time_limit)]
        if solver == "cpsat":
            cmd += ["--cpsat-time-limit", str(float(time_limit))]
    if board_cal:
        cmd += ["--board-calibration"] + ([board_cal] if isinstance(board_cal, str) else [])
    # CONSTRAIN RATHER THAN REJECT, where the solver can be constrained. When the
    # candidate uses shard mode, a packed-weight (convolution) dispatch must take one
    # core width across its periodic instances or ModelBlaster cannot generate it. CP-SAT
    # can express that -- a per-(dispatch, width) indicator linked to its
    # combination-presence variables, so the solver still CHOOSES the width and simply
    # has to choose one -- and asking for it here is what makes the `shard` lever's
    # output deployable instead of merely promising.
    #
    # Greedy has no combination-selection variable to couple, so for that arm the
    # contract can only be checked afterwards and the candidate rejected. The asymmetry
    # is real and is left visible rather than hidden: it understates greedy on exactly
    # the workloads where sharding is the answer.
    env = None
    try:
        _spec = json.load(open(spec_path))
        _mode = ((_spec.get("scheduler") or {}).get("machine_combination_mode"))
        if _mode == "shard":
            # BOTH ARMS, not just CP-SAT. This was cpsat-only because the constraint it
            # sets was a CP-SAT constraint; greedy's candidate was checked afterwards
            # and thrown away instead. The comment above predicted the cost of that
            # ("it understates greedy on exactly the workloads where sharding is the
            # answer") and w4_ffn_dronet_sensor is that workload: greedy's shard
            # schedule takes its misses from 10 to 5 and its worst lateness from
            # 17.95 ms to 3.90, and it was discarded over three dronet dispatches.
            # `codegen_contract.pin_uniform_widths` now gives the list scheduler the
            # same guarantee by pricing the losing widths out, so the flag means the
            # same thing to both arms.
            env = {"XPURT_UNIFORM_PACKED_WIDTH": "1"}
    except Exception:
        pass
    r = _run(cmd, env=env)
    metrics = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}_metrics.json")
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}.json")
    if not os.path.exists(metrics):
        return None, None, None, (r.stdout + r.stderr)[-800:]
    m = json.load(open(metrics))
    return float(m["makespan_ms"]), int(m["op_deadline_miss_count"]), sched, None


def instance_misses(sched_path, spec):
    """Count INSTANCE-level deadline misses in a schedule (a net-instance misses if its
    LATEST dispatch ends past its instance deadline = inst*period + window). This mirrors
    the evolution renderer's source-of-truth miss logic exactly, so the loop decides the
    board-feedback arm on the same number the figure will show — not on the per-dispatch
    op_deadline_miss_count the schedulers report. Returns (count, {net: count})."""
    try:
        from job_names import split_job_name
        sch = json.load(open(sched_path))["dispatches"]
        nets = spec["networks"] if isinstance(spec, dict) else json.load(open(spec))["networks"]
        known = set(nets)
        per = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
        win = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
        last = {}
        for d in sch.values():
            net, inst = split_job_name(d["job_name"], known)
            if not (net in per and per[net]):
                continue
            e = float(d["start_time"]) + float(d["duration"])
            last[(net, inst)] = max(last.get((net, inst), 0.0), e)
        miss, by = 0, {}
        for (net, inst), e in last.items():
            if win[net] > 0 and e > inst * per[net] + win[net] + 1e-6:
                miss += 1
                by[net] = by.get(net, 0) + 1
        return miss, by
    except Exception as e:
        return None, {"error": str(e)}


def mk_of(sched_path):
    """Makespan (ms) from a schedule's _metrics.json sidecar; 0.0 if unavailable."""
    try:
        m = json.load(open(sched_path.replace(".json", "_metrics.json")))
        return float(m.get("makespan_ms", m.get("makespan", 0.0)) or 0.0)
    except Exception:
        return 0.0


def total_lateness(sched_path, spec):
    """Sum over net-instances of max(0, last_dispatch_end - instance_deadline), in ms — the
    real-time-correct 'how badly do we miss' objective. Zero iff every periodic instance meets
    its deadline. Same instance-level (last-dispatch-per-instance) accounting as instance_misses,
    so a lever that pulls an instance in ahead of its deadline is credited even when it leaves the
    makespan critical path (and thus the makespan) unchanged — which is exactly how IME helps."""
    try:
        from job_names import split_job_name
        sch = json.load(open(sched_path))["dispatches"]
        nets = spec["networks"] if isinstance(spec, dict) else json.load(open(spec))["networks"]
        known = set(nets)
        per = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
        win = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
        last = {}
        for d in sch.values():
            net, inst = split_job_name(d["job_name"], known)
            if not (net in per and per[net]):
                continue
            e = float(d["start_time"]) + float(d["duration"])
            last[(net, inst)] = max(last.get((net, inst), 0.0), e)
        total = 0.0
        for (net, inst), e in last.items():
            if win[net] > 0:
                total += max(0.0, e - (inst * per[net] + win[net]))
        return total
    except Exception:
        return None


def recost_on_board(sched_path, spec_path, out_path, calibration=None):
    """Re-time a FIXED schedule under measured K1 board costs (scripts/recost_schedule_on_board.py):
    the AOT assignment kept, every dispatch's duration replaced by its board-faithful value, re-timed
    ASAP. This is the 'how the RUN differs from the GANTT' signal. Returns the out schedule path or None."""
    cmd = [PY, "scripts/recost_schedule_on_board.py", "--schedule", sched_path,
           "--spec", spec_path, "--out", out_path]
    if calibration:
        cmd += ["--calibration", calibration]
    r = _run(cmd)
    return out_path if os.path.exists(out_path) else None


def baseline(spec: dict) -> dict:
    """Strip all levers so the loop starts from a clean floor."""
    spec = copy.deepcopy(spec)
    sch = spec.setdefault("scheduler", {})
    sch["enable_impls"] = False
    sch["machine_combination_mode"] = "singletons"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"].setdefault("topo_tag_override", True)
    return spec


def apply_ime(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    built, failed, have_existing = [], [], False
    for key, info in spec.get("networks", {}).items():
        net, variant = _net_variant(info.get("dispatch_deps_path", ""), key)
        if not _is_conv_net(net, variant):
            continue
        if os.path.exists(_rvv_profile(net, variant, hw="ime_x60")):
            have_existing = True
            continue  # already has an ime profile (matmul nets, or a prior build) — do not clobber
        r = _run([PY, "scripts/make_ime_profile.py", "--net", net, "--variant", variant])
        if r.returncode == 0:
            built.append(f"{net}.{variant}")
        else:
            failed.append(f"{net}.{variant}: {(r.stderr or r.stdout or '').strip()[-160:]}")
    log(f"      ime: built ime_x60 profiles for {built or '(none new; existing reused)'}")
    if failed:
        # WHY THIS IS LOUD. Swallowing the failure while still setting enable_impls=True
        # would, on a checkout without scripts/make_ime_profile.py or xpu-rt/data/, make
        # the "ime" lever a no-op that the ledger records as applied -- and a rejected
        # lever and an unapplied one are not the same finding.
        for f in failed:
            log(f"      ime: BUILD FAILED {f}")
        if not built and not have_existing:
            raise SystemExit(
                "ime lever cannot be applied: no ime_x60 profile exists and every build "
                "failed (see above). Refusing to report an unapplied lever as applied.")
    spec.setdefault("scheduler", {})["enable_impls"] = True
    return spec


def apply_shard(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    spec.setdefault("scheduler", {})["machine_combination_mode"] = "shard"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"]["topo_tag_override"] = False
    return spec


def apply_unfuse(spec: dict, log) -> dict:
    """AOT graph-rewrite lever (automatic): point each fused net at ModelBlaster's UNFUSED
    dispatch graph and re-solve. The unfused graph is ModelBlaster's real rewrite output
    (`apply_unfuse_hint` on the IR, re-exported); it is adopted here rather than regenerated
    on the fly because kernel regeneration needs the RISC-V cross-toolchain (see
    generate_kernels.py) and board re-profiling needs the physical target — those are build/
    hardware steps, not scheduling. The lever is a no-op for a net with no unfused build,
    and (like every lever) is accepted only if it measures better with no new misses."""
    spec = copy.deepcopy(spec)
    swapped = []
    for key, info in spec.get("networks", {}).items():
        dp = info.get("dispatch_deps_path", "")
        cand = dp.replace(".ctrl.int8", ".unfused.int8").replace(".fused.", ".unfused.")
        cand_abs = cand if os.path.isabs(cand) else os.path.join(REPO, cand)
        if cand != dp and os.path.exists(cand_abs):
            info["dispatch_deps_path"] = cand
            swapped.append(key)
    log(f"      unfuse: adopted ModelBlaster unfused dispatch graph for "
        f"{swapped or '(none — no fused net with an unfused build on disk)'}")
    return spec


def apply_shard_only(net: str):
    """A shard lever that widens ONE network and holds the rest at a single core.

    WHY PER-NETWORK. `apply_shard` flips a global switch, so the loop could only take
    sharding for every network at once or not at all. On `w5_ffn_dronet_yolo` that made
    the lever unusable: widening also widens `yolov8_nano_64x96`, 191.6 core-ms that
    monopolises all eight harts while the 5 ms-period networks wait, so worst deadline
    lateness went 24.67 -> 34.87 ms and the whole lever was rejected -- including the
    part that helps. The right answer on that rung is to widen `ffn_block` and `dronet`
    and leave yolo alone, which is exactly the decision the ladder was built to force
    ("it has to choose WHICH nets to widen"), and which one global switch cannot say.

    Successive rounds compose: each accepted `shard:<net>` appends to the list, so the
    loop discovers a SET of networks to widen one at a time instead of guessing it.
    """
    def f(spec: dict, log) -> dict:
        spec = copy.deepcopy(spec)
        sch = spec.setdefault("scheduler", {})
        sch["machine_combination_mode"] = "shard"
        only = list(sch.get("shard_only_networks") or [])
        if net in only:
            return spec  # already widened; not a candidate (the caller skips no-ops)
        only.append(net)
        sch["shard_only_networks"] = only
        if "hardware" in spec and "profile" in spec["hardware"]:
            spec["hardware"]["profile"]["topo_tag_override"] = False
        log(f"      shard:{net}: widen {net}, hold {'+'.join(
            n for n in spec.get('networks', {}) if n not in only) or '(nothing)'} "
            f"at one core")
        return spec
    return f


LEVERS = {"ime": apply_ime, "shard": apply_shard, "unfuse": apply_unfuse}


def fusion_note():
    try:
        fb = {r["network"]: r for r in csv.DictReader(open(os.path.join(REPO, "xpu-rt/data/fusion_benefit.csv")))}
        worst = max(fb.values(), key=lambda r: float(r["epilogue_ceiling_pct"]))
        return (f"fuse: NOT applied — roofline decision-aid says the stack is compute-bound "
                f"(max fusible-epilogue ceiling {float(worst['epilogue_ceiling_pct']):.0f}% on "
                f"{worst['network']}); fusion collapses dispatches (scheduling) but is measured "
                f"~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.")
    except Exception:
        return "fuse: decision-aid unavailable."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--out-dir", default="results/codesign_loop")
    ap.add_argument("--solver", choices=["greedy", "cpsat"], default="greedy",
                    help="scheduler used for the predicted lever search (greedy is fast/low-mem; "
                         "cpsat gives an optimal 0-miss AOT schedule, needed for the sharp board arc).")
    ap.add_argument("--time-limit", type=int, default=45,
                    help="per-solve CP-SAT seconds during the lever search (ignored by greedy).")
    ap.add_argument("--search-calibration", nargs="?", const=True, default=None,
                    metavar="PATH",
                    help="run the INNER lever/rewrite search itself against measured "
                         "board costs instead of the isolated profile database. This is "
                         "the AOT decisions being re-taken in light of what the silicon "
                         "actually did: an op the board inflates (linear_f16 runs 2.05x "
                         "its profile) is worth splitting or widening even when the "
                         "isolated profile says it is cheap, and the AOT-cost search "
                         "cannot see that. Bare flag uses the default calibration "
                         "artifact; pass a path to override. Distinct from "
                         "--board-calibration, which only RE-SOLVES a fixed spec.")
    ap.add_argument("--board-calibration", nargs="?", const=True, default=None, metavar="PATH",
                    help="ENABLE THE BOARD-FEEDBACK ARM. After the predicted lever search converges, "
                         "re-cost the accepted schedule under measured K1 board costs; if that reveals "
                         "deadline misses the AOT Gantt hid, automatically re-solve the SAME spec with "
                         "--board-calibration and accept if the board misses drop. Bare flag uses "
                         "results/codesign_feedback/k1_board_calibration.json; pass a path to override.")
    ap.add_argument("--board-solver", choices=["greedy", "cpsat"], default="cpsat",
                    help="scheduler for the board-calibrated RE-SOLVE (the one final hard solve).")
    ap.add_argument("--board-time-limit", type=int, default=120,
                    help="CP-SAT seconds for the board-calibrated re-solve (the hard solve).")
    ap.add_argument("--levers", default=None,
                    help="comma-separated subset of {ime,shard,unfuse} the search may propose "
                         "(default: all). Restricting to a subset is honest scenario-scoping, e.g. "
                         "'--levers ime' asks 'with sharding off the table, does the loop autonomously "
                         "accept the matrix engine?' — the accept/reject is still measured, not forced.")
    ap.add_argument("--rewrite-arm", action="store_true",
                    help="ALSO propose ModelBlaster graph rewrites (fuse/unfuse/split/"
                         "shard) as candidates each round: advice -> bridge -> applier "
                         "-> graph gate -> bit-exact host verify -> MEASURE with "
                         "--runner -> schedule -> the same nine-term verdict. Without "
                         "it the loop searches scheduling levers only, which is what it "
                         "did before: a rewrite could never compete with a lever inside "
                         "one round.")
    ap.add_argument("--ir", action="append", default=[], metavar="NET=PATH",
                    help="ModelBlaster graph.json per net, for --rewrite-arm. "
                         "weights.npz/io.npz beside it enable the bit-exact gate.")
    ap.add_argument("--runner", choices=["board", "none"], default="board",
                    help="how a rewritten graph gets its costs. board: rebuild and "
                         "profile on the target. none: propose and gate rewrites but "
                         "never schedule them, since costing a rewrite from its "
                         "parent's profile measures the derivation, not the rewrite.")
    ap.add_argument("--rewrite-verbs", default=",".join(
        rewrite_arm.REWRITE_VERBS if rewrite_arm else ("fuse", "unfuse", "split",
                                                       "shard")))
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--hw", default="rvv_x60")
    ap.add_argument("--accel-hw", default="ime_x60",
                    help="the per-dispatch alternative the `ime` lever exposes; "
                         "set empty to disable that lever on a backend without one")
    ap.add_argument("--gen-root", default="gen/mb")
    ap.add_argument("--profile-root", default="gen/profile_mb")
    ap.add_argument("--replay", action="store_true",
                    help="deterministic offline replay: pin XPURT_CPSAT_WORKERS=1 and "
                         "refuse anything that would touch the board, so two runs of "
                         "the same inputs produce byte-identical schedules. This is the "
                         "mode someone without our hardware can check.")
    ap.add_argument("--accept-rule", choices=["objective", "legacy"], default="objective",
                    help="objective (default): candidate_objective.accept(), the project's "
                         "nine-term lexicographic rule. legacy: the old two-term "
                         "misses-not-worse AND delta>EPS rule (kept only to reproduce "
                         "pre-hardening runs).")
    ap.add_argument("--critical-models", default=None,
                    help="comma-separated nets carrying the hard-deadline term "
                         "(default: every deadline-bearing net)")
    ap.add_argument("--heavy-model", default=None,
                    help="the heavy perception net for terms 5-6 "
                         "(default: the longest-cadence net)")
    ap.add_argument("--objective", choices=["auto", "makespan", "lateness", "misses"], default="auto",
                    help="acceptance metric. 'auto' = the spec's declared objective (worst-response for "
                         "exact-cycle specs, else makespan). 'lateness' = total instance lateness "
                         "(sum max(0, end-deadline)); 'misses' = instance-miss count. The deadline-"
                         "correct objectives (lateness/misses) credit levers like IME that pull "
                         "instances in ahead of deadline without shortening the makespan critical path.")
    args = ap.parse_args()

    active_levers = list(LEVERS)
    if args.levers:
        want = [l.strip() for l in args.levers.split(",") if l.strip()]
        bad = [l for l in want if l not in LEVERS]
        if bad:
            print(f"--levers: unknown lever(s) {bad}; valid = {list(LEVERS)}")
            return 1
        active_levers = [l for l in LEVERS if l in want]

    DEFAULT_CAL = os.path.join(REPO,
                               "results/codesign_feedback/k1_board_calibration.json")
    board_cal_path = None
    if args.board_calibration is not None:
        board_cal_path = (DEFAULT_CAL if args.board_calibration is True
                          else args.board_calibration)
    # The calibration the INNER search solves against. None keeps the historical
    # behaviour: levers are chosen on isolated profile costs, and the board only ever
    # gets to re-solve what the AOT stage already decided.
    search_cal_path = None
    if args.search_calibration is not None:
        search_cal_path = (DEFAULT_CAL if args.search_calibration is True
                           else args.search_calibration)
        if not os.path.exists(search_cal_path):
            print(f"--search-calibration: no artifact at {search_cal_path}")
            return 1

    wl_stem = os.path.splitext(os.path.basename(args.workload))[0]
    out_dir = os.path.join(REPO, args.out_dir, wl_stem)
    spec_dir = os.path.join(out_dir, "specs")
    os.makedirs(spec_dir, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    if args.replay:
        # WHY PIN IT. scheduler_cpsat sets num_search_workers=1 and random_seed=42
        # precisely so a cold rerun matches bit-exactly, and says in as many words
        # that more workers under a time limit are NOT deterministic -- the docs then
        # mandate XPURT_CPSAT_WORKERS=0 for the published numbers, which is the fast
        # setting, not the reproducible one. A replay wants the reproducible one, and
        # it should say so rather than inherit whatever the shell had.
        os.environ["XPURT_CPSAT_WORKERS"] = "1"
        log("replay: XPURT_CPSAT_WORKERS=1 (deterministic; the published numbers use "
            "0/6, which is faster and not reproducible)")

    log(f"== co-design loop: {wl_stem} ==")
    if active_levers != list(LEVERS):
        log(f"lever menu restricted to {active_levers} (scenario-scoped; accept/reject still measured)")
    working = baseline(json.load(open(args.workload)))
    base_path = os.path.join(spec_dir, f"{wl_stem}_r0_baseline.json")
    json.dump(working, open(base_path, "w"), indent=1)
    if search_cal_path:
        # SAY IT LOUDLY. Every number the search reports now includes the board's
        # measured inflation, so it is not comparable with an AOT-cost run of the same
        # workload, and the baseline it improves on is the board-honest baseline.
        log(f"inner search runs against MEASURED board costs "
            f"({os.path.relpath(search_cal_path, REPO)}) -- lever scores here are "
            f"board-honest and are NOT comparable with an isolated-profile run")
    mk, miss, sched, err = solve(base_path, solver=args.solver,
                                 board_cal=search_cal_path,
                                 time_limit=args.time_limit)
    if mk is None:
        log(f"BASELINE SOLVE FAILED: {err}")
        return 1

    obj_mode = objective_of(working) if args.objective == "auto" else args.objective
    metric_name = {"worst_response": "worst-response", "lateness": "total-lateness",
                   "misses": "instance-misses"}.get(obj_mode, "makespan")
    # the new deadline-correct objectives judge "no new misses" on the INSTANCE-level
    # source-of-truth; worst_response/makespan keep their original op-count guard untouched.
    instance_guard = obj_mode in ("lateness", "misses")

    def score(sched_path, spec_dict, makespan):
        """The number the loop accepts on = the chosen objective (smaller is better)."""
        if obj_mode == "worst_response":
            w = worst_response_ms(sched_path, spec_dict)
            if w is not None:
                return w
        elif obj_mode == "lateness":
            v = total_lateness(sched_path, spec_dict)
            if v is not None:
                return v
        elif obj_mode == "misses":
            v = instance_misses(sched_path, spec_dict)[0]
            if v is not None:
                return float(v)
        return makespan

    def guard_miss(sched_path, spec_dict, op_miss):
        """The miss count the 'no new misses' acceptance guard uses: instance-level for the new
        lateness/misses objectives (the figure's source-of-truth), else the scheduler's op count
        (unchanged for the makespan/worst-response paths)."""
        if instance_guard:
            m = instance_misses(sched_path, spec_dict)[0]
            if m is not None:
                return m
        return op_miss

    # THE TWO COUNTERS DISAGREE AND MUST BE NAMED. guard_miss returns instance-level
    # misses only for the lateness/misses objectives; on the makespan and worst-response
    # paths it returns the scheduler's DISPATCH count. Printing either as "instance-miss"
    # would make a 22-instance workload report "72 instance-miss", which is impossible,
    # so the unit is named explicitly.
    miss_unit = "instance-miss" if instance_guard else "dispatch-miss"

    base_score = score(sched, working, mk)
    base_gmiss = guard_miss(sched, working, miss)
    backend = None
    irs = {}
    if rewrite_arm is not None:
        backend = rewrite_arm.Backend(
            target=args.target, hw=args.hw,
            accel_hw=(args.accel_hw or None), gen_root=args.gen_root,
            profile_root=args.profile_root)
    for spec_str in args.ir:
        if "=" in spec_str:
            k, v = spec_str.split("=", 1)
            irs[k] = v if os.path.isabs(v) else os.path.join(REPO, v)
    if args.rewrite_arm and (rewrite_arm is None or not irs):
        log("WARNING: --rewrite-arm needs xpu-rt/rewrite_arm.py and at least one "
            "--ir NET=PATH; searching scheduling levers only")

    critical, heavy = critical_and_heavy(working, args.critical_models, args.heavy_model)
    use_objective = (args.accept_rule == "objective") and objective is not None
    if args.accept_rule == "objective" and objective is None:
        log("WARNING: candidate_objective unavailable; falling back to the legacy two-term rule")
    base_out = outcome_of("baseline", sched, working, critical, heavy) if use_objective else None
    log(f"round 0 · baseline: {metric_name} {base_score:.3f} "
        f"(makespan {mk:.1f} ms), {base_gmiss} {miss_unit}")
    if use_objective:
        log(f"accept rule: candidate_objective.accept() · critical={list(critical)} "
            f"heavy={heavy}")

    applied, rounds, inapplicable = [], [], []
    traj = [{"round": 0, "lever": "baseline", "score_ms": round(base_score, 3),
             "makespan_ms": round(mk, 1), "misses": base_gmiss}]
    cur_mk, cur_score, cur_miss, cur_spec = mk, base_score, base_gmiss, working
    cur_sched, cur_spec_path = sched, base_path
    cur_out = base_out

    # EXPAND `shard` INTO ONE CANDIDATE PER NETWORK, plus the all-networks lever it
    # came from. The per-network candidates let a round widen `ffn_block` without also
    # widening yolo; keeping the global one means a workload where widening everything
    # IS right (w2, w3) still gets there in a single round.
    lever_fns = dict(LEVERS)
    if "shard" in active_levers:
        _nets = list((working.get("networks") or {}).keys())
        for _n in _nets:
            lever_fns[f"shard:{_n}"] = apply_shard_only(_n)
        active_levers = ([l for l in active_levers if l != "shard"]
                         + [f"shard:{n}" for n in _nets] + ["shard"])
        log(f"levers: shard expanded per network -> "
            f"{[f'shard:{n}' for n in _nets]} (+ shard = all networks)")

    for rnd in range(1, args.max_rounds + 1):
        cands = []
        for lever in active_levers:
            if lever in applied:
                continue
            cspec = lever_fns[lever](cur_spec, log)
            if cspec == cur_spec:
                # A LEVER THAT CHANGED NOTHING IS NOT A CANDIDATE. This is not
                # pedantry: on the sensor workload `unfuse` found no unfused build on
                # disk, returned the spec untouched, and the re-solve of that identical
                # spec came back with a different schedule (CP-SAT with >1 worker is not
                # deterministic under a time limit). The loop then credited the lever
                # with a p99 win of 38.86 -> 34.88 ms that came from solver noise, and
                # recorded an unapplied lever as applied. Skipped, and said so.
                log(f"round {rnd} · try {lever}: lever changed nothing on this spec "
                    f"(inapplicable here) — not a candidate")
                inapplicable.append(dict(round=rnd, lever=lever,
                                         reason="lever left the spec unchanged"))
                continue
            cpath = os.path.join(spec_dir, f"{wl_stem}_r{rnd}_{lever}.json")
            json.dump(cspec, open(cpath, "w"), indent=1)
            cmk, cmiss, csched, cerr = solve(cpath, solver=args.solver,
                                             board_cal=search_cal_path,
                                             time_limit=args.time_limit)
            if cmk is None:
                log(f"round {rnd} · try {lever}: SOLVE FAILED ({cerr[:120] if cerr else ''}) — reject")
                continue
            ok_build, why_build = buildable(csched, log, lever)
            if not ok_build:
                log(f"round {rnd} · try {lever}: {why_build}")
                log(f"round {rnd} · try {lever}: rejected -- the schedule is valid for "
                    f"the runtime and cannot be code-generated; not a candidate")
                inapplicable.append(dict(round=rnd, lever=lever, reason=why_build,
                                         kind="unbuildable"))
                continue
            csc = score(csched, cspec, cmk)
            cgmiss = guard_miss(csched, cspec, cmiss)
            delta = cur_score - csc
            pct = (-delta / cur_score * 100) if cur_score else 0.0
            if use_objective:
                cout = outcome_of(lever, csched, cspec, critical, heavy)
                ok, why = objective_verdict(cout, cur_out, csched, cur_sched, cspec)
                if ok is None:  # objective rule could not be evaluated -- do not guess
                    ok, why = False, "refused -- objective rule unavailable for this candidate"
            else:
                cout = None
                ok = (cgmiss <= cur_miss) and (delta > EPS)
                why = (f"legacy rule: misses {cur_miss}->{cgmiss}, "
                       f"{metric_name} delta {delta:+.3f} vs EPS {EPS}")
            log(f"round {rnd} · try {lever}: {metric_name} {cur_score:.3f} -> {csc:.3f} "
                f"({pct:+.1f}%), {cgmiss} {miss_unit} -> "
                f"{'ACCEPTABLE' if ok else 'reject'}")
            log(f"round {rnd} · try {lever}: {why}")
            cands.append(dict(lever=lever, mk=cmk, score=csc, miss=cgmiss, sched=csched,
                              spec=cspec, spec_path=cpath, ok=ok, why=why, out=cout))

        # ---- GRAPH-REWRITE CANDIDATES ----------------------------------------
        # The other half of the loop. A rewrite has to earn its place the same way a
        # scheduling lever does: proposed by the advisor, gated for correctness,
        # MEASURED, scheduled, and judged on the nine terms. Costing it from its
        # parent's profile would measure the derivation instead, so a rewrite that
        # cannot be measured is reported and skipped, never scored.
        if args.rewrite_arm and rewrite_arm is not None and irs and cur_sched:
            rw_dir = os.path.join(out_dir, "rewrites", f"round_{rnd}")
            verbs = tuple(v.strip() for v in args.rewrite_verbs.split(",") if v.strip())
            proposals = rewrite_arm.propose(cur_sched, irs, backend, rw_dir, log,
                                            verbs=verbs)
            for row in proposals:
                tag = f"r{rnd}{row['verb']}"
                label = f"{row['verb']}:{row['model']}"
                if not row["eligible"]:
                    log(f"round {rnd} · rewrite {label}: not eligible "
                        f"({row['stopped_at']}) — {str(row['why'])[:120]}")
                    inapplicable.append(dict(round=rnd, lever=label,
                                             reason=row["why"], kind="rewrite"))
                    continue
                if args.runner == "none":
                    log(f"round {rnd} · rewrite {label}: eligible, but --runner none "
                        f"— not scheduled (a rewrite has no profile until it is "
                        f"measured)")
                    inapplicable.append(dict(round=rnd, lever=label, kind="rewrite",
                                             reason="eligible but unmeasured "
                                                    "(--runner none)"))
                    continue
                seed = os.path.dirname(row["ir"])
                meas = rewrite_arm.measure_on_board(
                    row["rewritten_ir"], row["model"], tag, seed, backend, log)
                if meas is None:
                    inapplicable.append(dict(round=rnd, lever=label, kind="rewrite",
                                             reason="measurement failed"))
                    continue
                if not rewrite_arm.emit_graph(meas["staged_ir"], backend, log):
                    inapplicable.append(dict(round=rnd, lever=label, kind="rewrite",
                                             reason="emit_dispatch_graph failed"))
                    continue
                cpath = os.path.join(spec_dir, f"{wl_stem}_r{rnd}_{tag}.json")
                json.dump(cur_spec, open(cpath + ".base", "w"), indent=1)
                rewrite_arm.spec_with(cpath + ".base", row["model"], meas["model"],
                                      backend, cpath)
                cspec = json.load(open(cpath))
                cmk, cmiss, csched, cerr = solve(cpath, solver=args.solver,
                                                 board_cal=search_cal_path,
                                                 time_limit=args.time_limit)
                if cmk is None:
                    log(f"round {rnd} · rewrite {label}: SOLVE FAILED "
                        f"({(cerr or '')[:120]}) — reject")
                    continue
                ok_build, why_build = buildable(csched, log, label)
                if not ok_build:
                    log(f"round {rnd} · rewrite {label}: {why_build} — rejected")
                    inapplicable.append(dict(round=rnd, lever=label, kind="unbuildable",
                                             reason=why_build))
                    continue
                csc = score(csched, cspec, cmk)
                cgmiss = guard_miss(csched, cspec, cmiss)
                if use_objective:
                    cout = outcome_of(label, csched, cspec, critical, heavy)
                    ok, why = objective_verdict(cout, cur_out, csched, cur_sched, cspec)
                    if ok is None:
                        ok, why = False, "refused -- objective rule unavailable"
                else:
                    cout = None
                    ok = (cgmiss <= cur_miss) and ((cur_score - csc) > EPS)
                    why = "legacy rule"
                log(f"round {rnd} · rewrite {label}: {metric_name} {cur_score:.3f} -> "
                    f"{csc:.3f}, {cgmiss} {miss_unit} (MEASURED on "
                    f"{meas['runner']}) -> {'ACCEPTABLE' if ok else 'reject'}")
                log(f"round {rnd} · rewrite {label}: {why}")
                cands.append(dict(lever=label, mk=cmk, score=csc, miss=cgmiss,
                                  sched=csched, spec=cspec, spec_path=cpath, ok=ok,
                                  why=why, out=cout, kind="rewrite",
                                  measurement=meas))

        winners = [c for c in cands if c["ok"]]
        if not winners:
            # RECORD THE ROUND THAT ACCEPTED NOTHING. Rejections are recorded in their
            # own round entry rather than nested inside an accepted one, so the outcome
            # that most needs explaining -- "levers applied: none" -- keeps every reason
            # (e.g. w4_ffn_dronet_sensor): a reader can tell whether the levers were
            # rejected on measurement, failed to solve, or were skipped.
            rounds.append(dict(round=rnd, lever=None, accepted=False,
                               metric=metric_name,
                               score_before_ms=round(cur_score, 3),
                               accept_rule=args.accept_rule,
                               why="no candidate was accepted; the loop converged here",
                               rejected=[dict(lever=c["lever"], why=c.get("why"),
                                              score_ms=round(c["score"], 3),
                                              misses=c["miss"],
                                              kind=c.get("kind", "lever"))
                                         for c in cands]))
            log(f"round {rnd}: no lever is accepted by the "
                f"{'nine-term objective' if use_objective else 'legacy two-term'} rule "
                f"— CONVERGED")
            break
        # RANK WINNERS BY THE RULE THAT ACCEPTED THEM, not by makespan.
        # `min(winners, key=(c["score"], c["mk"]))` would rank by `score`, which IS the
        # makespan metric, so among several accepted candidates the loop would take the
        # FASTEST schedule even when another had fewer deadline misses, the term the
        # rule ranks first. With deterministic tolerances w5 offers several candidates
        # per round, and ranking by makespan takes its final miss count from 7 to NINE --
        # a hill-climb steered by the seventh term.
        #
        # The key mirrors the nine-term order: misses, worst lateness, frequency
        # shortfall, p99, then makespan as the final tie-break.
        def _rank(c):
            o = c.get("out")
            if o is None:  # legacy two-term rule; keep its old behaviour
                return (0, c["score"], c["mk"])
            return (o.total_misses(), o.worst_lateness(),
                    o.worst_frequency_shortfall(), o.worst_p99(), c["mk"])

        best = min(winners, key=_rank)
        pct = ((cur_score - best["score"]) / cur_score * 100) if cur_score else 0.0
        # A lever can be accepted on a HIGHER-priority term while this metric gets
        # worse -- w5 accepts shard:dronet because misses go 10 -> 7 even though makespan
        # grows -- so the change is signed once (never "(--12.9%)").
        log(f"round {rnd}: ACCEPT +{best['lever']}  {metric_name} "
            f"{cur_score:.3f} -> {best['score']:.3f} ({-pct:+.1f}%)")

        # render the accepted schedule's Gantt (IME dispatches darker+hatched)
        gstem = os.path.join(out_dir, f"round_{rnd}_{best['lever']}_gantt")
        _run([PY, "scripts/plot_scheduled_json.py", best["sched"], "--save", gstem,
              "--window-ms", str(round(best["mk"] * 1.05, 1))])

        rounds.append(dict(round=rnd, lever=best["lever"], metric=metric_name,
                           score_before_ms=round(cur_score, 3), score_after_ms=round(best["score"], 3),
                           makespan_before_ms=round(cur_mk, 1), makespan_after_ms=round(best["mk"], 1),
                           pct=round(-pct, 1), accepted=True, deadline_miss=best["miss"],
                           accept_rule=args.accept_rule, why=best.get("why"),
                           rejected=[dict(lever=c["lever"], why=c.get("why"),
                                          score_ms=round(c["score"], 3), misses=c["miss"])
                                     for c in cands if not c["ok"]],
                           terms=(objective.terms_dict(best["out"])
                                  if best.get("out") is not None else None)))
        applied.append(best["lever"])
        if best.get("kind") == "rewrite":
            # ITERATION. The next round must propose from the graph we just accepted,
            # not from the original -- otherwise a second rewrite is derived against an
            # IR that no longer describes what is scheduled, and two rewrites can never
            # compose (split then shard, say).
            _m = best["lever"].split(":", 1)[1]
            irs[_m] = best["measurement"]["staged_ir"]
            log(f"      rewrite arm: {_m} now proposes from "
                f"{os.path.relpath(irs[_m], REPO) if irs[_m].startswith(REPO) else irs[_m]}")
        cur_mk, cur_score, cur_miss, cur_spec = best["mk"], best["score"], best["miss"], best["spec"]
        cur_sched, cur_spec_path = best["sched"], best["spec_path"]
        if best.get("out") is not None:
            cur_out = best["out"]
        traj.append({"round": rnd, "lever": f"+{best['lever']}", "score_ms": round(cur_score, 3),
                     "makespan_ms": round(cur_mk, 1), "misses": cur_miss})

    fnote = fusion_note()
    log(fnote)

    # ---- BOARD-FEEDBACK ARM (the paper's headline beat, now automatic) --------------------
    # The predicted lever search above optimizes on ISOLATED profile costs — the AOT Gantt.
    # But the real K1 runs ~26-31% slower per op (measured; per-op exec inflation, not
    # contention). So after the AOT search converges we CLOSE THE LOOP against the board:
    #   1. re-cost the accepted AOT schedule under measured board costs (fixed assignment) —
    #      deadlines the Gantt promised can now be MISSED;
    #   2. if misses appear, automatically RE-SOLVE the same spec with --board-calibration so
    #      the scheduler knows the true costs, and accept if the board misses drop;
    #   3. repeat until misses==0 or no further improvement.
    # Every stage's verdict is a MEASURED instance-miss count (instance_misses(), the figure's
    # own source-of-truth).
    board = {"enabled": bool(board_cal_path)}
    panel_dir = os.path.join(out_dir, "panels")
    if board_cal_path:
        os.makedirs(panel_dir, exist_ok=True)

        def _stash(sched_src, name):
            dst = os.path.join(panel_dir, name + ".json")
            shutil.copy(sched_src, dst)
            msrc = sched_src.replace(".json", "_metrics.json")
            if os.path.exists(msrc):
                shutil.copy(msrc, dst.replace(".json", "_metrics.json"))
            return dst

        if not os.path.exists(board_cal_path):
            log(f"board arm: no calibration artifact at {board_cal_path} — arm SKIPPED (no-op).")
            board.update(enabled=False, reason="calibration artifact missing")
        else:
            log(f"\n== board-feedback arm ==  (calibration {os.path.relpath(board_cal_path, REPO)})")
            # The lever search above used short per-solve budgets to RANK levers cheaply; its
            # incumbent for the winning spec is not necessarily the optimum. Before closing the
            # board loop, re-solve the converged spec at the full budget so the AOT panel is the
            # definitive optimum for the loop's decision (same spec, solved to optimality) — and
            # so the reveal is measured against a schedule that genuinely meets every AOT deadline.
            amk, _aop, asched, aerr = solve(cur_spec_path, solver=args.board_solver,
                                            time_limit=args.board_time_limit)
            if asched is not None:
                cur_sched, cur_mk = asched, amk
                log(f"board arm · re-solved converged AOT spec at full budget "
                    f"({args.board_solver}, {args.board_time_limit}s): makespan {amk:.2f} ms")
            else:
                log(f"board arm · full-budget AOT re-solve failed ({(aerr or '')[:100]}); "
                    f"keeping the search incumbent")
            # canonical stage 1/2 schedules: baseline and the converged (polished) AOT schedule
            p1 = _stash(sched, "stage1_baseline")
            p2 = _stash(cur_sched, "stage2_aot")
            b_base = instance_misses(p1, working)[0]
            b_aot = instance_misses(p2, cur_spec)[0]
            log(f"board arm · stage1 baseline (AOT costs): {b_base} instance-miss")
            log(f"board arm · stage2 AOT-optimized (AOT costs): {b_aot} instance-miss  "
                f"[levers: {applied or 'none'}]")

            # stage 3 — re-cost the AOT schedule on the board (fixed assignment)
            p3 = os.path.join(panel_dir, "stage3_board_recost.json")
            recost_on_board(cur_sched, cur_spec_path, p3, calibration=board_cal_path)
            b_recost, b_recost_by = instance_misses(p3, cur_spec)
            log(f"board arm · stage3 board re-cost of the AOT schedule: {b_recost} instance-miss "
                f"{b_recost_by}  <- the run-honest reveal")

            stages = [
                dict(stage="baseline", cost="AOT", sched=os.path.relpath(p1, REPO),
                     instance_misses=b_base, makespan_ms=round(mk, 2)),
                dict(stage="aot-optimized", cost="AOT", sched=os.path.relpath(p2, REPO),
                     instance_misses=b_aot, makespan_ms=round(cur_mk, 2), levers=list(applied)),
                dict(stage="board-recost", cost="board", sched=os.path.relpath(p3, REPO),
                     instance_misses=b_recost, missed_by=b_recost_by),
            ]
            traj.append({"round": "board-recost", "lever": "recost(board)",
                         "makespan_ms": round(mk_of(p3), 1), "misses": b_recost})

            # stage 4 — automatic re-solve on board-calibrated costs, iterate the arm
            p4 = None
            b_final = b_recost
            if b_recost > 0:
                cur_board_miss = b_recost
                for it in range(1, args.max_rounds + 1):
                    rmk, rop, rsched, rerr = solve(cur_spec_path, solver=args.board_solver,
                                                   board_cal=board_cal_path,
                                                   time_limit=args.board_time_limit)
                    if rmk is None:
                        log(f"board arm · re-solve iter {it}: SOLVE FAILED ({(rerr or '')[:120]})")
                        break
                    b_re, b_re_by = instance_misses(rsched, cur_spec)
                    better = b_re < cur_board_miss
                    log(f"board arm · re-solve iter {it} (--board-calibration, {args.board_solver}): "
                        f"{cur_board_miss} -> {b_re} instance-miss {b_re_by} -> "
                        f"{'ACCEPT' if better else 'no improvement, stop'}")
                    if not better:
                        break
                    p4 = _stash(rsched, "stage4_board_resolve")
                    cur_board_miss = b_re
                    b_final = b_re
                    if b_re == 0:
                        break
                if p4 is not None:
                    stages.append(dict(stage="board-resolve", cost="board",
                                       sched=os.path.relpath(p4, REPO),
                                       instance_misses=b_final, makespan_ms=round(mk_of(p4), 2),
                                       solver=args.board_solver))
                    traj.append({"round": "board-resolve", "lever": "resolve(board)",
                                 "makespan_ms": round(mk_of(p4), 1), "misses": b_final})
            else:
                log("board arm · AOT schedule already meets every deadline on the board — "
                    "no re-solve needed.")

            board.update(
                calibration=os.path.relpath(board_cal_path, REPO),
                board_solver=args.board_solver,
                spec=os.path.relpath(cur_spec_path, REPO),
                baseline_instance_misses=b_base,
                aot_instance_misses=b_aot,
                board_recost_instance_misses=b_recost,
                board_resolve_instance_misses=(b_final if p4 is not None else None),
                recovered=(p4 is not None and b_final < b_recost),
                residual_misses=b_final,
                stages=stages,
            )
            arc = f"{b_base} -> {b_aot} -> {b_recost} -> {b_final if p4 is not None else b_recost}"
            log(f"board arm: instance-miss arc  baseline->AOT->board-recost->board-resolve = {arc}")

    denom = base_score if base_score else 1.0
    report = dict(workload=wl_stem, objective=metric_name,
                  accept_rule=("candidate_objective.accept (nine lexicographic terms)"
                               if use_objective else f"legacy two-term (EPS={EPS} ms)"),
                  critical_models=list(critical), heavy_model=heavy,
                  baseline_terms=(objective.terms_dict(base_out)
                                  if use_objective and base_out is not None else None),
                  final_terms=(objective.terms_dict(cur_out)
                               if use_objective and cur_out is not None else None),
                  # WHICH COSTS THE SEARCH SAW. Without this a board-honest run and an
                  # isolated-profile run of the same workload produce two reports that
                  # look comparable and are not.
                  search_costs=("measured board ("
                                + os.path.relpath(search_cal_path, REPO) + ")"
                                if search_cal_path else "isolated profile database"),
                  baseline_score_ms=round(base_score, 3), final_score_ms=round(cur_score, 3),
                  total_reduction_pct=round((base_score - cur_score) / denom * 100, 1),
                  baseline_makespan_ms=round(mk, 1), final_makespan_ms=round(cur_mk, 1),
                  levers_available=active_levers,
                  rewrite_arm=dict(
                      enabled=bool(args.rewrite_arm and rewrite_arm is not None
                                   and irs),
                      runner=args.runner, verbs=args.rewrite_verbs,
                      irs={k: os.path.relpath(v, REPO) if v.startswith(REPO) else v
                           for k, v in irs.items()},
                      backend=dict(target=args.target, hw=args.hw,
                                   accel_hw=args.accel_hw, gen_root=args.gen_root,
                                   profile_root=args.profile_root),
                      accepted=[r for r in rounds if ":" in str(r.get("lever"))]),
                  levers_applied=applied, levers_inapplicable=inapplicable,
                  rounds=rounds, trajectory=traj,
                  fusion=fnote, converged=True, board_feedback=board)
    json.dump(report, open(os.path.join(out_dir, "loop_report.json"), "w"), indent=1)

    _plot_traj(traj, os.path.join(out_dir, "objective_vs_round"), metric_name)
    _write_readme(out_dir, wl_stem, args, report, fnote)
    # WHAT THIS LINE MUST SAY. The declared objective alone is not enough, because the
    # nine-term rule may not have optimised it: the rule ranks hard deadline misses FIRST
    # and lateness fourth, so a large miss reduction can arrive with the objective going
    # UP (w3: total lateness 257.989 -> 274.292 ms while misses go 10 -> 1), which read
    # quickly looks like a failure. So: lead with the misses, name the term that actually
    # decided each acceptance, and give the objective its sign.
    _deciding = [f"{r['lever']} ({str(r.get('why', '')).split('--', 1)[-1].strip()})"
                 for r in rounds if r.get("accepted") and r.get("lever")]
    _red = report["total_reduction_pct"]
    # WHICH MISS COUNTER. There are two, and they disagree: guard_miss returns the
    # scheduler's per-DISPATCH op_deadline_miss_count for the makespan/worst-response
    # objectives, while candidate_objective's term 1 counts per-INSTANCE misses. On w2
    # that is 20 against 5 for the same schedule. Quoting one next to a verdict decided
    # on the other is how a reader ends up comparing two experiments, so report the
    # rule's own number when the rule is what decided, and label the fallback.
    if use_objective and base_out is not None and cur_out is not None:
        _mlabel = "hard deadline misses (instances)"
        _mbase, _mfinal = base_out.total_misses(), cur_out.total_misses()
    else:
        _mlabel = ("instance-misses" if instance_guard
                   else "dispatch-window misses")
        _mbase, _mfinal = base_gmiss, cur_miss
    log(f"\nCONVERGED: {_mlabel} {_mbase} -> {_mfinal}"
        f"; {metric_name} {base_score:.3f} -> {cur_score:.3f} ms "
        f"({-_red:+.1f}%), levers applied: {applied or 'none'}")
    if _deciding:
        for d in _deciding:
            log(f"  decided by: {d}")
    elif not applied:
        log("  no lever was accepted; every candidate and its reason is in "
            "loop_report.json under rounds[].rejected")
    if _red < 0:
        log(f"  note: {metric_name} rose while the rule accepted on a higher-ranked "
            f"term (misses rank first, {metric_name} lower) — this is the rule working, "
            f"not a regression")
    log(f"artifacts in {out_dir}")
    open(os.path.join(out_dir, "loop_log.txt"), "w").write("\n".join(lines) + "\n")
    return 0


def _plot_traj(traj, out, metric_name="makespan"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(traj)))
        ys = [t.get("score_ms", t["makespan_ms"]) for t in traj]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.step(xs, ys, where="post", color="#4c72b0", lw=2, marker="o")
        for i, t in enumerate(traj):
            ax.annotate(t["lever"], (i, ys[i]),
                        textcoords="offset points", xytext=(6, 8), fontsize=9,
                        weight="bold" if t["lever"] != "baseline" else "normal")
        if len(ys) > 1 and ys[0]:
            ax.annotate(f"-{(ys[0]-ys[-1])/ys[0]*100:.1f}%", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, -14), fontsize=9, color="#2f7d4f")
        ax.set_xlabel("feedback round"); ax.set_ylabel(f"{metric_name} (ms)")
        ax.set_xticks(xs); ax.set_ylim(0, max(ys) * 1.15)
        ax.set_title(f"Automatic co-design loop — {metric_name} per accepted round", weight="bold")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=160); fig.savefig(out + ".pdf")
    except Exception as e:
        print("traj plot skipped:", e)


def _write_readme(out_dir, stem, args, report, fnote):
    r = report
    md = [f"# Automatic co-design loop — `{stem}`", "",
          "Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →",
          "measure each → accept the largest measured makespan win with 0 added misses → repeat.",
          "", "```",
          f"scripts/run_codesign_loop.py --workload {args.workload} --max-rounds {args.max_rounds}",
          "```", "",
          f"**Baseline → final: {r['baseline_makespan_ms']} → {r['final_makespan_ms']} ms "
          f"(-{r['total_reduction_pct']}%)** — levers applied: {r['levers_applied'] or 'none'}.", "",
          "| round | lever | before (ms) | after (ms) | % | misses |", "|--:|--|--:|--:|--:|--:|"]
    for rr in r["rounds"]:
        # A round that accepted nothing is recorded too (so its rejections survive), and
        # it carries no before/after makespan -- there is no accepted candidate to have
        # one. Render it as the convergence row rather than crashing on the missing key.
        if not rr.get("lever"):
            md.append(f"| {rr['round']} | _(none accepted)_ | "
                      f"{rr.get('score_before_ms', '')} | — | — | — |")
            continue
        md.append(f"| {rr['round']} | +{rr['lever']} | "
                  f"{rr.get('makespan_before_ms', '')} | "
                  f"{rr.get('makespan_after_ms', '')} | {rr.get('pct', '')} | "
                  f"{rr.get('deadline_miss', '')} |")
    md += ["", f"Honest note — {fnote}", "",
           "Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, "
           "`round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), "
           "`specs/` (every candidate spec), `loop_log.txt`."]
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    sys.exit(main())
