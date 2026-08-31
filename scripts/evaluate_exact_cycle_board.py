#!/usr/bin/env python3
"""Validate repeated K1 traces for the exact-cycle feedback experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TICKS_PER_MS = 24_000.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stdout_audit(path: Path, expected_rows: int, require_pools: bool,
                  expected_models: set[str], fp16_models: set[str],
                  required_runner_policy: str | None) -> dict:
    text = path.read_text()
    forbidden = [line for line in text.splitlines()
                 if re.search(r"\b(?:FATAL|FAIL)\b", line)]
    workers = re.findall(
        r"pinned_hart=(-?\d+) observed_cpu=(-?\d+).*entries_done=(\d+)", text)
    pools = re.findall(r"modelblaster_pool\[block=\d+\]: width=(\d+) harts=([0-9+]+)", text)
    verifies = re.findall(
        r"MODELBLASTER_VERIFY \[([^]]+)\] === "
        r"max_abs_err=(\S+) max_rel_err=(\S+) n=(\d+) "
        r"instance=(\d+) ready=(\d+)", text)
    runner = re.findall(
        r"xpurt_runner: sched_policy=(\S+) priority=(\d+)", text)
    observed_runner = re.findall(
        r"xpurt: observed_sched_policy=(\S+) priority=(\d+)", text)
    golden_checks = {}
    for model, abs_text, rel_text, n_text, instance_text, ready_text in verifies:
        abs_err, rel_err = float(abs_text), float(rel_text)
        atol, rtol = ((1e-2, 1e-2) if model in fp16_models else (0.0, 0.0))
        golden_checks[model] = {
            "max_abs_err": abs_err, "max_rel_err": rel_err,
            "elements": int(n_text), "instance": int(instance_text),
            "ready": int(ready_text) == 1, "atol": atol, "rtol": rtol,
            "pass": (int(ready_text) == 1 and int(instance_text) == 0
                     and int(n_text) > 0
                     and (abs_err <= atol or rel_err <= rtol)),
        }
    golden_valid = (
        set(golden_checks) == expected_models
        and all(row["pass"] for row in golden_checks.values()))
    audit = {
        "path": str(path), "sha256": _sha(path),
        "forbidden_lines": forbidden,
        "worker_count": len(workers),
        "worker_entries_done": sum(int(row[2]) for row in workers),
        "all_workers_pinned_as_requested": all(a == b for a, b, _ in workers),
        "composite_pools": [{"width": int(w), "harts": h} for w, h in pools],
        "golden_checks": golden_checks,
        "all_golden_outputs_valid": golden_valid,
        "runner_policy": ({"policy": runner[-1][0],
                           "priority": int(runner[-1][1])}
                          if runner else None),
        "observed_runner_policy": (
            {"policy": observed_runner[-1][0],
             "priority": int(observed_runner[-1][1])}
            if observed_runner else None),
    }
    runner_policy_valid = (
        required_runner_policy is None
        or (runner and runner[-1][0] == required_runner_policy
            and observed_runner
            and observed_runner[-1][0] == required_runner_policy
            and runner[-1][1] == observed_runner[-1][1]))
    audit["required_runner_policy"] = required_runner_policy
    audit["runner_policy_valid"] = bool(runner_policy_valid)
    audit["pass"] = (
        not forbidden and workers
        and audit["worker_entries_done"] == expected_rows
        and audit["all_workers_pinned_as_requested"]
        and golden_valid
        and runner_policy_valid
        and (bool(pools) if require_pools else not pools)
    )
    return audit


def _trace_metrics(path: Path, workload: dict, schedule: dict,
                   critical: set[str], heavy: str | None) -> dict:
    rows = list(csv.DictReader(path.open()))
    expected_rows = len(schedule["dispatches"])
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: {len(rows)} rows, expected {expected_rows}")
    if any(int(r["actual_end_cycles"]) <= int(r["actual_start_cycles"])
           for r in rows):
        raise ValueError(f"{path}: non-positive execution interval")
    if any(int(r["worker_hart"]) != int(r["hart"]) for r in rows):
        raise ValueError(f"{path}: worker executed on a non-master hart")
    entry_ids = sorted(int(r["entry_id"]) for r in rows)
    if entry_ids != list(range(expected_rows)):
        raise ValueError(f"{path}: trace entry ids are not exactly 0..{expected_rows - 1}")
    expected_identity = Counter(
        (str(dispatch["job_name"]), int(dispatch["id"]))
        for dispatch in schedule["dispatches"].values())
    actual_identity = Counter(
        (f"{row['network']}{int(row['instance'])}", int(row["dispatch_id"]))
        for row in rows)
    if actual_identity != expected_identity:
        raise ValueError(f"{path}: trace jobs/dispatch ids do not match schedule")
    expected_by_identity = {
        (str(dispatch["job_name"]), int(dispatch["id"])): dispatch
        for dispatch in schedule["dispatches"].values()
    }
    for row in rows:
        identity = (f"{row['network']}{int(row['instance'])}",
                    int(row["dispatch_id"]))
        expected = expected_by_identity[identity]
        if abs(float(row["predicted_start_ms"])
               - float(expected["start_time"])) > 1e-5:
            raise ValueError(f"{path}: predicted start differs for {identity}")
        if abs(float(row["predicted_duration_ms"])
               - float(expected["duration"])) > 1e-5:
            raise ValueError(f"{path}: predicted duration differs for {identity}")
    early = [r for r in rows
             if int(r["actual_start_cycles"]) / TICKS_PER_MS + 1e-9
             < float(r["predicted_start_ms"])]
    if early:
        first = early[0]
        raise ValueError(
            f"{path}: {len(early)} entries began before their schedule-issued "
            f"start; first entry_id={first['entry_id']}")

    by_job: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_job[(row["network"], int(row["instance"]))].append(row)

    jobs = []
    for (model, instance), group in sorted(by_job.items()):
        spec = workload["networks"][model]
        period = float(spec["period"])
        phase = float(spec.get("phase", 0.0) or 0.0)
        release = phase + instance * period
        start = min(int(r["actual_start_cycles"]) for r in group) / TICKS_PER_MS
        end = max(int(r["actual_end_cycles"]) for r in group) / TICKS_PER_MS
        response = end - release
        lateness = max(0.0, response - period)
        jobs.append({
            "model": model, "instance": instance, "dispatches": len(group),
            "release_ms": release, "actual_start_ms": start,
            "actual_end_ms": end, "response_ms": response,
            "lateness_ms": lateness,
        })
    critical_response = max(
        row["response_ms"] for row in jobs if row["model"] in critical)
    heavy_response = (max(row["response_ms"] for row in jobs
                          if row["model"] == heavy) if heavy else None)
    max_actual_end = max(int(r["actual_end_cycles"]) for r in rows) / TICKS_PER_MS
    cycle_ms = float(workload["horizon_ms"])
    return {
        "path": str(path), "sha256": _sha(path), "trace_rows": len(rows),
        "jobs": jobs, "job_count": len(jobs),
        "schedule_identity_match": True,
        "early_start_count": 0,
        "max_actual_end_ms": max_actual_end,
        "cycle_boundary_clear": max_actual_end <= cycle_ms,
        "deadline_misses": sum(row["lateness_ms"] > 0 for row in jobs),
        "max_lateness_ms": max(row["lateness_ms"] for row in jobs),
        "worst_critical_response_ms": critical_response,
        "heavy_response_ms": heavy_response,
    }


def _aggregate(runs: list[dict], expected_rows: int) -> dict:
    critical = [r["worst_critical_response_ms"] for r in runs]
    heavy = [r["heavy_response_ms"] for r in runs]
    return {
        "runs": len(runs),
        "expected_trace_rows": expected_rows,
        "all_complete": all(r["trace_rows"] == expected_rows for r in runs),
        "all_release_gates_honored": all(r["early_start_count"] == 0 for r in runs),
        "all_cycle_boundaries_clear": all(r["cycle_boundary_clear"] for r in runs),
        "total_deadline_misses": sum(r["deadline_misses"] for r in runs),
        "critical_response_ms": {
            "samples": critical, "min": min(critical),
            "median": statistics.median(critical), "max": max(critical),
        },
        "heavy_response_ms": {
            "samples": heavy, "min": min(heavy),
            "median": statistics.median(heavy), "max": max(heavy),
        },
    }


def _exact_rank_sum_less_p(left: list[float], right: list[float]) -> float:
    """Exact one-sided rank-sum p-value for left values being smaller.

    The dynamic program enumerates every same-sized relabelling of the pooled
    observations. Doubled average ranks preserve exact tie handling without a
    floating-point comparison.
    """
    pooled = left + right
    ordered = sorted(range(len(pooled)), key=pooled.__getitem__)
    ranks2 = [0] * len(pooled)
    begin = 0
    while begin < len(ordered):
        end = begin + 1
        while end < len(ordered) and pooled[ordered[end]] == pooled[ordered[begin]]:
            end += 1
        average_rank_twice = (begin + 1) + end
        for position in range(begin, end):
            ranks2[ordered[position]] = average_rank_twice
        begin = end

    observed = sum(ranks2[:len(left)])
    # dp[k][rank_sum] is the number of ways to select k observations.
    dp: list[dict[int, int]] = [{0: 1}] + [{} for _ in left]
    for rank in ranks2:
        for count in range(min(len(left), len(pooled)), 0, -1):
            for subtotal, ways in list(dp[count - 1].items()):
                total = subtotal + rank
                dp[count][total] = dp[count].get(total, 0) + ways
    favourable = sum(ways for total, ways in dp[len(left)].items()
                     if total <= observed)
    return favourable / math.comb(len(pooled), len(left))


def _distribution_comparison(original: list[float], feedback: list[float]) -> dict:
    wins = sum(new < old for new in feedback for old in original)
    ties = sum(new == old for new in feedback for old in original)
    pairs = len(original) * len(feedback)
    return {
        "feedback_runs_below_original_min": sum(
            value < min(original) for value in feedback),
        "feedback_runs_below_original_median": sum(
            value < statistics.median(original) for value in feedback),
        "pairwise_feedback_superiority_pct": 100.0 * (wins + 0.5 * ties) / pairs,
        "exact_mann_whitney_less_p": _exact_rank_sum_less_p(feedback, original),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--original-workload", required=True)
    ap.add_argument("--original-schedule", required=True)
    ap.add_argument("--feedback-workload", required=True)
    ap.add_argument("--feedback-schedule", required=True)
    ap.add_argument("--critical-model", action="append", required=True)
    ap.add_argument("--heavy-model", required=True)
    ap.add_argument("--fp16-model", action="append", default=[],
                    help="model whose baked golden uses fp16 tolerances")
    ap.add_argument("--minimum-runs-per-phase", type=int, default=1)
    ap.add_argument("--required-runner-policy",
                    help="for example SCHED_FIFO; audited from every run log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    critical = set(args.critical_model)
    fp16_models = set(args.fp16_model)
    phases = {}
    stdout_audits = {}
    for label in ("original", "feedback"):
        workload = _load(Path(getattr(args, f"{label}_workload")))
        schedule = _load(Path(getattr(args, f"{label}_schedule")))
        traces = sorted(run_dir.glob(f"{label}_*_trace.csv"))
        if not traces:
            raise SystemExit(f"no {label} traces in {run_dir}")
        if len(traces) < args.minimum_runs_per_phase:
            raise SystemExit(
                f"only {len(traces)} {label} traces; "
                f"need {args.minimum_runs_per_phase}")
        runs = [_trace_metrics(p, workload, schedule, critical, args.heavy_model)
                for p in traces]
        audits = []
        for trace in traces:
            stdout = Path(str(trace).replace("_trace.csv", "_stdout.txt"))
            audits.append(_stdout_audit(
                stdout, len(schedule["dispatches"]), label == "feedback",
                set(workload["networks"]), fp16_models,
                args.required_runner_policy))
        phases[label] = {
            "aggregate": _aggregate(runs, len(schedule["dispatches"])),
            "runs": runs,
        }
        stdout_audits[label] = audits

    old = phases["original"]["aggregate"]
    new = phases["feedback"]["aggregate"]
    old_c = old["critical_response_ms"]["median"]
    new_c = new["critical_response_ms"]["median"]
    old_h = old["heavy_response_ms"]["median"]
    new_h = new["heavy_response_ms"]["median"]
    all_audits = stdout_audits["original"] + stdout_audits["feedback"]
    if old["runs"] != new["runs"]:
        raise SystemExit(
            f"unmatched sample counts: original={old['runs']}, feedback={new['runs']}")
    distribution = _distribution_comparison(
        old["critical_response_ms"]["samples"],
        new["critical_response_ms"]["samples"])
    result = {
        "schema_version": 2,
        "experiment_id": "k1-exact-cycle-feedback-board-v1",
        "board": "SpaceMiT K1", "timer_ticks_per_ms": TICKS_PER_MS,
        "runs_per_phase": min(old["runs"], new["runs"]),
        "execution_protocol": {
            "minimum_runs_per_phase": args.minimum_runs_per_phase,
            "required_runner_policy": args.required_runner_policy,
        },
        "critical_models": sorted(critical), "heavy_model": args.heavy_model,
        "fp16_models": sorted(fp16_models),
        "original": phases["original"], "feedback": phases["feedback"],
        "stdout_audits": stdout_audits,
        "comparison": {
            "original_median_critical_response_ms": old_c,
            "feedback_median_critical_response_ms": new_c,
            "original_median_heavy_response_ms": old_h,
            "feedback_median_heavy_response_ms": new_h,
            "median_critical_improvement_pct": 100.0 * (old_c - new_c) / old_c,
            "median_heavy_improvement_pct": 100.0 * (old_h - new_h) / old_h,
            "feedback_max_below_original_min": (
                new["critical_response_ms"]["max"]
                < old["critical_response_ms"]["min"]),
            **distribution,
        },
    }
    result["verdict"] = "CORROBORATED" if (
        old["total_deadline_misses"] == 0
        and new["total_deadline_misses"] == 0
        and old["all_complete"] and new["all_complete"]
        and old["all_release_gates_honored"]
        and new["all_release_gates_honored"]
        and old["all_cycle_boundaries_clear"]
        and new["all_cycle_boundaries_clear"]
        and new_c < old_c and new_h < old_h
        and all(a["pass"] for a in all_audits)
    ) else "FAILED"
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['verdict']}: critical median {old_c:.6f} -> {new_c:.6f} ms "
          f"({result['comparison']['median_critical_improvement_pct']:.2f}%)")
    return 0 if result["verdict"] == "CORROBORATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
