"""Certificates for exact-work cyclic feedback experiments.

The short ``repeat_window`` postprocessor proves a minimum service rate.  This
module is stricter: a cycle declares an exact horizon and an integral number of
releases for every model.  A schedule passes only when it contains exactly that
work, respects every release/deadline and dependency, is clear at the wrap
boundary, and has no physical-core overlap.

It also computes a solver-independent response-time lower bound.  For one
instance of each model, replace every dispatch duration by its fastest legal
measured implementation and take the DAG critical path.  No scheduler can make
that model respond faster, even with unlimited cores and zero interference.
When a feasible schedule attains the bound, its response objective is globally
optimal without relying on a solver-specific optimality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import job_names


_INFEASIBLE_COST = 1e8


def _sha256(value: dict) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def declared_contract(networks_data: dict, *, tol: float = 1e-9) -> dict:
    """Validate and describe an exact integral release cycle."""
    horizon = float(networks_data.get("horizon_ms") or 0.0)
    if horizon <= 0:
        raise ValueError("an exact cycle requires positive horizon_ms")
    models = {}
    for name, spec in sorted((networks_data.get("networks") or {}).items()):
        if spec.get("period") is None:
            raise ValueError(f"exact cycle model {name!r} has no period")
        period = float(spec["period"])
        count = spec.get("num_instances")
        if period <= 0 or not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"exact cycle model {name!r} has invalid period/count")
        exact = horizon / period
        expected = int(round(exact))
        if not math.isclose(exact, expected, rel_tol=tol, abs_tol=tol):
            raise ValueError(
                f"{name!r}: horizon/period={exact:.12g} is not integral")
        if count != expected:
            raise ValueError(
                f"{name!r}: declares {count} instances; exact cycle needs "
                f"{expected}")
        models[name] = {
            "period_ms": period,
            "window_ms": float(spec.get("window_duration", period)),
            "instances": count,
            "frequency_hz": count * 1000.0 / horizon,
        }
    return {
        "cycle_ms": horizon,
        "semantics": "exact release count and frequency; repeat indefinitely",
        "models": models,
        "total_instances": sum(m["instances"] for m in models.values()),
    }


def workload_lower_bounds(workload, networks_data: dict,
                          critical_models: Iterable[str],
                          heavy_model: Optional[str] = None) -> dict:
    """Return per-model fastest-DAG response lower bounds in milliseconds."""
    contract = declared_contract(networks_data)
    known = set(contract["models"])
    combos = workload.get_machine_combinations()
    machines = workload.machines
    by_job: Dict[int, List[int]] = defaultdict(list)
    for index, op in enumerate(workload.operations):
        if op.job_id is not None:
            by_job[int(op.job_id)].append(index)

    per_model: Dict[str, float] = {}
    for job_id, indices in sorted(by_job.items()):
        if job_id < 0 or job_id >= len(workload.job_names):
            continue
        model = job_names.model_of(str(workload.job_names[job_id]), known)
        if model not in known or model in per_model:
            continue  # one instance is enough; periodic copies are identical
        index_set = set(indices)
        memo: Dict[int, float] = {}

        def finish_lb(index: int) -> float:
            if index in memo:
                return memo[index]
            op = workload.operations[index]
            feasible = [
                float(op.get_duration_for_combination(k, combos, machines))
                for k in range(len(combos))
                if k not in (op.infeasible_combinations or ())
                and float(op.get_duration_for_combination(k, combos, machines))
                < _INFEASIBLE_COST
            ]
            if not feasible:
                raise ValueError(
                    f"{model}: operation {op.operation_name!r} has no legal profile")
            pred_finish = 0.0
            for pred in op.get_predecessors():
                try:
                    pred_index = workload.operations.index(pred)
                except ValueError:
                    continue
                # Cross-instance recurrence can only raise the response time.
                # Ignoring it preserves a valid lower bound for this instance.
                if pred_index in index_set:
                    pred_finish = max(pred_finish, finish_lb(pred_index))
            memo[index] = pred_finish + min(feasible)
            return memo[index]

        per_model[model] = max(finish_lb(i) for i in indices)

    missing = known - set(per_model)
    if missing:
        raise ValueError(f"no workload instance found for {sorted(missing)}")
    critical = list(critical_models)
    return {
        "kind": "fastest_legal_implementation_dag_critical_path",
        "interpretation": (
            "Solver-independent response lower bound: unlimited cores, zero "
            "contention, zero transfer cost, fastest legal measured duration "
            "for every dispatch. A real schedule cannot be faster."
        ),
        "per_model_ms": {k: round(v, 9) for k, v in sorted(per_model.items())},
        "worst_critical_response_lower_bound_ms": round(
            max(per_model[m] for m in critical), 9),
        "heavy_response_lower_bound_ms": (
            round(per_model[heavy_model], 9) if heavy_model else None),
    }


def assess_schedule(schedule: dict, networks_data: dict,
                    critical_models: Iterable[str],
                    heavy_model: Optional[str] = None,
                    *, tol_ms: float = 1e-6) -> dict:
    """Validate exact work, cyclic wrap safety, and the aligned objective."""
    contract = declared_contract(networks_data)
    horizon = float(contract["cycle_ms"])
    known = set(contract["models"])
    dispatches = schedule.get("dispatches") or {}
    repo_root = Path(__file__).resolve().parent.parent
    expected_dispatches_per_job = {}
    expected_graphs = {}
    for model, spec in (networks_data.get("networks") or {}).items():
        graph_path = Path(str(spec.get("dispatch_deps_path", "")))
        if not graph_path.is_absolute():
            graph_path = repo_root / graph_path
        try:
            graph = json.loads(graph_path.read_text())
            graph_dispatches = graph.get("dispatches") or {}
            expected_dispatches_per_job[model] = len(graph_dispatches)
            key_to_id = {
                str(key): int(value["id"])
                for key, value in graph_dispatches.items()
            }
            expected_graphs[model] = {
                int(value["id"]): {
                    key_to_id[str(dep)]
                    for dep in value.get("dependencies") or ()
                }
                for value in graph_dispatches.values()
            }
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                f"cannot read dispatch graph for exact-cycle model {model!r}: "
                f"{exc}") from exc
    errors = []
    groups: Dict[str, dict] = {}
    for key, dispatch in dispatches.items():
        job = str(dispatch.get("job_name", ""))
        model = job_names.model_of(job, known)
        instance = job_names.instance_index(job, known)
        row = groups.setdefault(job, {
            "model": model,
            "instance": instance,
            "keys": [],
            "by_id": {},
            "start_ms": math.inf,
            "end_ms": 0.0,
        })
        start = float(dispatch.get("start_time", 0.0))
        end = start + float(dispatch.get("duration", 0.0))
        row["keys"].append(str(key))
        try:
            dispatch_id = int(dispatch["id"])
            if dispatch_id in row["by_id"]:
                errors.append(f"{job}: duplicate dispatch id {dispatch_id}")
            row["by_id"][dispatch_id] = (str(key), dispatch)
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key}: missing or invalid dispatch id")
        row["start_ms"] = min(row["start_ms"], start)
        row["end_ms"] = max(row["end_ms"], end)

    objective_rows = []
    model_counts = {name: 0 for name in known}
    for job, row in sorted(groups.items()):
        model = row["model"]
        instance = row["instance"]
        if model not in known or instance is None:
            errors.append(f"unrecognized periodic job {job!r}")
            continue
        spec = contract["models"][model]
        release = instance * float(spec["period_ms"])
        deadline = release + float(spec["window_ms"])
        if row["start_ms"] < release - tol_ms:
            errors.append(f"{job} starts before release")
        if row["end_ms"] > deadline + tol_ms:
            errors.append(f"{job} misses deadline")
        if row["end_ms"] > horizon + tol_ms:
            errors.append(f"{job} crosses cycle boundary")
        response = row["end_ms"] - release
        lateness = max(0.0, row["end_ms"] - deadline)
        objective_rows.append({
            "job": job,
            "model": model,
            "instance": instance,
            "release_ms": release,
            "completion_ms": row["end_ms"],
            "response_ms": response,
            "lateness_ms": lateness,
        })
        expected_dispatches = expected_dispatches_per_job[model]
        if len(row["keys"]) != expected_dispatches:
            errors.append(
                f"{job}: found {len(row['keys'])} dispatches, expected "
                f"{expected_dispatches}")
        expected_ids = set(expected_graphs[model])
        actual_ids = set(row["by_id"])
        if actual_ids != expected_ids:
            errors.append(
                f"{job}: dispatch ids {sorted(actual_ids)} do not match graph "
                f"{sorted(expected_ids)}")
        for dispatch_id, expected_dep_ids in expected_graphs[model].items():
            actual_entry = row["by_id"].get(dispatch_id)
            if actual_entry is None:
                continue
            _, actual_dispatch = actual_entry
            actual_dep_ids = set()
            for dep_key in actual_dispatch.get("dependencies") or ():
                dep = dispatches.get(dep_key)
                if dep is not None and str(dep.get("job_name", "")) == job:
                    try:
                        actual_dep_ids.add(int(dep["id"]))
                    except (KeyError, TypeError, ValueError):
                        pass
            missing_dep_ids = expected_dep_ids - actual_dep_ids
            if missing_dep_ids:
                errors.append(
                    f"{job} dispatch {dispatch_id}: missing graph dependencies "
                    f"{sorted(missing_dep_ids)}")
        model_counts[model] += 1

    for model, spec in sorted(contract["models"].items()):
        if model_counts[model] != spec["instances"]:
            errors.append(
                f"{model}: found {model_counts[model]} complete jobs, expected "
                f"{spec['instances']}")
    expected_total_dispatches = sum(
        contract["models"][model]["instances"] * per_job
        for model, per_job in expected_dispatches_per_job.items())
    if len(dispatches) != expected_total_dispatches:
        errors.append(
            f"found {len(dispatches)} dispatches, expected "
            f"{expected_total_dispatches}")

    # Dependency closure and precedence.
    for key, dispatch in dispatches.items():
        start = float(dispatch.get("start_time", 0.0))
        for dep in dispatch.get("dependencies") or ():
            if dep not in dispatches:
                errors.append(f"{key}: missing dependency {dep}")
                continue
            pred = dispatches[dep]
            pred_end = float(pred.get("start_time", 0.0)) + float(
                pred.get("duration", 0.0))
            if pred_end > start + tol_ms:
                errors.append(f"{key}: dependency {dep} completes after start")

    # Physical cores are exclusive, including every lane of a multi-hart
    # target. Intervals that only touch at an endpoint are legal.
    per_core: Dict[str, List[tuple]] = defaultdict(list)
    for key, dispatch in dispatches.items():
        start = float(dispatch.get("start_time", 0.0))
        end = start + float(dispatch.get("duration", 0.0))
        for core in str(dispatch.get("hardware_target", "")).split("+"):
            per_core[core].append((start, end, str(key)))
    for core, intervals in per_core.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1] - tol_ms:
                errors.append(
                    f"{core}: {previous[2]} overlaps {current[2]}")

    critical = set(critical_models)
    critical_responses = [
        r["response_ms"] for r in objective_rows if r["model"] in critical]
    heavy_responses = [
        r["response_ms"] for r in objective_rows
        if heavy_model and r["model"] == heavy_model]
    deadline_misses = sum(r["lateness_ms"] > tol_ms for r in objective_rows)
    objective = {
        "job_deadline_misses": deadline_misses,
        "max_job_lateness_ms": max(
            (r["lateness_ms"] for r in objective_rows), default=0.0),
        "worst_critical_response_ms": max(critical_responses, default=0.0),
        "heavy_max_response_ms": max(heavy_responses, default=0.0),
    }
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "schedule_sha256": _sha256(schedule),
        "contract": contract,
        "model_instances_observed": dict(sorted(model_counts.items())),
        "dispatches_observed": len(dispatches),
        "dispatches_expected": expected_total_dispatches,
        "boundary_clear": not any("cycle boundary" in e for e in errors),
        "dependency_closed": not any("dependency" in e for e in errors),
        "physical_cores_exclusive": not any("overlaps" in e for e in errors),
        "objective": objective,
        "per_instance": objective_rows,
        "errors": errors,
    }


def separation_certificate(original_schedule: dict, original_workload: dict,
                           feedback_schedule: dict, feedback_workload: dict,
                           critical_models: Iterable[str],
                           heavy_model: Optional[str] = None) -> dict:
    """Prove feedback beats the original graph's analytic response floor."""
    original = assess_schedule(
        original_schedule, original_workload, critical_models, heavy_model)
    feedback = assess_schedule(
        feedback_schedule, feedback_workload, critical_models, heavy_model)
    bounds = (original_schedule.get("metadata") or {}).get(
        "analytic_response_lower_bounds")
    if not bounds:
        raise ValueError("original schedule has no analytic response lower bound")
    floor = float(bounds["worst_critical_response_lower_bound_ms"])
    original_achieved = float(
        original["objective"]["worst_critical_response_ms"])
    achieved = float(feedback["objective"]["worst_critical_response_ms"])
    feedback_bounds = (feedback_schedule.get("metadata") or {}).get(
        "analytic_response_lower_bounds") or {}
    feedback_floor = feedback_bounds.get(
        "worst_critical_response_lower_bound_ms")
    original_optimal = (original["status"] == "pass"
                        and math.isclose(original_achieved, floor,
                                         rel_tol=0.0, abs_tol=1e-6))
    feedback_optimal = (feedback["status"] == "pass"
                        and feedback_floor is not None
                        and math.isclose(achieved, float(feedback_floor),
                                         rel_tol=0.0, abs_tol=1e-6))
    proven = (original["status"] == "pass" and feedback["status"] == "pass"
              and achieved < floor - 1e-9)
    return {
        "schema_version": 1,
        "verdict": "PROVEN" if proven else "NOT_PROVEN",
        "claim": (
            "The feedback schedule's feasible worst critical response is below "
            "a solver-independent lower bound for every schedule available to "
            "the original implementation graph."
        ),
        "original": original,
        "feedback": feedback,
        "original_lower_bounds": bounds,
        "feedback_lower_bounds": feedback_bounds,
        "original_response_ms": original_achieved,
        "feedback_response_ms": achieved,
        "original_response_floor_ms": floor,
        "feedback_response_floor_ms": feedback_floor,
        "original_global_optimum_proven": original_optimal,
        "feedback_global_optimum_proven": feedback_optimal,
        "separation_ms": floor - achieved,
        "improvement_pct": 100.0 * (floor - achieved) / floor,
    }
