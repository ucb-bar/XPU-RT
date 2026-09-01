"""Qualify a finite schedule prefix as a safely repeatable steady-state frame.

This is deliberately a postprocessor, not a crop. A frame is accepted only
when it contains a complete anchor-model instance, is closed under data
dependencies, has no job crossing the wrap boundary, and contains enough
complete instances of every periodic model to meet its minimum average
frequency when repeated indefinitely.

The frequency contract is minimum-rate, not exact phase preservation. A model
may run faster than ``1 / period`` when the anchor model admits a shorter
frame; the report states both rates so that over-service is visible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence

import job_names


def _groups(schedule: dict, known: Iterable[str]) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    for key, dispatch in (schedule.get("dispatches") or {}).items():
        job = str(dispatch.get("job_name", ""))
        group = groups.setdefault(job, {
            "model": job_names.model_of(job, known),
            "instance": job_names.instance_index(job, known),
            "keys": [],
            "start_ms": math.inf,
            "end_ms": 0.0,
        })
        start = float(dispatch.get("start_time", 0.0))
        end = start + float(dispatch.get("duration", 0.0))
        group["keys"].append(str(key))
        group["start_ms"] = min(group["start_ms"], start)
        group["end_ms"] = max(group["end_ms"], end)
    return groups


def _content_sha256(value: dict) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def anchor_completion_ms(schedule: dict, anchor_model: str,
                         known: Iterable[str]) -> float:
    """Completion of the anchor's first instance, or raise if it is absent."""
    anchors = [g for g in _groups(schedule, known).values()
               if g["model"] == anchor_model]
    if not anchors:
        raise ValueError(f"anchor model {anchor_model!r} is absent")
    first = min(
        anchors,
        key=lambda g: (
            g["instance"] if g["instance"] is not None else math.inf,
            g["start_ms"],
        ),
    )
    return float(first["end_ms"])


def assess(schedule: dict, periods_ms: Dict[str, float], anchor_model: str,
           known: Iterable[str], window_ms: float, *, tol_ms: float = 1e-6,
           ) -> dict:
    """Return a machine-readable pass/fail report for one proposed frame."""
    known = set(known)
    window = float(window_ms)
    groups = _groups(schedule, known)
    included_jobs = {
        job for job, group in groups.items()
        if group["start_ms"] < window - tol_ms
    }
    crossing = sorted(
        job for job in included_jobs
        if groups[job]["end_ms"] > window + tol_ms)
    included_keys = {
        key for job in included_jobs for key in groups[job]["keys"]
    }

    missing_dependencies = []
    dispatches = schedule.get("dispatches") or {}
    for key in sorted(included_keys):
        for dep in dispatches[key].get("dependencies") or ():
            if dep not in included_keys:
                missing_dependencies.append({"dispatch": key, "dependency": dep})

    model_rows = {}
    frequency_ok = True
    for model, period in sorted(periods_ms.items()):
        period = float(period)
        complete = sum(
            group["model"] == model
            and job in included_jobs
            and group["end_ms"] <= window + tol_ms
            for job, group in groups.items())
        required = int(math.ceil((window - tol_ms) / period))
        ok = complete >= required
        frequency_ok = frequency_ok and ok
        model_rows[model] = {
            "complete_instances": complete,
            "required_instances": required,
            "achieved_hz": round(complete * 1000.0 / window, 3),
            "required_hz": round(1000.0 / period, 3),
            "pass": ok,
        }

    anchors = [g for job, g in groups.items()
               if job in included_jobs and g["model"] == anchor_model
               and g["end_ms"] <= window + tol_ms]
    anchor_ok = bool(anchors)
    passed = (window > 0 and anchor_ok and frequency_ok and not crossing
              and not missing_dependencies)
    return {
        "status": "pass" if passed else "fail",
        "window_ms": window,
        "semantics": "minimum average frequency; completed frame repeats from t=0",
        "anchor_model": anchor_model,
        "source_schedule_sha256": _content_sha256(schedule),
        "anchor_complete": anchor_ok,
        "boundary_clear": not crossing,
        "dependency_closed": not missing_dependencies,
        "crossing_jobs": crossing,
        "missing_dependencies": missing_dependencies,
        "models": model_rows,
        "dispatches_shown": len(included_keys),
        "dispatches_excluded": len(dispatches) - len(included_keys),
        "included_dispatches": sorted(included_keys),
    }


def find(schedule: dict, periods_ms: Dict[str, float], anchor_model: str,
         known: Iterable[str], *, quantum_ms: Optional[float] = None,
         max_window_ms: Optional[float] = None) -> dict:
    """Find the shortest clean frequency-safe frame on a regular grid."""
    if not periods_ms:
        raise ValueError("a repeat window needs at least one declared period")
    quantum = float(quantum_ms or min(periods_ms.values()))
    if quantum <= 0:
        raise ValueError("quantum_ms must be positive")
    anchor_end = anchor_completion_ms(schedule, anchor_model, known)
    max_window = float(max_window_ms or max(
        float(d.get("start_time", 0.0)) + float(d.get("duration", 0.0))
        for d in (schedule.get("dispatches") or {}).values()))
    step = max(1, int(math.ceil((anchor_end - 1e-9) / quantum)))
    while step * quantum <= max_window + 1e-6:
        report = assess(schedule, periods_ms, anchor_model, known,
                        step * quantum)
        report["quantum_ms"] = quantum
        report["anchor_completion_ms"] = anchor_end
        if report["status"] == "pass":
            return report
        step += 1
    raise ValueError(
        f"no repeatable {quantum:g} ms-grid frame for {anchor_model!r} "
        f"through {max_window:g} ms")


def find_common(schedules: Sequence[dict], periods_ms: Dict[str, float],
                anchor_model: str, known: Iterable[str], *,
                quantum_ms: Optional[float] = None,
                max_window_ms: Optional[float] = None) -> List[dict]:
    """Find the shortest one frame length that qualifies every schedule."""
    if not schedules:
        raise ValueError("no schedules supplied")
    if not periods_ms:
        raise ValueError("a repeat window needs at least one declared period")
    quantum = float(quantum_ms or min(periods_ms.values()))
    if quantum <= 0:
        raise ValueError("quantum_ms must be positive")
    anchor_end = max(anchor_completion_ms(s, anchor_model, known)
                     for s in schedules)
    if max_window_ms is None:
        max_window_ms = max(
            max(float(d.get("start_time", 0.0)) + float(d.get("duration", 0.0))
                for d in (s.get("dispatches") or {}).values())
            for s in schedules)
    step = max(1, int(math.ceil((anchor_end - 1e-9) / quantum)))
    while step * quantum <= float(max_window_ms) + 1e-6:
        reports = [assess(s, periods_ms, anchor_model, known, step * quantum)
                   for s in schedules]
        if all(r["status"] == "pass" for r in reports):
            for report in reports:
                report["quantum_ms"] = quantum
                report["common_anchor_completion_ms"] = anchor_end
            return reports
        step += 1
    raise ValueError(
        f"no common repeatable {quantum:g} ms-grid frame for "
        f"{anchor_model!r} through {float(max_window_ms):g} ms")


def extract_frame(schedule: dict, report: dict) -> dict:
    """Materialize the qualified prefix as a schedule that can be replayed.

    The returned schedule keeps the original dispatch identifiers and timing,
    removes every dispatch outside the frame, and records the repeat contract
    in metadata.  It is safe to concatenate copies at integer multiples of
    ``window_ms`` because :func:`assess` has already checked both dependency
    closure and the wrap boundary.
    """
    if report.get("status") != "pass":
        raise ValueError("cannot extract an unqualified repeat frame")
    included = set(report.get("included_dispatches") or ())
    result = copy.deepcopy(schedule)
    result["dispatches"] = {
        key: copy.deepcopy(dispatch)
        for key, dispatch in (schedule.get("dispatches") or {}).items()
        if key in included
    }
    contract = {
        "mode": "repeat_indefinitely",
        "window_ms": float(report["window_ms"]),
        "semantics": report["semantics"],
        "anchor_model": report["anchor_model"],
        "source_schedule_sha256": report["source_schedule_sha256"],
        "boundary_clear": bool(report["boundary_clear"]),
        "dependency_closed": bool(report["dependency_closed"]),
        "models": copy.deepcopy(report["models"]),
    }
    result["repeat_frame"] = contract
    metadata = result.setdefault("metadata", {})
    metadata["makespan"] = float(report["window_ms"])
    metadata["num_operations"] = len(result["dispatches"])
    metadata["repeat_frame"] = copy.deepcopy(contract)
    return result
