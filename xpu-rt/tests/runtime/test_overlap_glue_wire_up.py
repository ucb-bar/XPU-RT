"""End-to-end proof that ASYNC glue emit consumes overlap_planner output.

When ``05_execution_plan/overlap_schedule.solved.json`` is present,
the ASYNC emitter re-orders thread spawn to match the solver's
schedule. The manifest records ``solver_schedule_consumed=True``
plus the schedule path and the resulting region order. The
dependency wait/notify edges are unchanged (correctness is still
EventTensor-enforced); only the OS-level spawn order shifts to the
planner's preferred sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.runtime.execution_plan import (
    DependencyEdge,
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
)
from compgen.runtime.glue_emit.python_async import (
    _apply_overlap_schedule,
    _load_overlap_schedule,
    emit_python_async_executor,
)


def _write_plan(run_dir: Path) -> Path:
    plan_dir = run_dir / "05_execution_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    cert_dir = run_dir / "04_kernel_codegen" / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)

    plan = ExecutionPlan(workload="overlap_wire", target="host_cpu")
    plan.region_placement.extend(
        [
            RegionPlacement(region_id="rA", device="cpu0", queue="q0", stream_id=0),
            RegionPlacement(region_id="rB", device="cpu0", queue="q1", stream_id=0),
            RegionPlacement(region_id="rC", device="cpu0", queue="q2", stream_id=0),
        ]
    )
    # Independent regions: no dependency edges between them.
    for region_id in ("rA", "rB", "rC"):
        contract_hash = f"hash_{region_id}"
        cert_path = cert_dir / f"{contract_hash}.json"
        cert_body = {
            "schema_version": "kernel_certificate_v1",
            "contract_hash": contract_hash,
            "task_id": f"t_{region_id}",
            "region_id": region_id,
            "candidate_id": "c0",
            "accepted_at_utc": "2026-05-11T00:00:00Z",
            "artifact_hashes": {},
            "artifact_paths": {},
            "verifier_report_path": "",
            "verifier_report_hash": "",
            "claims": {},
        }
        cert_path.write_text(json.dumps(cert_body, sort_keys=True, indent=2))
        plan.region_kernel_bindings.append(
            RegionKernelBinding(
                region_id=region_id,
                contract_hash=contract_hash,
                certificate_path=str(cert_path.relative_to(run_dir)),
                kernel_artifact="",
                dispatch_model="async",
            )
        )
    plan_path = plan_dir / "execution_plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), sort_keys=True, indent=2))
    return run_dir


def _write_schedule(run_dir: Path, order: list[str], starts: list[int]) -> Path:
    plan_dir = run_dir / "05_execution_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "overlap_schedule_solver_v1",
        "solver_backend": "ortools_cp_sat",
        "status": "optimal",
        "schedule": [
            {
                "op_id": region_id,
                "start_tick": start,
                "end_tick": start + 1,
                "resource_id": f"q{i}",
            }
            for i, (region_id, start) in enumerate(zip(order, starts))
        ],
        "makespan": max(starts) + 1,
        "formulation_hash": "deadbeef",
    }
    path = plan_dir / "overlap_schedule.solved.json"
    path.write_text(json.dumps(body, sort_keys=True, indent=2))
    return path


def test_apply_overlap_schedule_reorders_by_start_tick():
    order = ["rA", "rB", "rC"]
    sched = {"rA": 10, "rB": 0, "rC": 5}
    out = _apply_overlap_schedule(order, sched)
    assert out == ["rB", "rC", "rA"]


def test_apply_overlap_schedule_stable_on_ties():
    order = ["rA", "rB", "rC"]
    sched = {"rA": 0, "rB": 0, "rC": 5}
    out = _apply_overlap_schedule(order, sched)
    assert out == ["rA", "rB", "rC"]


def test_apply_overlap_schedule_keeps_unscheduled_regions():
    order = ["rA", "rB", "rC"]
    sched = {"rA": 5, "rB": 0}
    out = _apply_overlap_schedule(order, sched)
    # rB is first, rA second, rC (unscheduled) appended at end.
    assert out == ["rB", "rA", "rC"]


def test_load_returns_none_when_no_schedule(tmp_path: Path):
    assert _load_overlap_schedule(tmp_path) is None


def test_load_parses_real_schedule(tmp_path: Path):
    _write_schedule(tmp_path, ["rA", "rB"], [3, 0])
    sched = _load_overlap_schedule(tmp_path)
    assert sched == {"rA": 3, "rB": 0}


def test_emit_consumes_overlap_schedule(tmp_path: Path):
    run = _write_plan(tmp_path)
    _write_schedule(run, ["rA", "rB", "rC"], [10, 0, 5])

    result = emit_python_async_executor(run)
    assert result.overall == "pass"

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["solver_schedule_consumed"] is True
    assert manifest["solver_schedule_path"].endswith("overlap_schedule.solved.json")
    # rB starts at 0, rC at 5, rA at 10 → expected emitted spawn order.
    assert manifest["solver_schedule_region_order"] == ["rB", "rC", "rA"]


def test_emit_without_schedule_marks_consumed_false(tmp_path: Path):
    run = _write_plan(tmp_path)
    # No overlap_schedule.solved.json written.

    result = emit_python_async_executor(run)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["solver_schedule_consumed"] is False
    assert manifest["solver_schedule_path"] is None
    assert manifest["solver_schedule_region_order"] is None


def test_emit_with_schedule_executor_uses_schedule_order(tmp_path: Path):
    """The generated executor's PLAN_REGION_ORDER reflects the schedule."""

    run = _write_plan(tmp_path)
    _write_schedule(run, ["rA", "rB", "rC"], [10, 0, 5])

    result = emit_python_async_executor(run)
    source = result.executor_path.read_text()
    # PLAN_REGION_ORDER is rendered from the schedule-reordered list.
    # Look for the slice of source defining the list.
    plan_order_idx = source.find("PLAN_REGION_ORDER")
    assert plan_order_idx >= 0
    snippet = source[plan_order_idx : plan_order_idx + 200]
    # In the snippet, "rB" should appear before "rA" and "rC" before "rA".
    rb = snippet.find("'rB'")
    rc = snippet.find("'rC'")
    ra = snippet.find("'rA'")
    assert rb < rc < ra, f"snippet order wrong: rB={rb} rC={rc} rA={ra}; snippet={snippet!r}"
