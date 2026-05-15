"""Cross-solver artifact integration tests (spec §11).

These tests prove that all four solver artifacts
(``placement_plan.solved.json``, ``overlap_schedule.solved.json``,
``memory_plan.solved.json``, ``bandwidth_plan.solved.json``) plus
the four solver responses live on disk after a real run, satisfy
the trust gates, and round-trip cleanly through their loader
APIs.

Spec §11 also requires that corrupt artifacts are rejected by the
ExecutionPlan validator. Those checks are pinned here too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.solve.bandwidth_planner import (
    BandwidthAllocation,
    BandwidthPlanInput,
    BandwidthPlanSolved,
    LinkCapacity,
    TransferDemand,
    _build_formulation as _build_bandwidth_formulation,
    plan_bandwidth,
)
from compgen.solve.memory_planner import (
    BufferAllocation,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
    _build_formulation as _build_memory_formulation,
    plan_memory,
)
from compgen.solve.overlap_planner import (
    Dependency,
    Operation,
    OverlapPlanInput,
    OverlapScheduleSolved,
    Resource,
    ScheduledOp,
    _build_formulation as _build_overlap_formulation,
    plan_overlap,
)
from compgen.solve.placement_planner import (
    Device,
    Edge,
    PlacementPlanInput,
    PlacementPlanSolved,
    Region,
    RegionAssignment,
    _build_formulation as _build_placement_formulation,
    plan_placement,
)
from compgen.solve.reports import write_solver_request, write_solver_response
from compgen.solve.solver_types import (
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)


def _emit_all_solvers(run_dir: Path) -> dict:
    """Run all four solvers on small problems and persist their
    request/response/solved.json artifacts under ``run_dir``."""

    solver_dir = run_dir / "05_execution_plan" / "solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "05_execution_plan"

    # 1. Placement
    p_input = PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("cpu",), memory_bytes=1024,
                   compute_cost_by_device={"cpu": 1.0}),
            Region("r1", allowed_devices=("cpu",), memory_bytes=1024,
                   compute_cost_by_device={"cpu": 1.0}),
        ),
        devices=(Device("cpu", memory_capacity=8192),),
    )
    p_resp, p_plan = plan_placement(p_input, problem_id="placement_solver")
    write_solver_request(
        SolverRequest(
            problem_id="placement_solver",
            problem_kind=SolverProblemKind.PLACEMENT,
            formulation=_build_placement_formulation(p_input),
        ),
        solver_dir / "placement_solver_request.json",
    )
    write_solver_response(p_resp, solver_dir / "placement_solver_response.json")
    if p_plan is not None:
        (out / "placement_plan.solved.json").write_text(
            json.dumps(p_plan.to_dict(), sort_keys=True, indent=2)
        )

    # 2. Overlap
    o_input = OverlapPlanInput(
        operations=(
            Operation("op0", duration=3, resource_id="cpu"),
            Operation("op1", duration=2, resource_id="cpu"),
        ),
        dependencies=(Dependency("op0", "op1"),),
        resources=(Resource("cpu"),),
    )
    o_resp, o_sched = plan_overlap(o_input, problem_id="overlap_solver")
    write_solver_request(
        SolverRequest(
            problem_id="overlap_solver",
            problem_kind=SolverProblemKind.OVERLAP_PLANNING,
            formulation=_build_overlap_formulation(o_input),
        ),
        solver_dir / "overlap_solver_request.json",
    )
    write_solver_response(o_resp, solver_dir / "overlap_solver_response.json")
    if o_sched is not None:
        (out / "overlap_schedule.solved.json").write_text(
            json.dumps(o_sched.to_dict(), sort_keys=True, indent=2)
        )

    # 3. Memory
    m_input = MemoryPlanInput(
        buffers=(
            BufferSpec("a", 1024, 0, 5, ("scratchpad",)),
            BufferSpec("b", 1024, 6, 10, ("scratchpad",)),
        ),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
    )
    m_resp, m_plan = plan_memory(m_input, problem_id="memory_solver")
    write_solver_request(
        SolverRequest(
            problem_id="memory_solver",
            problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
            formulation=_build_memory_formulation(m_input),
        ),
        solver_dir / "memory_solver_request.json",
    )
    write_solver_response(m_resp, solver_dir / "memory_solver_response.json")
    if m_plan is not None:
        (out / "memory_plan.solved.json").write_text(
            json.dumps(m_plan.to_dict(), sort_keys=True, indent=2)
        )

    # 4. Bandwidth
    b_input = BandwidthPlanInput(
        transfers=(
            TransferDemand("t0", bytes_=1, weight=2.0, max_bandwidth=60.0, link_id="L"),
            TransferDemand("t1", bytes_=1, weight=1.0, max_bandwidth=100.0, link_id="L"),
        ),
        links=(LinkCapacity("L", capacity=100.0),),
    )
    b_resp, b_plan = plan_bandwidth(b_input, problem_id="bandwidth_solver")
    write_solver_request(
        SolverRequest(
            problem_id="bandwidth_solver",
            problem_kind=SolverProblemKind.BANDWIDTH_ALLOCATION,
            formulation=_build_bandwidth_formulation(b_input),
        ),
        solver_dir / "bandwidth_solver_request.json",
    )
    write_solver_response(b_resp, solver_dir / "bandwidth_solver_response.json")
    if b_plan is not None:
        (out / "bandwidth_plan.solved.json").write_text(
            json.dumps(b_plan.to_dict(), sort_keys=True, indent=2)
        )

    return {
        "placement": (p_resp, p_plan),
        "overlap": (o_resp, o_sched),
        "memory": (m_resp, m_plan),
        "bandwidth": (b_resp, b_plan),
    }


# ---------------------------------------------------------------------------
# Artifact integration
# ---------------------------------------------------------------------------


def test_all_four_solvers_emit_artifacts(tmp_path: Path):
    out = _emit_all_solvers(tmp_path)
    for stage_name, (response, plan) in out.items():
        assert response.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}, (
            f"{stage_name} did not solve: {response}"
        )
        assert plan is not None, f"{stage_name} produced no plan"


def test_artifact_files_have_expected_layout(tmp_path: Path):
    _emit_all_solvers(tmp_path)
    solver_dir = tmp_path / "05_execution_plan" / "solver"
    plan_dir = tmp_path / "05_execution_plan"

    for name in (
        "placement_solver_request.json",
        "placement_solver_response.json",
        "overlap_solver_request.json",
        "overlap_solver_response.json",
        "memory_solver_request.json",
        "memory_solver_response.json",
        "bandwidth_solver_request.json",
        "bandwidth_solver_response.json",
    ):
        assert (solver_dir / name).is_file(), f"missing {name}"
    for name in (
        "placement_plan.solved.json",
        "overlap_schedule.solved.json",
        "memory_plan.solved.json",
        "bandwidth_plan.solved.json",
    ):
        assert (plan_dir / name).is_file(), f"missing {name}"


def test_m69_solver_gates_pass_on_artifact_tree(tmp_path: Path):
    """All five trust gates audit a real solver run-dir
    without failures."""

    _emit_all_solvers(tmp_path)
    from compgen.audit.solver_gates import all_solver_gates

    gates = all_solver_gates(run_dir=tmp_path)
    failed = [(g.name, g.detail) for g in gates if g.status == "fail"]
    assert not failed, f"M-69 gate failures on real artifact tree: {failed}"
    names = [g.name for g in gates]
    assert names == [
        "solver_backend_status",
        "solver_response_schema",
        "no_fake_solver_success",
        "formulation_hash_stability",
        "solver_artifact_traceability",
    ]


# ---------------------------------------------------------------------------
# Round-trip integrity
# ---------------------------------------------------------------------------


def test_memory_plan_solved_json_round_trips(tmp_path: Path):
    _emit_all_solvers(tmp_path)
    body = json.loads((tmp_path / "05_execution_plan" / "memory_plan.solved.json").read_text())
    plan = MemoryPlanSolved(
        schema_version=body["schema_version"],
        solver_backend=body["solver_backend"],
        status=body["status"],
        buffers=tuple(
            BufferAllocation(
                buffer_id=b["buffer_id"], tier=b["tier"],
                offset_bytes=int(b["offset_bytes"]),
                aliases_with=b.get("aliases_with"),
            )
            for b in body["buffers"]
        ),
        tier_peak_usage=dict(body["tier_peak_usage"]),
        objective_value=body.get("objective_value"),
        formulation_hash=body["formulation_hash"],
    )
    assert plan.to_dict() == body


def test_placement_plan_solved_json_round_trips(tmp_path: Path):
    _emit_all_solvers(tmp_path)
    body = json.loads((tmp_path / "05_execution_plan" / "placement_plan.solved.json").read_text())
    plan = PlacementPlanSolved(
        schema_version=body["schema_version"],
        solver_backend=body["solver_backend"],
        status=body["status"],
        assignments=tuple(
            RegionAssignment(region_id=a["region_id"], device_id=a["device_id"])
            for a in body["assignments"]
        ),
        objective_value=body.get("objective_value"),
        formulation_hash=body["formulation_hash"],
    )
    assert plan.to_dict() == body


def test_overlap_schedule_solved_json_round_trips(tmp_path: Path):
    _emit_all_solvers(tmp_path)
    body = json.loads((tmp_path / "05_execution_plan" / "overlap_schedule.solved.json").read_text())
    plan = OverlapScheduleSolved(
        schema_version=body["schema_version"],
        solver_backend=body["solver_backend"],
        status=body["status"],
        schedule=tuple(
            ScheduledOp(
                op_id=s["op_id"],
                start_tick=int(s["start_tick"]),
                end_tick=int(s["end_tick"]),
                resource_id=s["resource_id"],
            )
            for s in body["schedule"]
        ),
        makespan=int(body["makespan"]),
        formulation_hash=body["formulation_hash"],
    )
    assert plan.to_dict() == body


def test_bandwidth_plan_solved_json_round_trips(tmp_path: Path):
    _emit_all_solvers(tmp_path)
    body = json.loads((tmp_path / "05_execution_plan" / "bandwidth_plan.solved.json").read_text())
    plan = BandwidthPlanSolved(
        schema_version=body["schema_version"],
        solver_backend=body["solver_backend"],
        status=body["status"],
        allocations=tuple(
            BandwidthAllocation(
                transfer_id=a["transfer_id"],
                bandwidth=float(a["bandwidth"]),
                link_id=a["link_id"],
            )
            for a in body["allocations"]
        ),
        objective_value=float(body["objective_value"]),
        formulation_hash=body["formulation_hash"],
    )
    assert plan.to_dict() == body


# ---------------------------------------------------------------------------
# Corrupt-artifact rejection
# ---------------------------------------------------------------------------


def test_execution_plan_rejects_solver_memory_overlap(tmp_path: Path):
    """An execution plan built from a corrupt solver memory result
    (overlapping byte ranges) must fail validation. This is the
    contract between memory_planner and runtime/execution_plan."""

    from compgen.runtime.execution_plan import (
        BufferDescriptor,
        ExecutionPlan,
        Lifetime,
    )

    plan = ExecutionPlan(workload="x", target="host_cpu")
    plan.buffers.extend([
        BufferDescriptor(
            buffer_id="b0", size_bytes=1024, memory_space="scratchpad",
            lifetime=Lifetime(0, 10), ownership="exclusive", offset_bytes=0,
        ),
        BufferDescriptor(
            buffer_id="b1", size_bytes=1024, memory_space="scratchpad",
            lifetime=Lifetime(0, 10), ownership="exclusive", offset_bytes=512,
        ),
    ])
    with pytest.raises(ValueError, match="overlapping byte ranges"):
        plan.validate()


def test_corrupt_response_schema_rejected_by_loader(tmp_path: Path):
    """A response JSON missing required envelope fields cannot be
    loaded via SolverResponse.from_dict."""

    from compgen.solve.solver_types import SolverResponse

    body = {
        "schema_version": "solver_response_v1",
        "problem_id": "x",
        "problem_kind": "memory_allocation",
        "selected_backend": "mosek",
        "backend_availability": "available",
        # missing status
        "formulation_hash": "abcd",
        "time_ms": 0.0,
    }
    with pytest.raises(KeyError):
        SolverResponse.from_dict(body)


def test_formulation_hash_stable_across_rerun(tmp_path: Path):
    """Same input → same formulation_hash → byte-stable artifacts.
    Spec §11 determinism + §9.5 same-problem-same-hash."""

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _emit_all_solvers(run_a)
    _emit_all_solvers(run_b)

    for solver in ("placement", "overlap", "memory", "bandwidth"):
        a = json.loads((run_a / "05_execution_plan" / "solver" / f"{solver}_solver_response.json").read_text())
        b = json.loads((run_b / "05_execution_plan" / "solver" / f"{solver}_solver_response.json").read_text())
        assert a["formulation_hash"] == b["formulation_hash"], (
            f"{solver}: hash drift across reruns"
        )
        assert a["selected_backend"] == b["selected_backend"]
