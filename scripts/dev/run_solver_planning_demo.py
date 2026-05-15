"""End-to-end solver-planning demo on a synthetic problem.

evidence. Runs Z3 obligation, memory plan, placement plan, and
overlap plan in sequence; persists every request + response under
``<out>/05_execution_plan/solver/*`` and ``<out>/04_kernel_codegen/solver/*``,
plus a backend-status probe report.

This is the artifact source for the evidence pack and the five
audit gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    _build_formulation as _build_memory_formulation,
    plan_memory,
)
from compgen.solve.overlap_planner import (
    Dependency,
    Operation,
    OverlapPlanInput,
    Resource,
    _build_formulation as _build_overlap_formulation,
    plan_overlap,
)
from compgen.solve.placement_planner import (
    Device,
    Edge,
    PlacementPlanInput,
    Region,
    _build_formulation as _build_placement_formulation,
    plan_placement,
)
from compgen.solve.reports import (
    write_solver_request,
    write_solver_response,
)
from compgen.solve.solver_types import (
    SolverProblemKind,
    SolverRequest,
)
from compgen.solve.z3_obligations import (
    OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
    OBLIGATION_KIND_TILE_INDEX_BOUNDS,
)


def _run_z3_obligations(out_dir: Path) -> None:
    from compgen.solve.backend_registry import default_registry
    from compgen.solve.solver_types import SolverBackendName

    reg = default_registry()
    backend = reg.get_backend(SolverBackendName.Z3)
    assert backend is not None

    target = out_dir / "04_kernel_codegen" / "solver"
    target.mkdir(parents=True, exist_ok=True)

    # Obligation 1: tile bounds (proves)
    req1 = SolverRequest(
        problem_id="tile_bounds_safe",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS,
            "params": {"tile": 16, "dim_max": 1024, "use_safe_len": True},
        },
        artifact_dir=str(target),
    )
    resp1 = backend.solve(req1)
    write_solver_request(req1, target / "tile_bounds_safe_request.json")
    write_solver_response(resp1, target / "tile_bounds_safe_response.json")

    # Obligation 2: tile bounds with unsafe len (counterexample)
    req2 = SolverRequest(
        problem_id="tile_bounds_unsafe",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS,
            "params": {"tile": 16, "dim_max": 1024, "use_safe_len": False},
        },
        artifact_dir=str(target),
    )
    resp2 = backend.solve(req2)
    write_solver_request(req2, target / "tile_bounds_unsafe_request.json")
    write_solver_response(resp2, target / "tile_bounds_unsafe_response.json")

    # Obligation 3: shape predicate implication
    req3 = SolverRequest(
        problem_id="implication_div16_div8",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
            "params": {
                "variables": {"K": {"min": 1, "max": 4096}},
                "applies_when": [{"op": "divisible_by", "var": "K", "k": 16}],
                "precondition": {"op": "divisible_by", "var": "K", "k": 8},
            },
        },
        artifact_dir=str(target),
    )
    resp3 = backend.solve(req3)
    write_solver_request(req3, target / "implication_div16_div8_request.json")
    write_solver_response(resp3, target / "implication_div16_div8_response.json")

    # Roll up a combined z3_obligations.json (a small index of the three above).
    (target / "z3_obligations.json").write_text(
        json.dumps(
            {
                "schema_version": "z3_obligations_index_v1",
                "obligations": [
                    {"id": "tile_bounds_safe", "status": resp1.status.value},
                    {"id": "tile_bounds_unsafe", "status": resp2.status.value},
                    {"id": "implication_div16_div8", "status": resp3.status.value},
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )


def _run_memory_plan(out_dir: Path) -> None:
    plan_input = MemoryPlanInput(
        buffers=(
            BufferSpec("input", size_bytes=4096, lifetime_start=0, lifetime_end=4, allowed_tiers=("scratchpad", "host")),
            BufferSpec("weight", size_bytes=8192, lifetime_start=0, lifetime_end=10, allowed_tiers=("host",)),
            BufferSpec("tile_a", size_bytes=2048, lifetime_start=1, lifetime_end=3, allowed_tiers=("scratchpad",)),
            BufferSpec("tile_b", size_bytes=2048, lifetime_start=5, lifetime_end=7, allowed_tiers=("scratchpad",)),
            BufferSpec("output", size_bytes=4096, lifetime_start=8, lifetime_end=10, allowed_tiers=("scratchpad", "host")),
        ),
        tier_capacities=(
            TierCapacity("scratchpad", capacity_bytes=16 * 1024, weight=1.0),
            TierCapacity("host", capacity_bytes=1024 * 1024, weight=4.0),
        ),
        alias_candidates=(AliasCandidate("tile_a", "tile_b"),),
        objective_lambda=1e-9,
    )
    response, plan = plan_memory(plan_input, problem_id="memory_solver")
    target = out_dir / "05_execution_plan" / "solver"
    target.mkdir(parents=True, exist_ok=True)
    write_solver_request(
        SolverRequest(
            problem_id="memory_solver",
            problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
            formulation=_build_memory_formulation(plan_input),
            time_budget_ms=plan_input.time_budget_ms,
        ),
        target / "memory_solver_request.json",
    )
    write_solver_response(response, target / "memory_solver_response.json")
    if plan is not None:
        (out_dir / "05_execution_plan").mkdir(parents=True, exist_ok=True)
        (out_dir / "05_execution_plan" / "memory_plan.solved.json").write_text(
            json.dumps(plan.to_dict(), sort_keys=True, indent=2)
        )


def _run_placement(out_dir: Path) -> None:
    plan_input = PlacementPlanInput(
        regions=(
            Region(
                "embed",
                allowed_devices=("cpu0", "gpu0"),
                memory_bytes=4096,
                compute_cost_by_device={"cpu0": 10.0, "gpu0": 1.0},
            ),
            Region(
                "matmul",
                allowed_devices=("gpu0",),
                memory_bytes=8192,
                compute_cost_by_device={"gpu0": 1.0},
            ),
            Region(
                "softmax",
                allowed_devices=("cpu0", "gpu0"),
                memory_bytes=2048,
                compute_cost_by_device={"cpu0": 2.0, "gpu0": 1.0},
            ),
        ),
        devices=(
            Device("cpu0", memory_capacity=1024 * 1024, target_class="host_cpu"),
            Device("gpu0", memory_capacity=512 * 1024, target_class="cuda_sm75"),
        ),
        edges=(
            Edge(
                "embed",
                "matmul",
                bytes_=4096,
                transfer_cost_by_device_pair={("cpu0", "gpu0"): 1e-3, ("gpu0", "cpu0"): 1e-3},
            ),
            Edge(
                "matmul",
                "softmax",
                bytes_=2048,
                transfer_cost_by_device_pair={("cpu0", "gpu0"): 1e-3, ("gpu0", "cpu0"): 1e-3},
            ),
        ),
    )
    response, plan = plan_placement(plan_input, problem_id="placement_solver")
    target = out_dir / "05_execution_plan" / "solver"
    target.mkdir(parents=True, exist_ok=True)
    write_solver_request(
        SolverRequest(
            problem_id="placement_solver",
            problem_kind=SolverProblemKind.PLACEMENT,
            formulation=_build_placement_formulation(plan_input),
            time_budget_ms=plan_input.time_budget_ms,
        ),
        target / "placement_solver_request.json",
    )
    write_solver_response(response, target / "placement_solver_response.json")
    if plan is not None:
        (out_dir / "05_execution_plan" / "placement_plan.solved.json").write_text(
            json.dumps(plan.to_dict(), sort_keys=True, indent=2)
        )


def _run_overlap(out_dir: Path) -> None:
    plan_input = OverlapPlanInput(
        operations=(
            Operation("copy_H2D", duration=4, resource_id="dma0", kind="copy"),
            Operation("gpu_matmul", duration=6, resource_id="gpu_queue0"),
            Operation("gpu_softmax", duration=2, resource_id="gpu_queue0"),
            Operation("copy_D2H", duration=4, resource_id="dma0", kind="copy"),
        ),
        dependencies=(
            Dependency("copy_H2D", "gpu_matmul"),
            Dependency("gpu_matmul", "gpu_softmax"),
            Dependency("gpu_softmax", "copy_D2H"),
        ),
        resources=(Resource("dma0", "dma"), Resource("gpu_queue0", "queue")),
    )
    response, plan = plan_overlap(plan_input, problem_id="overlap_solver")
    target = out_dir / "05_execution_plan" / "solver"
    target.mkdir(parents=True, exist_ok=True)
    write_solver_request(
        SolverRequest(
            problem_id="overlap_solver",
            problem_kind=SolverProblemKind.OVERLAP_PLANNING,
            formulation=_build_overlap_formulation(plan_input),
            time_budget_ms=plan_input.time_budget_ms,
        ),
        target / "overlap_solver_request.json",
    )
    write_solver_response(response, target / "overlap_solver_response.json")
    if plan is not None:
        (out_dir / "05_execution_plan" / "overlap_schedule.solved.json").write_text(
            json.dumps(plan.to_dict(), sort_keys=True, indent=2)
        )


def _write_backend_status(out_dir: Path) -> None:
    import subprocess

    target = out_dir / "solver"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "scripts/dev/probe_solver_backends.py", "--out", str(target)],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/phase_e_audit"))
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_backend_status(out)
    _run_z3_obligations(out)
    _run_memory_plan(out)
    _run_placement(out)
    _run_overlap(out)
    print(f"wrote solver-planning demo artifacts to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
