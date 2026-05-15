"""CP-SAT copy/compute overlap planner.

. Schedules a set of operations (compute + copy) on a set of
resources (queues, DMA engines), respecting dependencies and
per-resource no-overlap. Returns issue times and resource
assignments that the ASYNC glue emitter consumes as `sync_edges`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from compgen.solve.backend_registry import default_registry
from compgen.solve.routing import choose_backend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)

__all__ = [
    "Operation",
    "Dependency",
    "Resource",
    "OverlapPlanInput",
    "OverlapScheduleSolved",
    "ScheduledOp",
    "plan_overlap",
]


OVERLAP_PLAN_SCHEMA_VERSION = "overlap_schedule_solver_v1"


@dataclass(frozen=True)
class Operation:
    op_id: str
    duration: int  # discrete ticks
    resource_id: str
    kind: str = "compute"  # "compute" | "copy"


@dataclass(frozen=True)
class Dependency:
    src_op: str
    dst_op: str


@dataclass(frozen=True)
class Resource:
    resource_id: str
    kind: str = "queue"  # "queue" | "dma" | "memory"


@dataclass(frozen=True)
class OverlapPlanInput:
    operations: tuple[Operation, ...]
    dependencies: tuple[Dependency, ...] = ()
    resources: tuple[Resource, ...] = ()
    deadline: int | None = None  # max-tick deadline
    time_budget_ms: int = 10_000


@dataclass(frozen=True)
class ScheduledOp:
    op_id: str
    start_tick: int
    end_tick: int
    resource_id: str


@dataclass(frozen=True)
class OverlapScheduleSolved:
    schema_version: str
    solver_backend: str
    status: str
    schedule: tuple[ScheduledOp, ...]
    makespan: int
    formulation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver_backend": self.solver_backend,
            "status": self.status,
            "schedule": [
                {
                    "op_id": s.op_id,
                    "start_tick": s.start_tick,
                    "end_tick": s.end_tick,
                    "resource_id": s.resource_id,
                }
                for s in self.schedule
            ],
            "makespan": self.makespan,
            "formulation_hash": self.formulation_hash,
        }


def _build_formulation(plan_input: OverlapPlanInput) -> dict[str, Any]:
    return {
        "operations": [
            {"op_id": o.op_id, "duration": o.duration, "resource_id": o.resource_id, "kind": o.kind}
            for o in plan_input.operations
        ],
        "dependencies": [{"src": d.src_op, "dst": d.dst_op} for d in plan_input.dependencies],
        "resources": [{"resource_id": r.resource_id, "kind": r.kind} for r in plan_input.resources],
        "deadline": plan_input.deadline,
    }


def _validate_input(plan_input: OverlapPlanInput) -> str | None:
    if not plan_input.operations:
        return "no operations"
    op_ids = {o.op_id for o in plan_input.operations}
    for d in plan_input.dependencies:
        if d.src_op not in op_ids:
            return f"dependency references unknown src op {d.src_op!r}"
        if d.dst_op not in op_ids:
            return f"dependency references unknown dst op {d.dst_op!r}"
    resources_declared = {r.resource_id for r in plan_input.resources} if plan_input.resources else None
    for o in plan_input.operations:
        if o.duration < 0:
            return f"op {o.op_id}: negative duration"
        if resources_declared is not None and o.resource_id not in resources_declared:
            return f"op {o.op_id}: resource {o.resource_id!r} not declared"
    return None


def _solve_cp_sat(
    request: SolverRequest,
    plan_input: OverlapPlanInput,
    *,
    probe_availability: BackendAvailabilityStatus,
) -> SolverResponse:
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    horizon = sum(o.duration for o in plan_input.operations)
    if plan_input.deadline is not None and plan_input.deadline > 0:
        horizon = min(horizon, plan_input.deadline)

    model = cp_model.CpModel()
    starts: dict[str, Any] = {}
    ends: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    for o in plan_input.operations:
        s = model.NewIntVar(0, horizon, f"start_{o.op_id}")
        e = model.NewIntVar(0, horizon, f"end_{o.op_id}")
        interval = model.NewIntervalVar(s, o.duration, e, f"int_{o.op_id}")
        starts[o.op_id] = s
        ends[o.op_id] = e
        intervals[o.op_id] = interval

    # Dependencies: end[src] <= start[dst]
    for d in plan_input.dependencies:
        model.Add(ends[d.src_op] <= starts[d.dst_op])

    # No-overlap per resource
    by_resource: dict[str, list[str]] = {}
    for o in plan_input.operations:
        by_resource.setdefault(o.resource_id, []).append(o.op_id)
    for res_id, op_ids in by_resource.items():
        if len(op_ids) > 1:
            model.AddNoOverlap([intervals[op] for op in op_ids])

    # Deadline
    if plan_input.deadline is not None and plan_input.deadline > 0:
        for o in plan_input.operations:
            model.Add(ends[o.op_id] <= plan_input.deadline)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [ends[o.op_id] for o in plan_input.operations])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(plan_input.time_budget_ms / 1000.0, 0.05)
    status_code = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = tuple(
            ScheduledOp(
                op_id=o.op_id,
                start_tick=int(solver.Value(starts[o.op_id])),
                end_tick=int(solver.Value(ends[o.op_id])),
                resource_id=o.resource_id,
            )
            for o in plan_input.operations
        )
        plan = OverlapScheduleSolved(
            schema_version=OVERLAP_PLAN_SCHEMA_VERSION,
            solver_backend=SolverBackendName.ORTOOLS_CP_SAT.value,
            status="optimal" if status_code == cp_model.OPTIMAL else "feasible",
            schedule=schedule,
            makespan=int(solver.Value(makespan)),
            formulation_hash=request.formulation_hash,
        )
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
            backend_availability=probe_availability,
            status=SolverStatus.OPTIMAL if status_code == cp_model.OPTIMAL else SolverStatus.FEASIBLE,
            formulation_hash=request.formulation_hash,
            time_ms=elapsed_ms,
            objective_value=float(plan.makespan),
            solution=plan.to_dict(),
        )
    if status_code == cp_model.INFEASIBLE:
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
            backend_availability=probe_availability,
            status=SolverStatus.INFEASIBLE,
            formulation_hash=request.formulation_hash,
            time_ms=elapsed_ms,
            infeasibility_reason="cp_sat infeasible (deadline or constraints unsatisfiable)",
        )
    return SolverResponse(
        problem_id=request.problem_id,
        problem_kind=request.problem_kind,
        selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
        backend_availability=probe_availability,
        status=SolverStatus.TIMEOUT,
        formulation_hash=request.formulation_hash,
        time_ms=elapsed_ms,
        infeasibility_reason=f"cp_sat status {status_code}",
    )


def plan_overlap(
    plan_input: OverlapPlanInput,
    *,
    registry: Any | None = None,
    problem_id: str = "overlap_schedule",
) -> tuple[SolverResponse, OverlapScheduleSolved | None]:
    reg = registry if registry is not None else default_registry()
    request = SolverRequest(
        problem_id=problem_id,
        problem_kind=SolverProblemKind.OVERLAP_PLANNING,
        formulation=_build_formulation(plan_input),
        time_budget_ms=plan_input.time_budget_ms,
    )
    err = _validate_input(plan_input)
    if err:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.OVERLAP_PLANNING,
                selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.INFEASIBLE,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=err,
            ),
            None,
        )

    backend_name = choose_backend(SolverProblemKind.OVERLAP_PLANNING, reg)
    if backend_name is None:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.OVERLAP_PLANNING,
                selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
                backend_availability=BackendAvailabilityStatus.IMPORT_MISSING,
                status=SolverStatus.BLOCKED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason="cp_sat unavailable",
            ),
            None,
        )
    probe = reg.probe(backend_name)
    response = _solve_cp_sat(request, plan_input, probe_availability=probe.availability)
    if response.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE} and isinstance(
        response.solution, dict
    ):
        body = response.solution
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
        return response, plan
    return response, None
