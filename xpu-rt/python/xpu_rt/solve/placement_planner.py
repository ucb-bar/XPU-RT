"""CP-SAT placement planner.

. Replaces the legacy ``solve/placement.py`` formulation with a
typed-envelope version: regions, devices, and inter-region transfer
edges. Solves via OR-Tools CP-SAT through the
:class:`OrToolsCpSatBackend`.

The legacy ``solve_placement`` remains as a shim forwarding to
:func:`plan_placement`, so callers in ``runtime/planner.py`` and
``agent/env/core.py`` keep working during migration.
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
    "Region",
    "Device",
    "Edge",
    "PlacementPlanInput",
    "PlacementPlanSolved",
    "RegionAssignment",
    "plan_placement",
]


PLACEMENT_PLAN_SCHEMA_VERSION = "placement_plan_solver_v1"


@dataclass(frozen=True)
class Region:
    region_id: str
    allowed_devices: tuple[str, ...]
    memory_bytes: int = 0
    compute_cost_by_device: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Device:
    device_id: str
    memory_capacity: int = 0
    target_class: str = ""


@dataclass(frozen=True)
class Edge:
    src_region: str
    dst_region: str
    bytes_: int = 0
    transfer_cost_by_device_pair: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementPlanInput:
    regions: tuple[Region, ...]
    devices: tuple[Device, ...]
    edges: tuple[Edge, ...] = ()
    warm_start: dict[str, str] = field(default_factory=dict)
    time_budget_ms: int = 10_000


@dataclass(frozen=True)
class RegionAssignment:
    region_id: str
    device_id: str


@dataclass(frozen=True)
class PlacementPlanSolved:
    schema_version: str
    solver_backend: str
    status: str
    assignments: tuple[RegionAssignment, ...]
    objective_value: float | None
    formulation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver_backend": self.solver_backend,
            "status": self.status,
            "assignments": [
                {"region_id": a.region_id, "device_id": a.device_id}
                for a in self.assignments
            ],
            "objective_value": self.objective_value,
            "formulation_hash": self.formulation_hash,
        }


def _build_formulation(plan_input: PlacementPlanInput) -> dict[str, Any]:
    return {
        "regions": [
            {
                "region_id": r.region_id,
                "allowed_devices": list(r.allowed_devices),
                "memory_bytes": r.memory_bytes,
                "compute_cost_by_device": dict(r.compute_cost_by_device),
            }
            for r in plan_input.regions
        ],
        "devices": [
            {"device_id": d.device_id, "memory_capacity": d.memory_capacity, "target_class": d.target_class}
            for d in plan_input.devices
        ],
        "edges": [
            {
                "src_region": e.src_region,
                "dst_region": e.dst_region,
                "bytes": e.bytes_,
                "transfer_cost_by_device_pair": {
                    f"{k[0]}->{k[1]}": v for k, v in e.transfer_cost_by_device_pair.items()
                },
            }
            for e in plan_input.edges
        ],
        "warm_start": dict(plan_input.warm_start),
    }


def _validate_input(plan_input: PlacementPlanInput) -> str | None:
    if not plan_input.regions:
        return "no regions"
    if not plan_input.devices:
        return "no devices"
    device_ids = {d.device_id for d in plan_input.devices}
    for r in plan_input.regions:
        if not r.allowed_devices:
            return f"region {r.region_id}: no allowed_devices"
        for d in r.allowed_devices:
            if d not in device_ids:
                return f"region {r.region_id}: allowed_device {d!r} not in declared devices"
    return None


def _solve_cp_sat(
    request: SolverRequest,
    plan_input: PlacementPlanInput,
    *,
    probe_availability: BackendAvailabilityStatus,
) -> SolverResponse:
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    model = cp_model.CpModel()
    regions = plan_input.regions
    devices = plan_input.devices
    region_ids = [r.region_id for r in regions]
    device_ids = [d.device_id for d in devices]

    # Binary x[r, d]
    x: dict[tuple[str, str], Any] = {}
    for r in regions:
        for d in devices:
            x[(r.region_id, d.device_id)] = model.NewBoolVar(f"x_{r.region_id}_{d.device_id}")

    # sum_d x[r,d] = 1, with disallowed entries forced to 0.
    for r in regions:
        model.AddExactlyOne([x[(r.region_id, d.device_id)] for d in devices])
        for d in devices:
            if d.device_id not in r.allowed_devices:
                model.Add(x[(r.region_id, d.device_id)] == 0)

    # Memory capacity per device.
    for d in devices:
        if d.memory_capacity > 0:
            model.Add(
                sum(r.memory_bytes * x[(r.region_id, d.device_id)] for r in regions)
                <= d.memory_capacity
            )

    scale = 1_000

    compute_terms = []
    for r in regions:
        for d in devices:
            cost = float(r.compute_cost_by_device.get(d.device_id, 0.0))
            if d.device_id not in r.allowed_devices:
                continue
            compute_terms.append(int(cost * scale) * x[(r.region_id, d.device_id)])

    transfer_terms = []
    for e in plan_input.edges:
        for d1 in devices:
            for d2 in devices:
                if d1.device_id == d2.device_id:
                    continue
                key = (d1.device_id, d2.device_id)
                cost = float(e.transfer_cost_by_device_pair.get(key, 0.0))
                if cost == 0.0 and e.bytes_ == 0:
                    continue
                # both[d1,d2] = 1 iff src on d1 AND dst on d2
                both = model.NewBoolVar(f"xfer_{e.src_region}_{e.dst_region}_{d1.device_id}_{d2.device_id}")
                model.AddImplication(both, x[(e.src_region, d1.device_id)])
                model.AddImplication(both, x[(e.dst_region, d2.device_id)])
                # both >= x_src_d1 + x_dst_d2 - 1 (linearization)
                model.Add(both >= x[(e.src_region, d1.device_id)] + x[(e.dst_region, d2.device_id)] - 1)
                weight = int(cost * e.bytes_ * scale) if e.bytes_ else int(cost * scale)
                transfer_terms.append(weight * both)

    model.Minimize(sum(compute_terms) + sum(transfer_terms))

    # Warm-start hints
    if plan_input.warm_start:
        for region_id, device_id in plan_input.warm_start.items():
            key = (region_id, device_id)
            if key in x:
                model.AddHint(x[key], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(plan_input.time_budget_ms / 1000.0, 0.05)
    status_code = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = tuple(
            RegionAssignment(
                region_id=r.region_id,
                device_id=next(
                    d.device_id
                    for d in devices
                    if solver.Value(x[(r.region_id, d.device_id)]) == 1
                ),
            )
            for r in regions
        )
        plan = PlacementPlanSolved(
            schema_version=PLACEMENT_PLAN_SCHEMA_VERSION,
            solver_backend=SolverBackendName.ORTOOLS_CP_SAT.value,
            status="optimal" if status_code == cp_model.OPTIMAL else "feasible",
            assignments=assignments,
            objective_value=float(solver.ObjectiveValue() / scale)
            if solver.ObjectiveValue() is not None
            else None,
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
            objective_value=plan.objective_value,
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
            infeasibility_reason="cp_sat returned INFEASIBLE",
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


def plan_placement(
    plan_input: PlacementPlanInput,
    *,
    registry: Any | None = None,
    problem_id: str = "placement_plan",
) -> tuple[SolverResponse, PlacementPlanSolved | None]:
    """High-level entry point.

    Routes through the registry; returns the typed response and, on
    OPTIMAL/FEASIBLE, the solved plan.
    """

    reg = registry if registry is not None else default_registry()
    request = SolverRequest(
        problem_id=problem_id,
        problem_kind=SolverProblemKind.PLACEMENT,
        formulation=_build_formulation(plan_input),
        time_budget_ms=plan_input.time_budget_ms,
    )
    err = _validate_input(plan_input)
    if err:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.PLACEMENT,
                selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.INFEASIBLE,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=err,
            ),
            None,
        )

    backend_name = choose_backend(SolverProblemKind.PLACEMENT, reg)
    if backend_name is None:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.PLACEMENT,
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
        return response, plan
    return response, None
