"""Bandwidth allocation LP planner.

Spec §10. Given a set of transfers sharing one or more links plus
per-transfer demand (bytes), maximum allowed bandwidth, and weight,
allocate bandwidth that maximizes weighted throughput subject to
the link capacity:

    maximize   Σ w[i] · bw[i]
    subject to bw[i] >= 0
               bw[i] <= max_bw[i]   (per-transfer cap; optional)
               Σ_{i on link L} bw[i] <= cap[L]   (per-link capacity)
               bw[i] >= min_bw[i]    (per-transfer minimum demand; infeasibility
                                      check)

Routed to MOSEK if available, else HiGHS. Both speak the LP
``scipy.optimize.linprog`` representation; the MOSEK path uses
``mosek.Task`` directly for parity with memory_planner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from compgen.solve.backends.mosek_backend import ensure_mosek_license_env
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)

__all__ = [
    "TransferDemand",
    "LinkCapacity",
    "BandwidthPlanInput",
    "BandwidthAllocation",
    "BandwidthPlanSolved",
    "plan_bandwidth",
]


BANDWIDTH_PLAN_SCHEMA_VERSION = "bandwidth_plan_solver_v1"


@dataclass(frozen=True)
class TransferDemand:
    transfer_id: str
    bytes_: int  # total bytes to move (informational)
    weight: float = 1.0  # objective weight
    min_bandwidth: float = 0.0  # bytes/us minimum required (0 = no floor)
    max_bandwidth: float | None = None  # bytes/us cap (None = no cap)
    link_id: str = "link0"  # which link this transfer uses


@dataclass(frozen=True)
class LinkCapacity:
    link_id: str
    capacity: float  # bytes/us


@dataclass(frozen=True)
class BandwidthPlanInput:
    transfers: tuple[TransferDemand, ...]
    links: tuple[LinkCapacity, ...]
    time_budget_ms: int = 5_000


@dataclass(frozen=True)
class BandwidthAllocation:
    transfer_id: str
    bandwidth: float
    link_id: str


@dataclass(frozen=True)
class BandwidthPlanSolved:
    schema_version: str
    solver_backend: str
    status: str
    allocations: tuple[BandwidthAllocation, ...]
    objective_value: float
    formulation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver_backend": self.solver_backend,
            "status": self.status,
            "allocations": [
                {"transfer_id": a.transfer_id, "bandwidth": a.bandwidth, "link_id": a.link_id}
                for a in self.allocations
            ],
            "objective_value": self.objective_value,
            "formulation_hash": self.formulation_hash,
        }


def _build_formulation(plan_input: BandwidthPlanInput) -> dict[str, Any]:
    return {
        "transfers": [
            {
                "transfer_id": t.transfer_id,
                "bytes": t.bytes_,
                "weight": t.weight,
                "min_bandwidth": t.min_bandwidth,
                "max_bandwidth": t.max_bandwidth,
                "link_id": t.link_id,
            }
            for t in plan_input.transfers
        ],
        "links": [{"link_id": l.link_id, "capacity": l.capacity} for l in plan_input.links],
    }


def _validate_input(plan_input: BandwidthPlanInput) -> str | None:
    link_ids = {l.link_id for l in plan_input.links}
    if not link_ids:
        return "no links declared"
    for t in plan_input.transfers:
        if t.link_id not in link_ids:
            return f"transfer {t.transfer_id}: link_id {t.link_id!r} not in links"
        if t.weight < 0:
            return f"transfer {t.transfer_id}: negative weight"
        if t.min_bandwidth < 0:
            return f"transfer {t.transfer_id}: negative min_bandwidth"
        if t.max_bandwidth is not None and t.max_bandwidth < t.min_bandwidth:
            return f"transfer {t.transfer_id}: max_bandwidth < min_bandwidth"
    for link in plan_input.links:
        if link.capacity < 0:
            return f"link {link.link_id}: negative capacity"
    # Feasibility precheck: per-link, sum of min_bandwidth must fit.
    per_link_min: dict[str, float] = {}
    for t in plan_input.transfers:
        per_link_min[t.link_id] = per_link_min.get(t.link_id, 0.0) + t.min_bandwidth
    for link in plan_input.links:
        if per_link_min.get(link.link_id, 0.0) > link.capacity + 1e-9:
            return (
                f"link {link.link_id}: total min_bandwidth "
                f"{per_link_min[link.link_id]} > capacity {link.capacity}"
            )
    return None


def _solve_lp_scipy(
    request: SolverRequest,
    plan_input: BandwidthPlanInput,
    *,
    backend: SolverBackendName,
    probe: BackendProbeResult,
    t0: float,
) -> SolverResponse:
    import numpy as np
    from scipy.optimize import LinearConstraint, linprog, Bounds

    transfers = plan_input.transfers
    n = len(transfers)
    # linprog minimizes; we negate weights for maximization.
    c = np.array([-t.weight for t in transfers], dtype=float)
    lb = np.array([t.min_bandwidth for t in transfers], dtype=float)
    ub = np.array(
        [t.max_bandwidth if t.max_bandwidth is not None else np.inf for t in transfers],
        dtype=float,
    )

    # Per-link capacity constraints.
    link_rows = []
    link_ubs = []
    for link in plan_input.links:
        row = np.array(
            [1.0 if t.link_id == link.link_id else 0.0 for t in transfers],
            dtype=float,
        )
        link_rows.append(row)
        link_ubs.append(link.capacity)
    a_mat = np.vstack(link_rows) if link_rows else None

    bounds = list(zip(lb, ub))
    res = linprog(
        c=c,
        A_ub=a_mat,
        b_ub=np.array(link_ubs, dtype=float) if link_rows else None,
        bounds=bounds,
        method="highs",
        options={"time_limit": max(plan_input.time_budget_ms / 1000.0, 0.05)},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not res.success:
        msg = str(getattr(res, "message", "")).lower()
        if "infeasible" in msg or res.status == 2:
            status = SolverStatus.INFEASIBLE
        elif "time" in msg or res.status == 1:
            status = SolverStatus.TIMEOUT
        else:
            status = SolverStatus.INFEASIBLE
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=backend,
            backend_availability=probe.availability,
            status=status,
            formulation_hash=request.formulation_hash,
            time_ms=elapsed_ms,
            infeasibility_reason=str(res.message),
        )
    allocations = tuple(
        BandwidthAllocation(
            transfer_id=t.transfer_id,
            bandwidth=float(res.x[i]),
            link_id=t.link_id,
        )
        for i, t in enumerate(transfers)
    )
    objective = -float(res.fun)
    plan = BandwidthPlanSolved(
        schema_version=BANDWIDTH_PLAN_SCHEMA_VERSION,
        solver_backend=backend.value,
        status="optimal",
        allocations=allocations,
        objective_value=objective,
        formulation_hash=request.formulation_hash,
    )
    return SolverResponse(
        problem_id=request.problem_id,
        problem_kind=request.problem_kind,
        selected_backend=backend,
        backend_availability=probe.availability,
        status=SolverStatus.OPTIMAL,
        formulation_hash=request.formulation_hash,
        time_ms=elapsed_ms,
        objective_value=objective,
        solution=plan.to_dict(),
    )


def _solve_lp_mosek(
    request: SolverRequest,
    plan_input: BandwidthPlanInput,
    *,
    probe: BackendProbeResult,
    t0: float,
) -> SolverResponse:
    """Real MOSEK LP solve for the bandwidth allocation problem.

    Mirrors the HiGHS LP formulation but uses ``mosek.Task`` directly
    so ``response.selected_backend`` is honest when MOSEK is selected
    by routing.
    """

    import mosek  # type: ignore[import-not-found]

    ensure_mosek_license_env()
    transfers = plan_input.transfers
    n = len(transfers)
    elapsed_ms = 0.0
    try:
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                task.appendvars(n)
                for j, t in enumerate(transfers):
                    task.putcj(j, -float(t.weight))  # minimise -weight·bw → maximise weight·bw
                    ub = t.max_bandwidth if t.max_bandwidth is not None else 1.0e30
                    if t.min_bandwidth == 0.0 and ub >= 1.0e30:
                        task.putvarbound(j, mosek.boundkey.lo, 0.0, +1.0e30)
                    else:
                        task.putvarbound(j, mosek.boundkey.ra, float(t.min_bandwidth), float(ub))
                task.putobjsense(mosek.objsense.minimize)
                for cid, link in enumerate(plan_input.links):
                    task.appendcons(1)
                    idxs, vals = [], []
                    for j, t in enumerate(transfers):
                        if t.link_id == link.link_id:
                            idxs.append(j)
                            vals.append(1.0)
                    if idxs:
                        task.putarow(cid, idxs, vals)
                    task.putconbound(cid, mosek.boundkey.up, -1.0e30, float(link.capacity))
                task.optimize()
                sol_sta = task.getsolsta(mosek.soltype.bas)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if sol_sta == mosek.solsta.optimal:
                    xx = [0.0] * n
                    task.getxx(mosek.soltype.bas, xx)
                    allocations = tuple(
                        BandwidthAllocation(
                            transfer_id=transfers[i].transfer_id,
                            bandwidth=float(xx[i]),
                            link_id=transfers[i].link_id,
                        )
                        for i in range(n)
                    )
                    obj = sum(transfers[i].weight * xx[i] for i in range(n))
                    plan = BandwidthPlanSolved(
                        schema_version=BANDWIDTH_PLAN_SCHEMA_VERSION,
                        solver_backend=SolverBackendName.MOSEK.value,
                        status="optimal",
                        allocations=allocations,
                        objective_value=obj,
                        formulation_hash=request.formulation_hash,
                    )
                    return SolverResponse(
                        problem_id=request.problem_id,
                        problem_kind=request.problem_kind,
                        selected_backend=SolverBackendName.MOSEK,
                        backend_availability=probe.availability,
                        status=SolverStatus.OPTIMAL,
                        formulation_hash=request.formulation_hash,
                        time_ms=elapsed_ms,
                        objective_value=obj,
                        solution=plan.to_dict(),
                    )
                if sol_sta in (
                    mosek.solsta.prim_infeas_cer,
                    mosek.solsta.dual_infeas_cer,
                ):
                    return SolverResponse(
                        problem_id=request.problem_id,
                        problem_kind=request.problem_kind,
                        selected_backend=SolverBackendName.MOSEK,
                        backend_availability=probe.availability,
                        status=SolverStatus.INFEASIBLE,
                        formulation_hash=request.formulation_hash,
                        time_ms=elapsed_ms,
                        infeasibility_reason=f"mosek solsta={sol_sta}",
                    )
                return SolverResponse(
                    problem_id=request.problem_id,
                    problem_kind=request.problem_kind,
                    selected_backend=SolverBackendName.MOSEK,
                    backend_availability=probe.availability,
                    status=SolverStatus.TIMEOUT,
                    formulation_hash=request.formulation_hash,
                    time_ms=elapsed_ms,
                    infeasibility_reason=f"mosek solsta={sol_sta}",
                )
    except Exception as exc:
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.MOSEK,
            backend_availability=probe.availability,
            status=SolverStatus.ERROR,
            formulation_hash=request.formulation_hash,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            infeasibility_reason=f"mosek bandwidth LP raised: {exc}",
        )


def plan_bandwidth(
    plan_input: BandwidthPlanInput,
    *,
    registry: Any | None = None,
    problem_id: str = "bandwidth_plan",
) -> tuple[SolverResponse, BandwidthPlanSolved | None]:
    """High-level entry point. Routes through the registry; MOSEK
    preferred, HiGHS fallback, BLOCKED when neither is available."""

    from compgen.solve.backend_registry import default_registry
    from compgen.solve.routing import choose_backend

    reg = registry if registry is not None else default_registry()
    request = SolverRequest(
        problem_id=problem_id,
        problem_kind=SolverProblemKind.BANDWIDTH_ALLOCATION,
        formulation=_build_formulation(plan_input),
        time_budget_ms=plan_input.time_budget_ms,
    )
    err = _validate_input(plan_input)
    t0 = time.perf_counter()
    if err:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.BANDWIDTH_ALLOCATION,
                selected_backend=SolverBackendName.HIGHS,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.INFEASIBLE,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=err,
            ),
            None,
        )

    backend = choose_backend(SolverProblemKind.BANDWIDTH_ALLOCATION, reg)
    if backend is None:
        return (
            SolverResponse(
                problem_id=problem_id,
                problem_kind=SolverProblemKind.BANDWIDTH_ALLOCATION,
                selected_backend=SolverBackendName.HIGHS,
                backend_availability=BackendAvailabilityStatus.IMPORT_MISSING,
                status=SolverStatus.BLOCKED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason="no_lp_milp_backend",
            ),
            None,
        )

    probe = reg.probe(backend)
    if backend is SolverBackendName.MOSEK:
        response = _solve_lp_mosek(request, plan_input, probe=probe, t0=t0)
    else:
        response = _solve_lp_scipy(request, plan_input, backend=backend, probe=probe, t0=t0)

    if response.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE} and isinstance(
        response.solution, dict
    ):
        body = response.solution
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
        return response, plan
    return response, None
