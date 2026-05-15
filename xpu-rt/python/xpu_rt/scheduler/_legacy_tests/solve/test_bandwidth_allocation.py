"""Bandwidth allocation LP tests (spec §10).

The LP form:

    maximize   sum_i w_i * bw_i
    subject to bw_i >= 0
               bw_i <= max_bw_i        (optional per-transfer cap)
               sum_{i on link L} bw_i <= cap[L]
               bw_i >= min_bw_i        (per-transfer floor)

Routed to MOSEK if available, else HiGHS. Both paths must return
honest typed status.
"""

from __future__ import annotations

import pytest

from compgen.solve.backend_registry import SolverBackendRegistry
from compgen.solve.backends.highs_backend import HighsBackend
from compgen.solve.backends.mosek_backend import MosekBackend
from compgen.solve.bandwidth_planner import (
    BandwidthPlanInput,
    LinkCapacity,
    TransferDemand,
    plan_bandwidth,
)
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverStatus,
)


def _two_transfer_single_link() -> BandwidthPlanInput:
    return BandwidthPlanInput(
        transfers=(
            TransferDemand(transfer_id="t0", bytes_=10_000_000, weight=2.0, max_bandwidth=60.0, link_id="L0"),
            TransferDemand(transfer_id="t1", bytes_=10_000_000, weight=1.0, max_bandwidth=100.0, link_id="L0"),
        ),
        links=(LinkCapacity(link_id="L0", capacity=100.0),),
    )


def test_known_optimum_two_transfers_one_link():
    """Spec example: max 2*bw0 + 1*bw1 s.t. bw0+bw1<=100, bw0<=60, bw1<=100.

    Optimum: bw0=60 (saturate cap), bw1=40 (fill remaining capacity).
    Objective = 2*60 + 40 = 160."""

    response, plan = plan_bandwidth(_two_transfer_single_link())
    assert response.status is SolverStatus.OPTIMAL
    assert plan is not None
    by_id = {a.transfer_id: a.bandwidth for a in plan.allocations}
    assert by_id["t0"] == pytest.approx(60.0, abs=1e-3)
    assert by_id["t1"] == pytest.approx(40.0, abs=1e-3)
    assert plan.objective_value == pytest.approx(160.0, abs=1e-3)


def test_link_capacity_respected():
    plan_input = BandwidthPlanInput(
        transfers=(
            TransferDemand("t0", bytes_=1, weight=1.0, max_bandwidth=200.0, link_id="L0"),
            TransferDemand("t1", bytes_=1, weight=1.0, max_bandwidth=200.0, link_id="L0"),
        ),
        links=(LinkCapacity("L0", capacity=50.0),),
    )
    response, plan = plan_bandwidth(plan_input)
    assert plan is not None
    total = sum(a.bandwidth for a in plan.allocations)
    assert total <= 50.0 + 1e-6


def test_two_links_independent():
    plan_input = BandwidthPlanInput(
        transfers=(
            TransferDemand("a0", bytes_=1, weight=1.0, max_bandwidth=200.0, link_id="L0"),
            TransferDemand("b0", bytes_=1, weight=1.0, max_bandwidth=200.0, link_id="L1"),
        ),
        links=(LinkCapacity("L0", capacity=80.0), LinkCapacity("L1", capacity=120.0)),
    )
    response, plan = plan_bandwidth(plan_input)
    assert plan is not None
    by_id = {a.transfer_id: a.bandwidth for a in plan.allocations}
    assert by_id["a0"] == pytest.approx(80.0, abs=1e-3)
    assert by_id["b0"] == pytest.approx(120.0, abs=1e-3)


def test_infeasible_when_min_demand_exceeds_capacity():
    plan_input = BandwidthPlanInput(
        transfers=(
            TransferDemand("t0", bytes_=1, min_bandwidth=60.0, weight=1.0, link_id="L0"),
            TransferDemand("t1", bytes_=1, min_bandwidth=60.0, weight=1.0, link_id="L0"),
        ),
        links=(LinkCapacity("L0", capacity=100.0),),
    )
    response, plan = plan_bandwidth(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_falls_back_to_highs_when_mosek_unavailable():
    reg = SolverBackendRegistry()

    class _UnavailableMosek(MosekBackend):
        def probe(self):  # type: ignore[override]
            return BackendProbeResult(
                backend=SolverBackendName.MOSEK,
                availability=BackendAvailabilityStatus.LICENSE_MISSING,
            )

    reg.register(_UnavailableMosek())
    reg.register(HighsBackend())
    response, plan = plan_bandwidth(_two_transfer_single_link(), registry=reg)
    assert response.selected_backend is SolverBackendName.HIGHS
    assert response.status is SolverStatus.OPTIMAL


def test_no_lp_backend_returns_blocked():
    reg = SolverBackendRegistry()
    response, plan = plan_bandwidth(_two_transfer_single_link(), registry=reg)
    assert response.status is SolverStatus.BLOCKED
    assert plan is None
    assert response.infeasibility_reason == "no_lp_milp_backend"


def test_mosek_native_path_runs_when_selected(monkeypatch):
    """When MOSEK is selected by routing, the bandwidth solve must
    go through ``_solve_lp_mosek`` — not scipy. Regression guard
    mirroring the memory_planner pin."""

    pytest.importorskip("mosek")
    from compgen.solve.backend_registry import default_registry
    from compgen.solve.solver_types import BackendAvailabilityStatus

    reg = default_registry()
    if reg.probe(SolverBackendName.MOSEK).availability is not BackendAvailabilityStatus.AVAILABLE:
        pytest.skip("MOSEK unavailable on this host")

    import scipy.optimize
    original_linprog = scipy.optimize.linprog
    scipy.optimize.linprog = lambda *a, **k: pytest.fail(
        "bandwidth MOSEK path accidentally called scipy.optimize.linprog"
    )
    try:
        response, plan = plan_bandwidth(_two_transfer_single_link(), registry=reg)
    finally:
        scipy.optimize.linprog = original_linprog

    assert response.selected_backend is SolverBackendName.MOSEK
    assert response.status is SolverStatus.OPTIMAL
    assert plan is not None
    by_id = {a.transfer_id: a.bandwidth for a in plan.allocations}
    assert by_id["t0"] == pytest.approx(60.0, abs=1e-3)
    assert by_id["t1"] == pytest.approx(40.0, abs=1e-3)


def test_unknown_link_in_transfer_returns_infeasible():
    plan_input = BandwidthPlanInput(
        transfers=(TransferDemand("t0", bytes_=1, weight=1.0, link_id="ghost"),),
        links=(LinkCapacity("L0", capacity=100.0),),
    )
    response, plan = plan_bandwidth(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


def test_zero_weight_transfer_gets_zero_bandwidth():
    plan_input = BandwidthPlanInput(
        transfers=(
            TransferDemand("greedy", bytes_=1, weight=10.0, max_bandwidth=100.0, link_id="L0"),
            TransferDemand("free", bytes_=1, weight=0.0, max_bandwidth=100.0, link_id="L0"),
        ),
        links=(LinkCapacity("L0", capacity=80.0),),
    )
    response, plan = plan_bandwidth(plan_input)
    assert plan is not None
    by_id = {a.transfer_id: a.bandwidth for a in plan.allocations}
    assert by_id["greedy"] == pytest.approx(80.0, abs=1e-3)
    # Zero-weight transfer gets 0 bandwidth (any non-zero allocation
    # would only hurt the objective).
    assert by_id["free"] == pytest.approx(0.0, abs=1e-3)
