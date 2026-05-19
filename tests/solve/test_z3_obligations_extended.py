"""Extended Z3 obligation harness tests.

One positive (provable) and one fault (counterexample) per new
obligation kind, plus a smoke test that ``solve_request`` dispatches
the new kinds correctly.
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from xpu_rt.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)
from xpu_rt.solve.z3_obligations import (
    OBLIGATION_KIND_ALIAS_DISJOINTNESS,
    OBLIGATION_KIND_COST_MONOTONICITY,
    OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
    OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
    prove_alias_disjointness,
    prove_cost_monotonicity,
    prove_layout_in_bounds,
    prove_no_accumulator_overflow,
    solve_request,
)


def _probe_available() -> BackendProbeResult:
    return BackendProbeResult(
        backend=SolverBackendName.Z3,
        availability=BackendAvailabilityStatus.AVAILABLE,
        version="test",
    )


# ----- alias_disjointness ------------------------------------------------


def test_alias_disjointness_proves_when_offsets_separated():
    status, _, _ = prove_alias_disjointness(
        in_buf_size=256, out_buf_size=256,
        tile_lo=0, tile_hi=16, write_offset=0, read_offset=64,
    )
    assert status is SolverStatus.PROVED


def test_alias_disjointness_yields_counterexample_when_write_ahead():
    status, cex, _ = prove_alias_disjointness(
        in_buf_size=128, out_buf_size=128,
        tile_lo=0, tile_hi=16, write_offset=8, read_offset=0,
    )
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None
    assert cex["i"] > cex["j"]


def test_alias_disjointness_no_alias_is_trivially_safe():
    status, _, _ = prove_alias_disjointness(
        in_buf_size=128, out_buf_size=128,
        tile_lo=0, tile_hi=16, write_offset=0, read_offset=0, alias=False,
    )
    assert status is SolverStatus.PROVED


def test_alias_disjointness_unsupported_for_large_span():
    status, _, detail = prove_alias_disjointness(
        in_buf_size=4096, out_buf_size=4096,
        tile_lo=0, tile_hi=4096, write_offset=0, read_offset=0,
    )
    assert status is SolverStatus.UNSUPPORTED
    assert "exceeds" in detail


# ----- no_accumulator_overflow ------------------------------------------


def test_accumulator_overflow_proves_int64_on_int8():
    status, _, _ = prove_no_accumulator_overflow(
        M=128, N=128, K=128, accum_bits=64, input_min=-128, input_max=127,
    )
    assert status is SolverStatus.PROVED


def test_accumulator_overflow_yields_counterexample_int8():
    status, cex, _ = prove_no_accumulator_overflow(
        M=128, N=128, K=128, accum_bits=8, input_min=-128, input_max=127,
    )
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None


# ----- cost_monotonicity -------------------------------------------------


def test_cost_monotonicity_proves_for_product():
    status, _, _ = prove_cost_monotonicity(
        cost_expr=lambda m, n, k: m * n * k,
    )
    assert status is SolverStatus.PROVED


def test_cost_monotonicity_rejects_negative_slope():
    status, cex, _ = prove_cost_monotonicity(
        cost_expr=lambda m, n, k: -m + n + k,
    )
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None
    assert cex["m_a"] < cex["m_b"]


# ----- layout_in_bounds --------------------------------------------------


def test_layout_in_bounds_proves_contiguous():
    status, _, _ = prove_layout_in_bounds(
        buf_size_bytes=48, dim_min=[3, 4], dim_max=[3, 4],
        stride_bytes=[16, 4], alignment=4,
    )
    assert status is SolverStatus.PROVED


def test_layout_in_bounds_rejects_oversized_stride():
    status, cex, _ = prove_layout_in_bounds(
        buf_size_bytes=48, dim_min=[3, 4], dim_max=[3, 4],
        stride_bytes=[64, 4], alignment=4,
    )
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None


# ----- solve_request dispatch -------------------------------------------


def test_solve_request_dispatches_alias_disjointness():
    request = SolverRequest(
        problem_id="ext_alias",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_ALIAS_DISJOINTNESS,
            "params": dict(
                in_buf_size=256, out_buf_size=256,
                tile_lo=0, tile_hi=16, write_offset=0, read_offset=64,
            ),
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.PROVED


def test_solve_request_dispatches_no_accumulator_overflow_negative():
    request = SolverRequest(
        problem_id="ext_overflow_neg",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
            "params": dict(M=128, N=128, K=128, accum_bits=8, input_min=-128, input_max=127),
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.SAT_COUNTEREXAMPLE


def test_solve_request_dispatches_layout_in_bounds():
    request = SolverRequest(
        problem_id="ext_layout",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
            "params": dict(
                buf_size_bytes=48, dim_min=[3, 4], dim_max=[3, 4],
                stride_bytes=[16, 4], alignment=4,
            ),
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.PROVED


def test_solve_request_cost_monotonicity_requires_callable():
    # The envelope path can't deliver a callable through the JSON
    # formulation; absent one, dispatch must surface a typed ERROR
    # rather than silently passing.
    request = SolverRequest(
        problem_id="ext_cost_no_callable",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_COST_MONOTONICITY,
            "params": {},
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.ERROR
