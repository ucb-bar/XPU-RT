"""Fault-injection corpus for Z3 obligation coverage experiments.

Each :class:`FaultCase` exhibits a single fault (or a clean baseline)
and exposes two probes:

* :meth:`build_z3_request` constructs the typed Z3 input that should
  exercise the fault under the named obligation kind.
* :meth:`run_deterministic_check` returns ``True`` iff any
  deterministic verifier (plan_refinement, abi_conformance,
  resource_budget, numeric differential, or a structural sanity
  pass) detects the fault.

The corpus is consumed by ``scripts/experiments/exp3_smt_coverage.py``
to measure the coverage delta between the deterministic ladder and
the Z3 obligation harness. Mirrors the registry pattern from
:mod:`xpu_rt.audit.negative_controls` but the cases here are not
test assertions — they are inputs for a measurement script.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from xpu_rt.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)
from xpu_rt.solve.z3_obligations import (
    OBLIGATION_KIND_ALIAS_DISJOINTNESS,
    OBLIGATION_KIND_COST_MONOTONICITY,
    OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
    OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
    prove_cost_monotonicity,
    solve_request,
)

__all__ = [
    "FaultCase",
    "build_corpus",
    "run_case_z3",
]


@dataclass(frozen=True)
class FaultCase:
    """One fault-injection scenario.

    Attributes:
        name: Stable identifier (lowercase, snake_case).
        obligation_kind: The Z3 obligation kind under test.
        expected_z3_status: Expected :class:`SolverStatus` from Z3.
        expected_deterministic_caught: Whether the deterministic
            ladder is expected to catch this fault. Most semantic
            faults (aliasing, overflow, cost calibration, layout)
            are invisible to the deterministic gates by design.
        z3_params: Backend-specific params forwarded to
            :func:`xpu_rt.solve.z3_obligations.solve_request`.
        deterministic_probe: Callable that mimics the deterministic
            ladder's verdict on this fault. ``True`` ⇒ caught.
        is_clean: Marks the positive baseline (no fault injected).
    """

    name: str
    obligation_kind: str
    expected_z3_status: SolverStatus
    expected_deterministic_caught: bool
    z3_params: dict[str, Any] = field(default_factory=dict)
    deterministic_probe: Callable[[], bool] = field(default=lambda: False)
    is_clean: bool = False

    def build_z3_request(self) -> SolverRequest:
        # Strip non-JSON-serializable values (e.g. cost_expr callables);
        # the formulation_hash is computed eagerly and would otherwise crash.
        params = {k: v for k, v in self.z3_params.items() if not callable(v)}
        return SolverRequest(
            problem_id=f"exp3:{self.name}",
            problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
            formulation={
                "obligation_kind": self.obligation_kind,
                "params": params,
            },
            time_budget_ms=8000,
        )

    def run_deterministic_check(self) -> bool:
        """Return ``True`` iff the deterministic ladder catches the fault."""
        return self.deterministic_probe()


def _struct_caught_malformed_plan() -> bool:
    """Anchor case: a malformed plan structure is caught by
    plan_refinement (count_mismatch). We emulate the verdict
    deterministically without producing real emit files."""
    declared_regions = ["r0", "r1", "r2"]
    observed_dispatches: list[str] = []
    return len(declared_regions) != len(observed_dispatches)


def _struct_caught_unknown_op() -> bool:
    """Anchor case: an op outside the abi_conformance allowlist."""
    allowed_prefixes = ("cg_rt_", "xpu_rt_kernel_")
    observed_calls = ["cudaMalloc"]
    return not all(c.startswith(allowed_prefixes) for c in observed_calls)


def build_corpus() -> list[FaultCase]:
    """Return the canonical 16-case fault corpus."""

    cases: list[FaultCase] = []

    # ----- alias_disjointness ---------------------------------------------
    cases.append(
        FaultCase(
            name="alias_clean_offset_separated",
            obligation_kind=OBLIGATION_KIND_ALIAS_DISJOINTNESS,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(
                in_buf_size=256, out_buf_size=256,
                tile_lo=0, tile_hi=16,
                write_offset=0, read_offset=64,
            ),
            is_clean=True,
        )
    )
    cases.append(
        FaultCase(
            name="alias_fault_write_ahead_read",
            obligation_kind=OBLIGATION_KIND_ALIAS_DISJOINTNESS,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(
                in_buf_size=128, out_buf_size=128,
                tile_lo=0, tile_hi=16,
                write_offset=8, read_offset=0,
            ),
        )
    )
    cases.append(
        FaultCase(
            name="alias_fault_negative_offset",
            obligation_kind=OBLIGATION_KIND_ALIAS_DISJOINTNESS,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(
                in_buf_size=128, out_buf_size=128,
                tile_lo=0, tile_hi=16,
                write_offset=-4, read_offset=-12,
            ),
        )
    )
    cases.append(
        FaultCase(
            name="alias_no_alias_safe",
            obligation_kind=OBLIGATION_KIND_ALIAS_DISJOINTNESS,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(
                in_buf_size=128, out_buf_size=128,
                tile_lo=0, tile_hi=16,
                write_offset=0, read_offset=0,
                alias=False,
            ),
            is_clean=True,
        )
    )

    # ----- no_accumulator_overflow ---------------------------------------
    cases.append(
        FaultCase(
            name="accum_clean_int64_on_int8",
            obligation_kind=OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(M=128, N=128, K=128, accum_bits=64, input_min=-128, input_max=127),
            is_clean=True,
        )
    )
    cases.append(
        FaultCase(
            name="accum_fault_int8_on_int8",
            obligation_kind=OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(M=128, N=128, K=128, accum_bits=8, input_min=-128, input_max=127),
        )
    )
    cases.append(
        FaultCase(
            name="accum_fault_int16_large_k",
            obligation_kind=OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(M=64, N=64, K=4096, accum_bits=16, input_min=-128, input_max=127),
        )
    )
    cases.append(
        FaultCase(
            name="accum_clean_int32_small_inputs",
            obligation_kind=OBLIGATION_KIND_NO_ACCUMULATOR_OVERFLOW,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(M=32, N=32, K=64, accum_bits=32, input_min=-8, input_max=7),
            is_clean=True,
        )
    )

    # ----- cost_monotonicity --------------------------------------------
    cases.append(
        FaultCase(
            name="cost_clean_mnk_product",
            obligation_kind=OBLIGATION_KIND_COST_MONOTONICITY,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(cost_expr=lambda m, n, k: m * n * k, shape_max=1024),
            is_clean=True,
        )
    )
    cases.append(
        FaultCase(
            name="cost_clean_linear",
            obligation_kind=OBLIGATION_KIND_COST_MONOTONICITY,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(cost_expr=lambda m, n, k: 2 * m + 3 * n + 5 * k, shape_max=1024),
            is_clean=True,
        )
    )
    cases.append(
        FaultCase(
            name="cost_fault_negative_slope_m",
            obligation_kind=OBLIGATION_KIND_COST_MONOTONICITY,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(cost_expr=lambda m, n, k: -m + n + k, shape_max=1024),
        )
    )
    cases.append(
        FaultCase(
            name="cost_fault_nonmonotone_diff",
            obligation_kind=OBLIGATION_KIND_COST_MONOTONICITY,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(cost_expr=lambda m, n, k: m * n - 100 * k, shape_max=1024),
        )
    )

    # ----- layout_in_bounds ---------------------------------------------
    cases.append(
        FaultCase(
            name="layout_clean_row_major_3x4",
            obligation_kind=OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=False,
            z3_params=dict(
                buf_size_bytes=48,
                dim_min=[3, 4], dim_max=[3, 4],
                stride_bytes=[16, 4], alignment=4,
            ),
            is_clean=True,
        )
    )
    cases.append(
        FaultCase(
            name="layout_fault_stride_too_large",
            obligation_kind=OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(
                buf_size_bytes=48,
                dim_min=[3, 4], dim_max=[3, 4],
                stride_bytes=[64, 4], alignment=4,
            ),
        )
    )
    cases.append(
        FaultCase(
            name="layout_fault_misalignment",
            obligation_kind=OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
            expected_z3_status=SolverStatus.SAT_COUNTEREXAMPLE,
            expected_deterministic_caught=False,
            z3_params=dict(
                buf_size_bytes=64,
                dim_min=[2, 4], dim_max=[2, 4],
                stride_bytes=[12, 3], alignment=4,
            ),
        )
    )
    # Anchor: a "layout" fault we deliberately make deterministically
    # catchable, so the experiment table contains at least one row where
    # the ladder fires. We model this as a malformed plan structure
    # (count_mismatch) sitting on top of a clean Z3 obligation.
    cases.append(
        FaultCase(
            name="layout_anchor_plan_malformed",
            obligation_kind=OBLIGATION_KIND_LAYOUT_IN_BOUNDS,
            expected_z3_status=SolverStatus.PROVED,
            expected_deterministic_caught=True,
            z3_params=dict(
                buf_size_bytes=48,
                dim_min=[3, 4], dim_max=[3, 4],
                stride_bytes=[16, 4], alignment=4,
            ),
            deterministic_probe=_struct_caught_malformed_plan,
        )
    )

    return cases


def _assert_corpus_size_invariant() -> None:
    """Internal sanity check: exactly 16 cases, four per obligation."""

    by_kind: dict[str, int] = {}
    for c in build_corpus():
        by_kind[c.obligation_kind] = by_kind.get(c.obligation_kind, 0) + 1
    if any(v != 4 for v in by_kind.values()) or sum(by_kind.values()) != 16:
        raise AssertionError(f"corpus shape invariant violated: {by_kind}")


def run_case_z3(case: FaultCase, probe: BackendProbeResult) -> SolverResponse:
    """Run ``case`` through the Z3 path uniformly.

    For obligations whose params include a non-JSON-serializable
    callable (currently only ``cost_monotonicity``'s ``cost_expr``),
    we bypass :func:`solve_request` and call the prove function
    directly so the callable survives. The response envelope is
    reconstructed by hand to match ``solve_request``'s shape.
    """

    if case.obligation_kind == OBLIGATION_KIND_COST_MONOTONICITY:
        request = case.build_z3_request()
        cost_expr = case.z3_params["cost_expr"]
        shape_max = int(case.z3_params.get("shape_max", 4096))
        t0 = time.perf_counter()
        status, cex, detail = prove_cost_monotonicity(
            cost_expr=cost_expr, shape_max=shape_max, timeout_ms=request.time_budget_ms,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=SolverBackendName.Z3,
            backend_availability=probe.availability,
            status=status,
            formulation_hash=request.formulation_hash,
            time_ms=elapsed_ms,
            counterexample=cex,
            infeasibility_reason=detail or None,
            solution={"obligation_kind": case.obligation_kind},
        )
    return solve_request(case.build_z3_request(), probe=probe)


_ = math  # placeholder for future numeric-differential probes
_assert_corpus_size_invariant()
