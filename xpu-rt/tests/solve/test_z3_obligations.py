"""Z3 obligation harness.

Production-path proofs for tile-index bounds, copy identity, and
shape-predicate implication. Negative controls verify the harness
rejects invalid obligations with concrete counterexamples.
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)
from compgen.solve.z3_obligations import (
    OBLIGATION_KIND_COPY_IDENTITY,
    OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
    OBLIGATION_KIND_TILE_INDEX_BOUNDS,
    prove_copy_identity,
    prove_shape_predicate_implication,
    prove_tile_index_bounds,
    solve_request,
)


def _probe_available() -> BackendProbeResult:
    return BackendProbeResult(
        backend=SolverBackendName.Z3,
        availability=BackendAvailabilityStatus.AVAILABLE,
        version="test",
    )


# ----- tile_index_bounds -------------------------------------------------


def test_tile_bounds_safe_len_proves():
    status, cex, detail = prove_tile_index_bounds(dim_max=1024, tile=16, use_safe_len=True)
    assert status is SolverStatus.PROVED
    assert cex is None


def test_tile_bounds_unsafe_len_yields_counterexample():
    status, cex, detail = prove_tile_index_bounds(dim_max=1024, tile=16, use_safe_len=False)
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None
    # boundary case: dim not a multiple of tile.
    assert cex["dim"] % 16 != 0


def test_tile_bounds_negative_tile_errors():
    status, cex, detail = prove_tile_index_bounds(dim_max=1024, tile=0)
    assert status is SolverStatus.ERROR


# ----- copy_identity -----------------------------------------------------


def test_copy_identity_proves():
    status, cex, _ = prove_copy_identity(lo=0, hi=64)
    assert status is SolverStatus.PROVED


def test_copy_identity_perturb_rejects():
    status, cex, _ = prove_copy_identity(lo=0, hi=64, perturb=1)
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None


def test_copy_identity_empty_range_errors():
    status, cex, _ = prove_copy_identity(lo=5, hi=5)
    assert status is SolverStatus.ERROR


# ----- shape_predicate_implication --------------------------------------


def test_implication_divisible_by_proves_weaker():
    status, cex, _ = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 16}],
        precondition={"op": "divisible_by", "var": "K", "k": 8},
    )
    assert status is SolverStatus.PROVED


def test_implication_divisible_by_rejects_stronger():
    status, cex, _ = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 16}],
        precondition={"op": "divisible_by", "var": "K", "k": 32},
    )
    assert status is SolverStatus.SAT_COUNTEREXAMPLE
    assert cex is not None
    # Concrete K: must satisfy K mod 16 == 0 but NOT K mod 32 == 0.
    assert cex["K"] % 16 == 0
    assert cex["K"] % 32 != 0


def test_implication_no_variables_returns_unsupported():
    status, _, _ = prove_shape_predicate_implication(
        variables={},
        applies_when=[],
        precondition={"op": "divisible_by", "var": "K", "k": 8},
    )
    assert status is SolverStatus.UNSUPPORTED


def test_implication_with_le_and_ge():
    status, _, _ = prove_shape_predicate_implication(
        variables={"N": {"min": 1, "max": 1024}},
        applies_when=[
            {"op": "ge", "a": "N", "b": 64},
            {"op": "le", "a": "N", "b": 256},
        ],
        precondition={"op": "ge", "a": "N", "b": 32},
    )
    assert status is SolverStatus.PROVED


def test_implication_in_set():
    status, _, _ = prove_shape_predicate_implication(
        variables={"BLOCK_K": {"min": 0, "max": 1024}},
        applies_when=[{"op": "in_set", "var": "BLOCK_K", "values": [16, 32, 64]}],
        precondition={"op": "divisible_by", "var": "BLOCK_K", "k": 8},
    )
    assert status is SolverStatus.PROVED


# ----- solve_request envelope ------------------------------------------


def test_solve_request_dispatches_tile_bounds():
    request = SolverRequest(
        problem_id="oblig_tile",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS,
            "params": {"tile": 16, "dim_max": 4096, "use_safe_len": True},
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.PROVED
    assert response.formulation_hash == request.formulation_hash
    assert response.selected_backend is SolverBackendName.Z3


def test_solve_request_dispatches_copy_identity_negative():
    request = SolverRequest(
        problem_id="oblig_copy_neg",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_COPY_IDENTITY,
            "params": {"lo": 0, "hi": 32, "perturb": 1},
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.SAT_COUNTEREXAMPLE
    assert response.counterexample is not None


def test_solve_request_dispatches_implication():
    request = SolverRequest(
        problem_id="oblig_implication",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
            "params": {
                "variables": {"K": {"min": 1, "max": 4096}},
                "applies_when": [{"op": "divisible_by", "var": "K", "k": 16}],
                "precondition": {"op": "divisible_by", "var": "K", "k": 8},
            },
        },
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.PROVED


def test_solve_request_unknown_obligation_kind_returns_unsupported():
    request = SolverRequest(
        problem_id="oblig_unknown",
        problem_kind=SolverProblemKind.PEEPHOLE_VERIFY,
        formulation={"obligation_kind": "nope", "params": {}},
    )
    response = solve_request(request, probe=_probe_available())
    assert response.status is SolverStatus.UNSUPPORTED


def test_solve_request_unavailable_returns_blocked():
    probe = BackendProbeResult(
        backend=SolverBackendName.Z3,
        availability=BackendAvailabilityStatus.IMPORT_MISSING,
        detail="missing for test",
    )
    request = SolverRequest(
        problem_id="x",
        problem_kind=SolverProblemKind.PEEPHOLE_VERIFY,
        formulation={"obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS, "params": {"tile": 16}},
    )
    response = solve_request(request, probe=probe)
    assert response.status is SolverStatus.BLOCKED
    assert response.infeasibility_reason is not None


def test_via_z3_backend_solve():
    """End-to-end: registry -> Z3Backend.solve -> z3_obligations."""

    from compgen.solve.backend_registry import default_registry

    reg = default_registry()
    backend = reg.get_backend(SolverBackendName.Z3)
    assert backend is not None
    request = SolverRequest(
        problem_id="end_to_end",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS,
            "params": {"tile": 32, "dim_max": 1024, "use_safe_len": True},
        },
    )
    response = backend.solve(request)
    assert response.status is SolverStatus.PROVED


def test_kernel_certificate_round_trips_z3_ref():
    from compgen.kernels.kernel_certificate import KernelCertificate

    cert = KernelCertificate(
        schema_version="kernel_certificate_v1",
        contract_hash="aaaa",
        task_id="t",
        region_id="r0",
        candidate_id="c0",
        accepted_at_utc="2026-05-11T00:00:00Z",
        artifact_hashes={},
        artifact_paths={},
        verifier_report_path="",
        verifier_report_hash="",
        claims={},
        z3_obligation_report_ref="04_kernel_codegen/solver/z3_obligations.json",
    )
    body = cert.to_dict()
    assert body["z3_obligation_report_ref"] == "04_kernel_codegen/solver/z3_obligations.json"
    recovered = KernelCertificate.from_dict(body)
    assert recovered.z3_obligation_report_ref == "04_kernel_codegen/solver/z3_obligations.json"
