"""typed envelope, JSON round-trip, formulation_hash stability."""

from __future__ import annotations

import json

import pytest

from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
    canonical_formulation_dump,
    compute_formulation_hash,
)


def test_formulation_hash_stable_across_key_order():
    a = {"x": 1, "y": [1, 2, 3], "nested": {"a": 1, "b": 2}}
    b = {"y": [1, 2, 3], "nested": {"b": 2, "a": 1}, "x": 1}
    assert compute_formulation_hash(a) == compute_formulation_hash(b)


def test_formulation_hash_unicode_stable():
    f = {"variable": "α", "values": ["β", "γ"]}
    h1 = compute_formulation_hash(f)
    h2 = compute_formulation_hash(dict(f))
    assert h1 == h2
    assert len(h1) == 16  # SHA256[:16]


def test_formulation_hash_float_repr_stable():
    """``repr``-based float encoding avoids platform double formatting."""

    f1 = {"x": 0.1 + 0.2}
    f2 = {"x": 0.1 + 0.2}
    assert compute_formulation_hash(f1) == compute_formulation_hash(f2)


def test_formulation_hash_changes_with_value():
    f1 = {"x": 1}
    f2 = {"x": 2}
    assert compute_formulation_hash(f1) != compute_formulation_hash(f2)


def test_request_round_trip():
    req = SolverRequest(
        problem_id="memplan_42",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        formulation={"buffers": [{"id": "b0", "size_bytes": 1024}]},
        time_budget_ms=5000,
        optimality_required=True,
        backend_preference=SolverBackendName.MOSEK,
    )
    body = req.to_dict()
    assert body["problem_kind"] == "memory_allocation"
    assert body["backend_preference"] == "mosek"
    assert body["formulation_hash"] == req.formulation_hash
    recovered = SolverRequest.from_dict(body)
    assert recovered.problem_kind is SolverProblemKind.MEMORY_ALLOCATION
    assert recovered.backend_preference is SolverBackendName.MOSEK
    assert recovered.formulation_hash == req.formulation_hash


def test_response_round_trip():
    resp = SolverResponse(
        problem_id="memplan_42",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        selected_backend=SolverBackendName.MOSEK,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.OPTIMAL,
        formulation_hash="deadbeefcafebabe",
        time_ms=12.5,
        objective_value=42.0,
        solution_path="/tmp/run/memplan.solved.json",
    )
    body = resp.to_dict()
    assert body["status"] == "optimal"
    recovered = SolverResponse.from_dict(body)
    assert recovered.status is SolverStatus.OPTIMAL
    assert recovered.selected_backend is SolverBackendName.MOSEK
    assert recovered.formulation_hash == "deadbeefcafebabe"


def test_enums_reject_unknown_strings():
    with pytest.raises(ValueError):
        SolverProblemKind("not_a_kind")
    with pytest.raises(ValueError):
        SolverStatus("kind_of_optimal")
    with pytest.raises(ValueError):
        SolverBackendName("gurobi")


def test_canonical_dump_sorts_and_strips_whitespace():
    dumped = canonical_formulation_dump({"b": 1, "a": [1, 2]})
    assert dumped == '{"a":[1,2],"b":1}'


def test_response_serializes_logs_and_caveats():
    resp = SolverResponse(
        problem_id="x",
        problem_kind=SolverProblemKind.PLACEMENT,
        selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.FEASIBLE,
        formulation_hash="abcd",
        time_ms=1.0,
        logs=("solved", "ok"),
        caveats=("warm_start_discarded",),
    )
    body = json.loads(json.dumps(resp.to_dict()))
    assert body["logs"] == ["solved", "ok"]
    assert body["caveats"] == ["warm_start_discarded"]
