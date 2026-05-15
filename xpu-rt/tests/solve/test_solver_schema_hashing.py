"""Solver envelope schema + formulation-hash invariants.

Spec §3 + §17: Every solver call carries a typed envelope; the
formulation hash is canonical-JSON SHA256[:16] and byte-stable
across reruns and JSON round-trips. These tests pin the contract
that every Phase E artifact downstream relies on.
"""

from __future__ import annotations

import json
import re

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


# ---------------------------------------------------------------------------
# formulation_hash invariants
# ---------------------------------------------------------------------------


def test_hash_is_16_lowercase_hex():
    h = compute_formulation_hash({"x": 1})
    assert re.fullmatch(r"[0-9a-f]{16}", h), h


def test_hash_stable_under_key_reordering():
    a = {"x": 1, "y": 2, "z": {"inner": [1, 2, 3]}}
    b = {"z": {"inner": [1, 2, 3]}, "y": 2, "x": 1}
    assert compute_formulation_hash(a) == compute_formulation_hash(b)


def test_hash_changes_when_value_changes():
    h1 = compute_formulation_hash({"x": 1})
    h2 = compute_formulation_hash({"x": 2})
    assert h1 != h2


def test_hash_handles_floats_repeatably():
    # repr-based float encoding insulates from platform double formatting.
    h1 = compute_formulation_hash({"a": 0.1 + 0.2, "b": 1e-9})
    h2 = compute_formulation_hash({"a": 0.1 + 0.2, "b": 1e-9})
    assert h1 == h2


def test_hash_handles_nested_lists_repeatably():
    h1 = compute_formulation_hash({"L": [[1, 2], [3, 4]]})
    h2 = compute_formulation_hash({"L": [[1, 2], [3, 4]]})
    assert h1 == h2


def test_hash_distinguishes_list_order():
    """Lists are order-significant; reordering changes the hash."""

    h1 = compute_formulation_hash({"L": [1, 2, 3]})
    h2 = compute_formulation_hash({"L": [3, 2, 1]})
    assert h1 != h2


def test_canonical_dump_strips_whitespace_and_sorts_keys():
    out = canonical_formulation_dump({"b": 2, "a": 1})
    assert out == '{"a":1,"b":2}'


def test_canonical_dump_normalises_enums():
    out = canonical_formulation_dump({"k": SolverProblemKind.PLACEMENT})
    assert "placement" in out
    assert "SolverProblemKind" not in out


def test_canonical_dump_normalises_nested_floats():
    out = canonical_formulation_dump({"x": 0.5})
    # repr(0.5) == "0.5", so the dump should embed that string.
    assert "0.5" in out


# ---------------------------------------------------------------------------
# SolverRequest invariants
# ---------------------------------------------------------------------------


def test_request_round_trip_via_dict():
    req = SolverRequest(
        problem_id="memplan_a",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        formulation={"buffers": [{"id": "b0", "size_bytes": 1024}]},
        time_budget_ms=5000,
        optimality_required=True,
        backend_preference=SolverBackendName.MOSEK,
    )
    body = req.to_dict()
    recovered = SolverRequest.from_dict(body)
    assert recovered.problem_kind is SolverProblemKind.MEMORY_ALLOCATION
    assert recovered.backend_preference is SolverBackendName.MOSEK
    assert recovered.time_budget_ms == 5000
    assert recovered.optimality_required is True
    assert recovered.formulation_hash == req.formulation_hash


def test_request_dict_carries_formulation_hash_and_schema_version():
    req = SolverRequest(
        problem_id="x",
        problem_kind=SolverProblemKind.PLACEMENT,
        formulation={"a": 1},
    )
    body = req.to_dict()
    assert body["formulation_hash"] == req.formulation_hash
    assert body["schema_version"].startswith("solver_request_v")


def test_request_problem_id_is_required():
    """SolverRequest with no problem_id is invalid (must be supplied
    so artifact filenames don't collide)."""

    with pytest.raises(TypeError):
        SolverRequest(  # type: ignore[call-arg]
            problem_kind=SolverProblemKind.PLACEMENT,
            formulation={},
        )


def test_request_problem_kind_must_be_typed_enum():
    with pytest.raises(ValueError):
        SolverProblemKind("placement_lol")


# ---------------------------------------------------------------------------
# SolverResponse invariants
# ---------------------------------------------------------------------------


_REQUIRED_RESPONSE_FIELDS = (
    "schema_version", "problem_id", "problem_kind",
    "selected_backend", "backend_availability",
    "status", "formulation_hash", "time_ms",
)


def test_response_dict_carries_required_envelope_fields():
    resp = SolverResponse(
        problem_id="x",
        problem_kind=SolverProblemKind.PLACEMENT,
        selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.OPTIMAL,
        formulation_hash="abcd1234abcd1234",
        time_ms=1.5,
    )
    body = resp.to_dict()
    for k in _REQUIRED_RESPONSE_FIELDS:
        assert k in body, k


def test_response_round_trip_via_dict():
    resp = SolverResponse(
        problem_id="y",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        selected_backend=SolverBackendName.MOSEK,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.OPTIMAL,
        formulation_hash="ffff0000ffff0000",
        time_ms=42.0,
        objective_value=3.14,
        solution_path="/tmp/path.json",
        solution={"k": "v"},
        caveats=("mosek_license_unavailable",),
    )
    body = resp.to_dict()
    rec = SolverResponse.from_dict(body)
    assert rec.objective_value == 3.14
    assert rec.solution_path == "/tmp/path.json"
    assert rec.caveats == ("mosek_license_unavailable",)


def test_response_status_must_be_typed_enum():
    with pytest.raises(ValueError):
        SolverStatus("nearly_optimal_lol")


def test_response_backend_availability_must_be_typed_enum():
    with pytest.raises(ValueError):
        BackendAvailabilityStatus("kinda_available")


def test_response_serialises_to_strict_json():
    """The response dict round-trips through ``json.dumps`` /
    ``json.loads`` byte-stable (no NaN, no Inf, sortable keys)."""

    resp = SolverResponse(
        problem_id="z",
        problem_kind=SolverProblemKind.PLACEMENT,
        selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.FEASIBLE,
        formulation_hash="dead0001dead0001",
        time_ms=0.1,
        objective_value=10.0,
    )
    body = resp.to_dict()
    serialised = json.dumps(body, sort_keys=True)
    reloaded = json.loads(serialised)
    assert reloaded["status"] == "feasible"
    assert reloaded["formulation_hash"] == "dead0001dead0001"


def test_response_canonical_serialisation_omits_python_specifics():
    """No tuples, enum classes, or sets appear in the serialised JSON
    — only JSON-native types."""

    resp = SolverResponse(
        problem_id="z",
        problem_kind=SolverProblemKind.PLACEMENT,
        selected_backend=SolverBackendName.ORTOOLS_CP_SAT,
        backend_availability=BackendAvailabilityStatus.AVAILABLE,
        status=SolverStatus.OPTIMAL,
        formulation_hash="cafe0001cafe0001",
        time_ms=0.0,
        logs=("first", "second"),
        caveats=("a", "b"),
    )
    body = resp.to_dict()
    # Tuples become lists.
    assert body["logs"] == ["first", "second"]
    assert body["caveats"] == ["a", "b"]
    # Enums become strings.
    assert body["status"] == "optimal"
    assert body["selected_backend"] == "ortools_cp_sat"


# ---------------------------------------------------------------------------
# Negative envelope contracts
# ---------------------------------------------------------------------------


def test_response_from_dict_rejects_missing_required_fields():
    body = {
        "schema_version": "solver_response_v1",
        # missing problem_id
        "problem_kind": "placement",
        "selected_backend": "ortools_cp_sat",
        "backend_availability": "available",
        "status": "optimal",
        "formulation_hash": "x",
        "time_ms": 0.0,
    }
    with pytest.raises(KeyError):
        SolverResponse.from_dict(body)


def test_response_from_dict_rejects_unknown_status():
    body = {
        "schema_version": "solver_response_v1",
        "problem_id": "x",
        "problem_kind": "placement",
        "selected_backend": "ortools_cp_sat",
        "backend_availability": "available",
        "status": "made_up_status",
        "formulation_hash": "x",
        "time_ms": 0.0,
    }
    with pytest.raises(ValueError):
        SolverResponse.from_dict(body)
