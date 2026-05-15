"""Architecture guards for the solver layer.

Per the solver-validation spec §14. These tests are deliberately
narrow and structural: they pin the contract that semantic-proof
problem kinds NEVER reach a numeric solver and discrete-scheduling
kinds NEVER reach a proof solver, that every response carries the
envelope fields, and that the architecture-audit script passes on
this repo's current source tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compgen.solve.backend_registry import default_registry
from compgen.solve.backends.highs_backend import HighsBackend
from compgen.solve.backends.mosek_backend import MosekBackend
from compgen.solve.backends.ortools_cp_sat_backend import OrToolsCpSatBackend
from compgen.solve.backends.z3_backend import Z3Backend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Guard 1: solver-purpose separation
# ---------------------------------------------------------------------------


_PROOF_KINDS = [
    SolverProblemKind.PEEPHOLE_VERIFY,
    SolverProblemKind.RECIPE_REFINEMENT,
    SolverProblemKind.TRANSLATION_VALIDATION,
    SolverProblemKind.SHAPE_PREDICATE_VERIFY,
    SolverProblemKind.PLAN_INVARIANT_VERIFY,
]

_DISCRETE_KINDS = [
    SolverProblemKind.PLACEMENT,
    SolverProblemKind.SCHEDULE,
    SolverProblemKind.NO_OVERLAP_SCHEDULE,
    SolverProblemKind.EVENT_ORDERING,
    SolverProblemKind.OVERLAP_PLANNING,
]

_NUMERIC_KINDS = [
    SolverProblemKind.MEMORY_ALLOCATION,
    SolverProblemKind.BUFFER_ALIASING,
    SolverProblemKind.BANDWIDTH_ALLOCATION,
    SolverProblemKind.COST_MODEL_FIT,
]


@pytest.mark.parametrize("kind", _PROOF_KINDS)
def test_mosek_rejects_proof_kinds(kind: SolverProblemKind):
    """Hard architecture rule: a proof kind sent to MOSEK returns
    UNSUPPORTED, never a numeric optimum."""

    request = SolverRequest(
        problem_id="arch_guard_mosek_proof",
        problem_kind=kind,
        formulation={"poison": "must_not_solve"},
    )
    response = MosekBackend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED
    assert response.selected_backend is SolverBackendName.MOSEK
    assert "solver-purpose violation" in (response.infeasibility_reason or "")


@pytest.mark.parametrize("kind", _PROOF_KINDS)
def test_highs_rejects_proof_kinds(kind: SolverProblemKind):
    request = SolverRequest(
        problem_id="arch_guard_highs_proof",
        problem_kind=kind,
        formulation={"poison": "must_not_solve"},
    )
    response = HighsBackend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED


@pytest.mark.parametrize("kind", _PROOF_KINDS)
def test_ortools_rejects_proof_kinds(kind: SolverProblemKind):
    request = SolverRequest(
        problem_id="arch_guard_ortools_proof",
        problem_kind=kind,
        formulation={"poison": "must_not_solve"},
    )
    response = OrToolsCpSatBackend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED


@pytest.mark.parametrize("kind", _DISCRETE_KINDS + _NUMERIC_KINDS)
def test_z3_rejects_non_proof_kinds(kind: SolverProblemKind):
    """Z3 is the proof engine only. Sending placement / scheduling /
    memory MILP to it must NOT solve — must return UNSUPPORTED."""

    request = SolverRequest(
        problem_id="arch_guard_z3_non_proof",
        problem_kind=kind,
        formulation={"poison": "must_not_solve"},
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# Guard 2: routing safety — backend_preference cannot override solver-purpose
# ---------------------------------------------------------------------------


def test_routing_preference_cannot_send_proof_to_mosek():
    from compgen.solve.routing import choose_backend

    reg = default_registry()
    chosen = choose_backend(
        SolverProblemKind.PEEPHOLE_VERIFY,
        reg,
        preference=SolverBackendName.MOSEK,
    )
    # Either Z3 or None — MUST never be MOSEK.
    assert chosen in (SolverBackendName.Z3, None)


def test_routing_preference_cannot_send_placement_to_z3():
    from compgen.solve.routing import choose_backend

    reg = default_registry()
    chosen = choose_backend(
        SolverProblemKind.PLACEMENT,
        reg,
        preference=SolverBackendName.Z3,
    )
    assert chosen in (SolverBackendName.ORTOOLS_CP_SAT, None)


def test_routing_preference_cannot_send_memory_to_z3():
    from compgen.solve.routing import choose_backend

    reg = default_registry()
    chosen = choose_backend(
        SolverProblemKind.MEMORY_ALLOCATION,
        reg,
        preference=SolverBackendName.Z3,
    )
    assert chosen in (
        SolverBackendName.MOSEK,
        SolverBackendName.HIGHS,
        None,
    )


# ---------------------------------------------------------------------------
# Guard 3: envelope completeness
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS = ("schema_version", "problem_id", "problem_kind",
                    "selected_backend", "status", "formulation_hash", "time_ms")


def test_every_backend_emits_envelope_fields():
    reg = default_registry()
    for name in reg.available_backends():
        impl = reg.get_backend(name)
        assert impl is not None
        request = SolverRequest(
            problem_id="envelope_pin",
            problem_kind=SolverProblemKind.BACKEND_PROBE,
            formulation={"k": "v"},
        )
        response = impl.solve(request)
        body = response.to_dict()
        missing = [f for f in _REQUIRED_FIELDS if f not in body]
        assert not missing, f"{name.value} response missing fields {missing}: {body}"


def test_formulation_hash_is_deterministic_and_16_hex():
    from compgen.solve.solver_types import compute_formulation_hash
    import re

    h1 = compute_formulation_hash({"a": 1, "b": [1, 2, 3]})
    h2 = compute_formulation_hash({"b": [1, 2, 3], "a": 1})
    assert h1 == h2
    assert re.fullmatch(r"[0-9a-f]{16}", h1)


# ---------------------------------------------------------------------------
# Guard 4: source scanner
# ---------------------------------------------------------------------------


def test_audit_solver_architecture_passes_on_repo():
    """``scripts/dev/audit_solver_architecture.py`` exits 0 on the
    current source tree. Catches optional-solver imports leaking
    outside ``solve/backends/`` (or the allowlist)."""

    script = REPO_ROOT / "scripts" / "dev" / "audit_solver_architecture.py"
    assert script.is_file(), f"audit script missing: {script}"
    rc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert rc.returncode == 0, (
        f"audit failed:\nstdout: {rc.stdout.decode()}\nstderr: {rc.stderr.decode()}"
    )


def test_audit_detects_a_leak_when_planted(tmp_path: Path):
    """Synthesize a fake repo with a forbidden import inside
    ``python/compgen/`` and verify the audit script rejects it."""

    fake_root = tmp_path / "fake_repo"
    (fake_root / "python" / "compgen" / "leaky").mkdir(parents=True)
    (fake_root / "python" / "compgen" / "leaky" / "bad.py").write_text(
        "import mosek\n"
    )
    # Copy the audit script to the fake repo's scripts/dev/ so its
    # default ``REPO_ROOT`` resolves to the fake tree.
    (fake_root / "scripts" / "dev").mkdir(parents=True)
    script_src = REPO_ROOT / "scripts" / "dev" / "audit_solver_architecture.py"
    (fake_root / "scripts" / "dev" / "audit_solver_architecture.py").write_text(
        script_src.read_text()
    )
    rc = subprocess.run(
        [
            sys.executable,
            str(fake_root / "scripts" / "dev" / "audit_solver_architecture.py"),
            "--repo-root",
            str(fake_root),
        ],
        capture_output=True,
    )
    assert rc.returncode == 2, "audit must reject planted leak"
    assert b"leaky/bad.py" in rc.stderr
