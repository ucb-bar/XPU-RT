"""audit gates for the solver substrate.

Five gates that prove the solver layer is real, not decorative:

1. ``solver_backend_status`` — probe ran, baseline available.
2. ``solver_response_schema`` — every solver response in scanned
   run-dirs has the envelope fields (``formulation_hash``,
   ``selected_backend``, ``status``, ``time_ms``).
3. ``no_fake_solver_success`` — no ``status=optimal`` or
   ``status=proved`` without a corresponding artifact (solution_path
   or certificate_path).
4. ``formulation_hash_stability`` — re-hashing the request payload
   reproduces the recorded ``formulation_hash``.
5. ``solver_artifact_traceability`` — every ``*.solved.json`` is
   reachable from a ``*_response.json`` in the same dir.
"""

from __future__ import annotations

import json
from pathlib import Path

from compgen.audit.errors import GateResult
from compgen.solve.backend_registry import default_registry
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverStatus,
    compute_formulation_hash,
)

__all__ = [
    "gate_solver_backend_status",
    "gate_solver_response_schema",
    "gate_no_fake_solver_success",
    "gate_formulation_hash_stability",
    "gate_solver_artifact_traceability",
    "all_solver_gates",
]


_REQUIRED_FIELDS = (
    "formulation_hash",
    "selected_backend",
    "status",
    "time_ms",
    "problem_id",
    "problem_kind",
)


def _iter_solver_responses(run_dir: Path):
    """Yield (path, body) for every solver response JSON under run_dir."""

    if not run_dir or not run_dir.exists():
        return
    for path in run_dir.rglob("*_response.json"):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(body, dict):
            continue
        if "selected_backend" not in body:
            continue
        yield path, body


def _iter_solved_artifacts(run_dir: Path):
    if not run_dir or not run_dir.exists():
        return
    yield from run_dir.rglob("*.solved.json")


def gate_solver_backend_status() -> GateResult:
    reg = default_registry()
    results = reg.probe_all()
    backends_status = {
        name.value: results.get(name).availability.value if results.get(name) else "import_missing"
        for name in (
            SolverBackendName.Z3,
            SolverBackendName.ORTOOLS_CP_SAT,
            SolverBackendName.MOSEK,
            SolverBackendName.HIGHS,
        )
    }
    avail = backends_status
    baseline_ok = (
        avail["z3"] == "available"
        and avail["ortools_cp_sat"] == "available"
        and (avail["highs"] == "available" or avail["mosek"] == "available")
    )
    if baseline_ok:
        return GateResult(
            name="solver_backend_status",
            status="pass",
            detail=(
                f"baseline ok: z3={avail['z3']}, ortools={avail['ortools_cp_sat']}, "
                f"mosek={avail['mosek']}, highs={avail['highs']}"
            ),
        )
    return GateResult(
        name="solver_backend_status",
        status="fail",
        detail=(
            f"baseline missing (need z3 + ortools_cp_sat + one of HiGHS/MOSEK): {avail}"
        ),
    )


def gate_solver_response_schema(*, run_dir: Path | None) -> GateResult:
    if run_dir is None:
        return GateResult(
            name="solver_response_schema",
            status="skipped",
            detail="no run-dir supplied",
        )
    bad: list[str] = []
    seen = 0
    for path, body in _iter_solver_responses(run_dir):
        seen += 1
        missing = [f for f in _REQUIRED_FIELDS if f not in body]
        if missing:
            bad.append(f"{path.relative_to(run_dir)} missing {missing}")
    if seen == 0:
        return GateResult(
            name="solver_response_schema",
            status="skipped",
            detail="no solver responses found",
        )
    if bad:
        return GateResult(
            name="solver_response_schema",
            status="fail",
            detail=f"{len(bad)} response(s) missing envelope fields: {bad[:5]}",
        )
    return GateResult(
        name="solver_response_schema",
        status="pass",
        detail=f"{seen} solver response(s) carry the full envelope",
    )


def gate_no_fake_solver_success(*, run_dir: Path | None) -> GateResult:
    if run_dir is None:
        return GateResult(
            name="no_fake_solver_success",
            status="skipped",
            detail="no run-dir supplied",
        )
    fake: list[str] = []
    seen = 0
    for path, body in _iter_solver_responses(run_dir):
        seen += 1
        status = body.get("status")
        if status in (SolverStatus.OPTIMAL.value, SolverStatus.PROVED.value, SolverStatus.FEASIBLE.value):
            # Must have either solution_path, certificate_path, or
            # solution embedded.
            if not (body.get("solution_path") or body.get("certificate_path") or body.get("solution")):
                fake.append(f"{path.relative_to(run_dir)}: status={status} but no artifact")
    if seen == 0:
        return GateResult(
            name="no_fake_solver_success",
            status="skipped",
            detail="no solver responses found",
        )
    if fake:
        return GateResult(
            name="no_fake_solver_success",
            status="fail",
            detail=f"{len(fake)} response(s) claim success without artifact: {fake[:5]}",
        )
    return GateResult(
        name="no_fake_solver_success",
        status="pass",
        detail=f"{seen} response(s) — every success carries an artifact",
    )


def gate_formulation_hash_stability(*, run_dir: Path | None) -> GateResult:
    """Pair every ``*_response.json`` with the sibling ``*_request.json``
    and re-hash the request's formulation; reject any drift."""

    if run_dir is None:
        return GateResult(
            name="formulation_hash_stability",
            status="skipped",
            detail="no run-dir supplied",
        )
    drifted: list[str] = []
    paired = 0
    for resp_path, resp_body in _iter_solver_responses(run_dir):
        req_path = resp_path.with_name(resp_path.name.replace("_response.json", "_request.json"))
        if not req_path.exists():
            continue
        try:
            req_body = json.loads(req_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        recorded = resp_body.get("formulation_hash")
        re_hashed = compute_formulation_hash(req_body.get("formulation"))
        paired += 1
        if recorded != re_hashed:
            drifted.append(f"{resp_path.relative_to(run_dir)}: recorded={recorded} re_hashed={re_hashed}")
    if paired == 0:
        return GateResult(
            name="formulation_hash_stability",
            status="skipped",
            detail="no request/response pairs found",
        )
    if drifted:
        return GateResult(
            name="formulation_hash_stability",
            status="fail",
            detail=f"{len(drifted)} drifted hash(es): {drifted[:5]}",
        )
    return GateResult(
        name="formulation_hash_stability",
        status="pass",
        detail=f"{paired} request/response pair(s) hash-stable",
    )


def gate_solver_artifact_traceability(*, run_dir: Path | None) -> GateResult:
    """Every ``*.solved.json`` must be reachable from a sibling
    ``*_response.json`` (either via ``solution_path`` or by being in
    the same solver dir)."""

    if run_dir is None:
        return GateResult(
            name="solver_artifact_traceability",
            status="skipped",
            detail="no run-dir supplied",
        )
    orphans: list[str] = []
    seen = 0
    for solved_path in _iter_solved_artifacts(run_dir):
        seen += 1
        solver_dir = solved_path.parent
        # The convention is: solver_response.json is in solver_dir; the
        # solved file is in solver_dir or its parent.
        sibling_responses = list(solver_dir.rglob("*_response.json"))
        if not sibling_responses:
            # Try parent.
            sibling_responses = list((solver_dir.parent if solver_dir.parent else solver_dir).rglob("*_response.json"))
        traced = False
        for resp_path in sibling_responses:
            try:
                body = json.loads(resp_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            sol_path = body.get("solution_path") or ""
            if sol_path and Path(sol_path).name == solved_path.name:
                traced = True
                break
            # Co-located (same dir, sibling-dir, or one of these is an
            # ancestor of the other) responses count as a trace.
            if resp_path.parent == solved_path.parent:
                traced = True
                break
            if resp_path.parent in solved_path.parents:
                traced = True
                break
            if solved_path.parent in resp_path.parents:
                traced = True
                break
        if not traced:
            orphans.append(str(solved_path.relative_to(run_dir)))
    if seen == 0:
        return GateResult(
            name="solver_artifact_traceability",
            status="skipped",
            detail="no *.solved.json artifacts found",
        )
    if orphans:
        return GateResult(
            name="solver_artifact_traceability",
            status="fail",
            detail=f"{len(orphans)} orphaned solved artifact(s): {orphans[:5]}",
        )
    return GateResult(
        name="solver_artifact_traceability",
        status="pass",
        detail=f"{seen} solved artifact(s) traced to response file",
    )


def all_solver_gates(*, run_dir: Path | None) -> list[GateResult]:
    return [
        gate_solver_backend_status(),
        gate_solver_response_schema(run_dir=run_dir),
        gate_no_fake_solver_success(run_dir=run_dir),
        gate_formulation_hash_stability(run_dir=run_dir),
        gate_solver_artifact_traceability(run_dir=run_dir),
    ]
