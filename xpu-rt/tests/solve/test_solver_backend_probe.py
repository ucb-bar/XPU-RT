"""backend availability probe tests.

These tests pass whether MOSEK is installed/licensed or not. The
baseline requirement (Z3 + OR-Tools + HiGHS) is asserted; MOSEK is
verified to return a typed availability status — not raise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compgen.solve.backend_registry import SolverBackendRegistry, default_registry
from compgen.solve.backends.highs_backend import HighsBackend
from compgen.solve.backends.mosek_backend import MosekBackend, ensure_mosek_license_env
from compgen.solve.backends.ortools_cp_sat_backend import OrToolsCpSatBackend
from compgen.solve.backends.z3_backend import Z3Backend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)


def test_z3_probe_returns_typed_status():
    result = Z3Backend().probe()
    # Z3 must be installed for the baseline to be ok.
    assert result.availability in (
        BackendAvailabilityStatus.AVAILABLE,
        BackendAvailabilityStatus.IMPORT_MISSING,
    )
    if result.availability is BackendAvailabilityStatus.AVAILABLE:
        assert result.version is not None


def test_ortools_probe_returns_typed_status():
    result = OrToolsCpSatBackend().probe()
    assert result.availability in (
        BackendAvailabilityStatus.AVAILABLE,
        BackendAvailabilityStatus.IMPORT_MISSING,
    )


def test_mosek_probe_returns_typed_status():
    # MOSEK may legitimately be import_missing, license_missing, or available.
    result = MosekBackend().probe()
    assert result.availability in (
        BackendAvailabilityStatus.AVAILABLE,
        BackendAvailabilityStatus.IMPORT_MISSING,
        BackendAvailabilityStatus.LICENSE_MISSING,
        BackendAvailabilityStatus.LICENSE_TOKEN_UNAVAILABLE,
        BackendAvailabilityStatus.PROBE_ERROR,
    )


def test_highs_probe_returns_typed_status():
    result = HighsBackend().probe()
    assert result.availability in (
        BackendAvailabilityStatus.AVAILABLE,
        BackendAvailabilityStatus.IMPORT_MISSING,
        BackendAvailabilityStatus.PROBE_ERROR,
    )


def test_baseline_available_on_this_host():
    reg = default_registry()
    available = set(reg.available_backends())
    # Hard baseline: Z3 + OR-Tools + at least one of {HiGHS, MOSEK}
    assert SolverBackendName.Z3 in available, "Z3 must be available for baseline"
    assert SolverBackendName.ORTOOLS_CP_SAT in available, "OR-Tools must be available for baseline"
    assert (SolverBackendName.HIGHS in available) or (
        SolverBackendName.MOSEK in available
    ), "Need at least HiGHS or MOSEK for LP/MILP baseline"


def test_mosek_license_env_auto_set_from_repo_local(tmp_path, monkeypatch):
    monkeypatch.delenv("MOSEKLM_LICENSE_FILE", raising=False)
    # Build a fake mosek.lic at a synthetic repo root.
    fake_root = tmp_path / "compgen"
    (fake_root / "python" / "compgen" / "solve" / "backends").mkdir(parents=True)
    fake_lic = fake_root / "mosek.lic"
    fake_lic.write_text("FAKE LICENSE — not actually used; never read.")
    # Monkey-patch the function's _repo_root computation by adjusting
    # MOSEKLM_LICENSE_FILE detection: we pass the real repo's lic if
    # present, otherwise simulate.
    from compgen.solve.backends import mosek_backend

    monkeypatch.setattr(mosek_backend, "_repo_root", lambda: fake_root)
    chosen = ensure_mosek_license_env()
    assert chosen == str(fake_lic)
    assert os.environ["MOSEKLM_LICENSE_FILE"] == str(fake_lic)


def test_mosek_license_env_respects_existing(monkeypatch):
    monkeypatch.setenv("MOSEKLM_LICENSE_FILE", "/some/explicit/path")
    chosen = ensure_mosek_license_env()
    assert chosen == "/some/explicit/path"


def test_probe_cli_emits_json_and_md(tmp_path):
    import importlib.util

    script_path = Path("scripts/dev/probe_solver_backends.py").resolve()
    spec = importlib.util.spec_from_file_location("probe_solver_backends", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    rc = mod.main(["--out", str(tmp_path)])
    assert rc in (0, 2)  # 0 = baseline ok; 2 = baseline missing (still valid output)
    json_path = tmp_path / "solver_backend_status.json"
    md_path = tmp_path / "solver_backend_status.md"
    assert json_path.exists()
    assert md_path.exists()
    body = json.loads(json_path.read_text())
    assert body["schema_version"] == "solver_backend_status_v1"
    assert set(body["backends"].keys()) == {"z3", "ortools_cp_sat", "mosek", "highs"}
    for backend_body in body["backends"].values():
        assert backend_body["availability"] in {a.value for a in BackendAvailabilityStatus}


def test_backend_probe_request_succeeds_for_available_backends():
    reg = default_registry()
    for name in reg.available_backends():
        impl = reg.get_backend(name)
        assert impl is not None
        request = SolverRequest(
            problem_id="probe_test",
            problem_kind=SolverProblemKind.BACKEND_PROBE,
            formulation={"hello": "world"},
        )
        response = impl.solve(request)
        assert response.status in (SolverStatus.OPTIMAL, SolverStatus.PROVED), (
            f"{name.value} probe returned {response.status}"
        )
        assert response.formulation_hash == request.formulation_hash
