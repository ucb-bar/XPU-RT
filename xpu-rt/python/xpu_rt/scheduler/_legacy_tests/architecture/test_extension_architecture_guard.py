"""spec'd path for the architecture audit tests.

Mirrors :mod:`tests.audit.test_extension_architecture_audit` but
imports through the user-spec ``compgen.extensions.architecture_guard``
re-export shim. Asserts both surfaces resolve to the same audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Spec'd path
from compgen.extensions import architecture_guard as guard
# Underlying module
from compgen.audit import extension_architecture as audit_impl


def test_spec_path_imports():
    assert hasattr(guard, "run_audit")
    assert hasattr(guard, "AuditReport")
    assert hasattr(guard, "Violation")


def test_spec_path_routes_to_real_impl():
    """The shim re-exports the same callable as the underlying module."""

    assert guard.run_audit is audit_impl.run_audit
    assert guard.AuditReport is audit_impl.AuditReport


def test_audit_through_spec_path_passes_on_current_repo():
    report = guard.run_audit()
    assert report.passed, f"violations: {[v.to_dict() for v in report.violations]}"
    assert "extension_card_completeness" in report.checks_run
    assert "execution_evidence" in report.checks_run


def test_audit_violation_reason_enum_includes_execution_evidence():
    assert "execution_evidence_malformed" in guard.VIOLATION_REASONS


def test_audit_forbidden_imports_list_is_typed():
    assert "cuda_tile" in guard.FORBIDDEN_OPTIONAL_IMPORTS
    assert "hexagon_mlir" in guard.FORBIDDEN_OPTIONAL_IMPORTS


def test_audit_with_malformed_evidence_pack(tmp_path: Path):
    """End-to-end: build a malformed evidence pack, audit through the
    spec'd path catches it."""

    bad_dir = tmp_path / "per_provider" / "rogue"
    bad_dir.mkdir(parents=True)
    (bad_dir / "kernel_source.py").write_text("x")
    (bad_dir / "run_report.json").write_text("not json")
    (bad_dir / "certificate.json").write_text("{}")
    report = guard.run_audit(evidence_pack=tmp_path)
    assert not report.passed
    assert any(
        v.reason == "execution_evidence_malformed" for v in report.violations
    )
