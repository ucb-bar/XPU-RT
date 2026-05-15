"""extension architecture audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.audit.extension_architecture import (
    AuditReport,
    FORBIDDEN_OPTIONAL_IMPORTS,
    VIOLATION_REASONS,
    Violation,
    check_blocked_provider_not_paper_claimable,
    check_card_completeness,
    check_optional_provider_imports_quarantined,
    check_pass_tool_no_direct_ir_mutation,
    check_provider_result_is_not_certificate,
    run_audit,
)


def test_audit_passes_on_current_repo():
    """Phase F substrate as-shipped must pass every architecture check."""
    report = run_audit()
    assert report.passed, f"{len(report.violations)} violations:\n" + "\n".join(
        f"  - {v.check}: {v.path} ({v.reason})" for v in report.violations
    )
    assert len(report.checks_run) == 6
    assert "extension_card_completeness" in report.checks_run
    assert "execution_evidence" in report.checks_run


def test_audit_reports_all_six_checks():
    report = run_audit()
    expected = {
        "extension_card_completeness",
        "blocked_provider_not_paper_claimable",
        "pass_tool_no_direct_ir_mutation",
        "optional_provider_imports_quarantined",
        "provider_result_is_not_certificate",
        "execution_evidence",
    }
    assert set(report.checks_run) == expected


def test_audit_report_round_trips_to_dict():
    report = run_audit()
    body = report.to_dict()
    assert body["schema_version"] == "extension_architecture_audit_v1"
    assert body["passed"] is True
    assert body["violation_count"] == 0
    assert isinstance(body["summary"], dict)


def test_forbidden_optional_imports_enum():
    assert "cuda_tile" in FORBIDDEN_OPTIONAL_IMPORTS
    assert "tilelang" in FORBIDDEN_OPTIONAL_IMPORTS
    assert "thunderkittens" in FORBIDDEN_OPTIONAL_IMPORTS


def test_violation_reasons_enum():
    assert "missing_integration_level" in VIOLATION_REASONS
    assert "paper_claimable_at_blocked_level" in VIOLATION_REASONS
    assert "forbidden_optional_import" in VIOLATION_REASONS


# ---------------------------------------------------------------------------
# Negative controls: synthesize a violating tree and confirm the audit
# catches each rule.
# ---------------------------------------------------------------------------


def test_optional_provider_import_violation_detected(tmp_path: Path):
    """A core module that imports an optional package is flagged."""

    fake_repo = tmp_path / "fake_repo"
    src = fake_repo / "python" / "compgen" / "core_thing.py"
    src.parent.mkdir(parents=True)
    src.write_text("import cuda_tile\n")

    report = AuditReport()
    check_optional_provider_imports_quarantined(report, repo_root=fake_repo)
    violations = [v for v in report.violations if v.reason == "forbidden_optional_import"]
    assert violations, "expected forbidden_optional_import violation"
    assert "cuda_tile" in violations[0].detail


def test_optional_import_inside_adapter_allowed(tmp_path: Path):
    """The same import in the adapter tree is permitted."""

    fake_repo = tmp_path / "fake_repo"
    src = fake_repo / "python" / "compgen" / "providers" / "adapters" / "x.py"
    src.parent.mkdir(parents=True)
    src.write_text("import cuda_tile\n")

    report = AuditReport()
    check_optional_provider_imports_quarantined(report, repo_root=fake_repo)
    forbidden = [v for v in report.violations if v.reason == "forbidden_optional_import"]
    assert not forbidden, "adapter directory must be allowed to import optional providers"


def test_provider_result_certificate_alias_violation_detected(tmp_path: Path):
    fake_repo = tmp_path / "fake_repo"
    src = fake_repo / "python" / "compgen" / "rogue.py"
    src.parent.mkdir(parents=True)
    src.write_text("ProviderResult = KernelCertificate\n")

    report = AuditReport()
    check_provider_result_is_not_certificate(report, repo_root=fake_repo)
    assert any(v.reason == "provider_result_used_as_certificate" for v in report.violations)


def test_audit_summary_records_cards_audited():
    report = run_audit()
    assert "cards_audited" in report.summary
    assert report.summary["cards_audited"] >= 1
