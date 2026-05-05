"""Tests for compgen.audit.negative_controls (M-31A.5).

Each control must raise its declared typed error. These tests prove
the gates are real — that the audit doesn't silently swallow injected
faults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.audit.negative_controls import (
    NegativeControlOutcome,
    control_applies_when_predicate_violated,
    control_certificate_artifact_hash_changed,
    control_evidence_pack_source_missing,
    control_pass_card_missing,
    control_pass_precondition_violation,
    control_promotion_contract_hash_corrupted,
    control_replay_input_hash_mismatch,
    control_stale_analysis_consumed,
    control_task_pack_missing_required_file,
    run_all_negative_controls,
)


@pytest.mark.parametrize("control,expected", [
    (control_evidence_pack_source_missing, "EvidencePackSourceMissing"),
    (control_promotion_contract_hash_corrupted, "ContractHashMismatch"),
    (control_applies_when_predicate_violated, "AppliesWhenViolation"),
    (control_certificate_artifact_hash_changed, "CertificateInvalidated"),
    (control_pass_card_missing, "MissingPassCard"),
    # M-33: real fault injection now raises VerificationGateMissing
    # (the closest existing typed error for "preconditions for safe
    # consumption are not satisfied"). PreconditionViolation remains
    # as the family-level placeholder for M-34's per-pass IR-level
    # precondition check.
    (control_pass_precondition_violation, "VerificationGateMissing"),
    # M-33: StaleAnalysisAudit is the family root for stale-summary
    # failures; UnannouncedInvalidation is a subclass. The control
    # declares expected_error=StaleAnalysisAudit so a future
    # consumer-side stale-read injection can also resolve to the same
    # row without a parametrize-table change.
    (control_stale_analysis_consumed, "StaleAnalysisAudit"),
])
def test_synthetic_controls_raise_expected(control, expected, tmp_path: Path) -> None:
    outcome = control(tmp_path)
    assert outcome.raised, f"control {outcome.name} did not raise"
    assert outcome.actual_error == expected, (
        f"control {outcome.name}: expected {expected}, got {outcome.actual_error}"
    )
    assert outcome.passes


def test_task_pack_missing_required_file_control(tmp_path: Path) -> None:
    outcome = control_task_pack_missing_required_file(tmp_path)
    assert outcome.passes
    assert outcome.actual_error == "TaskPackIncomplete"


def test_replay_input_hash_mismatch_control(tmp_path: Path) -> None:
    outcome = control_replay_input_hash_mismatch(tmp_path)
    assert outcome.passes
    assert outcome.actual_error == "ReplayHashMismatch"


def test_run_all_negative_controls_all_pass(tmp_path: Path) -> None:
    """The aggregator: all 9 controls fire correctly."""
    report = run_all_negative_controls(tmp_path)
    assert len(report.outcomes) == 9
    failures = [o for o in report.outcomes if not o.passes]
    assert not failures, (
        f"failing controls: {[(o.name, o.expected_error, o.actual_error) for o in failures]}"
    )
    assert report.all_pass


def test_negative_control_outcome_passes_only_on_match() -> None:
    o = NegativeControlOutcome(
        name="x", expected_error="A", raised=True, actual_error="B",
    )
    assert o.passes is False
    o2 = NegativeControlOutcome(
        name="x", expected_error="A", raised=True, actual_error="A",
    )
    assert o2.passes is True
    o3 = NegativeControlOutcome(
        name="x", expected_error="A", raised=False, actual_error="",
    )
    assert o3.passes is False
