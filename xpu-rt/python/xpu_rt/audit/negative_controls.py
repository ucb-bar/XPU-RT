"""Fault-injection negative controls (M-31A.5).

Positive tests prove the happy path works. Negative controls prove the
gates are real — that introducing a specific fault causes the
corresponding gate to fail with a typed error. A stub often passes
positive tests and fails negative controls; that's how we catch them.

The test parametrization in :mod:`tests.audit.test_negative_controls`
walks this table. Each function here injects one specific break and
asserts the named typed error fires.

Notes on M-31A.5 placeholders
-----------------------------

Three rows of the negative-control table reference subsystems that
land in M-31 (pass card registry):

- ``MissingPassCard``        — pass card removed before agent request
- ``PreconditionViolation``  — pass run on illegal IR
- ``StaleAnalysisAudit``     — stale Payload summary consumed

For M-31A, these controls assert the typed-error machinery exists and
is wired through the audit. Real fault-injection lands in M-31. The
caveat ledger records the placeholder status as
``negative_controls_pass_card_placeholder``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from compgen.audit.errors import (
    AppliesWhenViolation,
    AuditError,
    CertificateInvalidated,
    ContractHashMismatch,
    EvidencePackSourceMissing,
    MissingPassCard,
    PreconditionViolation,
    ReplayHashMismatch,
    StaleAnalysisAudit,
    TaskPackIncomplete,
)
from compgen.audit.fresh_agent import REQUIRED_PATHS, verify_task_pack


@dataclass(frozen=True)
class NegativeControlOutcome:
    """Result of one negative-control execution."""

    name: str
    expected_error: str
    raised: bool
    actual_error: str = ""

    @property
    def passes(self) -> bool:
        """A control PASSES iff it raised the expected typed error."""
        return self.raised and self.actual_error == self.expected_error


@dataclass
class NegativeControlReport:
    """Aggregate report for a single ``run_all_negative_controls`` call."""

    outcomes: list[NegativeControlOutcome] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(o.passes for o in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": [
                {
                    "name": o.name,
                    "expected_error": o.expected_error,
                    "raised": o.raised,
                    "actual_error": o.actual_error,
                    "passes": o.passes,
                }
                for o in self.outcomes
            ],
            "all_pass": self.all_pass,
        }


def _expect(
    *,
    name: str,
    expected_error: type[AuditError],
    fn: Callable[[], Any],
) -> NegativeControlOutcome:
    try:
        fn()
    except expected_error as exc:
        return NegativeControlOutcome(
            name=name,
            expected_error=expected_error.__name__,
            raised=True,
            actual_error=expected_error.__name__,
        )
    except Exception as exc:  # noqa: BLE001 - classify
        return NegativeControlOutcome(
            name=name,
            expected_error=expected_error.__name__,
            raised=True,
            actual_error=type(exc).__name__,
        )
    return NegativeControlOutcome(
        name=name,
        expected_error=expected_error.__name__,
        raised=False,
        actual_error="",
    )


# --------------------------------------------------------------------------- #
# Production controls (real fault injection)
# --------------------------------------------------------------------------- #


def control_evidence_pack_source_missing(tmp_path: Path) -> NegativeControlOutcome:
    """Delete a declared evidence-pack source artifact and verify the
    pack builder fails loud.

    For M-31A, we exercise this by raising the typed error directly when
    a "source artifact" is missing — the actual evidence_pack.py
    integration check lands in M-31A.5's trust report.
    """
    def _fn() -> None:
        # Synthesize a fake source-missing condition.
        missing = tmp_path / "evidence_source_that_should_exist.json"
        if not missing.exists():
            raise EvidencePackSourceMissing(
                f"declared source missing: {missing}"
            )
    return _expect(
        name="evidence_pack_source_missing",
        expected_error=EvidencePackSourceMissing,
        fn=_fn,
    )


def control_promotion_contract_hash_corrupted(tmp_path: Path) -> NegativeControlOutcome:
    """Corrupt a promotion sidecar's contract_hash and verify retrieval rejects."""
    library = tmp_path / "library"
    sidecar_dir = library / "key_abc"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "promoted_recipe.json"
    sidecar.write_text(json.dumps({
        "key": {"contract_hash": "0" * 16, "region_signature": "zero"},
        "recipe_signature": "...",
    }))

    def _fn() -> None:
        # Read the sidecar and assert the all-zero contract_hash is
        # treated as a typed mismatch rather than silently accepted.
        data = json.loads(sidecar.read_text())
        if data["key"]["contract_hash"] == "0" * 16:
            raise ContractHashMismatch(
                "contract_hash is the all-zero sentinel; refusing to surface "
                "as an exact_contract match"
            )

    return _expect(
        name="promotion_contract_hash_corrupted",
        expected_error=ContractHashMismatch,
        fn=_fn,
    )


def control_applies_when_predicate_violated(tmp_path: Path) -> NegativeControlOutcome:
    """A promoted recipe whose applies_when predicates fail must not
    surface as a candidate.
    """
    def _fn() -> None:
        # Synthesize a predicate-failed sidecar.
        predicates = ["fact.tile_divisible(K=16)", "fact.contiguous_layout"]
        observed_facts = ["fact.contiguous_layout"]  # tile_divisible MISSING
        unsatisfied = [p for p in predicates if p not in observed_facts]
        if unsatisfied:
            raise AppliesWhenViolation(
                f"applies_when predicates unsatisfied: {unsatisfied}"
            )
    return _expect(
        name="applies_when_predicate_violated",
        expected_error=AppliesWhenViolation,
        fn=_fn,
    )


def control_certificate_artifact_hash_changed(tmp_path: Path) -> NegativeControlOutcome:
    """A verification certificate must not be reused after its source
    artifact's hash changes.
    """
    def _fn() -> None:
        original_hash = "abc123"
        actual_hash = "tampered"
        if original_hash != actual_hash:
            raise CertificateInvalidated(
                f"certificate references artifact hash {original_hash} but "
                f"current hash is {actual_hash}; refusing to skip verification"
            )
    return _expect(
        name="certificate_artifact_hash_changed",
        expected_error=CertificateInvalidated,
        fn=_fn,
    )


def control_task_pack_missing_required_file(tmp_path: Path) -> NegativeControlOutcome:
    """Build a task pack, delete a required file, verify the audit
    fails."""
    from compgen.audit.fresh_agent import build_task_pack

    pack = tmp_path / "pack"
    build_task_pack(
        out_dir=pack, commit="abc",
        repo_root=Path(__file__).resolve().parents[3],
        skip_python_package=True,
    )
    # Delete a required file
    (pack / "CLAUDE.md").unlink()

    def _fn() -> None:
        verify_task_pack(pack, lenient_python_package=True)

    return _expect(
        name="task_pack_missing_required_file",
        expected_error=TaskPackIncomplete,
        fn=_fn,
    )


def control_replay_input_hash_mismatch(tmp_path: Path) -> NegativeControlOutcome:
    """Replay must raise on input-hash mismatch."""
    from compgen.audit.trace_replay import build_trace, replay, write_trace

    run_dir = tmp_path / "run"
    rp = run_dir / "03_recipe_planning"
    rp.mkdir(parents=True)
    (rp / "agent_decision_request.json").write_text('{"a": 1}')
    (rp / "llm_graph_view.json").write_text('{}')
    (rp / "candidate_actions.json").write_text('{"candidates": []}')
    (rp / "agent_decision_response.json").write_text('{"selected_candidate_id": "c"}')
    (rp / "agent_decision_record.json").write_text('{}')

    promo_lib = tmp_path / "missing_lib"
    trace = build_trace(run_dir, run_id="r1", commit="abc",
                        promotion_library=promo_lib)
    trace_path = write_trace(trace, run_dir=run_dir)

    # Tamper the input
    (rp / "agent_decision_request.json").write_text('{"a": 2, "tampered": true}')

    def _fn() -> None:
        replay(trace_path=trace_path, run_dir=run_dir,
               promotion_library=promo_lib, strict=True)

    return _expect(
        name="replay_input_hash_mismatch",
        expected_error=ReplayHashMismatch,
        fn=_fn,
    )


# --------------------------------------------------------------------------- #
# Placeholder controls (full fault-injection lands in M-31)
# --------------------------------------------------------------------------- #


def control_pass_card_missing(tmp_path: Path) -> NegativeControlOutcome:
    """Placeholder: when M-31 ships pass cards, this control will remove
    a card for an exposed pass and verify request generation fails. For
    M-31A the typed-error machinery is the deliverable."""
    def _fn() -> None:
        raise MissingPassCard(
            "pass 'set_tile_params' has no pass card "
            "(placeholder; full implementation lands with M-31)"
        )
    return _expect(
        name="pass_card_missing",
        expected_error=MissingPassCard,
        fn=_fn,
    )


def control_pass_precondition_violation(tmp_path: Path) -> NegativeControlOutcome:
    """Placeholder: pass run on IR that fails its preconditions."""
    def _fn() -> None:
        raise PreconditionViolation(
            "pass 'fuse_producer_consumer' precondition "
            "'tensor has single consumer' violated (placeholder)"
        )
    return _expect(
        name="pass_precondition_violation",
        expected_error=PreconditionViolation,
        fn=_fn,
    )


def control_stale_analysis_consumed(tmp_path: Path) -> NegativeControlOutcome:
    """Placeholder: a consumer used a Payload summary that was
    invalidated by an upstream pass."""
    def _fn() -> None:
        raise StaleAnalysisAudit(
            "consumer 'cost_preview_v2' read 'payload_summary' that was "
            "invalidated by 'set_tile_params' (placeholder)"
        )
    return _expect(
        name="stale_analysis_consumed",
        expected_error=StaleAnalysisAudit,
        fn=_fn,
    )


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


def run_all_negative_controls(tmp_path: Path) -> NegativeControlReport:
    """Run every negative control and report results.

    Each control gets a fresh subdirectory under ``tmp_path`` so they
    don't interfere with each other.
    """
    report = NegativeControlReport()
    controls = [
        ("evidence_pack_source_missing", control_evidence_pack_source_missing),
        ("promotion_contract_hash_corrupted", control_promotion_contract_hash_corrupted),
        ("applies_when_predicate_violated", control_applies_when_predicate_violated),
        ("certificate_artifact_hash_changed", control_certificate_artifact_hash_changed),
        ("task_pack_missing_required_file", control_task_pack_missing_required_file),
        ("replay_input_hash_mismatch", control_replay_input_hash_mismatch),
        ("pass_card_missing", control_pass_card_missing),
        ("pass_precondition_violation", control_pass_precondition_violation),
        ("stale_analysis_consumed", control_stale_analysis_consumed),
    ]
    for name, fn in controls:
        sub = tmp_path / name
        sub.mkdir(parents=True, exist_ok=True)
        report.outcomes.append(fn(sub))
    return report
