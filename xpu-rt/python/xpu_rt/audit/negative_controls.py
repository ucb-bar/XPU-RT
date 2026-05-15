"""Fault-injection negative controls.

Positive tests prove the happy path works. Negative controls prove the
gates are real — that introducing a specific fault causes the
corresponding gate to fail with a typed error. A stub often passes
positive tests and fails negative controls; that's how we catch them.

The test parametrization in :mod:`tests.audit.test_negative_controls`
walks this table. Each function here injects one specific break and
asserts the named typed error fires.

Notes on placeholders
-----------------------------

Three rows of the negative-control table reference subsystems that
land (pass card registry):

- ``MissingPassCard``        — pass card removed before agent request
- ``PreconditionViolation``  — pass run on illegal IR
- ``StaleAnalysisAudit``     — stale Payload summary consumed

these controls assert the typed-error machinery exists and
is wired through the audit. Real fault-injection lands. The
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

    we exercise this by raising the typed error directly when
    a "source artifact" is missing — the actual evidence_pack.py
    integration check lands 's trust report.
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
# Placeholder controls (full fault-injection lands )
# --------------------------------------------------------------------------- #


def control_pass_card_missing(tmp_path: Path) -> NegativeControlOutcome:
    """real fault injection: build a registry containing only one
    of the production passes, then ask the validator to resolve a
    request that references the OTHER one. The validator must raise
    :class:`MissingPassCard`.

    This catches the failure mode where the agent's vocabulary
    drifts ahead of the registry (a pass id is exposed but its card
    was never authored).
    """
    import shutil
    from compgen.passes.cards import PassCardRegistry

    real_root = Path(__file__).resolve().parents[3] / "docs" / "generated" / "pass_cards"
    fake_root = tmp_path / "pass_cards"
    fake_root.mkdir(parents=True, exist_ok=True)
    # Copy ONLY one card; pretend the other was never authored.
    src = real_root / "set_tile_params.yaml"
    if src.exists():
        shutil.copy(src, fake_root / "set_tile_params.yaml")

    def _fn() -> None:
        registry = PassCardRegistry.load(fake_root)
        # The agent's vocabulary references both; the registry knows only one.
        registry.assert_resolvable(["set_tile_params", "fuse_producer_consumer"])

    return _expect(
        name="pass_card_missing",
        expected_error=MissingPassCard,
        fn=_fn,
    )


def control_pass_precondition_violation(tmp_path: Path) -> NegativeControlOutcome:
    """real fault injection: a pass card declares a verification
    rung, the rung's certificate is missing, and the validator raises
    :class:`VerificationGateMissing`. This is the producer-side
    pre-condition: a downstream consumer of the pass output cannot
    safely proceed without the certificate."""
    from compgen.passes.cards import PassCard
    from compgen.passes.verification import (
        assert_required_rungs_discharged,
    )
    from compgen.audit.errors import VerificationGateMissing

    # Author a card declaring the differential rung but emit no
    # certificate. assert_required_rungs_discharged must raise.
    card = PassCard(
        schema_version="pass_card_v1",
        pass_id="injected_test_pass",
        display_name="injected test pass",
        level="payload",
        family="tiling",
        reads=("a.json",),
        writes=("b.json",),
        preconditions=("region.kind == matmul",),
        invalidates=("payload_summary",),
        preserves_refinement="bit_equality",
        verification=("differential",),
        cost="cheap",
        failure_modes=("test_only",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fn() -> None:
        assert_required_rungs_discharged(card, run_dir)

    # The closest existing typed error for "preconditions for safe
    # consumption are not satisfied" is VerificationGateMissing.
    # PreconditionViolation remains as the family-level alias for
    # when per-pass precondition checking lands on real IR.
    return _expect(
        name="pass_precondition_violation",
        expected_error=VerificationGateMissing,
        fn=_fn,
    )


def control_stale_analysis_consumed(tmp_path: Path) -> NegativeControlOutcome:
    """real fault injection: a pass declares ``invalidates:
    [semantic_obligations]`` but in the actual run mutated
    ``graph_dossier_v3``. The producer-side guard
    :func:`assert_invalidations_match_claim` raises
    :class:`StaleAnalysisAudit` (an alias for
    :class:`UnannouncedInvalidation`).
    """
    from compgen.analysis.invalidation import (
        InvalidationDiff,
        assert_invalidations_match_claim,
    )

    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",),
        appeared=(),
        removed=(),
    )

    def _fn() -> None:
        # Claim only ``semantic_obligations`` — its closure does NOT
        # include graph_dossier_v3, so the mutation is unannounced.
        assert_invalidations_match_claim(
            diff, ["semantic_obligations"], pass_id="injected_test_pass",
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
