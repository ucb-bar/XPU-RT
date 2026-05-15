"""Tests for compgen.analysis.invalidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.analysis.checkpoints import AnalysisIndex, AnalysisSummary
from compgen.analysis.invalidation import (
    InvalidationDiff,
    append_invalidation_log,
    assert_invalidations_match_claim,
    compute_invalidation_diff,
    make_log_entry,
    write_invalidation_log,
)
from compgen.audit.errors import (
    StaleAnalysisAudit,
    UnannouncedInvalidation,
)


def _make_synthetic_run(tmp_path: Path) -> Path:
    """Mirror tests/analysis/test_checkpoints.py::_make_synthetic_run."""
    run_dir = tmp_path / "run"
    (run_dir / "00_graph_capture").mkdir(parents=True)
    (run_dir / "00_graph_capture" / "capture_report.json").write_text('{"ok": true}')
    (run_dir / "00_graph_capture" / "dynamo_summary.json").write_text('{"breaks": 0}')
    (run_dir / "01_payload_lowering").mkdir(parents=True)
    (run_dir / "01_payload_lowering" / "lowering_summary.json").write_text('{"ops": 4}')
    (run_dir / "02_graph_analysis").mkdir(parents=True)
    (run_dir / "02_graph_analysis" / "graph_dossier_v3.json").write_text('{"regions": []}')
    (run_dir / "02_graph_analysis" / "candidate_actions.json").write_text('{"candidates": []}')
    (run_dir / "02_graph_analysis" / "cost_preview_v2.json").write_text('{"per_candidate": []}')
    return run_dir


# --------------------------------------------------------------------------- #
# Diff primitive
# --------------------------------------------------------------------------- #


def test_diff_empty_when_no_change(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    a = AnalysisIndex.from_run_dir(run_dir)
    b = AnalysisIndex.from_run_dir(run_dir)
    diff = compute_invalidation_diff(a, b)
    assert diff.is_empty
    assert diff.mutated == ()
    assert diff.appeared == ()
    assert diff.removed == ()


def test_diff_detects_mutation(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    before = AnalysisIndex.from_run_dir(run_dir)
    # Mutate one summary's content
    (run_dir / "02_graph_analysis" / "graph_dossier_v3.json").write_text(
        '{"regions": ["m0"]}'
    )
    after = AnalysisIndex.from_run_dir(run_dir)
    diff = compute_invalidation_diff(before, after)
    assert "graph_dossier_v3" in diff.mutated
    assert diff.appeared == ()
    assert diff.removed == ()


def test_diff_detects_appearance(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    before = AnalysisIndex.from_run_dir(run_dir)
    # Emit a new summary
    (run_dir / "03_recipe_planning").mkdir(parents=True)
    (run_dir / "03_recipe_planning" / "recipe_summary.json").write_text('{"ops": 1}')
    after = AnalysisIndex.from_run_dir(run_dir)
    diff = compute_invalidation_diff(before, after)
    assert "recipe_summary" in diff.appeared
    assert diff.mutated == ()


def test_diff_detects_removal(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    before = AnalysisIndex.from_run_dir(run_dir)
    (run_dir / "02_graph_analysis" / "graph_dossier_v3.json").unlink()
    after = AnalysisIndex.from_run_dir(run_dir)
    diff = compute_invalidation_diff(before, after)
    assert "graph_dossier_v3" in diff.removed


# --------------------------------------------------------------------------- #
# Claim-vs-actual enforcement
# --------------------------------------------------------------------------- #


def test_claim_matches_when_no_mutation() -> None:
    diff = InvalidationDiff(mutated=(), appeared=(), removed=())
    # No claim, no mutation — passes.
    assert_invalidations_match_claim(diff, [], pass_id="set_tile_params")
    # Empty diff with non-empty claim is also fine (overclaim is okay).
    assert_invalidations_match_claim(diff, ["graph_dossier_v3"], pass_id="x")


def test_claim_matches_direct_mutation() -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    assert_invalidations_match_claim(diff, ["graph_dossier_v3"], pass_id="x")


def test_claim_via_transitive_closure() -> None:
    """Mutating ``cost_preview`` must be accepted when claim was
    ``[graph_dossier_v3]`` because cost_preview transitively depends
    on graph_dossier_v3."""
    diff = InvalidationDiff(
        mutated=("cost_preview",), appeared=(), removed=()
    )
    assert_invalidations_match_claim(diff, ["graph_dossier_v3"], pass_id="x")


def test_unannounced_invalidation_raises() -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    # Claim only the unrelated payload_summary — the closure of payload_summary
    # transitively includes graph_dossier_v3, so it would actually pass.
    # We need a true unrelated claim; use semantic_obligations which is
    # downstream of recipe_summary.
    with pytest.raises(UnannouncedInvalidation, match="graph_dossier_v3"):
        assert_invalidations_match_claim(
            diff, ["semantic_obligations"], pass_id="lying_pass",
        )


def test_stale_analysis_audit_is_alias() -> None:
    """keeps StaleAnalysisAudit as a subclass alias of
    UnannouncedInvalidation so negative-control references
    still resolve."""
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    with pytest.raises(StaleAnalysisAudit):
        assert_invalidations_match_claim(
            diff, ["semantic_obligations"], pass_id="x",
        )


def test_removal_treated_as_invalidation() -> None:
    """A summary that disappears between snapshots requires a claim."""
    diff = InvalidationDiff(
        mutated=(), appeared=(), removed=("graph_dossier_v3",)
    )
    with pytest.raises(UnannouncedInvalidation):
        assert_invalidations_match_claim(diff, [], pass_id="vanishing_pass")
    # Claimed → fine
    assert_invalidations_match_claim(
        diff, ["graph_dossier_v3"], pass_id="vanishing_pass",
    )


def test_appearance_does_not_require_claim() -> None:
    diff = InvalidationDiff(
        mutated=(), appeared=("recipe_summary",), removed=()
    )
    # No claim needed for new artifacts
    assert_invalidations_match_claim(diff, [], pass_id="emit_pass")


def test_unknown_claim_id_raises() -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    with pytest.raises(Exception):  # AnalysisSummaryError from assert_resolvable
        assert_invalidations_match_claim(
            diff, ["totally_made_up_id"], pass_id="x",
        )


# --------------------------------------------------------------------------- #
# Log entry + persistence
# --------------------------------------------------------------------------- #


def test_make_log_entry_records_match() -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    entry = make_log_entry(
        pass_id="set_tile_params",
        region_id="matmul_0",
        candidate_id="tile_M16_N16_K16",
        claimed=["graph_dossier_v3"],
        diff=diff,
    )
    assert entry.matches_claim
    assert entry.pass_id == "set_tile_params"
    assert "graph_dossier_v3" in entry.closure


def test_make_log_entry_records_mismatch() -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    entry = make_log_entry(
        pass_id="lying_pass",
        claimed=["semantic_obligations"],
        diff=diff,
    )
    assert entry.matches_claim is False


def test_write_invalidation_log(tmp_path: Path) -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    entry = make_log_entry(
        pass_id="set_tile_params",
        claimed=["graph_dossier_v3"],
        diff=diff,
    )
    out = write_invalidation_log(tmp_path, [entry])
    assert out.exists()
    raw = json.loads(out.read_text())
    assert raw["schema_version"] == "invalidation_log_v1"
    assert raw["entry_count"] == 1
    assert raw["entries"][0]["pass_id"] == "set_tile_params"


def test_append_invalidation_log(tmp_path: Path) -> None:
    diff = InvalidationDiff(
        mutated=("graph_dossier_v3",), appeared=(), removed=()
    )
    e1 = make_log_entry(
        pass_id="set_tile_params",
        claimed=["graph_dossier_v3"],
        diff=diff,
    )
    e2 = make_log_entry(
        pass_id="fuse_producer_consumer",
        claimed=["graph_dossier_v3"],
        diff=diff,
    )
    append_invalidation_log(tmp_path, e1)
    out = append_invalidation_log(tmp_path, e2)
    raw = json.loads(out.read_text())
    assert raw["entry_count"] == 2
    pass_ids = [e["pass_id"] for e in raw["entries"]]
    assert pass_ids == ["set_tile_params", "fuse_producer_consumer"]


# --------------------------------------------------------------------------- #
# AnalysisSummary.generation
# --------------------------------------------------------------------------- #


def test_analysis_summary_default_generation_is_zero() -> None:
    s = AnalysisSummary(
        id="x", level="payload", relative_path="x.json",
        content_hash="h", dependencies=(), available=True,
        last_modified_utc="2026-05-05T00:00:00Z",
        description="x", optional=False,
    )
    assert s.generation == 0
    assert s.to_dict()["generation"] == 0


def test_analysis_summary_explicit_generation() -> None:
    s = AnalysisSummary(
        id="x", level="payload", relative_path="x.json",
        content_hash="h", dependencies=(), available=True,
        last_modified_utc="2026-05-05T00:00:00Z",
        description="x", optional=False, generation=3,
    )
    assert s.generation == 3
    assert s.to_dict()["generation"] == 3


# --------------------------------------------------------------------------- #
# consumer-side stale-read detection
# --------------------------------------------------------------------------- #


def test_summary_read_round_trip() -> None:
    from compgen.analysis.invalidation import SummaryRead

    r = SummaryRead(
        summary_id="graph_dossier_v3",
        consumer_id="cost_preview_v2",
        generation_observed=2,
        timestamp_utc="2026-05-05T00:00:00Z",
    )
    assert SummaryRead.from_dict(r.to_dict()) == r


def test_append_read_log_writes(tmp_path: Path) -> None:
    from compgen.analysis.invalidation import (
        SummaryRead,
        append_read_log,
        load_read_log,
    )

    r1 = SummaryRead(
        summary_id="graph_dossier_v3",
        consumer_id="cost_preview_v2",
        generation_observed=0,
    )
    r2 = SummaryRead(
        summary_id="graph_dossier_v3",
        consumer_id="kernel_readiness",
        generation_observed=1,
    )
    append_read_log(tmp_path, r1)
    append_read_log(tmp_path, r2)
    reads = load_read_log(tmp_path)
    assert len(reads) == 2
    assert reads[0].consumer_id == "cost_preview_v2"
    assert reads[1].consumer_id == "kernel_readiness"


def test_assert_no_stale_reads_clean(tmp_path: Path) -> None:
    """All consumers observing the same generation: clean."""
    from compgen.analysis.invalidation import (
        SummaryRead,
        append_read_log,
        assert_no_stale_reads,
    )

    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c1", generation_observed=2,
    ))
    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c2", generation_observed=2,
    ))
    assert_no_stale_reads(tmp_path)  # no raise


def test_assert_no_stale_reads_strictly_increasing(tmp_path: Path) -> None:
    """Consumers reading later see higher generations: clean."""
    from compgen.analysis.invalidation import (
        SummaryRead,
        append_read_log,
        assert_no_stale_reads,
    )

    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c1", generation_observed=0,
    ))
    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c2", generation_observed=1,
    ))
    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c3", generation_observed=2,
    ))
    assert_no_stale_reads(tmp_path)


def test_assert_no_stale_reads_raises_on_regression(tmp_path: Path) -> None:
    """A later reader observing an OLDER generation than an earlier one
    is the failure mode: someone bumped the summary, then a later
    consumer used a stale value."""
    from compgen.analysis.invalidation import (
        SummaryRead,
        append_read_log,
        assert_no_stale_reads,
    )
    from compgen.audit.errors import StaleAnalysisAudit

    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c1", generation_observed=2,
    ))
    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3",
        consumer_id="lying_consumer",
        generation_observed=1,  # older!
    ))
    with pytest.raises(StaleAnalysisAudit, match="lying_consumer"):
        assert_no_stale_reads(tmp_path)


def test_assert_no_stale_reads_no_log_is_clean(tmp_path: Path) -> None:
    from compgen.analysis.invalidation import assert_no_stale_reads

    assert_no_stale_reads(tmp_path)  # no log → no-op


def test_assert_no_stale_reads_per_summary_isolated(tmp_path: Path) -> None:
    """Stale read on one summary must not contaminate another summary's check."""
    from compgen.analysis.invalidation import (
        SummaryRead,
        append_read_log,
        assert_no_stale_reads,
    )

    append_read_log(tmp_path, SummaryRead(
        summary_id="graph_dossier_v3", consumer_id="c1", generation_observed=5,
    ))
    # c2 reads cost_preview at gen=0 — that's fine for cost_preview
    append_read_log(tmp_path, SummaryRead(
        summary_id="cost_preview", consumer_id="c2", generation_observed=0,
    ))
    assert_no_stale_reads(tmp_path)  # no raise — different summaries
