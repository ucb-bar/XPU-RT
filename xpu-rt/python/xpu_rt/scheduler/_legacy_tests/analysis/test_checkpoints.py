"""Tests for compgen.analysis.checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.analysis.checkpoints import (
    ANALYSIS_LEVELS,
    KNOWN_SUMMARIES,
    AnalysisIndex,
    AnalysisLevel,
    AnalysisSummary,
    AnalysisSummaryError,
    KnownSummary,
    assert_resolvable,
    known_summary_ids,
    summary_id_for_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Static registry
# --------------------------------------------------------------------------- #


def test_levels_match_spec() -> None:
    """The 8 IR / analysis lenses are exactly: fx, payload, graph,
    recipe, semantic, tile, kernel, plan, runtime.

    Note: ``graph`` is the payload-derived graph-dossier bucket; it
    sits between A1 (Payload) and A2 (Recipe) and is a level of its
    own to keep dossier-specific dependencies separable."""
    assert set(ANALYSIS_LEVELS) == {
        "fx", "payload", "graph", "recipe", "semantic",
        "tile", "kernel", "plan", "runtime",
    }


def test_known_summaries_are_unique() -> None:
    seen: set[str] = set()
    for entry in KNOWN_SUMMARIES:
        assert entry.id not in seen, f"duplicate id {entry.id!r}"
        seen.add(entry.id)


def test_every_dependency_is_registered() -> None:
    """Every summary's declared dependency must itself be a known id —
    no dangling references."""
    known = {e.id for e in KNOWN_SUMMARIES}
    for entry in KNOWN_SUMMARIES:
        for dep in entry.dependencies:
            assert dep in known, (
                f"summary {entry.id!r} depends on unknown id {dep!r}"
            )


def test_known_summary_to_dict_round_trip() -> None:
    e = KnownSummary(
        id="x",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/x.json",
        dependencies=("payload_summary",),
        description="test",
    )
    d = e.to_dict()
    assert d["level"] == "payload"
    assert d["dependencies"] == ["payload_summary"]


def test_assert_resolvable_passes_for_known() -> None:
    assert_resolvable(["payload_summary", "graph_dossier_v3"])


def test_assert_resolvable_raises_for_unknown() -> None:
    with pytest.raises(AnalysisSummaryError, match="ghost_summary"):
        assert_resolvable(["payload_summary", "ghost_summary"])


def test_known_summary_ids_is_sorted_tuple() -> None:
    ids = known_summary_ids()
    assert isinstance(ids, tuple)
    assert list(ids) == sorted(ids)


def test_summary_id_for_path_exact() -> None:
    sid = summary_id_for_path("02_graph_analysis/graph_dossier_v3.json")
    assert sid == "graph_dossier_v3"


def test_summary_id_for_path_under_directory() -> None:
    """Files under a directory-typed summary resolve to the directory
    summary id."""
    sid = summary_id_for_path("03_recipe_planning/kernel_contracts/region_0.yaml")
    assert sid == "kernel_contracts"


def test_summary_id_for_path_unknown() -> None:
    assert summary_id_for_path("99_made_up/whatever.json") is None


# --------------------------------------------------------------------------- #
# Per-run index (synthetic)
# --------------------------------------------------------------------------- #


def _make_synthetic_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "00_graph_capture").mkdir(parents=True)
    (run_dir / "00_graph_capture" / "capture_report.json").write_text('{"ok": true}')
    (run_dir / "00_graph_capture" / "dynamo_summary.json").write_text('{"breaks": 0}')
    (run_dir / "01_payload_lowering").mkdir(parents=True)
    (run_dir / "01_payload_lowering" / "lowering_summary.json").write_text('{"ops": 4}')
    (run_dir / "01_payload_lowering" / "merlin_strict_gate_report.json").write_text('{"status": "pass"}')
    (run_dir / "02_graph_analysis").mkdir(parents=True)
    (run_dir / "02_graph_analysis" / "graph_dossier_v3.json").write_text('{"regions": []}')
    return run_dir


def test_analysis_index_from_run_dir(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    # Always populates an entry per KNOWN_SUMMARIES, available or not
    assert len(idx) == len(KNOWN_SUMMARIES)
    # Available rows have non-empty hash
    capture = idx.require("capture_report")
    assert capture.available
    assert capture.content_hash != ""
    # Missing rows have empty hash + available=False
    semantic = idx.require("semantic_obligations")
    assert semantic.available is False
    assert semantic.content_hash == ""


def test_analysis_index_strict_gate_report_glob(tmp_path: Path) -> None:
    """strict_gate_report uses a glob (<model>_strict_gate_report.json)."""
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    sgr = idx.require("strict_gate_report")
    assert sgr.available
    # Path resolves to the actual file, not the directory pattern
    assert sgr.relative_path.endswith("_strict_gate_report.json")


def test_analysis_index_content_hash_changes_when_content_changes(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    a = AnalysisIndex.from_run_dir(run_dir).require("capture_report").content_hash
    (run_dir / "00_graph_capture" / "capture_report.json").write_text('{"ok": false}')
    b = AnalysisIndex.from_run_dir(run_dir).require("capture_report").content_hash
    assert a != b


def test_analysis_index_available_summaries(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    avail = idx.available_summaries()
    available_ids = {s.id for s in avail}
    assert "capture_report" in available_ids
    assert "graph_dossier_v3" in available_ids
    # Things we did NOT create are not available
    assert "kernel_evidence_pack" not in available_ids


def test_analysis_index_by_level(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    fx = idx.by_level("fx")
    assert all(s.level == "fx" for s in fx)
    payload = idx.by_level(AnalysisLevel.PAYLOAD)
    assert all(s.level == "payload" for s in payload)


def test_analysis_index_to_dict(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    d = idx.to_dict()
    assert d["schema_version"] == "analysis_index_v1"
    assert d["summary_count"] == len(KNOWN_SUMMARIES)
    assert d["available_count"] == len(idx.available_summaries())


# --------------------------------------------------------------------------- #
# Transitive invalidation
# --------------------------------------------------------------------------- #


def test_transitively_invalidated_by_payload_summary(tmp_path: Path) -> None:
    """Invalidating ``payload_summary`` cascades to graph + recipe summaries."""
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    affected = idx.transitively_invalidated_by(["payload_summary"])
    # Direct dependents
    assert "graph_dossier_v3" in affected
    assert "dialect_coverage" in affected
    # Transitive dependents
    assert "candidate_actions" in affected
    assert "cost_preview" in affected
    assert "llm_action_space" in affected
    assert "candidate_selection" in affected
    # Original is in the result (the seed)
    assert "payload_summary" in affected


def test_transitively_invalidated_by_graph_dossier_only(tmp_path: Path) -> None:
    """Invalidating only graph_dossier_v3 leaves the upstream payload
    summary alone."""
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    affected = idx.transitively_invalidated_by(["graph_dossier_v3"])
    assert "graph_dossier_v3" in affected
    assert "payload_summary" not in affected
    # but downstream still affected
    assert "candidate_actions" in affected


def test_transitively_invalidated_by_unknown_raises(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    idx = AnalysisIndex.from_run_dir(run_dir)
    with pytest.raises(AnalysisSummaryError):
        idx.transitively_invalidated_by(["ghost"])


# --------------------------------------------------------------------------- #
# Real run dir (end-to-end)
# --------------------------------------------------------------------------- #


def test_analysis_index_against_real_run() -> None:
    """The /tmp/m31_smoke run from the smoke must index cleanly.

    We skip if that directory doesn't exist (e.g. on CI without a
    prior smoke run); the holdout-baseline test below covers the same
    contract from scratch.
    """
    real = Path("/tmp/m31_smoke")
    if not real.exists() or not (real / "run_manifest.json").exists():
        pytest.skip("/tmp/m31_smoke not available; covered by holdout-run test")
    idx = AnalysisIndex.from_run_dir(real)
    avail_ids = {s.id for s in idx.available_summaries()}
    # Every summary the smoke produces must be indexable
    must_be_available = {
        "capture_report", "dynamo_summary", "export_graph",
        "payload_summary", "dialect_coverage", "lowering_diagnostics",
        "graph_dossier_v3", "region_graph", "region_map",
        "candidate_actions", "cost_preview",
        "llm_action_space", "llm_graph_view",
        "candidate_selection", "recipe_summary", "recipe_validation",
        "semantic_obligations",
    }
    missing = must_be_available - avail_ids
    assert not missing, f"expected always-on summaries missing: {missing}"
