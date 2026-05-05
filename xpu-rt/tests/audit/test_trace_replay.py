"""Tests for compgen.audit.trace_replay (M-31A.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.errors import ReplayHashMismatch, DecisionIdMismatch
from compgen.audit.trace_replay import (
    DecisionTrace,
    TRACE_SCHEMA_VERSION,
    assert_decision_ids_match,
    build_trace,
    compute_decision_id,
    compute_input_hashes,
    compute_output_hashes,
    load_trace,
    replay,
    write_trace,
)


def _build_synthetic_run(tmp_path: Path) -> Path:
    """Lay out the minimum 03_recipe_planning structure for a trace test."""
    run_dir = tmp_path / "synthetic_run"
    rp = run_dir / "03_recipe_planning"
    rp.mkdir(parents=True)
    (rp / "agent_decision_request.json").write_text(
        json.dumps({"schema_version": "agent_decision_request_v1",
                    "candidate_ids_allowed": ["cand_a", "cand_b"]},
                   sort_keys=True)
    )
    (rp / "llm_graph_view.json").write_text(
        json.dumps({"regions": [{"region_id": "matmul_0"}]}, sort_keys=True)
    )
    (rp / "candidate_actions.json").write_text(
        json.dumps({"candidates": [
            {"candidate_id": "cand_a", "legality": {"ok": True}},
            {"candidate_id": "cand_b", "legality": {"ok": True}},
        ]}, sort_keys=True)
    )
    (rp / "agent_decision_response.json").write_text(
        json.dumps({"schema_version": "agent_decision_response_v1",
                    "selected_candidate_id": "cand_a",
                    "rationale": {"why": "shortest predicted runtime"}},
                   sort_keys=True)
    )
    (rp / "agent_decision_record.json").write_text(
        json.dumps({"selected": "cand_a", "committed": True}, sort_keys=True)
    )
    return run_dir


def test_decision_id_is_deterministic() -> None:
    a = compute_decision_id(
        run_id="r1", region_id="m0", decision_index=0, request_hash="abc",
    )
    b = compute_decision_id(
        run_id="r1", region_id="m0", decision_index=0, request_hash="abc",
    )
    assert a == b
    assert len(a) == 16


def test_decision_id_changes_when_inputs_change() -> None:
    a = compute_decision_id(
        run_id="r1", region_id="m0", decision_index=0, request_hash="abc",
    )
    b = compute_decision_id(
        run_id="r1", region_id="m0", decision_index=0, request_hash="DIFFERENT",
    )
    assert a != b


def test_compute_input_hashes(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    hashes = compute_input_hashes(run_dir, promotion_library=tmp_path / "missing")
    assert "agent_decision_request" in hashes
    assert hashes["agent_decision_request"] != ""
    assert "promotion_library_state" in hashes  # empty dir → hash ""
    # Same inputs → same hash
    again = compute_input_hashes(run_dir, promotion_library=tmp_path / "missing")
    assert hashes == again


def test_build_and_round_trip_trace(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    trace = build_trace(
        run_dir,
        run_id="run_x",
        region_id="matmul_0",
        decision_index=0,
        commit="deadbeef",
    )
    assert trace.schema_version == TRACE_SCHEMA_VERSION
    assert len(trace.decision_id) == 16
    assert trace.run_id == "run_x"
    assert trace.chosen_action.get("candidate_id") == "cand_a"

    out = write_trace(trace, run_dir=run_dir)
    assert out.exists()
    reloaded = load_trace(out)
    assert reloaded.to_dict() == trace.to_dict()


def test_replay_matches_when_run_dir_unchanged(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    promo_lib = tmp_path / "missing"
    trace = build_trace(run_dir, run_id="run_x", commit="abc",
                        promotion_library=promo_lib)
    out = write_trace(trace, run_dir=run_dir)
    report = replay(trace_path=out, run_dir=run_dir, promotion_library=promo_lib)
    assert report.all_match
    assert report.decision_id_match
    assert report.input_hashes_match
    assert report.output_hashes_match


def test_replay_raises_on_input_mismatch(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    promo_lib = tmp_path / "missing"
    trace = build_trace(run_dir, run_id="run_x", commit="abc",
                        promotion_library=promo_lib)
    out = write_trace(trace, run_dir=run_dir)
    # Corrupt one input artifact
    (run_dir / "03_recipe_planning" / "agent_decision_request.json").write_text(
        '{"schema_version": "agent_decision_request_v1", "tampered": true}'
    )
    with pytest.raises(ReplayHashMismatch, match="input"):
        replay(trace_path=out, run_dir=run_dir, promotion_library=promo_lib)


def test_replay_raises_on_output_mismatch(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    promo_lib = tmp_path / "missing"
    trace = build_trace(run_dir, run_id="run_x", commit="abc",
                        promotion_library=promo_lib)
    out = write_trace(trace, run_dir=run_dir)
    (run_dir / "03_recipe_planning" / "agent_decision_response.json").write_text(
        '{"selected_candidate_id": "tampered"}'
    )
    with pytest.raises(ReplayHashMismatch, match="output"):
        replay(trace_path=out, run_dir=run_dir, promotion_library=promo_lib)


def test_replay_lenient_returns_report(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run(tmp_path)
    promo_lib = tmp_path / "missing"
    trace = build_trace(run_dir, run_id="run_x", commit="abc",
                        promotion_library=promo_lib)
    out = write_trace(trace, run_dir=run_dir)
    (run_dir / "03_recipe_planning" / "agent_decision_request.json").write_text(
        '{"tampered": true}'
    )
    report = replay(
        trace_path=out, run_dir=run_dir,
        promotion_library=promo_lib, strict=False,
    )
    assert report.all_match is False
    assert "agent_decision_request" in report.input_deltas


def test_assert_decision_ids_match_no_op_when_request_lacks_id() -> None:
    # Pre-M-31A requests have no decision_id; the assert is a no-op
    assert_decision_ids_match(
        request={"schema_version": "old"},
        response={"selected_candidate_id": "cand_a"},
    )


def test_assert_decision_ids_match_passes_on_echo() -> None:
    assert_decision_ids_match(
        request={"decision_id": "abc123def456"},
        response={"decision_id": "abc123def456",
                  "selected_candidate_id": "cand_a"},
    )


def test_assert_decision_ids_match_raises_on_mismatch() -> None:
    with pytest.raises(DecisionIdMismatch, match="decision_id"):
        assert_decision_ids_match(
            request={"decision_id": "abc123def456"},
            response={"decision_id": "wrong_value",
                      "selected_candidate_id": "cand_a"},
        )


def test_trace_round_trip_preserves_all_fields() -> None:
    trace = DecisionTrace(
        decision_id="abcdef1234567890",
        commit="deadbeef" * 5,
        run_id="run_y",
        region_id="conv_0",
        decision_index=3,
        input_hashes={"agent_decision_request": "h1", "llm_graph_view": "h2"},
        chosen_action={"kind": "select_candidate", "candidate_id": "cand_z"},
        rationale_paths=["llm_graph_view.regions[0].legal_candidates[1]"],
        output_hashes={"agent_decision_response": "h3"},
    )
    raw = trace.to_dict()
    reloaded = DecisionTrace.from_dict(raw)
    assert reloaded == trace
