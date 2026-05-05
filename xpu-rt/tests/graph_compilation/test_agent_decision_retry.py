"""Tests for M-15A Agent Decision Rejection / Retry Loop.

Verifies the bounded retry protocol around M-14A validation failures:

- Failed first attempt emits retry_request.json with typed reason.
- Corrected second attempt commits recipe.
- Exhausted retries leave recipe.mlir unwritten.
- Retry artifacts under attempts/attempt_<N>/ are preserved.
- retry_summary.json records the full attempt history.
- MCP commit tool returns retry_required vs committed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WIDE = REPO_ROOT / "results" / "graph_compilation" / "m14a_wide_llm_stub_suite"


def _need_wide() -> None:
    if not WIDE.is_dir():
        pytest.skip(f"wide fixture suite missing: {WIDE}")


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write(p: Path, body: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Response builders for negative / positive cases
# --------------------------------------------------------------------------- #


def _good_response_for_merlin_mlp_wide() -> dict:
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e",
        "rationale": {
            "summary": "Selecting the M-12-verified SetTileParams tile on matmul_0.",
            "evidence": [
                {"field": "candidate.kind", "value": "set_tile_params",
                 "reason": "Structured tiling on matmul region."},
                {"field": "cost_preview_v2.features.real_transform_verified",
                 "value": True,
                 "reason": "Carries M-12 differential evidence."},
                {"field": "cost_preview_v2.confidence", "value": 0.75,
                 "reason": "Verification boost above 0.55 baseline."},
            ],
        },
    }


def _bad_response_nonexistent() -> dict:
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": "cand_does_not_exist_xyz",
        "rationale": {
            "summary": "test",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }


def _bad_response_correctness_claim() -> dict:
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e",
        "rationale": {
            "summary": "this transform is verified correct end-to-end",
            "evidence": [
                {"field": "candidate.kind", "value": "set_tile_params", "reason": "x"},
                {"field": "candidate.label", "value": "tile", "reason": "y"},
            ],
        },
    }


def _bad_response_perf_claim() -> dict:
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e",
        "rationale": {
            "summary": "we benchmarked this and measured fastest",
            "evidence": [
                {"field": "candidate.kind", "value": "set_tile_params", "reason": "x"},
                {"field": "candidate.label", "value": "tile", "reason": "y"},
            ],
        },
    }


def _bad_response_missing_evidence() -> dict:
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e",
        "rationale": {"summary": "test", "evidence": []},
    }


def _bad_response_illegal() -> dict:
    """Pick an illegal candidate from the suite. We have to discover its
    id from candidate_actions at test time."""
    src = WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "candidate_actions.json"
    cas = _read(src)
    illegal = next(
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    )
    return {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": illegal,
        "rationale": {
            "summary": "illegal pick test",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def fresh_run(tmp_path: Path) -> Path:
    """Yields a freshly-constructed run dir (via subprocess) that has
    been driven through agent-decision-request with greedy mode. The
    M-14A request artifact is present; agent_decision/ is otherwise
    populated by the greedy run, so we wipe it before retry tests."""
    _need_wide()
    out = tmp_path / "m15a_run"
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "greedy",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return out


def _invoke_with_responses(
    *, run_dir: Path, response_paths: list[Path], max_retries: int = 3,
) -> subprocess.CompletedProcess:
    """Re-invoke the pipeline with --selection-mode agent-file +
    repeatable --agent-decision-response. Re-creates run_dir from
    scratch (the wipe is part of run.py)."""
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(run_dir),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "agent-file",
        "--agent-max-retries", str(max_retries),
    ]
    for p in response_paths:
        cmd += ["--agent-decision-response", str(p)]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# Single-attempt failure → retry_request emitted, no recipe
# --------------------------------------------------------------------------- #


def test_bad_first_response_emits_retry_request(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    _write(bad, _bad_response_nonexistent())
    res = _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=3,
    )
    assert res.returncode != 0  # retries exhausted at 1 bad attempt
    rr_path = (
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    assert rr_path.exists()
    rr = _read(rr_path)
    assert rr["schema_version"] == "agent_decision_retry_request_v1"
    assert rr["status"] == "retry_required"
    failed_names = {c["name"] for c in rr["validation"]["failed_checks"]}
    assert "selected_candidate_exists" in failed_names
    assert len(rr["candidate_ids_allowed"]) >= 1
    assert not (fresh_run / "03_recipe_planning" / "recipe.mlir").exists()


def test_retry_request_excludes_illegal_from_allowed_set(
    fresh_run: Path, tmp_path: Path,
) -> None:
    """The retry_request's candidate_ids_allowed should be the same
    bounded set the agent saw originally — no illegal IDs."""
    bad = tmp_path / "bad.json"
    _write(bad, _bad_response_nonexistent())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=1,
    )
    rr = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    cas = _read(
        fresh_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    }
    illegal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    }
    allowed = set(rr["candidate_ids_allowed"])
    # All allowed are legal.
    assert allowed.issubset(legal_ids)
    # No illegal sneaks in.
    assert not (allowed & illegal_ids)


# --------------------------------------------------------------------------- #
# Bad → good retry sequence (the headline test)
# --------------------------------------------------------------------------- #


def test_bad_then_good_retry_commits(fresh_run: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    good = tmp_path / "good.json"
    _write(bad, _bad_response_nonexistent())
    _write(good, _good_response_for_merlin_mlp_wide())
    res = _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad, good], max_retries=3,
    )
    assert res.returncode == 0, res.stderr

    # retry_summary records both attempts in order.
    summary = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "retry_summary.json"
    )
    assert summary["status"] == "pass"
    assert summary["recipe_committed"] is True
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][0]["status"] == "fail"
    assert summary["attempts"][1]["status"] == "pass"
    assert (
        summary["attempts"][1]["selected_candidate_id"]
        == "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e"
    )

    # attempts/attempt_000 + attempt_001 both exist.
    ad = fresh_run / "03_recipe_planning" / "agent_decision"
    assert (ad / "attempts" / "attempt_000" / "agent_decision_response.json").exists()
    assert (ad / "attempts" / "attempt_000" / "agent_decision_validation.json").exists()
    assert (ad / "attempts" / "attempt_000" / "retry_request.json").exists()
    assert (ad / "attempts" / "attempt_001" / "agent_decision_response.json").exists()
    assert (ad / "attempts" / "attempt_001" / "agent_decision_validation.json").exists()
    # No retry_request under attempt_001 (it passed).
    assert not (ad / "attempts" / "attempt_001" / "retry_request.json").exists()

    # Top-level retry_request.json removed after final pass.
    assert not (ad / "retry_request.json").exists()

    # Final accepted response copied to top-level.
    final_validation = _read(ad / "agent_decision_validation.json")
    assert final_validation["overall"] == "pass"
    assert (
        final_validation["selected_candidate_id"]
        == "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e"
    )

    # recipe.mlir references the corrected candidate.
    recipe = (fresh_run / "03_recipe_planning" / "recipe.mlir").read_text(
        encoding="utf-8",
    )
    assert "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e" in recipe


# --------------------------------------------------------------------------- #
# Exhausted retries
# --------------------------------------------------------------------------- #


def test_exhausted_retries_leave_no_recipe(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad1 = tmp_path / "bad1.json"
    bad2 = tmp_path / "bad2.json"
    _write(bad1, _bad_response_nonexistent())
    _write(bad2, _bad_response_correctness_claim())
    res = _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad1, bad2], max_retries=2,
    )
    assert res.returncode != 0
    summary = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "retry_summary.json"
    )
    assert summary["status"] == "failed_exhausted_retries"
    assert summary["recipe_committed"] is False
    assert summary["final_selected_candidate_id"] is None
    assert len(summary["attempts"]) == 2
    assert all(a["status"] == "fail" for a in summary["attempts"])
    assert not (fresh_run / "03_recipe_planning" / "recipe.mlir").exists()


# --------------------------------------------------------------------------- #
# Per-failure-mode coverage
# --------------------------------------------------------------------------- #


def test_illegal_candidate_attempt_emits_retry_request(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "illegal.json"
    _write(bad, _bad_response_illegal())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=1,
    )
    rr = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    failed = {c["name"] for c in rr["validation"]["failed_checks"]}
    assert "selected_candidate_is_legal" in failed


def test_correctness_claim_attempt_emits_retry_request(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "correctness.json"
    _write(bad, _bad_response_correctness_claim())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=1,
    )
    rr = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    failed = {c["name"] for c in rr["validation"]["failed_checks"]}
    assert "no_correctness_claim" in failed


def test_perf_claim_attempt_emits_retry_request(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "perf.json"
    _write(bad, _bad_response_perf_claim())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=1,
    )
    rr = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    failed = {c["name"] for c in rr["validation"]["failed_checks"]}
    assert "no_measured_performance_claim" in failed


def test_missing_evidence_attempt_emits_retry_request(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "noevidence.json"
    _write(bad, _bad_response_missing_evidence())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad], max_retries=1,
    )
    rr = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "attempts" / "attempt_000" / "retry_request.json"
    )
    failed = {c["name"] for c in rr["validation"]["failed_checks"]}
    assert "rationale_evidence_present" in failed


# --------------------------------------------------------------------------- #
# retry_summary determinism
# --------------------------------------------------------------------------- #


def test_retry_summary_records_attempts_in_order(
    fresh_run: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    good = tmp_path / "good.json"
    _write(bad, _bad_response_nonexistent())
    _write(good, _good_response_for_merlin_mlp_wide())
    _invoke_with_responses(
        run_dir=fresh_run, response_paths=[bad, good], max_retries=3,
    )
    summary = _read(
        fresh_run / "03_recipe_planning" / "agent_decision"
        / "retry_summary.json"
    )
    indices = [a["attempt_index"] for a in summary["attempts"]]
    assert indices == [0, 1]


# --------------------------------------------------------------------------- #
# MCP tool returns retry_required vs committed
# --------------------------------------------------------------------------- #


def test_mcp_commit_tool_returns_retry_required_on_validation_fail(
    fresh_run: Path, tmp_path: Path,
) -> None:
    """The MCP commit tool's response should distinguish retry_required
    from committed for downstream programmatic consumers."""
    from compgen.mcp.tools.agent_decision import (
        compgen_commit_agent_decision_response,
    )

    class _S: pass

    bad = _bad_response_nonexistent()
    r = compgen_commit_agent_decision_response(
        _S(),
        model_config="configs/models/merlin_mlp_wide.yaml",
        target_config="configs/targets/host_cpu.yaml",
        out_dir=str(fresh_run),
        response=bad,
        stop_after="agent-decision-request",
        timeout_sec=300,
    )
    assert r["validation_overall"] == "fail"
    # The pipeline raises (returncode != 0).
    assert not r["ok"]
    assert r.get("validation_failed_checks")
    assert any(
        c["name"] == "selected_candidate_exists"
        for c in r["validation_failed_checks"]
    )


def test_mcp_commit_tool_returns_committed_on_pass(
    fresh_run: Path, tmp_path: Path,
) -> None:
    from compgen.mcp.tools.agent_decision import (
        compgen_commit_agent_decision_response,
    )

    class _S: pass

    r = compgen_commit_agent_decision_response(
        _S(),
        model_config="configs/models/merlin_mlp_wide.yaml",
        target_config="configs/targets/host_cpu.yaml",
        out_dir=str(fresh_run),
        response=_good_response_for_merlin_mlp_wide(),
        stop_after="agent-decision-request",
        timeout_sec=300,
    )
    assert r["ok"] is True
    assert r["validation_overall"] == "pass"
    assert (
        r["selected_candidate_id"]
        == "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e"
    )


# --------------------------------------------------------------------------- #
# Greedy still passes after M-15A
# --------------------------------------------------------------------------- #


def test_greedy_mode_unaffected_by_m15a(tmp_path: Path) -> None:
    out = tmp_path / "greedy_run"
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "greedy",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    # Greedy doesn't emit retry artifacts.
    ad = out / "03_recipe_planning" / "agent_decision"
    assert not (ad / "retry_summary.json").exists()
    assert not (ad / "attempts").exists()
    assert (out / "03_recipe_planning" / "recipe.mlir").exists()
