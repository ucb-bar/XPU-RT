"""Tests for M-14A Agent Candidate Decision Loop.

Covers the agent-driven selection mode (agent-file) end-to-end + the
spec's required negative cases against a copy of a real run dir.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compgen.graph_compilation.agent_decision import (
    _evidence_field_resolves,
    _scan_forbidden,
    _FORBIDDEN_CORRECTNESS_PATTERNS,
    _FORBIDDEN_PERF_PATTERNS,
    build_agent_decision_request,
    run_agent_decision,
    validate_agent_decision_response,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WIDE = REPO_ROOT / "results" / "graph_compilation" / "m14a_wide_llm_stub_suite"


def _need_wide() -> None:
    if not WIDE.is_dir():
        pytest.skip(f"wide fixture suite missing: {WIDE}")


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


_CANONICAL = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
)


# --------------------------------------------------------------------------- #
# agent-file end-to-end
# --------------------------------------------------------------------------- #


def test_merlin_mlp_wide_agent_file_e2e(tmp_path: Path) -> None:
    """End-to-end: reuse a recorded agent_decision_response.json (real
    Claude-Code-style file, written by an external agent, not synthesized
    by a stub), run with --selection-mode agent-file, validate the
    recipe references the agent-selected candidate."""
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    response_path = (
        src / "03_recipe_planning" / "agent_decision"
        / "agent_decision_response.json"
    )
    if not response_path.exists():
        pytest.skip(f"agent_decision_response.json missing: {response_path}")

    out = tmp_path / "agent_file_run"
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "agent-file",
        "--agent-decision-response", str(response_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    val = _read(
        out / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    assert val["overall"] == "pass"
    assert val["selection_mode"] == "agent-file"
    recipe = (out / "03_recipe_planning" / "recipe.mlir").read_text(
        encoding="utf-8",
    )
    assert val["selected_candidate_id"] in recipe


# --------------------------------------------------------------------------- #
# greedy mode unchanged
# --------------------------------------------------------------------------- #


def test_greedy_mode_does_not_emit_agent_decision_artifacts() -> None:
    """Greedy is the default and must remain identical in behavior. It
    does NOT call run_agent_decision, so no agent_decision/ directory
    should exist."""
    GREEDY_SUITE = (
        REPO_ROOT / "results" / "graph_compilation" / "m14a_greedy_suite"
    )
    if not GREEDY_SUITE.is_dir():
        pytest.skip(f"greedy fixture missing: {GREEDY_SUITE}")
    for model in _CANONICAL:
        ad = (
            GREEDY_SUITE / model / "03_recipe_planning" / "agent_decision"
        )
        # The post-M-13 emission writes agent_decision_request.json for
        # any mode; that's allowed. But response/validation/trace must
        # NOT exist for greedy.
        assert not (ad / "agent_decision_response.json").exists(), (
            f"{model}: greedy mode emitted a response (should not)"
        )
        assert not (ad / "agent_decision_validation.json").exists(), (
            f"{model}: greedy mode emitted a validation report"
        )
        assert not (ad / "agent_decision_trace.json").exists(), (
            f"{model}: greedy mode emitted a trace"
        )


# --------------------------------------------------------------------------- #
# Forbidden-claim scanner unit tests
# --------------------------------------------------------------------------- #


def test_correctness_claims_detected() -> None:
    bad_strings = (
        "this is verified correct",
        "correctness guaranteed",
        "bit equivalent to eager",
        "Correctness verified.",
    )
    for s in bad_strings:
        assert _scan_forbidden(s, _FORBIDDEN_CORRECTNESS_PATTERNS), (
            f"failed to flag: {s!r}"
        )


def test_perf_claims_detected() -> None:
    bad_strings = (
        "measured fastest",
        "we benchmarked it",
        "profiled and confirmed",
        "executed faster than baseline",
    )
    for s in bad_strings:
        assert _scan_forbidden(s, _FORBIDDEN_PERF_PATTERNS), (
            f"failed to flag: {s!r}"
        )


def test_innocent_phrasing_passes() -> None:
    ok_strings = (
        "this candidate is legal",
        "the cost preview suggests this is cheap",
        "the obligation is bit_equality",
        "real_transform_verified=true (M-12 evidence)",
    )
    for s in ok_strings:
        assert not _scan_forbidden(s, _FORBIDDEN_CORRECTNESS_PATTERNS)
        assert not _scan_forbidden(s, _FORBIDDEN_PERF_PATTERNS)


def test_evidence_field_resolves_against_candidate() -> None:
    cand = {"kind": "set_tile_params", "cost_preview": {"static_relative_cost": 0.5}}
    request = {"sources": {}}
    assert _evidence_field_resolves(
        "candidate.kind", request=request, cand=cand,
    )
    assert _evidence_field_resolves(
        "candidate.cost_preview.static_relative_cost",
        request=request, cand=cand,
    )
    assert not _evidence_field_resolves(
        "candidate.fictional_field", request=request, cand=cand,
    )


# --------------------------------------------------------------------------- #
# Negative tests — copy a real run dir and mutate
# --------------------------------------------------------------------------- #


@pytest.fixture
def merlin_mlp_wide_run(tmp_path: Path) -> Path:
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    dst = tmp_path / "merlin_mlp_wide"
    shutil.copytree(src, dst)
    # Build a fresh request from the copy so SHAs match.
    build_agent_decision_request(dst)
    return dst


def _run_validation(run_dir: Path, response: dict) -> dict:
    request = _read(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    candidate_actions = _read(
        run_dir / "02_graph_analysis" / "candidate_actions.json"
    )
    return validate_agent_decision_response(
        request=request, response=response,
        candidate_actions=candidate_actions, run_dir=run_dir,
        selection_mode="agent-file",
    )


def test_nonexistent_candidate_id_fails(merlin_mlp_wide_run: Path) -> None:
    response = {
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
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(c for c in val["checks"] if c["name"] == "selected_candidate_exists")
    assert chk["status"] == "fail"


def test_illegal_candidate_id_fails(merlin_mlp_wide_run: Path) -> None:
    cas = _read(merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json")
    illegal = next(
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": illegal,
        "rationale": {
            "summary": "test",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(c for c in val["checks"] if c["name"] == "selected_candidate_is_legal")
    assert chk["status"] == "fail"


def test_legal_but_not_in_llm_view_fails(merlin_mlp_wide_run: Path) -> None:
    """Find a LEGAL candidate that's not in candidate_ids_allowed by
    artificially shrinking the request's allow-list."""
    request_path = (
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    request = _read(request_path)
    cas = _read(merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json")
    legal = [
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    ]
    target = legal[0]
    request["candidate_ids_allowed"] = [
        c for c in request["candidate_ids_allowed"] if c != target
    ]
    request_path.write_text(json.dumps(request), encoding="utf-8")

    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": target,
        "rationale": {
            "summary": "test",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(
        c for c in val["checks"] if c["name"] == "selected_candidate_visible_to_agent"
    )
    assert chk["status"] == "fail"


def test_missing_rationale_summary_fails(merlin_mlp_wide_run: Path) -> None:
    request = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": request["candidate_ids_allowed"][0],
        "rationale": {
            "summary": "",  # empty
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(c for c in val["checks"] if c["name"] == "rationale_summary_present")
    assert chk["status"] == "fail"


def test_empty_rationale_evidence_fails(merlin_mlp_wide_run: Path) -> None:
    request = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": request["candidate_ids_allowed"][0],
        "rationale": {"summary": "test", "evidence": []},
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(c for c in val["checks"] if c["name"] == "rationale_evidence_present")
    assert chk["status"] == "fail"


def test_evidence_referencing_nonexistent_field_fails(
    merlin_mlp_wide_run: Path,
) -> None:
    request = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": request["candidate_ids_allowed"][0],
        "rationale": {
            "summary": "test",
            "evidence": [
                {"field": "candidate.fake_field_1", "value": "x", "reason": "y"},
                {"field": "candidate.fake_field_2", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(
        c for c in val["checks"] if c["name"] == "rationale_references_real_fields"
    )
    assert chk["status"] == "fail"


def test_correctness_claim_in_rationale_fails(merlin_mlp_wide_run: Path) -> None:
    request = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": request["candidate_ids_allowed"][0],
        "rationale": {
            "summary": "this candidate is verified correct end-to-end",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(c for c in val["checks"] if c["name"] == "no_correctness_claim")
    assert chk["status"] == "fail"


def test_measured_perf_claim_in_rationale_fails(merlin_mlp_wide_run: Path) -> None:
    request = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": request["candidate_ids_allowed"][0],
        "rationale": {
            "summary": "we measured fastest among all alternatives",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = _run_validation(merlin_mlp_wide_run, response)
    assert val["overall"] == "fail"
    chk = next(
        c for c in val["checks"] if c["name"] == "no_measured_performance_claim"
    )
    assert chk["status"] == "fail"


def test_invalid_json_response_fails_cleanly(
    merlin_mlp_wide_run: Path, tmp_path: Path,
) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{this is not json", encoding="utf-8")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="agent-file",
        agent_response_path=bad_path,
    )
    assert result.overall == "fail"
    assert "invalid JSON" in result.rejection_reason


def test_agent_file_mode_without_response_path_fails_cleanly(
    merlin_mlp_wide_run: Path,
) -> None:
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="agent-file",
        agent_response_path=None,
    )
    assert result.overall == "fail"
    assert "requires --agent-decision-response" in result.rejection_reason


def test_invalid_response_fails_before_recipe_commit(tmp_path: Path) -> None:
    """Run a fresh M-14A pipeline with --selection-mode agent-file and
    a response that selects an illegal candidate; recipe.mlir must NOT
    be written / must not reference the bad candidate."""
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    cas = _read(src / "02_graph_analysis" / "candidate_actions.json")
    illegal = next(
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    )
    bad_response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": illegal,
        "rationale": {
            "summary": "test",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    bad_path = tmp_path / "bad_response.json"
    bad_path.write_text(json.dumps(bad_response), encoding="utf-8")

    out = tmp_path / "fail_run"
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "agent-file",
        "--agent-decision-response", str(bad_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode != 0, (
        f"agent-file with illegal candidate should fail; stdout={result.stdout}"
    )
    # recipe.mlir must NOT exist.
    recipe_path = out / "03_recipe_planning" / "recipe.mlir"
    assert not recipe_path.exists() or illegal not in recipe_path.read_text(
        encoding="utf-8",
    )


def test_no_compiler_core_imports_in_module() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "agent_decision.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"agent_decision must not import: {pat}"
