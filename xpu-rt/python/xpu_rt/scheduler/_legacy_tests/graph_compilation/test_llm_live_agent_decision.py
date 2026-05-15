"""Tests for the live LLM provider integration.

Exercises:

- The provider plug-in surface (register_provider) using REAL test
  providers — not mocks. A test-registered provider is a real
  ProviderCallResult-returning callable with hand-crafted JSON; the
  validator path it exercises is identical to the path a real
  anthropic/openai response takes.
- Dry-run halts before recipe.mlir without calling the provider.
- Provider failures (typed, malformed, prose, illegal/hidden/false-claim
  responses) all fail BEFORE recipe.mlir is written.
- API keys never leak into emitted artifacts.
- Fallback semantics (none).

Note: there is no built-in ``mock`` provider in CompGen — for
no-API-key workflows use ``--selection-mode agent-file`` with Claude
Code instead. Tests that need a deterministic provider register one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compgen.graph_compilation.agent_decision import (
    LiveProviderConfig,
    run_agent_decision,
)
from compgen.graph_compilation.llm_live_provider import (
    ProviderCallResult,
    ProviderError,
    build_prompt,
    parse_provider_response_text,
    register_provider,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WIDE = REPO_ROOT / "results" / "graph_compilation" / "m14a_wide_llm_stub_suite"


def _need_wide() -> None:
    if not WIDE.is_dir():
        pytest.skip(f"wide fixture suite missing: {WIDE}")


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def merlin_mlp_wide_run(tmp_path: Path) -> Path:
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    dst = tmp_path / "merlin_mlp_wide"
    shutil.copytree(src, dst)
    ad = dst / "03_recipe_planning" / "agent_decision"
    if ad.exists():
        shutil.rmtree(ad)
    recipe = dst / "03_recipe_planning" / "recipe.mlir"
    if recipe.exists():
        recipe.unlink()
    return dst


def _register_for_test(name: str, fn) -> None:  # type: ignore[no-untyped-def]
    register_provider(name, fn)


def _result_with_text(text: str, *, model: str = "test-model") -> ProviderCallResult:
    return ProviderCallResult(
        raw_response={
            "provider": "test", "model": model,
            "completion_text": text, "completion_kind": "json",
            "metadata": {},
        },
        parsed_response=None,
        latency_ms=10,
        prompt="<test>",
        provider_name="test",
        model=model,
    )


def _build_good_test_provider(merlin_mlp_wide_run: Path):  # type: ignore[no-untyped-def]
    """Build a test-provider that returns a real, valid response shaped
    like an anthropic/openai completion. The selected candidate is the
    first legal candidate in the run's candidate_actions.json. This is a
    real provider implementation registered for the test, not a mock —
    the response shape and validation path are identical to a live LLM
    response."""
    from compgen.graph_compilation.agent_decision import (
        build_agent_decision_request,
    )
    build_agent_decision_request(merlin_mlp_wide_run)
    cas = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal = next(
        c for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    )
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": legal["candidate_id"],
        "rationale": {
            "summary": (
                f"Selected legal {legal['kind']!r} candidate; cost "
                f"preview indicates a low static_relative_cost."
            ),
            "evidence": [
                {"field": "candidate.kind", "value": legal["kind"],
                 "reason": "matches the legal action surface"},
                {"field": "candidate.label",
                 "value": legal.get("label", ""),
                 "reason": "label resolves against candidate_actions"},
            ],
        },
    }

    def provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(json.dumps(response), model="test-good-1")
    return provider


# --------------------------------------------------------------------------- #
# Good provider end-to-end (real test-registered provider, not a mock)
# --------------------------------------------------------------------------- #


def test_test_provider_emits_all_artifacts(merlin_mlp_wide_run: Path) -> None:
    _register_for_test("test_good", _build_good_test_provider(merlin_mlp_wide_run))
    cfg = LiveProviderConfig(provider="test_good", model="test-good-1")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "pass", result.rejection_reason
    ad = merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
    for name in (
        "agent_decision_request.json",
        "agent_decision_prompt.txt",
        "agent_decision_provider_request.json",
        "agent_decision_provider_response.raw.json",
        "agent_decision_response.json",
        "agent_decision_validation.json",
        "agent_decision_trace.json",
    ):
        assert (ad / name).exists(), f"missing {name}"


def test_test_provider_validation_passes(merlin_mlp_wide_run: Path) -> None:
    _register_for_test("test_good2", _build_good_test_provider(merlin_mlp_wide_run))
    cfg = LiveProviderConfig(provider="test_good2", model="test-good-1")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    val = _read(result.validation_path)
    assert val["overall"] == "pass"
    assert val["selection_mode"] == "llm-live"
    assert all(c["status"] == "pass" for c in val["checks"])


def test_trace_records_provider_block(merlin_mlp_wide_run: Path) -> None:
    _register_for_test("test_good3", _build_good_test_provider(merlin_mlp_wide_run))
    cfg = LiveProviderConfig(provider="test_good3", model="test-good-1")
    run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    trace = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_trace.json"
    )
    assert trace["selection_mode"] == "llm-live"
    p = trace["provider"]
    assert p["dry_run"] is False
    assert p["fallback_used"] is False
    assert p["latency_ms"] >= 0
    assert p["prompt_sha256"].startswith("sha256:")
    assert p["raw_response_sha256"].startswith("sha256:")


# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_emits_prompt_and_request_but_no_recipe(
    merlin_mlp_wide_run: Path,
) -> None:
    _register_for_test("test_good_dry", _build_good_test_provider(merlin_mlp_wide_run))
    cfg = LiveProviderConfig(provider="test_good_dry", dry_run=True)
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"  # dry-run halts before commit
    assert "dry_run" in result.rejection_reason
    ad = merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
    assert (ad / "agent_decision_prompt.txt").exists()
    assert (ad / "agent_decision_provider_request.json").exists()
    assert (ad / "agent_decision_dry_run.json").exists()
    assert not (ad / "agent_decision_provider_response.raw.json").exists()
    assert not (
        merlin_mlp_wide_run / "03_recipe_planning" / "recipe.mlir"
    ).exists()


# --------------------------------------------------------------------------- #
# Bad-provider scenarios via register_provider
# --------------------------------------------------------------------------- #


def test_provider_returning_nonexistent_candidate_fails(
    merlin_mlp_wide_run: Path,
) -> None:
    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(json.dumps({
            "schema_version": "agent_decision_response_v1",
            "selected_candidate_id": "cand_does_not_exist_xyz",
            "rationale": {
                "summary": "test",
                "evidence": [
                    {"field": "candidate.kind", "value": "x", "reason": "y"},
                    {"field": "candidate.label", "value": "x", "reason": "y"},
                ],
            },
        }))
    _register_for_test("bad_nonexistent", bad_provider)
    cfg = LiveProviderConfig(provider="bad_nonexistent", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    val = _read(result.validation_path)
    chk = next(
        c for c in val["checks"] if c["name"] == "selected_candidate_exists"
    )
    assert chk["status"] == "fail"


def test_provider_returning_illegal_candidate_fails(
    merlin_mlp_wide_run: Path,
) -> None:
    cas = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json"
    )
    illegal = next(
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    )

    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(json.dumps({
            "schema_version": "agent_decision_response_v1",
            "selected_candidate_id": illegal,
            "rationale": {
                "summary": "test",
                "evidence": [
                    {"field": "candidate.kind", "value": "x", "reason": "y"},
                    {"field": "candidate.label", "value": "x", "reason": "y"},
                ],
            },
        }))
    _register_for_test("bad_illegal", bad_provider)
    cfg = LiveProviderConfig(provider="bad_illegal", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    val = _read(result.validation_path)
    chk = next(
        c for c in val["checks"] if c["name"] == "selected_candidate_is_legal"
    )
    assert chk["status"] == "fail"


def test_provider_returning_correctness_claim_fails(
    merlin_mlp_wide_run: Path,
) -> None:
    request_path = (
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    from compgen.graph_compilation.agent_decision import (
        build_agent_decision_request,
    )
    build_agent_decision_request(merlin_mlp_wide_run)
    req = _read(request_path)
    legal_id = req["candidate_ids_allowed"][0]

    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(json.dumps({
            "schema_version": "agent_decision_response_v1",
            "selected_candidate_id": legal_id,
            "rationale": {
                "summary": "this is verified correct end-to-end",
                "evidence": [
                    {"field": "candidate.kind", "value": "x", "reason": "y"},
                    {"field": "candidate.label", "value": "x", "reason": "y"},
                ],
            },
        }))
    _register_for_test("bad_correctness", bad_provider)
    cfg = LiveProviderConfig(provider="bad_correctness", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    val = _read(result.validation_path)
    chk = next(c for c in val["checks"] if c["name"] == "no_correctness_claim")
    assert chk["status"] == "fail"


def test_provider_returning_perf_claim_fails(merlin_mlp_wide_run: Path) -> None:
    from compgen.graph_compilation.agent_decision import (
        build_agent_decision_request,
    )
    build_agent_decision_request(merlin_mlp_wide_run)
    req = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    legal_id = req["candidate_ids_allowed"][0]

    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(json.dumps({
            "schema_version": "agent_decision_response_v1",
            "selected_candidate_id": legal_id,
            "rationale": {
                "summary": "we benchmarked this and it's measured fastest",
                "evidence": [
                    {"field": "candidate.kind", "value": "x", "reason": "y"},
                    {"field": "candidate.label", "value": "x", "reason": "y"},
                ],
            },
        }))
    _register_for_test("bad_perf", bad_provider)
    cfg = LiveProviderConfig(provider="bad_perf", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    val = _read(result.validation_path)
    chk = next(
        c for c in val["checks"] if c["name"] == "no_measured_performance_claim"
    )
    assert chk["status"] == "fail"


def test_provider_returning_malformed_json_fails(
    merlin_mlp_wide_run: Path,
) -> None:
    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text("{this is not json")
    _register_for_test("bad_malformed", bad_provider)
    cfg = LiveProviderConfig(provider="bad_malformed", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    err = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "provider_error.json"
    )
    assert err["error_type"] == "ProviderError"
    assert "valid JSON" in err["error_message"]


def test_provider_returning_prose_fails(merlin_mlp_wide_run: Path) -> None:
    def bad_provider(**kwargs):  # type: ignore[no-untyped-def]
        return _result_with_text(
            "I think you should pick the first candidate, no need for JSON."
        )
    _register_for_test("bad_prose", bad_provider)
    cfg = LiveProviderConfig(provider="bad_prose", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    err = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "provider_error.json"
    )
    assert err["error_type"] == "ProviderError"


def test_provider_exception_emits_provider_error(
    merlin_mlp_wide_run: Path,
) -> None:
    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise ProviderError("provider exploded for testing")
    _register_for_test("boom", boom)
    cfg = LiveProviderConfig(provider="boom", model="x")
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    err = _read(
        merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
        / "provider_error.json"
    )
    assert err["error_message"] == "provider exploded for testing"


def test_provider_hidden_candidate_fails(merlin_mlp_wide_run: Path) -> None:
    """Direct validator test for "hidden but selected" — the visibility
    check rejects any selection not in candidate_ids_allowed."""
    from compgen.graph_compilation.agent_decision import (
        build_agent_decision_request,
        validate_agent_decision_response,
    )
    request_path = build_agent_decision_request(merlin_mlp_wide_run)
    request = _read(request_path)
    cas = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal_target = next(
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    )
    request["candidate_ids_allowed"] = [
        c for c in request["candidate_ids_allowed"] if c != legal_target
    ]
    response = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": legal_target,
        "rationale": {
            "summary": "provider picked a hidden but legal candidate",
            "evidence": [
                {"field": "candidate.kind", "value": "x", "reason": "y"},
                {"field": "candidate.label", "value": "x", "reason": "y"},
            ],
        },
    }
    val = validate_agent_decision_response(
        request=request, response=response,
        candidate_actions=cas, run_dir=merlin_mlp_wide_run,
        selection_mode="llm-live",
    )
    assert val["overall"] == "fail"
    chk = next(
        c for c in val["checks"]
        if c["name"] == "selected_candidate_visible_to_agent"
    )
    assert chk["status"] == "fail"


# --------------------------------------------------------------------------- #
# Secret-handling: API keys never written to disk
# --------------------------------------------------------------------------- #


def test_no_api_key_in_emitted_artifacts(merlin_mlp_wide_run: Path) -> None:
    """Set a fake API key in the env, run a test-registered provider,
    scan every emitted artifact for the secret."""
    secret = "sk-test-do-not-leak-this-secret-12345"
    import os

    _register_for_test(
        "test_secret", _build_good_test_provider(merlin_mlp_wide_run),
    )
    old = os.environ.get("COMPGEN_LLM_API_KEY")
    os.environ["COMPGEN_LLM_API_KEY"] = secret
    try:
        cfg = LiveProviderConfig(provider="test_secret", model="x")
        run_agent_decision(
            merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
        )
    finally:
        if old is None:
            os.environ.pop("COMPGEN_LLM_API_KEY", None)
        else:
            os.environ["COMPGEN_LLM_API_KEY"] = old

    ad = merlin_mlp_wide_run / "03_recipe_planning" / "agent_decision"
    for path in ad.iterdir():
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8")
        assert secret not in body, (
            f"API key leaked into {path.name}"
        )


# --------------------------------------------------------------------------- #
# Fallback semantics
# --------------------------------------------------------------------------- #


def test_fallback_none_aborts_on_failure(merlin_mlp_wide_run: Path) -> None:
    def boom(**kwargs):  # type: ignore[no-untyped-def]
        raise ProviderError("test")
    _register_for_test("boom2", boom)
    cfg = LiveProviderConfig(
        provider="boom2", model="x", fallback="none",
    )
    result = run_agent_decision(
        merlin_mlp_wide_run, selection_mode="llm-live", live_config=cfg,
    )
    assert result.overall == "fail"
    trace = _read(result.trace_path)
    assert trace["provider"]["fallback_used"] is False


# --------------------------------------------------------------------------- #
# Parser unit tests
# --------------------------------------------------------------------------- #


def test_parse_bare_json_object() -> None:
    obj = parse_provider_response_text('{"a": 1, "b": "two"}')
    assert obj == {"a": 1, "b": "two"}


def test_parse_fenced_json_object() -> None:
    text = '```json\n{"a": 1}\n```'
    assert parse_provider_response_text(text) == {"a": 1}
    text2 = '```\n{"a": 2}\n```'
    assert parse_provider_response_text(text2) == {"a": 2}


def test_parse_rejects_prose() -> None:
    with pytest.raises(ProviderError):
        parse_provider_response_text("just some text")


def test_parse_rejects_array() -> None:
    with pytest.raises(ProviderError):
        parse_provider_response_text("[1, 2, 3]")


def test_parse_rejects_empty() -> None:
    with pytest.raises(ProviderError):
        parse_provider_response_text("")


# --------------------------------------------------------------------------- #
# Prompt-shape sanity
# --------------------------------------------------------------------------- #


def test_build_prompt_contains_request_and_view() -> None:
    request = {"candidate_ids_allowed": ["cand_a", "cand_b"]}
    view = {"regions": [{"region_id": "matmul_0"}]}
    prompt = build_prompt(request=request, llm_graph_view=view)
    assert "cand_a" in prompt
    assert "matmul_0" in prompt
    assert "Return only JSON" in prompt
    # PE-3 strengthened the prompt (case-emphasized "NOT").
    assert "Do NOT invent candidate IDs" in prompt
