"""Tests for the Gemini client adapter."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.gemini_client import GeminiClient


def _make_request(prompt: str = "Say hello") -> GenerationRequest:
    return GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="matmul r0",
            target_profile_summary="test-target",
            available_transforms=["tile", "eqsat"],
            kernel_contracts=["r0: layouts=rowmajor"],
            objective=Objective.LATENCY,
            frontend_diagnostics_summary="graph_breaks=0",
            analysis_dossier_summary="regions=1",
            unsupported_operator_summary="no unsupported operators",
        ),
        config=LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=128),
    )


class _FakeGeminiModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: str, config: object) -> object:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            text=self.text,
            usage_metadata=SimpleNamespace(prompt_token_count=17, candidates_token_count=5),
        )


class _FakeGeminiClient:
    def __init__(self, text: str) -> None:
        self.models = _FakeGeminiModels(text)


def test_gemini_client_defaults() -> None:
    client = GeminiClient()
    assert client.model == "gemini-2.5-flash"


def test_gemini_client_generate_uses_rendered_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGeminiClient("ready")
    client = GeminiClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_client", lambda: fake)

    response = client.generate(_make_request("Analyze this graph"))

    assert response.raw_text == "ready"
    assert response.model_id == "gemini-2.5-flash"
    assert response.prompt_tokens == 17
    assert response.completion_tokens == 5
    contents = str(fake.models.calls[0]["contents"])
    assert "## Frontend Diagnostics" in contents
    assert "## Analysis Dossier" in contents
    assert "Analyze this graph" in contents


def test_gemini_client_generate_structured_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeGeminiClient('{"tile_m": 64, "tile_n": 64, "tile_k": 32}')
    client = GeminiClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_client", lambda: fake)

    response = client.generate_structured(
        _make_request("Suggest tile sizes"),
        {
            "type": "object",
            "properties": {
                "tile_m": {"type": "integer"},
                "tile_n": {"type": "integer"},
                "tile_k": {"type": "integer"},
            },
        },
    )

    parsed = json.loads(response.parsed_artifacts[0])
    assert parsed["tile_m"] == 64
    assert response.metadata["format"] == "json"


@pytest.mark.slow
def test_gemini_client_generate_real_smoke() -> None:
    if os.environ.get("COMPGEN_RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("Set COMPGEN_RUN_REAL_LLM_TESTS=1 to enable real LLM smoke tests.")
    if os.environ.get("COMPGEN_REAL_LLM_BACKEND", "").strip().lower() not in {"gemini", "gemmini"}:
        pytest.skip("Real LLM smoke test is configured for another backend.")

    client = GeminiClient(model=os.environ.get("COMPGEN_REAL_LLM_MODEL", "gemini-2.5-flash"))
    response = client.generate(_make_request("Say ready in one word."))
    assert response.raw_text.strip()
