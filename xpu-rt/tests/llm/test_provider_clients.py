"""Unit tests for non-Gemini provider and CLI adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compgen.llm.anthropic_client import AnthropicClient
from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.cli_client import ClaudeCLIClient, CodexCLIClient
from compgen.llm.factory import create_llm_client
from compgen.llm.gemini_client import GeminiClient
from compgen.llm.openai_client import OpenAIClient


def _make_request(prompt: str = "Say hello") -> GenerationRequest:
    return GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="region r0",
            target_profile_summary="target",
            available_transforms=["tile", "eqsat"],
            kernel_contracts=["r0: backends=triton"],
            objective=Objective.LATENCY,
            frontend_diagnostics_summary="graph_breaks=1",
            analysis_dossier_summary="regions=2",
        ),
        config=LLMConfig(model="test-model", temperature=0.0, max_tokens=128),
    )


def test_openai_client_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Responses:
        def create(self, **kwargs: object) -> object:
            assert kwargs["model"] == "test-model"
            return SimpleNamespace(
                id="resp_123",
                output_text="```mlir\ntransform.sequence {}\n```",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    fake_client = SimpleNamespace(responses=_Responses())
    client = OpenAIClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.generate(_make_request("Generate a transform"))
    assert response.model_id == "test-model"
    assert "transform.sequence" in response.parsed_artifacts[0]


def test_anthropic_client_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Messages:
        def create(self, **kwargs: object) -> object:
            assert kwargs["model"] == "test-model"
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ready")],
                usage=SimpleNamespace(input_tokens=9, output_tokens=3),
                stop_reason="end_turn",
            )

    fake_client = SimpleNamespace(messages=_Messages())
    client = AnthropicClient(api_key="test-key")
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    response = client.generate(_make_request("Say ready"))
    assert response.raw_text == "ready"
    assert response.metadata["stop_reason"] == "end_turn"


def test_claude_cli_client_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(args: list[str], **kwargs: object) -> object:
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='{"action":"noop"}', stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    client = ClaudeCLIClient(model="sonnet", working_dir=Path("/tmp"))
    response = client.generate_structured(
        _make_request("Return json"),
        {"type": "object", "properties": {"action": {"type": "string"}}},
    )

    assert json.loads(response.parsed_artifacts[0])["action"] == "noop"
    assert "--json-schema" in calls[0]
    assert "--tools" in calls[0]


def test_codex_cli_client_uses_output_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(args: list[str], **kwargs: object) -> object:
        calls.append(args)
        out_path = Path(args[args.index("-o") + 1])
        if "--output-schema" in args:
            out_path.write_text('{"decision":"eqsat"}')
        else:
            out_path.write_text("ready")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)
    client = CodexCLIClient(model="gpt-5.4-mini", working_dir=Path("/tmp"))

    response = client.generate(_make_request("Say ready"))
    assert response.raw_text == "ready"
    assert "exec" in calls[0]

    structured = client.generate_structured(
        _make_request("Return json"),
        {"type": "object", "properties": {"decision": {"type": "string"}}},
    )
    assert json.loads(structured.parsed_artifacts[0])["decision"] == "eqsat"


def test_create_llm_client_explicit_providers() -> None:
    assert isinstance(create_llm_client("gemini"), GeminiClient)
    assert isinstance(create_llm_client("openai"), OpenAIClient)
    assert isinstance(create_llm_client("anthropic"), AnthropicClient)
    assert isinstance(create_llm_client("claude-cli"), ClaudeCLIClient)
    assert isinstance(create_llm_client("codex-cli"), CodexCLIClient)
