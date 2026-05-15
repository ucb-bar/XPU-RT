"""Tests for LLM interaction recorder."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.recorder import LLMRecorder


def _make_request(prompt: str = "Say hello") -> GenerationRequest:
    return GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="region r0",
            target_profile_summary="a100",
            available_transforms=["tile"],
            kernel_contracts=["r0: backends=triton"],
            objective=Objective.LATENCY,
            frontend_diagnostics_summary="graph_breaks=0",
            analysis_dossier_summary="regions=1",
            unsupported_operator_summary="no unsupported operators",
            frontier_summary="step=0",
        ),
        config=LLMConfig(model="mock", temperature=0.0, max_tokens=50),
    )


def _make_client() -> MockLLMClient:
    client = MockLLMClient(strict=False)
    client.add_response("Say", "test")
    return client


def test_recorder_wraps_client(tmp_path: Path) -> None:
    recorder = LLMRecorder(wrapped=_make_client(), log_dir=tmp_path / "llm_logs")

    response = recorder.generate(_make_request("Say 'test' in one word."))
    assert response.raw_text == "test"
    assert recorder.total_calls == 1

    log_files = list((tmp_path / "llm_logs").glob("*.json"))
    assert len(log_files) == 1


def test_recorder_logs_contain_context_metadata(tmp_path: Path) -> None:
    recorder = LLMRecorder(wrapped=_make_client(), log_dir=tmp_path / "logs")
    recorder.generate(_make_request("Say hello"))

    log_files = list((tmp_path / "logs").glob("*.json"))
    data = json.loads(log_files[0].read_text())
    assert "model" in data
    assert "timestamp" in data
    assert data["context"]["frontend_diagnostics_summary"] == "graph_breaks=0"
    assert "## Frontend Diagnostics" in data["prompt_preview"]


def test_recorder_disabled(tmp_path: Path) -> None:
    recorder = LLMRecorder(wrapped=_make_client(), log_dir=tmp_path / "logs", enabled=False)
    recorder.generate(_make_request("Say hello"))

    log_files = list((tmp_path / "logs").glob("*.json"))
    assert len(log_files) == 0
