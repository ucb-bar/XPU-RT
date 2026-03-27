"""Tests for LLM interaction recorder."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.gemini_client import GeminiClient
from compgen.llm.recorder import LLMRecorder


def _make_request(prompt: str = "Say hello") -> GenerationRequest:
    return GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="",
            target_profile_summary="",
            available_transforms=[],
            kernel_contracts=[],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=50),
    )


def test_recorder_wraps_client(tmp_path: Path) -> None:
    """Recorder should forward calls and log to disk."""
    client = GeminiClient(model="gemini-2.5-flash")
    recorder = LLMRecorder(wrapped=client, log_dir=tmp_path / "llm_logs")

    response = recorder.generate(_make_request("Say 'test' in one word."))
    assert len(response.raw_text) > 0
    assert recorder.total_calls == 1

    log_files = list((tmp_path / "llm_logs").glob("*.json"))
    assert len(log_files) == 1


def test_recorder_logs_contain_metadata(tmp_path: Path) -> None:
    """Log files should contain model, tokens, timestamp."""
    client = GeminiClient(model="gemini-2.5-flash")
    recorder = LLMRecorder(wrapped=client, log_dir=tmp_path / "logs")
    recorder.generate(_make_request("Say hello"))

    log_files = list((tmp_path / "logs").glob("*.json"))
    data = json.loads(log_files[0].read_text())
    assert "model" in data
    assert "timestamp" in data
    assert data["response"]["prompt_tokens"] > 0


def test_recorder_disabled(tmp_path: Path) -> None:
    """Disabled recorder should not create log files."""
    client = GeminiClient(model="gemini-2.5-flash")
    recorder = LLMRecorder(wrapped=client, log_dir=tmp_path / "logs", enabled=False)
    recorder.generate(_make_request("Say hello"))

    log_files = list((tmp_path / "logs").glob("*.json"))
    assert len(log_files) == 0
