"""Tests for MockLLMClient replay behavior."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.mock_client import MockLLMClient


def _make_request(prompt: str) -> GenerationRequest:
    """Helper to create a GenerationRequest with a prompt string."""
    return GenerationRequest(
        prompt_template=prompt,
        context=PromptContext(
            model_ir_summary="",
            target_profile_summary="",
            available_transforms=[],
            kernel_contracts=[],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="mock"),
    )


def test_mock_client_instantiation() -> None:
    client = MockLLMClient()
    assert client.strict is True


def test_fragment_response() -> None:
    client = MockLLMClient()
    client.add_response("matmul", "Use Triton kernel for matmul")

    response = client.generate(_make_request("Optimize this matmul operation"))
    assert response.raw_text == "Use Triton kernel for matmul"
    assert response.model_id == "mock"


def test_exact_response() -> None:
    client = MockLLMClient()
    prompt = "Generate a tiling pass"
    client.add_exact_response(prompt, "class TilePass: pass")

    response = client.generate(_make_request(prompt))
    assert response.raw_text == "class TilePass: pass"


def test_strict_mode_raises() -> None:
    client = MockLLMClient(strict=True)
    with pytest.raises(KeyError, match="No mock response"):
        client.generate(_make_request("unknown prompt"))


def test_lenient_mode_returns_empty() -> None:
    client = MockLLMClient(strict=False)
    response = client.generate(_make_request("unknown prompt"))
    assert response.raw_text == ""


def test_generate_structured() -> None:
    client = MockLLMClient()
    client.add_response("test", '{"key": "value"}')

    response = client.generate_structured(_make_request("test structured"), schema={"type": "object"})
    assert response.raw_text == '{"key": "value"}'


def test_load_replay_from_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = {"prompt": "hello world", "response": "hi there"}
        with open(Path(tmp) / "001.json", "w") as f:
            json.dump(data, f)

        client = MockLLMClient()
        client.load_replay(Path(tmp))

        response = client.generate(_make_request("hello world"))
        assert response.raw_text == "hi there"


def test_load_replay_nonexistent_dir() -> None:
    client = MockLLMClient()
    client.load_replay(Path("/nonexistent"))


def test_multiple_fragment_responses() -> None:
    client = MockLLMClient()
    client.add_response("matmul", "matmul response")
    client.add_response("conv2d", "conv response")

    r1 = client.generate(_make_request("optimize matmul"))
    r2 = client.generate(_make_request("optimize conv2d"))
    assert r1.raw_text == "matmul response"
    assert r2.raw_text == "conv response"
