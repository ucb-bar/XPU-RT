"""Tests for GeminiClient adapter."""

from __future__ import annotations

from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.llm.gemini_client import GeminiClient


def test_gemini_client_defaults() -> None:
    """GeminiClient should have sensible defaults."""
    client = GeminiClient()
    assert client.model == "gemini-2.5-flash"


def test_gemini_client_custom_model() -> None:
    client = GeminiClient(model="gemini-2.5-pro", api_key="test-key")
    assert client.model == "gemini-2.5-pro"
    assert client.api_key == "test-key"


def test_gemini_client_generate_real() -> None:
    """Real API call to Gemini. Requires GEMMINI_API in .env."""
    client = GeminiClient(model="gemini-2.5-flash")

    request = GenerationRequest(
        prompt_template="You are a compiler optimization assistant. Say 'ready' in one word.",
        context=PromptContext(
            model_ir_summary="",
            target_profile_summary="",
            available_transforms=[],
            kernel_contracts=[],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=50),
    )

    response = client.generate(request)
    assert len(response.raw_text) > 0
    assert response.model_id == "gemini-2.5-flash"
    assert response.prompt_tokens > 0
    assert response.latency_ms > 0


def test_gemini_client_generate_with_context() -> None:
    """Generate with IR summary and target profile context."""
    client = GeminiClient(model="gemini-2.5-flash")

    request = GenerationRequest(
        prompt_template="What optimization would you suggest for this IR?",
        context=PromptContext(
            model_ir_summary="matmul_0: 8x768 @ 768x3072 → 8x3072, 37.7M FLOPs, memory-bound",
            target_profile_summary="A100-SXM4-80GB: 312 TFLOPS FP16, 2039 GB/s HBM",
            available_transforms=["tile", "fuse", "vectorize"],
            kernel_contracts=["matmul: tile_sizes must divide dimensions"],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="gemini-2.5-flash", temperature=0.3, max_tokens=200),
    )

    response = client.generate(request)
    assert len(response.raw_text) > 0
    assert response.prompt_tokens > 0


def test_gemini_client_generate_structured() -> None:
    """Generate structured JSON output."""
    client = GeminiClient(model="gemini-2.5-flash")

    request = GenerationRequest(
        prompt_template="Suggest tile sizes for a matmul of shape 8x768x3072.",
        context=PromptContext(
            model_ir_summary="",
            target_profile_summary="",
            available_transforms=[],
            kernel_contracts=[],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="gemini-2.5-flash", temperature=0.0, max_tokens=200),
    )

    schema = {
        "type": "object",
        "properties": {
            "tile_m": {"type": "integer"},
            "tile_n": {"type": "integer"},
            "tile_k": {"type": "integer"},
        },
    }

    response = client.generate_structured(request, schema)
    assert len(response.raw_text) > 0
    assert len(response.parsed_artifacts) > 0
    # Should be valid JSON
    import json
    parsed = json.loads(response.parsed_artifacts[0])
    assert isinstance(parsed, dict)
