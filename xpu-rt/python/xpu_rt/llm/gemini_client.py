"""Gemini API adapter — the primary LLM provider for CompGen.

Uses the google-genai SDK to call Gemini models. Handles API key loading
from .env (GEMMINI_API → GOOGLE_API_KEY), structured output via JSON mode,
and token/cost tracking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from compgen.llm._env import resolve_api_key
from compgen.observability.gemini_usage import (
    install_genai_instrumentation,
    tracking_source,
)
from compgen.llm._prompt import (
    extract_markdown_artifacts,
    parse_json_payload,
    render_request_prompt,
    stringify_json_payload,
)
from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


def _ensure_api_key() -> str:
    """Load Gemini API key from environment or .env file."""
    key = resolve_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API")
    return key


@dataclass
class GeminiClient:
    """Gemini API adapter implementing CompGenLLMProtocol."""

    model: str = "gemini-2.5-flash"
    api_key: str | None = None

    def _get_client(self) -> Any:
        """Get or create the genai client."""
        from google import genai

        key = self.api_key or _ensure_api_key()
        if not key:
            raise RuntimeError("No Gemini API key. Set GEMMINI_API in .env or GOOGLE_API_KEY in environment.")
        # Patch the SDK so every call (ours + autocomp + anyone else) is
        # logged. Idempotent and best-effort.
        install_genai_instrumentation()
        return genai.Client(api_key=key)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response from Gemini.

        Renders the prompt template with context, calls the API,
        extracts artifacts from the response.
        """
        client = self._get_client()
        prompt = render_request_prompt(request)
        model = request.config.model or self.model

        t0 = time.perf_counter()
        with tracking_source("gemini_client.generate"):
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": request.config.temperature,
                    "max_output_tokens": request.config.max_tokens,
                    "top_p": request.config.top_p,
                },
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_text = response.text or ""
        usage = response.usage_metadata

        artifacts = extract_markdown_artifacts(raw_text)

        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=artifacts,
            model_id=model,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=latency_ms,
            metadata={"finish_reason": "stop"},
        )

    def generate_structured(self, request: GenerationRequest, schema: dict[str, Any]) -> GenerationResponse:
        """Generate structured (JSON) output from Gemini using response_mime_type."""
        try:
            from google.genai import types
        except ImportError:
            types = None  # type: ignore[assignment]

        client = self._get_client()
        prompt = render_request_prompt(request)
        prompt += f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        model = request.config.model or self.model

        t0 = time.perf_counter()
        if types is not None:
            config = types.GenerateContentConfig(
                temperature=request.config.temperature,
                max_output_tokens=request.config.max_tokens,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        else:
            config = {  # type: ignore[assignment]
                "temperature": request.config.temperature,
                "max_output_tokens": request.config.max_tokens,
                "response_mime_type": "application/json",
            }
        with tracking_source("gemini_client.generate_structured"):
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_text = response.text or ""
        usage = response.usage_metadata

        # Parse JSON
        artifacts = []
        try:
            parsed = parse_json_payload(raw_text)
            artifacts.append(stringify_json_payload(parsed))
        except json.JSONDecodeError:
            artifacts.append(raw_text)

        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=artifacts,
            model_id=model,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=latency_ms,
            metadata={"format": "json"},
        )


# Protocol compliance check
def _check_protocol() -> None:
    client: CompGenLLMProtocol = GeminiClient()  # noqa: F841


__all__ = ["GeminiClient"]
