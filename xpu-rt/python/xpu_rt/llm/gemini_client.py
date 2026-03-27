"""Gemini API adapter — the primary LLM provider for CompGen.

Uses the google-genai SDK to call Gemini models. Handles API key loading
from .env (GEMMINI_API → GOOGLE_API_KEY), structured output via JSON mode,
and token/cost tracking.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


def _ensure_api_key() -> str:
    """Load Gemini API key from environment or .env file."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        key = os.environ.get("GEMMINI_API", "")
    if not key:
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMMINI_API="):
                    key = line.split("=", 1)[1].strip()
                    break
    if key:
        os.environ["GOOGLE_API_KEY"] = key
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
        return genai.Client(api_key=key)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response from Gemini.

        Renders the prompt template with context, calls the API,
        extracts artifacts from the response.
        """
        client = self._get_client()

        # Build prompt from request
        prompt = self._build_prompt(request)

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
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

        # Extract artifacts (code blocks, YAML blocks)
        artifacts = self._extract_artifacts(raw_text)

        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=artifacts,
            model_id=self.model,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=latency_ms,
            metadata={"finish_reason": "stop"},
        )

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any]
    ) -> GenerationResponse:
        """Generate structured (JSON) output from Gemini using response_mime_type."""
        from google.genai import types

        client = self._get_client()
        prompt = self._build_prompt(request)
        prompt += f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=request.config.temperature,
                max_output_tokens=request.config.max_tokens,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_text = response.text or ""
        usage = response.usage_metadata

        # Parse JSON
        artifacts = []
        try:
            parsed = json.loads(raw_text)
            artifacts.append(json.dumps(parsed))
        except json.JSONDecodeError:
            artifacts.append(raw_text)

        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=artifacts,
            model_id=self.model,
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=latency_ms,
            metadata={"format": "json"},
        )

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build prompt string from a GenerationRequest."""
        parts: list[str] = []

        if request.prompt_template:
            parts.append(request.prompt_template)

        ctx = request.context
        if ctx.model_ir_summary:
            parts.append(f"\n## IR Summary\n{ctx.model_ir_summary}")
        if ctx.target_profile_summary:
            parts.append(f"\n## Target Hardware\n{ctx.target_profile_summary}")
        if ctx.available_transforms:
            parts.append("\n## Available Transforms\n" + "\n".join(f"- {t}" for t in ctx.available_transforms))
        if ctx.kernel_contracts:
            parts.append("\n## Kernel Contracts\n" + "\n".join(ctx.kernel_contracts))
        if ctx.hardware_feedback:
            parts.append(f"\n## Hardware Feedback\n{ctx.hardware_feedback}")
        if ctx.prior_attempts:
            parts.append("\n## Prior Attempts\n" + "\n".join(ctx.prior_attempts))

        return "\n".join(parts)

    def _extract_artifacts(self, text: str) -> list[str]:
        """Extract code/YAML blocks from response text."""
        artifacts: list[str] = []
        in_block = False
        block_lines: list[str] = []

        for line in text.splitlines():
            if line.strip().startswith("```"):
                if in_block:
                    artifacts.append("\n".join(block_lines))
                    block_lines = []
                    in_block = False
                else:
                    in_block = True
            elif in_block:
                block_lines.append(line)

        return artifacts


# Protocol compliance check
def _check_protocol() -> None:
    client: CompGenLLMProtocol = GeminiClient()  # noqa: F841


__all__ = ["GeminiClient"]
