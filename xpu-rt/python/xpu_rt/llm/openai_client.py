"""OpenAI Responses API adapter for CompGen."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from compgen.llm._env import resolve_api_key
from compgen.llm._prompt import (
    extract_markdown_artifacts,
    parse_json_payload,
    render_request_prompt,
    stringify_json_payload,
)
from compgen.llm.base import (
    GenerationRequest,
    GenerationResponse,
)


def _ensure_api_key() -> str:
    return resolve_api_key("OPENAI_API_KEY")


@dataclass
class OpenAIClient:
    """OpenAI adapter implementing ``CompGenLLMProtocol``."""

    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    base_url: str | None = None

    def _get_client(self) -> Any:
        from openai import OpenAI

        key = self.api_key or _ensure_api_key()
        if not key:
            raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY in environment or .env.")
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        prompt = render_request_prompt(request)
        model = request.config.model or self.model

        t0 = time.perf_counter()
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=request.config.temperature,
            max_output_tokens=request.config.max_tokens,
            top_p=request.config.top_p,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            raw_text=raw_text,
            parsed_artifacts=extract_markdown_artifacts(raw_text),
            model_id=model,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
            metadata={"response_id": getattr(response, "id", "")},
        )

    def generate_structured(
        self,
        request: GenerationRequest,
        schema: dict[str, Any],
    ) -> GenerationResponse:
        structured_request = GenerationRequest(
            prompt_template=(
                f"{request.prompt_template}\n\nRespond with valid JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            ),
            context=request.context,
            config=request.config,
            artifact_type=request.artifact_type,
        )
        response = self.generate(structured_request)
        artifacts = []
        try:
            parsed = parse_json_payload(response.raw_text)
            artifacts = [stringify_json_payload(parsed)]
        except json.JSONDecodeError:
            artifacts = response.parsed_artifacts or [response.raw_text]
        response.parsed_artifacts = artifacts
        return response


__all__ = ["OpenAIClient"]
