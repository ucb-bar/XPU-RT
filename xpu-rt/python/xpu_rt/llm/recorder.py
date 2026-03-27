"""LLM interaction recorder middleware.

Every LLM call in CompGen passes through the recorder. It wraps any
CompGenLLMProtocol implementor and logs all interactions to disk as
JSON for reproducibility, debugging, and cost tracking.

Each interaction is saved as a JSON file with:
- The full prompt (rendered)
- The raw response text
- Model ID, token counts, latency
- Timestamp
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


@dataclass
class LLMRecorder:
    """Recording middleware for LLM interactions.

    Wraps any CompGenLLMProtocol client and logs all calls to disk.
    """

    wrapped: CompGenLLMProtocol
    log_dir: Path
    enabled: bool = True
    _call_count: int = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Forward to wrapped client and record the interaction."""
        t0 = time.time()
        response = self.wrapped.generate(request)
        self._record(request, response, t0)
        return response

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any]
    ) -> GenerationResponse:
        """Forward structured generation and record."""
        t0 = time.time()
        response = self.wrapped.generate_structured(request, schema)
        self._record(request, response, t0, schema=schema)
        return response

    def _record(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
        start_time: float,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Write interaction to disk as JSON."""
        if not self.enabled:
            return

        self._call_count += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt text for hashing
        prompt_text = request.prompt_template or ""
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]

        record = {
            "call_id": self._call_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "model": response.model_id,
            "config": {
                "temperature": request.config.temperature,
                "max_tokens": request.config.max_tokens,
                "top_p": request.config.top_p,
            },
            "prompt_hash": prompt_hash,
            "prompt_template": request.prompt_template[:500] if request.prompt_template else "",
            "artifact_type": request.artifact_type,
            "response": {
                "raw_text_length": len(response.raw_text),
                "raw_text_preview": response.raw_text[:200],
                "num_artifacts": len(response.parsed_artifacts),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_ms": response.latency_ms,
            },
            "metadata": response.metadata,
        }

        if schema:
            record["schema"] = schema

        filename = f"{self._call_count:04d}_{prompt_hash}.json"
        (self.log_dir / filename).write_text(json.dumps(record, indent=2, default=str))

    def replay_log(self, log_path: Path) -> GenerationResponse:
        """Load a previously recorded response from a log file."""
        data = json.loads(log_path.read_text())
        return GenerationResponse(
            raw_text=data.get("response", {}).get("raw_text_preview", ""),
            parsed_artifacts=[],
            model_id=data.get("model", ""),
            prompt_tokens=data.get("response", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("response", {}).get("completion_tokens", 0),
            latency_ms=data.get("response", {}).get("latency_ms", 0),
        )

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls recorded."""
        return self._call_count

    @property
    def total_tokens(self) -> int:
        """Approximate total tokens from log files."""
        total = 0
        if self.log_dir.exists():
            for f in self.log_dir.glob("*.json"):
                data = json.loads(f.read_text())
                resp = data.get("response", {})
                total += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
        return total


__all__ = ["LLMRecorder"]
