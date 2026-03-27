"""Deterministic mock/replay LLM client for testing.

The mock client serves two roles:

1. **Replay mode**: Load pre-recorded prompt/response pairs from a directory
   and return deterministic responses. Used for regression testing.

2. **Direct mode**: Manually add prompt→response mappings for test setup.

Invariants:
    - Responses are deterministic given the same prompts.
    - Missing replay entries raise an error (strict) or return empty (lenient).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
)


def _normalize_prompt(prompt: str) -> str:
    """Normalize whitespace for fuzzy matching."""
    return " ".join(prompt.split())


def _hash_prompt(prompt: str) -> str:
    """Hash a normalized prompt for lookup."""
    return hashlib.sha256(_normalize_prompt(prompt).encode()).hexdigest()[:16]


@dataclass
class MockLLMClient:
    """Deterministic mock LLM client for testing.

    Attributes:
        replay_dir: Directory containing recorded prompt/response JSON files.
        strict: If True, raise on prompt mismatch. If False, return empty response.
    """

    replay_dir: Path | None = None
    strict: bool = True
    _responses: dict[str, str] = field(default_factory=dict, repr=False)
    _fragment_responses: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def add_response(self, prompt_fragment: str, response: str) -> None:
        """Add a prompt→response mapping for testing.

        Matches if prompt_fragment appears anywhere in the prompt.

        Args:
            prompt_fragment: Substring to match in prompts.
            response: Response text to return.
        """
        self._fragment_responses.append((prompt_fragment, response))

    def add_exact_response(self, prompt: str, response: str) -> None:
        """Add an exact prompt→response mapping."""
        self._responses[_hash_prompt(prompt)] = response

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Return a pre-recorded response for the given request."""
        prompt = request.prompt_template

        # Check exact matches first
        key = _hash_prompt(prompt)
        if key in self._responses:
            text = self._responses[key]
            return GenerationResponse(
                raw_text=text,
                parsed_artifacts=[text],
                model_id="mock",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
            )

        # Check fragment matches
        normalized = _normalize_prompt(prompt)
        for fragment, response in self._fragment_responses:
            if fragment in normalized or fragment in prompt:
                return GenerationResponse(
                    raw_text=response,
                    parsed_artifacts=[response],
                    model_id="mock",
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(response.split()),
                )

        if self.strict:
            raise KeyError(
                f"No mock response for prompt (hash={key}). "
                f"Add one with add_response() or add_exact_response()."
            )

        return GenerationResponse(
            raw_text="",
            parsed_artifacts=[],
            model_id="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=0,
        )

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any]
    ) -> GenerationResponse:
        """Return a pre-recorded structured response."""
        return self.generate(request)

    def load_replay(self, replay_dir: Path) -> None:
        """Load recorded interactions from a directory.

        Expected format: one JSON file per interaction with keys:
        - "prompt": the original prompt text
        - "response": the response text

        Args:
            replay_dir: Directory containing JSON files.
        """
        self.replay_dir = replay_dir
        replay_path = Path(replay_dir)
        if not replay_path.exists():
            return

        for json_file in sorted(replay_path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            self.add_exact_response(prompt, response)


# Type check
def _check_protocol() -> None:
    client: CompGenLLMProtocol = MockLLMClient()  # noqa: F841


__all__ = ["MockLLMClient"]
