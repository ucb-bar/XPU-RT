"""Provider-agnostic LLM protocol and shared types.

This module defines the interface that all CompGen LLM clients must satisfy.
It does NOT reimplement autocomp's ``LLMClient`` -- that is used directly for
kernel search via ``compgen.kernels.autocomp_adapter``.

CompGen's protocol adds structured output, prompt context management, and
recording support that are needed for graph-level transform generation.

Invariants:
    - Every LLM call must be recordable (prompt + response + metadata).
    - Structured output must be parseable as a known schema.
    - Temperature and sampling config must be explicit, never implicit defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Objective(Enum):
    """Optimization objective for the generation pipeline."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ENERGY = "energy"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM generation call.

    Invariant: all fields must be explicit. No hidden defaults.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 1.0
    seed: int | None = None
    # TODO: Add structured output schema support (JSON mode, tool use, etc.)


@dataclass(frozen=True)
class PromptContext:
    """Context provided to the LLM alongside a generation request.

    This bundles all the information the LLM needs to generate a
    transform script, lowering policy, or other recipe artifact.
    """

    model_ir_summary: str
    """Canonical IR summary (abbreviated payload.mlir)."""

    target_profile_summary: str
    """Serialized target profile (devices, memory, constraints)."""

    available_transforms: list[str]
    """List of transform templates/operations the LLM may reference."""

    kernel_contracts: list[str]
    """Kernel contracts (layout, aliasing, cost) for ops in the IR."""

    objective: Objective
    """Optimization objective (latency, throughput, memory, energy)."""

    prior_attempts: list[str] = field(default_factory=list)
    """Summaries of prior generation attempts and their verification results."""

    hardware_feedback: str = ""
    """Profiling or diagnostic feedback from prior runs."""


@dataclass(frozen=True)
class GenerationRequest:
    """A request to generate a recipe artifact via LLM."""

    prompt_template: str
    """Jinja2 template name or raw prompt string."""

    context: PromptContext
    """Structured context for template rendering."""

    config: LLMConfig
    """LLM configuration (model, temperature, etc.)."""

    artifact_type: str = "transform_script"
    """Expected artifact type: transform_script | lowering_policy | kernel_plan | backend_glue."""


@dataclass
class GenerationResponse:
    """Response from an LLM generation call."""

    raw_text: str
    """Raw text output from the LLM."""

    parsed_artifacts: list[str]
    """Extracted artifacts (e.g., MLIR transform scripts, YAML policies)."""

    model_id: str
    """Exact model identifier used for this generation."""

    prompt_tokens: int = 0
    """Number of prompt tokens consumed."""

    completion_tokens: int = 0
    """Number of completion tokens generated."""

    latency_ms: float = 0.0
    """Wall-clock latency of the LLM call in milliseconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata (request ID, safety ratings, etc.)."""


@runtime_checkable
class CompGenLLMProtocol(Protocol):
    """Protocol that all CompGen LLM clients must satisfy.

    This is intentionally minimal. Adapters (Gemini, OpenAI, mock) implement
    this protocol. The ``LLMRecorder`` wraps any implementor.

    TODO: Add generate_structured() for JSON-schema-constrained output.
    TODO: Add batch generation for parallel candidate search.
    """

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a recipe artifact from a prompt + context.

        Args:
            request: The generation request with template, context, and config.

        Returns:
            The generation response with raw text, parsed artifacts, and metadata.

        Raises:
            NotImplementedError: Until a concrete adapter is implemented.
        """
        ...

    def generate_structured(
        self, request: GenerationRequest, schema: dict[str, Any]
    ) -> GenerationResponse:
        """Generate a structured (JSON-schema-constrained) artifact.

        Args:
            request: The generation request.
            schema: JSON schema the output must conform to.

        Returns:
            The generation response with parsed_artifacts conforming to schema.

        Raises:
            NotImplementedError: Until a concrete adapter is implemented.
        """
        ...


__all__ = [
    "CompGenLLMProtocol",
    "GenerationRequest",
    "GenerationResponse",
    "LLMConfig",
    "Objective",
    "PromptContext",
]
