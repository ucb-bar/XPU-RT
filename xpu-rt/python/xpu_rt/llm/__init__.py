"""LLM interface subpackage for CompGen.

CompGen uses a two-layer LLM architecture:

1. **Autocomp's LLMClient** -- for kernel-level search loops (beam search,
   plan/code generation, hardware feedback). Accessed via the adapter at
   ``compgen.kernels.autocomp_adapter``. Never duplicated here.

2. **CompGen's LLM interface** -- for graph-level transform generation,
   lowering policy synthesis, and structured recipe output. Defined here
   as ``CompGenLLMProtocol`` with adapters for Gemini (primary), OpenAI,
   Anthropic, and a deterministic mock for testing.

All LLM interactions pass through the ``LLMRecorder`` middleware for
reproducibility and audit.
"""

from __future__ import annotations

from compgen.llm.anthropic_client import AnthropicClient
from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
    LLMConfig,
    Objective,
    PromptContext,
)
from compgen.llm.cli_client import ClaudeCLIClient, CodexCLIClient
from compgen.llm.config import (
    LLMSelection,
    SUPPORTED_PROVIDERS,
    apply_selection_to_env,
    build_llm_runtime,
    resolve_llm_selection,
    selection_status,
)
from compgen.llm.factory import create_llm_client
from compgen.llm.gemini_client import GeminiClient
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.openai_client import OpenAIClient
from compgen.llm.recorder import LLMRecorder, ToolCallRecord, ToolCallRecorder
from compgen.llm.registry import (
    InventSlot,
    Registry,
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)

__all__ = [
    "AnthropicClient",
    "ClaudeCLIClient",
    "CodexCLIClient",
    "CompGenLLMProtocol",
    "GenerationRequest",
    "GenerationResponse",
    "GeminiClient",
    "InventSlot",
    "LLMSelection",
    "LLMConfig",
    "LLMRecorder",
    "MockLLMClient",
    "Objective",
    "OpenAIClient",
    "PromptContext",
    "Registry",
    "SUPPORTED_PROVIDERS",
    "Tool",
    "ToolArg",
    "ToolCallRecord",
    "ToolCallRecorder",
    "ToolResult",
    "apply_selection_to_env",
    "build_llm_runtime",
    "create_llm_client",
    "get_registry",
    "resolve_llm_selection",
    "selection_status",
]
