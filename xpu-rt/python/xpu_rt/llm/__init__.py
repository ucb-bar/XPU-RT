"""LLM interface subpackage for XPU-RT.

XPU-RT uses a two-layer LLM architecture:

1. **Autocomp's LLMClient** -- for kernel-level search loops (beam search,
   plan/code generation, hardware feedback). Accessed via the adapter at
   ``xpu_rt.kernels.autocomp_adapter``. Never duplicated here.

2. **XPU-RT's LLM interface** -- for graph-level transform generation,
   lowering policy synthesis, and structured recipe output. Defined here
   as ``CompGenLLMProtocol`` with adapters for Gemini (primary), OpenAI,
   Anthropic, and a deterministic mock for testing.

All LLM interactions pass through the ``LLMRecorder`` middleware for
reproducibility and audit.
"""

from __future__ import annotations

from xpu_rt.llm.anthropic_client import AnthropicClient
from xpu_rt.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    GenerationResponse,
    LLMConfig,
    Objective,
    PromptContext,
)
from xpu_rt.llm.cli_client import ClaudeCLIClient, CodexCLIClient
from xpu_rt.llm.config import (
    SUPPORTED_PROVIDERS,
    LLMSelection,
    apply_selection_to_env,
    build_llm_runtime,
    resolve_llm_selection,
    selection_status,
)
from xpu_rt.llm.factory import create_llm_client
from xpu_rt.llm.gemini_client import GeminiClient
from xpu_rt.llm.mock_client import MockLLMClient
from xpu_rt.llm.openai_client import OpenAIClient
from xpu_rt.llm.recorder import LLMRecorder, ToolCallRecord, ToolCallRecorder
from xpu_rt.llm.registry import (
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
