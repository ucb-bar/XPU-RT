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

__all__: list[str] = []
