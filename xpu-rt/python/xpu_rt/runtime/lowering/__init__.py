"""FX-graph → MegakernelGraph lowering — Phase 10a.

The conformance-harness path used to require a hand-built
:class:`compgen.runtime.megakernel.MegakernelGraph` per workload. This
subpackage replaces that with a pattern matcher that walks an FX
graph from ``torch.export``, recognises supported op shapes, and
emits the corresponding MegakernelGraph + device-function bodies
automatically.

Public surface:

- :class:`UnsupportedShape` — raised when no pattern matches.
- :func:`lower_torch_to_megakernel` — single entry point. Takes a
  ``torch.nn.Module`` + sample inputs, returns
  ``(megakernel_graph, device_function_sources, user_buffer_layout,
  decision_log)``.

Round-1 pattern coverage: the diamond-DAG shape ``y = (linear_a(x)
+ linear_b(x)).relu()``. Round 2+ adds the FFN portion of
decoder_layer and the gemm_rs row-sharded matmul.
"""

from __future__ import annotations

from compgen.runtime.lowering.fx_to_megakernel import (
    LoweringDecision,
    LoweringResult,
    UnsupportedShape,
    lower_torch_to_megakernel,
)

__all__ = [
    "LoweringDecision",
    "LoweringResult",
    "UnsupportedShape",
    "lower_torch_to_megakernel",
]
