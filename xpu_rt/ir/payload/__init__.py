"""Payload IR -- the canonical computational IR layer.

Layer 1 of the three-layer IR stack. Payload IR represents the model's
computational structure after import from PyTorch FX and canonicalization.

This is the "compiler IR" -- optimized for progressive lowering, typed
transforms, and verification. The LLM does NOT edit Payload IR directly;
it edits Recipe IR (Layer 2), which lowers into transforms over Payload IR.

Modules:
    import_fx       -- FX graph → xDSL conversion
    canonicalize    -- canonical form enforcement
    contracts       -- kernel/op contracts (layouts, dtypes, costs)
"""

from __future__ import annotations

__all__: list[str] = []
