"""``compgen.linalg_ext`` -- named structured ops for high-level patterns.

xDSL's ``linalg`` dialect lacks named ops for common LLM / vision
patterns (softmax, layernorm, rmsnorm, rope, swiglu, gelu, silu). The
Wave 2 ``raise_special_ops`` pass detects these patterns inside
``linalg.generic`` bodies and rewrites them into these named ops, so
downstream passes (kernel dispatch, library-call matching, Triton
lowering) can reason about them structurally instead of re-parsing
arithmetic every time.

Register with::

    ctx.register_dialect("compgen.linalg_ext", lambda: LinalgExt)
"""

from __future__ import annotations

from compgen.ir.linalg_ext.dialect import ALL_ATTRS, ALL_OPS, LinalgExt
from compgen.ir.linalg_ext.ops import (
    GeluOp,
    LayerNormOp,
    RMSNormOp,
    RoPEOp,
    SiluOp,
    SoftmaxOp,
    SwiGLUOp,
)

__all__ = [
    "ALL_ATTRS",
    "ALL_OPS",
    "GeluOp",
    "LayerNormOp",
    "LinalgExt",
    "RMSNormOp",
    "RoPEOp",
    "SiluOp",
    "SoftmaxOp",
    "SwiGLUOp",
]
