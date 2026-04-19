"""Ported xDSL-level payload passes (P7 + P8 from the patch plan).

Importing this package triggers registration of every ported pass as a
typed :class:`compgen.llm.registry.Tool`. Real ports set ``stub=False``;
scaffolded stubs set ``stub=True`` and run as identity rewrites until
fully implemented.

See `user_perspective/analysis/iree_global_optimization_audit.md` and
`xla_optimization_audit.md` for per-pass rationale.
"""

from __future__ import annotations

from compgen.ir.payload.passes.base import PayloadPass

# IREE ports (real)
from compgen.ir.payload.passes.decompose_concat import DecomposeConcat
from compgen.ir.payload.passes.demote_contraction_inputs import DemoteContractionInputs

# XLA ports (real)
from compgen.ir.payload.passes.normalize_subbyte import NormalizeSubByte

# Scaffolded stubs (real impl pending)
from compgen.ir.payload.passes.stubs import (
    FoldTransposesIntoDots,
    FuseDequantMatmul,
    FuseSoftmaxToTriton,
    LowerConvToImg2Col,
    LowerQuantizedConv,
    LowerQuantizedMatmul,
    MatchLibraryCall,
    PlanReduction,
    PropagateTransposes,
    RaiseSpecialOps,
    SetNumericsPolicy,
)

# Phase 5 runtime stubs (P15)
from compgen.ir.payload.passes.runtime_stubs import (
    AliasIoBuffers,
    AssignMemorySpace,
    AssignQueue,
    AssignStreams,
    InsertCopies,
    InsertHostOffload,
    NormalizeSubBytePostLayout,
    PlanBuffers,
)

# Registration (idempotent)
_ALL_PASSES: list[PayloadPass] = [
    # Real (MVP annotation passes)
    DecomposeConcat(),
    DemoteContractionInputs(),
    NormalizeSubByte(),
    # Phase 2/3 stubs
    FoldTransposesIntoDots(),
    FuseDequantMatmul(),
    FuseSoftmaxToTriton(),
    LowerConvToImg2Col(),
    LowerQuantizedConv(),
    LowerQuantizedMatmul(),
    MatchLibraryCall(),
    PlanReduction(),
    PropagateTransposes(),
    RaiseSpecialOps(),
    SetNumericsPolicy(),
    # Phase 5 runtime stubs (P15)
    AliasIoBuffers(),
    AssignMemorySpace(),
    AssignQueue(),
    AssignStreams(),
    InsertCopies(),
    InsertHostOffload(),
    NormalizeSubBytePostLayout(),
    PlanBuffers(),
]


def register_all() -> None:
    """Register every pass into the global LLM registry (idempotent)."""
    for p in _ALL_PASSES:
        p.register()


# Auto-register on import.
register_all()


__all__ = [
    "AliasIoBuffers",
    "AssignMemorySpace",
    "AssignQueue",
    "AssignStreams",
    "DecomposeConcat",
    "DemoteContractionInputs",
    "FoldTransposesIntoDots",
    "FuseDequantMatmul",
    "FuseSoftmaxToTriton",
    "InsertCopies",
    "InsertHostOffload",
    "LowerConvToImg2Col",
    "LowerQuantizedConv",
    "LowerQuantizedMatmul",
    "MatchLibraryCall",
    "NormalizeSubByte",
    "NormalizeSubBytePostLayout",
    "PayloadPass",
    "PlanBuffers",
    "PlanReduction",
    "PropagateTransposes",
    "RaiseSpecialOps",
    "SetNumericsPolicy",
    "register_all",
]
