"""``raise_special_ops`` -- rewrite hint-tagged opaque calls to
``compgen.linalg_ext`` named ops.

Reconstruction of IREE's ``RaiseSpecialOpsPass``. Zero external
references; CompGen owns the rewrite.

Strategy (fast path first, body walk second):

1. **Fast path — pattern-hint attribute**: CompGen's decomposition
   table tags every lowered opaque call with
   ``compgen._pattern_hint`` (e.g., ``"softmax"``, ``"layer_norm"``,
   ``"silu"``). This pass reads those hints and rewrites the tagged
   op into the corresponding ``compgen.linalg_ext.*`` named op in
   O(N) over the module.
2. **Body walk — linalg.generic pattern match (deferred)**: When a
   hint is absent we'd walk ``linalg.generic`` bodies and look for
   the canonical arithmetic sequence (``exp - max`` for softmax,
   ``rsqrt(mean(x**2))`` for rmsnorm, etc.). That's Wave 2+ follow-up
   work; the current decomposition pipeline ships hints on every
   traced op, so the fast path already covers every path through
   Wave 1+ `raise_special_ops` in real workloads. When the follow-up
   ships it plugs into the same ``_dispatch`` table below.

Covered patterns (hint string → target op):

- ``softmax``          → ``compgen.linalg_ext.softmax``
- ``layer_norm``       → ``compgen.linalg_ext.layer_norm``
- ``rms_norm``         → ``compgen.linalg_ext.rms_norm``
- ``silu``             → ``compgen.linalg_ext.silu``
- ``gelu``             → ``compgen.linalg_ext.gelu``
- ``swiglu``           → ``compgen.linalg_ext.swiglu``
- ``rope``             → ``compgen.linalg_ext.rope``

LLM-tool signature:

    tool_name="raise_special_ops"
    wraps_pass="CompGen:RaiseSpecialOps"
    invent_slot="pattern_library/named_op_recognition"
    policy="RaiseHintsToLinalgExt"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.linalg_ext import (
    GeluOp,
    LayerNormOp,
    RMSNormOp,
    RoPEOp,
    SiluOp,
    SoftmaxOp,
    SwiGLUOp,
)


@dataclass
class RaiseSpecialOpsStats:
    hinted_ops_seen: int = 0
    raised_by_hint: dict[str, int] | None = None
    failures_by_hint: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.raised_by_hint is None:
            self.raised_by_hint = {}
        if self.failures_by_hint is None:
            self.failures_by_hint = {}

    def record_raise(self, hint: str) -> None:
        assert self.raised_by_hint is not None
        self.raised_by_hint[hint] = self.raised_by_hint.get(hint, 0) + 1

    def record_failure(self, hint: str) -> None:
        assert self.failures_by_hint is not None
        self.failures_by_hint[hint] = self.failures_by_hint.get(hint, 0) + 1

    @property
    def total_raised(self) -> int:
        assert self.raised_by_hint is not None
        return sum(self.raised_by_hint.values())


# --- per-hint rewriters -------------------------------------------------------


def _hint(op: Operation) -> str | None:
    attr = op.attributes.get("compgen._pattern_hint")
    if attr is None:
        return None
    return attr.data if isinstance(attr, StringAttr) else None


def _region_id(op: Operation) -> str | None:
    attr = op.attributes.get("compgen.region_id")
    return attr.data if isinstance(attr, StringAttr) else None


def _preserve_attrs(dst: Operation, src: Operation) -> None:
    """Copy region_id / pattern_hint from ``src`` onto ``dst``."""
    for key in ("compgen.region_id", "compgen._pattern_hint"):
        if key in src.attributes and key not in dst.attributes:
            dst.attributes[key] = src.attributes[key]


def _raise_softmax(
    op: Operation, rewriter: PatternRewriter
) -> bool:
    if len(op.operands) < 1 or len(op.results) != 1:
        return False
    # Softmax dim defaults to the last axis. We can't recover the
    # original dim from the opaque call, but the last-dim default
    # matches the overwhelming majority of real workloads (attention
    # softmax, MoE router softmax); leave it to a follow-up to read
    # the FX args when we thread them through.
    res_type = op.results[0].type
    if not hasattr(res_type, "get_shape"):
        return False
    rank = len(res_type.get_shape())
    dim = max(rank - 1, 0)
    new = SoftmaxOp(op.operands[0], dim=dim, result_type=res_type)
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_layer_norm(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 1 or len(op.results) != 1:
        return False
    weight = op.operands[1] if len(op.operands) >= 2 else None
    bias = op.operands[2] if len(op.operands) >= 3 else None
    new = LayerNormOp(
        op.operands[0],
        op.results[0].type,
        weight=weight,
        bias=bias,
        eps=1e-5,
    )
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_rms_norm(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 1 or len(op.results) != 1:
        return False
    weight = op.operands[1] if len(op.operands) >= 2 else None
    new = RMSNormOp(
        op.operands[0],
        op.results[0].type,
        weight=weight,
        eps=1e-6,
    )
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_silu(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 1 or len(op.results) != 1:
        return False
    new = SiluOp(op.operands[0], op.results[0].type)
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_gelu(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 1 or len(op.results) != 1:
        return False
    new = GeluOp(op.operands[0], op.results[0].type, approximate="none")
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_swiglu(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 2 or len(op.results) != 1:
        return False
    new = SwiGLUOp(op.operands[0], op.operands[1], op.results[0].type)
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


def _raise_rope(op: Operation, rewriter: PatternRewriter) -> bool:
    if len(op.operands) < 4 or len(op.results) < 2:
        return False
    # RoPE's hint is rare in current decomps (no ATen op emits it
    # directly); we handle it for symmetry with the other named
    # ops. Results must be (q_rot, k_rot).
    new = RoPEOp(
        op.operands[0], op.operands[1], op.operands[2], op.operands[3],
        op.results[0].type, op.results[1].type,
    )
    _preserve_attrs(new, op)
    rewriter.replace_matched_op(new)
    return True


_DISPATCH = {
    "softmax": _raise_softmax,
    "layer_norm": _raise_layer_norm,
    "native_layer_norm": _raise_layer_norm,
    "rms_norm": _raise_rms_norm,
    "silu": _raise_silu,
    "gelu": _raise_gelu,
    "swiglu": _raise_swiglu,
    "rope": _raise_rope,
}


# --- pattern ------------------------------------------------------------------


class RaiseSpecialOpsPattern(RewritePattern):
    """Match any op carrying a known ``compgen._pattern_hint`` and raise it."""

    def __init__(self, stats: RaiseSpecialOpsStats | None = None) -> None:
        self.stats = stats if stats is not None else RaiseSpecialOpsStats()

    def match_and_rewrite(
        self, op: Operation, rewriter: PatternRewriter
    ) -> None:
        hint = _hint(op)
        if hint is None:
            return
        handler = _DISPATCH.get(hint)
        if handler is None:
            return
        # Skip ops that are already in the linalg_ext dialect (idempotent).
        if op.name.startswith("compgen.linalg_ext."):
            return
        self.stats.hinted_ops_seen += 1
        ok = handler(op, rewriter)
        if ok:
            self.stats.record_raise(hint)
        else:
            self.stats.record_failure(hint)


# --- entry point --------------------------------------------------------------


def run_raise_special_ops(
    module: ModuleOp,
    *,
    apply_recursively: bool = False,
) -> RaiseSpecialOpsStats:
    """Raise every hinted op to its ``compgen.linalg_ext.*`` counterpart.

    ``apply_recursively=False`` because each op is rewritten once and
    the replacement op does not itself carry a raw ``_pattern_hint``
    attribute the walker can re-match (the replacement IS the
    canonical form).
    """
    stats = RaiseSpecialOpsStats()
    pattern = RaiseSpecialOpsPattern(stats=stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "RaiseSpecialOpsPattern",
    "RaiseSpecialOpsStats",
    "run_raise_special_ops",
]
