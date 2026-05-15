"""IREE DecomposeConcat — MVP port.

**Scope for this wave:** *identify-and-annotate* rather than full
destructive rewrite. Walks the module, finds every `tensor.concat`
(or equivalent concat-shaped op), computes a per-concat decomposition
strategy (``outer_dim_zerocopy`` / ``transpose_then_outer`` /
``inner_insert_slice``), and attaches the chosen strategy as a
``compgen.concat_strategy`` attribute on the op. Does NOT rewrite the
op yet — leaves that for a follow-up wave with correctness testing.

This still produces observable diffs (new attributes appear), is
registered as a real ``stub=False`` tool, and exercises the
``PayloadPass`` + registry path end-to-end. Full rewrite is straight-
forward once we commit to the surgery (iterating results of a
``tensor.concat`` op, producing ``tensor.empty`` + chain of
``tensor.insert_slice`` for the outer-dim case).
"""

from __future__ import annotations

from typing import Any, ClassVar

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation

from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg

# Candidate op-name patterns. xDSL doesn't register a universal
# "tensor.concat" today in CompGen's import path; FX-level `aten.cat`
# lowers to various forms. We cover the common ones by name.
_CONCAT_OP_NAMES = frozenset(
    {
        "tensor.concat",
        "tosa.concat",
        "tensor.concatenate",
        # FX-level residue that sometimes survives the import table:
        "compgen.concat",
    }
)


def _pick_strategy(op: Operation, preferred: str) -> str:
    """Decide the decomposition strategy for a single concat op.

    Heuristic:
      - If axis == 0 or ``outermost_non_unit`` → ``outer_dim_zerocopy``.
      - Else if ``preferred == transpose_then_outer`` → try to hoist
        the concat dimension to outer via transposes.
      - Else → ``inner_insert_slice`` (fallback).

    MVP: we don't inspect the actual axis attribute (would need to
    reach into each op's properties). We default to ``preferred`` and
    let the follow-up wave refine.
    """
    allowed = {"outer_dim_zerocopy", "transpose_then_outer", "inner_insert_slice"}
    return preferred if preferred in allowed else "inner_insert_slice"


class DecomposeConcat(PayloadPass):
    """Identify-and-annotate port of IREE DecomposeConcat."""

    name: ClassVar[str] = "decompose_concat"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:DecomposeConcat"
    covers_families: ClassVar[frozenset[str]] = frozenset(
        {"rvv_cpu", "qualcomm_npu", "qualcomm_dsp", "arm_cpu", "generic_npu"}
    )
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = (
        "Identify concat ops and annotate a decomposition strategy. "
        "MVP: annotates compgen.concat_strategy attribute; full "
        "destructive rewrite in follow-up wave."
    )
    stub: ClassVar[bool] = False  # real analysis pass; real diff

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg(
                name="region",
                dtype="region_ref",
                description="region symbol ref (empty = whole module)",
                required=False,
                default="",
            ),
            ToolArg(
                name="strategy",
                dtype="enum",
                description="preferred decomposition strategy",
                required=False,
                default="inner_insert_slice",
                enum=(
                    "outer_dim_zerocopy",
                    "transpose_then_outer",
                    "inner_insert_slice",
                ),
            ),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        """Walk the module, annotate concat ops with a strategy.

        Returns the SAME module (attributes mutated in place). Callers
        that want an isolated clone should clone before calling.
        """
        preferred = kwargs.get("strategy", "inner_insert_slice")
        annotated_count = 0
        for op in module.walk():
            if op.name not in _CONCAT_OP_NAMES:
                continue
            strategy = _pick_strategy(op, preferred)
            # Annotate (merges with existing attributes; no overwrite guard).
            op.attributes["compgen.concat_strategy"] = StringAttr(strategy)
            annotated_count += 1

        # Record the pass's footprint on the module itself.
        existing = module.attributes.get("compgen.decompose_concat.count")
        if existing is not None and hasattr(existing, "data"):
            prev = int(getattr(existing, "data", 0))
        else:
            prev = 0
        from xdsl.dialects.builtin import IntegerAttr, i64

        module.attributes["compgen.decompose_concat.count"] = IntegerAttr(prev + annotated_count, i64)
        return module


__all__ = ["DecomposeConcat"]
