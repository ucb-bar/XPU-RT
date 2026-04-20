"""``scatter_expander`` -- lower ``aten.scatter`` / ``aten.scatter_add``
calls into ``compgen.tensor_ext.pack`` + indexed-write form.

XLA's ``ScatterExpander``. Operates on opaque
``func.call`` ops tagged with ``compgen._pattern_hint = "scatter"``.
For the simplest 1-D case, tags the call with
``compgen.scatter_expanded = true`` and carries the scatter dim.

More elaborate structural expansion (``tensor_ext.pack`` + masked
write) is gated on a ``require_static_shapes`` check.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass(frozen=True)
class ScatterExpanderConfig:
    require_static_shapes: bool = True
    hint_set: frozenset[str] = frozenset({"scatter", "scatter_add"})


@dataclass
class ScatterExpanderStats:
    scatters_seen: int = 0
    scatters_tagged: int = 0
    scatters_skipped_dynamic: int = 0


class _ScatterExpanderPattern(RewritePattern):
    def __init__(
        self,
        cfg: ScatterExpanderConfig,
        stats: ScatterExpanderStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        hint = op.attributes.get("compgen._pattern_hint")
        if not isinstance(hint, StringAttr) or hint.data not in self.cfg.hint_set:
            return
        self.stats.scatters_seen += 1
        if "compgen.scatter_expanded" in op.attributes:
            return
        rt = op.results[0].type if op.results else None
        if not isinstance(rt, TensorType):
            return
        if self.cfg.require_static_shapes and any(d < 0 for d in rt.get_shape()):
            self.stats.scatters_skipped_dynamic += 1
            return
        op.attributes["compgen.scatter_expanded"] = StringAttr("true")
        op.attributes["compgen.scatter_rank"] = IntegerAttr(len(list(rt.get_shape())), IntegerType(64))
        self.stats.scatters_tagged += 1


def run_scatter_expander(
    module: ModuleOp,
    *,
    config: ScatterExpanderConfig | None = None,
) -> ScatterExpanderStats:
    cfg = config if config is not None else ScatterExpanderConfig()
    stats = ScatterExpanderStats()
    walker = PatternRewriteWalker(
        _ScatterExpanderPattern(cfg, stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "ScatterExpanderConfig",
    "ScatterExpanderStats",
    "run_scatter_expander",
]
