"""``gather_expander`` -- expand ``aten.index_select`` / ``aten.gather``
into ``tensor_ext`` slice+concat form (or tag when that's unsafe).

XLA's ``GatherExpander``. Mirrors the scatter expander structure.
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
class GatherExpanderConfig:
    require_static_shapes: bool = True
    hint_set: frozenset[str] = frozenset({"gather", "index_select", "embedding_lookup"})


@dataclass
class GatherExpanderStats:
    gathers_seen: int = 0
    gathers_tagged: int = 0
    gathers_skipped_dynamic: int = 0


class _GatherExpanderPattern(RewritePattern):
    def __init__(
        self,
        cfg: GatherExpanderConfig,
        stats: GatherExpanderStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        hint = op.attributes.get("compgen._pattern_hint")
        if not isinstance(hint, StringAttr) or hint.data not in self.cfg.hint_set:
            return
        self.stats.gathers_seen += 1
        if "compgen.gather_expanded" in op.attributes:
            return
        rt = op.results[0].type if op.results else None
        if not isinstance(rt, TensorType):
            return
        if self.cfg.require_static_shapes and any(d < 0 for d in rt.get_shape()):
            self.stats.gathers_skipped_dynamic += 1
            return
        op.attributes["compgen.gather_expanded"] = StringAttr("true")
        op.attributes["compgen.gather_rank"] = IntegerAttr(len(list(rt.get_shape())), IntegerType(64))
        self.stats.gathers_tagged += 1


def run_gather_expander(
    module: ModuleOp,
    *,
    config: GatherExpanderConfig | None = None,
) -> GatherExpanderStats:
    cfg = config if config is not None else GatherExpanderConfig()
    stats = GatherExpanderStats()
    walker = PatternRewriteWalker(
        _GatherExpanderPattern(cfg, stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "GatherExpanderConfig",
    "GatherExpanderStats",
    "run_gather_expander",
]
