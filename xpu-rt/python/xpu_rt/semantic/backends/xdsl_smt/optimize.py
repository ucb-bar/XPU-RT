"""Semantic IR optimization before SMT solving.

When using the xDSL SMT dialect pipeline (rather than Z3 API directly),
this module provides optimization passes that reduce solver time by
simplifying the SMT query before sending it to Z3.

The paper reported 24.6% and 59.5% solver-time reductions from this
optimization pipeline on two translation-validation workloads.

Pipeline: canonicalize → CSE → DCE → canonicalize.
"""

from __future__ import annotations

import structlog
from xdsl.context import Context as MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.dead_code_elimination import DeadCodeElimination

log = structlog.get_logger()


def optimize_smt_module(ctx: MLContext, module: ModuleOp) -> int:
    """Optimize an SMT-dialect module to reduce solver time.

    Applies canonicalization, CSE, and DCE in sequence. This mirrors
    the ``--opt`` flag from the xdsl-tv tool.

    Args:
        ctx: MLContext with all required dialects registered.
        module: The SMT-dialect module to optimize (modified in place).

    Returns:
        Number of ops removed (approximate, based on before/after count).
    """
    before_count = sum(1 for _ in module.walk())

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    DeadCodeElimination().apply(ctx, module)
    CanonicalizePass().apply(ctx, module)

    after_count = sum(1 for _ in module.walk())
    removed = before_count - after_count

    log.debug(
        "smt.optimize",
        before=before_count,
        after=after_count,
        removed=removed,
    )
    return removed


__all__ = ["optimize_smt_module"]
