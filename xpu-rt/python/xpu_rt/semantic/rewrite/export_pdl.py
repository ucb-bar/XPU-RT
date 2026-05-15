"""Export eqsat rules and RewritePatterns to PDL for verification.

Pure DAG-to-DAG rewrites on side-effect-free arith ops can be exported
to the callable verification path. More complex rewrites (with memory,
control flow, or side effects) cannot be exported and return None.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

log = structlog.get_logger()


def eqsat_rule_to_pdl(
    rule: Any,
    module: ModuleOp | None = None,
) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Export a pure eqsat rule to callable pattern/replacement pair.

    Returns a (pattern_fn, replacement_fn) tuple suitable for
    ``PDLVerificationBackend.verify_arith_rewrite()``, or None if the
    rule is not exportable.

    Args:
        rule: An ``EqSatRewriteRule`` instance.
        module: Optional module context for type inference.

    Returns:
        (pattern_fn, replacement_fn) callables or None.
    """
    name = getattr(rule, "name", "unknown")

    # Known exportable patterns
    exportable = _KNOWN_RULE_EXPORTS.get(name)
    if exportable is not None:
        log.debug("rewrite.export_pdl.known", rule=name)
        return exportable

    log.debug("rewrite.export_pdl.not_exportable", rule=name)
    return None


def pass_pattern_to_pdl(pattern_source: str) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Export a RewritePattern source to callable pair for verification.

    Currently returns None — full RewritePattern-to-PDL export requires
    static analysis of the pattern's match_and_rewrite method.

    Args:
        pattern_source: Python source code of the RewritePattern.

    Returns:
        (pattern_fn, replacement_fn) or None.
    """
    return None


# ---- Known exportable rules ----
# These are hand-written exports for CompGen's built-in eqsat rules.
# LLM-generated rules go through the callable verification path directly.


def _commutativity_addi_pattern(operands: list[Any]) -> Any:
    """add(a, b) pattern."""
    return operands[0] + operands[1]


def _commutativity_addi_replacement(operands: list[Any]) -> Any:
    """add(b, a) replacement — same by commutativity."""
    return operands[1] + operands[0]


def _commutativity_muli_pattern(operands: list[Any]) -> Any:
    """mul(a, b) pattern."""
    return operands[0] * operands[1]


def _commutativity_muli_replacement(operands: list[Any]) -> Any:
    """mul(b, a) replacement — same by commutativity."""
    return operands[1] * operands[0]


_KNOWN_RULE_EXPORTS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "commutativity_addi": (_commutativity_addi_pattern, _commutativity_addi_replacement),
    "commutativity_muli": (_commutativity_muli_pattern, _commutativity_muli_replacement),
}


__all__ = ["eqsat_rule_to_pdl", "pass_pattern_to_pdl"]
