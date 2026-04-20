"""Shared helpers for MVP annotation passes.

Every annotation pass in this package (``decompose_concat``,
``normalize_subbyte``, and the 11 Wave-2/3 ports upgraded in the fifth
wave) follows the same shape: walk the module, match a predicate, tag
the op with an attribute, and record the count on the module itself.
This module hosts the common helpers so the individual pass files
stay small.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Operation


def annotate_matching_ops(
    module: ModuleOp,
    *,
    match: Callable[[Operation], str | None],
    attr_name: str,
    count_attr: str,
) -> int:
    """Walk the module; for each op where ``match(op)`` returns a non-``None``
    string, attach ``attr_name`` = ``StringAttr(value)``. Record the total
    count on the module as ``count_attr``. Returns the annotation count.
    """
    annotated = 0
    for op in module.walk():
        value = match(op)
        if value is None:
            continue
        op.attributes[attr_name] = StringAttr(value)
        annotated += 1
    module.attributes[count_attr] = IntegerAttr(annotated, i64)
    return annotated


def walk_ops_by_name(module: ModuleOp, names: Iterable[str]) -> Iterable[Operation]:
    """Yield every op in the module whose ``.name`` is in ``names``."""
    name_set = frozenset(names)
    for op in module.walk():
        if op.name in name_set:
            yield op


def operand_defining_op(op: Operation, index: int) -> Operation | None:
    """Return the defining op of ``op.operands[index]``, or ``None``."""
    if index >= len(op.operands):
        return None
    operand = op.operands[index]
    return getattr(operand, "owner", None)


def op_matches_any_prefix(op: Operation, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if op.name.startswith(prefix):
            return True
    return False


__all__ = [
    "annotate_matching_ops",
    "op_matches_any_prefix",
    "operand_defining_op",
    "walk_ops_by_name",
]
