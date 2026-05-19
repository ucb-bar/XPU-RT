"""Declarative constraint evaluator for ukernel matching.

Evaluates constraint strings from ``UkernelMatchOp`` against a
``ConstraintContext``. No ``eval()``, no code execution — pure string
parsing with ``re``.

Supported constraint syntax:
    - Shape predicates: ``M%16==0``, ``K>=32``, ``N<=4096``
    - Feature predicates: ``has_tensor_core``, ``has_rvv``, ``has_npu_engine``
    - Device predicates: ``device_type==gpu``, ``device_type==accelerator``
    - Dtype predicates: ``dtype==float32``, ``dtype_in(float16,bfloat16)``
    - Layout predicates: ``lhs_rowmajor``, ``rhs_prepacked``
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

# Regex patterns for constraint parsing
_SHAPE_MOD_RE = re.compile(r"^([A-Z_]\w*)%(\d+)==(\d+)$")  # M%16==0
_SHAPE_CMP_RE = re.compile(r"^([A-Z_]\w*)(>=|<=|==|!=|>|<)(\d+)$")  # K>=32
_DEVICE_RE = re.compile(r"^device_type==(\w+)$")  # device_type==gpu
_DTYPE_EQ_RE = re.compile(r"^dtype==(\w+)$")  # dtype==float32
_DTYPE_IN_RE = re.compile(r"^dtype_in\(([^)]+)\)$")  # dtype_in(float16,bfloat16)
_LAYOUT_RE = re.compile(r"^(lhs|rhs|out)_(\w+)$")  # lhs_rowmajor, rhs_prepacked


@dataclass(frozen=True)
class ConstraintContext:
    """Context for evaluating ukernel match constraints.

    Attributes:
        shapes: Dimension values (M, N, K, etc.).
        dtypes: Active dtypes in the computation.
        target_features: Target capability features (e.g., "has_tensor_core", "has_rvv").
        device_type: Device type string ("gpu", "cpu", "accelerator", "npu").
        layouts: Operand layouts ("lhs" -> "rowmajor", "rhs" -> "prepacked").
    """

    shapes: dict[str, int] = field(default_factory=dict)
    dtypes: tuple[str, ...] = ()
    target_features: frozenset[str] = frozenset()
    device_type: str = ""
    layouts: dict[str, str] = field(default_factory=dict)


def evaluate_constraint(constraint: str, context: ConstraintContext) -> bool:
    """Evaluate a single constraint string against a context.

    Args:
        constraint: Declarative constraint string.
        context: Evaluation context with shapes, dtypes, features, etc.

    Returns:
        True if the constraint is satisfied, False otherwise.
        Unknown constraint formats return False (safe default).
    """
    constraint = constraint.strip()
    if not constraint:
        return True

    # Shape modulo: M%16==0
    m = _SHAPE_MOD_RE.match(constraint)
    if m:
        dim_name, divisor, remainder = m.group(1), int(m.group(2)), int(m.group(3))
        val = context.shapes.get(dim_name)
        if val is None:
            return False
        return val % divisor == remainder

    # Shape comparison: K>=32, N<=4096, M==128
    m = _SHAPE_CMP_RE.match(constraint)
    if m:
        dim_name, op, threshold = m.group(1), m.group(2), int(m.group(3))
        val = context.shapes.get(dim_name)
        if val is None:
            return False
        ops = {
            ">=": val >= threshold,
            "<=": val <= threshold,
            "==": val == threshold,
            "!=": val != threshold,
            ">": val > threshold,
            "<": val < threshold,
        }
        return ops.get(op, False)

    # Device type: device_type==gpu
    m = _DEVICE_RE.match(constraint)
    if m:
        return context.device_type == m.group(1)

    # Dtype equality: dtype==float32
    m = _DTYPE_EQ_RE.match(constraint)
    if m:
        return m.group(1) in context.dtypes

    # Dtype set: dtype_in(float16,bfloat16)
    m = _DTYPE_IN_RE.match(constraint)
    if m:
        allowed = {s.strip() for s in m.group(1).split(",")}
        return bool(set(context.dtypes) & allowed)

    # Layout predicate: lhs_rowmajor, rhs_prepacked
    m = _LAYOUT_RE.match(constraint)
    if m:
        operand, layout = m.group(1), m.group(2)
        return context.layouts.get(operand) == layout

    # Feature predicate: has_tensor_core, has_rvv, has_npu_engine
    if constraint.startswith("has_") or constraint.startswith("supports_"):
        return constraint in context.target_features

    log.debug("ukernel.constraint.unknown", constraint=constraint)
    return False


def evaluate_all_constraints(
    constraints: Iterable[str],
    context: ConstraintContext,
) -> bool:
    """Evaluate all constraints — all must pass.

    Args:
        constraints: Iterable of constraint strings.
        context: Evaluation context.

    Returns:
        True if ALL constraints pass, False if any fails.
        Empty constraints = always True.
    """
    return all(evaluate_constraint(c, context) for c in constraints)


__all__ = ["ConstraintContext", "evaluate_all_constraints", "evaluate_constraint"]
