"""Shape + rank utilities for Wave 1+ pattern rewrites.

xDSL's ``TensorType`` distinguishes static dims (positive ints) from
dynamic dims (``-1``). Many rewrites need a bounded "is this shape
static" check or a fused "infer result shape from operands + op
semantics" helper. Implementing both once here is better than letting
every pass re-roll the same logic.

All helpers accept raw ``TensorType``s or xDSL ``SSAValue``s (in
which case they inspect ``value.type``).
"""

from __future__ import annotations

from collections.abc import Iterable

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Attribute, SSAValue

_DYNAMIC_DIM = -1


def _get_tensor_type(source: TensorType | SSAValue | Attribute) -> TensorType | None:
    """Return ``source.type`` when it's a ``TensorType``; else ``None``."""
    if isinstance(source, TensorType):
        return source
    if isinstance(source, SSAValue):
        t = source.type
        return t if isinstance(t, TensorType) else None
    if isinstance(source, Attribute):
        return source if isinstance(source, TensorType) else None
    return None


def static_shape_or_none(
    source: TensorType | SSAValue | Attribute,
) -> tuple[int, ...] | None:
    """Return the fully-static shape tuple, or ``None`` if any dim is dynamic.

    Use this as a gate at the top of pattern rewrites that can't
    correctly handle dynamic extents (e.g. ``lower_conv_to_img2col``).
    """
    t = _get_tensor_type(source)
    if t is None:
        return None
    shape = list(t.get_shape())
    if any(d == _DYNAMIC_DIM for d in shape):
        return None
    return tuple(shape)


def is_static_shape(source: TensorType | SSAValue | Attribute) -> bool:
    """Convenience boolean wrapper around :func:`static_shape_or_none`."""
    return static_shape_or_none(source) is not None


def rank_of(source: TensorType | SSAValue | Attribute) -> int | None:
    """Return the rank, or ``None`` if ``source`` is not a tensor."""
    t = _get_tensor_type(source)
    if t is None:
        return None
    return len(list(t.get_shape()))


def infer_result_shape(
    op_kind: str,
    operand_shapes: list[tuple[int, ...]],
    *,
    axis: int | None = None,
    reduction_dims: list[int] | None = None,
    perm: list[int] | None = None,
) -> tuple[int, ...] | None:
    """Infer the result shape of a handful of canonical ops.

    This is the small subset needed by Wave 1-4 rewrites:

    - ``elementwise`` -- all operand shapes must match; result is
      the first.
    - ``matmul`` -- standard 2D matmul shape: ``[M, K]`` x
      ``[K, N]`` -> ``[M, N]``.
    - ``concat`` -- shapes must agree on every non-``axis`` dim; the
      ``axis`` extent is the sum (or ``-1`` if any input is dynamic).
    - ``reduction`` -- drops ``reduction_dims`` from the first
      operand's shape.
    - ``transpose`` -- permutes the first operand's shape by ``perm``.

    Returns ``None`` if the op is unrecognised or the inputs are
    inconsistent (e.g. mismatched matmul K, missing ``axis`` for
    concat).
    """
    if not operand_shapes:
        return None

    if op_kind == "elementwise":
        first = operand_shapes[0]
        for s in operand_shapes[1:]:
            if s != first:
                return None
        return first

    if op_kind == "matmul":
        if len(operand_shapes) != 2:
            return None
        lhs, rhs = operand_shapes
        if len(lhs) != 2 or len(rhs) != 2:
            return None
        m, k_lhs = lhs
        k_rhs, n = rhs
        if k_lhs != k_rhs and _DYNAMIC_DIM not in (k_lhs, k_rhs):
            return None
        return (m, n)

    if op_kind == "concat":
        if axis is None:
            return None
        first = operand_shapes[0]
        rank = len(first)
        if axis < 0 or axis >= rank:
            return None
        total = 0
        for s in operand_shapes:
            if len(s) != rank:
                return None
            for i, (a, b) in enumerate(zip(first, s, strict=True)):
                if i == axis:
                    continue
                if a != b and _DYNAMIC_DIM not in (a, b):
                    return None
            if s[axis] == _DYNAMIC_DIM or total == _DYNAMIC_DIM:
                total = _DYNAMIC_DIM
            else:
                total += s[axis]
        return tuple(total if i == axis else first[i] for i in range(rank))

    if op_kind == "reduction":
        if reduction_dims is None:
            return None
        shape = list(operand_shapes[0])
        rank = len(shape)
        for d in reduction_dims:
            if d < 0 or d >= rank:
                return None
        kept = [shape[i] for i in range(rank) if i not in reduction_dims]
        return tuple(kept)

    if op_kind == "transpose":
        if perm is None:
            return None
        shape = list(operand_shapes[0])
        rank = len(shape)
        if sorted(perm) != list(range(rank)):
            return None
        return tuple(shape[p] for p in perm)

    return None


def common_element_type(
    sources: Iterable[TensorType | SSAValue | Attribute],
) -> Attribute | None:
    """Return the element type when all sources agree, else ``None``."""
    first: Attribute | None = None
    for s in sources:
        t = _get_tensor_type(s)
        if t is None:
            return None
        elem = t.get_element_type()
        if first is None:
            first = elem
        elif elem != first:
            return None
    return first


__all__ = [
    "common_element_type",
    "infer_result_shape",
    "is_static_shape",
    "rank_of",
    "static_shape_or_none",
]
