"""Per-dim semantics: tag every dim of every op as parallel / reduce /
broadcast / batch.

Why it matters:
  * The tile oracle picks BLOCK_K only along dims marked ``reduce``.
  * The fusion oracle decides whether two ops share parallel-dim
    structure (cheap to fuse) vs need transposes (expensive).
  * The kernel codegen prompt names dims by role rather than position,
    which makes Claude Code's emitted kernel correct-by-construction.

Stored as ``compgen.dim_role`` IR attribute — an ArrayAttr of role
strings, one per output-tensor dim. The agent reads this via
``dim_roles_for_op(op)`` without re-traversing the dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation


class DimRole(Enum):
    """One per output-tensor dim of an op."""

    PARALLEL = "parallel"  # output dim corresponds to a parallel axis (no reduction)
    REDUCE = "reduce"  # this dim is reduced over (matmul K, softmax axis)
    BROADCAST = "broadcast"  # this dim is broadcast (size 1, expanded by consumer)
    BATCH = "batch"  # outer batch dim — parallel but special (scheduling unit)
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OpDimAnnotation:
    """Roles for one op's output-tensor dims."""

    op_name: str
    output_roles: tuple[DimRole, ...]
    # Reduced-input dims (input_idx, input_dim) pairs — useful for tile_oracle
    reduce_axes: tuple[tuple[int, int], ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-op-family analyzers
# ---------------------------------------------------------------------------


def _shape_of_first_result(op: Operation) -> tuple[int, ...]:
    if not op.results:
        return ()
    t = op.results[0].type
    if not hasattr(t, "get_shape"):
        return ()
    return tuple(t.get_shape())


def _analyze_matmul(op: Operation) -> OpDimAnnotation:
    """linalg.matmul (M,K) × (K,N) → (M,N): output dims are both PARALLEL,
    K is the reduced axis (input 0 dim 1, input 1 dim 0)."""
    shape = _shape_of_first_result(op)
    rank = len(shape)
    if rank == 2:
        return OpDimAnnotation(
            op_name=op.name,
            output_roles=(DimRole.PARALLEL, DimRole.PARALLEL),
            reduce_axes=((0, 1), (1, 0)),
            notes="matmul: M, N parallel; K reduced",
        )
    if rank == 3:
        return OpDimAnnotation(
            op_name=op.name,
            output_roles=(DimRole.BATCH, DimRole.PARALLEL, DimRole.PARALLEL),
            reduce_axes=((0, 2), (1, 1)),
            notes="batch_matmul: B batch, M/N parallel, K reduced",
        )
    return OpDimAnnotation(
        op_name=op.name,
        output_roles=tuple(DimRole.UNKNOWN for _ in range(rank)),
    )


def _analyze_reduce(op: Operation, *, reduced_axis: int = -1) -> OpDimAnnotation:
    """Reductions (softmax, mean, sum) along a known axis."""
    shape = _shape_of_first_result(op)
    rank = len(shape)
    roles: list[DimRole] = []
    axis = reduced_axis if reduced_axis >= 0 else rank + reduced_axis
    for i in range(rank):
        roles.append(DimRole.REDUCE if i == axis else DimRole.PARALLEL)
    return OpDimAnnotation(
        op_name=op.name,
        output_roles=tuple(roles),
        reduce_axes=((0, axis),),
        notes=f"reduction axis={axis}",
    )


def _analyze_pointwise(op: Operation) -> OpDimAnnotation:
    """Elementwise ops: every dim is parallel."""
    shape = _shape_of_first_result(op)
    rank = len(shape)
    return OpDimAnnotation(
        op_name=op.name,
        output_roles=tuple(DimRole.PARALLEL for _ in range(rank)),
        notes="pointwise — all dims parallel",
    )


# Op-family → analyzer dispatch.
_ANALYZERS = {
    "linalg.matmul": _analyze_matmul,
    "linalg.batch_matmul": lambda op: _analyze_matmul(op),  # rank-3 path
    "softmax": lambda op: _analyze_reduce(op, reduced_axis=-1),
    "rmsnorm": lambda op: _analyze_reduce(op, reduced_axis=-1),
    "reduce_mean": lambda op: _analyze_reduce(op, reduced_axis=-1),
    "rsqrt": _analyze_pointwise,
    "silu": _analyze_pointwise,
    "sigmoid": _analyze_pointwise,
    "tanh": _analyze_pointwise,
    "arith.mulf": _analyze_pointwise,
    "arith.addf": _analyze_pointwise,
    "arith.subf": _analyze_pointwise,
    "arith.divf": _analyze_pointwise,
    "neg": _analyze_pointwise,
    "where": _analyze_pointwise,
}


def _hint_or_name(op: Operation) -> str:
    """Prefer the compgen._pattern_hint when present (tracks the op family
    even when the op is wrapped in a func.call)."""
    attrs = getattr(op, "attributes", {})
    hint = attrs.get("compgen._pattern_hint") if attrs else None
    if hint is not None and hasattr(hint, "data"):
        return hint.data
    return op.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_op(op: Operation) -> OpDimAnnotation | None:
    """Return the dim annotation for ``op``, or None when the op is
    structural (func.func / return / module)."""
    if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
        return None
    if not op.results:
        return None
    name = _hint_or_name(op)
    fn = _ANALYZERS.get(name)
    if fn is None:
        # Default: pointwise unless we know otherwise. Safer to assume
        # parallel than to leave UNKNOWN, since most decomposed ops are.
        return _analyze_pointwise(op)
    return fn(op)


def annotate_dim_roles(module: ModuleOp) -> int:
    """Walk ``module`` and stamp ``compgen.dim_role`` on every op's
    attributes. Returns number of ops annotated.
    """
    annotated = 0
    for op in module.walk():
        ann = analyze_op(op)
        if ann is None:
            continue
        op.attributes["compgen.dim_role"] = ArrayAttr([StringAttr(r.value) for r in ann.output_roles])
        annotated += 1
    return annotated


def dim_roles_for_op(op: Operation) -> tuple[DimRole, ...]:
    """Read back the dim roles previously stamped on ``op``.

    Returns empty tuple when no annotation has been written.
    """
    attrs = getattr(op, "attributes", {})
    arr = attrs.get("compgen.dim_role") if attrs else None
    if arr is None or not hasattr(arr, "data"):
        return ()
    out: list[DimRole] = []
    for el in arr.data:
        try:
            out.append(DimRole(el.data))
        except ValueError:
            out.append(DimRole.UNKNOWN)
    return tuple(out)


__all__ = [
    "DimRole",
    "OpDimAnnotation",
    "analyze_op",
    "annotate_dim_roles",
    "dim_roles_for_op",
]
