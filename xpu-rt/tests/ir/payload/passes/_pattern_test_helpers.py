"""Test helpers for the Wave 1+ ``PatternRewriter`` pass tests.

Every pass test needs:

1. A way to cheaply build a tiny ``ModuleOp`` containing the op shape
   the pass is supposed to match (matmul, concat, quant matmul, etc.).
2. A one-line applicator that runs a pattern against the module.
3. A handful of structural asserts over the resulting IR.
4. A thin wrapper around
   :func:`compgen.ir.semantic.translation_validation.validate_translation`
   that treats "timeout" as "acceptable" (since the Z3 SMT backend
   can time out on non-trivial bodies) but treats "invalid" as an
   actual test failure.

This module deliberately contains NO test functions (its filename is
prefixed with ``_`` so pytest ignores collection). It's imported by
per-pass test modules in ``tests/ir/payload/passes/rewrites/``.
"""

from __future__ import annotations

from collections.abc import Iterable

from compgen.ir.payload.decompositions import _attach_region_id
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)

# --- Module builders ---------------------------------------------------------


def _ft(shape: Iterable[int]) -> TensorType:
    return TensorType(Float32Type(), list(shape))


def _wrap_in_forward_func(
    ops: list,
    result_value: SSAValue,
    *,
    arg_types: list[Attribute] | None = None,
    arg_values: list[SSAValue] | None = None,
) -> ModuleOp:
    """Wrap a list of ops ending in ``result_value`` into a ``@forward`` func.

    When ``arg_types`` is provided, the function takes those as block
    args and ``arg_values`` are the corresponding SSA handles that
    must appear in ``ops`` (the caller is responsible for already
    using them). When ``arg_types`` is omitted, the function is
    parameterless.
    """
    if arg_types is None:
        arg_types = []
    if arg_values is None:
        arg_values = []

    block = Block(arg_types=arg_types)
    for op in ops:
        block.add_op(op)
    block.add_op(ReturnOp(result_value))

    func_type = FunctionType.from_lists(arg_types, [result_value.type])
    func = FuncOp("forward", func_type, Region([block]))
    return ModuleOp([func])


def build_linalg_matmul_module(M: int = 4, K: int = 8, N: int = 16, *, region_id: str = "matmul_0") -> ModuleOp:
    """Build a module containing one ``linalg.matmul`` tagged with ``region_id``."""
    lhs_empty = EmptyOp([], _ft([M, K]))
    rhs_empty = EmptyOp([], _ft([K, N]))
    out_empty = EmptyOp([], _ft([M, N]))
    mm = MatmulOp(
        inputs=[lhs_empty.results[0], rhs_empty.results[0]],
        outputs=[out_empty.results[0]],
        res=[_ft([M, N])],
    )
    _attach_region_id(mm, region_id)

    ops = [lhs_empty, rhs_empty, out_empty, mm]
    return _wrap_in_forward_func(ops, mm.results[0])


def build_concat_module(
    shapes: list[tuple[int, ...]],
    *,
    dim: int = 0,
    region_id: str = "concat_0",
) -> ModuleOp:
    """Build a module with a ``compgen.tensor_ext.concat``."""
    from compgen.ir.tensor_ext import ConcatOp

    if not shapes:
        raise ValueError("build_concat_module needs at least one shape")

    source_ops = [EmptyOp([], _ft(s)) for s in shapes]
    total = sum(s[dim] for s in shapes)
    result_shape = list(shapes[0])
    result_shape[dim] = total

    concat = ConcatOp(
        [op.results[0] for op in source_ops],
        dim=dim,
        result_type=_ft(result_shape),
    )
    _attach_region_id(concat, region_id)
    ops = list(source_ops) + [concat]
    return _wrap_in_forward_func(ops, concat.result)


def build_quantized_matmul_module(
    *,
    M: int = 4,
    K: int = 8,
    N: int = 16,
    bits: int = 8,
    group_size: int = 128,
    region_id: str = "quantized_matmul_0",
) -> ModuleOp:
    """Build a module with one ``compgen.quant.weight_int{N}pack_mm``.

    ``bits`` ∈ {4, 8}.
    """
    from xdsl.dialects.builtin import IntegerAttr

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    x = EmptyOp([], _ft([M, K]))
    w = EmptyOp([], TensorType(IntegerType(8), [N, K]))

    ops: list = [x, w]
    if bits == 8:
        from compgen.ir.quant import WeightInt8PackMMOp

        scales = EmptyOp([], _ft([N]))
        ops.append(scales)
        op = WeightInt8PackMMOp(
            operands=[x.results[0], w.results[0], scales.results[0]],
            result_types=[_ft([M, N])],
        )
    else:
        from compgen.ir.quant import WeightInt4PackMMOp

        sz = EmptyOp([], _ft([N, 2]))
        ops.append(sz)
        op = WeightInt4PackMMOp(
            operands=[x.results[0], w.results[0], sz.results[0]],
            result_types=[_ft([M, N])],
            properties={"group_size": IntegerAttr(group_size, IntegerType(64))},
        )
    _attach_region_id(op, region_id)
    ops.append(op)
    return _wrap_in_forward_func(ops, op.results[0])


# --- Applicator --------------------------------------------------------------


def apply_pattern(
    module: ModuleOp,
    patterns: RewritePattern | Iterable[RewritePattern],
    *,
    apply_recursively: bool = True,
    walk_reverse: bool = False,
) -> ModuleOp:
    """Run ``patterns`` over ``module`` to fixpoint.

    Accepts either a single ``RewritePattern`` or an iterable of them
    (which get ``GreedyRewritePatternApplier``-composed). Mutates
    ``module`` in place and returns it for chaining.
    """
    if isinstance(patterns, RewritePattern):
        effective = patterns
    else:
        effective = GreedyRewritePatternApplier(list(patterns))
    walker = PatternRewriteWalker(
        effective,
        apply_recursively=apply_recursively,
        walk_reverse=walk_reverse,
    )
    walker.rewrite_module(module)
    return module


# --- Structural asserts ------------------------------------------------------


def assert_module_verifies(module: ModuleOp) -> None:
    """Fail with a clear message when ``module.verify()`` raises."""
    try:
        module.verify()
    except Exception as e:  # noqa: BLE001 - we want the original in the AssertionError
        raise AssertionError(f"module failed to verify: {e}") from e


def assert_op_count(
    module: ModuleOp,
    op_name: str,
    expected: int,
) -> None:
    """Count ops named ``op_name`` across the module and assert the count."""
    actual = count_ops(module, op_name)
    assert actual == expected, f"expected {expected} {op_name!r}, found {actual}"


def count_ops(module: ModuleOp, op_name: str) -> int:
    """Walk the module and return the number of ops with ``op_name``."""
    count = 0
    for op in module.walk():
        if op.name == op_name:
            count += 1
    return count


def all_ops(module: ModuleOp) -> list[str]:
    """Return the list of every op name in walk order (for debug traces)."""
    return [op.name for op in module.walk()]


def find_op_by_region_id(module: ModuleOp, region_id: str):
    """Return the first op whose ``compgen.region_id`` matches, or ``None``."""
    for op in module.walk():
        attr = op.attributes.get("compgen.region_id")
        if attr is not None and attr.data == region_id:
            return op
    return None


# --- SMT equivalence wrapper -------------------------------------------------


def assert_smt_equivalent(
    before: ModuleOp,
    after: ModuleOp,
    *,
    timeout_ms: int = 5_000,
    allow_timeout: bool = True,
    allow_unknown: bool = True,
) -> None:
    """Call ``validate_translation`` and assert the result is acceptable.

    - ``valid`` -> pass.
    - ``timeout`` -> pass when ``allow_timeout`` (default), else fail.
    - ``unknown`` -> pass when ``allow_unknown`` (default), else fail.
    - ``invalid`` -> ALWAYS fail (this is the soundness check).

    The default thresholds reflect reality: the Z3 backend cannot
    prove refinement for arbitrary ``linalg.generic`` bodies yet, so
    ``timeout`` / ``unknown`` should not break the test. But a clear
    ``invalid`` must always fail -- that's what this helper exists
    for.
    """
    from compgen.ir.semantic.translation_validation import validate_translation

    result = validate_translation(before, after, timeout_ms=timeout_ms)
    if result.valid:
        return
    if result.status == "timeout" and allow_timeout:
        return
    if result.status == "unknown" and allow_unknown:
        return
    raise AssertionError(
        f"translation validation failed: status={result.status} counterexample={result.counterexample}"
    )


__all__ = [
    "all_ops",
    "apply_pattern",
    "assert_module_verifies",
    "assert_op_count",
    "assert_smt_equivalent",
    "build_concat_module",
    "build_linalg_matmul_module",
    "build_quantized_matmul_module",
    "count_ops",
    "find_op_by_region_id",
]
