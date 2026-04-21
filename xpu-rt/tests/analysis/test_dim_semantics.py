"""Tests for ``compgen.analysis.dim_semantics``.

Locks in:
  * matmul output dims tagged (PARALLEL, PARALLEL); reduce axis recorded
  * batch_matmul tagged (BATCH, PARALLEL, PARALLEL)
  * softmax / rmsnorm get a REDUCE on axis -1
  * pointwise ops get all PARALLEL
  * annotate_dim_roles writes IR attrs that dim_roles_for_op reads back
"""

from __future__ import annotations

import pytest

from compgen.analysis.dim_semantics import (
    DimRole,
    analyze_op,
    annotate_dim_roles,
    dim_roles_for_op,
)


# Minimal fakes — avoid the cost of real xDSL ops in unit tests.
class _FakeType:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape
    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _FakeResult:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.type = _FakeType(shape)


class _FakeOp:
    def __init__(self, name: str, shape: tuple[int, ...], hint: str | None = None) -> None:
        self.name = name
        self.results = [_FakeResult(shape)]
        self.attributes: dict = {}
        if hint is not None:
            class _H:
                def __init__(self, d): self.data = d
            self.attributes["compgen._pattern_hint"] = _H(hint)


def test_matmul_output_dims_are_parallel_with_reduce_axes() -> None:
    op = _FakeOp("linalg.matmul", shape=(64, 64))
    ann = analyze_op(op)
    assert ann is not None
    assert ann.output_roles == (DimRole.PARALLEL, DimRole.PARALLEL)
    # K reduction recorded as (input_idx, dim) pairs
    assert (0, 1) in ann.reduce_axes
    assert (1, 0) in ann.reduce_axes


def test_batch_matmul_first_dim_is_batch() -> None:
    op = _FakeOp("linalg.batch_matmul", shape=(8, 64, 64))
    ann = analyze_op(op)
    assert ann is not None
    assert ann.output_roles == (DimRole.BATCH, DimRole.PARALLEL, DimRole.PARALLEL)


def test_softmax_last_dim_is_reduce() -> None:
    op = _FakeOp("func.call", shape=(2, 4, 8, 128), hint="softmax")
    ann = analyze_op(op)
    assert ann is not None
    # Last dim (-1) is REDUCE; others PARALLEL.
    assert ann.output_roles[-1] is DimRole.REDUCE
    assert all(r is DimRole.PARALLEL for r in ann.output_roles[:-1])


def test_rmsnorm_uses_pattern_hint() -> None:
    op = _FakeOp("func.call", shape=(2, 4, 128), hint="rmsnorm")
    ann = analyze_op(op)
    assert ann is not None
    assert ann.output_roles[-1] is DimRole.REDUCE


def test_pointwise_ops_are_all_parallel() -> None:
    op = _FakeOp("arith.addf", shape=(64, 128))
    ann = analyze_op(op)
    assert ann is not None
    assert ann.output_roles == (DimRole.PARALLEL, DimRole.PARALLEL)
    assert "pointwise" in ann.notes


def test_annotate_dim_roles_round_trips_through_ir_attrs() -> None:
    """Writing + reading-back via the compgen.dim_role attr."""
    from xdsl.dialects.builtin import ArrayAttr, Float32Type, ModuleOp, StringAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import MatmulOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region

    f32 = Float32Type()
    M, K, N = 4, 8, 4
    lhs_t = TensorType(f32, [M, K])
    rhs_t = TensorType(f32, [K, N])
    out_t = TensorType(f32, [M, N])

    block = Block(arg_types=[lhs_t, rhs_t])
    out_empty = EmptyOp([], out_t)
    block.add_op(out_empty)
    mm = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[out_empty.results[0]],
        res=[out_t],
    )
    block.add_op(mm)
    block.add_op(ReturnOp(mm.results[0]))
    func = FuncOp("forward", ((lhs_t, rhs_t), (out_t,)), Region([block]))
    module = ModuleOp([func])

    n = annotate_dim_roles(module)
    assert n >= 1   # matmul (and tensor.empty if it counts) annotated

    roles = dim_roles_for_op(mm)
    assert roles == (DimRole.PARALLEL, DimRole.PARALLEL)
