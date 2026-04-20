"""Tests for the ``compgen.collective`` dialect."""

from __future__ import annotations

import pytest
from compgen.ir.collective import (
    ALL_ATTRS,
    ALL_OPS,
    AllGatherOp,
    AllReduceOp,
    BroadcastOp,
    Collective,
    ReduceKindAttr,
    ReduceScatterOp,
    ShardingSpecAttr,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float32Type,
    IntegerAttr,
    IntegerType,
    TensorType,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Dialect
from xdsl.utils.exceptions import VerifyException


def _t(shape):
    return TensorType(Float32Type(), list(shape))


def _groups(n=4):
    return ArrayAttr([ArrayAttr([IntegerAttr(i, IntegerType(64)) for i in range(n)])])


# --- dialect registration ------------------------------------------------


def test_collective_dialect():
    assert isinstance(Collective, Dialect)
    assert Collective.name == "compgen.collective"


def test_dialect_has_four_ops():
    assert len(ALL_OPS) == 4


def test_dialect_has_two_attrs():
    assert len(ALL_ATTRS) == 2


# --- ShardingSpecAttr ---------------------------------------------------


def test_sharding_spec_builds():
    s = ShardingSpecAttr(devices=[4, 2], dim_map=["axis0", "replicated"])
    assert s.partial.data == "none"
    assert len(s.devices.data) == 2


def test_sharding_spec_rejects_bad_partial():
    with pytest.raises(ValueError, match="partial"):
        ShardingSpecAttr(devices=[4], dim_map=["axis0"], partial="weird")


def test_reduce_kind_rejects_unknown():
    with pytest.raises(ValueError, match="kind"):
        ReduceKindAttr("zoombooster")


# --- AllReduce ----------------------------------------------------------


def test_all_reduce_builds_and_verifies():
    e = EmptyOp([], _t([8, 16]))
    op = AllReduceOp.build(
        operands=[e.results[0]],
        result_types=[_t([8, 16])],
        properties={
            "reduce_kind": ReduceKindAttr("sum"),
            "replica_groups": _groups(4),
        },
    )
    op.verify()


def test_all_reduce_rejects_empty_replica_groups():
    e = EmptyOp([], _t([8, 16]))
    op = AllReduceOp.build(
        operands=[e.results[0]],
        result_types=[_t([8, 16])],
        properties={
            "reduce_kind": ReduceKindAttr("sum"),
            "replica_groups": ArrayAttr([]),
        },
    )
    with pytest.raises(VerifyException, match="replica_groups"):
        op.verify()


# --- AllGather ----------------------------------------------------------


def test_all_gather_builds():
    e = EmptyOp([], _t([8, 16]))
    op = AllGatherOp.build(
        operands=[e.results[0]],
        result_types=[_t([32, 16])],
        properties={
            "all_gather_dim": IntegerAttr(0, IntegerType(64)),
            "replica_groups": _groups(4),
        },
    )
    op.verify()


def test_all_gather_rejects_negative_dim():
    e = EmptyOp([], _t([8, 16]))
    op = AllGatherOp.build(
        operands=[e.results[0]],
        result_types=[_t([32, 16])],
        properties={
            "all_gather_dim": IntegerAttr(-1, IntegerType(64)),
            "replica_groups": _groups(4),
        },
    )
    with pytest.raises(VerifyException, match="all_gather_dim"):
        op.verify()


# --- ReduceScatter ------------------------------------------------------


def test_reduce_scatter_builds():
    e = EmptyOp([], _t([32, 16]))
    op = ReduceScatterOp.build(
        operands=[e.results[0]],
        result_types=[_t([8, 16])],
        properties={
            "scatter_dim": IntegerAttr(0, IntegerType(64)),
            "reduce_kind": ReduceKindAttr("sum"),
            "replica_groups": _groups(4),
        },
    )
    op.verify()


# --- Broadcast ----------------------------------------------------------


def test_broadcast_builds():
    e = EmptyOp([], _t([8, 16]))
    op = BroadcastOp.build(
        operands=[e.results[0]],
        result_types=[_t([8, 16])],
        properties={
            "source_replica": IntegerAttr(0, IntegerType(64)),
            "replica_groups": _groups(4),
        },
    )
    op.verify()


# --- purity trait -------------------------------------------------------


@pytest.mark.parametrize("cls", [AllReduceOp, AllGatherOp, ReduceScatterOp, BroadcastOp])
def test_collective_ops_are_pure(cls):
    from xdsl.traits import Pure

    assert any(isinstance(t, Pure) for t in cls.traits.traits)
