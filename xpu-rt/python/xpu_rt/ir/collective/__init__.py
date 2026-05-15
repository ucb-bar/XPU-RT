"""``compgen.collective`` -- distributed / collective communication ops."""

from __future__ import annotations

from compgen.ir.collective.attrs import ReduceKindAttr, ShardingSpecAttr
from compgen.ir.collective.dialect import ALL_ATTRS, ALL_OPS, Collective
from compgen.ir.collective.ops import (
    AllGatherOp,
    AllReduceOp,
    BroadcastOp,
    ReduceScatterOp,
)

__all__ = [
    "ALL_ATTRS",
    "ALL_OPS",
    "AllGatherOp",
    "AllReduceOp",
    "BroadcastOp",
    "Collective",
    "ReduceKindAttr",
    "ReduceScatterOp",
    "ShardingSpecAttr",
]
