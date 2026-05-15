"""``xpu_rt.collective`` -- distributed / collective communication ops."""

from __future__ import annotations

from xpu_rt.ir.collective.attrs import ReduceKindAttr, ShardingSpecAttr
from xpu_rt.ir.collective.dialect import ALL_ATTRS, ALL_OPS, Collective
from xpu_rt.ir.collective.ops import (
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
