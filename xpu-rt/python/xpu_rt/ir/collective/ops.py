"""Ops for the ``compgen.collective`` dialect.

Four collective primitives modelled after XLA's HLO collectives +
NCCL / RCCL device collectives:

- ``all_reduce``     -- reduce across replicas, replicate result.
- ``all_gather``     -- gather per-device shards along an axis.
- ``reduce_scatter`` -- reduce then scatter along an axis.
- ``broadcast``      -- replicate one device's tensor across all.

Every op carries a ``ShardingSpecAttr`` describing the producer +
consumer sharding and a ``replica_group`` list identifying the
participating devices.
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Attribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.collective.attrs import ReduceKindAttr, ShardingSpecAttr


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    """Reduce across replicas, every device ends up with the full result."""

    name = "compgen.collective.all_reduce"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    reduce_kind = prop_def(ReduceKindAttr)
    replica_groups = prop_def(ArrayAttr)  # ArrayAttr of ArrayAttr<i64>
    channel_id = opt_prop_def(IntegerAttr)
    sharding_in = opt_prop_def(ShardingSpecAttr)
    sharding_out = opt_prop_def(ShardingSpecAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if not self.replica_groups.data:
            raise VerifyException(
                f"{self.name}: replica_groups must list at least one group"
            )


@irdl_op_definition
class AllGatherOp(IRDLOperation):
    """Gather shards from every device along one axis."""

    name = "compgen.collective.all_gather"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    all_gather_dim = prop_def(IntegerAttr)
    replica_groups = prop_def(ArrayAttr)
    channel_id = opt_prop_def(IntegerAttr)
    sharding_in = opt_prop_def(ShardingSpecAttr)
    sharding_out = opt_prop_def(ShardingSpecAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.all_gather_dim.value.data < 0:
            raise VerifyException(
                f"{self.name}: all_gather_dim must be non-negative"
            )


@irdl_op_definition
class ReduceScatterOp(IRDLOperation):
    """Reduce across replicas + scatter along one axis."""

    name = "compgen.collective.reduce_scatter"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    scatter_dim = prop_def(IntegerAttr)
    reduce_kind = prop_def(ReduceKindAttr)
    replica_groups = prop_def(ArrayAttr)
    channel_id = opt_prop_def(IntegerAttr)
    sharding_in = opt_prop_def(ShardingSpecAttr)
    sharding_out = opt_prop_def(ShardingSpecAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.scatter_dim.value.data < 0:
            raise VerifyException(
                f"{self.name}: scatter_dim must be non-negative"
            )


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    """Replicate one device's tensor across every other device."""

    name = "compgen.collective.broadcast"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    source_replica = prop_def(IntegerAttr)
    replica_groups = prop_def(ArrayAttr)
    channel_id = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())


COLLECTIVE_OPS: list[type[IRDLOperation]] = [
    AllReduceOp,
    AllGatherOp,
    ReduceScatterOp,
    BroadcastOp,
]


__all__ = [
    "AllGatherOp",
    "AllReduceOp",
    "BroadcastOp",
    "COLLECTIVE_OPS",
    "ReduceScatterOp",
]
