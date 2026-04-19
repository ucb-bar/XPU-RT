"""Registration of ``compgen.collective``."""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.collective.attrs import ReduceKindAttr, ShardingSpecAttr
from compgen.ir.collective.ops import COLLECTIVE_OPS

ALL_OPS = list(COLLECTIVE_OPS)
ALL_ATTRS = [ShardingSpecAttr, ReduceKindAttr]

Collective = Dialect("compgen.collective", ALL_OPS, ALL_ATTRS)


__all__ = ["ALL_ATTRS", "ALL_OPS", "Collective"]
