"""Registration of ``xpu_rt.collective``."""

from __future__ import annotations

from xdsl.ir import Dialect

from xpu_rt.ir.collective.attrs import ReduceKindAttr, ShardingSpecAttr
from xpu_rt.ir.collective.ops import COLLECTIVE_OPS

ALL_OPS = list(COLLECTIVE_OPS)
ALL_ATTRS = [ShardingSpecAttr, ReduceKindAttr]

Collective = Dialect("xpu_rt.collective", ALL_OPS, ALL_ATTRS)


__all__ = ["ALL_ATTRS", "ALL_OPS", "Collective"]
