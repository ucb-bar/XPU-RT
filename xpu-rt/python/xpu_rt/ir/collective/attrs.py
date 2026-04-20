"""Attributes for the ``compgen.collective`` dialect."""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def

_VALID_REDUCE_OPS: frozenset[str] = frozenset({"sum", "mean", "max", "min", "prod"})


@irdl_attr_definition
class ShardingSpecAttr(ParametrizedAttribute):
    """Per-dim sharding specification for a tensor.

    Parameters:
        devices: flat list of device counts per mesh axis, e.g.
            ``[4, 2]`` for a 4×2 mesh.
        dim_map: one entry per tensor dim; each entry is a ``StringAttr``
            naming the mesh axis sharded on that dim, or ``"replicated"``
            if replicated.
        partial: whether the sharding carries an unreduced partial
            result (``sum`` / ``mean`` etc. pending). One of
            ``"none"`` / ``"sum"`` / ``"mean"``.
    """

    name = "compgen.collective.sharding_spec"

    devices: ArrayAttr = param_def(ArrayAttr)
    dim_map: ArrayAttr = param_def(ArrayAttr)
    partial: StringAttr = param_def(StringAttr)

    _VALID_PARTIAL: ClassVar[frozenset[str]] = frozenset({"none", "sum", "mean", "max", "min"})

    def __init__(
        self,
        devices: list[int] | ArrayAttr,
        dim_map: list[str] | ArrayAttr,
        partial: str | StringAttr = "none",
    ) -> None:
        if isinstance(devices, list):
            devices = ArrayAttr([IntegerAttr(d, IntegerType(64)) for d in devices])
        if isinstance(dim_map, list):
            dim_map = ArrayAttr([StringAttr(s) for s in dim_map])
        if isinstance(partial, str):
            if partial not in self._VALID_PARTIAL:
                raise ValueError(f"partial must be one of {sorted(self._VALID_PARTIAL)}, got {partial!r}")
            partial = StringAttr(partial)
        super().__init__(devices, dim_map, partial)


@irdl_attr_definition
class ReduceKindAttr(ParametrizedAttribute):
    """Reduction-kind enum for collective reductions."""

    name = "compgen.collective.reduce_kind"

    kind: StringAttr = param_def(StringAttr)

    def __init__(self, kind: str | StringAttr) -> None:
        if isinstance(kind, str):
            if kind not in _VALID_REDUCE_OPS:
                raise ValueError(f"kind must be one of {sorted(_VALID_REDUCE_OPS)}, got {kind!r}")
            kind = StringAttr(kind)
        super().__init__(kind)


__all__ = [
    "ReduceKindAttr",
    "ShardingSpecAttr",
]
