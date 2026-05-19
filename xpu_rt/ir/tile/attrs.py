"""Tile IR custom attributes.

Defines ParametrizedAttribute types for tile operations:
    - MemoryClassAttr: memory space classification
    - FragmentLayoutAttr: tile fragment data layout
    - TileShapeAttr: tile dimensions
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


@irdl_attr_definition
class MemoryClassAttr(ParametrizedAttribute):
    """Memory class for tile operations.

    Valid kinds: "global", "shared", "local", "accumulator", "register".
    """

    name = "tile.memory_class"
    kind: StringAttr = param_def(StringAttr)

    def __init__(self, kind: str | StringAttr) -> None:
        if isinstance(kind, str):
            kind = StringAttr(kind)
        super().__init__(kind)


@irdl_attr_definition
class FragmentLayoutAttr(ParametrizedAttribute):
    """Layout descriptor for a tile fragment.

    Valid layouts: "row_major", "col_major", "accumulator".
    """

    name = "tile.fragment_layout"
    layout: StringAttr = param_def(StringAttr)

    def __init__(self, layout: str | StringAttr) -> None:
        if isinstance(layout, str):
            layout = StringAttr(layout)
        super().__init__(layout)


@irdl_attr_definition
class TileShapeAttr(ParametrizedAttribute):
    """Tile shape: dimensions (M, N, optional K).

    Attributes:
        dims: ArrayAttr of IntegerAttr representing tile dimensions.
    """

    name = "tile.shape"
    dims: ArrayAttr = param_def(ArrayAttr)

    def __init__(self, dims: list[int] | ArrayAttr) -> None:
        if isinstance(dims, list):
            dims = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in dims],
            )
        super().__init__(dims)


__all__ = [
    "FragmentLayoutAttr",
    "MemoryClassAttr",
    "TileShapeAttr",
]
