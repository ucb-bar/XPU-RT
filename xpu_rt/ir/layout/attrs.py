"""Layout IR custom attributes.

Defines ParametrizedAttribute types for layout operations:
    - LayoutEncodingAttr: virtual layout encoding for a tensor operand
    - PackSpecAttr: physical pack/unpack specification
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


@irdl_attr_definition
class LayoutEncodingAttr(ParametrizedAttribute):
    """Virtual layout encoding for a tensor operand.

    Captures the logical layout decision for a specific operand of an
    operation, including tile dimensions and element types.

    Attributes:
        op_type: What kind of op (matmul, conv, elementwise, etc.).
        operand_index: Which operand (0=lhs, 1=rhs, 2=output).
        logical_layout: Layout kind -- rowmajor, colmajor, tiled, blocked.
        tile_dims: Tile shape dimensions (empty ArrayAttr if untiled).
        element_types: Dtype strings (e.g. ["f32"]).
    """

    name = "layout.encoding"
    op_type: StringAttr = param_def(StringAttr)
    operand_index: IntegerAttr = param_def(IntegerAttr)
    logical_layout: StringAttr = param_def(StringAttr)
    tile_dims: ArrayAttr = param_def(ArrayAttr)
    element_types: ArrayAttr = param_def(ArrayAttr)

    def __init__(
        self,
        op_type: str | StringAttr,
        operand_index: int | IntegerAttr,
        logical_layout: str | StringAttr,
        tile_dims: list[int] | ArrayAttr,
        element_types: list[str] | ArrayAttr,
    ) -> None:
        if isinstance(op_type, str):
            op_type = StringAttr(op_type)
        if isinstance(operand_index, int):
            operand_index = IntegerAttr(operand_index, IntegerType(64))
        if isinstance(logical_layout, str):
            logical_layout = StringAttr(logical_layout)
        if isinstance(tile_dims, list):
            tile_dims = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in tile_dims],
            )
        if isinstance(element_types, list):
            element_types = ArrayAttr(
                [StringAttr(t) for t in element_types],
            )
        super().__init__(op_type, operand_index, logical_layout, tile_dims, element_types)


@irdl_attr_definition
class PackSpecAttr(ParametrizedAttribute):
    """Physical pack/unpack specification.

    Describes how to materialize a layout encoding into concrete
    pack/transpose operations at tensor boundaries.

    Attributes:
        inner_tiles: Inner tiling dimensions (e.g. [16, 16]).
        outer_perm: Outer dimension permutation (e.g. [0, 1]).
        padding_value: Padding strategy -- "zero" or "none".
    """

    name = "layout.pack_spec"
    inner_tiles: ArrayAttr = param_def(ArrayAttr)
    outer_perm: ArrayAttr = param_def(ArrayAttr)
    padding_value: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        inner_tiles: list[int] | ArrayAttr,
        outer_perm: list[int] | ArrayAttr,
        padding_value: str | StringAttr,
    ) -> None:
        if isinstance(inner_tiles, list):
            inner_tiles = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in inner_tiles],
            )
        if isinstance(outer_perm, list):
            outer_perm = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in outer_perm],
            )
        if isinstance(padding_value, str):
            padding_value = StringAttr(padding_value)
        super().__init__(inner_tiles, outer_perm, padding_value)


__all__ = [
    "LayoutEncodingAttr",
    "PackSpecAttr",
]
