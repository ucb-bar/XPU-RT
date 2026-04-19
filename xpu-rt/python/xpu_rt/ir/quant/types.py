"""Type attributes for the ``compgen.quant`` dialect.

These are ParametrizedAttribute types that mirror TorchAO's quantized
tensor taxonomy one-for-one:

- ``AffineQuantizedTensorType`` -- TorchAO's canonical
  ``AffineQuantizedTensor``. Carries the storage element type, scale
  dtype, optional zero-point dtype, granularity
  ("per_tensor" / "per_channel" / "per_group"), block size (for
  per-group granularity), and an optional layout tag (for packing
  variants like tiled or plain).
- ``PackedIntTensorType`` -- carries the logical sub-byte bit width
  and the physical pack dimension so sub-byte normalization passes
  know how packed elements are laid out.
- ``MXQuantizedTensorType`` -- block-scaled FP formats (MX4, MX6,
  MX9) with E8M0 block scales.
- ``NVFP4TensorType`` -- NVIDIA's block-scaled FP4 format.

These types are *supplementary* metadata: the SSA value itself is
still typed as an ordinary ``tensor<...x iN>`` (or ``tensor<...x fN>``),
and the quantization type is attached to the producing op as a
property via ``prop_def``. Passes that need the quantization metadata
consult the property; passes that only care about the storage tensor
(buffer layout, DMA, memory space) operate on the ``TensorType``
directly.

Why not make quantization types the element type of ``TensorType``?
Because many downstream ops (``linalg.generic``, ``arith.muli``, etc.)
expect primitive element types. Layering the quantization metadata
as a side-attribute keeps round-tripping through upstream MLIR
straightforward.
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.ir import Attribute, ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


_VALID_GRANULARITY: frozenset[str] = frozenset({
    "per_tensor",
    "per_channel",
    "per_group",
    "per_token",
    "per_row",
})
"""Allowed values for the ``AffineQuantizedTensorType.granularity`` tag."""


_VALID_AFFINE_LAYOUTS: frozenset[str] = frozenset({
    "plain",
    "tensor_core_tiled",
    "marlin",
    "marlin_sparse",
    "block_sparse",
    "int4_cpu",
})
"""Allowed values for the ``AffineQuantizedTensorType.layout`` tag.

Mirrors the TorchAO layout variants documented in
``torchao.dtypes.affine_quantized_tensor_ops.LAYOUT_REGISTRY``.
"""


@irdl_attr_definition
class AffineQuantizedTensorType(ParametrizedAttribute):
    """Metadata type describing a TorchAO-style affine-quantized tensor.

    A value of this type is represented at the SSA level by a plain
    ``TensorType`` whose element type is ``storage_type``; this attribute
    is attached to the producing op as a property carrying the
    supplementary metadata.

    Parameters:
        storage_type: the element type of the stored tensor, e.g.
            ``IntegerType(8)`` or ``IntegerType(4)``.
        scale_dtype: the element type of the scale tensor, e.g.
            ``Float32Type()`` or ``BFloat16Type()``.
        zero_point_dtype: the element type of the zero-point tensor.
            Use a zero-width ``IntegerType(0)`` to indicate "absent"
            when the quantization is symmetric.
        granularity: one of ``per_tensor``, ``per_channel``,
            ``per_group``, ``per_token``, ``per_row``.
        block_size: a list of block dimensions (length matches tensor
            rank). ``per_tensor`` uses the whole tensor, ``per_channel``
            uses ``[1, ..., C, ..., 1]``, and ``per_group`` uses an
            explicit inner-dim block.
        layout: one of the TorchAO layouts in ``_VALID_AFFINE_LAYOUTS``.
    """

    name = "compgen.quant.affine_tensor"

    storage_type: Attribute = param_def(Attribute)
    scale_dtype: Attribute = param_def(Attribute)
    zero_point_dtype: Attribute = param_def(Attribute)
    granularity: StringAttr = param_def(StringAttr)
    block_size: ArrayAttr = param_def(ArrayAttr)
    layout: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        storage_type: Attribute,
        scale_dtype: Attribute,
        zero_point_dtype: Attribute | None = None,
        granularity: str | StringAttr = "per_tensor",
        block_size: list[int] | ArrayAttr | None = None,
        layout: str | StringAttr = "plain",
    ) -> None:
        if zero_point_dtype is None:
            zero_point_dtype = IntegerType(0)
        if isinstance(granularity, str):
            if granularity not in _VALID_GRANULARITY:
                raise ValueError(
                    f"Invalid granularity {granularity!r}; "
                    f"expected one of {sorted(_VALID_GRANULARITY)}"
                )
            granularity = StringAttr(granularity)
        if block_size is None:
            block_size = ArrayAttr([])
        elif isinstance(block_size, list):
            block_size = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in block_size],
            )
        if isinstance(layout, str):
            if layout not in _VALID_AFFINE_LAYOUTS:
                raise ValueError(
                    f"Invalid layout {layout!r}; "
                    f"expected one of {sorted(_VALID_AFFINE_LAYOUTS)}"
                )
            layout = StringAttr(layout)
        super().__init__(
            storage_type,
            scale_dtype,
            zero_point_dtype,
            granularity,
            block_size,
            layout,
        )


@irdl_attr_definition
class PackedIntTensorType(ParametrizedAttribute):
    """Metadata type for a sub-byte packed integer tensor.

    Parameters:
        bit_width: logical bit width per element (2/3/4/6).
        pack_dim: the physical packing dimension.
        storage_type: the byte-level storage element type (e.g.
            ``IntegerType(8)`` when packing 4-bit values two-per-byte).
    """

    name = "compgen.quant.packed_int_tensor"

    bit_width: IntegerAttr = param_def(IntegerAttr)
    pack_dim: IntegerAttr = param_def(IntegerAttr)
    storage_type: Attribute = param_def(Attribute)

    _VALID_BITS: ClassVar[frozenset[int]] = frozenset({2, 3, 4, 6})

    def __init__(
        self,
        bit_width: int | IntegerAttr,
        pack_dim: int | IntegerAttr,
        storage_type: Attribute | None = None,
    ) -> None:
        if isinstance(bit_width, int):
            if bit_width not in self._VALID_BITS:
                raise ValueError(
                    f"PackedIntTensorType bit_width must be one of "
                    f"{sorted(self._VALID_BITS)}, got {bit_width}"
                )
            bit_width = IntegerAttr(bit_width, IntegerType(64))
        if isinstance(pack_dim, int):
            pack_dim = IntegerAttr(pack_dim, IntegerType(64))
        if storage_type is None:
            storage_type = IntegerType(8)
        super().__init__(bit_width, pack_dim, storage_type)


@irdl_attr_definition
class MXQuantizedTensorType(ParametrizedAttribute):
    """Metadata type for the OCP MX block-scaled formats (MX4, MX6, MX9).

    Parameters:
        element_bit_width: 4 (MXFP4), 6 (MXFP6), or 9 (MXFP9 / INT8).
        block_size: block size in elements (commonly 32).
        scale_bit_width: scale element bit width (E8M0 = 8).
        scale_kind: ``"e8m0"`` (the OCP-standard scale format) or
            ``"fp32"`` for debug/reference builds.
    """

    name = "compgen.quant.mx_tensor"

    element_bit_width: IntegerAttr = param_def(IntegerAttr)
    block_size: IntegerAttr = param_def(IntegerAttr)
    scale_bit_width: IntegerAttr = param_def(IntegerAttr)
    scale_kind: StringAttr = param_def(StringAttr)

    _VALID_ELEMENT_BITS: ClassVar[frozenset[int]] = frozenset({4, 6, 8, 9})
    _VALID_SCALE_KINDS: ClassVar[frozenset[str]] = frozenset({"e8m0", "fp32"})

    def __init__(
        self,
        element_bit_width: int | IntegerAttr,
        block_size: int | IntegerAttr = 32,
        scale_bit_width: int | IntegerAttr = 8,
        scale_kind: str | StringAttr = "e8m0",
    ) -> None:
        if isinstance(element_bit_width, int):
            if element_bit_width not in self._VALID_ELEMENT_BITS:
                raise ValueError(
                    f"MXQuantizedTensorType element_bit_width must be one of "
                    f"{sorted(self._VALID_ELEMENT_BITS)}, got {element_bit_width}"
                )
            element_bit_width = IntegerAttr(element_bit_width, IntegerType(64))
        if isinstance(block_size, int):
            block_size = IntegerAttr(block_size, IntegerType(64))
        if isinstance(scale_bit_width, int):
            scale_bit_width = IntegerAttr(scale_bit_width, IntegerType(64))
        if isinstance(scale_kind, str):
            if scale_kind not in self._VALID_SCALE_KINDS:
                raise ValueError(
                    f"MXQuantizedTensorType scale_kind must be one of "
                    f"{sorted(self._VALID_SCALE_KINDS)}, got {scale_kind!r}"
                )
            scale_kind = StringAttr(scale_kind)
        super().__init__(
            element_bit_width,
            block_size,
            scale_bit_width,
            scale_kind,
        )


@irdl_attr_definition
class NVFP4TensorType(ParametrizedAttribute):
    """Metadata type for NVIDIA's block-scaled FP4 format.

    Parameters:
        block_size: elements per scale block (commonly 16).
        scale_dtype: dtype of the block scale (typically
            ``Float32Type`` or ``BFloat16Type``).
    """

    name = "compgen.quant.nvfp4_tensor"

    block_size: IntegerAttr = param_def(IntegerAttr)
    scale_dtype: Attribute = param_def(Attribute)

    def __init__(
        self,
        block_size: int | IntegerAttr = 16,
        scale_dtype: Attribute | None = None,
    ) -> None:
        if isinstance(block_size, int):
            block_size = IntegerAttr(block_size, IntegerType(64))
        if scale_dtype is None:
            from xdsl.dialects.builtin import Float32Type
            scale_dtype = Float32Type()
        super().__init__(block_size, scale_dtype)


__all__ = [
    "AffineQuantizedTensorType",
    "MXQuantizedTensorType",
    "NVFP4TensorType",
    "PackedIntTensorType",
]
