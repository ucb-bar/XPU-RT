"""Operations for ``compgen.tensor_ext``.

Each op mirrors the corresponding MLIR upstream op so that future
round-trips through MLIR parse-back cleanly:

- ``ConcatOp`` -- ``tensor.concat``. Concatenates a variadic list of
  input tensors along ``dim``.
- ``PackOp`` -- ``tensor.pack``. Tiles a source tensor into a packed
  tensor by splitting ``inner_dims_pos`` dimensions into
  ``inner_tiles`` outer + inner pairs, optionally with an
  ``outer_dims_perm`` permutation and a ``padding_value`` scalar.
- ``UnpackOp`` -- ``tensor.unpack``. The inverse of ``PackOp``: folds
  inner tile dimensions back into the original source layout.

All three ops are ``Pure`` (no side effects). Verification mirrors
MLIR's checks where applicable -- axis bounds, rank relations,
uniqueness of ``inner_dims_pos``.
"""

from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    IntegerAttr,
    IntegerType,
    TensorType,
    i64,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


# --- helpers -----------------------------------------------------------------


def _dense_i64(values: list[int]) -> DenseArrayBase:
    return DenseArrayBase.from_list(i64, values)


def _dense_i64_values(attr: DenseArrayBase) -> list[int]:
    """Extract Python ints from a dense i64 array attribute."""
    data = attr.get_values()
    return [int(v) for v in data]


# --- ConcatOp ----------------------------------------------------------------


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """Concatenate tensors along ``dim``.

    Mirrors ``tensor.concat`` from MLIR upstream::

        %out = compgen.tensor_ext.concat dim(d) %a, %b, %c
            : (tensor<2x4xf32>, tensor<3x4xf32>, tensor<5x4xf32>)
            -> tensor<10x4xf32>

    All inputs must have the same rank and the same shape on every
    non-concat dimension; the result's extent along ``dim`` is the
    sum of each input's extent along ``dim``.
    """

    name = "compgen.tensor_ext.concat"

    inputs = var_operand_def(Attribute)
    result = result_def(Attribute)

    dim = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        inputs: list[SSAValue],
        dim: int | IntegerAttr,
        result_type: Attribute,
    ) -> None:
        if isinstance(dim, int):
            dim = IntegerAttr(dim, IntegerType(64))
        super().__init__(
            operands=[list(inputs)],
            result_types=[result_type],
            properties={"dim": dim},
        )

    def verify_(self) -> None:
        if len(self.inputs) == 0:
            raise VerifyException(
                f"{self.name}: requires at least one input tensor"
            )

        d = self.dim.value.data
        # Fetch input + result types. Only enforce structural checks
        # when they are TensorType (the standard shape).
        input_types = [t for t in (op.type for op in self.inputs)]
        result_type = self.result.type
        if not all(isinstance(t, TensorType) for t in input_types):
            return
        if not isinstance(result_type, TensorType):
            return

        first_shape = list(input_types[0].get_shape())
        rank = len(first_shape)
        if d < 0 or d >= rank:
            raise VerifyException(
                f"{self.name}: dim {d} is out of range for rank-{rank} inputs"
            )

        elem = input_types[0].get_element_type()
        for t in input_types[1:]:
            shape = list(t.get_shape())
            if len(shape) != rank:
                raise VerifyException(
                    f"{self.name}: all inputs must have the same rank "
                    f"(got {rank} and {len(shape)})"
                )
            for i, (lhs, rhs) in enumerate(zip(first_shape, shape, strict=True)):
                if i == d:
                    continue
                if lhs != rhs:
                    raise VerifyException(
                        f"{self.name}: mismatched extent on non-concat dim {i} "
                        f"({lhs} vs {rhs})"
                    )
            if t.get_element_type() != elem:
                raise VerifyException(
                    f"{self.name}: all inputs must share an element type"
                )

        res_shape = list(result_type.get_shape())
        if len(res_shape) != rank:
            raise VerifyException(
                f"{self.name}: result rank {len(res_shape)} must equal "
                f"input rank {rank}"
            )
        # The only dim permitted to be ``-1`` (dynamic) is ``d``; every
        # other dim must match the common input shape.
        for i in range(rank):
            if i == d:
                continue
            if res_shape[i] != first_shape[i]:
                raise VerifyException(
                    f"{self.name}: result dim {i} ({res_shape[i]}) "
                    f"must equal input dim ({first_shape[i]})"
                )


# --- PackOp ------------------------------------------------------------------


@irdl_op_definition
class PackOp(IRDLOperation):
    """Tile a source tensor by splitting ``inner_dims_pos`` into inner tiles.

    Mirrors ``tensor.pack`` semantics::

        %packed = compgen.tensor_ext.pack %src
            inner_dims_pos = [1, 0]
            inner_tiles    = [32, 16]
            outer_dims_perm = [0, 1]
            padding_value  = %pad
            : (tensor<128x128xf32>, f32) -> tensor<8x4x16x32xf32>

    Semantics:

    - The source tensor has rank ``N``.
    - ``inner_dims_pos`` (length ``K``) names which source dims get
      split into inner tiles. Entries must be unique and in ``[0, N)``.
    - ``inner_tiles`` (length ``K``) gives the inner tile size for
      each position.
    - ``outer_dims_perm`` (length ``N``, optional) permutes the outer
      dims of the result.
    - ``padding_value`` is an optional scalar used when ``inner_tiles``
      don't divide the source shape evenly.

    Result rank = ``N + K``: first ``N`` dims are the (permuted)
    outer dims (each equal to ``ceildiv(src_dim, tile)`` for dims in
    ``inner_dims_pos``, unchanged otherwise); last ``K`` dims are the
    inner tiles in the order given by ``inner_dims_pos``.
    """

    name = "compgen.tensor_ext.pack"

    source = operand_def(Attribute)
    padding_value = opt_operand_def(Attribute)
    result = result_def(Attribute)

    inner_dims_pos = prop_def(DenseArrayBase)
    inner_tiles = prop_def(DenseArrayBase)
    outer_dims_perm = opt_prop_def(DenseArrayBase)

    traits = traits_def(Pure())

    def __init__(
        self,
        source: SSAValue,
        inner_dims_pos: list[int] | DenseArrayBase,
        inner_tiles: list[int] | DenseArrayBase,
        result_type: Attribute,
        *,
        outer_dims_perm: list[int] | DenseArrayBase | None = None,
        padding_value: SSAValue | None = None,
    ) -> None:
        if isinstance(inner_dims_pos, list):
            inner_dims_pos = _dense_i64(inner_dims_pos)
        if isinstance(inner_tiles, list):
            inner_tiles = _dense_i64(inner_tiles)
        if isinstance(outer_dims_perm, list):
            outer_dims_perm = _dense_i64(outer_dims_perm)

        properties: dict[str, Attribute] = {
            "inner_dims_pos": inner_dims_pos,
            "inner_tiles": inner_tiles,
        }
        if outer_dims_perm is not None:
            properties["outer_dims_perm"] = outer_dims_perm

        operands: list[list[SSAValue]] = [
            [source],
            [padding_value] if padding_value is not None else [],
        ]
        super().__init__(
            operands=operands,
            result_types=[result_type],
            properties=properties,
        )

    def verify_(self) -> None:
        dims_pos = _dense_i64_values(self.inner_dims_pos)
        tiles = _dense_i64_values(self.inner_tiles)
        if len(dims_pos) != len(tiles):
            raise VerifyException(
                f"{self.name}: inner_dims_pos ({len(dims_pos)}) and "
                f"inner_tiles ({len(tiles)}) must have the same length"
            )
        if len(set(dims_pos)) != len(dims_pos):
            raise VerifyException(
                f"{self.name}: inner_dims_pos entries must be unique: {dims_pos}"
            )
        for p in dims_pos:
            if p < 0:
                raise VerifyException(
                    f"{self.name}: inner_dims_pos entries must be non-negative, "
                    f"got {p}"
                )
        for t in tiles:
            if t <= 0:
                raise VerifyException(
                    f"{self.name}: inner_tiles must be strictly positive, got {t}"
                )

        src_type = self.source.type
        res_type = self.result.type
        if not isinstance(src_type, TensorType) or not isinstance(
            res_type, TensorType
        ):
            return
        src_rank = len(src_type.get_shape())
        res_rank = len(res_type.get_shape())
        if res_rank != src_rank + len(dims_pos):
            raise VerifyException(
                f"{self.name}: result rank {res_rank} must equal "
                f"source rank {src_rank} + {len(dims_pos)} inner tiles"
            )
        for p in dims_pos:
            if p >= src_rank:
                raise VerifyException(
                    f"{self.name}: inner_dims_pos entry {p} out of range "
                    f"for rank-{src_rank} source"
                )
        if self.outer_dims_perm is not None:
            perm = _dense_i64_values(self.outer_dims_perm)
            if len(perm) != src_rank:
                raise VerifyException(
                    f"{self.name}: outer_dims_perm must have length "
                    f"{src_rank}, got {len(perm)}"
                )
            if sorted(perm) != list(range(src_rank)):
                raise VerifyException(
                    f"{self.name}: outer_dims_perm must be a permutation of "
                    f"[0, {src_rank}), got {perm}"
                )


# --- UnpackOp ----------------------------------------------------------------


@irdl_op_definition
class UnpackOp(IRDLOperation):
    """Inverse of :class:`PackOp`.

    Folds inner tile dims back into the source layout. Result rank =
    ``source rank - len(inner_dims_pos)``.
    """

    name = "compgen.tensor_ext.unpack"

    source = operand_def(Attribute)
    result = result_def(Attribute)

    inner_dims_pos = prop_def(DenseArrayBase)
    inner_tiles = prop_def(DenseArrayBase)
    outer_dims_perm = opt_prop_def(DenseArrayBase)

    traits = traits_def(Pure())

    def __init__(
        self,
        source: SSAValue,
        inner_dims_pos: list[int] | DenseArrayBase,
        inner_tiles: list[int] | DenseArrayBase,
        result_type: Attribute,
        *,
        outer_dims_perm: list[int] | DenseArrayBase | None = None,
    ) -> None:
        if isinstance(inner_dims_pos, list):
            inner_dims_pos = _dense_i64(inner_dims_pos)
        if isinstance(inner_tiles, list):
            inner_tiles = _dense_i64(inner_tiles)
        if isinstance(outer_dims_perm, list):
            outer_dims_perm = _dense_i64(outer_dims_perm)

        properties: dict[str, Attribute] = {
            "inner_dims_pos": inner_dims_pos,
            "inner_tiles": inner_tiles,
        }
        if outer_dims_perm is not None:
            properties["outer_dims_perm"] = outer_dims_perm

        super().__init__(
            operands=[source],
            result_types=[result_type],
            properties=properties,
        )

    def verify_(self) -> None:
        dims_pos = _dense_i64_values(self.inner_dims_pos)
        tiles = _dense_i64_values(self.inner_tiles)
        if len(dims_pos) != len(tiles):
            raise VerifyException(
                f"{self.name}: inner_dims_pos ({len(dims_pos)}) and "
                f"inner_tiles ({len(tiles)}) must have the same length"
            )
        if len(set(dims_pos)) != len(dims_pos):
            raise VerifyException(
                f"{self.name}: inner_dims_pos entries must be unique: {dims_pos}"
            )
        for t in tiles:
            if t <= 0:
                raise VerifyException(
                    f"{self.name}: inner_tiles must be strictly positive, got {t}"
                )

        src_type = self.source.type
        res_type = self.result.type
        if not isinstance(src_type, TensorType) or not isinstance(
            res_type, TensorType
        ):
            return
        src_rank = len(src_type.get_shape())
        res_rank = len(res_type.get_shape())
        if src_rank != res_rank + len(dims_pos):
            raise VerifyException(
                f"{self.name}: source rank {src_rank} must equal "
                f"result rank {res_rank} + {len(dims_pos)} inner tiles"
            )
        for p in dims_pos:
            if p < 0 or p >= res_rank:
                raise VerifyException(
                    f"{self.name}: inner_dims_pos entry {p} out of range "
                    f"for rank-{res_rank} result"
                )
        if self.outer_dims_perm is not None:
            perm = _dense_i64_values(self.outer_dims_perm)
            if len(perm) != res_rank:
                raise VerifyException(
                    f"{self.name}: outer_dims_perm must have length "
                    f"{res_rank}, got {len(perm)}"
                )
            if sorted(perm) != list(range(res_rank)):
                raise VerifyException(
                    f"{self.name}: outer_dims_perm must be a permutation of "
                    f"[0, {res_rank}), got {perm}"
                )


TENSOR_EXT_OPS: list[type[IRDLOperation]] = [ConcatOp, PackOp, UnpackOp]


__all__ = [
    "ConcatOp",
    "PackOp",
    "TENSOR_EXT_OPS",
    "UnpackOp",
]
