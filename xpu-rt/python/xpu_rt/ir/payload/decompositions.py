"""ATen to xDSL decomposition table.

Maps PyTorch ATen operator targets to functions that produce real xDSL
linalg/arith/tensor operations. This replaces the opaque ``func.call``
approach with structured IR the agent can reason about.

Each decomposition function takes the FX node's args (as xDSL SSAValues)
and metadata, and returns a list of xDSL Operations to insert into the block.

Ops without decompositions fall back to ``func.call`` (flagged as opaque).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import (
    DenseArrayBase,
    Float32Type,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import MatmulOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Operation, SSAValue


def _static_shape(shape_like: Any) -> list[int]:
    """Coerce a sequence of possibly-symbolic dims to xDSL-friendly ints.

    ``torch.export`` may emit ``SymInt`` dims (rendered as ``u6`` etc.)
    when the graph is traced under data-dependent or unbacked shape
    constraints (e.g. SmolVLA's image-tile counts). xDSL's
    :class:`TensorType` only accepts :class:`builtin.int`; a symbolic dim
    would otherwise surface as ``VerifyException: u6 should be of base
    attribute builtin.int``.

    This helper mirrors the more-narrow
    :func:`compgen.ir.payload.import_fx._coerce_static_dim`. Symbolic
    dims become xDSL's dynamic-dim sentinel (``-1``) so import
    completes; downstream passes that need static shapes handle the
    dynamic case through their own paths.
    """
    out: list[int] = []
    for dim in shape_like:
        try:
            out.append(int(dim))
        except Exception:
            out.append(-1)
    return out


@dataclass
class DecompResult:
    """Result of decomposing one FX node into xDSL ops.

    Attributes:
        ops: xDSL operations to insert into the block.
        result: The SSAValue that represents this node's output.
        region_ids: region_id labels attached to linalg ops.
        pattern_hint: Optional canonical pattern name (e.g. "layer_norm",
            "softmax", "rms_norm", "dequantize_per_channel"). The
            ``FXImporter`` propagates this onto every emitted op as the
            ``compgen._pattern_hint`` attribute so Phase 2 passes
            (``raise_special_ops``, ``fuse_dequant_matmul``, etc.) can
            recognize the op's origin without re-detecting.
    """

    ops: list[Operation] = field(default_factory=list)
    result: SSAValue | None = None
    region_ids: list[str] = field(default_factory=list)
    pattern_hint: str | None = None


# Type for decomposition functions
DecompFn = Callable[
    [
        list[SSAValue],  # positional operands (resolved FX args)
        dict[str, Any],  # FX node metadata (shapes, dtypes)
        str,  # node name (for region_id generation)
    ],
    DecompResult,
]


# ============================================================================
# Counters for unique region IDs
# ============================================================================

_region_counters: dict[str, int] = {}


def _next_region_id(prefix: str) -> str:
    """Generate a unique region ID like 'matmul_0', 'matmul_1'."""
    count = _region_counters.get(prefix, 0)
    _region_counters[prefix] = count + 1
    return f"{prefix}_{count}"


def reset_region_counters() -> None:
    """Reset counters between imports."""
    _region_counters.clear()


def _make_empty(result_type: TensorType) -> EmptyOp:
    """Create a tensor.empty for an output tensor."""
    return EmptyOp([], result_type)


def _attach_region_id(op: Operation, region_id: str) -> None:
    """Attach a compgen.region_id attribute to an operation."""
    op.attributes["compgen.region_id"] = StringAttr(region_id)


# ============================================================================
# Decomposition functions
# ============================================================================


def decompose_linear(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.linear.default(input, weight, bias?) -> matmul + bias add.

    linear(x, w, b) = x @ w^T + b
    - x: [M, K], w: [N, K] (note: weight is transposed), b: [N]
    - output: [M, N]
    """
    ops: list[Operation] = []
    region_ids: list[str] = []

    x = operands[0]  # input: [M, K]
    w = operands[1]  # weight: [N, K]

    # Get result type from metadata
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))

    # Step 1: Transpose weight [N, K] -> [K, N]
    w_type = w.type
    assert isinstance(w_type, TensorType)
    w_shape = w_type.get_shape()
    wt_type = TensorType(Float32Type(), [w_shape[1], w_shape[0]])
    wt_empty = _make_empty(wt_type)
    ops.append(wt_empty)

    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=w,
        init=wt_empty.results[0],
        permutation=perm,
        result=wt_type,
    )
    ops.append(transpose)

    # Step 2: Matmul: x [M, K] @ w^T [K, N] -> [M, N]
    mm_empty = _make_empty(result_type)
    ops.append(mm_empty)

    matmul = MatmulOp(
        inputs=[x, transpose.results[0]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)
    region_ids.append(rid)
    ops.append(matmul)

    result = matmul.results[0]

    # Step 3: Bias addition deferred — requires broadcast lowering
    # (bias is [N] but result is [M, N], needs linalg.generic with
    # indexing_maps for proper broadcast semantics)

    return DecompResult(ops=ops, result=result, region_ids=region_ids)


def decompose_gelu(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.gelu.default(input) -> element-wise GELU.

    For MVP, represent as a func.call to @aten_gelu (element-wise ops in
    linalg.generic require indexing_maps and a body region, which we'll
    add in a later phase). The region_id is still attached.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))

    # Create external function declaration for gelu
    # (In later phase, this becomes a linalg.generic with the GELU body)
    rid = _next_region_id("gelu")
    call = CallOp("aten_gelu", [operands[0]], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[call], result=call.res[0], region_ids=[rid])


def _coerce_static_dim(d: Any) -> int:
    try:
        return int(d)
    except Exception:
        return -1


def _scalar_to_tensor(scalar: Any, like_type: TensorType) -> tuple[list[Operation], SSAValue]:
    """Materialize a Python scalar as a constant tensor matching ``like_type``.

    Returns ``(ops_to_emit, ssa_value_to_use_as_operand)``.

    xDSL's ``DenseIntOrFPElementsAttr.from_list`` packs data through the
    element type's ``pack`` method. Some element types (notably
    ``BFloat16Type`` as of xDSL 0.24) raise ``NotImplementedError`` in
    ``pack`` — SmolVLA's vision tower carries bf16 weights, so we hit
    this on import. The fallback below materialises the constant as f32
    and emits an ``arith.truncf`` / ``arith.extf`` cast when the target
    element type differs, keeping the IR well-typed.
    """
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType

    elem = like_type.element_type
    is_int = isinstance(elem, IntegerType)
    data = [int(scalar)] if is_int else [float(scalar)]
    try:
        attr = DenseIntOrFPElementsAttr.from_list(like_type, data)
        const = ConstantOp(attr, like_type)
        return [const], const.result
    except NotImplementedError:
        pass

    # Pack fallback: build the constant in f32 and cast to the target.
    # Emits an opaque ``func.call @_compgen_cast`` rather than an xDSL
    # arith cast, because the linalg-on-tensor cast path would require
    # shape-aware loop nests that we don't have here. The cast call
    # carries a ``compgen.cast_to`` attribute so downstream passes can
    # lower it alongside the rest of the opaque fallbacks.
    from xdsl.dialects.func import CallOp

    f32_like = TensorType(Float32Type(), like_type.get_shape())
    f32_attr = DenseIntOrFPElementsAttr.from_list(f32_like, [float(scalar)])
    f32_const = ConstantOp(f32_attr, f32_like)
    cast = CallOp("_compgen_cast_scalar", [f32_const.result], [like_type])
    cast.attributes["compgen.cast_to"] = StringAttr(str(elem))
    return [f32_const, cast], cast.res[0]


def _binary_operands(operands: list[SSAValue], meta: dict[str, Any]) -> tuple[list[Operation], SSAValue, SSAValue]:
    """Resolve a binary aten op's two operands, materializing a scalar second
    operand from FX args when only one SSA value was supplied.

    Pre-fix this raised ``IndexError: list index out of range`` for
    ``aten.add.Tensor(tensor, scalar_int)`` because the scalar arrives as
    a Python int via ``meta['_fx_args']`` rather than as an SSA operand.
    """
    if len(operands) >= 2:
        return [], operands[0], operands[1]
    if len(operands) == 1:
        # Scalar second operand — broadcastable constant of result dtype.
        val: Any = meta["val"]
        elem = _element_type_from_meta(meta)
        # 1-element tensor; the elementwise add will broadcast against it.
        like = TensorType(elem, [1])
        scalar = _fx_arg(meta, 1, 0)
        ops, ssa = _scalar_to_tensor(scalar, like)
        return ops, operands[0], ssa
    raise IndexError("binary op with zero SSA operands")


def decompose_add_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.add.Tensor(a, b) -> element-wise add.

    Handles the (tensor, scalar) form by materialising the scalar as a
    1-element constant tensor of the result dtype.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, [_coerce_static_dim(d) for d in val.shape])

    pre, lhs, rhs = _binary_operands(operands, meta)

    rid = _next_region_id("add")
    call = CallOp("aten_add", [lhs, rhs], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[*pre, call], result=call.res[0], region_ids=[rid])


def decompose_mul_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mul.Tensor(a, b) -> element-wise mul.

    Handles the (tensor, scalar) form like ``decompose_add_tensor``.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, [_coerce_static_dim(d) for d in val.shape])

    pre, lhs, rhs = _binary_operands(operands, meta)

    rid = _next_region_id("mul")
    call = CallOp("aten_mul", [lhs, rhs], [result_type])
    _attach_region_id(call, rid)

    return DecompResult(ops=[*pre, call], result=call.res[0], region_ids=[rid])


def decompose_mm(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mm.default(a, b) -> linalg.matmul."""
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))

    mm_empty = _make_empty(result_type)
    matmul = MatmulOp(
        inputs=[operands[0], operands[1]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)

    return DecompResult(ops=[mm_empty, matmul], result=matmul.results[0], region_ids=[rid])


def decompose_transpose(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.t.default(input) -> linalg.transpose."""
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))

    t_empty = _make_empty(result_type)
    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=operands[0],
        init=t_empty.results[0],
        permutation=perm,
        result=result_type,
    )
    rid = _next_region_id("transpose")
    _attach_region_id(transpose, rid)

    return DecompResult(ops=[t_empty, transpose], result=transpose.results[0], region_ids=[rid])


def decompose_permute(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.permute.default(input, dims) for the common 2D transpose case."""

    operand_type = operands[0].type
    val: Any = meta["val"]
    result_shape = _static_shape(val.shape)
    if not isinstance(operand_type, TensorType):
        dims = None
    else:
        input_shape = list(operand_type.get_shape())
        dims = (1, 0) if len(input_shape) == 2 and input_shape == list(reversed(result_shape)) else None

    if dims != (1, 0):
        from xdsl.dialects.func import CallOp

        result_type = TensorType(Float32Type(), _static_shape(val.shape))
        rid = _next_region_id("permute")
        call = CallOp("aten_permute", [operands[0]], [result_type])
        _attach_region_id(call, rid)
        return DecompResult(ops=[call], result=call.res[0], region_ids=[rid])

    return decompose_transpose(operands, meta, node_name)


def decompose_addmm(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.addmm.default(bias, mat1, mat2, ...) -> matmul + bias add."""

    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))
    ops: list[Operation] = []
    region_ids: list[str] = []

    mm_empty = _make_empty(result_type)
    ops.append(mm_empty)

    matmul = MatmulOp(
        inputs=[operands[1], operands[2]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    matmul_rid = _next_region_id("matmul")
    _attach_region_id(matmul, matmul_rid)
    region_ids.append(matmul_rid)
    ops.append(matmul)

    bias_add = CallOp("aten_bias_add", [operands[0], matmul.results[0]], [result_type])
    add_rid = _next_region_id("add")
    _attach_region_id(bias_add, add_rid)
    region_ids.append(add_rid)
    ops.append(bias_add)

    return DecompResult(ops=ops, result=bias_add.res[0], region_ids=region_ids)


# ============================================================================
#  expansion — real-model coverage (smolVLA + Gemma-decode)
# ============================================================================
#
# Each entry below follows the established MVP pattern:
# - emit a real linalg op where cleanly supported (bmm, convolution as GEMM)
# - otherwise emit an opaque func.call (same pattern as decompose_gelu today)
# - set ``pattern_hint`` so downstream Phase 2 passes can reason about intent
#   even when the body is a black box.
#
# Destructive lowerings into full linalg.generic bodies land in a follow-up
# wave alongside the MVP-annotator → real-rewrite upgrade.


def _opaque_decomp(
    op_name: str,
    operands: list[SSAValue],
    meta: dict[str, Any],
    region_prefix: str,
    *,
    pattern_hint: str | None = None,
) -> DecompResult:
    """Shared helper for MVP decompositions that lower to a typed
    ``func.call @op_name`` carrying a ``compgen.region_id`` and
    optional pattern hint. Used for every op below whose full linalg
    body is deferred to the destructive-rewrite wave.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))
    rid = _next_region_id(region_prefix)
    call = CallOp(op_name, operands, [result_type])
    _attach_region_id(call, rid)
    return DecompResult(
        ops=[call],
        result=call.res[0],
        region_ids=[rid],
        pattern_hint=pattern_hint,
    )


def decompose_bmm(operands, meta, node_name):
    """aten.bmm.default(a, b) -> linalg.batch_matmul."""
    val: Any = meta["val"]
    result_type = TensorType(Float32Type(), _static_shape(val.shape))

    # We don't have a dedicated BatchMatmulOp in the xdsl dialect here;
    # emit as opaque call but with a canonical hint so the propagation
    # tags reach Recipe IR.
    return _opaque_decomp(
        "aten_bmm",
        operands,
        meta,
        "batch_matmul",
        pattern_hint="batch_matmul",
    )


def decompose_native_layer_norm(operands, meta, node_name):
    """aten.native_layer_norm.default(input, normalized_shape, weight, bias, eps).

    Emits a func.call with pattern_hint='layer_norm' so
    ``raise_special_ops`` picks it up.
    """
    return _opaque_decomp(
        "aten_native_layer_norm",
        operands,
        meta,
        "layer_norm",
        pattern_hint="layer_norm",
    )


def decompose_softmax(operands, meta, node_name):
    """aten._softmax.default(input, dim, half_to_float)."""
    return _opaque_decomp(
        "aten_softmax",
        operands,
        meta,
        "softmax",
        pattern_hint="softmax",
    )


def decompose_rsqrt(operands, meta, node_name):
    """aten.rsqrt.default(input) -> math.rsqrt (opaque call MVP)."""
    return _opaque_decomp(
        "aten_rsqrt",
        operands,
        meta,
        "rsqrt",
        pattern_hint="rsqrt",
    )


def decompose_pow_tensor_scalar(operands, meta, node_name):
    """aten.pow.Tensor_Scalar(input, exponent) - common in RMS norm (pow 2)."""
    return _opaque_decomp(
        "aten_pow",
        operands,
        meta,
        "pow",
        pattern_hint="pow_tensor_scalar",
    )


def decompose_mean_dim(operands, meta, node_name):
    """aten.mean.dim(input, dims, keepdim?, dtype?) -> linalg.reduce (MVP: opaque)."""
    return _opaque_decomp(
        "aten_mean_dim",
        operands,
        meta,
        "reduce",
        pattern_hint="reduce_mean",
    )


def decompose_convolution(operands, meta, node_name):
    """aten.convolution.default(input, weight, bias, stride, padding, ...).

    The full decomposition to linalg.conv_2d_nhwc_hwcf depends on the
    stride/padding/groups fields which live in operands[3:]. For MVP
    we tag pattern_hint='convolution' and keep the call opaque so the
    Phase 2 ``lower_conv_to_img2col`` pass can still annotate it.
    """
    # convolution may have >= 3 tensor operands (input, weight, optional bias);
    # pass only the tensor operands to the opaque call (scalars become func-level).
    tensor_operands = [op for op in operands[:3]]
    return _opaque_decomp(
        "aten_convolution",
        tensor_operands,
        meta,
        "convolution",
        pattern_hint="convolution",
    )


def decompose_embedding(operands, meta, node_name):
    """aten.embedding.default(weight, indices, ...) -> gather-style op (MVP: opaque)."""
    # weight + indices are the first two operands; scalar-flag kwargs beyond that.
    tensor_operands = operands[:2] if len(operands) >= 2 else operands
    return _opaque_decomp(
        "aten_embedding",
        tensor_operands,
        meta,
        "embedding",
        pattern_hint="embedding_lookup",
    )


def decompose_sigmoid(operands, meta, node_name):
    """aten.sigmoid.default(input) -> elementwise sigmoid (MVP: opaque)."""
    return _opaque_decomp(
        "aten_sigmoid",
        operands,
        meta,
        "elementwise",
        pattern_hint="sigmoid",
    )


def decompose_neg(operands, meta, node_name):
    """aten.neg.default(input) -> elementwise negation."""
    return _opaque_decomp(
        "aten_neg",
        operands,
        meta,
        "elementwise",
        pattern_hint="neg",
    )


def decompose_silu(operands, meta, node_name):
    """aten.silu.default(input) -> elementwise silu (used in many MLPs)."""
    return _opaque_decomp(
        "aten_silu",
        operands,
        meta,
        "elementwise",
        pattern_hint="silu",
    )


def decompose_sub_tensor(operands, meta, node_name):
    """aten.sub.Tensor(a, b, alpha?) -> elementwise sub."""
    return _opaque_decomp(
        "aten_sub",
        operands[:2],
        meta,
        "elementwise",
        pattern_hint="sub",
    )


def decompose_div_tensor(operands, meta, node_name):
    """aten.div.Tensor(a, b) -> elementwise div."""
    return _opaque_decomp(
        "aten_div",
        operands[:2],
        meta,
        "elementwise",
        pattern_hint="div",
    )


# ---- layout / structural (preserve shape metadata; no compute) ----


def decompose_view(operands, meta, node_name):
    """aten.view.default(input, shape) -> tensor reshape (opaque MVP)."""
    # view's shape is a scalar-list operand; keep only the tensor operand.
    return _opaque_decomp(
        "aten_view",
        operands[:1],
        meta,
        "layout",
        pattern_hint="view",
    )


def decompose_unsqueeze(operands, meta, node_name):
    """aten.unsqueeze.default(input, dim) -> insert a size-1 dim."""
    return _opaque_decomp(
        "aten_unsqueeze",
        operands[:1],
        meta,
        "layout",
        pattern_hint="unsqueeze",
    )


def decompose_expand(operands, meta, node_name):
    """aten.expand.default(input, sizes, implicit?) -> broadcast."""
    return _opaque_decomp(
        "aten_expand",
        operands[:1],
        meta,
        "layout",
        pattern_hint="expand",
    )


def decompose_cat(operands, meta, node_name):
    """aten.cat.default(tensors, dim?) -> tensor concat.

    Concat preserves the tensor inputs but not the scalar ``dim`` — the
    lowering wave will read the axis from node.args[1].
    """
    # The first positional arg is a list of tensors; FX expands that
    # list onto operands, so everything that's already an SSAValue stays.
    return _opaque_decomp(
        "aten_cat",
        operands,
        meta,
        "layout",
        pattern_hint="cat",
    )


def decompose_split_with_sizes(operands, meta, node_name):
    """aten.split_with_sizes.default(input, split_sizes, dim?).

    Returns a list of tensors; for MVP we emit a single opaque call
    with pattern_hint='split' — the FX graph's getitem ops disambiguate
    which chunk each downstream consumer needs.
    """
    return _opaque_decomp(
        "aten_split_with_sizes",
        operands[:1],
        meta,
        "layout",
        pattern_hint="split",
    )


def decompose_clone(operands, meta, node_name):
    """aten.clone.default(input) -> identity (metadata-only MVP)."""
    return _opaque_decomp(
        "aten_clone",
        operands[:1],
        meta,
        "identity",
        pattern_hint="clone",
    )


# --- layout / reshape / contiguous (production-readiness fill-ins) -----------


def decompose_contiguous(operands, meta, node_name):
    """aten.contiguous.default(input) -> layout no-op; same shape same dtype."""
    return _opaque_decomp(
        "aten_contiguous",
        operands[:1],
        meta,
        "layout",
        pattern_hint="contiguous",
    )


def decompose_transpose_int(operands, meta, node_name):
    """aten.transpose.int(input, dim0, dim1) -> shape-swapped tensor.

    ``dim0`` / ``dim1`` are scalar ints and don't appear as SSA
    operands. The result's shape comes from ``meta['val'].shape``
    which already reflects the transposition.
    """
    return _opaque_decomp(
        "aten_transpose",
        operands[:1],
        meta,
        "layout",
        pattern_hint="transpose",
    )


def decompose_matmul(operands, meta, node_name):
    """aten.matmul.default(a, b) -> linalg.matmul when rank-2, else opaque.

    Structural emission for the 2D × 2D case drops the opaque rate
    on real LLM fixtures. Higher-rank matmul stays opaque with
    ``pattern_hint="batch_matmul"`` so  dispatch handles it.
    """
    val = meta.get("val")
    shape = getattr(val, "shape", ()) if val is not None else ()
    out_rank = len(shape)

    if out_rank == 2 and len(operands) == 2:
        lhs, rhs = operands[0], operands[1]
        lhs_type = getattr(lhs, "type", None)
        rhs_type = getattr(rhs, "type", None)
        if (
            isinstance(lhs_type, TensorType)
            and isinstance(rhs_type, TensorType)
            and len(list(lhs_type.get_shape())) == 2
            and len(list(rhs_type.get_shape())) == 2
        ):
            result_type = TensorType(Float32Type(), _static_shape(shape))
            init = _make_empty(result_type)
            mm = MatmulOp(
                inputs=[lhs, rhs],
                outputs=[init.results[0]],
                res=[result_type],
            )
            rid = _next_region_id("matmul")
            _attach_region_id(mm, rid)
            return DecompResult(
                ops=[init, mm],
                result=mm.results[0],
                region_ids=[rid],
                pattern_hint="matmul",
            )

    hint = "batch_matmul" if out_rank > 2 else "matmul"
    return _opaque_decomp(
        "aten_matmul",
        operands[:2],
        meta,
        "matmul",
        pattern_hint=hint,
    )


# ---------------------------------------------------------------------------
# Wave 7 — TinyLlama opaque-tail closure: 10 new families that previously
# fell through to the unhinted opaque fallback. Each emits a typed
# func.call with a pattern_hint so the kernel selector recognises them
# as members of a known family (not unknown-tail).
# ---------------------------------------------------------------------------


def decompose_to_copy(operands, meta, node_name):
    """aten._to_copy.default — dtype/device cast. Emits a DTYPE_CAST kernel."""
    return _opaque_decomp("aten_to_dtype", operands[:1], meta, "cast", pattern_hint="dtype_cast")


def decompose_where_self(operands, meta, node_name):
    """aten.where.self(condition, x, y) — elementwise selection."""
    return _opaque_decomp("aten_where", operands[:3], meta, "select", pattern_hint="where")


def decompose_scalar_tensor(operands, meta, node_name):
    """aten.scalar_tensor.default(value, ...) — 0-rank constant fill."""
    return _opaque_decomp("aten_scalar_tensor", [], meta, "fill", pattern_hint="fill")


def decompose_full_like(operands, meta, node_name):
    """aten.full_like.default(input, fill_value, ...) — same-shape fill."""
    return _opaque_decomp("aten_full_like", operands[:1], meta, "fill", pattern_hint="fill")


def decompose_full(operands, meta, node_name):
    """aten.full.default(size, fill_value, ...) — explicit-shape fill."""
    return _opaque_decomp("aten_full", [], meta, "fill", pattern_hint="fill")


def decompose_arange(operands, meta, node_name):
    """aten.arange.start_step(start, end, step, ...) — index generator."""
    return _opaque_decomp("aten_arange", [], meta, "arange", pattern_hint="arange")


def decompose_logical_not(operands, meta, node_name):
    """aten.logical_not.default — pointwise boolean NOT."""
    return _opaque_decomp("aten_logical_not", operands[:1], meta, "logical", pattern_hint="logical_not")


def decompose_bitwise_and(operands, meta, node_name):
    """aten.bitwise_and.Tensor — pointwise bitwise AND."""
    return _opaque_decomp("aten_bitwise_and", operands[:2], meta, "bitwise", pattern_hint="bitwise_and")


def decompose_any_dim(operands, meta, node_name):
    """aten.any.dim(input, dim, keepdim) — boolean OR reduction along dim."""
    return _opaque_decomp("aten_any_dim", operands[:1], meta, "bool_reduce", pattern_hint="bool_reduce")


def decompose_index_tensor(operands, meta, node_name):
    """aten.index.Tensor — multi-dim gather. Operand 0 is source; index
    tensors arrive via meta['_fx_args'][1] as a list."""
    # Forward only the source SSA value; the index list is a Python list of
    # tensors that the kernel will receive via the contract metadata.
    return _opaque_decomp("aten_index", operands[:1], meta, "gather", pattern_hint="gather")


def decompose_compare(operands, meta, node_name):
    """aten.{eq,ne,le,lt,gt,ge}.{Tensor,Scalar} — pointwise comparison."""
    return _opaque_decomp("aten_compare", operands[:2], meta, "compare", pattern_hint="compare")


def decompose_cos(operands, meta, node_name):
    """aten.cos.default — pointwise cosine (RoPE)."""
    return _opaque_decomp("aten_cos", operands[:1], meta, "trig", pattern_hint="cos")


def decompose_sin(operands, meta, node_name):
    """aten.sin.default — pointwise sine (RoPE)."""
    return _opaque_decomp("aten_sin", operands[:1], meta, "trig", pattern_hint="sin")


def decompose_cumsum(operands, meta, node_name):
    """aten.cumsum.default — prefix sum along dim."""
    return _opaque_decomp("aten_cumsum", operands[:1], meta, "scan", pattern_hint="cumsum")


def decompose_slice_tensor(operands, meta, node_name):
    """aten.slice.Tensor(input, dim, start, end, step) -> tensor slice.

    All parameters except ``input`` are scalars. Result shape
    comes from meta.
    """
    return _opaque_decomp(
        "aten_slice",
        operands[:1],
        meta,
        "layout",
        pattern_hint="slice",
    )


# ============================================================================
#  C.2 — TorchAO quantized_decomposed + _weight_int*pack_mm
# W0.1: lowered to real compgen.quant ops (no more opaque func.call).
# ============================================================================


def _element_type_from_meta(meta: dict[str, Any]) -> Any:
    """Map the meta['val'].dtype (torch dtype) to an xDSL element type.

    Defaults to ``Float32Type`` when dtype is unavailable (e.g. in unit
    tests that stub out ``meta``).
    """
    from xdsl.dialects.builtin import (
        BFloat16Type,
        Float16Type,
        Float64Type,
        IntegerType,
    )

    val = meta.get("val")
    if val is None or not hasattr(val, "dtype"):
        return Float32Type()
    try:
        import torch
    except ImportError:
        return Float32Type()

    d = val.dtype
    if d == torch.float32:
        return Float32Type()
    if d == torch.float64:
        return Float64Type()
    if d == torch.float16:
        return Float16Type()
    if d == torch.bfloat16:
        return BFloat16Type()
    if d == torch.int8 or d == torch.uint8:
        return IntegerType(8)
    if d == torch.int16:
        return IntegerType(16)
    if d == torch.int32:
        return IntegerType(32)
    if d == torch.int64:
        return IntegerType(64)
    if hasattr(torch, "float8_e4m3fn") and d == torch.float8_e4m3fn:
        from compgen.ir.payload.types import Float8E4M3FNType

        return Float8E4M3FNType()
    if hasattr(torch, "float8_e5m2") and d == torch.float8_e5m2:
        from compgen.ir.payload.types import Float8E5M2Type

        return Float8E5M2Type()
    return Float32Type()


def _fx_arg(meta: dict[str, Any], index: int, default: Any = None) -> Any:
    """Read a scalar from the forwarded FX positional args, if present.

    ``import_fx.FXImporter`` attaches ``_fx_args`` (a tuple of raw FX
    call arguments) to ``meta`` before invoking a decomposition so
    scalar kwargs (group_size, axis, quant_min, quant_max) can flow
    through without becoming SSA operands.
    """
    args = meta.get("_fx_args") or ()
    if not isinstance(args, (tuple, list)):
        return default
    if index >= len(args):
        return default
    return args[index]


def _int_attr(value: int, width: int = 64) -> Any:
    """Shorthand for ``IntegerAttr(value, IntegerType(width))``."""
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    return IntegerAttr(int(value), IntegerType(width))


def _string_attr(value: str) -> Any:
    from xdsl.dialects.builtin import StringAttr

    return StringAttr(str(value))


def _torch_dtype_tag(val: Any) -> str:
    """Best-effort string tag for the result dtype.

    Used as the ``output_dtype`` / ``input_dtype`` informative property
    on quantize / dequantize ops.
    """
    if val is None or not hasattr(val, "dtype"):
        return ""
    return str(val.dtype).replace("torch.", "")


def decompose_quantize_per_tensor(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_tensor.default.

    FX signature: ``quantize_per_tensor(input, scale, zero_point,
    quant_min, quant_max, dtype)`` -- scale/zero_point are scalar
    tensor operands in the traced graph.
    """
    from compgen.ir.quant.ops import QuantizePerTensorOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    # Require at least input + scale + zero_point as SSA operands. In
    # the real FX path these all exist; in unit tests the caller passes
    # three tensor placeholders which we accept as-is.
    if len(operands) < 3:
        raise IndexError(
            f"decompose_quantize_per_tensor expects input + scale + zero_point (3 operands), got {len(operands)}"
        )

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 3)
    qmax = _fx_arg(meta, 4)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)
    tag = _torch_dtype_tag(val)
    if tag:
        properties["output_dtype"] = _string_attr(tag)

    rid = _next_region_id("quantize")
    op = QuantizePerTensorOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_tensor",
    )


def decompose_dequantize_per_tensor(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_tensor.default."""
    from compgen.ir.quant.ops import DequantizePerTensorOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_tensor expects input + scale + zero_point, got {len(operands)}")

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 3)
    qmax = _fx_arg(meta, 4)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerTensorOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_tensor",
    )


def decompose_quantize_per_channel(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_channel.default.

    FX signature: ``(input, scales, zero_points, axis, quant_min,
    quant_max, dtype)``. ``scales`` + ``zero_points`` are 1-D tensors
    along ``axis``.
    """
    from compgen.ir.quant.ops import QuantizePerChannelOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_quantize_per_channel expects input + scales + zero_points, got {len(operands)}")

    axis = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)
    tag = _torch_dtype_tag(val)
    if tag:
        properties["output_dtype"] = _string_attr(tag)

    rid = _next_region_id("quantize")
    op = QuantizePerChannelOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_channel",
    )


def decompose_dequantize_per_channel(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_channel.default."""
    from compgen.ir.quant.ops import DequantizePerChannelOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_channel expects input + scales + zero_points, got {len(operands)}")

    axis = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerChannelOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_channel",
    )


def decompose_quantize_per_group(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_group_along_last_dim.default.

    FX signature: ``(input, scales, zero_points, group_size, quant_min,
    quant_max, dtype)``.
    """
    from compgen.ir.quant.ops import QuantizePerGroupOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_quantize_per_group expects input + scales + zero_points, got {len(operands)}")

    gs = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        # Default group_size = 128 (TorchAO's most common setting) when
        # the FX arg is unavailable (e.g. under test fixtures).
        "group_size": _int_attr(gs if isinstance(gs, int) and gs > 0 else 128),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("quantize")
    op = QuantizePerGroupOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_group",
    )


def decompose_dequantize_per_group(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_group_along_last_dim.default."""
    from compgen.ir.quant.ops import DequantizePerGroupOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_group expects input + scales + zero_points, got {len(operands)}")

    gs = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "group_size": _int_attr(gs if isinstance(gs, int) and gs > 0 else 128),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerGroupOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_group",
    )


def decompose_weight_int8pack_mm(operands, meta, node_name):
    """aten._weight_int8pack_mm.default(input, weight_int8, scales)."""
    from compgen.ir.quant.ops import WeightInt8PackMMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_weight_int8pack_mm expects input + weight + scales, got {len(operands)}")

    rid = _next_region_id("quantized_matmul")
    op = WeightInt8PackMMOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int8pack_mm",
    )


def decompose_weight_int4pack_mm(operands, meta, node_name):
    """aten._weight_int4pack_mm.default(input, weight_int4, group_size, scales_and_zeros).

    ``group_size`` is a Python scalar in the FX signature so it is
    forwarded via ``meta['_fx_args'][2]`` rather than an SSA operand;
    in the traced graph the remaining tensor operands are ``[input,
    weight, scales_and_zeros]``. Tests may supply a 4-operand list
    with a placeholder in slot 2 which is skipped.
    """
    from compgen.ir.quant.ops import WeightInt4PackMMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) == 3:
        tensor_operands = [operands[0], operands[1], operands[2]]
    elif len(operands) >= 4:
        tensor_operands = [operands[0], operands[1], operands[3]]
    else:
        raise IndexError(f"decompose_weight_int4pack_mm expects >= 3 operands, got {len(operands)}")

    gs = _fx_arg(meta, 2)
    if not isinstance(gs, int) or gs <= 0:
        gs = 128  # TorchAO default
    # Snap to nearest valid group_size (32/64/128/256); the verifier
    # enforces this set. 128 is the TorchAO / Marlin default.
    valid = (32, 64, 128, 256)
    if gs not in valid:
        gs = 128

    rid = _next_region_id("quantized_matmul")
    op = WeightInt4PackMMOp(
        operands=tensor_operands,
        result_types=[result_type],
        properties={"group_size": _int_attr(gs)},
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int4pack_mm",
    )


def decompose_weight_int4pack_qm(operands, meta, node_name):
    """aten._weight_int4pack_qm.default — batched int4 packed GEMM."""
    from compgen.ir.quant.ops import WeightInt4PackQMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    tensor_operands = [o for o in operands if hasattr(o, "type") and isinstance(o.type, TensorType)]
    if len(tensor_operands) < 3:
        raise IndexError(f"decompose_weight_int4pack_qm expects >= 3 tensor operands, got {len(tensor_operands)}")

    gs = _fx_arg(meta, 2)
    if not isinstance(gs, int) or gs <= 0:
        gs = 128

    rid = _next_region_id("quantized_matmul")
    op = WeightInt4PackQMOp(
        operands=[tensor_operands[0], tensor_operands[1], tensor_operands[2]],
        result_types=[result_type],
        properties={"group_size": _int_attr(gs)},
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int4pack_qm",
    )


def decompose_choose_qparams_per_tensor(operands, meta, node_name):
    """aten._choose_qparams_per_tensor.default."""
    from compgen.ir.quant.ops import ChooseQParamsPerTensorOp

    # Produces (scale: f32, zero_point: i64) scalar tensors.
    scale_type = TensorType(Float32Type(), [])
    from xdsl.dialects.builtin import IntegerType

    zp_type = TensorType(IntegerType(64), [])

    if len(operands) < 1:
        raise IndexError("decompose_choose_qparams_per_tensor needs an input")

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 1)
    qmax = _fx_arg(meta, 2)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("choose_qparams")
    op = ChooseQParamsPerTensorOp(
        operands=[operands[0]],
        result_types=[scale_type, zp_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    # DecompResult carries a single ``result`` SSAValue; downstream
    # consumers that need both scale + zero_point read them off the
    # op directly via op.results[0] / op.results[1]. We pick
    # ``scale`` as the canonical ``result`` since that's what the FX
    # node's downstream ops most commonly consume first.
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="choose_qparams_per_tensor",
    )


def decompose_choose_qparams_per_channel(operands, meta, node_name):
    """aten._choose_qparams_per_channel.default."""
    from xdsl.dialects.builtin import IntegerType

    from compgen.ir.quant.ops import ChooseQParamsPerChannelOp

    val: Any = meta.get("val")
    # For channel qparams we produce two 1-D vectors of size C along
    # the channel axis. When the test fixture lacks a concrete shape
    # we fall back to rank-0 scalars so the op still verifies.
    shape = _static_shape(val.shape) if val is not None and hasattr(val, "shape") else []
    scale_type = TensorType(Float32Type(), shape)
    zp_type = TensorType(IntegerType(64), shape)

    if len(operands) < 1:
        raise IndexError("decompose_choose_qparams_per_channel needs an input")

    axis = _fx_arg(meta, 1)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 2)
    qmax = _fx_arg(meta, 3)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("choose_qparams")
    op = ChooseQParamsPerChannelOp(
        operands=[operands[0]],
        result_types=[scale_type, zp_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="choose_qparams_per_channel",
    )


# ============================================================================
# Decomposition table
# ============================================================================

DECOMPOSITION_TABLE: dict[str, DecompFn] = {
    # --- pre-wave-6 entries (kept) ---
    "aten.addmm.default": decompose_addmm,
    "aten.linear.default": decompose_linear,
    "aten.gelu.default": decompose_gelu,
    "aten.add.Tensor": decompose_add_tensor,
    "aten.mul.Tensor": decompose_mul_tensor,
    "aten.mm.default": decompose_mm,
    "aten.permute.default": decompose_permute,
    "aten.t.default": decompose_transpose,
    # --- wave 6: real-model coverage ---
    # compute / semantic
    "aten.bmm.default": decompose_bmm,
    "aten.native_layer_norm.default": decompose_native_layer_norm,
    "aten.layer_norm.default": decompose_native_layer_norm,
    "aten._softmax.default": decompose_softmax,
    "aten.softmax.int": decompose_softmax,
    "aten.rsqrt.default": decompose_rsqrt,
    "aten.pow.Tensor_Scalar": decompose_pow_tensor_scalar,
    "aten.mean.dim": decompose_mean_dim,
    "aten.convolution.default": decompose_convolution,
    "aten.embedding.default": decompose_embedding,
    "aten.sigmoid.default": decompose_sigmoid,
    "aten.neg.default": decompose_neg,
    "aten.silu.default": decompose_silu,
    "aten.sub.Tensor": decompose_sub_tensor,
    "aten.div.Tensor": decompose_div_tensor,
    # layout / structural
    "aten.view.default": decompose_view,
    "aten.unsqueeze.default": decompose_unsqueeze,
    "aten.expand.default": decompose_expand,
    "aten.cat.default": decompose_cat,
    "aten.split_with_sizes.default": decompose_split_with_sizes,
    "aten.clone.default": decompose_clone,
    # production-readiness fill-ins:
    "aten.contiguous.default": decompose_contiguous,
    "aten.transpose.int": decompose_transpose_int,
    "aten.transpose.default": decompose_transpose_int,
    "aten.matmul.default": decompose_matmul,
    "aten.slice.Tensor": decompose_slice_tensor,
    # --- wave 6 C.2: TorchAO quantized_decomposed + packed GEMMs ---
    "torch.ops.quantized_decomposed.quantize_per_tensor.default": decompose_quantize_per_tensor,
    "torch.ops.quantized_decomposed.dequantize_per_tensor.default": decompose_dequantize_per_tensor,
    "torch.ops.quantized_decomposed.quantize_per_channel.default": decompose_quantize_per_channel,
    "torch.ops.quantized_decomposed.dequantize_per_channel.default": decompose_dequantize_per_channel,
    "torch.ops.quantized_decomposed.quantize_per_group_along_last_dim.default": decompose_quantize_per_group,
    "torch.ops.quantized_decomposed.dequantize_per_group_along_last_dim.default": decompose_dequantize_per_group,
    "aten._weight_int8pack_mm.default": decompose_weight_int8pack_mm,
    "aten._weight_int4pack_mm.default": decompose_weight_int4pack_mm,
    "aten._weight_int4pack_qm.default": decompose_weight_int4pack_qm,
    "aten._choose_qparams_per_tensor.default": decompose_choose_qparams_per_tensor,
    "aten._choose_qparams_per_channel.default": decompose_choose_qparams_per_channel,
    # Short-form aliases some PyTorch versions emit
    "quantized_decomposed.quantize_per_tensor.default": decompose_quantize_per_tensor,
    "quantized_decomposed.dequantize_per_tensor.default": decompose_dequantize_per_tensor,
    "quantized_decomposed.quantize_per_channel.default": decompose_quantize_per_channel,
    "quantized_decomposed.dequantize_per_channel.default": decompose_dequantize_per_channel,
    "quantized_decomposed.quantize_per_group_along_last_dim.default": decompose_quantize_per_group,
    "quantized_decomposed.dequantize_per_group_along_last_dim.default": decompose_dequantize_per_group,
    # --- wave 7: TinyLlama opaque-tail closure (10 new families) ---
    "aten._to_copy.default": decompose_to_copy,
    "aten.where.self": decompose_where_self,
    "aten.scalar_tensor.default": decompose_scalar_tensor,
    "aten.full_like.default": decompose_full_like,
    "aten.full.default": decompose_full,
    "aten.arange.start_step": decompose_arange,
    "aten.arange.default": decompose_arange,
    "aten.logical_not.default": decompose_logical_not,
    "aten.bitwise_and.Tensor": decompose_bitwise_and,
    "aten.any.dim": decompose_any_dim,
    "aten.index.Tensor": decompose_index_tensor,
    # Comparisons + trig + scan (RoPE / mask construction)
    "aten.eq.Scalar": decompose_compare,
    "aten.eq.Tensor": decompose_compare,
    "aten.ne.Scalar": decompose_compare,
    "aten.ne.Tensor": decompose_compare,
    "aten.le.Scalar": decompose_compare,
    "aten.le.Tensor": decompose_compare,
    "aten.lt.Scalar": decompose_compare,
    "aten.lt.Tensor": decompose_compare,
    "aten.gt.Scalar": decompose_compare,
    "aten.gt.Tensor": decompose_compare,
    "aten.ge.Scalar": decompose_compare,
    "aten.ge.Tensor": decompose_compare,
    "aten.cos.default": decompose_cos,
    "aten.sin.default": decompose_sin,
    "aten.cumsum.default": decompose_cumsum,
}


__all__ = [
    "DECOMPOSITION_TABLE",
    "DecompFn",
    "DecompResult",
    "reset_region_counters",
]
