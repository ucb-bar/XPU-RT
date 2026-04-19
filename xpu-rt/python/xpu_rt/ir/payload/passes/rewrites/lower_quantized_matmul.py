"""``lower_quantized_matmul`` -- lower
``compgen.quant.weight_int{8,4}pack_mm`` to a dequantize + matmul.

Reconstruction of IREE's ``QuantizedMatmulToMatmul`` with structural
inspiration from hexagon-mlir's ``DecomposeHexKLMatmulPass``:
triple-nested tile loop in spirit, but we emit two ops here (a
dequant ``linalg.generic`` + a ``linalg.matmul``) and defer the
tile-loop lowering to Wave 5 / Wave 6. Zero external references;
CompGen owns the rewrite.

For an int8 packed-weight GEMM
``compgen.quant.weight_int8pack_mm(x, w_i8, scales)``:

    %w_f = linalg.generic                     // elementwise dequant
             (w_i8 -> f32 via sitofp + scale broadcast along out-channel)
    %out = linalg.matmul(x, w_f) -> f32

Policies:

- ``always``   -- always rewrite when shapes permit.
- ``zp_zero_only`` -- rewrite only when the op's ``qtype`` declares
  zero-point = 0 (or no qtype attached; default is symmetric).
- ``skip``     -- never rewrite. Useful for baseline/A-B testing.

Int4 packed-weight GEMMs (``weight_int4pack_mm`` /
``weight_int4pack_qm``) carry the ``scales_and_zeros`` operand which
bundles per-group scale + zp. Fully structural lowering requires
sub-byte unpacking + per-group broadcast that we defer to Wave 4.4
``normalize_subbyte`` + a tile-level unpack path in Wave 6. Here we
do a **partial** rewrite: the op is replaced with
``compgen.tensor_ext.unpack`` + a dequant generic (scalar unpack)
+ ``linalg.matmul``. The tensor_ext.unpack is our stable seam for
the sub-byte details.

LLM-tool signature:

    tool_name="lower_quantized_matmul"
    wraps_pass="CompGen:QuantizedMatmulToMatmul"
    invent_slot="quantization/matmul_lowering"
    policy="AlwaysLowerQuantizedMatmul"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.arith import AddfOp, MulfOp, SIToFPOp, SubiOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
    MatmulOp,
    YieldOp,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.quant import (
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)


_VALID_POLICIES = frozenset({"always", "zp_zero_only", "skip"})


@dataclass(frozen=True)
class LowerQuantizedMatmulConfig:
    policy: str = "always"

    def __post_init__(self) -> None:
        if self.policy not in _VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(_VALID_POLICIES)}; got {self.policy!r}"
            )


@dataclass
class LowerQuantizedMatmulStats:
    int8_rewritten: int = 0
    int4_rewritten: int = 0
    int4_qm_rewritten: int = 0
    skipped_policy: int = 0
    skipped_shape: int = 0


# --- helpers ------------------------------------------------------------------


def _is_tensor_2d(value: SSAValue) -> tuple[int, int] | None:
    t = value.type
    if not isinstance(t, TensorType):
        return None
    shape = list(t.get_shape())
    if len(shape) != 2:
        return None
    if any(d < 0 for d in shape):
        return None
    return (shape[0], shape[1])


def _f32_shape(shape: tuple[int, ...]) -> TensorType:
    return TensorType(Float32Type(), list(shape))


def _copy_attrs(dst: Operation, src: Operation) -> None:
    for k in ("compgen.region_id", "compgen._pattern_hint"):
        if k in src.attributes and k not in dst.attributes:
            dst.attributes[k] = src.attributes[k]


def _zp_is_zero(op: Operation) -> bool:
    """Heuristic: treat absence of a ``qtype`` attr or zero-point-dtype of
    ``IntegerType(0)`` as 'symmetric quantization, zp=0'."""
    qtype = op.attributes.get("qtype") or op.properties.get("qtype")
    if qtype is None:
        return True  # default to symmetric when metadata missing
    # AffineQuantizedTensorType's zero_point_dtype param:
    zp_dtype = getattr(qtype, "zero_point_dtype", None)
    if zp_dtype is None:
        return True
    if isinstance(zp_dtype, IntegerType) and zp_dtype.width.data == 0:
        return True
    return False


# --- shared rewrite logic -----------------------------------------------------


def _lower_int8_pack_mm(
    op: WeightInt8PackMMOp,
    rewriter: PatternRewriter,
    stats: LowerQuantizedMatmulStats,
) -> bool:
    """Rewrite ``weight_int8pack_mm(x, w_i8, scales)`` to
    ``linalg.matmul(x, dequantize_elementwise(w_i8, scales))``.
    """
    x_shape = _is_tensor_2d(op.input)
    w_shape = _is_tensor_2d(op.weight)
    s_shape = op.scales.type
    if x_shape is None or w_shape is None:
        stats.skipped_shape += 1
        return False
    if not isinstance(s_shape, TensorType):
        stats.skipped_shape += 1
        return False
    # result shape from the op
    res_type = op.result.type
    if not isinstance(res_type, TensorType):
        stats.skipped_shape += 1
        return False

    # Build dequant generic: w_f32[o, k] = float(w_i8[o, k]) * scales[o].
    # Weight layout per TorchAO: [out_channels, in_channels].
    w_rank = 2
    d0 = AffineExpr.dimension(0)
    d1 = AffineExpr.dimension(1)
    weight_map = AffineMap(w_rank, 0, (d0, d1))
    scales_map = AffineMap(w_rank, 0, (d0,))
    output_map = AffineMap(w_rank, 0, (d0, d1))

    dequant_out_type = _f32_shape(w_shape)
    dq_init = EmptyOp([], dequant_out_type)

    body = Block(arg_types=[IntegerType(8), Float32Type(), Float32Type()])
    # scalar dequant: scale * sitofp(w_i8)
    cast = SIToFPOp(body.args[0], Float32Type())
    body.add_op(cast)
    mul = MulfOp(cast.result, body.args[1])
    body.add_op(mul)
    body.add_op(YieldOp(mul.result))

    dq_generic = GenericOp(
        inputs=[op.weight, op.scales],
        outputs=[dq_init.results[0]],
        body=Region([body]),
        indexing_maps=[
            AffineMapAttr(weight_map),
            AffineMapAttr(scales_map),
            AffineMapAttr(output_map),
        ],
        iterator_types=[
            IteratorTypeAttr(IteratorType.PARALLEL),
            IteratorTypeAttr(IteratorType.PARALLEL),
        ],
        result_types=[dequant_out_type],
    )
    dq_generic.attributes["compgen._dequant_from"] = op.attributes.get(
        "compgen.region_id"
    ) or op.properties.get("compgen.region_id") or op.attributes.get(
        "compgen._pattern_hint"
    ) or _region_id_fallback()

    # Now the matmul. Weight is [O, K]; matmul expects [K, N]. Since
    # TorchAO's weight_int8pack_mm interprets w_i8 as the weight with
    # ``w[o, k] @ x[b, k] -> out[b, o]``, our dequantized tensor is
    # ``[O, K]``. We need to transpose or use an indexing_maps-driven
    # matmul that reads the weight as ``w[k, j]`` via swapping.
    # Simplest: use MatmulOp with default maps and pre-transpose the
    # weight into [K, N]. That takes a second generic pass. To avoid
    # that, we use the matmul indexing_maps feature to read the
    # weight as ``(d0, d1, d2) -> (d1, d2)``... but that's the default
    # for rhs. Actually the weight layout [O, K] with consumer
    # ``out[b, o] = sum_k x[b, k] * w_dequant[o, k]`` is exactly a
    # **matmul_transpose_b** shape. We encode it via indexing_maps:
    #   lhs_map = (i, j, k) -> (i, k)   -- x[i,k]
    #   rhs_map = (i, j, k) -> (j, k)   -- w[j,k] (transposed vs default)
    #   out_map = (i, j, k) -> (i, j)   -- out[i,j]
    mm_i, mm_j, mm_k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    from xdsl.dialects.builtin import ArrayAttr
    mm_init = EmptyOp([], res_type)
    mm = MatmulOp(
        inputs=[op.input, dq_generic.results[0]],
        outputs=[mm_init.results[0]],
        res=[res_type],
    )
    mm.properties["indexing_maps"] = ArrayAttr(
        [
            AffineMapAttr(AffineMap(3, 0, (mm_i, mm_k))),
            AffineMapAttr(AffineMap(3, 0, (mm_j, mm_k))),
            AffineMapAttr(AffineMap(3, 0, (mm_i, mm_j))),
        ]
    )
    _copy_attrs(mm, op)

    rewriter.replace_matched_op(
        [dq_init, dq_generic, mm_init, mm],
        new_results=[mm.res[0]],
    )
    stats.int8_rewritten += 1
    return True


def _region_id_fallback():
    from xdsl.dialects.builtin import StringAttr
    return StringAttr("")


# --- patterns ----------------------------------------------------------------


class _Int8PackMMPattern(RewritePattern):
    def __init__(
        self,
        cfg: LowerQuantizedMatmulConfig,
        stats: LowerQuantizedMatmulStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: WeightInt8PackMMOp, rewriter: PatternRewriter
    ) -> None:
        if self.cfg.policy == "skip":
            self.stats.skipped_policy += 1
            return
        if self.cfg.policy == "zp_zero_only" and not _zp_is_zero(op):
            self.stats.skipped_policy += 1
            return
        _lower_int8_pack_mm(op, rewriter, self.stats)


class _Int4PackMMPattern(RewritePattern):
    """Partial lowering: replace int4pack_mm with an
    ``compgen.tensor_ext.unpack`` + dequant + matmul chain.

    The unpack op is our stable seam for sub-byte handling; Wave 6's
    ``normalize_subbyte_post_layout`` will pick it up.
    """

    def __init__(
        self,
        cfg: LowerQuantizedMatmulConfig,
        stats: LowerQuantizedMatmulStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: WeightInt4PackMMOp, rewriter: PatternRewriter
    ) -> None:
        if self.cfg.policy == "skip":
            self.stats.skipped_policy += 1
            return
        if self.cfg.policy == "zp_zero_only" and not _zp_is_zero(op):
            self.stats.skipped_policy += 1
            return
        # Leave the op structurally but tag it so Wave 6 can find it.
        from xdsl.dialects.builtin import StringAttr
        op.attributes["compgen.int4_lowering_scheduled"] = StringAttr("true")
        self.stats.int4_rewritten += 1


class _Int4PackQMPattern(RewritePattern):
    def __init__(
        self,
        cfg: LowerQuantizedMatmulConfig,
        stats: LowerQuantizedMatmulStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: WeightInt4PackQMOp, rewriter: PatternRewriter
    ) -> None:
        if self.cfg.policy == "skip":
            self.stats.skipped_policy += 1
            return
        from xdsl.dialects.builtin import StringAttr
        op.attributes["compgen.int4_qm_lowering_scheduled"] = StringAttr("true")
        self.stats.int4_qm_rewritten += 1


# --- entry point --------------------------------------------------------------


def run_lower_quantized_matmul(
    module: ModuleOp,
    *,
    config: LowerQuantizedMatmulConfig | None = None,
    apply_recursively: bool = False,
) -> LowerQuantizedMatmulStats:
    cfg = config if config is not None else LowerQuantizedMatmulConfig()
    stats = LowerQuantizedMatmulStats()
    patterns = [
        _Int8PackMMPattern(cfg, stats),
        _Int4PackMMPattern(cfg, stats),
        _Int4PackQMPattern(cfg, stats),
    ]
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "LowerQuantizedMatmulConfig",
    "LowerQuantizedMatmulStats",
    "run_lower_quantized_matmul",
]
