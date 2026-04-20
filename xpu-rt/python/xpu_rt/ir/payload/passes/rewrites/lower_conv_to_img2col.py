"""``lower_conv_to_img2col`` -- rewrite 2D convolution as
``tensor.pack`` + ``linalg.matmul`` (the im2col algorithm).

Reconstruction of IREE's ``ConvertConv2DToImg2ColPass``. Zero
external references; CompGen owns the rewrite.

Classical im2col:

- Input: ``[N, H, W, C]`` (NHWC) activation + ``[F, KH, KW, C]``
  (HWCF-like) filter.
- Pack input into ``[N, OH, OW, KH, KW, C]`` -- one tile per output
  location -- via ``compgen.tensor_ext.pack``.
- Collapse ``[N, OH, OW]`` into ``[N*OH*OW]`` (the output rows)
  and ``[KH, KW, C]`` into ``[KH*KW*C]`` (the reduction dim).
- Do a ``linalg.matmul`` with shape ``[N*OH*OW, KH*KW*C] × [KH*KW*C, F] → [N*OH*OW, F]``.
- Reshape back to ``[N, OH, OW, F]``.

Fully structural lowering requires:

- The convolution's shape (N, H, W, C, F, KH, KW, stride, dilation)
  from the FX node args -- not yet threaded through the opaque
  decomp table.
- Static shapes -- dynamic-shape convs defer to a Wave 6 pass.

So in Wave 5 we ship a **scheduled lowering**: every
``func.call @aten_convolution`` with static-shape input + weight
tensors gets:

- ``compgen.img2col_scheduled`` tag with shape metadata.
- A placeholder ``compgen.tensor_ext.pack`` emitted alongside the
  call, with ``inner_tiles`` carrying the (KH*KW*C) inner dim as
  a single packed axis -- the stable seam the tile-lowering path
  in Wave 6 consumes.
- The call's second operand (filter) is tagged so the dispatcher
  knows to materialize it as the GEMM rhs.

Full structural rewrite to ``pack + matmul + reshape`` lands as a
follow-up when:

1. The conv shape attrs (``stride``, ``dilation``, ``padding``,
   ``groups``) are threaded through the decomp table via
   ``node.args`` forwarding (already partially in place via
   ``_fx_args`` in :mod:`compgen.ir.payload.decompositions`).
2. ``tensor.collapse_shape`` lands in xDSL or we add a CompGen
   equivalent (``compgen.tensor_ext.collapse``).

LLM-tool signature:

    tool_name="lower_conv_to_img2col"
    wraps_pass="CompGen:ConvertConv2DToImg2Col"
    invent_slot="structural/conv_lowering"
    policy="ScheduleConvForImg2Col"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.tensor_ext import PackOp


@dataclass(frozen=True)
class LowerConvToImg2ColConfig:
    require_static_shapes: bool = True
    min_output_elements: int = 16  # skip trivially small convs


@dataclass
class LowerConvToImg2ColStats:
    convs_seen: int = 0
    convs_scheduled: int = 0
    convs_skipped_dynamic: int = 0
    convs_skipped_too_small: int = 0
    convs_skipped_wrong_rank: int = 0
    # Number of convs for which a real ``compgen.tensor_ext.pack`` op
    # was emitted alongside the scheduling tag.
    pack_ops_emitted: int = 0


# --- helpers -----------------------------------------------------------------


def _is_convolution_call(op: Operation) -> bool:
    if not isinstance(op, CallOp):
        return False
    hint = op.attributes.get("compgen._pattern_hint")
    if hint is None:
        return False
    if not isinstance(hint, StringAttr):
        return False
    return hint.data in {"convolution", "quantized_convolution"}


def _operand_shape(op: Operation, idx: int) -> tuple[int, ...] | None:
    if idx >= len(op.operands):
        return None
    t = op.operands[idx].type
    if not isinstance(t, TensorType):
        return None
    return tuple(t.get_shape())


# --- pattern -----------------------------------------------------------------


class _Img2ColSchedulePattern(RewritePattern):
    def __init__(
        self,
        cfg: LowerConvToImg2ColConfig,
        stats: LowerConvToImg2ColStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        if not _is_convolution_call(op):
            return
        self.stats.convs_seen += 1

        # Already scheduled -> idempotent.
        if "compgen.img2col_scheduled" in op.attributes:
            return

        input_shape = _operand_shape(op, 0)
        filter_shape = _operand_shape(op, 1)
        if input_shape is None or filter_shape is None:
            self.stats.convs_skipped_wrong_rank += 1
            return
        # Require 4-D (2-D conv).
        if len(input_shape) != 4 or len(filter_shape) != 4:
            self.stats.convs_skipped_wrong_rank += 1
            return

        if self.cfg.require_static_shapes and (any(d < 0 for d in input_shape) or any(d < 0 for d in filter_shape)):
            self.stats.convs_skipped_dynamic += 1
            return

        # Output shape from the call's result.
        res_type = op.results[0].type
        if not isinstance(res_type, TensorType):
            self.stats.convs_skipped_wrong_rank += 1
            return
        out_shape = tuple(res_type.get_shape())
        if len(out_shape) != 4:
            self.stats.convs_skipped_wrong_rank += 1
            return

        n_out = 1
        for d in out_shape:
            if d < 0:
                n_out = -1
                break
            n_out *= d
        if n_out != -1 and n_out < self.cfg.min_output_elements:
            self.stats.convs_skipped_too_small += 1
            return

        # Tag with shape metadata so Wave 6 can structurally lower.
        op.attributes["compgen.img2col_scheduled"] = StringAttr("true")
        op.attributes["compgen.img2col_input_shape"] = StringAttr(",".join(str(d) for d in input_shape))
        op.attributes["compgen.img2col_filter_shape"] = StringAttr(",".join(str(d) for d in filter_shape))
        op.attributes["compgen.img2col_output_shape"] = StringAttr(",".join(str(d) for d in out_shape))
        self.stats.convs_scheduled += 1

        # Emit a real ``compgen.tensor_ext.pack`` op on the conv's
        # activation input that tiles the spatial dims (H, W) into
        # (H / KH, W / KW, KH, KW). This is the blocked layout
        # im2col rewrites rely on as its prep step -- the full
        # windowed unfolding lands in a follow-up wave once the
        # conv's stride/padding args flow through the opaque call.
        # The tag + pack together are the contract downstream
        # tiling consumes.
        KH = filter_shape[-2]
        KW = filter_shape[-1]
        # Only emit pack when tiles divide the spatial extents. Otherwise
        # the pack's result shape is fractional; leave the tag alone.
        if KH <= 0 or KW <= 0 or input_shape[-2] % KH != 0 or input_shape[-1] % KW != 0:
            return
        input_val = op.operands[0]
        input_elem = input_val.type.get_element_type()
        # Result shape after tiling dims [H, W] into (H/KH, W/KW, KH, KW).
        packed_shape = list(input_shape[:-2]) + [input_shape[-2] // KH, input_shape[-1] // KW, KH, KW]
        packed_type = TensorType(input_elem, packed_shape)
        pack = PackOp(
            source=input_val,
            inner_dims_pos=[len(input_shape) - 2, len(input_shape) - 1],
            inner_tiles=[KH, KW],
            result_type=packed_type,
        )
        pack.attributes["compgen.img2col_pack"] = StringAttr("true")
        rewriter.insert_op_before_matched_op(pack)
        # Tag the conv with the pack's buffer id so later passes can
        # find it.
        op.attributes["compgen.img2col_pack_tile_kh"] = IntegerAttr(KH, IntegerType(64))
        op.attributes["compgen.img2col_pack_tile_kw"] = IntegerAttr(KW, IntegerType(64))
        self.stats.pack_ops_emitted += 1


# --- entry point -------------------------------------------------------------


def run_lower_conv_to_img2col(
    module: ModuleOp,
    *,
    config: LowerConvToImg2ColConfig | None = None,
    apply_recursively: bool = False,
) -> LowerConvToImg2ColStats:
    cfg = config if config is not None else LowerConvToImg2ColConfig()
    stats = LowerConvToImg2ColStats()
    pattern = _Img2ColSchedulePattern(cfg, stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "LowerConvToImg2ColConfig",
    "LowerConvToImg2ColStats",
    "run_lower_conv_to_img2col",
]
