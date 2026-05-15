"""Tests for W4.3 ``fuse_dequant_matmul``."""

from __future__ import annotations

from compgen.ir.payload.passes.rewrites.fuse_dequant_matmul import (
    FuseDequantMatmulConfig,
    FuseDequantMatmulStats,
    run_fuse_dequant_matmul,
)
from compgen.ir.quant import (
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
)
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _ft(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _make_module(
    *,
    dequant_kind: str,
    per_channel_axis: int = 1,
    group_size: int = 32,
    extra_user_on_dequant: bool = False,
) -> tuple[ModuleOp, MatmulOp]:
    """Build a dequant -> matmul module and optionally add an extra user."""
    B, K, N = 4, 128, 32
    xt = _ft([B, K])
    wi_t = _ft([K, N], IntegerType(8))
    wf_t = _ft([K, N])
    ot = _ft([B, N])

    x = EmptyOp([], xt)
    wi = EmptyOp([], wi_t)

    if dequant_kind == "per_tensor":
        scale = EmptyOp([], _ft([]))
        zp = EmptyOp([], _ft([], IntegerType(32)))
        dq = DequantizePerTensorOp(
            operands=[wi.results[0], scale.results[0], zp.results[0]],
            result_types=[wf_t],
        )
        dq_ops = [scale, zp, dq]
    elif dequant_kind == "per_channel":
        scales = EmptyOp([], _ft([N]))
        zps = EmptyOp([], _ft([N], IntegerType(32)))
        dq = DequantizePerChannelOp(
            operands=[wi.results[0], scales.results[0], zps.results[0]],
            result_types=[wf_t],
            properties={"axis": IntegerAttr(per_channel_axis, IntegerType(64))},
        )
        dq_ops = [scales, zps, dq]
    else:  # per_group
        scales = EmptyOp([], _ft([K // group_size, N]))
        zps = EmptyOp([], _ft([K // group_size, N], IntegerType(32)))
        dq = DequantizePerGroupOp(
            operands=[wi.results[0], scales.results[0], zps.results[0]],
            result_types=[wf_t],
            properties={"group_size": IntegerAttr(group_size, IntegerType(64))},
        )
        dq_ops = [scales, zps, dq]

    init = EmptyOp([], ot)
    mm = MatmulOp(inputs=[x.results[0], dq.result], outputs=[init.results[0]], res=[ot])
    body_ops = [x, wi, *dq_ops, init, mm]
    return_value = mm.res[0]

    if extra_user_on_dequant:
        # Second matmul consuming the dequant -> dequant has 2 uses.
        init2 = EmptyOp([], ot)
        mm2 = MatmulOp(
            inputs=[x.results[0], dq.result],
            outputs=[init2.results[0]],
            res=[ot],
        )
        body_ops.extend([init2, mm2])
        return_value = mm2.res[0]

    block = Block()
    for op in body_ops:
        block.add_op(op)
    block.add_op(ReturnOp(return_value))
    func = FuncOp("forward", FunctionType.from_lists([], [ot]), Region([block]))
    return ModuleOp([func]), mm


# --- happy path ------------------------------------------------------------


def test_per_channel_dequant_matmul_fuses():
    m, mm = _make_module(dequant_kind="per_channel")
    stats = run_fuse_dequant_matmul(m)
    assert stats.fusions_applied == 1
    assert mm.attributes["compgen.fused_dequant_kind"].data == "per_channel"
    assert mm.attributes["compgen.fused_dequant_side"].data == "rhs"
    assert_module_verifies(m)


def test_per_tensor_dequant_matmul_fuses():
    m, mm = _make_module(dequant_kind="per_tensor")
    stats = run_fuse_dequant_matmul(m)
    assert stats.fusions_applied == 1
    assert mm.attributes["compgen.fused_dequant_kind"].data == "per_tensor"


def test_per_group_with_allow_per_group_fuses():
    m, mm = _make_module(dequant_kind="per_group")
    cfg = FuseDequantMatmulConfig(reassoc_safe_only=True, allow_per_group=True)
    stats = run_fuse_dequant_matmul(m, config=cfg)
    assert stats.fusions_applied == 1
    assert mm.attributes["compgen.fused_dequant_kind"].data == "per_group"


def test_per_group_without_allow_per_group_is_skipped():
    m, mm = _make_module(dequant_kind="per_group")
    cfg = FuseDequantMatmulConfig(reassoc_safe_only=True, allow_per_group=False)
    stats = run_fuse_dequant_matmul(m, config=cfg)
    assert stats.fusions_applied == 0
    assert stats.skipped_reassoc_unsafe == 1
    assert "compgen.fused_dequant_kind" not in mm.attributes


# --- non-matching cases ---------------------------------------------------


def test_plain_matmul_is_skipped():
    t = _ft([4, 8])
    t2 = _ft([8, 16])
    t3 = _ft([4, 16])
    a = EmptyOp([], t)
    b = EmptyOp([], t2)
    init = EmptyOp([], t3)
    mm = MatmulOp(inputs=[a.results[0], b.results[0]], outputs=[init.results[0]], res=[t3])
    block = Block()
    for op in (a, b, init, mm):
        block.add_op(op)
    block.add_op(ReturnOp(mm.res[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t3]), Region([block]))
    m = ModuleOp([func])

    stats = run_fuse_dequant_matmul(m)
    assert stats.fusions_applied == 0
    assert stats.skipped_no_dequant_producer == 1


def test_multi_use_dequant_is_skipped():
    m, mm = _make_module(dequant_kind="per_channel", extra_user_on_dequant=True)
    stats = run_fuse_dequant_matmul(m)
    # Neither matmul gets the dequant absorbed because the dequant
    # has two users -> single-use gate fails.
    assert stats.fusions_applied == 0


# --- idempotence + stats --------------------------------------------------


def test_idempotent_second_run_is_noop():
    """After body fusion the matmul is replaced by a linalg.generic,
    so the second run sees zero matmul ops to process.

    The idempotence contract is: rerunning the pass does not mutate
    the module further.
    """
    m, _ = _make_module(dequant_kind="per_channel")
    first = run_fuse_dequant_matmul(m)
    assert first.fusions_applied == 1
    # First run emitted the fused body.
    assert first.fusions_body_inlined == 1
    # Second run sees no remaining matmul.
    second = run_fuse_dequant_matmul(m)
    assert second.matmuls_seen == 0
    assert second.fusions_applied == 0


def test_stats_initial_values():
    s = FuseDequantMatmulStats()
    assert s.matmuls_seen == 0
    assert s.fusions_applied == 0


# --- region_id preservation ----------------------------------------------


def test_matmul_attributes_preserved():
    m, mm = _make_module(dequant_kind="per_channel")
    mm.attributes["compgen.region_id"] = StringAttr("mm_test")
    mm.attributes["compgen._pattern_hint"] = StringAttr("gemm")
    run_fuse_dequant_matmul(m)
    assert mm.attributes["compgen.region_id"].data == "mm_test"
    assert mm.attributes["compgen._pattern_hint"].data == "gemm"


# --- allow_numerics_relaxation ---------------------------------------


def test_allow_numerics_relaxation_lets_per_group_through():
    m, mm = _make_module(dequant_kind="per_group")
    cfg = FuseDequantMatmulConfig(reassoc_safe_only=False)
    stats = run_fuse_dequant_matmul(m, config=cfg)
    assert stats.fusions_applied == 1


# --- real body fusion (not just tags) -----------------------------------


def test_per_channel_emits_fused_linalg_generic():
    """Real body fusion: matmul replaced by linalg.generic with
    inline dequant + mul + accumulate body.
    """
    m, _ = _make_module(dequant_kind="per_channel")
    stats = run_fuse_dequant_matmul(m)
    assert stats.fusions_body_inlined == 1
    # No matmul remains.
    assert count_ops(m, "linalg.matmul") == 0
    # A new linalg.generic was emitted with multiple inputs
    # (lhs, q_weight, scales, zeros).
    generics = [op for op in m.walk() if op.name == "linalg.generic"]
    assert len(generics) >= 1
    fused = generics[0]
    # 4 inputs (lhs, q, scales, zeros) + 1 output operand.
    assert len(fused.operands) == 5
    # Body contains arith.subi / sitofp / mulf / addf.
    body_ops = [o.name for o in fused.body.walk()]
    assert "arith.subi" in body_ops
    assert "arith.sitofp" in body_ops
    assert "arith.mulf" in body_ops
    assert "arith.addf" in body_ops
    assert_module_verifies(m)


def test_per_tensor_emits_fused_generic_with_scalar_scale():
    m, _ = _make_module(dequant_kind="per_tensor")
    stats = run_fuse_dequant_matmul(m)
    assert stats.fusions_body_inlined == 1
    # The scalar scale operand has rank-0.
    generic = next(op for op in m.walk() if op.name == "linalg.generic")
    scale_operand = generic.operands[2]
    assert scale_operand.type.get_shape() == ()


def test_per_group_keeps_tag_only_path():
    """Per-group dequant falls through to the tag-only path because
    the body fusion template doesn't yet handle per-group
    broadcasts (2-D scales tensor)."""
    m, mm = _make_module(dequant_kind="per_group")
    cfg = FuseDequantMatmulConfig(reassoc_safe_only=False)
    stats = run_fuse_dequant_matmul(m, config=cfg)
    assert stats.fusions_applied == 1
    assert stats.fusions_body_inlined == 0
    # Matmul still present with the tag.
    assert count_ops(m, "linalg.matmul") == 1
    assert "compgen.fused_dequant_kind" in mm.attributes


def test_body_fusion_preserves_accumulator_f32():
    m, _ = _make_module(dequant_kind="per_channel")
    run_fuse_dequant_matmul(m)
    generic = next(op for op in m.walk() if op.name == "linalg.generic")
    # The output operand (last input) has float32 element type.
    out_operand = generic.outputs[0]
    assert out_operand.type.get_element_type() == Float32Type()
