"""Tests for the P3 propose-op family in Recipe IR."""

from __future__ import annotations

import pytest
from compgen.ir.recipe.dialect import ALL_OPS, Recipe
from compgen.ir.recipe.ops_propose import (
    _PROPOSE_OPS,
    ProposeBufferLifetimePlanOp,
    ProposeCollectivePipelineOp,
    ProposeDequantFusionOp,
    ProposeFusionOp,
    ProposeLayoutPlanOp,
    ProposeMegakernelSynthesisOp,
    ProposeMultiOutputFusionOp,
    ProposeNumericsPlanOp,
    ProposePayload,
    ProposePeepholePatternOp,
    ProposeRematerializationPlanOp,
    ProposeSchedulingPolicyOp,
    ProposeShardingPlanOp,
)
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr, i64
from xdsl.utils.exceptions import VerifyException


def _valid_payload() -> str:
    return ProposePayload(
        candidates=[{"name": "a", "est_cost": 1.0}],
        chosen={"name": "a", "est_cost": 1.0},
        target_feature_justification="supported_kernel_families[?family=='gemm']",
        gate_result={"status": "accepted", "details": {}},
        select_vs_invent="invent",
    ).to_json()


def test_all_propose_ops_in_dialect() -> None:
    for op_cls in _PROPOSE_OPS:
        assert op_cls in ALL_OPS


def test_payload_roundtrip() -> None:
    p = ProposePayload(
        candidates=[{"x": 1}],
        chosen={"x": 1},
        target_feature_justification="justif",
        gate_result={"status": "accepted"},
        select_vs_invent="invent",
    )
    j = p.to_json()
    p2 = ProposePayload.from_json(j)
    assert p2.chosen == {"x": 1}
    assert p2.target_feature_justification == "justif"
    assert p2.select_vs_invent == "invent"


def test_propose_layout_plan_verifies() -> None:
    op = ProposeLayoutPlanOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "payload": StringAttr(_valid_payload()),
        },
    )
    op.verify_()
    recovered = op.get_payload()
    assert recovered.chosen["name"] == "a"


def test_propose_layout_plan_rejects_malformed_payload() -> None:
    op = ProposeLayoutPlanOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "payload": StringAttr('{"candidates": []}'),  # missing chosen
        },
    )
    with pytest.raises(VerifyException, match="chosen"):
        op.verify_()


def test_propose_layout_plan_rejects_bad_json() -> None:
    op = ProposeLayoutPlanOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "payload": StringAttr("not json"),
        },
    )
    with pytest.raises(VerifyException, match="not valid JSON"):
        op.verify_()


def test_propose_layout_plan_rejects_bad_select_vs_invent() -> None:
    bad = ProposePayload(
        chosen={"x": 1},
        select_vs_invent="bogus",  # type: ignore[arg-type]
    ).to_json()
    op = ProposeLayoutPlanOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "payload": StringAttr(bad),
        },
    )
    with pytest.raises(VerifyException, match="select_vs_invent"):
        op.verify_()


def test_propose_fusion_requires_grouped_region() -> None:
    op = ProposeFusionOp.create(
        properties={
            "grouped_regions": ArrayAttr([]),
            "payload": StringAttr(_valid_payload()),
        },
    )
    with pytest.raises(VerifyException, match="grouped_regions"):
        op.verify_()


def test_propose_fusion_accepts_single_region() -> None:
    op = ProposeFusionOp.create(
        properties={
            "grouped_regions": ArrayAttr([SymbolRefAttr("r0")]),
            "payload": StringAttr(_valid_payload()),
        },
    )
    op.verify_()


def test_propose_multi_output_requires_min_outputs() -> None:
    bad = ProposeMultiOutputFusionOp.create(
        properties={
            "grouped_regions": ArrayAttr([SymbolRefAttr("r0")]),
            "producer_output_count": IntegerAttr(1, i64),
            "payload": StringAttr(_valid_payload()),
        },
    )
    with pytest.raises(VerifyException, match="producer_output_count"):
        bad.verify_()
    ok = ProposeMultiOutputFusionOp.create(
        properties={
            "grouped_regions": ArrayAttr([SymbolRefAttr("r0"), SymbolRefAttr("r1")]),
            "producer_output_count": IntegerAttr(2, i64),
            "payload": StringAttr(_valid_payload()),
        },
    )
    ok.verify_()


def test_propose_remat_plan_memory_budget_positive() -> None:
    bad = ProposeRematerializationPlanOp.create(
        properties={
            "plan_ref": SymbolRefAttr("plan0"),
            "memory_budget_bytes": IntegerAttr(0, i64),
            "payload": StringAttr(_valid_payload()),
        },
    )
    with pytest.raises(VerifyException, match="memory_budget_bytes"):
        bad.verify_()
    ok = ProposeRematerializationPlanOp.create(
        properties={
            "plan_ref": SymbolRefAttr("plan0"),
            "memory_budget_bytes": IntegerAttr(1024, i64),
            "payload": StringAttr(_valid_payload()),
        },
    )
    ok.verify_()


def test_propose_collective_pipeline_direction_enum() -> None:
    bad = ProposeCollectivePipelineOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "direction": StringAttr("sideways"),
            "payload": StringAttr(_valid_payload()),
        },
    )
    with pytest.raises(VerifyException, match="direction"):
        bad.verify_()
    ok = ProposeCollectivePipelineOp.create(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "direction": StringAttr("forward"),
            "payload": StringAttr(_valid_payload()),
        },
    )
    ok.verify_()


def test_other_propose_ops_verify_on_valid_payload() -> None:
    cases = [
        (
            ProposePeepholePatternOp,
            {
                "region_ref": SymbolRefAttr("r0"),
                "pattern_class": StringAttr("attention_variant"),
                "payload": StringAttr(_valid_payload()),
            },
        ),
        (
            ProposeNumericsPlanOp,
            {
                "region_ref": SymbolRefAttr("r0"),
                "payload": StringAttr(_valid_payload()),
            },
        ),
        (
            ProposeDequantFusionOp,
            {
                "region_ref": SymbolRefAttr("r0"),
                "payload": StringAttr(_valid_payload()),
            },
        ),
        (
            ProposeShardingPlanOp,
            {
                "module_ref": SymbolRefAttr("main"),
                "payload": StringAttr(_valid_payload()),
            },
        ),
        (
            ProposeBufferLifetimePlanOp,
            {
                "plan_ref": SymbolRefAttr("plan0"),
                "payload": StringAttr(_valid_payload()),
            },
        ),
    ]
    for cls, props in cases:
        op = cls.create(properties=props)
        op.verify_()


def test_recipe_dialect_includes_propose_family() -> None:
    # Recipe dialect should register all propose ops.
    registered = {op.name for op in Recipe.operations}
    for op_cls in _PROPOSE_OPS:
        assert op_cls.name in registered, f"{op_cls.name} not in Recipe dialect"


# ---------------------------------------------------------------------------
# Event Tensor Compiler (ETC) integration: propose-megakernel + propose-policy
# ---------------------------------------------------------------------------


def _mk_megakernel_payload() -> str:
    return ProposePayload(
        candidates=[
            {
                "megakernel_name": "mm_rs_static",
                "fused_region_refs": ["region_mm", "region_rs"],
                "event_tensor_decls": [{"name": "E", "shape": [4], "wait_count": 2, "scope": "device"}],
            }
        ],
        chosen={
            "megakernel_name": "mm_rs_static",
            "fused_region_refs": ["region_mm", "region_rs"],
            "event_tensor_decls": [{"name": "E", "shape": [4], "wait_count": 2, "scope": "device"}],
            "task_partition": {"region_mm": [4], "region_rs": [4]},
        },
        target_feature_justification=(
            "B200 capabilities.persistent_kernels=true and capabilities.semaphore_atomics=true; ETC abstraction"
        ),
        gate_result={"status": "accepted", "details": {}},
        select_vs_invent="invent",
    ).to_json()


def test_propose_megakernel_synthesis_verifies() -> None:
    op = ProposeMegakernelSynthesisOp.create(
        properties={
            "fused_region_refs": ArrayAttr([SymbolRefAttr("region_mm"), SymbolRefAttr("region_rs")]),
            "payload": StringAttr(_mk_megakernel_payload()),
        },
    )
    op.verify_()
    chosen = op.get_payload().chosen
    assert chosen["megakernel_name"] == "mm_rs_static"


def test_propose_megakernel_requires_at_least_one_region() -> None:
    op = ProposeMegakernelSynthesisOp.create(
        properties={
            "fused_region_refs": ArrayAttr([]),
            "payload": StringAttr(_mk_megakernel_payload()),
        },
    )
    with pytest.raises(VerifyException, match="fused_region_refs"):
        op.verify_()


def _mk_policy_payload(policy: str = "static") -> str:
    return ProposePayload(
        candidates=[{"policy": "static"}, {"policy": "dynamic"}],
        chosen={"policy": policy, "sm_count": 108, "early_push": False},
        target_feature_justification="static is appropriate for predictable GEMM+RS",
        gate_result={"status": "accepted", "details": {}},
        select_vs_invent="select",
    ).to_json()


def test_propose_scheduling_policy_static_verifies() -> None:
    op = ProposeSchedulingPolicyOp.create(
        properties={
            "megakernel_ref": SymbolRefAttr("mm_rs_static"),
            "payload": StringAttr(_mk_policy_payload("static")),
        },
    )
    op.verify_()
    assert op.get_payload().chosen["policy"] == "static"


def test_propose_scheduling_policy_dynamic_verifies() -> None:
    op = ProposeSchedulingPolicyOp.create(
        properties={
            "megakernel_ref": SymbolRefAttr("moe"),
            "payload": StringAttr(_mk_policy_payload("dynamic")),
        },
    )
    op.verify_()


def test_propose_scheduling_policy_rejects_unknown_policy() -> None:
    op = ProposeSchedulingPolicyOp.create(
        properties={
            "megakernel_ref": SymbolRefAttr("mm_rs_static"),
            "payload": StringAttr(_mk_policy_payload("greedy")),
        },
    )
    with pytest.raises(VerifyException, match="chosen.policy"):
        op.verify_()
