"""Tests for :mod:`compgen.agent.recipe_bridge_invent`.

Asserts that every slot-name → propose-op conversion:
  * constructs a Recipe-IR op that passes ``verify()``
  * serialises a ``ProposePayload`` that survives JSON round-trip
  * round-trips through ``recipe_to_mlir`` → ``mlir_to_recipe``

Unmapped slot names return ``None`` so unknown slots fall back to side-log only.
"""

from __future__ import annotations

import json

import pytest
from compgen.agent.recipe_bridge_invent import (
    proposal_to_recipe_op,
    supported_slot_names,
)
from compgen.ir.recipe.ops_propose import (
    ProposeDequantFusionOp,
    ProposeFusionOp,
    ProposeLayoutPlanOp,
    ProposeMegakernelSynthesisOp,
    ProposePayload,
)
from compgen.ir.recipe.serialize import mlir_to_recipe, recipe_to_mlir
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region


def test_supported_slots_exposes_minimum_set() -> None:
    names = supported_slot_names()
    assert "propose_fusion" in names
    assert "propose_megakernel_synthesis" in names
    assert "propose_layout_plan" in names
    assert "propose_dequant_fusion" in names


def test_unknown_slot_returns_none() -> None:
    op = proposal_to_recipe_op("not_a_real_slot", {"chosen": {}})
    assert op is None


# ---------------------------------------------------------------------------
# propose_fusion
# ---------------------------------------------------------------------------


def test_propose_fusion_from_grouped_regions() -> None:
    op = proposal_to_recipe_op(
        "propose_fusion",
        {
            "chosen": {
                "grouped_regions": ["r_3", "r_4"],
                "fusion_kind": "producer_consumer",
            },
            "candidates": [],
            "target_feature_justification": "HVX 32x32 tile alignment",
            "gate_result": {"status": "accepted"},
            "select_vs_invent": "invent",
        },
        iteration=7,
        llm_turn_id="turn_x",
    )
    assert isinstance(op, ProposeFusionOp)
    # verify() was called inside the bridge; check the shape landed.
    refs = [r.root_reference.data for r in op.grouped_regions.data]
    assert refs == ["r_3", "r_4"]
    payload = ProposePayload.from_json(op.payload.data)
    assert payload.chosen["grouped_regions"] == ["r_3", "r_4"]
    assert payload.llm_turn_id == "turn_x"
    assert payload.target_feature_justification == "HVX 32x32 tile alignment"


def test_propose_fusion_accepts_string_region() -> None:
    op = proposal_to_recipe_op(
        "propose_fusion",
        {"chosen": {"grouped_regions": "r_0"}, "select_vs_invent": "invent"},
    )
    assert isinstance(op, ProposeFusionOp)
    assert len(op.grouped_regions.data) == 1


def test_propose_fusion_rejects_empty_regions() -> None:
    with pytest.raises(ValueError):
        proposal_to_recipe_op(
            "propose_fusion",
            {"chosen": {"grouped_regions": []}, "select_vs_invent": "invent"},
        )


# ---------------------------------------------------------------------------
# propose_megakernel_synthesis
# ---------------------------------------------------------------------------


def test_propose_megakernel_synthesis_happy_path() -> None:
    op = proposal_to_recipe_op(
        "propose_megakernel_synthesis",
        {
            "chosen": {
                "megakernel_name": "gemma_block_mega",
                "fused_region_refs": ["r_2", "r_3", "r_4"],
                "event_tensor_decls": [{"name": "done", "shape": [1], "wait_count": 3, "scope": "block"}],
                "task_partition": {"sm_0": ["r_2"], "sm_1": ["r_3", "r_4"]},
            },
            "target_feature_justification": "persistent_kernels + semaphore_atomics",
            "select_vs_invent": "invent",
        },
        iteration=1,
    )
    assert isinstance(op, ProposeMegakernelSynthesisOp)
    regions = [r.root_reference.data for r in op.fused_region_refs.data]
    assert regions == ["r_2", "r_3", "r_4"]
    payload = ProposePayload.from_json(op.payload.data)
    assert payload.chosen["megakernel_name"] == "gemma_block_mega"


def test_propose_megakernel_synthesis_target_device_ref_optional() -> None:
    op = proposal_to_recipe_op(
        "propose_megakernel_synthesis",
        {
            "chosen": {
                "fused_region_refs": ["r_0"],
                "target_device_ref": "QRB5165",
            },
            "select_vs_invent": "invent",
        },
    )
    assert op.target_device_ref is not None
    assert op.target_device_ref.root_reference.data == "QRB5165"


# ---------------------------------------------------------------------------
# propose_layout_plan
# ---------------------------------------------------------------------------


def test_propose_layout_plan_requires_region_ref() -> None:
    with pytest.raises(ValueError):
        proposal_to_recipe_op(
            "propose_layout_plan",
            {"chosen": {}, "select_vs_invent": "invent"},
        )


def test_propose_layout_plan_happy_path() -> None:
    op = proposal_to_recipe_op(
        "propose_layout_plan",
        {
            "chosen": {"region_ref": "r_5", "layout": "blocked_32x32"},
            "select_vs_invent": "invent",
        },
    )
    assert isinstance(op, ProposeLayoutPlanOp)
    assert op.region_ref.root_reference.data == "r_5"
    assert ProposePayload.from_json(op.payload.data).chosen["layout"] == "blocked_32x32"


# ---------------------------------------------------------------------------
# propose_dequant_fusion
# ---------------------------------------------------------------------------


def test_propose_dequant_fusion_happy_path() -> None:
    op = proposal_to_recipe_op(
        "propose_dequant_fusion",
        {
            "chosen": {
                "region_ref": "r_1",
                "pattern": "int4_per_group_then_matmul",
                "tolerance_hint": 3e-2,
            },
            "select_vs_invent": "invent",
        },
    )
    assert isinstance(op, ProposeDequantFusionOp)
    assert op.region_ref.root_reference.data == "r_1"


# ---------------------------------------------------------------------------
# Round-trip through MLIR text
# ---------------------------------------------------------------------------


def test_propose_fusion_round_trips_through_mlir() -> None:
    """Construct, embed in a module, serialise, parse back, verify."""
    module = ModuleOp(Region([Block()]))
    op = proposal_to_recipe_op(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_a", "r_b"]},
            "select_vs_invent": "invent",
        },
        iteration=3,
    )
    module.body.block.add_op(op)

    mlir_text = recipe_to_mlir(module)
    assert "recipe.propose_fusion" in mlir_text
    assert '"r_a"' in mlir_text or "@r_a" in mlir_text

    reparsed = mlir_to_recipe(mlir_text)
    # Reparsed module's first op must still be ProposeFusionOp.
    body_ops = list(reparsed.body.block.ops)
    assert len(body_ops) == 1
    reparsed_op = body_ops[0]
    assert reparsed_op.name == "recipe.propose_fusion"
    assert "r_a" in reparsed_op.payload.data


def test_payload_json_survives_construction() -> None:
    """The payload StringAttr must be valid JSON with the canonical keys."""
    op = proposal_to_recipe_op(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_0"]},
            "candidates": [{"id": "c0"}, {"id": "c1"}],
            "target_feature_justification": "why",
            "gate_result": {"status": "accepted", "details": {"x": 1}},
            "select_vs_invent": "invent",
        },
    )
    decoded = json.loads(op.payload.data)
    for key in (
        "candidates",
        "chosen",
        "target_feature_justification",
        "gate_result",
        "select_vs_invent",
        "llm_turn_id",
    ):
        assert key in decoded
