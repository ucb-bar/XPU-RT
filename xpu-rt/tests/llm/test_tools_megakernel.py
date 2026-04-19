"""Tests for the Phase-4 megakernel LLM tools + target coverage."""

from __future__ import annotations

from compgen.llm.registry import get_registry
from compgen.llm.target_coverage import cost_weight_for, get_coverage
from compgen.llm.tools.megakernel import (
    pick_scheduler_strategy,
    propose_megakernel_layout,
    register,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_both_megakernel_tools_are_registered_in_phase_4() -> None:
    register()  # idempotent
    reg = get_registry()
    assert reg.lookup_tool("propose_megakernel_layout", phase=4) is not None
    assert reg.lookup_tool("pick_scheduler_strategy", phase=4) is not None


def test_register_is_idempotent() -> None:
    first = set(register())
    second = set(register())
    assert second == set()
    # `first` may be empty if a previous test already registered.
    assert first <= {"propose_megakernel_layout", "pick_scheduler_strategy"}


# ---------------------------------------------------------------------------
# propose_megakernel_layout
# ---------------------------------------------------------------------------


def test_propose_megakernel_layout_returns_payload_with_named_kernel() -> None:
    result = propose_megakernel_layout.impl(
        candidate_regions=["region_mm", "region_rs"],
        inter_region_edges=[{"shape": [4], "wait_count": 2}],
        task_shape=[4],
    )
    assert result["status"] == "ok"
    assert result["megakernel_name"] == "mk_region_mm_region_rs"
    chosen = result["payload"]["chosen"]
    assert chosen["fused_region_refs"] == ["region_mm", "region_rs"]
    assert chosen["event_tensor_decls"][0]["wait_count"] == 2


def test_propose_megakernel_layout_supports_explicit_name() -> None:
    result = propose_megakernel_layout.impl(
        candidate_regions=["a", "b"],
        megakernel_name="qwen3_decode_step",
    )
    assert result["megakernel_name"] == "qwen3_decode_step"


def test_propose_megakernel_layout_metadata_advertises_phase_and_cost() -> None:
    assert propose_megakernel_layout.phase == 4
    assert propose_megakernel_layout.autocomp_cost_impact == "very_high"
    assert propose_megakernel_layout.wraps_pass == "recipe.propose_megakernel_synthesis"


# ---------------------------------------------------------------------------
# pick_scheduler_strategy
# ---------------------------------------------------------------------------


def test_pick_scheduler_strategy_defaults_to_static() -> None:
    result = pick_scheduler_strategy.impl(
        megakernel_ref="mk_test",
        has_data_dependent_edges=False,
    )
    assert result["status"] == "ok"
    assert result["policy"] == "static"
    assert result["payload"]["chosen"]["sm_count"] == 108


def test_pick_scheduler_strategy_picks_dynamic_for_data_dep_edges() -> None:
    result = pick_scheduler_strategy.impl(
        megakernel_ref="moe",
        has_data_dependent_edges=True,
        data_dependent_edges=["topk", "exp_indptr"],
        early_push=True,
    )
    assert result["policy"] == "dynamic"
    chosen = result["payload"]["chosen"]
    assert chosen["dynamic_features"] == ["topk", "exp_indptr"]
    assert chosen["early_push"] is True


def test_pick_scheduler_strategy_metadata_advertises_phase_and_cost() -> None:
    assert pick_scheduler_strategy.phase == 4
    assert pick_scheduler_strategy.autocomp_cost_impact == "high"
    assert pick_scheduler_strategy.wraps_pass == "recipe.propose_scheduling_policy"


# ---------------------------------------------------------------------------
# target coverage row
# ---------------------------------------------------------------------------


def test_megakernel_coverage_row_present_for_cuda_and_amd() -> None:
    cov = get_coverage("megakernel_static_schedule", "cuda")
    assert cov is not None
    assert cov.coverage == "none"
    assert cov.cost_weight_bias == "prefer"
    cov_amd = get_coverage("megakernel_static_schedule", "amd")
    assert cov_amd is not None
    assert cov_amd.cost_weight_bias == "prefer"


def test_megakernel_cost_weight_is_below_neutral_on_cuda() -> None:
    """`prefer` bias should drop the cost weight below 1.0."""
    weight = cost_weight_for("megakernel_static_schedule", "cuda")
    assert weight < 1.0
