"""End-to-end: step_invent on accept appends a propose-op to the live recipe.

Before P5.1, accepted proposals were recorded in the side-log only — the
recipe ModuleOp never changed. This test asserts the new behaviour:

  1. A fresh driver has a recipe with some seed ops.
  2. step_invent('propose_fusion', {...accepted shape...}) lands.
  3. The recipe now contains a `recipe.propose_fusion` op referring to
     the proposed regions.
  4. `current_view()` surfaces the new op.
  5. A malformed proposal is rejected with a remediation hint, not a crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model, device as _device
from compgen.ir.recipe.ops_propose import ProposeFusionOp
from compgen.llm.mock_client import MockLLMClient

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _driver(tmp_path: Path) -> LLMDrivenCompiler:
    from compgen.agent.invent_slots.registrar import register_invent_slots
    from compgen.llm.registry import Registry

    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _TinyMLP().eval(), dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    env = compiled.create_agent_env(budget=4)
    # Use a per-test scratch registry + populate the canonical slots so
    # propose_fusion / propose_megakernel_synthesis are looked up.
    reg = Registry()
    register_invent_slots(reg)
    return LLMDrivenCompiler(
        env=env, target=dev.profile, llm_client=MockLLMClient(strict=False),
        transcript_dir=tmp_path / "transcripts", budget=4,
        registry=reg,
    )


def _count_ops(recipe_module, op_name: str) -> int:
    return sum(1 for op in recipe_module.body.block.ops if op.name == op_name)


def test_step_invent_appends_propose_fusion_on_accept(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    assert driver.env.recipe is not None
    before = _count_ops(driver.env.recipe, "recipe.propose_fusion")

    result = driver.step_invent(
        "propose_fusion",
        {
            "chosen": {
                "grouped_regions": ["r_0", "r_1"],
                "fusion_kind": "producer_consumer",
            },
            "candidates": [{"id": "c0"}],
            "target_feature_justification": "HVX tile alignment",
            "select_vs_invent": "invent",
        },
    )
    assert result.status == "accepted", result.gate_result
    assert result.tool_result is not None
    assert result.tool_result["appended_op"] == "recipe.propose_fusion"

    after = _count_ops(driver.env.recipe, "recipe.propose_fusion")
    assert after == before + 1, "propose_fusion not appended to recipe"


def test_current_view_surfaces_appended_op(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    view_before = driver.current_view(max_ops=150)
    hash_before = view_before["hash"]

    driver.step_invent(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_0", "r_2"]},
            "select_vs_invent": "invent",
        },
    )

    view_after = driver.current_view(max_ops=150)
    assert view_after["hash"] != hash_before
    all_rows = view_after["banner"] + [
        r for r in view_after["middle"] if "_op" in r
    ]
    assert any(r["_op"] == "recipe.propose_fusion" for r in all_rows)


def test_diff_since_reports_the_added_op(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    # driver auto-seeds ckpt_0 in __post_init__; use it as our baseline.
    driver.step_invent(
        "propose_fusion",
        {
            "chosen": {"grouped_regions": ["r_0", "r_1"]},
            "select_vs_invent": "invent",
        },
    )
    diff = driver.diff_since("ckpt_0")
    assert diff["status"] == "ok"
    added_ops = {e["_op"] for e in diff["added"]}
    assert "recipe.propose_fusion" in added_ops


def test_malformed_proposal_demotes_to_rejected_with_hint(tmp_path: Path) -> None:
    """Missing grouped_regions → bridge raises ValueError → status flips to rejected."""
    driver = _driver(tmp_path)
    before_count = _count_ops(driver.env.recipe, "recipe.propose_fusion")

    result = driver.step_invent(
        "propose_fusion",
        {
            # intentionally omit 'grouped_regions' inside 'chosen'
            "chosen": {},
            "select_vs_invent": "invent",
        },
    )
    assert result.status == "rejected"
    hint = result.remediation_hint
    assert hint is not None
    assert "grouped_regions" in hint or "chosen" in hint

    # And nothing landed in the recipe.
    assert _count_ops(driver.env.recipe, "recipe.propose_fusion") == before_count


def test_unknown_slot_does_not_append(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    # Unknown slot name returns 'unknown' from step_invent (slot lookup fails).
    result = driver.step_invent(
        "not_a_real_slot",
        {"chosen": {}, "select_vs_invent": "invent"},
    )
    assert result.status == "unknown"


def test_accept_megakernel_synthesis_appends(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    before = _count_ops(driver.env.recipe, "recipe.propose_megakernel_synthesis")

    result = driver.step_invent(
        "propose_megakernel_synthesis",
        {
            "chosen": {
                "megakernel_name": "demo_mk",
                "fused_region_refs": ["r_0", "r_1"],
            },
            "target_feature_justification": "persistent_kernels + semaphore_atomics",
            "select_vs_invent": "invent",
        },
    )
    assert result.status == "accepted", result.gate_result
    after = _count_ops(driver.env.recipe, "recipe.propose_megakernel_synthesis")
    assert after == before + 1
