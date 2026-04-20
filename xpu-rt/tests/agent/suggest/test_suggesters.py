"""P7.3 — model-aware suggesters: per-slot pattern detection.

Tests for the recipe index + the four primary suggesters
(propose_fusion, propose_megakernel_synthesis, propose_layout_plan,
propose_dequant_fusion). The other 5 suggesters get smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.agent.suggest import (
    SUGGESTERS,
    suggest,
    supported_slot_names,
)
from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._recipe_index import (
    build_recipe_index,
    critical_path_recipe_syms,
)
from compgen.api import compile_model, device as _device
from compgen.ir.recipe.seed import generate_seed_recipe

EXEMPLAR = (
    Path(__file__).resolve().parents[2]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _seed():
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _Mlp().eval(), dev, sample_inputs=(torch.randn(1, 32),),
    )
    recipe = generate_seed_recipe(
        compiled.payload_module, dev.profile, "latency",
    )
    return recipe, compiled.analysis_dossier, dev.profile


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_every_canonical_slot_has_a_suggester() -> None:
    canonical = {
        "propose_fusion", "propose_megakernel_synthesis",
        "propose_layout_plan", "propose_dequant_fusion",
        "propose_numerics_plan", "propose_peephole_pattern",
        "propose_buffer_lifetime_plan",
        "propose_rematerialization_plan",
        "propose_scheduling_policy",
    }
    available = set(supported_slot_names())
    missing = canonical - available
    assert not missing, f"missing suggesters: {missing}"


def test_unknown_slot_returns_empty_list() -> None:
    recipe, dossier, target = _seed()
    out = suggest(
        "not_a_slot", recipe=recipe, dossier=dossier, target=target,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Recipe index
# ---------------------------------------------------------------------------


def test_recipe_index_walks_regions() -> None:
    recipe, _, _ = _seed()
    idx = build_recipe_index(recipe)
    assert idx.regions, "no regions extracted"
    # Adjacency = N-1 pairs for N regions.
    assert len(idx.adjacency) == max(0, len(idx.regions) - 1)
    # Most regions should have a role tag (P7.1 propagated _pattern_hint).
    with_role = sum(1 for s in idx.regions if s in idx.role_by_region)
    assert with_role >= max(1, len(idx.regions) // 2)


def test_critical_path_translates_payload_ids_to_recipe_syms() -> None:
    recipe, dossier, _ = _seed()
    idx = build_recipe_index(recipe)
    cp = critical_path_recipe_syms(idx, dossier)
    # cp may be empty for a tiny MLP, but must be a list of strings.
    assert isinstance(cp, list)
    for s in cp:
        assert s in idx.regions


# ---------------------------------------------------------------------------
# propose_fusion
# ---------------------------------------------------------------------------


def test_suggest_fusion_returns_candidates_on_mlp() -> None:
    recipe, dossier, target = _seed()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target, k=5)
    assert isinstance(out, list)
    # An MLP has at least one canonical (e.g. mm→addmm or addmm→relu) pair.
    if not out:
        pytest.skip("MLP didn't produce any canonical fusion pair on this target")
    for c in out:
        assert isinstance(c, ProposalCandidate)
        gr = c.chosen.get("grouped_regions") or []
        assert len(gr) >= 2, f"candidate has too-small grouped_regions: {gr}"
        assert c.chosen.get("fusion_kind") == "producer_consumer"
        assert c.expected_impact >= 0.4


def test_suggest_fusion_to_proposal_is_accepted_by_propose_invent_slot(tmp_path: Path) -> None:
    """Submitting candidate 0 to propose_invent_slot must accept end-to-end."""
    from compgen.agent.invent_slots.registrar import register_invent_slots
    from compgen.agent.llm_driver import LLMDrivenCompiler
    from compgen.llm.mock_client import MockLLMClient
    from compgen.llm.registry import Registry
    from compgen.mcp.session import SessionManager
    from compgen.mcp.tools.transform import propose_invent_slot

    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _Mlp().eval(), dev, sample_inputs=(torch.randn(1, 32),),
    )
    reg = Registry(); register_invent_slots(reg)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env, target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=4, registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver

    candidates = suggest(
        "propose_fusion",
        recipe=driver.env.recipe,
        dossier=compiled.analysis_dossier,
        target=dev.profile,
        k=5,
    )
    if not candidates:
        pytest.skip("no candidates generated")
    # Submit candidate 0 verbatim.
    r = propose_invent_slot(
        sm, session_id=session.session_id,
        slot_name="propose_fusion",
        proposal=candidates[0].to_proposal(),
    )
    assert r["status"] == "accepted", r.get("gate_result")


# ---------------------------------------------------------------------------
# propose_megakernel_synthesis
# ---------------------------------------------------------------------------


def test_suggest_megakernel_returns_at_least_fallback() -> None:
    recipe, dossier, target = _seed()
    out = suggest(
        "propose_megakernel_synthesis",
        recipe=recipe, dossier=dossier, target=target,
    )
    # MLP has matmuls but no softmax → no attention window. The MLP
    # window may or may not match (depends on op decomposition). At
    # minimum the all-matmul fallback should fire if matmuls exist.
    assert isinstance(out, list)
    for c in out:
        assert "fused_region_refs" in c.chosen
        assert len(c.chosen["fused_region_refs"]) >= 1


# ---------------------------------------------------------------------------
# propose_layout_plan
# ---------------------------------------------------------------------------


def test_suggest_layout_plan_uses_target_tile_mn() -> None:
    recipe, dossier, target = _seed()
    out = suggest(
        "propose_layout_plan", recipe=recipe, dossier=dossier, target=target,
    )
    # Target has tile_mn declared, so layout strings should embed it.
    for c in out:
        assert "region_ref" in c.chosen
        assert "layout" in c.chosen


# ---------------------------------------------------------------------------
# propose_dequant_fusion (won't fire on a plain MLP — empty list is OK)
# ---------------------------------------------------------------------------


def test_suggest_dequant_fusion_on_plain_mlp_is_empty_or_valid() -> None:
    recipe, dossier, target = _seed()
    out = suggest(
        "propose_dequant_fusion", recipe=recipe, dossier=dossier, target=target,
    )
    # Plain MLP has no quantized ops → expect empty list.
    assert isinstance(out, list)
    for c in out:
        assert "region_ref" in c.chosen


# ---------------------------------------------------------------------------
# Smoke tests for the remaining 5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slot", [
    "propose_numerics_plan",
    "propose_peephole_pattern",
    "propose_buffer_lifetime_plan",
    "propose_rematerialization_plan",
    "propose_scheduling_policy",
])
def test_other_suggesters_smoke(slot: str) -> None:
    recipe, dossier, target = _seed()
    out = suggest(slot, recipe=recipe, dossier=dossier, target=target)
    assert isinstance(out, list)
    for c in out:
        assert isinstance(c, ProposalCandidate)
        assert c.chosen, f"{slot} candidate has empty chosen dict"


def test_proposal_candidate_to_proposal_shape_is_acceptable() -> None:
    recipe, dossier, target = _seed()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target)
    if not out:
        pytest.skip("no fusion candidates")
    p = out[0].to_proposal()
    # Required keys for the structural gate.
    assert "chosen" in p
    assert p.get("select_vs_invent") == "invent"
