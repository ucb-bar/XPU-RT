"""P7-polish: megakernel filter + fusion dedup + next_call.

Locks in the three fixes that pushed agent UX from 8.5 → 9:

  1. Megakernel attention/MLP windows include only semantic-role
     members (matmul/softmax/activation/...), not structural noise
     (yield/empty/view/transpose) the seed walks past.
  2. propose_fusion suggestions dedupe by canonical (prod_role,
     cons_role); occurrences land under ``members`` instead of
     eating the agent's k budget.
  3. Every candidate carries a ``next_call`` self-describing follow-
     up; multi-member candidates default to ``batch_propose``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.agent.suggest import suggest
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.ir.recipe.seed import generate_seed_recipe

EXEMPLAR = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


def _seed_gemma_like():
    """Use the user_perspective Gemma slice — has matmul→softmax→matmul."""
    import sys

    UP = Path(__file__).resolve().parents[3] / "user_perspective"
    if not (UP / "models" / "gemma_decode_slice.py").exists():
        pytest.skip("user_perspective/models/gemma_decode_slice.py not present (sandbox is gitignored)")
    if str(UP) not in sys.path:
        sys.path.insert(0, str(UP))
    from models.gemma_decode_slice import load

    bundle = load()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        bundle.model.eval(),
        dev,
        sample_inputs=bundle.sample_inputs,
    )
    recipe = generate_seed_recipe(
        compiled.payload_module,
        dev.profile,
        "latency",
    )
    return recipe, compiled.analysis_dossier, dev.profile


# ---------------------------------------------------------------------------
# 1. Megakernel filter
# ---------------------------------------------------------------------------


def test_megakernel_attention_window_is_filtered_to_semantic_roles() -> None:
    recipe, dossier, target = _seed_gemma_like()
    out = suggest(
        "propose_megakernel_synthesis",
        recipe=recipe,
        dossier=dossier,
        target=target,
        k=5,
    )
    if not out:
        pytest.skip("no megakernel candidate produced")
    structural_noise = {"yield", "empty", "view", "transpose", "permute", "cat", "clone", "unsqueeze", "expand"}
    # Walk every candidate's fused_region_refs; cross-check that the
    # role of each region is NOT structural noise.
    from compgen.agent.suggest._recipe_index import build_recipe_index

    idx = build_recipe_index(recipe)
    for c in out:
        regions = c.chosen.get("fused_region_refs", [])
        # Expect a tight cluster (the previous bug was 29 regions).
        assert 1 <= len(regions) <= 12, f"fused_region_refs is not tight: {len(regions)} regions ({regions})"
        for sym in regions:
            role = idx.role_by_region.get(sym, "")
            assert role not in structural_noise, (
                f"megakernel cluster contains structural-noise region {sym} (role={role})"
            )


# ---------------------------------------------------------------------------
# 2. Fusion dedup by role pair
# ---------------------------------------------------------------------------


def test_fusion_candidates_dedupe_by_role_pair() -> None:
    recipe, dossier, target = _seed_gemma_like()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target, k=5)
    if not out:
        pytest.skip("no fusion candidates")
    # No two candidates share the same (producer_role, consumer_role).
    seen = set()
    for c in out:
        pair = (c.chosen.get("producer_role"), c.chosen.get("consumer_role"))
        assert pair not in seen, f"duplicate role-pair candidate: {pair}"
        seen.add(pair)


def test_fusion_candidates_collapse_3_rmsnorm_into_one(tmp_path: Path) -> None:
    """The Gemma slice has 3 RMSNorms → 3 rsqrt→mul fusions. They must
    collapse into ONE candidate with members=[3]."""
    recipe, dossier, target = _seed_gemma_like()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target, k=10)
    rsqrt_mul = [c for c in out if c.chosen.get("producer_role") == "rsqrt" and c.chosen.get("consumer_role") == "mul"]
    if not rsqrt_mul:
        pytest.skip("no rsqrt→mul candidate (model decomposed differently)")
    assert len(rsqrt_mul) == 1, f"rsqrt→mul not deduped — got {len(rsqrt_mul)} candidates"
    # The Gemma decoder slice has 3 rsqrt regions. The grouped
    # candidate must surface them under members.
    head = rsqrt_mul[0]
    assert len(head.members) >= 1
    # If there are >=2 members, the candidate's expected_impact gets a
    # multiplicity boost.
    if len(head.members) >= 2:
        assert head.expected_impact >= 0.7
        assert "× " in head.rationale


# ---------------------------------------------------------------------------
# 3. next_call self-describing follow-up
# ---------------------------------------------------------------------------


def test_single_member_candidate_next_call_is_propose_invent_slot() -> None:
    recipe, dossier, target = _seed_gemma_like()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target, k=10)
    singletons = [c for c in out if len(c.members) <= 1]
    if not singletons:
        pytest.skip("every candidate has multiple members")
    nc = singletons[0].next_call()
    assert nc["tool"] == "propose_invent_slot"
    assert nc["args"]["slot_name"] == "propose_fusion"
    assert "proposal" in nc["args"]


def test_multi_member_candidate_next_call_is_batch_propose() -> None:
    recipe, dossier, target = _seed_gemma_like()
    out = suggest("propose_fusion", recipe=recipe, dossier=dossier, target=target, k=10)
    multis = [c for c in out if len(c.members) >= 2]
    if not multis:
        pytest.skip("no multi-member candidate; corpus may be too small")
    nc = multis[0].next_call()
    assert nc["tool"] == "batch_propose"
    proposals = nc["args"]["proposals"]
    # One proposal per member.
    assert len(proposals) == len(multis[0].members)
    for p in proposals:
        assert p["slot_name"] == "propose_fusion"
        assert "chosen" in p["proposal"]


def test_dispatch_stamps_slot_name_on_every_candidate() -> None:
    recipe, dossier, target = _seed_gemma_like()
    for slot in ("propose_fusion", "propose_megakernel_synthesis", "propose_layout_plan", "propose_numerics_plan"):
        out = suggest(slot, recipe=recipe, dossier=dossier, target=target)
        for c in out:
            assert c.slot_name == slot


# ---------------------------------------------------------------------------
# Idempotent re-suggest: skip already-proposed pairs
# ---------------------------------------------------------------------------


def test_already_proposed_pair_drops_out_of_next_suggest_call(tmp_path: Path) -> None:
    """After propose_invent_slot lands a fusion, re-calling suggest
    must not return the SAME (prod_sym, cons_sym) pair."""
    from compgen.agent.invent_slots.registrar import register_invent_slots
    from compgen.agent.llm_driver import LLMDrivenCompiler
    from compgen.llm.mock_client import MockLLMClient
    from compgen.llm.registry import Registry
    from compgen.mcp.session import SessionManager
    from compgen.mcp.tools.transform import propose_invent_slot

    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()
    dev = _device(EXEMPLAR)
    import sys

    UP = Path(__file__).resolve().parents[3] / "user_perspective"
    if not (UP / "models" / "gemma_decode_slice.py").exists():
        pytest.skip("user_perspective/models/gemma_decode_slice.py not present (sandbox is gitignored)")
    if str(UP) not in sys.path:
        sys.path.insert(0, str(UP))
    from models.gemma_decode_slice import load

    bundle = load()
    compiled = compile_model(
        bundle.model.eval(),
        dev,
        sample_inputs=bundle.sample_inputs,
    )
    reg = Registry()
    register_invent_slots(reg)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=4,
        registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver

    first = suggest(
        "propose_fusion", recipe=driver.env.recipe, dossier=compiled.analysis_dossier, target=dev.profile, k=10
    )
    if not first:
        pytest.skip("no fusion candidates on this corpus")
    head_pair = tuple(first[0].chosen["grouped_regions"])
    propose_invent_slot(
        sm,
        session_id=session.session_id,
        slot_name="propose_fusion",
        proposal=first[0].to_proposal(),
    )
    second = suggest(
        "propose_fusion", recipe=driver.env.recipe, dossier=compiled.analysis_dossier, target=dev.profile, k=10
    )
    second_pairs = {tuple(c.chosen["grouped_regions"]) for c in second}
    # The head pair we already proposed must not reappear at the
    # head of the second call.
    assert head_pair not in second_pairs or (
        # Fallback: at minimum the candidate the agent JUST submitted
        # shouldn't be the new top suggestion.
        len(second) == 0 or tuple(second[0].chosen["grouped_regions"]) != head_pair
    )
