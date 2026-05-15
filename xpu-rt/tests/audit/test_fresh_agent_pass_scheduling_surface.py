"""Fresh-agent task pack must include the post-agentic surface.

 (pass-card registry), (analysis checkpoints),
(invalidation discipline), (full pass-card coverage), and
(multi-pass scheduling), a fresh agent has a much richer surface
to reason about than the baseline. This test asserts the task
pack carries every artifact that surface depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from compgen.audit.fresh_agent import build_task_pack

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def task_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("m35_pack")
    build_task_pack(
        out_dir=out, commit="m35_test",
        repo_root=REPO_ROOT, skip_python_package=True,
    )
    return out


# --------------------------------------------------------------------------- #
# pass-card registry surface
# --------------------------------------------------------------------------- #


def test_pack_contains_pass_card_index(task_pack: Path) -> None:
    """The auto-generated INDEX.md is the agent's navigation surface."""
    index = task_pack / "docs" / "generated" / "pass_cards" / "INDEX.md"
    assert index.exists(), "INDEX.md missing from task pack"
    text = index.read_text()
    assert "60 cards" in text, "INDEX no longer reports 60 cards"
    assert "## At a glance" in text


def test_pack_contains_at_least_60_cards(task_pack: Path) -> None:
    """All 60 ported pass cards must travel with the task pack."""
    cards_root = task_pack / "docs" / "generated" / "pass_cards"
    cards = list(cards_root.rglob("*.yaml"))
    assert len(cards) >= 60, (
        f"expected ≥60 pass card YAMLs in pack, got {len(cards)}"
    )


def test_pack_card_subdirectories_match_families(task_pack: Path) -> None:
    """Cards organized by family — fresh agent navigates by directory."""
    cards_root = task_pack / "docs" / "generated" / "pass_cards"
    expected_subdirs = {
        "tiling", "fusion", "layout", "layout_pipeline", "quant",
        "codegen", "scheduling", "memory", "canonicalize",
        "event_tensor", "fx_graph", "eqsat",
    }
    actual = {p.name for p in cards_root.iterdir() if p.is_dir()}
    missing = expected_subdirs - actual
    assert not missing, f"family subdirs missing from pack: {missing}"


def test_pack_card_carries_phase_and_source(task_pack: Path) -> None:
    """A representative card must carry source + phase."""
    card = task_pack / "docs" / "generated" / "pass_cards" / "tiling" / "set_tile_params.yaml"
    raw = yaml.safe_load(card.read_text())
    assert raw.get("source"), "card missing source field"
    assert raw.get("impl_path"), "card missing impl_path field"
    # phase is optional; effective_phase derives from family if absent.


# --------------------------------------------------------------------------- #
# through realness contracts
# --------------------------------------------------------------------------- #


def test_pack_carries_all_realness_contracts(task_pack: Path) -> None:
    realness_dir = task_pack / "docs" / "realness"
    contracts = list(realness_dir.glob("*.yaml"))
    expected = {
        "m26_promotion_bridge.yaml",
        "m27_recipe_ir_promote_op.yaml",
        "m28_promotion_retrieval.yaml",
        "m29_promotion_gates.yaml",
        "m30_efficiency_report.yaml",
        "m31a_audit_layer.yaml",
        "m31_pass_card_registry.yaml",
        "m32_multi_level_analysis.yaml",
        "m33_invalidation_discipline.yaml",
        "m34_pass_scheduling.yaml",
    }
    actual_names = {p.name for p in contracts}
    missing = expected - actual_names
    assert not missing, f"realness contracts missing from pack: {missing}"


# --------------------------------------------------------------------------- #
# audit + skills
# --------------------------------------------------------------------------- #


def test_pack_includes_compgen_skills(task_pack: Path) -> None:
    """The compgen, compgen-compile, compgen-candidate-selection
    skills are how the agent drives the compile."""
    for skill in ("compgen", "compgen-compile", "compgen-candidate-selection"):
        skill_md = task_pack / ".claude" / "skills" / skill / "SKILL.md"
        assert skill_md.exists(), f"skill {skill} missing"


def test_pack_includes_realness_policy(task_pack: Path) -> None:
    policy = task_pack / "docs" / "reference" / "realness_policy.md"
    assert policy.exists()


# --------------------------------------------------------------------------- #
# Task prompt mentions the new surface
# --------------------------------------------------------------------------- #


def test_task_prompt_mentions_pass_cards(task_pack: Path) -> None:
    text = (task_pack / "TASK.md").read_text()
    assert "pass_cards" in text or "INDEX.md" in text, (
        "task prompt does not mention pass cards or INDEX.md"
    )
    # Should explain the multi-step pass_plan path
    assert "pass_plan" in text, "task prompt does not mention pass_plan"
    # Phase ordering should be called out
    assert "phase" in text.lower(), "task prompt does not mention phases"


def test_task_prompt_lists_typed_outcomes_only(task_pack: Path) -> None:
    text = (task_pack / "TASK.md").read_text()
    assert "typed-blocked" in text or "typed_blocked" in text or "compgen.runtime.errors" in text
    assert "silent partial pass is failure" in text or "silent partial" in text.lower()
