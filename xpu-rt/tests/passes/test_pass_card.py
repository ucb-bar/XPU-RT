"""Tests for compgen.passes.cards."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from compgen.audit.errors import MissingPassCard
from compgen.passes.cards import (
    COST_KINDS,
    PASS_FAMILIES,
    PASS_LEVELS,
    REFINEMENT_KINDS,
    PassCard,
    PassCardError,
    PassCardRegistry,
    default_registry_root,
    iter_cards,
    load_card,
    validate_card,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_ROOT = REPO_ROOT / "docs" / "generated" / "pass_cards"


def _good_card_dict() -> dict[str, object]:
    return {
        "schema_version": "pass_card_v1",
        "pass_id": "demo_pass",
        "display_name": "Demo pass",
        "level": "payload",
        "family": "fusion",
        "reads": ["a.json"],
        "writes": ["b.json"],
        "preconditions": ["x == y"],
        "invalidates": ["payload_summary"],
        "preserves_refinement": "bit_equality",
        "verification": ["structural"],
        "cost": "cheap",
        "failure_modes": ["unsupported_op"],
        "mcp_tool": "",
        "example_invocation": {"kind": "demo"},
    }


def test_round_trip(tmp_path: Path) -> None:
    raw = _good_card_dict()
    card = PassCard.from_dict(raw)
    out = tmp_path / "demo.yaml"
    out.write_text(yaml.safe_dump(card.to_dict(), sort_keys=True))
    reloaded = load_card(out)
    assert reloaded.to_dict() == card.to_dict()


def test_content_hash_is_deterministic() -> None:
    a = PassCard.from_dict(_good_card_dict())
    b = PassCard.from_dict(_good_card_dict())
    assert a.content_hash() == b.content_hash()
    # Modifying any field changes the hash
    raw = _good_card_dict()
    raw["family"] = "tiling"
    c = PassCard.from_dict(raw)
    assert c.content_hash() != a.content_hash()


def test_invalid_pass_id_rejected() -> None:
    raw = _good_card_dict()
    raw["pass_id"] = "Bad-Name"
    with pytest.raises(PassCardError, match="pass_id"):
        validate_card(PassCard.from_dict(raw))


def test_invalid_level_rejected() -> None:
    raw = _good_card_dict()
    raw["level"] = "totally_made_up"
    with pytest.raises(PassCardError, match="level"):
        validate_card(PassCard.from_dict(raw))


def test_invalid_family_rejected() -> None:
    raw = _good_card_dict()
    raw["family"] = "not_real"
    with pytest.raises(PassCardError, match="family"):
        validate_card(PassCard.from_dict(raw))


def test_invalid_refinement_rejected() -> None:
    raw = _good_card_dict()
    raw["preserves_refinement"] = "lossy"
    with pytest.raises(PassCardError, match="preserves_refinement"):
        validate_card(PassCard.from_dict(raw))


def test_invalid_cost_rejected() -> None:
    raw = _good_card_dict()
    raw["cost"] = "ludicrous"
    with pytest.raises(PassCardError, match="cost"):
        validate_card(PassCard.from_dict(raw))


def test_empty_preconditions_rejected() -> None:
    raw = _good_card_dict()
    raw["preconditions"] = []
    with pytest.raises(PassCardError, match="preconditions"):
        validate_card(PassCard.from_dict(raw))


def test_missing_required_field_raises_typed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "schema_version": "pass_card_v1",
            # missing pass_id
            "display_name": "x",
            "level": "payload",
            "family": "fusion",
            "preserves_refinement": "bit_equality",
            "cost": "cheap",
        })
    )
    with pytest.raises(PassCardError, match="pass_id"):
        load_card(bad)


def test_constants_are_published() -> None:
    """The published constants must include the canonical values used
    by the agent_decision_request."""
    assert "payload" in PASS_LEVELS
    assert "tiling" in PASS_FAMILIES
    assert "fusion" in PASS_FAMILIES
    assert "bit_equality" in REFINEMENT_KINDS
    assert "cheap" in COST_KINDS


def test_registry_load_smoke(tmp_path: Path) -> None:
    raw_a = _good_card_dict()
    raw_b = _good_card_dict()
    raw_b["pass_id"] = "another_pass"
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw_a))
    (tmp_path / "another_pass.yaml").write_text(yaml.safe_dump(raw_b))
    reg = PassCardRegistry.load(tmp_path)
    assert "demo_pass" in reg
    assert "another_pass" in reg
    assert len(reg) == 2
    assert reg.passes_allowed() == ("another_pass", "demo_pass")


def test_registry_duplicate_id_rejected(tmp_path: Path) -> None:
    raw = _good_card_dict()
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(raw))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(PassCardError, match="duplicate"):
        PassCardRegistry.load(tmp_path)


def test_registry_assert_resolvable_passes(tmp_path: Path) -> None:
    raw = _good_card_dict()
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    reg = PassCardRegistry.load(tmp_path)
    reg.assert_resolvable(["demo_pass"])


def test_registry_assert_resolvable_missing_raises(tmp_path: Path) -> None:
    raw = _good_card_dict()
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    reg = PassCardRegistry.load(tmp_path)
    with pytest.raises(MissingPassCard, match="ghost_pass"):
        reg.assert_resolvable(["demo_pass", "ghost_pass"])


def test_registry_require_returns_card(tmp_path: Path) -> None:
    raw = _good_card_dict()
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    reg = PassCardRegistry.load(tmp_path)
    card = reg.require("demo_pass")
    assert card.pass_id == "demo_pass"


def test_registry_require_missing_raises(tmp_path: Path) -> None:
    reg = PassCardRegistry.load(tmp_path)
    with pytest.raises(MissingPassCard, match="ghost"):
        reg.require("ghost")


def test_registry_underscore_prefixed_files_ignored(tmp_path: Path) -> None:
    raw = _good_card_dict()
    raw["pass_id"] = "real_one"
    (tmp_path / "real_one.yaml").write_text(yaml.safe_dump(raw))
    raw_skip = _good_card_dict()
    raw_skip["pass_id"] = "private"
    (tmp_path / "_private.yaml").write_text(yaml.safe_dump(raw_skip))
    reg = PassCardRegistry.load(tmp_path)
    assert "real_one" in reg
    assert "private" not in reg


def test_default_registry_root_resolves_under_repo_root() -> None:
    root = default_registry_root()
    assert root.parent.name == "generated"
    assert root.name == "pass_cards"


def test_iter_cards_returns_sorted(tmp_path: Path) -> None:
    raw_a = _good_card_dict()
    raw_a["pass_id"] = "z_pass"
    raw_b = _good_card_dict()
    raw_b["pass_id"] = "a_pass"
    (tmp_path / "z_pass.yaml").write_text(yaml.safe_dump(raw_a))
    (tmp_path / "a_pass.yaml").write_text(yaml.safe_dump(raw_b))
    cards = list(iter_cards(tmp_path))
    assert [c.pass_id for c in cards] == ["a_pass", "z_pass"]


def test_registry_rejects_unknown_invalidates_id(tmp_path: Path) -> None:
    """cross-link: invalidates ids must resolve to known summaries."""
    raw = _good_card_dict()
    raw["invalidates"] = ["totally_made_up_summary"]
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(PassCardError, match="known analysis summary"):
        PassCardRegistry.load(tmp_path)


def test_registry_accepts_known_invalidates_id(tmp_path: Path) -> None:
    """The registry default loader runs the cross-link; known ids pass."""
    raw = _good_card_dict()
    raw["invalidates"] = ["payload_summary", "graph_dossier_v3"]
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    reg = PassCardRegistry.load(tmp_path)
    assert "demo_pass" in reg


def test_registry_skip_cross_link_with_flag(tmp_path: Path) -> None:
    """When an integrator wants to test card schema without summary
    cross-link, the flag disables the check."""
    raw = _good_card_dict()
    raw["invalidates"] = ["totally_made_up_summary"]
    (tmp_path / "demo_pass.yaml").write_text(yaml.safe_dump(raw))
    # Should not raise with the flag off
    reg = PassCardRegistry.load(tmp_path, validate_summary_invalidates=False)
    assert "demo_pass" in reg
