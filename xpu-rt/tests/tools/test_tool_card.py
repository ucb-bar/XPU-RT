"""Unit tests for :mod:`compgen.tools.tool_card`.

Coverage:

* positive — a minimal valid card constructs cleanly.
* negative controls — every closed-enum branch and every required
  field individually triggers :class:`ToolCardError`.
* round-trip — ``ToolCard.from_dict(card.to_dict()) == card`` for any
  card built through the public API.
"""

from __future__ import annotations

import pytest
from compgen.tools.errors import ToolCardError
from compgen.tools.tool_card import (
    FORBIDDEN_ACTIONS,
    MATURITY_LEVELS,
    PROMOTION_REQUIREMENT_KEYS,
    SCHEMA_VERSION,
    TOOL_PHASES,
    ToolCard,
)


def _minimal_card_body(**overrides):
    body = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": "compgen_unit_test_tool",
        "maturity": "T0",
        "phase": "evidence",
        "description": "unit test fixture",
        "entrypoints": {
            "python": "compgen.tools.builtin.echo:run",
        },
        "input_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "output_schema": {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"enum": ["ok", "error"]},
            },
        },
        "writes": {"allowed_roots": ["${run_dir}/"]},
        "forbidden": ["mutate_payload_ir"],
        "promotion_requirements": {},
    }
    body.update(overrides)
    return body


def test_minimal_card_constructs():
    card = ToolCard.from_dict(_minimal_card_body())
    assert card.tool_id == "compgen_unit_test_tool"
    assert card.maturity == "T0"
    assert card.maturity_index == 0
    assert card.phase == "evidence"
    assert card.entrypoints.python == "compgen.tools.builtin.echo:run"


def test_roundtrip_to_dict_from_dict():
    card = ToolCard.from_dict(_minimal_card_body())
    roundtrip = ToolCard.from_dict(card.to_dict())
    assert roundtrip == card


def test_unknown_maturity_rejected():
    body = _minimal_card_body(maturity="T99")
    with pytest.raises(ToolCardError, match="maturity"):
        ToolCard.from_dict(body)


def test_unknown_phase_rejected():
    body = _minimal_card_body(phase="warp_drive_dispatch")
    with pytest.raises(ToolCardError, match="phase"):
        ToolCard.from_dict(body)


def test_unknown_forbidden_action_rejected():
    body = _minimal_card_body(forbidden=["definitely_not_real"])
    with pytest.raises(ToolCardError, match="forbidden action"):
        ToolCard.from_dict(body)


def test_unknown_promotion_requirement_key_rejected():
    body = _minimal_card_body(promotion_requirements={"warp_speed": True})
    with pytest.raises(ToolCardError, match="promotion_requirements"):
        ToolCard.from_dict(body)


def test_missing_required_field_rejected():
    body = _minimal_card_body()
    del body["tool_id"]
    with pytest.raises(ToolCardError, match="tool_id"):
        ToolCard.from_dict(body)


def test_missing_python_entrypoint_rejected():
    body = _minimal_card_body()
    body["entrypoints"] = {"cli": "compgen-tool run foo"}
    with pytest.raises(ToolCardError, match="entrypoints.python"):
        ToolCard.from_dict(body)


def test_wrong_schema_version_rejected():
    body = _minimal_card_body(schema_version="some_old_version")
    with pytest.raises(ToolCardError, match="schema_version"):
        ToolCard.from_dict(body)


def test_output_schema_must_have_status_enum():
    body = _minimal_card_body()
    body["output_schema"] = {"type": "object", "properties": {"foo": {"type": "string"}}}
    with pytest.raises(ToolCardError, match="status"):
        ToolCard.from_dict(body)


def test_output_schema_status_enum_must_be_subset():
    body = _minimal_card_body()
    body["output_schema"] = {
        "type": "object",
        "properties": {"status": {"enum": ["ok", "happy", "sad"]}},
    }
    with pytest.raises(ToolCardError, match="status enum"):
        ToolCard.from_dict(body)


def test_input_schema_must_be_object_type():
    body = _minimal_card_body()
    body["input_schema"] = {"type": "string"}
    with pytest.raises(ToolCardError, match="input_schema.type"):
        ToolCard.from_dict(body)


def test_output_schema_must_be_object_type():
    body = _minimal_card_body()
    body["output_schema"] = {"type": "array"}
    with pytest.raises(ToolCardError, match="output_schema.type"):
        ToolCard.from_dict(body)


def test_high_maturity_without_evidence_rejected():
    """A card cannot claim T5 if mcp_wrapper=false."""

    body = _minimal_card_body(maturity="T5")
    body["promotion_requirements"] = {
        "unit_tests": True,
        "negative_controls": True,
        "cli_wrapper": True,
        "artifact_outputs": True,
        "skill_doc": True,
        "mcp_wrapper": False,  # <- inconsistent with T5
    }
    with pytest.raises(ToolCardError, match="mcp_wrapper"):
        ToolCard.from_dict(body)


def test_t2_requires_unit_tests_and_negative_controls():
    """A T2 tool must declare positive unit tests AND negative controls.

    The card also needs ``cli_wrapper`` (transitive T1 requirement);
    we set it explicitly so the failure surfaces on the T2 flag.
    """

    body = _minimal_card_body(maturity="T2")
    body["promotion_requirements"] = {
        "cli_wrapper": True,
        "unit_tests": True,
        "negative_controls": False,
    }
    with pytest.raises(ToolCardError, match="negative_controls"):
        ToolCard.from_dict(body)


def test_promotion_requirements_to_dict_is_total():
    """``to_dict`` must list every key in PROMOTION_REQUIREMENT_KEYS."""

    card = ToolCard.from_dict(_minimal_card_body())
    serialised = card.promotion_requirements.to_dict()
    assert set(serialised.keys()) == set(PROMOTION_REQUIREMENT_KEYS)


def test_closed_enums_are_documented_completely():
    """Smoke test that the closed-enum tuples are non-empty and stable."""

    assert len(MATURITY_LEVELS) == 8
    assert len(TOOL_PHASES) == 6
    assert "mutate_payload_ir" in FORBIDDEN_ACTIONS
    assert "bypass_verifier" in FORBIDDEN_ACTIONS


def test_real_echo_card_loads_from_disk():
    """The shipped ``echo.yaml`` card is itself a valid card."""

    from pathlib import Path

    from compgen.tools.tool_registry import load_tool_card

    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "compgen"
        / "tools"
        / "cards"
        / "echo.yaml"
    )
    card = load_tool_card(path)
    assert card.tool_id == "compgen_echo"
    # together lift echo to T2: Python entrypoint, the
    # ``compgen tool run`` CLI, and the positive + negative-control
    # tests in this directory.
    assert card.maturity == "T2"
    assert card.entrypoints.cli == "compgen-tool run compgen_echo"
    assert card.promotion_requirements.get("unit_tests") is True
    assert card.promotion_requirements.get("negative_controls") is True
    assert card.promotion_requirements.get("cli_wrapper") is True
    assert card.promotion_requirements.get("skill_doc") is False
