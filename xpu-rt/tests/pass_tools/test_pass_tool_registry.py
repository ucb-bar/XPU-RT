"""PassToolCard schema tests (registry implementation lands )."""

from __future__ import annotations

import pytest

from compgen.pass_tools.pass_tool_types import (
    PASS_PHASES,
    REFINEMENT_KINDS,
    PassToolCard,
    PassToolCardError,
)


def _minimal_body(**overrides):
    body = {
        "schema_version": "pass_tool_card_v1",
        "tool_id": "fuse_matmul_bias_relu",
        "phase": "recipe_authoring",
        "reads": ["graph_dossier", "payload_ir_summary"],
        "writes": ["recipe_delta"],
        "allowed_recipe_ops": ["FuseElementwise", "SetAccumulator"],
        "refinement": {
            "kind": "tolerance_eps",
            "verifier": "differential_then_z3_if_promoted",
        },
        "entrypoint": "compgen.pass_tools.builtin.fuse_matmul_bias_relu:run",
    }
    body.update(overrides)
    return body


def test_pass_tool_card_round_trips():
    card = PassToolCard.from_dict(_minimal_body())
    restored = PassToolCard.from_dict(card.to_dict())
    assert restored == card


def test_pass_tool_card_unknown_phase_rejected():
    with pytest.raises(PassToolCardError, match="phase"):
        PassToolCard.from_dict(_minimal_body(phase="totally_made_up"))


def test_pass_tool_card_unknown_refinement_rejected():
    body = _minimal_body()
    body["refinement"]["kind"] = "magic"
    with pytest.raises(PassToolCardError, match="refinement.kind"):
        PassToolCard.from_dict(body)


def test_pass_tool_card_writes_payload_ir_rejected():
    """Hard rule 4: pass tools never mutate Payload IR directly."""
    body = _minimal_body(writes=["payload_ir"])
    with pytest.raises(PassToolCardError, match="payload_ir"):
        PassToolCard.from_dict(body)


def test_pass_tool_card_missing_required_field_rejected():
    body = _minimal_body()
    body.pop("entrypoint")
    with pytest.raises(PassToolCardError, match="entrypoint"):
        PassToolCard.from_dict(body)


def test_pass_phases_and_refinement_kinds_are_typed_enums():
    assert "recipe_authoring" in PASS_PHASES
    assert "tolerance_eps" in REFINEMENT_KINDS
    assert "differential_then_z3_if_promoted" in REFINEMENT_KINDS
