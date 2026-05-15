"""Tests for :mod:`compgen.tools.tool_registry`.

Coverage:

* the shipped cards directory loads cleanly;
* unknown cards directory yields nothing (does not crash);
* a malformed card raises :class:`ToolCardError` rather than being
  silently skipped (hard rule: every card is audited or the loader
  fails);
* iteration order is deterministic.
"""

from __future__ import annotations

import pytest
import yaml
from compgen.tools.errors import ToolCardError
from compgen.tools.tool_registry import (
    iter_tool_cards,
    load_tool_card,
    tool_cards_root,
)


def test_shipped_cards_load(tmp_path):
    cards = list(iter_tool_cards())
    assert any(card.tool_id == "compgen_echo" for card in cards), (
        "echo card must be discoverable from the shipped cards directory"
    )


def test_missing_directory_yields_nothing(tmp_path):
    out = list(iter_tool_cards(tmp_path / "does_not_exist"))
    assert out == []


def test_malformed_card_raises(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "schema_version": "compgen_tool_card_v1",
                "tool_id": "broken",
                "maturity": "T0",
                "phase": "made_up_phase",  # closed-enum violation
                "entrypoints": {"python": "compgen.tools.builtin.echo:run"},
                "input_schema": {"type": "object"},
                "output_schema": {
                    "type": "object",
                    "properties": {"status": {"enum": ["ok"]}},
                },
                "writes": {"allowed_roots": ["${run_dir}/"]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolCardError, match="phase"):
        list(iter_tool_cards(tmp_path))


def test_iteration_order_is_deterministic(tmp_path):
    for name in ("c.yaml", "a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(
                {
                    "schema_version": "compgen_tool_card_v1",
                    "tool_id": f"tool_{name.split('.')[0]}",
                    "maturity": "T0",
                    "phase": "evidence",
                    "entrypoints": {"python": "compgen.tools.builtin.echo:run"},
                    "input_schema": {"type": "object"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"status": {"enum": ["ok"]}},
                    },
                    "writes": {"allowed_roots": ["${run_dir}/"]},
                }
            ),
            encoding="utf-8",
        )
    cards = list(iter_tool_cards(tmp_path))
    ids = [c.tool_id for c in cards]
    assert ids == ["tool_a", "tool_b", "tool_c"]


def test_non_mapping_yaml_rejected(tmp_path):
    bad = tmp_path / "list.yaml"
    bad.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ToolCardError, match="YAML mapping"):
        load_tool_card(bad)


def test_tool_cards_root_resolves_under_compgen():
    root = tool_cards_root()
    assert root.name == "cards"
    assert root.parent.name == "tools"
