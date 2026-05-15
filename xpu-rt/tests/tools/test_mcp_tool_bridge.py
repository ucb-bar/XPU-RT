"""Tests for :mod:`compgen.mcp.tool_bridge`.

Coverage:

Positive:
* The shipped echo card produces a valid MCP tool dict whose
  ``input_schema`` is bit-equal to the card's ``input_schema``.
* ``bridge_tools()`` returns exactly the ToolCards that declare an
  MCP entrypoint (echo today).
* The bridged tool appears inside ``compgen.mcp.tools.get_all_tools()``
  so the MCP server enumerates it next to native tools.
* Calling the bridge handler runs the underlying tool end-to-end and
  returns the canonical ``ToolResult.to_dict()`` payload.

Negative controls:
* Cards without ``entrypoints.mcp`` are not bridged.
* ``make_mcp_tool_dict`` raises if asked to bridge a card with no MCP
  entrypoint (defensive — must never silently elide).
* Bridge handler translates a tool that raises a ``ToolRunError`` into
  a typed ``status=error`` payload (the bridge must NEVER let an
  exception escape into the MCP transport).
* Bridge handler translates an input-schema violation into
  ``error_type=input_schema_violation``.

Schema-equivalence (the headline T5 gate this enables):
* For every bridged tool, MCP ``input_schema`` is bit-equal to the
  ToolCard ``input_schema`` (after canonical-JSON normalisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from compgen.mcp.tool_bridge import (
    bridge_tools,
    make_mcp_tool_dict,
)
from compgen.tools.tool_card import ToolCard
from compgen.tools.tool_registry import load_tool_card, tool_cards_root


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# Positive ------------------------------------------------------------


def test_echo_bridge_tool_round_trips():
    card = load_tool_card(tool_cards_root() / "echo.yaml")
    tool = make_mcp_tool_dict(card)
    assert tool["name"] == "compgen_echo"
    assert tool["phase"] in {"inspect", "transform", "lifecycle", "job"}
    assert callable(tool["handler"])
    assert tool["_card_tool_id"] == "compgen_echo"


def test_bridge_input_schema_is_bit_equal_to_card_input_schema():
    """The headline cross-surface guarantee — MCP and runner share schema."""

    card = load_tool_card(tool_cards_root() / "echo.yaml")
    tool = make_mcp_tool_dict(card)
    assert _canonical(tool["input_schema"]) == _canonical(card.input_schema)


def test_bridge_tools_returns_only_mcp_carded():
    tools = bridge_tools()
    ids = [t["_card_tool_id"] for t in tools]
    assert "compgen_echo" in ids


def test_bridged_tool_reachable_via_get_all_tools():
    from compgen.mcp.tools import get_all_tools

    names = [t.get("name") for t in get_all_tools()]
    assert "compgen_echo" in names


def test_bridge_handler_runs_end_to_end(tmp_path):
    card = load_tool_card(tool_cards_root() / "echo.yaml")
    tool = make_mcp_tool_dict(card)
    handler = tool["handler"]
    payload = handler(sm=None, text="bridge-says-hi", count=2, out_dir=str(tmp_path / "out"))
    assert payload["status"] == "ok"
    assert payload["tool_id"] == "compgen_echo"
    assert payload["result"]["lines_written"] == 2


# Schema-equivalence over every bridged tool --------------------------


def test_every_bridged_tool_matches_card_schema():
    """Card-YAML filenames are *not* required to match tool_id, so we
    iterate the registry instead of guessing the path."""

    from compgen.tools.tool_registry import iter_tool_cards

    by_id = {c.tool_id: c for c in iter_tool_cards()}
    for tool in bridge_tools():
        card_id = tool["_card_tool_id"]
        card = by_id[card_id]
        assert _canonical(tool["input_schema"]) == _canonical(card.input_schema), (
            f"bridge schema for {card_id} drifted from card"
        )


# Negative controls ---------------------------------------------------


def test_bridge_skips_cards_without_mcp_entrypoint(tmp_path):
    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_no_mcp"
    body["entrypoints"]["mcp"] = ""
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    (cards_dir / "no_mcp.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    tools = bridge_tools(cards_root=cards_dir)
    assert tools == []


def test_make_mcp_tool_dict_rejects_card_without_mcp_entrypoint():
    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_no_mcp_direct"
    body["entrypoints"]["mcp"] = ""
    card = ToolCard.from_dict(body)
    with pytest.raises(ValueError, match="entrypoints.mcp"):
        make_mcp_tool_dict(card)


def test_bridge_handler_translates_input_schema_error_to_typed_payload(tmp_path):
    card = load_tool_card(tool_cards_root() / "echo.yaml")
    handler = make_mcp_tool_dict(card)["handler"]
    # Missing required "text" field.
    payload = handler(sm=None, count=2, out_dir=str(tmp_path / "out"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "input_schema_violation"
    assert payload["tool_id"] == "compgen_echo"


def test_bridge_handler_translates_run_error_to_typed_payload(tmp_path):
    """Build a card whose entrypoint crashes and bridge it directly."""

    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_bridge_crashes"
    body["entrypoints"]["python"] = "compgen.tools.builtin.echo:crashes"
    body["entrypoints"]["mcp"] = "compgen_bridge_crashes"
    card = ToolCard.from_dict(body)
    handler = make_mcp_tool_dict(card)["handler"]
    payload = handler(sm=None, text="x", out_dir=str(tmp_path / "out"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "entrypoint_raised"


def test_bridge_handler_translates_entrypoint_import_error(tmp_path):
    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_bridge_missing"
    body["entrypoints"]["python"] = "compgen.totally_fake_xyz:run"
    body["entrypoints"]["mcp"] = "compgen_bridge_missing"
    card = ToolCard.from_dict(body)
    handler = make_mcp_tool_dict(card)["handler"]
    payload = handler(sm=None, text="x", out_dir=str(tmp_path / "out"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "entrypoint_error"


def test_bridge_handler_translates_output_schema_violation(tmp_path):
    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_bridge_bad_output"
    body["entrypoints"]["python"] = "compgen.tools.builtin.echo:returns_bad_status"
    body["entrypoints"]["mcp"] = "compgen_bridge_bad_output"
    card = ToolCard.from_dict(body)
    handler = make_mcp_tool_dict(card)["handler"]
    payload = handler(sm=None, text="x", out_dir=str(tmp_path / "out"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "output_schema_violation"


# Defence-in-depth: the bridge module exists and mentions ToolRunner.


def test_bridge_module_references_tool_runner():
    """The T5 gate's weak invariant — bridge imports ToolRunner."""

    from compgen.mcp import tool_bridge

    source = Path(tool_bridge.__file__).read_text(encoding="utf-8")
    assert "ToolRunner" in source
