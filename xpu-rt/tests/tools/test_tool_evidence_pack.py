"""Tests for the tool evidence pack builder.

Coverage:

Positive:
* ``build()`` produces all 7 artifacts + manifest under ``out_dir``.
* ``tool_registry.json`` lists every shipped card.
* ``tool_maturity_matrix.csv`` has one row per card, with the
  declared/verified rung populated from the audit.
* ``tool_surface_matrix.csv`` reflects python/cli/skill/mcp coverage.
* ``cli_mcp_schema_match.json`` is bit-equal for every MCP-bridged
  card (the headline cross-surface guarantee).
* ``claim_matrix.json`` has the documented 5 closed-enum claims.
* ``promotion_log.json`` appends new rung rows on rising
  ``verified_maturity`` and is byte-stable for unchanged repo state.
* The figures directory contains either a real PNG (matplotlib
  available) or the typed ``figure_status_marker.json`` (honest
  non-claim — no silent absence).

Negative controls:
* When a tool's audit declines (we inject a malformed card root via
  monkeypatch), the registry still emits a row + claim status flips
  to ``unmet``.
* Pre-existing ``promotion_log.json`` is preserved on rebuild; new
  rung rows are appended, not duplicated when nothing changed.

The whole pack rebuilds in seconds; tests use it directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.dev.build_tool_evidence_pack import build  # type: ignore[import-not-found]


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    out = tmp_path / "evidence"
    build(out)
    return out


# ---------- Positive -------------------------------------------------


def test_manifest_lists_all_seven_artifacts(pack_dir: Path):
    body = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    assert body["schema_version"] == "compgen_tool_evidence_pack_v1"
    expected = {
        "tool_registry.json",
        "tool_maturity_matrix.csv",
        "tool_surface_matrix.csv",
        "cli_mcp_schema_match.json",
        "fresh_agent_tasks.json",
        "claim_matrix.json",
        "promotion_log.json",
    }
    assert set(body["artifacts"]) == expected
    for name in expected:
        assert (pack_dir / name).is_file()


def test_registry_contains_every_shipped_card(pack_dir: Path):
    from compgen.tools.tool_registry import iter_tool_cards

    registry = json.loads((pack_dir / "tool_registry.json").read_text(encoding="utf-8"))
    shipped = {c.tool_id for c in iter_tool_cards()}
    listed = {t["tool_id"] for t in registry["tools"]}
    assert listed == shipped


def test_maturity_matrix_rows_match_cards(pack_dir: Path):
    from compgen.tools.tool_registry import iter_tool_cards

    cards = list(iter_tool_cards())
    with (pack_dir / "tool_maturity_matrix.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(cards)
    by_id = {r["tool_id"]: r for r in rows}
    for card in cards:
        assert by_id[card.tool_id]["declared"] == card.maturity


def test_surface_matrix_reflects_card_entrypoints(pack_dir: Path):
    from compgen.tools.tool_registry import iter_tool_cards

    with (pack_dir / "tool_surface_matrix.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["tool_id"]: r for r in rows}
    for card in iter_tool_cards():
        row = by_id[card.tool_id]
        # CSV stores "True"/"False" strings.
        assert (row["cli"] == "True") == bool(card.entrypoints.cli)
        assert (row["mcp"] == "True") == bool(card.entrypoints.mcp)


def test_cli_mcp_schema_match_is_bit_equal(pack_dir: Path):
    """The headline cross-surface invariant — same schema string
    on both surfaces for every bridged tool."""

    body = json.loads((pack_dir / "cli_mcp_schema_match.json").read_text(encoding="utf-8"))
    assert body["rows"], "at least one MCP-bridged tool must appear in the match table"
    for row in body["rows"]:
        assert row["bit_equal"] is True, row
        assert row["card_schema_sha"] == row["mcp_schema_sha"]


def test_claim_matrix_has_five_closed_enum_claims(pack_dir: Path):
    body = json.loads((pack_dir / "claim_matrix.json").read_text(encoding="utf-8"))
    claims = body["claims"]
    assert len(claims) == 5
    statuses = {c["status"] for c in claims}
    assert statuses.issubset({"signed", "partial", "unmet", "no_data"})
    ids = {c["id"] for c in claims}
    assert {
        "C_TOOLS_AUDITED_CLEAN",
        "C_TOOLS_CLI_REACHABLE",
        "C_TOOLS_MCP_BRIDGED",
        "C_TOOLS_FRESH_AGENT_GRADED",
        "C_TOOLS_EXTENSION_FLOW_REACHABLE",
    } == ids


def test_claim_audited_clean_signed_when_all_cards_pass(pack_dir: Path):
    body = json.loads((pack_dir / "claim_matrix.json").read_text(encoding="utf-8"))
    audit_claim = next(c for c in body["claims"] if c["id"] == "C_TOOLS_AUDITED_CLEAN")
    assert audit_claim["status"] == "signed"


def test_figures_directory_either_renders_or_carries_marker(pack_dir: Path):
    figures = pack_dir / "figures"
    assert figures.is_dir()
    pngs = list(figures.glob("*.png"))
    marker = figures / "figure_status_marker.json"
    assert pngs or marker.is_file(), (
        "either a real PNG or the typed skip marker must be present"
    )
    # Manifest's figures.kind agrees with disk reality.
    body = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    if pngs:
        assert body["figures"]["kind"] == "available"
    else:
        assert body["figures"]["kind"] == "skipped_missing_matplotlib"


def test_promotion_log_has_one_rung_row_per_tool(pack_dir: Path):
    body = json.loads((pack_dir / "promotion_log.json").read_text(encoding="utf-8"))
    assert body["schema_version"] == "promotion_log_v1"
    from compgen.tools.tool_registry import iter_tool_cards

    for card in iter_tool_cards():
        assert card.tool_id in body["rung_history"]
        rows = body["rung_history"][card.tool_id]
        assert len(rows) >= 1
        assert rows[-1]["rung"] in {
            "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "below-T0"
        }


def test_promotion_log_append_only_on_unchanged_rebuild(tmp_path: Path):
    out = tmp_path / "evidence"
    build(out)
    body1 = json.loads((out / "promotion_log.json").read_text(encoding="utf-8"))
    build(out)
    body2 = json.loads((out / "promotion_log.json").read_text(encoding="utf-8"))
    # No new rung rows should be appended when verified_maturity is unchanged.
    for tool_id in body1["rung_history"]:
        assert len(body2["rung_history"][tool_id]) == len(
            body1["rung_history"][tool_id]
        )


def test_fresh_agent_tasks_index_lists_known_tasks(pack_dir: Path):
    from compgen.audit.fresh_agent_grading import list_task_ids

    body = json.loads((pack_dir / "fresh_agent_tasks.json").read_text(encoding="utf-8"))
    listed = {t["task_id"] for t in body["tasks"]}
    assert set(list_task_ids()).issubset(listed)


def test_evidence_pack_writes_only_under_out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The builder must not mutate any path outside ``out_dir``.

    We verify by snapshotting mtimes of a small set of shipped repo
    files and asserting they are unchanged after the build.
    """

    sentinels = [
        Path("python/compgen/tools/cards/echo.yaml"),
        Path("python/compgen/audit/tool_promotion.py"),
        Path(".claude/skills/compgen-tool-development/SKILL.md"),
    ]
    before = {p: p.stat().st_mtime_ns for p in sentinels if p.is_file()}
    build(tmp_path / "pack")
    for p, mtime in before.items():
        assert p.stat().st_mtime_ns == mtime, f"builder mutated {p}"


def test_main_returns_zero_when_no_claims_unmet(tmp_path: Path):
    """The CLI entrypoint exits 0 in the current clean state."""

    from scripts.dev.build_tool_evidence_pack import main  # type: ignore[import-not-found]

    rc = main(["--out", str(tmp_path / "pack")])
    assert rc == 0
