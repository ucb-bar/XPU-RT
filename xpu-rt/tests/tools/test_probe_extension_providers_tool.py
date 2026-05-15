"""Tests for the `compgen_probe_extension_providers` ToolCard (P1 T6 promotion).

Coverage:

Positive:
* The ToolCard loads cleanly and declares T6 maturity.
* Running the wrapper end-to-end on real hardware (no env shortcuts)
  produces all seven probe artifacts, returns status=ok, and the
  audit verifies the tool at T6.

Negative controls:
* Unknown input field is rejected by the input_schema before the
  wrapper runs (no probe subprocess is started).
* timeout_s=0.0001 blocks with a typed `probe_timeout` reason — the
  subprocess can't even start in that window, so the wrapper must
  return status=blocked rather than silently succeeding.
* The shipped fresh-agent task still has `last_grading_result.passed=true`
  on disk — the T6 evidence persists across reruns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.tools.errors import ToolInputSchemaError
from compgen.tools.tool_registry import load_tool_card, tool_cards_root
from compgen.tools.tool_runner import ToolRunner


def _card():
    return load_tool_card(tool_cards_root() / "probe_extension_providers.yaml")


def test_card_declares_t6():
    card = _card()
    assert card.tool_id == "compgen_probe_extension_providers"
    assert card.maturity == "T6"
    assert card.entrypoints.cli == "compgen-tool run compgen_probe_extension_providers"
    assert card.entrypoints.mcp == "compgen_probe_extension_providers"
    assert card.skill_path == ".claude/skills/compgen-provider-integration/SKILL.md"
    assert card.fresh_agent_task_id == "probe_providers_v1"


def test_probe_runs_clean(tmp_path):
    """End-to-end on real hardware via ToolRunner."""

    card = _card()
    result = ToolRunner().run(card, request={}, out_dir=tmp_path / "probe_run")
    assert result.status == "ok", result.result
    assert result.result["provider_count"] >= 1
    # All 7 contracted artifacts produced.
    assert len(result.artifacts) == 7
    for name in (
        "provider_status.json",
        "target_status.json",
        "dialect_status.json",
        "pass_tool_status.json",
        "provider_target_matrix.csv",
        "provider_contract_matrix.csv",
        "probe_summary.md",
    ):
        assert (tmp_path / "probe_run" / name).is_file()


def test_probe_blocks_on_timeout(tmp_path):
    """The smallest schema-legal timeout (1 ms) cannot complete the
    probe subprocess; the wrapper must surface `status=blocked` with
    the typed `probe_timeout` reason rather than silently succeeding."""

    card = _card()
    result = ToolRunner().run(
        card,
        request={"timeout_s": 0.001},
        out_dir=tmp_path / "probe_timeout",
    )
    assert result.status == "blocked"
    assert result.result["reason"] == "probe_timeout"


def test_probe_rejects_unknown_input_field(tmp_path):
    """input_schema declares ``additionalProperties: false``."""

    card = _card()
    with pytest.raises(ToolInputSchemaError):
        ToolRunner().run(
            card,
            request={"definitely_not_real": "x"},
            out_dir=tmp_path / "probe_bad",
        )


def test_audit_verifies_t6():
    """The audit must mark this tool at verified=T6.

    Pre-condition: ``last_grading_result.json`` under
    ``.rcg-artifacts/fresh_agent_tasks/probe_providers_v1/`` shows
    ``passed=true`` (committed alongside this card).
    """

    from compgen.audit.tool_promotion import run_tool_promotion_audit

    report = run_tool_promotion_audit()
    outcome = next(
        o for o in report.outcomes if o.tool_id == "compgen_probe_extension_providers"
    )
    assert outcome.verified_maturity == "T6", [v.to_dict() for v in outcome.violations]


def test_fresh_agent_grading_sidecar_passed():
    """The shipped fresh-agent grading sidecar is committed clean."""

    repo_root = Path(__file__).resolve().parents[2]
    sidecar = (
        repo_root
        / ".rcg-artifacts"
        / "fresh_agent_tasks"
        / "probe_providers_v1"
        / "last_grading_result.json"
    )
    assert sidecar.is_file()
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    assert body["passed"] is True
    assert body["task_id"] == "probe_providers_v1"
