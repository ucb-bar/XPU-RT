"""Agent-experience tests for the MCP surface.

These are not unit tests of individual functions — they're scenario
tests that simulate what a Claude-Code-style agent does with the MCP
tools, and assert the surface stays *helpful* (not just functional):

* Cold-start → can the agent discover what to call?
* Typos → does the surface tell me what was meant?
* Region naming → can the agent translate between recipe-level and
  payload-level region names without guessing?
* Mid-session diff → can the agent see what its own decisions did?
* End-to-end → does a multi-step session with real region picks
  produce a structurally different bundle?

If any of these regress, the agent loses the ability to recover from
its own mistakes, which is the whole point of MCP-driven compilation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.mcp.session import SessionManager
from compgen.mcp.tools import ALL_TOOLS
from compgen.mcp.tools.inspect import (
    diff_recipe,
    get_dossier,
    list_phase_tools,
    view_recipe,
)
from compgen.mcp.tools.lifecycle import bundle_export
from compgen.mcp.tools.recipe_apply import apply_recipe
from compgen.mcp.tools.transform import (
    invoke_tool,
    propose_invent_slot,
    step_proposal,
)

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _TwoLinear(nn.Module):
    """Two stacked linear layers — gives the recipe several regions."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _open_session(tmp_path: Path) -> tuple[SessionManager, str]:
    """Open a real session with a real model + driver."""
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _TwoLinear().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        transcript_dir=session.scratch_dir / "transcripts",
        budget=4,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver
    return sm, session.session_id


# ---------------------------------------------------------------------------
# UX-1: discoverability after load_model
# ---------------------------------------------------------------------------


def test_slots_auto_register_on_session_open(tmp_path: Path) -> None:
    """An agent that never calls register_invent_slots explicitly must
    still find the canonical slots via list_phase_tools after open."""
    sm, sid = _open_session(tmp_path)
    r = list_phase_tools(sm, session_id=sid)
    slot_names = [s["name"] for s in r["invent_slots"]]
    # The 9 canonical slots (defined in invent_slots/registrar.py) must
    # all appear without any explicit registration call by the agent.
    for must_have in (
        "propose_fusion",
        "propose_layout_plan",
        "propose_megakernel_synthesis",
        "propose_dequant_fusion",
    ):
        assert must_have in slot_names, f"slot {must_have!r} missing — agent has nothing to propose!"


def test_list_phase_tools_works_with_session_id(tmp_path: Path) -> None:
    """The catalogue endpoint must accept (and tolerate) a session_id arg
    so the agent can pass it uniformly with every other call."""
    sm, sid = _open_session(tmp_path)
    r = list_phase_tools(sm, session_id=sid)
    assert r["ok"] is True
    assert "tools" in r and "invent_slots" in r


# ---------------------------------------------------------------------------
# UX-2: typos / fuzzy recovery
# ---------------------------------------------------------------------------


def test_unknown_slot_returns_available_list_and_nearest(tmp_path: Path) -> None:
    """A typo'd slot name must come back with (a) the full available
    list and (b) edit-distance suggestions, not a bare 'unknown'."""
    sm, sid = _open_session(tmp_path)
    r = propose_invent_slot(
        sm,
        session_id=sid,
        slot_name="propse_fusion",  # typo
        proposal={"chosen": {}, "select_vs_invent": "invent"},
    )
    assert r["status"] == "unknown"
    assert r["remediation_hint"] is not None
    tr = r.get("tool_result") or {}
    assert tr.get("nearest"), "no nearest-match suggestions"
    assert "propose_fusion" in tr["nearest"]
    assert tr.get("available_slots"), "no available_slots in tool_result"


def test_unknown_tool_returns_available_list(tmp_path: Path) -> None:
    sm, sid = _open_session(tmp_path)
    r = invoke_tool(sm, session_id=sid, tool_name="not_a_real_tool")
    assert r["status"] == "unknown"
    assert r.get("remediation_hint") is not None
    tr = r.get("tool_result") or {}
    assert "available_tools" in tr


# ---------------------------------------------------------------------------
# UX-3: region naming — both name forms must be discoverable
# ---------------------------------------------------------------------------


def test_view_recipe_middle_carries_sym_name_so_agent_sees_region_ids(
    tmp_path: Path,
) -> None:
    """An agent paging through view_recipe.middle must see sym_name on
    every recipe.region row — without it, the agent can't construct
    a propose_fusion against a region it found in the middle section."""
    sm, sid = _open_session(tmp_path)
    r = view_recipe(sm, session_id=sid, max_ops=200)
    middle = r["view"]["middle"]
    region_rows = [m for m in middle if m.get("_op") == "recipe.region"]
    assert region_rows, "expected recipe.region rows in middle"
    with_sym = [m for m in region_rows if "sym_name" in m]
    assert len(with_sym) == len(region_rows), "every region row must include sym_name, agent can't address it otherwise"


def test_get_dossier_exposes_recipe_to_payload_translation(tmp_path: Path) -> None:
    """Agents see payload region names in the dossier (mm_1, rmsnorm_0)
    but propose_invent_slot expects recipe sym names (r_0/r_1). The
    region_map must bridge them and carry the role tag."""
    sm, sid = _open_session(tmp_path)
    r = get_dossier(sm, session_id=sid)
    assert "region_map" in r, "dossier must expose a recipe sym → payload id translation"
    rmap = r["region_map"]
    assert rmap, "region_map is empty"
    # New shape: {sym: {payload_id, role?}}.
    for sym, info in list(rmap.items())[:3]:
        assert sym.startswith("r_"), f"unexpected sym format: {sym}"
        assert isinstance(info, dict)
        assert info.get("payload_id"), f"empty payload_id for {sym}"
    # P7.1: at least one region must carry a non-empty role.
    roles_present = [info.get("role") for info in rmap.values() if info.get("role")]
    assert roles_present, "no role tags propagated from compgen._pattern_hint — P7.1 regression"
    # Reverse-index must agree with forward.
    assert "regions_by_role" in r
    for role, syms in r["regions_by_role"].items():
        for s in syms:
            assert rmap[s]["role"] == role


# ---------------------------------------------------------------------------
# UX-4: agent decisions surface in the IR diff
# ---------------------------------------------------------------------------


def test_propose_then_diff_shows_added_op(tmp_path: Path) -> None:
    """The basic feedback loop: propose → diff_recipe(from='ckpt_0') must
    return the added propose_fusion op so the agent can confirm its
    decision actually landed."""
    sm, sid = _open_session(tmp_path)
    propose_invent_slot(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        proposal={
            "chosen": {"grouped_regions": ["r_0", "r_1"]},
            "select_vs_invent": "invent",
        },
    )
    r = diff_recipe(sm, session_id=sid, from_ckpt="ckpt_0")
    added_ops = {e["_op"] for e in r["diff"]["added"]}
    assert "recipe.propose_fusion" in added_ops


def test_apply_recipe_returns_mutation_report(tmp_path: Path) -> None:
    """After apply_recipe, the agent needs to know what concretely
    happened — not just 'ok'. Mutation report must surface counts."""
    sm, sid = _open_session(tmp_path)
    propose_invent_slot(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        proposal={
            "chosen": {"grouped_regions": ["r_0", "r_1"]},
            "select_vs_invent": "invent",
        },
    )
    r = apply_recipe(sm, session_id=sid)
    assert r["ok"]
    assert "mutation_report" in r
    mr = r["mutation_report"]
    # The report must carry the per-kind counters; even zero entries
    # are acceptable as long as the keys exist (so the agent can do
    # `mr["fusions_applied"]` without a KeyError safety dance).
    for k in ("fusions_applied", "structural_fusions", "structural_callees_added", "payload_ops_touched"):
        assert k in mr


# ---------------------------------------------------------------------------
# UX-5: end-to-end — agent decisions reach the bundle
# ---------------------------------------------------------------------------


def test_full_agent_loop_changes_bundle_payload(tmp_path: Path) -> None:
    """This is the headline UX claim: agent → propose → apply → bundle
    yields a payload.mlir that's BYTE-DIFFERENT from the no-proposal
    baseline. If this regresses, the agent's surface is theatre."""
    sm, sid = _open_session(tmp_path)
    base = bundle_export(
        sm,
        session_id=sid,
        output_dir=str(tmp_path / "base"),
    )
    base_payload = (Path(base["path"]) / "payload.mlir").read_text()

    propose_invent_slot(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        proposal={
            "chosen": {"grouped_regions": ["r_0", "r_1"]},
            "select_vs_invent": "invent",
        },
    )
    apply_recipe(sm, session_id=sid)

    after = bundle_export(
        sm,
        session_id=sid,
        output_dir=str(tmp_path / "after"),
    )
    after_payload = (Path(after["path"]) / "payload.mlir").read_text()
    assert base["sha"] != after["sha"], "bundle SHA unchanged"
    assert base_payload != after_payload, "payload.mlir bytes unchanged"


def test_step_proposal_unknown_action_is_noop_not_crash(tmp_path: Path) -> None:
    """A bogus action_type must come back as 'noop' with the agent
    still able to keep going — not a stack trace."""
    sm, sid = _open_session(tmp_path)
    r = step_proposal(
        sm,
        session_id=sid,
        action_type="not_a_known_action",
    )
    assert r["ok"] is True
    assert r["status"] == "noop"


# ---------------------------------------------------------------------------
# UX-6: catalogue stability — what an MCP client first sees
# ---------------------------------------------------------------------------


def test_all_tools_have_input_schema_with_required_session_id() -> None:
    """Every tool that needs a session_id must declare it as required —
    otherwise an MCP client that strictly follows JSON Schema can omit
    it and we'd silently dispatch to the wrong session."""
    needs_session = {
        "load_model",
        "compile",
        "bundle_export",
        "view_recipe",
        "diff_recipe",
        "checkpoint",
        "get_dossier",
        "session_summary",
        "diagnose_model_compatibility",
        "invoke_tool",
        "propose_invent_slot",
        "verify_proposal",
        "step_proposal",
        "synthesize_decomp",
        "synthesize_translation",
        "register_blackbox",
        "resolve_unsupported_op",
        "recovery_status",
        "apply_recipe",
        "register_pack",
    }
    for t in ALL_TOOLS:
        if t["name"] not in needs_session:
            continue
        required = t["input_schema"].get("required", [])
        assert "session_id" in required, (
            f"{t['name']!r} doesn't list session_id in 'required' — agents will hit obscure errors when they omit it"
        )


def test_every_tool_has_a_human_readable_description() -> None:
    """Sanity: a description an agent can read in tool-selection."""
    for t in ALL_TOOLS:
        d = t.get("description", "")
        assert isinstance(d, str) and len(d) > 12, f"{t['name']!r} description too short: {d!r}"
