"""P7.3 — `suggest_proposals` MCP tool: end-to-end via the MCP surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.registry import Registry
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.batch import batch_propose
from compgen.mcp.tools.suggest import SUGGEST_TOOLS, suggest_proposals
from compgen.mcp.tools.transform import propose_invent_slot

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _open(tmp_path: Path) -> tuple[SessionManager, str]:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _MLP().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    reg = Registry()
    register_invent_slots(reg)
    env = compiled.create_agent_env(budget=8)
    driver = LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=8,
        registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver
    return sm, session.session_id


def test_suggest_tool_is_registered() -> None:
    names = [t["name"] for t in SUGGEST_TOOLS]
    assert "suggest_proposals" in names


def test_suggest_proposals_returns_candidates_with_proposal_payload(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = suggest_proposals(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        k=5,
    )
    assert r["ok"]
    if r["candidate_count"] == 0:
        pytest.skip("no fusion pairs detected on this MLP")
    for c in r["candidates"]:
        # Required for ranking + display.
        for k in (
            "rank",
            "chosen",
            "rationale",
            "expected_impact",
            "target_feature_justification",
            "metadata",
            "proposal",
        ):
            assert k in c
        # The pre-built proposal must carry the keys propose_invent_slot needs.
        p = c["proposal"]
        assert "chosen" in p
        assert p.get("select_vs_invent") == "invent"


def test_suggest_proposals_unknown_slot_returns_available(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = suggest_proposals(sm, session_id=sid, slot_name="not_a_slot")
    assert r["ok"]
    assert r["candidates"] == []
    assert r["available_slots"]
    assert r["remediation_hint"]


def test_full_flow_suggest_then_submit(tmp_path: Path) -> None:
    """The headline UX: agent calls suggest, picks one, propose_invent_slot accepts."""
    sm, sid = _open(tmp_path)
    r = suggest_proposals(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        k=3,
    )
    if r["candidate_count"] == 0:
        pytest.skip("no candidates to test the round-trip with")
    pick = r["candidates"][0]["proposal"]
    accept = propose_invent_slot(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        proposal=pick,
    )
    assert accept["status"] == "accepted", accept.get("gate_result")


def test_full_flow_suggest_then_batch_submit(tmp_path: Path) -> None:
    """Suggest k → batch_propose all → accepted count matches."""
    sm, sid = _open(tmp_path)
    r = suggest_proposals(
        sm,
        session_id=sid,
        slot_name="propose_fusion",
        k=4,
    )
    if r["candidate_count"] == 0:
        pytest.skip("no candidates")
    proposals = [{"slot_name": "propose_fusion", "proposal": c["proposal"]} for c in r["candidates"]]
    batch = batch_propose(sm, session_id=sid, proposals=proposals)
    assert batch["ok"]
    assert batch["accepted"] >= 1
    assert len(batch["results"]) == len(proposals)


def test_suggest_proposals_megakernel_returns_attention_or_fallback(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = suggest_proposals(
        sm,
        session_id=sid,
        slot_name="propose_megakernel_synthesis",
        k=3,
    )
    assert r["ok"]
    # MLP doesn't have an attention block but may have an MLP window or
    # all-matmul fallback. Either way: 0 or more candidates, never crashes.
    assert isinstance(r["candidates"], list)


def test_suggest_proposals_layout_plan_uses_target_tile(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = suggest_proposals(
        sm,
        session_id=sid,
        slot_name="propose_layout_plan",
        k=5,
    )
    assert r["ok"]
    for c in r["candidates"]:
        layout = c["chosen"]["layout"]
        assert layout in {"row_major"} or layout.startswith("blocked_")
