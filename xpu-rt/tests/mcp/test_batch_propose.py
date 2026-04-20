"""P7.4 — ``batch_propose`` runs many invent-slot calls in one MCP roundtrip
+ supports atomic rollback on first rejection.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.registry import Registry
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.batch import BATCH_TOOLS, batch_propose

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


def test_batch_tool_is_registered() -> None:
    names = [t["name"] for t in BATCH_TOOLS]
    assert "batch_propose" in names


def test_batch_propose_happy_path_all_accepted(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    proposals = [
        {
            "slot_name": "propose_fusion",
            "proposal": {
                "chosen": {"grouped_regions": ["r_0", "r_1"]},
                "select_vs_invent": "invent",
            },
        },
        {
            "slot_name": "propose_fusion",
            "proposal": {
                "chosen": {"grouped_regions": ["r_2", "r_3"]},
                "select_vs_invent": "invent",
            },
        },
        {
            "slot_name": "propose_fusion",
            "proposal": {
                "chosen": {"grouped_regions": ["r_4", "r_5"]},
                "select_vs_invent": "invent",
            },
        },
    ]
    r = batch_propose(sm, session_id=sid, proposals=proposals)
    assert r["ok"]
    assert r["accepted"] == 3
    assert r["rejected"] == 0
    assert r["rolled_back"] is False
    assert len(r["results"]) == 3


def test_batch_propose_continues_on_rejection_when_not_atomic(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    proposals = [
        {
            "slot_name": "propose_fusion",
            "proposal": {"chosen": {"grouped_regions": ["r_0", "r_1"]}, "select_vs_invent": "invent"},
        },
        # Bad: missing chosen.grouped_regions → rejected with hint
        {"slot_name": "propose_fusion", "proposal": {"chosen": {}, "select_vs_invent": "invent"}},
        # Should still attempt the third even after the rejection.
        {
            "slot_name": "propose_fusion",
            "proposal": {"chosen": {"grouped_regions": ["r_2", "r_3"]}, "select_vs_invent": "invent"},
        },
    ]
    r = batch_propose(sm, session_id=sid, proposals=proposals, atomic=False)
    assert r["ok"]
    assert r["accepted"] >= 1
    assert r["rejected"] >= 1
    assert r["rolled_back"] is False
    assert len(r["results"]) == 3


def test_batch_propose_atomic_rolls_back_on_rejection(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    session = sm.get(sid)
    driver = session.driver
    assert driver is not None
    # Snapshot the recipe op count before the batch.
    before_ops = len(list(driver.env.recipe.body.block.ops))
    before_payload_text = str(driver.env.payload_module)

    proposals = [
        {
            "slot_name": "propose_fusion",
            "proposal": {"chosen": {"grouped_regions": ["r_0", "r_1"]}, "select_vs_invent": "invent"},
        },
        # Bad — should trigger rollback in atomic mode.
        {"slot_name": "propose_fusion", "proposal": {"chosen": {}, "select_vs_invent": "invent"}},
    ]
    r = batch_propose(sm, session_id=sid, proposals=proposals, atomic=True)
    assert r["ok"]
    assert r["rolled_back"] is True
    assert r["accepted"] == 0  # the first acceptance was rolled back

    # Recipe op count must equal pre-batch.
    after_ops = len(list(driver.env.recipe.body.block.ops))
    assert after_ops == before_ops, f"recipe was mutated after rollback: {before_ops} -> {after_ops}"
    # Payload bytes unchanged.
    after_payload_text = str(driver.env.payload_module)
    assert before_payload_text == after_payload_text


def test_batch_propose_unknown_slot_in_batch(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = batch_propose(
        sm,
        session_id=sid,
        proposals=[
            {
                "slot_name": "definitely_not_a_slot",
                "proposal": {"chosen": {"grouped_regions": ["r_0", "r_1"]}, "select_vs_invent": "invent"},
            },
        ],
    )
    assert r["ok"]
    assert r["accepted"] == 0
    # Result should carry the unknown status + remediation_hint from
    # the per-call _unknown_slot_result path (P7-UX fix).
    assert r["results"][0]["status"] == "unknown"
    assert r["results"][0].get("remediation_hint")


def test_batch_propose_invalid_input_returns_error(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    r = batch_propose(sm, session_id=sid, proposals="not_a_list")  # type: ignore[arg-type]
    assert r["ok"] is False
    assert "must be a list" in r["error"]
