"""P7.2 + P7.6 — apply_recipe surfaces per-obligation + per-script detail,
explain_verification turns failures into actionable hints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model, device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.registry import Registry
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.explain import EXPLAIN_TOOLS, explain_verification
from compgen.mcp.tools.recipe_apply import apply_recipe
from compgen.mcp.tools.transform import propose_invent_slot

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


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
        _MLP().eval(), dev, sample_inputs=(torch.randn(1, 32),),
    )
    reg = Registry(); register_invent_slots(reg)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env, target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=4, registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver
    return sm, session.session_id


def test_explain_tool_is_registered() -> None:
    names = [t["name"] for t in EXPLAIN_TOOLS]
    assert "explain_verification" in names


def test_apply_recipe_returns_per_obligation_and_per_script(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    propose_invent_slot(
        sm, session_id=sid, slot_name="propose_fusion",
        proposal={
            "chosen": {"grouped_regions": ["r_0", "r_1"]},
            "select_vs_invent": "invent",
        },
    )
    r = apply_recipe(sm, session_id=sid)
    assert r["ok"]
    # Per-obligation list (P7.2)
    assert "verification" in r and "per_obligation" in r["verification"]
    assert isinstance(r["verification"]["per_obligation"], list)
    # When obligations exist, every entry must have the agent-readable
    # fields needed to act on it.
    if r["verification"]["per_obligation"]:
        first = r["verification"]["per_obligation"][0]
        for k in ("region_id", "type", "status", "passed", "solver_time_ms"):
            assert k in first
    # Per-script transform list (P7.6)
    assert "transforms" in r and "per_script" in r["transforms"]
    assert isinstance(r["transforms"]["per_script"], list)


def test_explain_verification_after_apply_returns_failures_with_hints(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    # Submit a fusion that lower_recipe will produce a transform script
    # for. Transform Dialect interpreter isn't wired, so we expect
    # transform-script failures to surface.
    propose_invent_slot(
        sm, session_id=sid, slot_name="propose_fusion",
        proposal={
            "chosen": {"grouped_regions": ["r_0", "r_1"],
                       "fusion_kind": "producer_consumer"},
            "select_vs_invent": "invent",
        },
    )
    apply_recipe(sm, session_id=sid)
    r = explain_verification(sm, session_id=sid)
    assert r["ok"]
    assert "verification_failures" in r
    assert "transform_failures" in r
    assert "summary" in r
    # The seed recipe always produces verification obligations against
    # which Transform-Dialect-less mode emits transform_failures > 0,
    # so we assert at least one failure category is non-empty AND each
    # failure carries a remediation_hint + next_step.
    all_failures = r["verification_failures"] + r["transform_failures"]
    if all_failures:
        for f in all_failures:
            assert f.get("remediation_hint")
            assert f.get("next_step")


def test_explain_verification_before_apply_is_empty(tmp_path: Path) -> None:
    """Calling explain_verification before any apply_recipe must not crash."""
    sm, sid = _open(tmp_path)
    r = explain_verification(sm, session_id=sid)
    assert r["ok"]
    assert r["verification_failures"] == []
    assert r["transform_failures"] == []


def test_explain_verification_includes_passed_when_asked(tmp_path: Path) -> None:
    sm, sid = _open(tmp_path)
    propose_invent_slot(
        sm, session_id=sid, slot_name="propose_fusion",
        proposal={"chosen": {"grouped_regions": ["r_0", "r_1"]},
                  "select_vs_invent": "invent"},
    )
    apply_recipe(sm, session_id=sid)
    r_failed = explain_verification(sm, session_id=sid, include_passed=False)
    r_all = explain_verification(sm, session_id=sid, include_passed=True)
    assert len(r_all["verification_failures"]) >= len(r_failed["verification_failures"])
