"""G4 wire-in tests — Strategist + Tactician plumbed into ``LLMDrivenCompiler``.

Coverage:

1. With ``COMPGEN_USE_STRATEGIST_TACTICIAN=1``, the first
   ``current_view`` call materialises a non-empty :class:`Plan` on
   ``driver._plan`` with a valid fallback ladder for every region.
2. With the flag *off* (default), ``driver._plan`` stays ``None``
   and the driver is byte-identical to the pre-G4 codepath.
3. A ``step_proposal`` call that targets a known region while the
   wire-in is on appends an audit entry to
   ``driver._tactician_audit`` carrying the Tactician's decision.
4. With the flag off, no Tactician audit is recorded — proves the
   wire-in is fully gated.
5. The plan is idempotent: re-calling ``current_view`` does not
   rebuild ``_plan`` (a Strategist hiccup must not double-charge).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.llm.mock_client import MockLLMClient

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen"
    / "exemplars"
    / "test_gpu_simt.yaml"
)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_driver() -> LLMDrivenCompiler:
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _TinyMLP().eval(),
        dev,
        sample_inputs=(torch.randn(1, 16),),
    )
    env = compiled.create_agent_env(budget=4)
    return LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=4,
    )


# ----------------------------------------------------------------------
# Positive: flag on → Plan materialised, audit logged
# ----------------------------------------------------------------------


def test_plan_materialises_with_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPGEN_USE_STRATEGIST_TACTICIAN", "1")
    drv = _make_driver()
    assert drv._plan is None
    _ = drv.current_view()
    # Plan exists; every region has a non-empty fallback ladder.
    assert drv._plan is not None
    assert len(drv._plan.region_partition) >= 1
    for rp in drv._plan.region_partition:
        assert len(rp.fallback_ladder) >= 1
        assert rp.tactic in rp.fallback_ladder


def test_plan_init_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPGEN_USE_STRATEGIST_TACTICIAN", "1")
    drv = _make_driver()
    _ = drv.current_view()
    first_plan = drv._plan
    _ = drv.current_view()
    second_plan = drv._plan
    assert first_plan is second_plan


# ----------------------------------------------------------------------
# Negative: flag off → no Plan, no audit
# ----------------------------------------------------------------------


def test_plan_stays_none_with_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPGEN_USE_STRATEGIST_TACTICIAN", raising=False)
    drv = _make_driver()
    _ = drv.current_view()
    assert drv._plan is None
    assert drv._tactician_audit == []


def test_no_tactician_audit_with_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPGEN_USE_STRATEGIST_TACTICIAN", raising=False)
    drv = _make_driver()
    _ = drv.current_view()
    # An arbitrary proposal targeting any region must NOT touch
    # _tactician_audit when the flag is off.
    target = drv.env._regions[0].region_id if drv.env._regions else ""
    drv.step_proposal(action_type="fuse_elementwise", target=target)
    assert drv._tactician_audit == []


# ----------------------------------------------------------------------
# Positive: audit recorded when the flag is on and the proposal
# names a known region
# ----------------------------------------------------------------------


def test_step_proposal_logs_tactician_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPGEN_USE_STRATEGIST_TACTICIAN", "1")
    drv = _make_driver()
    _ = drv.current_view()
    if not drv.env._regions:
        pytest.skip("no regions extracted from this MLP — env-dependent")
    target = drv.env._regions[0].region_id
    drv.step_proposal(action_type="fuse_elementwise", target=target)
    assert len(drv._tactician_audit) >= 1
    entry = drv._tactician_audit[-1]
    assert entry["region_id"] == target
    assert entry["agent_proposal"] == "fuse_elementwise"
    assert entry["tactician_action"] in ("apply", "escalate", "exhausted")
    assert "matches_agent" in entry
