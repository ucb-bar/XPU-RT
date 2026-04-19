"""Tests for :mod:`compgen.agent.llm_driver_recovery`.

Covers the three core paths:
  1. No LLM — deterministic classifier picks every strategy.
  2. LLM consulted on a low-confidence case.
  3. LLM pick fails to apply; fallback kicks in.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.agent.llm_driver_recovery import (
    OpRecoveryDecision,
    RecoveryPlan,
    plan_recovery,
)
from compgen.api import compile_model, device as _device
from compgen.capture.unsupported import UnsupportedOpResolution
from compgen.capture.unsupported.classify import UnsupportedClassification
from compgen.llm.base import GenerationResponse
from compgen.llm.mock_client import MockLLMClient

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


class _TanhModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))


def _capture_artifact():
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _TanhModel().eval(), dev,
        sample_inputs=(torch.randn(1, 32),),
        recover_unsupported=False,   # so plan_recovery is the only pass under test
    )
    return compiled.capture_artifact


def test_plan_recovery_deterministic_without_llm() -> None:
    artifact = _capture_artifact()
    assert len(artifact.unsupported_resolutions) >= 1
    plan = plan_recovery(artifact, llm_client=None)
    assert isinstance(plan, RecoveryPlan)
    assert plan.ok()
    assert plan.llm_consulted == 0
    for d in plan.decisions:
        assert d.source == "classifier"


def test_plan_recovery_llm_consulted_on_low_confidence() -> None:
    """Synthesise a resolution whose classification confidence is 'low'
    so the LLM is consulted. Verify the LLM's pick is honoured."""
    artifact = _capture_artifact()
    # Force a low-confidence classification on the only issue.
    downgraded_resolutions = []
    for r in artifact.unsupported_resolutions:
        forced_cls = UnsupportedClassification(
            bucket="blackbox_boundary",
            strategy="explicit_blackbox",
            confidence="low",
            reason="forced for test",
        )
        downgraded_resolutions.append(replace(r, classification=forced_cls))
    artifact.unsupported_resolutions = downgraded_resolutions  # type: ignore[misc]

    class _MockLLM(MockLLMClient):
        def generate(self, request):
            return GenerationResponse(
                raw_text="translation\npick translation because the schema is Tensor→Tensor",
                parsed_artifacts=["translation"],
                model_id="mock",
            )

    plan = plan_recovery(artifact, llm_client=_MockLLM(strict=False))
    assert plan.llm_consulted >= 1
    llm_decisions = [d for d in plan.decisions if d.source == "llm"]
    assert llm_decisions
    assert all(d.strategy == "translation" for d in llm_decisions)


def test_plan_recovery_llm_bad_answer_falls_back() -> None:
    """LLM returns unparseable text → decision source becomes ``fallback``."""
    artifact = _capture_artifact()
    downgraded_resolutions = []
    for r in artifact.unsupported_resolutions:
        forced_cls = UnsupportedClassification(
            bucket="blackbox_boundary",
            strategy="explicit_blackbox",
            confidence="low",
            reason="forced for test",
        )
        downgraded_resolutions.append(replace(r, classification=forced_cls))
    artifact.unsupported_resolutions = downgraded_resolutions  # type: ignore[misc]

    class _BadLLM(MockLLMClient):
        def generate(self, request):
            return GenerationResponse(
                raw_text="I have no idea honestly",
                parsed_artifacts=[""], model_id="mock",
            )

    plan = plan_recovery(artifact, llm_client=_BadLLM(strict=False))
    assert plan.llm_consulted >= 1
    fallback = [d for d in plan.decisions if d.source == "fallback"]
    assert fallback


def test_plan_recovery_on_empty_artifact_is_noop() -> None:
    """A capture artifact with no unsupported ops yields an empty plan."""
    class _NoIssues(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 8)
        def forward(self, x): return self.fc(x)

    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _NoIssues().eval(), dev, sample_inputs=(torch.randn(1, 16),),
    )
    plan = plan_recovery(compiled.capture_artifact, llm_client=None)
    assert plan.decisions == []
    assert plan.ok()


def test_plan_recovery_to_dict_is_serialisable() -> None:
    artifact = _capture_artifact()
    plan = plan_recovery(artifact, llm_client=None)
    d = plan.to_dict()
    import json
    json.dumps(d)   # must not raise
    assert d["num_issues"] == len(plan.decisions)
