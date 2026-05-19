"""Tests for the KB-v2 loader."""

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest

from xpu_rt.benchmarks.loaders.kb_v2_loader import (
    CostSnapshot,
    cycle_source_for_target,
    load_kb_v2_row,
)


# Small stub mimicking AgentLoopResult + Candidate without importing
# the real ones (keeps the test hermetic).
@dataclass
class _StubReport:
    correct: bool
    score: float = 0.0
    cycles: int | None = None


@dataclass
class _StubProposal:
    kernel_code: str = ""
    language: str = "c"
    action: str = "tile-ws-dataflow"


@dataclass
class _StubCandidate:
    attempt: int
    proposal: _StubProposal
    report: _StubReport


@dataclass
class _StubResult:
    best: _StubCandidate | None
    history: list = None
    aborted: bool = False
    abort_reason: str = ""

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def test_cycle_source_for_target_dispatches_correctly() -> None:
    assert cycle_source_for_target("gemmini_mx") == "MAIN_LD_ST_EX_CYCLES"
    assert cycle_source_for_target("saturn_opu_v128") == "mcycle"
    assert cycle_source_for_target("opu_v128_alt") == "mcycle"


def test_load_kb_v2_row_correct_run() -> None:
    cand = _StubCandidate(
        attempt=2, proposal=_StubProposal(),
        report=_StubReport(correct=True, cycles=12251),
    )
    result = _StubResult(best=cand, history=[cand])
    row = load_kb_v2_row(
        result, target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 320]", repeat=0,
        cost=CostSnapshot(cost_usd=0.012, tokens_in=2000, tokens_out=400, wall_s=18.5),
    )
    assert row.backend == "kb-v2"
    assert row.correctness is True
    assert row.cycles == 12251
    assert row.rounds_used == 3  # attempt is 0-based; rounds_used is 1-based
    assert row.cost_usd == 0.012
    assert row.tokens_in == 2000
    assert row.cycle_source == "MAIN_LD_ST_EX_CYCLES"
    assert "tile-ws-dataflow" in row.notes


def test_load_kb_v2_row_no_candidate() -> None:
    """When the agent loop never finds a correct candidate, the row
    must still serialise with correctness=False, cycles=None, and
    a notes field that captures the failure mode."""
    result = _StubResult(best=None, history=[None, None, None])
    row = load_kb_v2_row(
        result, target="saturn_opu_v128", workload="smolvla_mlp_block",
        shape_id="action_expert.layer0.mlp", repeat=2,
        cost=CostSnapshot(cost_usd=0.03, tokens_in=5000, tokens_out=800, wall_s=42.0),
    )
    assert row.correctness is False
    assert row.cycles is None
    assert row.rounds_used == 3
    assert row.cycle_source == "none"
    assert "no candidate" in row.notes


def test_load_kb_v2_row_aborted_run() -> None:
    """An aborted loop (budget exceeded etc.) must surface the
    abort_reason in the notes."""
    result = _StubResult(best=None, aborted=True, abort_reason="budget_exceeded")
    row = load_kb_v2_row(
        result, target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[64, 32]×[32, 720]", repeat=0,
        cost=CostSnapshot(cost_usd=0.5),
    )
    assert row.correctness is False
    assert "budget_exceeded" in row.notes
    assert "aborted=True" in row.notes


def test_load_kb_v2_row_saturn_uses_mcycle() -> None:
    cand = _StubCandidate(
        attempt=0, proposal=_StubProposal(),
        report=_StubReport(correct=True, cycles=22899),
    )
    row = load_kb_v2_row(
        _StubResult(best=cand, history=[cand]),
        target="saturn_opu_v128", workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 720]", repeat=0,
        cost=CostSnapshot(),
    )
    assert row.cycle_source == "mcycle"
    assert row.cycles == 22899
