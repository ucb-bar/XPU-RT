"""Tests for the LLM-driven runtime scheduler."""

from __future__ import annotations

from compgen.llm.mock_client import MockLLMClient
from compgen.runtime.llm_scheduler import LLMScheduler, SchedulingDecision


def _make_mock_client() -> MockLLMClient:
    client = MockLLMClient(strict=False)
    client.add_response("runtime scheduling", '{"decision": "re_solve", "reason": "high drift", "parameters": {}}')
    return client


def test_llm_scheduler_decide() -> None:
    """LLM scheduler makes a decision."""
    client = _make_mock_client()
    scheduler = LLMScheduler(llm_client=client)

    decision = scheduler.decide(
        measured_latency_us=100.0,
        estimated_latency_us=50.0,
        device_utilization={"gpu0": 80.0, "cpu0": 20.0},
        batch_size=32,
    )

    assert isinstance(decision, SchedulingDecision)
    assert decision.action in {"keep", "re_solve", "change_batch_tier", "migrate_ops", "throttle"}
    assert decision.reason


def test_llm_scheduler_tracks_history() -> None:
    """Scheduler records decision history."""
    client = _make_mock_client()
    scheduler = LLMScheduler(llm_client=client)

    scheduler.decide(measured_latency_us=100, estimated_latency_us=80)
    scheduler.decide(measured_latency_us=90, estimated_latency_us=80)

    assert len(scheduler.decision_history) == 2


def test_llm_scheduler_reset() -> None:
    """Scheduler history can be cleared."""
    client = _make_mock_client()
    scheduler = LLMScheduler(llm_client=client)
    scheduler.decide(measured_latency_us=100, estimated_latency_us=80)
    scheduler.reset_history()
    assert len(scheduler.decision_history) == 0


def test_llm_scheduler_should_re_solve() -> None:
    """Quick heuristic check for re-solve threshold."""
    client = _make_mock_client()
    scheduler = LLMScheduler(llm_client=client)
    assert scheduler.should_re_solve(25.0)
    assert not scheduler.should_re_solve(10.0)


def test_llm_scheduler_fallback_on_error() -> None:
    """Scheduler falls back to 'keep' on LLM error."""
    client = MockLLMClient(strict=True)  # Will raise on any prompt
    scheduler = LLMScheduler(llm_client=client)

    # Should not raise, should return "keep"
    decision = scheduler.decide(measured_latency_us=100, estimated_latency_us=80)
    assert decision.action == "keep"


def test_llm_scheduler_high_drift_triggers_re_solve() -> None:
    """When drift is high, LLM should suggest re-solve."""
    client = MockLLMClient(strict=False)
    client.add_response("runtime scheduling", '{"decision": "re_solve", "reason": "100% drift"}')

    scheduler = LLMScheduler(llm_client=client)
    decision = scheduler.decide(
        measured_latency_us=200.0,
        estimated_latency_us=100.0,
    )
    assert decision.action == "re_solve"
