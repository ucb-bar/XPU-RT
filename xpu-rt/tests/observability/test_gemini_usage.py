"""Tests for compgen.observability.gemini_usage."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from compgen.observability import gemini_usage as gu


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    # Make sure repo-root override doesn't pull in real configs.
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Pricing / cost
# ---------------------------------------------------------------------------


def test_compute_cost_known_model_flash() -> None:
    # gemini-2.5-flash: $0.30/1M input, $2.50/1M output
    cost = gu.compute_cost_usd("gemini-2.5-flash", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == pytest.approx(0.30, rel=1e-6)
    cost = gu.compute_cost_usd("gemini-2.5-flash", prompt_tokens=0, completion_tokens=1_000_000)
    assert cost == pytest.approx(2.50, rel=1e-6)


def test_compute_cost_long_context_pro() -> None:
    # >200k tokens triggers the long-context tier on 2.5-pro
    cost_short = gu.compute_cost_usd("gemini-2.5-pro", 100_000, 0)
    cost_long = gu.compute_cost_usd("gemini-2.5-pro", 250_000, 0)
    # Long tier is exactly 2x short tier on the prompt rate.
    assert cost_long > cost_short
    short_rate = cost_short / 100_000
    long_rate = cost_long / 250_000
    assert long_rate == pytest.approx(2 * short_rate, rel=1e-6)


def test_compute_cost_cached_tokens_discounted() -> None:
    # Cached portion of the prompt is billed at the cached (cheaper) rate.
    full = gu.compute_cost_usd("gemini-2.5-flash", 1_000_000, 0, cached_tokens=0)
    half_cached = gu.compute_cost_usd("gemini-2.5-flash", 1_000_000, 0, cached_tokens=500_000)
    assert half_cached < full


def test_compute_cost_unknown_model_uses_fallback() -> None:
    cost = gu.compute_cost_usd("gemini-9.9-fictional", 1_000_000, 0)
    # Falls back to flash-equivalent rate, not zero.
    assert cost > 0


def test_normalize_model_strips_version_suffix() -> None:
    assert gu._normalize_model("gemini-2.5-flash-001") == "gemini-2.5-flash"
    assert gu._normalize_model("models/gemini-2.5-pro") == "gemini-2.5-pro"
    assert gu._normalize_model("gemini-2.5-flash-lite-preview") == "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Recording + summary
# ---------------------------------------------------------------------------


def test_record_call_writes_event(isolated_storage: Path) -> None:
    event = gu.record_call("gemini-2.5-flash", 1000, 200, source="test")
    assert event is not None
    assert event.cost_usd > 0

    log = (isolated_storage / "events.jsonl").read_text().strip().splitlines()
    assert len(log) == 1
    payload = json.loads(log[0])
    assert payload["model"] == "gemini-2.5-flash"
    assert payload["prompt_tokens"] == 1000
    assert payload["source"] == "test"


def test_summary_aggregates_across_calls(isolated_storage: Path) -> None:
    gu.record_call("gemini-2.5-flash", 1000, 200, source="a")
    gu.record_call("gemini-2.5-pro", 500, 100, source="b")
    gu.record_call("gemini-2.5-flash", 2000, 400, source="a")

    summary = gu.load_summary()
    assert summary.total_calls == 3
    assert summary.total_prompt_tokens == 3500
    assert summary.total_completion_tokens == 700
    assert summary.total_cost_usd > 0
    assert set(summary.by_model.keys()) == {"gemini-2.5-flash", "gemini-2.5-pro"}
    assert summary.by_model["gemini-2.5-flash"]["calls"] == 2


def test_monthly_bucketing(isolated_storage: Path) -> None:
    jan = datetime(2026, 1, 15, tzinfo=timezone.utc)
    feb = datetime(2026, 2, 5, tzinfo=timezone.utc)
    gu.record_call("gemini-2.5-flash", 1000, 100, timestamp=jan)
    gu.record_call("gemini-2.5-flash", 2000, 200, timestamp=jan)
    gu.record_call("gemini-2.5-flash", 3000, 300, timestamp=feb)

    summary = gu.load_summary()
    assert "2026-01" in summary.by_month
    assert "2026-02" in summary.by_month
    assert summary.by_month["2026-01"].calls == 2
    assert summary.by_month["2026-02"].calls == 1


def test_record_from_response_extracts_usage(isolated_storage: Path) -> None:
    fake_usage = SimpleNamespace(
        prompt_token_count=4242,
        candidates_token_count=128,
        cached_content_token_count=64,
    )
    fake_response = SimpleNamespace(usage_metadata=fake_usage, text="hi")
    event = gu.record_from_response("gemini-2.5-flash", fake_response, source="test")
    assert event is not None
    assert event.prompt_tokens == 4242
    assert event.completion_tokens == 128
    assert event.cached_tokens == 64


def test_record_from_response_tolerates_missing_usage(isolated_storage: Path) -> None:
    fake_response = SimpleNamespace(usage_metadata=None, text="")
    event = gu.record_from_response("gemini-2.5-flash", fake_response)
    assert event is not None
    assert event.prompt_tokens == 0
    assert event.completion_tokens == 0


def test_record_call_never_raises_on_corrupt_input(isolated_storage: Path) -> None:
    # Pass clearly bogus arguments: tracker must swallow and return None.
    result = gu.record_call("gemini-2.5-flash", "not-an-int", 100)  # type: ignore[arg-type]
    # int("not-an-int") raises inside record_call; should be caught.
    assert result is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _writer(storage_dir: str, n: int) -> None:
    os.environ["COMPGEN_GEMINI_USAGE_DIR"] = storage_dir
    os.environ["COMPGEN_REPO_ROOT"] = storage_dir
    # Re-import so child uses fresh env-resolved paths.
    from compgen.observability import gemini_usage as child_gu
    for _ in range(n):
        child_gu.record_call("gemini-2.5-flash", 100, 10, source="concurrent")


def test_concurrent_writes_are_serialized(isolated_storage: Path) -> None:
    procs = [mp.Process(target=_writer, args=(str(isolated_storage), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    summary = gu.load_summary()
    assert summary.total_calls == 100
    # Every line should be valid JSON.
    lines = (isolated_storage / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 100
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_warning_at_80_pct(isolated_storage: Path) -> None:
    b = gu.Budget(monthly_usd=1.0)
    b.save()
    # Spend ~85¢ at flash rates: 2.83M input + 0 output ≈ $0.85
    gu.record_call("gemini-2.5-flash", 2_834_000, 0)
    status = gu.evaluate_budget()
    assert status.monthly_usd_pct is not None
    assert 80 <= status.monthly_usd_pct < 100
    assert status.warnings
    assert not status.exceeded


def test_budget_exceeded_at_100_pct(isolated_storage: Path) -> None:
    b = gu.Budget(monthly_usd=0.10)
    b.save()
    gu.record_call("gemini-2.5-flash", 1_000_000, 0)  # $0.30
    status = gu.evaluate_budget()
    assert status.exceeded


def test_budget_unset_returns_none_pct(isolated_storage: Path) -> None:
    gu.record_call("gemini-2.5-flash", 1000, 100)
    status = gu.evaluate_budget()
    assert status.monthly_usd_pct is None
    assert not status.warnings
    assert not status.exceeded
