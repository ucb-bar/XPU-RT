"""Tests for :func:`xpu_rt.observability.gemini_usage.check_pre_call`.

Covers the pre-call hard gate that protects the configured cumulative-USD
cap before any Gemini request fires. Pairs with the retrospective
``evaluate_budget`` tests in ``test_gemini_usage.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from xpu_rt.observability import gemini_usage as gu


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("XPU_RT_REPO_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_soft_warn_memo() -> None:
    gu._SOFT_WARN_FIRED.clear()


# ---------------------------------------------------------------------------
# No-cap behaviour
# ---------------------------------------------------------------------------


def test_check_pre_call_no_budget_is_noop(isolated_storage: Path) -> None:
    gu.record_call("gemini-2.5-flash", 1_000_000, 1_000_000)
    gu.check_pre_call(projected_cost_usd=100.0, source="test")


def test_check_pre_call_zero_cap_is_noop(isolated_storage: Path) -> None:
    gu.Budget(cumulative_usd=0.0).save()
    gu.record_call("gemini-2.5-flash", 1_000_000, 1_000_000)
    gu.check_pre_call(projected_cost_usd=10.0, source="test")


# ---------------------------------------------------------------------------
# Hard cap
# ---------------------------------------------------------------------------


def test_check_pre_call_blocks_when_already_at_cap(isolated_storage: Path) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    # gemini-2.5-flash: $0.30/1M in, $2.50/1M out — burn through $100:
    # 40M output tokens = $100.00 flat.
    gu.record_call("gemini-2.5-flash", 0, 40_000_000)
    with pytest.raises(gu.GeminiBudgetExceeded) as excinfo:
        gu.check_pre_call(source="knowledge.ingest")
    err = excinfo.value
    assert err.limit_usd == pytest.approx(100.0)
    assert err.current_usd >= 100.0
    assert err.source == "knowledge.ingest"
    assert "knowledge.ingest" in str(err)


def test_check_pre_call_blocks_when_projected_would_exceed(
    isolated_storage: Path,
) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    # Burn $99.50 of the cap.
    gu.record_call("gemini-2.5-flash", 0, 39_800_000)
    # Projected $0.75 more would cross.
    with pytest.raises(gu.GeminiBudgetExceeded) as excinfo:
        gu.check_pre_call(projected_cost_usd=0.75, source="kernelblaster_v2")
    err = excinfo.value
    assert err.current_usd == pytest.approx(99.5, rel=1e-3)
    assert err.projected_usd is not None
    assert err.projected_usd > 100.0


def test_check_pre_call_allows_when_room_remains(isolated_storage: Path) -> None:
    gu.Budget(cumulative_usd=100.0).save()
    gu.record_call("gemini-2.5-flash", 0, 10_000_000)  # $25
    gu.check_pre_call(projected_cost_usd=0.10, source="ingest")  # well under cap


# ---------------------------------------------------------------------------
# Soft warn
# ---------------------------------------------------------------------------


def test_check_pre_call_soft_warn_logs_once_per_source(
    isolated_storage: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gu.Budget(cumulative_usd=100.0, soft_warn_usd=75.0).save()
    gu.record_call("gemini-2.5-flash", 0, 32_000_000)  # $80

    with caplog.at_level(logging.WARNING, logger=gu.logger.name):
        gu.check_pre_call(source="ingest")
        gu.check_pre_call(source="ingest")  # second call must not re-warn

    soft_warn_records = [
        r for r in caplog.records if "soft-warn" in r.getMessage()
    ]
    assert len(soft_warn_records) == 1
    assert "ingest" in soft_warn_records[0].getMessage()


def test_check_pre_call_soft_warn_does_not_block(isolated_storage: Path) -> None:
    gu.Budget(cumulative_usd=100.0, soft_warn_usd=75.0).save()
    gu.record_call("gemini-2.5-flash", 0, 35_000_000)  # $87.50
    gu.check_pre_call(projected_cost_usd=0.10, source="ingest")  # warns, returns


def test_check_pre_call_soft_warn_below_threshold_silent(
    isolated_storage: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gu.Budget(cumulative_usd=100.0, soft_warn_usd=75.0).save()
    gu.record_call("gemini-2.5-flash", 0, 10_000_000)  # $25
    with caplog.at_level(logging.WARNING, logger=gu.logger.name):
        gu.check_pre_call(source="ingest")
    assert not [r for r in caplog.records if "soft-warn" in r.getMessage()]


# ---------------------------------------------------------------------------
# Budget round-trip with new field
# ---------------------------------------------------------------------------


def test_budget_round_trip_with_soft_warn(isolated_storage: Path) -> None:
    gu.Budget(cumulative_usd=100.0, soft_warn_usd=75.0).save()
    reloaded = gu.Budget.load()
    assert reloaded.cumulative_usd == pytest.approx(100.0)
    assert reloaded.soft_warn_usd == pytest.approx(75.0)
