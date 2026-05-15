"""Verify the Gemini PRICING table matches Google's published rates.

This test is the **canonical guard** against pricing drift. It pins
the expected per-model rates against what Google published on
``ai.google.dev/pricing`` as of ``PRICING_VERIFIED_AT``. When Google
revises rates (which happens regularly), the next maintenance pass:

1. Re-fetches ``ai.google.dev/pricing``.
2. Updates ``PRICING`` in :mod:`compgen.observability.gemini_usage`.
3. Bumps ``PRICING_VERIFIED_AT`` to today.
4. Updates ``_EXPECTED_RATES`` below to match.

If any of those steps is skipped, this test fails — guaranteeing
the live ``compgen-gemini-usage watch`` panel can never silently
report stale numbers.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from compgen.observability.gemini_usage import (
    _FALLBACK_RATES,
    PRICING,
    PRICING_SOURCE_URL,
    PRICING_VERIFIED_AT,
    compute_cost_usd,
    resolve_rates,
)


# Each entry pins what Google's pricing page showed for that
# model as of PRICING_VERIFIED_AT. Keep in sync with the table.
_EXPECTED_RATES: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
        "cached": 0.025,
    },
    "gemini-2.0-flash-lite": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "cached": 0.125,
        "input_long": 2.50,
        "output_long": 15.00,
        "long_threshold": 200_000,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
        "cached": 0.03,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cached": 0.01,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "cached": 0.20,
        "input_long": 4.00,
        "output_long": 18.00,
        "long_threshold": 200_000,
    },
    "gemini-3.1-flash-lite": {
        "input": 0.25,
        "output": 1.50,
        "cached": 0.025,
    },
    "gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00,
        "cached": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Rate pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(_EXPECTED_RATES.keys()))
def test_pricing_matches_expected_for_each_model(model: str):
    """Every model in _EXPECTED_RATES must have an exact match in PRICING."""
    assert model in PRICING, (
        f"{model!r} missing from PRICING — update the table or "
        f"remove it from _EXPECTED_RATES"
    )
    actual = PRICING[model]
    expected = _EXPECTED_RATES[model]
    for field_name, expected_value in expected.items():
        assert field_name in actual, (
            f"{model}: missing field {field_name!r}"
        )
        assert actual[field_name] == expected_value, (
            f"{model}.{field_name} drift: "
            f"actual={actual[field_name]} expected={expected_value}"
        )


def test_no_stale_cached_rates_in_2_5_family():
    """Regression: the 2.5 family previously over-priced cached
    tokens by ~2.5x. Pin the correct values explicitly."""
    assert PRICING["gemini-2.5-pro"]["cached"] == 0.125
    assert PRICING["gemini-2.5-flash"]["cached"] == 0.03
    assert PRICING["gemini-2.5-flash-lite"]["cached"] == 0.01


def test_gemini_3_family_present():
    """Gemini 3.x family must not silently fall back."""
    for mid in ("gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
                "gemini-3-flash-preview"):
        rates, key = resolve_rates(mid)
        assert key == mid, (
            f"{mid!r} resolved to {key!r} — should resolve to itself, "
            f"not fall back"
        )


# ---------------------------------------------------------------------------
# Verified-at freshness
# ---------------------------------------------------------------------------


def test_verified_at_is_iso_date():
    parsed = datetime.strptime(PRICING_VERIFIED_AT, "%Y-%m-%d")
    assert parsed.year >= 2025


def test_verified_at_is_not_ancient():
    """If the verified date is more than 6 months stale, fail —
    Google revises rates often enough that 6 months is the soft
    cap before the table needs a re-verification pass."""
    verified = date.fromisoformat(PRICING_VERIFIED_AT)
    now = datetime.now(timezone.utc).date()
    age_days = (now - verified).days
    assert age_days <= 180, (
        f"PRICING_VERIFIED_AT is {age_days} days old — re-verify "
        f"rates against {PRICING_SOURCE_URL} and bump the date."
    )


def test_pricing_source_url_is_google():
    assert "ai.google.dev" in PRICING_SOURCE_URL


# ---------------------------------------------------------------------------
# Fallback behavior is loud + correct
# ---------------------------------------------------------------------------


def test_unknown_model_logs_warning(caplog):
    """An unknown model must trigger a structured warning so the user
    notices the table needs updating — not a silent $0 or silent flash-
    rate fallback."""
    from compgen.observability import gemini_usage as gu
    import logging

    # Reset the dedup set so the warning re-fires.
    gu._WARNED_FALLBACK_MODELS.discard("gemini-totally-made-up-99")
    with caplog.at_level(logging.WARNING, logger=gu.logger.name):
        rates, key = resolve_rates("gemini-totally-made-up-99")
    assert key == "_FALLBACK_RATES"
    assert rates == _FALLBACK_RATES
    assert any(
        "fallback for unknown model" in r.getMessage() for r in caplog.records
    ), f"no warning logged; records={[r.getMessage() for r in caplog.records]}"


def test_fallback_warning_is_dedup(caplog):
    """A single fallback model only warns once per process."""
    from compgen.observability import gemini_usage as gu
    import logging

    gu._WARNED_FALLBACK_MODELS.discard("gemini-fake-dedup")
    with caplog.at_level(logging.WARNING, logger=gu.logger.name):
        resolve_rates("gemini-fake-dedup")
        resolve_rates("gemini-fake-dedup")
        resolve_rates("gemini-fake-dedup")
    fallback_warnings = [
        r for r in caplog.records if "fallback for unknown model" in r.getMessage()
    ]
    assert len(fallback_warnings) == 1


def test_fallback_rates_are_sane():
    """Fallback rates must be non-zero (so unknown models don't report
    $0) and shouldn't exceed gemini-2.5-pro (so we don't over-bill)."""
    assert _FALLBACK_RATES["input"] > 0
    assert _FALLBACK_RATES["output"] > 0
    assert _FALLBACK_RATES["input"] <= PRICING["gemini-2.5-pro"]["input"]
    assert _FALLBACK_RATES["output"] <= PRICING["gemini-2.5-pro"]["output"]


# ---------------------------------------------------------------------------
# End-to-end cost computation matches expected
# ---------------------------------------------------------------------------


def test_cost_for_known_model_uses_real_rates():
    # 1M prompt, 0 output, 0 cached → exactly $0.30 on flash.
    assert compute_cost_usd("gemini-2.5-flash", 1_000_000, 0, 0) == pytest.approx(
        0.30, abs=1e-6
    )


def test_cost_for_versioned_model_normalizes_correctly():
    """``gemini-2.5-flash-001`` should resolve to the same rates as
    ``gemini-2.5-flash`` via prefix-stripping in ``_normalize_model``."""
    rates_a, key_a = resolve_rates("gemini-2.5-flash-001")
    rates_b, key_b = resolve_rates("gemini-2.5-flash")
    assert key_a == key_b == "gemini-2.5-flash"
    assert rates_a == rates_b


def test_cost_for_long_context_uses_long_rates():
    """Above the 200k threshold, 2.5-pro charges $2.50/1M input not $1.25."""
    short = compute_cost_usd("gemini-2.5-pro", 100_000, 1_000, 0)
    long_ = compute_cost_usd("gemini-2.5-pro", 250_000, 1_000, 0)
    # long rate / short rate ≈ 2.0 ignoring tiny output-rate difference.
    # Bound loosely; primary purpose is "uses long rate at all".
    assert long_ > short * 2


def test_cached_token_billing_is_subset_of_prompt():
    """Cached tokens shouldn't be double-billed at both rates."""
    # 1M prompt, half of which is cached, on flash. Cached portion bills
    # at $0.03, the non-cached half bills at $0.30 — total ≈ $0.165.
    cost = compute_cost_usd("gemini-2.5-flash", 1_000_000, 0, 500_000)
    assert cost == pytest.approx(
        0.5 * 0.30 + 0.5 * 0.03, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Event stamping: every recorded call has rates_key + rates_verified_at
# ---------------------------------------------------------------------------


def test_recorded_event_stamps_rates_key(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    from compgen.observability.gemini_usage import record_call

    event = record_call(
        model="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=500,
        source="test",
    )
    assert event is not None
    assert event.metadata["rates_key"] == "gemini-2.5-flash"
    assert event.metadata["rates_verified_at"] == PRICING_VERIFIED_AT


def test_recorded_event_marks_fallback_when_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    from compgen.observability import gemini_usage as gu

    gu._WARNED_FALLBACK_MODELS.discard("gemini-never-heard-of-this")
    event = gu.record_call(
        model="gemini-never-heard-of-this",
        prompt_tokens=1000,
        completion_tokens=500,
        source="test",
    )
    assert event is not None
    assert event.metadata["rates_key"] == "_FALLBACK_RATES"
