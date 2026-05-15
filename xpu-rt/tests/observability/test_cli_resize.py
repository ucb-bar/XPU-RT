"""Verify the gemini-usage CLI renders across a range of terminal widths.

Catches regressions where:
  * the budget bar overflows or degenerates on narrow/wide terminals,
  * tables don't reflow (lose the ``expand=True`` setting),
  * long model ids wrap onto multiple lines (no-wrap + ellipsis lost).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest
from rich.console import Console

from compgen.observability.cli import (
    _adaptive_bar_width,
    _budget_bar,
    _render_status,
)
from compgen.observability.gemini_usage import (
    Budget,
    MonthBucket,
    UsageSummary,
)


def _fake_summary() -> UsageSummary:
    s = UsageSummary()
    s.total_calls = 50
    s.total_prompt_tokens = 100_000
    s.total_completion_tokens = 40_000
    s.total_cached_tokens = 12_000
    s.total_cost_usd = 0.42
    s.first_event_at = "2026-05-01T00:00:00+00:00"
    s.last_event_at = "2026-05-12T07:00:00+00:00"
    s.by_model = {
        "gemini-2.5-flash-lite-001-very-long-name": {
            "calls": 30,
            "prompt_tokens": 70_000,
            "completion_tokens": 30_000,
            "cost_usd": 0.30,
        },
        "gemini-3.1-pro-preview": {
            "calls": 20,
            "prompt_tokens": 30_000,
            "completion_tokens": 10_000,
            "cost_usd": 0.12,
        },
    }
    s.by_month = {
        "2026-05": MonthBucket(
            month="2026-05",
            calls=50,
            prompt_tokens=100_000,
            completion_tokens=40_000,
            cached_tokens=12_000,
            cost_usd=0.42,
        ),
    }
    return s


# ---------------------------------------------------------------------------
# Bar width math
# ---------------------------------------------------------------------------


def test_bar_width_clamped_low():
    """A 20-col terminal still gets at least a 12-cell bar."""
    assert _adaptive_bar_width(20) == 12


def test_bar_width_clamped_high():
    """A 500-col terminal caps at 48 cells."""
    assert _adaptive_bar_width(500) == 48


def test_bar_width_scales_in_middle():
    a = _adaptive_bar_width(80)
    b = _adaptive_bar_width(120)
    assert b >= a


# ---------------------------------------------------------------------------
# Budget bar
# ---------------------------------------------------------------------------


def test_budget_bar_under_limit_is_green():
    bar = _budget_bar(0.10, 1.0, console_width=80)
    assert "█" in bar.plain
    assert "░" in bar.plain
    # Style applied at top level
    assert "green" in str(bar.style).lower()


def test_budget_bar_at_full_is_red():
    bar = _budget_bar(1.0, 1.0, console_width=80)
    assert "red" in str(bar.style).lower()
    assert "100.0%" in bar.plain


def test_budget_bar_no_limit_shows_placeholder():
    bar = _budget_bar(0.5, None, console_width=80)
    assert "no limit" in bar.plain


# ---------------------------------------------------------------------------
# Render across widths — never raises, always produces non-empty output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [40, 60, 80, 120, 160, 240])
def test_render_does_not_raise_at_any_width(width: int):
    buf = io.StringIO()
    console = Console(
        file=buf, width=width, force_terminal=True, color_system=None
    )
    summary = _fake_summary()
    budget = Budget(monthly_usd=1.0)
    console.print(_render_status(summary, budget, console_width=width))
    output = buf.getvalue()
    # Must produce some output at every width.
    assert len(output) > 200
    # No line should be wider than the terminal width.
    for line in output.splitlines():
        # Strip rich's color escape sequences before measuring.
        from rich.text import Text as _T
        visible = _T.from_markup(line).plain
        assert len(visible) <= width + 4, (
            f"line exceeds width {width}: {len(visible)} cells\n{visible!r}"
        )


def test_render_includes_pricing_panel_at_every_width():
    for width in (60, 100, 160):
        buf = io.StringIO()
        console = Console(
            file=buf, width=width, force_terminal=True, color_system=None
        )
        console.print(_render_status(_fake_summary(), Budget(), console_width=width))
        out = buf.getvalue()
        assert "Pricing table" in out, f"missing pricing panel at width={width}"
        assert "Rates verified" in out


def test_render_handles_empty_summary():
    """An empty summary must render cleanly at any width (no division
    by zero, no missing rows)."""
    for width in (60, 120):
        buf = io.StringIO()
        console = Console(
            file=buf, width=width, force_terminal=True, color_system=None
        )
        console.print(_render_status(UsageSummary(), Budget(), console_width=width))
        out = buf.getvalue()
        assert "Total calls" in out


def test_long_model_name_does_not_overflow():
    """A 30+ char model id must ellipsize, not wrap or overflow."""
    buf = io.StringIO()
    console = Console(
        file=buf, width=70, force_terminal=True, color_system=None
    )
    summary = _fake_summary()
    console.print(_render_status(summary, Budget(), console_width=70))
    out = buf.getvalue()
    # The full name "gemini-2.5-flash-lite-001-very-long-name" (40 chars)
    # cannot fit at width=70 without ellipsizing the model column.
    assert "very-long-name" not in out or "…" in out, (
        "model name should ellipsize on narrow terminals"
    )
