"""``compgen-gemini-usage`` command — inspect and monitor Gemini API spend.

Subcommands:
    status   Print a one-shot snapshot (default).
    watch    Live-updating panel that tails events.jsonl.
    budget   Inspect / set spending limits.
    record   Manually log a call (useful when external tools call Gemini).
    paths    Print the on-disk locations the tracker uses.
    json     Emit the raw summary as JSON for scripting.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from compgen.observability.gemini_usage import (
    PRICING,
    PRICING_SOURCE_URL,
    PRICING_VERIFIED_AT,
    Budget,
    UsageSummary,
    budget_path,
    evaluate_budget,
    events_path,
    get_storage_dir,
    iter_events,
    load_pricing_overrides,
    load_summary,
    record_call,
    resolve_rates,
    summary_path,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _fmt_usd(value: float) -> str:
    if value >= 100:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _adaptive_bar_width(console_width: int) -> int:
    """Pick a budget-bar width that scales with the current terminal.

    Bounds: at least 12 cells (so the bar isn't degenerate on a
    narrow terminal), at most 48 (so it doesn't dominate the
    panel on a wide one). Leaves ~50 cells for the label + %
    suffix.
    """

    return max(12, min(48, console_width - 50))


def _budget_bar(
    used: float,
    limit: float | None,
    *,
    console_width: int | None = None,
    width: int | None = None,
) -> Text:
    if not limit or limit <= 0:
        return Text("(no limit)", style="dim")
    if width is None:
        width = _adaptive_bar_width(console_width or 80)
    pct = min(used / limit, 1.0) * 100
    filled = int(round(pct / 100 * width))
    if pct >= 100:
        style = "bold red"
    elif pct >= 80:
        style = "bold yellow"
    else:
        style = "green"
    bar = "█" * filled + "░" * (width - filled)
    return Text(f"{bar} {pct:5.1f}%", style=style)


def _render_status(
    summary: UsageSummary,
    budget: Budget,
    *,
    console_width: int = 80,
) -> Group:
    now = datetime.now(timezone.utc)
    month = summary.current_month(now)
    status = evaluate_budget(summary, budget)

    totals = Table.grid(padding=(0, 2))
    totals.add_column(style="dim")
    totals.add_column(justify="right")
    totals.add_row("Total calls", _fmt_int(summary.total_calls))
    totals.add_row("Total tokens (in/out/cached)",
                   f"{_fmt_int(summary.total_prompt_tokens)} / "
                   f"{_fmt_int(summary.total_completion_tokens)} / "
                   f"{_fmt_int(summary.total_cached_tokens)}")
    totals.add_row("Total cost", Text(_fmt_usd(summary.total_cost_usd), style="bold cyan"))
    totals.add_row("First event", summary.first_event_at or "—")
    totals.add_row("Last event", summary.last_event_at or "—")
    cumulative_panel = Panel(totals, title="[bold]Cumulative[/bold]", border_style="cyan")

    monthly = Table.grid(padding=(0, 2))
    monthly.add_column(style="dim")
    monthly.add_column(justify="right")
    monthly.add_row("Month", month.month)
    monthly.add_row("Calls", _fmt_int(month.calls))
    monthly.add_row("Tokens (in/out)",
                    f"{_fmt_int(month.prompt_tokens)} / {_fmt_int(month.completion_tokens)}")
    monthly.add_row("Cost", Text(_fmt_usd(month.cost_usd), style="bold cyan"))
    monthly.add_row(
        "Monthly USD budget",
        _budget_bar(month.cost_usd, budget.monthly_usd, console_width=console_width),
    )
    monthly.add_row(
        "Monthly token budget",
        _budget_bar(month.total_tokens, budget.monthly_tokens, console_width=console_width),
    )
    monthly_panel = Panel(monthly, title=f"[bold]Current month — {month.month}[/bold]",
                          border_style="magenta")

    by_model = Table(title="By model", show_edge=False, border_style="dim", expand=True)
    # Model column may carry long ids ("gemini-2.5-flash-lite-001"); ellipsize on
    # narrow terminals rather than wrap onto two lines.
    by_model.add_column("Model", overflow="ellipsis", no_wrap=True, ratio=4)
    by_model.add_column("Calls", justify="right", ratio=1)
    by_model.add_column("Prompt tok", justify="right", ratio=1)
    by_model.add_column("Output tok", justify="right", ratio=1)
    by_model.add_column("Cost", justify="right", ratio=1)
    rows = sorted(summary.by_model.items(), key=lambda kv: -kv[1]["cost_usd"])
    for model, stats in rows[:10]:
        by_model.add_row(
            model,
            _fmt_int(int(stats["calls"])),
            _fmt_int(int(stats["prompt_tokens"])),
            _fmt_int(int(stats["completion_tokens"])),
            _fmt_usd(stats["cost_usd"]),
        )
    if not rows:
        by_model.add_row("—", "0", "0", "0", "$0.00")

    months_table = Table(title="By month", show_edge=False, border_style="dim", expand=True)
    months_table.add_column("Month", ratio=2)
    months_table.add_column("Calls", justify="right", ratio=1)
    months_table.add_column("Tokens", justify="right", ratio=1)
    months_table.add_column("Cost", justify="right", ratio=1)
    for key in sorted(summary.by_month.keys(), reverse=True)[:6]:
        b = summary.by_month[key]
        months_table.add_row(key, _fmt_int(b.calls), _fmt_int(b.total_tokens), _fmt_usd(b.cost_usd))
    if not summary.by_month:
        months_table.add_row("—", "0", "0", "$0.00")

    alerts: list[Text] = []
    for warn in status.warnings:
        alerts.append(Text(f"⚠  {warn}", style="bold yellow"))
    for exc in status.exceeded:
        alerts.append(Text(f"✗ {exc}", style="bold red"))
    alerts_panel = (
        Panel(Group(*alerts), title="[bold]Budget alerts[/bold]", border_style="yellow")
        if alerts
        else Panel(Text("budget OK", style="green"), title="[bold]Budget alerts[/bold]",
                   border_style="green")
    )

    # Pricing panel — verified date + fallback-event count so the user
    # can see at a glance whether the rates table is fresh and whether
    # any historical events were priced via the fallback path.
    fallback_calls = 0
    fallback_models: set[str] = set()
    for m, _stats in summary.by_model.items():
        _rates, key = resolve_rates(m)
        if key == "_FALLBACK_RATES":
            fallback_calls += int(summary.by_model[m]["calls"])
            fallback_models.add(m)
    pricing_grid = Table.grid(padding=(0, 2))
    pricing_grid.add_column(style="dim")
    pricing_grid.add_column(justify="right")
    pricing_grid.add_row("Rates verified", PRICING_VERIFIED_AT)
    pricing_grid.add_row("Source", PRICING_SOURCE_URL)
    pricing_grid.add_row("Models in table", _fmt_int(len(PRICING)))
    if fallback_calls:
        pricing_grid.add_row(
            Text("⚠ fallback-priced calls", style="bold yellow"),
            Text(f"{fallback_calls} ({len(fallback_models)} model(s))",
                 style="bold yellow"),
        )
    else:
        pricing_grid.add_row("Fallback-priced calls", Text("0", style="green"))
    pricing_panel = Panel(
        pricing_grid,
        title="[bold]Pricing table[/bold]",
        border_style=("yellow" if fallback_calls else "green"),
    )

    return Group(
        cumulative_panel,
        monthly_panel,
        by_model,
        months_table,
        alerts_panel,
        pricing_panel,
    )


def _watch_footer(refresh_s: float, event_count: int) -> Text:
    now = datetime.now().strftime("%H:%M:%S")
    return Text(
        f"  refresh={refresh_s:.1f}s   events={event_count}   updated={now}   (Ctrl-C to exit)",
        style="dim",
    )


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("--json-out", "json_out", is_flag=True, help="Emit raw summary JSON.")
@click.pass_context
def main(ctx: click.Context, json_out: bool) -> None:
    """Inspect and monitor Gemini API spend for this CompGen project."""
    if ctx.invoked_subcommand is not None:
        return
    if json_out:
        ctx.invoke(json_cmd)
    else:
        ctx.invoke(status)


@main.command()
def status() -> None:
    """Print a one-shot usage snapshot."""
    console = Console()
    summary = load_summary()
    budget = Budget.load()
    console.print(
        _render_status(summary, budget, console_width=console.size.width)
    )


@main.command()
@click.option("--refresh", default=1.0, show_default=True, help="Refresh interval (seconds).")
@click.option(
    "--alt-screen/--inline",
    default=True,
    show_default=True,
    help=(
        "Render in the terminal's alternate-screen buffer "
        "(reflows cleanly on resize). Use --inline to keep "
        "scrollback-friendly inline rendering."
    ),
)
def watch(refresh: float, alt_screen: bool) -> None:
    """Live-updating dashboard that tails events.jsonl.

    Defaults to alt-screen mode so resizing the terminal reflows the
    layout cleanly (Rich's Live handles SIGWINCH for us in this
    mode). Pass ``--inline`` for the legacy in-place behavior that
    preserves scrollback.
    """
    console = Console()
    refresh = max(0.2, refresh)
    with Live(
        console=console,
        refresh_per_second=max(1.0, 1.0 / refresh),
        screen=alt_screen,
        transient=False,
        auto_refresh=False,
    ) as live:
        try:
            while True:
                summary = load_summary()
                budget = Budget.load()
                # Re-read width on every tick so SIGWINCH-driven resize
                # immediately recomputes the budget-bar width and
                # expanded-table layout.
                view = Group(
                    _render_status(
                        summary, budget, console_width=console.size.width
                    ),
                    _watch_footer(refresh, summary.total_calls),
                )
                live.update(view, refresh=True)
                time.sleep(refresh)
        except KeyboardInterrupt:
            console.print("[dim]watch stopped[/dim]")


@main.group("budget")
def budget_grp() -> None:
    """View or modify the spending budget."""


@budget_grp.command("show")
def budget_show() -> None:
    """Print the configured budget."""
    b = Budget.load()
    console = Console()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("Monthly USD", f"${b.monthly_usd:.2f}" if b.monthly_usd else "(unset)")
    table.add_row("Monthly tokens", _fmt_int(b.monthly_tokens) if b.monthly_tokens else "(unset)")
    table.add_row("Cumulative USD", f"${b.cumulative_usd:.2f}" if b.cumulative_usd else "(unset)")
    table.add_row("Cumulative tokens",
                  _fmt_int(b.cumulative_tokens) if b.cumulative_tokens else "(unset)")
    table.add_row("File", str(budget_path()))
    console.print(Panel(table, title="[bold]Budget[/bold]", border_style="cyan"))


@budget_grp.command("set")
@click.option("--monthly-usd", type=float, default=None, help="USD limit per calendar month.")
@click.option("--monthly-tokens", type=int, default=None, help="Token limit per calendar month.")
@click.option("--cumulative-usd", type=float, default=None, help="USD limit total.")
@click.option("--cumulative-tokens", type=int, default=None, help="Token limit total.")
@click.option("--clear", is_flag=True, help="Clear all limits.")
def budget_set(
    monthly_usd: float | None,
    monthly_tokens: int | None,
    cumulative_usd: float | None,
    cumulative_tokens: int | None,
    clear: bool,
) -> None:
    """Set one or more spending limits. Unspecified fields are kept."""
    if clear:
        Budget().save()
        click.echo(f"cleared {budget_path()}")
        return
    current = Budget.load()
    if monthly_usd is not None:
        current.monthly_usd = monthly_usd
    if monthly_tokens is not None:
        current.monthly_tokens = monthly_tokens
    if cumulative_usd is not None:
        current.cumulative_usd = cumulative_usd
    if cumulative_tokens is not None:
        current.cumulative_tokens = cumulative_tokens
    current.save()
    click.echo(f"saved budget to {budget_path()}")


@main.command("record")
@click.option("--model", required=True)
@click.option("--prompt-tokens", type=int, required=True)
@click.option("--completion-tokens", type=int, required=True)
@click.option("--cached-tokens", type=int, default=0)
@click.option("--source", default="manual")
def record_cmd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    source: str,
) -> None:
    """Manually log a call (e.g. for tools outside the Python tracker)."""
    event = record_call(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        source=source,
    )
    if event is None:
        raise click.ClickException("record_call failed; see logs")
    click.echo(json.dumps(
        {"recorded": True, "cost_usd": event.cost_usd, "timestamp": event.timestamp},
        indent=2,
    ))


@main.command("paths")
def paths_cmd() -> None:
    """Print the on-disk paths used by the tracker."""
    console = Console()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("storage dir", str(get_storage_dir()))
    table.add_row("events.jsonl", str(events_path()))
    table.add_row("summary.json", str(summary_path()))
    table.add_row("budget.json", str(budget_path()))
    table.add_row("override env",
                  os.environ.get("COMPGEN_GEMINI_USAGE_DIR", "(COMPGEN_GEMINI_USAGE_DIR unset)"))
    console.print(Panel(table, title="[bold]Tracker paths[/bold]", border_style="cyan"))


@main.command("pricing")
@click.option("--model", default=None, help="Print rates for one model id.")
@click.option(
    "--json-out", "json_out", is_flag=True, help="Emit the pricing table as JSON."
)
def pricing_cmd(model: str | None, json_out: bool) -> None:
    """Print the active pricing table + verified date + overrides.

    Useful for sanity-checking what rates the tracker is using
    right now — particularly when a new Gemini model lands and you
    want to know whether the tracker recognized it or is falling
    back to the mid-tier rate.
    """
    merged = dict(PRICING)
    overrides = load_pricing_overrides()
    merged.update(overrides)

    if model:
        rates, key = resolve_rates(model)
        if json_out:
            click.echo(json.dumps(
                {"model": model, "resolved_key": key, "rates": rates},
                indent=2, sort_keys=True,
            ))
            return
        console = Console()
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(justify="right")
        for field_name in ("input", "output", "cached", "input_long",
                           "output_long", "long_threshold"):
            if field_name in rates:
                value = rates[field_name]
                if field_name == "long_threshold":
                    table.add_row(field_name, _fmt_int(int(value)) + " tokens")
                else:
                    table.add_row(field_name, f"${value} / 1M")
        style = "yellow" if key == "_FALLBACK_RATES" else "green"
        title_note = " (FALLBACK)" if key == "_FALLBACK_RATES" else ""
        console.print(Panel(
            table,
            title=f"[bold]{model} → {key}{title_note}[/bold]",
            border_style=style,
        ))
        return

    if json_out:
        click.echo(json.dumps(
            {
                "verified_at": PRICING_VERIFIED_AT,
                "source": PRICING_SOURCE_URL,
                "overrides_applied": list(overrides.keys()),
                "models": merged,
            },
            indent=2, sort_keys=True,
        ))
        return

    console = Console()
    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("Verified at", PRICING_VERIFIED_AT)
    header.add_row("Source", PRICING_SOURCE_URL)
    if overrides:
        header.add_row(
            "Override file applied",
            Text("yes", style="bold yellow") + Text(
                f" ({len(overrides)} model(s))", style="dim"
            ),
        )
    console.print(Panel(header, title="[bold]Active pricing[/bold]",
                        border_style="cyan"))

    tbl = Table(border_style="dim", show_edge=False)
    tbl.add_column("Model")
    tbl.add_column("input $/1M", justify="right")
    tbl.add_column("output $/1M", justify="right")
    tbl.add_column("cached $/1M", justify="right")
    tbl.add_column("long_threshold", justify="right")
    for name, rates in sorted(merged.items()):
        tbl.add_row(
            name,
            f"${rates.get('input', '—')}",
            f"${rates.get('output', '—')}",
            f"${rates.get('cached', '—')}",
            (
                _fmt_int(int(rates["long_threshold"]))
                if "long_threshold" in rates
                else "—"
            ),
        )
    console.print(tbl)


@main.command("json")
@click.option("--events", is_flag=True, help="Emit the raw event log instead of the summary.")
def json_cmd(events: bool) -> None:
    """Emit raw summary or events as JSON for scripting."""
    if events:
        for ev in iter_events():
            click.echo(ev.to_json_line())
        return
    summary = load_summary()
    click.echo(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
