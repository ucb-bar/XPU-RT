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
    Budget,
    UsageSummary,
    budget_path,
    evaluate_budget,
    events_path,
    get_storage_dir,
    iter_events,
    load_summary,
    record_call,
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


def _budget_bar(used: float, limit: float | None, width: int = 24) -> Text:
    if not limit or limit <= 0:
        return Text("(no limit)", style="dim")
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


def _render_status(summary: UsageSummary, budget: Budget) -> Group:
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
    monthly.add_row("Monthly USD budget", _budget_bar(month.cost_usd, budget.monthly_usd))
    monthly.add_row("Monthly token budget",
                    _budget_bar(month.total_tokens, budget.monthly_tokens))
    monthly_panel = Panel(monthly, title=f"[bold]Current month — {month.month}[/bold]",
                          border_style="magenta")

    by_model = Table(title="By model", show_edge=False, border_style="dim")
    by_model.add_column("Model")
    by_model.add_column("Calls", justify="right")
    by_model.add_column("Prompt tok", justify="right")
    by_model.add_column("Output tok", justify="right")
    by_model.add_column("Cost", justify="right")
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

    months_table = Table(title="By month", show_edge=False, border_style="dim")
    months_table.add_column("Month")
    months_table.add_column("Calls", justify="right")
    months_table.add_column("Tokens", justify="right")
    months_table.add_column("Cost", justify="right")
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

    return Group(cumulative_panel, monthly_panel, by_model, months_table, alerts_panel)


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
    console.print(_render_status(summary, budget))


@main.command()
@click.option("--refresh", default=1.0, show_default=True, help="Refresh interval (seconds).")
def watch(refresh: float) -> None:
    """Live-updating dashboard that tails events.jsonl."""
    console = Console()
    refresh = max(0.2, refresh)
    with Live(console=console, refresh_per_second=max(1.0, 1.0 / refresh), screen=False) as live:
        try:
            while True:
                summary = load_summary()
                budget = Budget.load()
                view = Group(_render_status(summary, budget),
                             _watch_footer(refresh, summary.total_calls))
                live.update(view)
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
