"""ASCII Gantt renderer for the QNN dashboard.

Renders one row per machine (DSP/HTA, GPU, CPU) with dispatch bars
scaled to the terminal width. Used by ``QnnDashboard`` to show the
current schedule in-place; mirrors what
``targets/backends/qnn/plot.gantt`` produces as a PNG, except this
version is text-only and updateable every frame.

The renderer is deliberately tolerant: it accepts either a
``ScheduleResult`` (from ``targets/backends/qnn/scheduler``) or a
plain dict / loaded ``schedule.json``. The latter is what the
heterogeneous loop writes out, so the dashboard never has to
re-instantiate the dataclass to draw.
"""

from __future__ import annotations

import colorsys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DispatchSpan:
    """One bar to draw on the Gantt."""

    name: str
    machine: str
    start_us: float
    finish_us: float
    workload: str = "default"


def _palette(workloads: list[str]) -> dict[str, str]:
    """Stable Rich color tags, one per workload identity."""
    out: dict[str, str] = {}
    n = max(1, len(workloads))
    for i, wl in enumerate(workloads):
        # Walk HSV at a fixed saturation/value so adjacent hues are
        # easy to distinguish in a typical 256-colour terminal.
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        out[wl] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return out


def _normalise(schedule: Any) -> list[DispatchSpan]:
    """Coerce either ``ScheduleResult`` or a dict into spans."""
    if hasattr(schedule, "start_us") and hasattr(schedule, "finish_us"):
        # ScheduleResult dataclass.
        spans: list[DispatchSpan] = []
        machine = getattr(schedule, "machine", {})
        for cid, s in schedule.start_us.items():
            f = schedule.finish_us.get(cid, s)
            m = machine.get(cid, "CPU")
            spans.append(DispatchSpan(
                name=cid, machine=m, start_us=float(s), finish_us=float(f),
                workload=cid.split(".")[0] if "." in cid else "default",
            ))
        return spans

    if isinstance(schedule, Mapping):
        if "ops" in schedule and isinstance(schedule["ops"], list):
            spans = []
            for op in schedule["ops"]:
                spans.append(DispatchSpan(
                    name=str(op.get("name", "?")),
                    machine=str(op.get("machine") or op.get("hardware_target") or "CPU"),
                    start_us=float(op.get("start_us", 0.0)),
                    finish_us=float(op.get("finish_us", op.get("start_us", 0.0))),
                    workload=str(op.get("workload", op.get("name", "?").split(".")[0])),
                ))
            return spans
        if "dispatches" in schedule and isinstance(schedule["dispatches"], dict):
            spans = []
            for name, row in schedule["dispatches"].items():
                start = float(row.get("start_us", 0.0))
                finish = float(row.get("finish_us", row.get("end_us", start)))
                spans.append(DispatchSpan(
                    name=name,
                    machine=str(row.get("machine") or row.get("hardware_target") or "CPU"),
                    start_us=start,
                    finish_us=finish,
                    workload=str(row.get("workload", name.split(".")[0])),
                ))
            return spans
    return []


def render_ascii_gantt(
    schedule: Any,
    *,
    width: int = 80,
    machines: tuple[str, ...] = ("HTA", "GPU", "CPU"),
    palette: dict[str, str] | None = None,
    title: str | None = None,
):
    """Render a Rich Panel showing the schedule.

    ``schedule`` may be a ``ScheduleResult`` or a parsed schedule.json
    dict (the heterogeneous loop produces the latter, with ``ops`` or
    ``dispatches``).
    """
    # Lazy import — keeps Rich off the hot path for callers that just
    # want a string representation.
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    spans = _normalise(schedule)
    if not spans:
        return Panel(
            Text("(no schedule yet)", style="dim"),
            title="Schedule",
            box=box.MINIMAL,
        )

    makespan = max(s.finish_us for s in spans) or 1.0
    workloads = sorted({s.workload for s in spans})
    pal = palette or _palette(workloads)

    # Account for the device-name column + a space + edges.
    bar_width = max(20, width - 6)

    table = Table(
        box=box.MINIMAL,
        show_header=True,
        header_style="bold",
        title=title or f"Schedule  makespan={makespan:.1f}µs",
        title_style="bold",
        expand=False,
    )
    table.add_column("dev", style="bold", width=4)
    table.add_column(f"0 .. {makespan:.0f} µs")

    for m in machines:
        cells: list[str] = [" "] * bar_width
        colors: list[str | None] = [None] * bar_width
        for sp in spans:
            if sp.machine != m:
                continue
            a = max(0, int(sp.start_us / makespan * bar_width))
            b = max(a + 1, int(sp.finish_us / makespan * bar_width))
            b = min(b, bar_width)
            color = pal.get(sp.workload, "white")
            for i in range(a, b):
                cells[i] = "█"
                colors[i] = color
        rich_text = Text()
        for ch, col in zip(cells, colors):
            if col is None:
                rich_text.append(ch)
            else:
                rich_text.append(ch, style=col)
        table.add_row(m, rich_text)

    legend = Text()
    for i, wl in enumerate(workloads):
        if i:
            legend.append("  ")
        legend.append("█", style=pal.get(wl, "white"))
        legend.append(f" {wl}")

    return Panel.fit(
        Table.grid().__call__ if False else _stack(table, legend),
        title="Schedule",
        box=box.MINIMAL,
    )


def _stack(*renderables):
    """Tiny helper to stack two Rich renderables vertically."""
    from rich.console import Group

    return Group(*renderables)
