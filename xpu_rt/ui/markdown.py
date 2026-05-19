"""Markdown renderers for the agentic QNN flow.

Used by the MCP tools to embed display-ready blocks in their return
values so the agent (this Claude Code session) can paste them
verbatim into the chat. The standalone Rich dashboard in
``xpu_rt.ui.dashboard`` is for users who run ``xpu-rt qnn run`` in a
real terminal; this module is for the in-CLI agentic path where
Claude Code itself is the renderer.

Three blocks:

* :func:`render_gantt_markdown` — ASCII Gantt in a fenced code
  block, one row per machine, with a legend.
* :func:`render_deltas_markdown` — predicted-vs-measured comparison
  as a markdown table with status markers on outliers.
* :func:`render_decision_markdown` — the granularity decision the
  agent is about to commit, plus the legal alternatives.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# A small fixed palette used as visual prefixes per workload — Claude
# Code's chat surface is plain monospace inside code blocks, so we
# pair each workload with a unique block-character marker rather than
# relying on ANSI colour.
_WORKLOAD_MARKERS = ["▓", "▒", "░", "█", "▚", "▞", "▙", "▟", "◧", "◨", "◩", "◪", "◫"]


def _normalise_ops(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    ops = schedule.get("ops")
    if isinstance(ops, list) and ops:
        return [dict(o) for o in ops if isinstance(o, Mapping)]
    out: list[dict[str, Any]] = []
    dispatches = schedule.get("dispatches")
    if isinstance(dispatches, Mapping):
        for name, row in dispatches.items():
            d = dict(row) if isinstance(row, Mapping) else {}
            d.setdefault("name", name)
            out.append(d)
    out.sort(key=lambda o: float(o.get("start_us", 0.0)))
    return out


def _workload_of(op: Mapping[str, Any]) -> str:
    if op.get("workload"):
        return str(op["workload"])
    name = str(op.get("name", ""))
    return name.split(".")[0] if "." in name else name or "default"


def render_gantt_markdown(
    schedule: Mapping[str, Any] | str | Path,
    *,
    width: int = 64,
    machines: tuple[str, ...] = ("HTA", "GPU", "CPU"),
    title: str | None = None,
) -> str:
    """Render a Gantt as a fenced-code block markdown string.

    ``schedule`` may be the dict, a JSON string, or a path to a
    schedule.json. The rendered block is safe to paste straight into
    a Claude Code message.
    """
    if isinstance(schedule, (str, Path)):
        try:
            schedule = json.loads(Path(schedule).read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            return "_no schedule available_\n"
    ops = _normalise_ops(schedule)
    if not ops:
        return "_no schedule available_\n"

    makespan = float(schedule.get("makespan_us", 0.0)) or max(
        float(o.get("finish_us", 0.0)) for o in ops
    ) or 1.0
    workloads = sorted({_workload_of(o) for o in ops})
    marker_for = {w: _WORKLOAD_MARKERS[i % len(_WORKLOAD_MARKERS)]
                  for i, w in enumerate(workloads)}

    rows = {m: [" "] * width for m in machines}
    for op in ops:
        m = str(op.get("machine") or op.get("hardware_target") or "CPU")
        if m not in rows:
            continue
        a = max(0, int(float(op.get("start_us", 0.0)) / makespan * width))
        b = max(a + 1, int(float(op.get("finish_us", a + 1)) / makespan * width))
        b = min(b, width)
        marker = marker_for[_workload_of(op)]
        for i in range(a, b):
            rows[m][i] = marker

    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
    lines.append(f"_makespan: {makespan:,.0f} µs · {len(ops)} dispatches "
                 f"· {len(workloads)} workload(s)_")
    lines.append("")
    lines.append("```")
    # Axis labels.
    axis = (
        f"     0{' ' * (width - 2)}{int(makespan):>5} µs"
    )
    lines.append(axis)
    lines.append(f"     {'─' * width}")
    for m in machines:
        lines.append(f" {m:>3} │{''.join(rows[m])}│")
    lines.append(f"     {'─' * width}")
    lines.append("")
    legend = "  ".join(
        f"{marker_for[w]} {w}" for w in workloads
    )
    lines.append(f"legend: {legend}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def render_deltas_markdown(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 12,
    title: str | None = None,
) -> str:
    """Render the predicted-vs-measured rows as a markdown table.

    Rows where the ratio exceeds 1.3 are flagged with ``⚠️``; rows
    above 1.1 get a ``·`` marker so the agent can call out hotspots
    without re-reading the whole table.
    """
    rows = list(rows)
    if not rows:
        return "_no deltas yet — no profile measurements ingested_\n"
    rows = rows[:limit]

    out: list[str] = []
    if title:
        out.append(f"**{title}**")
    out.append("")
    out.append("| | dispatch | dev | predicted (µs) | measured (µs) | Δ (µs) | ratio |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        ratio = r.get("ratio")
        if ratio is not None and ratio > 1.30:
            mark = "⚠️"
        elif ratio is not None and ratio > 1.10:
            mark = "·"
        else:
            mark = "•"
        delta = r.get("delta_us")
        meas = r.get("measured_us")
        pred = float(r.get("predicted_us") or 0.0)
        name = str(r.get("dispatch", "?"))[:42]
        machine = str(r.get("machine", "-"))
        meas_cell = f"{meas:,.0f}" if meas is not None else "_missing_"
        delta_cell = f"{delta:+,.0f}" if delta is not None else "-"
        ratio_cell = f"{ratio:.2f}" if ratio is not None else "-"
        out.append(
            f"| {mark} | `{name}` | {machine} "
            f"| {pred:,.0f} | {meas_cell} | {delta_cell} | {ratio_cell} |"
        )
    return "\n".join(out) + "\n"


def render_decision_markdown(
    *,
    round_index: int,
    makespan_us: float,
    greedy_pick: str,
    split_candidates: Iterable[Mapping[str, Any]] = (),
    coarsen_candidates: Iterable[Mapping[str, Any]] = (),
    legal_candidate_ids: Iterable[str] = (),
    prev_makespan_us: float | None = None,
) -> str:
    """Render the agent-facing decision request as a markdown block."""
    splits = list(split_candidates)
    coarsens = list(coarsen_candidates)
    legal = list(legal_candidate_ids)

    lines: list[str] = []
    delta_part = ""
    if prev_makespan_us is not None and prev_makespan_us > 0:
        d = (makespan_us - prev_makespan_us) / prev_makespan_us * 100
        delta_part = f"  _(Δ {d:+.1f}% vs round {round_index - 1})_"
    lines.append(
        f"### Round {round_index} decision  ·  makespan = "
        f"{makespan_us:,.0f} µs{delta_part}"
    )
    lines.append("")
    lines.append(f"**Greedy pick:** `{greedy_pick}`")
    lines.append("")
    if splits:
        lines.append(f"**Split candidates ({len(splits)})** "
                     "_(measured / predicted > 1.3 AND ≥10% of makespan)_:")
        lines.append("")
        for c in splits[:6]:
            lines.append(
                f"- `split:{c.get('dispatch_id', '?')}` "
                f"on **{c.get('machine', '-')}** — "
                f"ratio **{c.get('ratio', 0):.2f}**, "
                f"region share **{c.get('region_share', 0) * 100:.1f}%**, "
                f"pred={c.get('predicted_us', 0):,.0f}µs, "
                f"meas={c.get('measured_us', 0):,.0f}µs"
            )
        lines.append("")
    if coarsens:
        lines.append(f"**Coarsen candidates ({len(coarsens)})** "
                     "_(same-backend transfer ≥20% of combined compute)_:")
        lines.append("")
        for c in coarsens[:6]:
            lines.append(
                f"- `coarsen:{c.get('first_dispatch_id', '?')}"
                f"+{c.get('second_dispatch_id', '?')}` "
                f"on **{c.get('machine', '-')}** — "
                f"transfer **{c.get('ratio', 0) * 100:.0f}%** "
                f"({c.get('transfer_us', 0):.0f}µs / "
                f"{c.get('combined_compute_us', 0):.0f}µs combined)"
            )
        lines.append("")
    if not splits and not coarsens:
        lines.append("_no split or coarsen candidates above threshold — "
                     "schedule is close to predicted; "
                     "agent should commit `keep:all`._")
        lines.append("")
    if legal:
        lines.append("<details><summary>Legal candidate IDs "
                     f"({len(legal)})</summary>")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in legal))
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def render_round_summary_markdown(
    *,
    round_index: int,
    makespan_us: float,
    schedule: Mapping[str, Any] | None,
    deltas: Iterable[Mapping[str, Any]] = (),
    greedy_pick: str = "keep:all",
    split_candidates: Iterable[Mapping[str, Any]] = (),
    coarsen_candidates: Iterable[Mapping[str, Any]] = (),
    legal_candidate_ids: Iterable[str] = (),
    prev_makespan_us: float | None = None,
) -> str:
    """Stitch the three blocks into one round-level message."""
    parts = [
        render_decision_markdown(
            round_index=round_index,
            makespan_us=makespan_us,
            greedy_pick=greedy_pick,
            split_candidates=split_candidates,
            coarsen_candidates=coarsen_candidates,
            legal_candidate_ids=legal_candidate_ids,
            prev_makespan_us=prev_makespan_us,
        ),
    ]
    if schedule is not None:
        parts.append(render_gantt_markdown(
            schedule, title=f"Schedule (round {round_index})",
        ))
    parts.append(render_deltas_markdown(
        deltas, title=f"Predicted vs measured (round {round_index})",
    ))
    return "\n".join(parts)
