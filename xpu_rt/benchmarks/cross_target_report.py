"""Aggregator for the cross-target matrix study.

Reads every per-cell :class:`CellResult` from
``<out_dir>/per_cell/*/status.json`` and emits the headline report
answering the three questions the study was set up to ask:

  * **Q1 — Is KB as good as autocomp?** Cell-by-cell at single-kernel
    level, per target.
  * **Q2 — Does the agentic graph analysis pay off?** KB-v2 vs
    KB-vanilla cells (where both are wired live), per target +
    workload.
  * **Q3 — Does the agentic approach generalise across targets?**
    Same backend across Gemmini and OPU, per workload.

Cells with status other than ``"ok"`` (deferred / env_missing /
budget_exceeded / error) are surfaced in the report with their
notes, so the reader can see *why* a measurement is missing — the
matrix never silently drops data.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_OK = "ok"
_HEAD_ICON = {
    "ok": "✓",
    "deferred": "—",
    "env_missing": "⊘",
    "budget_exceeded": "$",
    "error": "✗",
}


def write_reports(out_dir: Path, results: Iterable[Any], *, mode: str = "plan") -> None:
    """Write ``report.md`` + ``report.json`` under ``out_dir``."""
    results_list = list(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(_render_json(results_list, mode=mode))
    (out_dir / "report.md").write_text(_render_markdown(results_list, mode=mode))


def _render_json(results: list[Any], *, mode: str) -> str:
    return json.dumps(
        {
            "mode": mode,
            "n_cells": len(results),
            "cells": [dataclasses.asdict(c) for c in results],
        },
        indent=2,
    )


def _render_markdown(results: list[Any], *, mode: str) -> str:
    lines: list[str] = []
    lines.append("# Cross-target × cross-backend comparison\n")
    lines.append(f"Mode: **{mode}**  |  Cells: **{len(results)}**\n")

    by_status = _count_by_status(results)
    lines.append("Cell status breakdown:")
    lines.append("")
    for status in ("ok", "deferred", "env_missing", "budget_exceeded", "error"):
        n = by_status.get(status, 0)
        if n:
            lines.append(f"- **{status}**: {n}")
    lines.append("")

    # Headline matrix table.
    lines.append("## Matrix (status × backend × target × workload)\n")
    lines.append("| backend | target | workload | status | n_vanilla | n_agentic | planner_speedup | notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in results:
        notes = (c.notes or "").replace("\n", " ").replace("|", "/")
        if len(notes) > 110:
            notes = notes[:107] + "..."
        speedup = (
            f"{c.planner_estimated_speedup:.2f}x"
            if c.planner_estimated_speedup and c.planner_estimated_speedup > 0 else "—"
        )
        lines.append(
            f"| `{c.backend}` | `{c.target}` | `{c.workload}` | "
            f"{_HEAD_ICON.get(c.status, '?')} {c.status} | "
            f"{c.n_kernels_vanilla or '—'} | {c.n_kernels_agentic or '—'} | "
            f"{speedup} | {notes} |"
        )

    # Q1 — KB vs autocomp per target.
    lines.append("\n## Q1 — Is KB (KB-v2) as good as autocomp?\n")
    lines += _q1_rows(results)

    # Q2 — does agentic graph analysis pay off?
    lines.append("\n## Q2 — Does the agentic graph analysis pay off (KB-vanilla vs KB-v2)?\n")
    lines += _q2_rows(results)

    # Q3 — cross-target generalisation.
    lines.append("\n## Q3 — Does the agentic flow generalise across targets?\n")
    lines += _q3_rows(results)

    # Env-readiness section.
    missing_cells = [c for c in results if c.status == "env_missing"]
    if missing_cells:
        lines.append("\n## Env-readiness — what to set for the deferred cells\n")
        env_to_cells: dict[str, list[Any]] = defaultdict(list)
        for c in missing_cells:
            for e in c.env_missing:
                env_to_cells[e].append(c)
        for env, cells in sorted(env_to_cells.items()):
            ids = ", ".join(f"`{c.backend}`×`{c.target}`×`{c.workload}`" for c in cells)
            lines.append(f"- **{env}** — needed by: {ids}")

    # Methodology footer.
    lines.append("\n## Methodology\n")
    lines.append(
        "- **plan mode**: structural numbers + planner verdicts. Re-uses "
        "`xpu_rt.benchmarks.run_pipeline_comparison.run` for the KB-v2 cells; "
        "no Gemini spend, no Spike execution. Numbers below are derived "
        "from the FusionPlanner's cost model (after the Track 0 weight-"
        "tiling discount fix) + the MegaContractEmitter.\n"
        "- **full mode**: wires KB v2's agent loop + the per-target "
        "`CRiscvEvaluator` (Gemmini → `--extension=gemmini pk`, Saturn → "
        "`--isa=rv64gcv_zvl128b pk`). Each per-cell call honours the "
        "$0.50 ceiling via `gemini_usage.check_pre_call`; cells past "
        "the budget cap fail clean with `status=budget_exceeded`.\n"
    )
    return "\n".join(lines)


def _count_by_status(results: list[Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for c in results:
        out[c.status] += 1
    return dict(out)


def _q1_rows(results: list[Any]) -> list[str]:
    """KB-v2 vs autocomp per (target, workload)."""
    by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for c in results:
        if c.backend in ("kb-v2", "autocomp"):
            by_key[(c.target, c.workload)][c.backend] = c

    lines: list[str] = []
    lines.append("| target | workload | KB-v2 | autocomp | comparison |")
    lines.append("|---|---|---|---|---|")
    for (target, workload), cells in sorted(by_key.items()):
        kb = cells.get("kb-v2")
        ac = cells.get("autocomp")
        kb_str = _cell_metric_str(kb)
        ac_str = _cell_metric_str(ac)
        comparison = _compare_two(kb, ac)
        lines.append(f"| `{target}` | `{workload}` | {kb_str} | {ac_str} | {comparison} |")
    return lines


def _q2_rows(results: list[Any]) -> list[str]:
    """KB-vanilla vs KB-v2 per (target, workload). Only Gemmini has
    KB-vanilla wired today."""
    by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for c in results:
        if c.backend in ("kb-vanilla", "kb-v2"):
            by_key[(c.target, c.workload)][c.backend] = c
    lines: list[str] = []
    lines.append("| target | workload | KB-vanilla | KB-v2 (agentic) | agentic vs vanilla |")
    lines.append("|---|---|---|---|---|")
    for (target, workload), cells in sorted(by_key.items()):
        v = cells.get("kb-vanilla")
        v2 = cells.get("kb-v2")
        lines.append(
            f"| `{target}` | `{workload}` | {_cell_metric_str(v)} | "
            f"{_cell_metric_str(v2)} | {_compare_two(v, v2)} |"
        )
    return lines


def _q3_rows(results: list[Any]) -> list[str]:
    """For each (backend, workload), show side-by-side Gemmini vs Saturn."""
    by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for c in results:
        by_key[(c.backend, c.workload)][c.target] = c
    lines: list[str] = []
    lines.append("| backend | workload | Gemmini | Saturn | cross-target ratio |")
    lines.append("|---|---|---|---|---|")
    for (backend, workload), cells in sorted(by_key.items()):
        g = cells.get("gemmini_mx")
        s = cells.get("saturn_opu_v128")
        lines.append(
            f"| `{backend}` | `{workload}` | {_cell_metric_str(g)} | "
            f"{_cell_metric_str(s)} | {_compare_two(g, s)} |"
        )
    return lines


def _cell_metric_str(c: Any | None) -> str:
    if c is None:
        return "—"
    if c.status != _OK:
        return f"{_HEAD_ICON.get(c.status, '?')} {c.status}"
    if not math.isnan(c.geomean_cycles) and c.geomean_cycles > 0:
        return f"{c.geomean_cycles:.0f} cyc"
    if not math.isnan(c.correctness_rate):
        return f"{c.correctness_rate:.0%} correct"
    if c.planner_estimated_speedup and c.planner_estimated_speedup > 1.0:
        return f"{c.planner_estimated_speedup:.2f}x planner"
    return "ok"


def _compare_two(a: Any | None, b: Any | None) -> str:
    if a is None or b is None:
        return "—"
    if a.status != _OK or b.status != _OK:
        return f"a={a.status} / b={b.status}"
    # Prefer measured cycles when present.
    if (
        not math.isnan(a.geomean_cycles) and not math.isnan(b.geomean_cycles)
        and a.geomean_cycles > 0 and b.geomean_cycles > 0
    ):
        return f"{a.geomean_cycles / b.geomean_cycles:.2f}× (a/b cycles)"
    # Fall back to planner speedup ratio.
    if a.planner_estimated_speedup and b.planner_estimated_speedup:
        return f"{a.planner_estimated_speedup / b.planner_estimated_speedup:.2f}× (planner)"
    return "—"


__all__ = ["write_reports"]
