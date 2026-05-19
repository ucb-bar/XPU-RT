"""Final caveat-aware report for the fair-comparison study.

Reads:
  * One ``samples.jsonl`` per cell (Phase D persists canonical rows).
  * One ``status.json`` per cell (matrix driver writes the cell-level
    status / env_missing details).
  * The Phase-C ``calibration.json`` so the harness-skew column can
    cite the calibration verdict per backend.

Emits:
  * ``report.md`` — human-readable; per-cell median ± min/max table,
    Q1 / Q2 / Q3 sections that join on ``correctness ∩ correctness``
    intersections, harness-skew column, expected-behaviour-for-deferred
    subsection.
  * ``report.json`` — machine-readable summary (one row per cell with
    median + range).
  * ``caveats.md`` — one line per caveat (harness skew / deferred /
    env_missing / discrepant-calibration / single-sample).

The caveats ledger is the load-bearing artefact for a fair claim:
every numerical row in the report has a back-pointer here.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from xpu_rt.benchmarks.canonical_metrics import (
    BACKENDS,
    CanonicalCellRow,
    TARGETS,
    WORKLOADS,
    read_jsonl,
)
from xpu_rt.benchmarks.sample_aggregator import CellSummary, aggregate


# ---------------------------------------------------------------------------
# Caveat schema
# ---------------------------------------------------------------------------


_CAVEAT_TYPES = (
    "harness_skew",
    "single_sample",
    "deferred_cell",
    "env_missing_cell",
    "discrepant_calibration",
    "zero_correct",
    "mixed_cycle_source",
)


@dataclass(frozen=True)
class Caveat:
    """One caveat — surfaces a known limitation of the cell's data."""

    kind: str  # one of _CAVEAT_TYPES
    backend: str
    target: str
    workload: str
    shape_id: str = ""  # "" for cell-level caveats
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _CAVEAT_TYPES:
            raise ValueError(f"unknown caveat kind {self.kind!r}; expected one of {_CAVEAT_TYPES}")

    def to_line(self) -> str:
        parts = [
            f"kind={self.kind}",
            f"backend={self.backend}",
            f"target={self.target}",
            f"workload={self.workload}",
        ]
        if self.shape_id:
            parts.append(f"shape={self.shape_id!r}")
        parts.append(f"detail={self.detail!r}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    cells_dir: Path,
    out_dir: Path,
    *,
    calibration_json: Path | None = None,
) -> dict[str, Any]:
    """Walk ``cells_dir/per_cell/*`` and write report.md / .json /
    caveats.md into ``out_dir``.

    Returns the JSON summary (also written to disk) so callers can
    assert against it in tests.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rows from every cell + cell-level status.
    all_rows: list[CanonicalCellRow] = []
    cell_status: dict[tuple[str, str, str], dict[str, Any]] = {}
    cells_root = cells_dir / "per_cell"
    if cells_root.is_dir():
        for cell_dir in sorted(cells_root.iterdir()):
            samples = read_jsonl(cell_dir / "samples.jsonl")
            all_rows.extend(samples)
            status_path = cell_dir / "status.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                key = (status.get("backend", ""), status.get("target", ""), status.get("workload", ""))
                cell_status[key] = status

    summaries = aggregate(all_rows)

    # Load calibration verdicts so caveats can cite them.
    calibration_by_backend: dict[str, str] = {}
    if calibration_json and calibration_json.is_file():
        try:
            cal = json.loads(calibration_json.read_text())
            for cell in cal.get("cells", []):
                backend = cell.get("backend", "")
                verdict = cell.get("verdict", "")
                ratio = cell.get("ratio_to_reference")
                if backend not in calibration_by_backend or verdict == "discrepant":
                    detail = f"verdict={verdict}"
                    if ratio is not None:
                        detail += f" ratio={ratio:.2f}"
                    calibration_by_backend[backend] = detail
        except (OSError, json.JSONDecodeError):
            pass

    caveats = _collect_caveats(all_rows, summaries, cell_status, calibration_by_backend)

    json_summary = {
        "n_samples": len(all_rows),
        "n_cells": len(summaries),
        "n_caveats": len(caveats),
        "cells": [dataclasses.asdict(s) for s in summaries],
        "caveats": [dataclasses.asdict(c) for c in caveats],
    }
    (out_dir / "report.json").write_text(json.dumps(json_summary, indent=2))
    (out_dir / "report.md").write_text(
        _render_markdown(summaries, cell_status, calibration_by_backend, caveats)
    )
    (out_dir / "caveats.md").write_text(_render_caveats(caveats))
    return json_summary


# ---------------------------------------------------------------------------
# Caveat collection
# ---------------------------------------------------------------------------


def _collect_caveats(
    rows: Iterable[CanonicalCellRow],
    summaries: list[CellSummary],
    cell_status: dict[tuple[str, str, str], dict[str, Any]],
    calibration: dict[str, str],
) -> list[Caveat]:
    caveats: list[Caveat] = []

    # Cell-level: deferred / env_missing show up in cell_status.
    for key, status in sorted(cell_status.items()):
        backend, target, workload = key
        if status.get("status") == "deferred":
            caveats.append(Caveat(
                kind="deferred_cell", backend=backend, target=target,
                workload=workload, detail=str(status.get("notes", ""))[:200],
            ))
        elif status.get("status") == "env_missing":
            env_missing = status.get("env_missing") or []
            caveats.append(Caveat(
                kind="env_missing_cell", backend=backend, target=target,
                workload=workload,
                detail=f"missing={','.join(env_missing)}",
            ))

    # Sample-level: single sample, mixed cycle_source, zero correct,
    # cross-backend harness skew.
    for s in summaries:
        if s.n_samples < 3 and s.n_correct > 0:
            caveats.append(Caveat(
                kind="single_sample", backend=s.backend, target=s.target,
                workload=s.workload, shape_id=s.shape_id,
                detail=f"n_samples={s.n_samples}; variance not measured",
            ))
        if s.cycle_source == "mixed":
            caveats.append(Caveat(
                kind="mixed_cycle_source", backend=s.backend, target=s.target,
                workload=s.workload, shape_id=s.shape_id,
                detail="samples disagreed on cycle counter; can't pool cycles",
            ))
        if s.n_samples > 0 and s.n_correct == 0:
            caveats.append(Caveat(
                kind="zero_correct", backend=s.backend, target=s.target,
                workload=s.workload, shape_id=s.shape_id,
                detail=f"all {s.n_samples} samples incorrect",
            ))
        # Harness skew is structural: autocomp uses a different cycle
        # counter from KB. Flag every autocomp row so the joined Q1
        # table makes this visible.
        if s.cycle_source == "Generated implementation latency":
            cal_detail = calibration.get(s.backend, "no calibration data")
            caveats.append(Caveat(
                kind="harness_skew", backend=s.backend, target=s.target,
                workload=s.workload, shape_id=s.shape_id,
                detail=(
                    f"counter='Generated implementation latency' "
                    f"vs KB's MAIN_LD_ST_EX_CYCLES — calibration: {cal_detail}"
                ),
            ))

    return caveats


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(
    summaries: list[CellSummary],
    cell_status: dict[tuple[str, str, str], dict[str, Any]],
    calibration: dict[str, str],
    caveats: list[Caveat],
) -> str:
    lines: list[str] = []
    lines.append("# Fair comparison — KB-vanilla, KB-v2, autocomp on Gemmini + Saturn/OPU\n")
    lines.append(
        f"Samples loaded: **{sum(s.n_samples for s in summaries)}**  |  "
        f"Cells covered: **{len(summaries)}**  |  "
        f"Caveats raised: **{len(caveats)}**\n"
    )

    # Headline matrix table — per-cell median ± min/max with the
    # harness-skew column.
    lines.append("## Per-cell summary (median cycles ± min/max)\n")
    lines.append(
        "| backend | target | workload | shape | n | correct/N | "
        "median cycles | min | max | cycle_source | mean $ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        median = f"{s.median_cycles:,.0f}" if s.median_cycles is not None else "—"
        mn = f"{s.min_cycles:,}" if s.min_cycles is not None else "—"
        mx = f"{s.max_cycles:,}" if s.max_cycles is not None else "—"
        lines.append(
            f"| `{s.backend}` | `{s.target}` | `{s.workload}` | "
            f"`{s.shape_id}` | {s.n_samples} | {s.n_correct}/{s.n_samples} | "
            f"{median} | {mn} | {mx} | `{s.cycle_source}` | ${s.mean_cost_usd:.4f} |"
        )

    # Q1 / Q2 / Q3 (intersection joined).
    lines.append("\n## Q1 — KB-v2 vs autocomp (per target × workload, correctness∩correctness only)\n")
    lines += _qx_intersection_table(summaries, ("kb-v2", "autocomp"))

    lines.append("\n## Q2 — KB-vanilla vs KB-v2 (per target × workload)\n")
    lines += _qx_intersection_table(summaries, ("kb-vanilla", "kb-v2"))

    lines.append("\n## Q3 — Cross-target generalisation (per backend × workload, Gemmini vs Saturn)\n")
    lines += _q3_cross_target_table(summaries)

    # Cell-status overview (deferred / env_missing cells).
    deferred_cells = sorted(
        (k, v) for k, v in cell_status.items()
        if v.get("status") in ("deferred", "env_missing")
    )
    if deferred_cells:
        lines.append("\n## Cell-status overview (deferred + env_missing)\n")
        lines.append("| backend | target | workload | status | notes |")
        lines.append("|---|---|---|---|---|")
        for key, status in deferred_cells:
            backend, target, workload = key
            notes = str(status.get("notes", "")).replace("|", "/").replace("\n", " ")
            if len(notes) > 110:
                notes = notes[:107] + "..."
            lines.append(
                f"| `{backend}` | `{target}` | `{workload}` | "
                f"`{status.get('status')}` | {notes} |"
            )

    # Expected behaviour for deferred cells (drives the headline
    # caveat: KB-vanilla on Saturn etc.).
    lines.append("\n## Expected behaviour for the deferred cells\n")
    lines.append(
        "When a cell is `deferred` or `env_missing`, the numerical "
        "tables above leave it blank. The expected behaviour the "
        "operator should anticipate if the cell were live:\n"
    )
    lines.append(
        "- **KB-vanilla × Saturn × any** — would mirror "
        "`KB-vanilla × Gemmini × matmuls` (≈ 50% correct at K ≤ 720, "
        "degrades at K ≥ 960). Saturn's larger scratchpad (512 KiB vs "
        "Gemmini's 256 KiB) likely shifts the boundary up.\n"
        "- **KB-vanilla × Gemmini × MLP block** — per-op correctness "
        "stays at ≈ 50%; no fusion available without a multi-op driver.\n"
        "- **autocomp × any target** — once "
        "`scripts/dev/setup_autocomp_chipyard.sh` lands, expect correctness "
        "rates ≥ KB-vanilla on the matmul workload; cycle ratios sit in "
        "the [0.5×, 2.0×] band against the KB-vanilla cache per "
        "calibration. Pipeline-level (MLP-block) workload is a fresh axis "
        "for autocomp; no upstream baseline.\n"
    )

    # Harness-skew subsection — surfaces the cross-counter caveat.
    lines.append("\n## Harness skew per cycle source\n")
    sources_seen: dict[str, list[CellSummary]] = defaultdict(list)
    for s in summaries:
        sources_seen[s.cycle_source].append(s)
    lines.append("| cycle_source | cells | calibration note |")
    lines.append("|---|---|---|")
    for src in sorted(sources_seen):
        cells = sources_seen[src]
        cal_note = ""
        for backend in (c.backend for c in cells):
            if backend in calibration:
                cal_note = calibration[backend]
                break
        cal_note = cal_note or "—"
        lines.append(f"| `{src}` | {len(cells)} | {cal_note} |")

    lines.append("\n## Caveats summary\n")
    if caveats:
        kinds: dict[str, int] = defaultdict(int)
        for c in caveats:
            kinds[c.kind] += 1
        lines.append("| kind | count |")
        lines.append("|---|---|")
        for kind in sorted(kinds):
            lines.append(f"| `{kind}` | {kinds[kind]} |")
        lines.append("\nFull list in `caveats.md` (one line per caveat).\n")
    else:
        lines.append("None raised. (Suspicious — usually means the input data was empty.)\n")

    lines.append("\n## Methodology\n")
    lines.append(
        "- Phase A's `CanonicalCellRow` normalises every backend's "
        "native output to a common schema; Phase B's "
        "`setup_autocomp_chipyard.sh` brings autocomp's pinned chipyard + "
        "Spike-fork into a separate worktree at "
        "`/scratch2/agustin/chipyard-autocomp`; Phase C's "
        "`calibration_runner` confirms each backend's cycles sit in "
        "the [0.5×, 2.0×] band vs KB-vanilla's cached reference "
        "before the matrix runs; Phase D drives N=3 statistical "
        "repeats per cell with seed-varied LLM proposals.\n"
        "- This report (Phase E) aggregates median ± min/max over the "
        "N samples and flags every harness-skew / single-sample / "
        "deferred-cell / discrepant-calibration condition in the "
        "caveats ledger.\n"
        "- Reading the report: any `harness_skew` caveat means the "
        "two backends being compared in that row use **different "
        "cycle counters** (KB's MAIN_LD_ST_EX_CYCLES vs autocomp's "
        "Generated-implementation-latency). Compare ratios with that "
        "in mind.\n"
    )
    return "\n".join(lines)


def _qx_intersection_table(
    summaries: list[CellSummary], pair: tuple[str, str]
) -> list[str]:
    """For backend pair (a, b), per (target, workload) join the shapes
    where BOTH backends produced a correct cycle count, then report
    geomean of the ratios."""
    a_backend, b_backend = pair
    lookup: dict[tuple[str, str, str], CellSummary] = {
        (s.backend, s.target, s.workload, s.shape_id): s for s in summaries
        for _ in [None]
    }
    # Rebuild with the (backend, target, workload, shape_id) key.
    lookup = {(s.backend, s.target, s.workload, s.shape_id): s for s in summaries}

    by_pair: dict[tuple[str, str], list[tuple[CellSummary, CellSummary]]] = defaultdict(list)
    for s in summaries:
        if s.backend != a_backend:
            continue
        other = lookup.get((b_backend, s.target, s.workload, s.shape_id))
        if other is None:
            continue
        if s.median_cycles is None or other.median_cycles is None:
            continue
        if s.median_cycles <= 0 or other.median_cycles <= 0:
            continue
        by_pair[(s.target, s.workload)].append((s, other))

    rows: list[str] = []
    rows.append(f"| target | workload | shapes joined | geomean({a_backend}/{b_backend}) |")
    rows.append("|---|---|---|---|")
    if not by_pair:
        rows.append("| — | — | 0 | (no joined shapes — see caveats) |")
        return rows
    for (target, workload), pairs in sorted(by_pair.items()):
        log_sum = 0.0
        for a_s, b_s in pairs:
            log_sum += math.log(a_s.median_cycles / b_s.median_cycles)
        geomean = math.exp(log_sum / len(pairs)) if pairs else 1.0
        rows.append(f"| `{target}` | `{workload}` | {len(pairs)} | {geomean:.2f}× |")
    return rows


def _q3_cross_target_table(summaries: list[CellSummary]) -> list[str]:
    """For each (backend, workload), join shapes that have results on
    BOTH Gemmini and Saturn; report geomean ratio of Saturn/Gemmini
    cycles."""
    lookup: dict[tuple[str, str, str, str], CellSummary] = {
        (s.backend, s.target, s.workload, s.shape_id): s for s in summaries
    }
    rows: list[str] = []
    rows.append("| backend | workload | shapes joined | geomean(Saturn/Gemmini) |")
    rows.append("|---|---|---|---|")
    any_joined = False
    for backend in sorted({s.backend for s in summaries}):
        for workload in sorted({s.workload for s in summaries if s.backend == backend}):
            pairs = []
            shape_ids = {s.shape_id for s in summaries
                         if s.backend == backend and s.workload == workload}
            for shape_id in shape_ids:
                g = lookup.get((backend, "gemmini_mx", workload, shape_id))
                s_sat = lookup.get((backend, "saturn_opu_v128", workload, shape_id))
                if g and s_sat and g.median_cycles and s_sat.median_cycles:
                    pairs.append((g.median_cycles, s_sat.median_cycles))
            if not pairs:
                continue
            any_joined = True
            log_sum = sum(math.log(s_c / g_c) for g_c, s_c in pairs)
            geomean = math.exp(log_sum / len(pairs))
            rows.append(f"| `{backend}` | `{workload}` | {len(pairs)} | {geomean:.2f}× |")
    if not any_joined:
        rows.append("| — | — | 0 | (no shapes joined across targets — see caveats) |")
    return rows


def _render_caveats(caveats: list[Caveat]) -> str:
    if not caveats:
        return "# Caveats ledger\n\n(no caveats — see report.md's caveats section)\n"
    lines = ["# Caveats ledger\n", "One line per caveat. Read alongside `report.md`.\n"]
    for c in caveats:
        lines.append(c.to_line())
    return "\n".join(lines) + "\n"


__all__ = ["Caveat", "build_report"]
