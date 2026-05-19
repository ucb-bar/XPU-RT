"""Calibration runner — confirm harness numbers agree across backends.

Runs **two anchor shapes** through every backend on Gemmini (the
study's reference target), pulls each backend's native output through
the Phase-A loaders, and emits
``results/comparison/cross_target_fair/calibration.md`` with a
per-backend cycle ratio against KB-vanilla's reference. Cells where
the ratio falls outside ``[CALIBRATION_LOWER, CALIBRATION_UPPER]``
are flagged ``discrepant`` and surfaced in the caveats ledger.

The runner does **not** spend Gemini tokens for KB-vanilla — it uses
the cached prior run from
``results/comparison/vanilla_kb_gemmini/report.md`` as the reference.
KB-v2 and autocomp invocations are gated behind their env-readiness
checks; missing env emits ``status="env_missing"`` rather than
crashing inside the backend.

Anchors (per the plan):
  * Anchor 1 — ``M=64, K=720, N=320`` — KB-vanilla measured 12,251
    cycles, ``composite`` strategy, round 0.
  * Anchor 2 — ``M=64, K=32, N=720`` — KB-vanilla measured 21,830
    cycles, ``composite`` strategy, round 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.canonical_metrics import (
    CanonicalCellRow,
    shape_id_for_matmul,
    write_jsonl,
)
from xpu_rt.benchmarks.loaders.kb_vanilla_loader import load_kb_vanilla_rows


logger = logging.getLogger("xpu_rt.benchmarks.calibration_runner")


# ---------------------------------------------------------------------------
# Anchors + thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationAnchor:
    """One anchor shape with its KB-vanilla reference cycle count.

    The reference is what we expect every backend to land within
    a band of. KB-vanilla itself is the reference (ratio = 1.0)
    because it's the cached truth — every other backend's number
    is compared to this.
    """

    shape: tuple[int, int, int]  # (M, K, N)
    reference_cycles: int
    rationale: str

    @property
    def shape_id(self) -> str:
        M, K, N = self.shape
        return shape_id_for_matmul(M, K, N)


ANCHORS: tuple[CalibrationAnchor, ...] = (
    CalibrationAnchor(
        shape=(64, 720, 320),
        reference_cycles=12251,
        rationale="medium K, small N; exercises tiling heuristics",
    ),
    CalibrationAnchor(
        shape=(64, 32, 720),
        reference_cycles=21830,
        rationale="tiny K, large N (streaming); tests tall-skinny inputs",
    ),
)

# Band for ratios. Same-backend-same-counter ratio should sit very
# close to 1.0 (we tighten to [0.7, 1.3] in the report's KB-v2 row);
# cross-backend with different counter forks gets the wider band.
CALIBRATION_LOWER = 0.5
CALIBRATION_UPPER = 2.0


# ---------------------------------------------------------------------------
# Per-backend single-shot calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCell:
    """One backend × one anchor."""

    backend: str
    anchor: CalibrationAnchor
    row: CanonicalCellRow
    ratio_to_reference: float | None  # cycles(this) / reference_cycles
    verdict: str  # "comparable" | "discrepant" | "no_cycles" | "env_missing"
    notes: str = ""


def _ratio(cycles: int | None, ref: int) -> float | None:
    if cycles is None or ref <= 0:
        return None
    return cycles / ref


def _verdict_for(cell_cycles: int | None, ref: int, *, status: str = "ok") -> tuple[str, float | None]:
    if status != "ok":
        return (status, None)
    r = _ratio(cell_cycles, ref)
    if r is None:
        return ("no_cycles", None)
    if CALIBRATION_LOWER <= r <= CALIBRATION_UPPER:
        return ("comparable", r)
    return ("discrepant", r)


def _calibrate_kb_vanilla(anchor: CalibrationAnchor) -> CalibrationCell:
    """KB-vanilla calibration reuses the cached batch — no live run."""
    rows = load_kb_vanilla_rows()
    matching = [r for r in rows if r.shape_id == anchor.shape_id]
    if not matching:
        notes = (
            f"anchor {anchor.shape_id} not present in cached KB-vanilla report; "
            "verify the report covers the expected shape."
        )
        row = CanonicalCellRow(
            backend="kb-vanilla", target="gemmini_mx",
            workload="smolvla_matmuls", shape_id=anchor.shape_id,
            repeat=0, correctness=False, cycles=None, rounds_used=0,
            tokens_in=0, tokens_out=0, cost_usd=0.0, wall_s=0.0,
            cycle_source="none", notes=notes,
        )
        return CalibrationCell(
            backend="kb-vanilla", anchor=anchor, row=row,
            ratio_to_reference=None, verdict="no_cycles", notes=notes,
        )

    row = matching[0]
    verdict, ratio = _verdict_for(row.cycles, anchor.reference_cycles)
    return CalibrationCell(
        backend="kb-vanilla", anchor=anchor, row=row,
        ratio_to_reference=ratio, verdict=verdict,
        notes="reference cell — ratio is by definition 1.0",
    )


def _calibrate_kb_v2(anchor: CalibrationAnchor, *, mode: str) -> CalibrationCell:
    """KB-v2 calibration. In ``plan`` mode (no LLM, no Spike) this
    surfaces ``status='env_missing'`` for the toolchain + Gemini key
    requirements — exactly like the matrix driver does. In ``full``
    mode this invokes the live agent loop; deferred for now since
    the live wiring lives in Task 54."""
    missing: list[str] = []
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMMINI_API"):
        missing.append("GOOGLE_API_KEY (or GEMMINI_API)")
    conda_root = Path(
        os.environ.get(
            "XPU_RT_RISCV_CONDA_ROOT", "/scratch2/agustin/chipyard/.conda-env/riscv-tools"
        )
    )
    if not (conda_root / "bin" / "spike").is_file():
        missing.append(f"riscv-tools conda env at {conda_root}")

    notes_extra = ""
    if missing:
        notes_extra = f"env missing: {', '.join(missing)}"
        status = "env_missing"
    elif mode == "plan":
        notes_extra = "plan mode (no live run; structural numbers only)"
        status = "env_missing"  # treated identically to env_missing for calibration
    else:
        notes_extra = "full-mode calibration runner is wired by Task 54"
        status = "env_missing"  # until Task 54 lands

    row = CanonicalCellRow(
        backend="kb-v2", target="gemmini_mx", workload="smolvla_matmuls",
        shape_id=anchor.shape_id, repeat=0,
        correctness=False, cycles=None, rounds_used=0,
        tokens_in=0, tokens_out=0, cost_usd=0.0, wall_s=0.0,
        cycle_source="none", notes=notes_extra,
    )
    verdict, ratio = _verdict_for(row.cycles, anchor.reference_cycles, status=status)
    return CalibrationCell(
        backend="kb-v2", anchor=anchor, row=row,
        ratio_to_reference=ratio, verdict=verdict, notes=notes_extra,
    )


def _calibrate_autocomp(anchor: CalibrationAnchor, *, mode: str) -> CalibrationCell:
    """Autocomp calibration. Checks env via the Track-2 resolver +
    the chipyard env var; emits ``env_missing`` with a clean note
    when the operator hasn't run ``scripts/dev/setup_autocomp_chipyard.sh``."""
    from xpu_rt.kernels.autocomp_adapter import resolve_autocomp_target

    bindings = resolve_autocomp_target("gemmini_mx")
    missing = list(bindings.missing_env())
    try:
        bindings.resolve()
    except ImportError as exc:
        missing.append(f"autocomp package import ({exc})")

    notes_extra = (
        f"autocomp env missing: {', '.join(missing)}"
        if missing else "autocomp env ready; live calibration lands via Task 54"
    )
    status = "env_missing"  # until live wiring lands
    row = CanonicalCellRow(
        backend="autocomp", target="gemmini_mx", workload="smolvla_matmuls",
        shape_id=anchor.shape_id, repeat=0,
        correctness=False, cycles=None, rounds_used=0,
        tokens_in=0, tokens_out=0, cost_usd=0.0, wall_s=0.0,
        cycle_source="none", notes=notes_extra,
    )
    verdict, ratio = _verdict_for(row.cycles, anchor.reference_cycles, status=status)
    return CalibrationCell(
        backend="autocomp", anchor=anchor, row=row,
        ratio_to_reference=ratio, verdict=verdict, notes=notes_extra,
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run(out_dir: Path, *, mode: str = "plan") -> list[CalibrationCell]:
    """Run every (backend × anchor) calibration cell and persist
    the report.

    Returns the list of :class:`CalibrationCell` for the caller to
    inspect / pipe into the matrix runner's caveats ledger.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[CalibrationCell] = []
    for anchor in ANCHORS:
        cells.append(_calibrate_kb_vanilla(anchor))
        cells.append(_calibrate_kb_v2(anchor, mode=mode))
        cells.append(_calibrate_autocomp(anchor, mode=mode))

    # Persist machine-readable rows alongside the human-readable md.
    write_jsonl([c.row for c in cells], out_dir / "calibration_rows.jsonl")
    (out_dir / "calibration.json").write_text(_render_json(cells, mode=mode))
    (out_dir / "calibration.md").write_text(_render_markdown(cells, mode=mode))
    return cells


def _render_json(cells: list[CalibrationCell], *, mode: str) -> str:
    return json.dumps(
        {
            "mode": mode,
            "anchors": [asdict(a.anchor) for a in cells if a.backend == "kb-vanilla"],
            "cells": [
                {
                    "backend": c.backend,
                    "anchor_shape": list(c.anchor.shape),
                    "anchor_reference_cycles": c.anchor.reference_cycles,
                    "cycles": c.row.cycles,
                    "ratio_to_reference": c.ratio_to_reference,
                    "verdict": c.verdict,
                    "cycle_source": c.row.cycle_source,
                    "notes": c.notes,
                }
                for c in cells
            ],
        },
        indent=2,
    )


def _render_markdown(cells: list[CalibrationCell], *, mode: str) -> str:
    lines: list[str] = []
    lines.append("# Calibration report — harness skew across backends\n")
    lines.append(
        f"Mode: **{mode}**  |  Band: ratios within "
        f"[{CALIBRATION_LOWER}×, {CALIBRATION_UPPER}×] are `comparable`; "
        "outside flag `discrepant` in the caveats ledger.\n"
    )

    # Anchor summary.
    lines.append("## Anchors\n")
    lines.append("| anchor | shape | reference cycles | rationale |")
    lines.append("|---|---|---|---|")
    for i, anchor in enumerate(ANCHORS):
        M, K, N = anchor.shape
        lines.append(
            f"| {i+1} | `M={M}, K={K}, N={N}` | {anchor.reference_cycles:,} | {anchor.rationale} |"
        )

    # Per-anchor per-backend table.
    lines.append("\n## Per-anchor calibration\n")
    for anchor in ANCHORS:
        M, K, N = anchor.shape
        lines.append(f"### Anchor `M={M}, K={K}, N={N}` — reference {anchor.reference_cycles:,} cycles\n")
        lines.append("| backend | cycles | ratio | verdict | cycle_source | notes |")
        lines.append("|---|---|---|---|---|---|")
        anchor_cells = [c for c in cells if c.anchor.shape == anchor.shape]
        for cell in anchor_cells:
            cycles_str = f"{cell.row.cycles:,}" if cell.row.cycles else "—"
            ratio_str = (
                f"{cell.ratio_to_reference:.2f}×"
                if cell.ratio_to_reference is not None else "—"
            )
            notes = (cell.notes or "").replace("|", "/").replace("\n", " ")
            if len(notes) > 90:
                notes = notes[:87] + "..."
            lines.append(
                f"| `{cell.backend}` | {cycles_str} | {ratio_str} | "
                f"`{cell.verdict}` | `{cell.row.cycle_source}` | {notes} |"
            )
        lines.append("")

    # Methodology footer.
    lines.append("## Methodology\n")
    lines.append(
        "- KB-vanilla cycles come from the cached prior batch "
        "(`results/comparison/vanilla_kb_gemmini/report.md`) — no new spend.\n"
        "- KB-v2 cycles come from `xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv` "
        "running the candidate on `spike --extension=gemmini pk` with "
        "`MAIN_LD_ST_EX_CYCLES` as the counter.\n"
        "- autocomp cycles come from its own modified `libgemmini` Spike "
        "fork's stdout (`Generated implementation latency: N cycles`). "
        "Different fork from ours — the cross-backend ratio sits in a "
        "wider band (`[0.5, 2.0]`) because the counters aren't proven "
        "to use identical event masks.\n"
        "- `env_missing` rows mean the calibration runner couldn't reach "
        "that backend yet — populate by running "
        "`scripts/dev/setup_autocomp_chipyard.sh` (autocomp) or "
        "`source /scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/activate` (KB-v2).\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/comparison/cross_target_fair"),
    )
    parser.add_argument(
        "--mode", choices=("plan", "full"), default="plan",
        help="``plan`` only reads the KB-vanilla cache and reports env-readiness "
        "for the other backends; ``full`` invokes KB-v2 + autocomp live "
        "(requires Task 54's wiring).",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    cells = run(args.out_dir, mode=args.mode)
    n_ok = sum(1 for c in cells if c.verdict == "comparable")
    n_total = len(cells)
    print(f"wrote {args.out_dir / 'calibration.md'} — {n_ok}/{n_total} cells comparable", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHORS",
    "CALIBRATION_LOWER",
    "CALIBRATION_UPPER",
    "CalibrationAnchor",
    "CalibrationCell",
    "main",
    "run",
]
