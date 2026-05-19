"""Load KB-vanilla's per-shape results from its markdown report.

The KB-vanilla batch driver
(:mod:`xpu_rt.kb_gemmini.kb_pipeline_driver`) emits
``results/comparison/vanilla_kb_gemmini/report.md`` with a per-shape
table:

    | # | shape | result | round | strategy | cycles |

This loader parses that table into
:class:`~xpu_rt.benchmarks.canonical_metrics.CanonicalCellRow`
records. The KB-vanilla batch is the cached, $0.16 prior result —
every row this loader emits is tagged ``cycle_source="cached"`` and
``repeat=0``.

The report also carries an aggregate header (correctness rate,
Gemini $, wall time); the loader doesn't try to back-compute
per-shape ``cost_usd`` and ``wall_s`` from that aggregate (the
per-shape breakdown isn't in the markdown — just totals). Per-row
``cost_usd`` and ``wall_s`` are set to ``0.0`` and the aggregator
re-derives the per-cell totals from the row's headline numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow, shape_id_for_matmul


DEFAULT_REPORT_PATH = Path(
    "/scratch2/agustin/xpu-rt-integration/results/comparison/vanilla_kb_gemmini/report.md"
)


# Regex for the per-contract table rows. Matches:
#   | 4 | [64, 720]×[720, 2048] | **✓** | 0 | composite | 62 179 |
#   | 1 | [64, 960]×[960, 960] | ✗ | — | — | — |
_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*"          # index
    r"\|\s*(\[[\d, ]+\]×\[[\d, ]+\])\s*"  # shape
    r"\|\s*(\*\*✓\*\*|✓|✗)\s*"  # result icon
    r"\|\s*([\d—-]+)\s*"        # round (or — for failures)
    r"\|\s*([\w_—-]+)\s*"        # strategy
    r"\|\s*([\d  —-]+)\s*\|"  # cycles (spaces / nbsp may show up)
)


def _strip_thousands(s: str) -> int | None:
    """Convert ``"62 179"`` / ``"12 251"`` / ``"—"`` → int / None."""
    cleaned = s.replace(" ", "").replace(" ", "").replace(",", "").strip()
    if cleaned in ("", "—", "-"):
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _parse_shape(shape_str: str) -> tuple[int, int, int] | None:
    """Parse ``"[64, 720]×[720, 2048]"`` → (64, 720, 2048)."""
    m = re.match(r"\[(\d+),\s*(\d+)\]×\[\d+,\s*(\d+)\]", shape_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_kb_vanilla_rows(
    report_path: Path | None = None,
    *,
    target: str = "gemmini_mx",
    workload: str = "smolvla_matmuls",
) -> list[CanonicalCellRow]:
    """Parse KB-vanilla's report.md and emit one canonical row per
    table entry.

    Args:
        report_path: Override the default report location (handy for
            tests). When ``None``, reads
            ``results/comparison/vanilla_kb_gemmini/report.md``.
        target: Stamped into every emitted row. The report is
            Gemmini-only today; passing ``saturn_opu_v128`` is a no-op
            because there's no Saturn-side KB-vanilla batch.
        workload: ``smolvla_matmuls`` for the standard report; the
            block-level workload doesn't have a KB-vanilla batch yet.

    Returns:
        14 rows for the canonical SmolVLA-matmuls report. Empty list
        when the report file is missing.
    """
    path = report_path or DEFAULT_REPORT_PATH
    if not path.is_file():
        return []
    rows: list[CanonicalCellRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        idx, shape_str, result_icon, round_str, strategy, cycles_str = m.groups()
        shape = _parse_shape(shape_str)
        if shape is None:
            continue
        M, K, N = shape
        correct = "✓" in result_icon
        cycles = _strip_thousands(cycles_str) if correct else None
        round_num = _strip_thousands(round_str) or 0
        rows.append(
            CanonicalCellRow(
                backend="kb-vanilla",
                target=target,
                workload=workload,
                shape_id=shape_id_for_matmul(M, K, N),
                repeat=0,
                correctness=correct,
                cycles=cycles,
                rounds_used=int(round_num) if correct else 0,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                wall_s=0.0,
                cycle_source="cached",
                notes=(
                    f"strategy={strategy.strip()}"
                    if correct and strategy.strip() not in ("—", "-")
                    else ""
                ),
            )
        )
    return rows


__all__ = ["DEFAULT_REPORT_PATH", "load_kb_vanilla_rows"]
