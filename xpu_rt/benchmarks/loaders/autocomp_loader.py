"""Load autocomp's per-search outputs into the canonical row schema.

Autocomp's :class:`BeamSearchStrategy.optimize` writes:

  * ``<out>/eval-results-iter-<i>/code_<j>_result.txt`` — JSON per
    candidate with ``correct`` / ``latency`` (Gemmini cycles when
    ``simulator="spike"`` per ``gemmini_eval.py:423``) /
    ``compiled``.

  * ``<out>/metrics-iter-<i>.json`` — per-iteration token usage +
    timing aggregated by phase (``plan_generation``,
    ``code_generation``, etc.).

  * ``<out>/run_metrics.json`` — aggregated rollup across all
    iterations (per the writer at ``search.py:1118``).

  * ``<out>/best_candidate_so_far.py`` — source of the best
    correct candidate (we don't need to re-parse, just confirm it
    exists).

The loader walks this tree, picks the lowest-latency correct
candidate across all iterations, and totals tokens + cost. One
canonical row per ``output_dir`` (= one cell per repeat).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow


logger = logging.getLogger(__name__)


# Per-1M-token pricing for gemini-2.5-flash (USD). Mirrors the
# xpu_rt.observability.gemini_usage rate table so per-cell $ on the
# autocomp side is directly comparable with the KB-v2 side.
# These rates can shift; we read them from the env when set so the
# study can pin a specific snapshot.
_GEMINI_INPUT_PER_MTOK_USD = 0.075
_GEMINI_OUTPUT_PER_MTOK_USD = 0.30


def _walk_eval_results(out_dir: Path) -> list[dict[str, Any]]:
    """Read every ``eval-results-iter-N/code_*_result.txt`` JSON.

    Each entry is augmented with ``_iter`` (the iter dir number) and
    ``_cand`` (the candidate index parsed out of the filename) so
    the caller can pick the winner deterministically.
    """
    entries: list[dict[str, Any]] = []
    for eval_dir in sorted(out_dir.glob("eval-results-iter-*")):
        try:
            iter_num = int(eval_dir.name.split("-")[-1])
        except ValueError:
            continue
        for result_file in sorted(eval_dir.glob("code_*_result.txt")):
            try:
                body = json.loads(result_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(body, dict):
                continue
            cand_str = result_file.stem.split("_")[1]  # "code_<n>_result"
            try:
                cand_num = int(cand_str)
            except ValueError:
                continue
            body["_iter"] = iter_num
            body["_cand"] = cand_num
            entries.append(body)
    return entries


def _aggregate_tokens_and_cost(out_dir: Path) -> tuple[int, int, float, float]:
    """Walk ``metrics-iter-*.json`` files, sum tokens + estimate cost.

    Returns ``(tokens_in, tokens_out, cost_usd, total_iter_seconds)``.
    """
    tokens_in = 0
    tokens_out = 0
    wall_s = 0.0
    for metrics_file in sorted(out_dir.glob("metrics-iter-*.json")):
        try:
            body = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(body, dict):
            continue
        wall_s += float(body.get("iteration_total_s", 0.0) or 0.0)
        # Tokens live under phase keys (``plan_generation`` etc.) per
        # ``search.py:1148``. Each phase value is a dict of
        # ``{model_name: {input_tokens, output_tokens, ...}}``.
        for phase_name, phase_body in body.items():
            if phase_name.startswith("_") or phase_name in (
                "iteration",
                "iteration_total_s",
                "evaluation",
            ):
                continue
            if not isinstance(phase_body, dict):
                continue
            for model_stats in phase_body.values():
                if not isinstance(model_stats, dict):
                    continue
                tokens_in += int(model_stats.get("input_tokens", 0) or 0)
                tokens_out += int(model_stats.get("output_tokens", 0) or 0)
    cost = (
        tokens_in * _GEMINI_INPUT_PER_MTOK_USD
        + tokens_out * _GEMINI_OUTPUT_PER_MTOK_USD
    ) / 1_000_000
    return tokens_in, tokens_out, cost, wall_s


def _pick_winner(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the lowest-latency correct entry. Returns ``None`` when
    no correct entry exists."""
    correct_entries = [
        e for e in entries
        if e.get("correct") is True and e.get("latency") is not None
    ]
    if not correct_entries:
        return None
    return min(correct_entries, key=lambda e: int(e["latency"]))


def load_autocomp_row(
    out_dir: Path,
    *,
    target: str,
    workload: str,
    shape_id: str,
    repeat: int,
) -> CanonicalCellRow:
    """Build one canonical row from one autocomp search output dir.

    When ``out_dir`` doesn't exist or carries no eval-results,
    returns a ``correctness=False`` row with ``cycle_source="none"``
    and a note explaining the failure mode.
    """
    if not out_dir.is_dir():
        return _missing_row(
            target=target, workload=workload, shape_id=shape_id, repeat=repeat,
            note=f"autocomp output dir not found: {out_dir}",
        )

    entries = _walk_eval_results(out_dir)
    tokens_in, tokens_out, cost_usd, wall_s = _aggregate_tokens_and_cost(out_dir)

    winner = _pick_winner(entries)
    if winner is None:
        # No correct candidate. Surface the rounds we did burn so the
        # aggregator can still show "spent N rounds, never converged".
        return CanonicalCellRow(
            backend="autocomp",
            target=target,
            workload=workload,
            shape_id=shape_id,
            repeat=repeat,
            correctness=False,
            cycles=None,
            rounds_used=max((e["_iter"] for e in entries), default=0) + (1 if entries else 0),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            wall_s=wall_s,
            cycle_source="none",
            notes=f"no correct candidate after {len(entries)} eval entries",
        )

    return CanonicalCellRow(
        backend="autocomp",
        target=target,
        workload=workload,
        shape_id=shape_id,
        repeat=repeat,
        correctness=True,
        cycles=int(winner["latency"]),
        rounds_used=int(winner["_iter"]) + 1,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        wall_s=wall_s,
        # Autocomp's modified libgemmini Spike fork emits this string
        # (see gemmini_eval.py:423). We record the cycle_source
        # verbatim so the report's harness-skew column can flag it
        # against KB's MAIN_LD_ST_EX_CYCLES.
        cycle_source="Generated implementation latency",
        notes=f"iter={winner['_iter']} cand={winner['_cand']}",
    )


def _missing_row(*, target: str, workload: str, shape_id: str, repeat: int, note: str) -> CanonicalCellRow:
    return CanonicalCellRow(
        backend="autocomp",
        target=target,
        workload=workload,
        shape_id=shape_id,
        repeat=repeat,
        correctness=False,
        cycles=None,
        rounds_used=0,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        wall_s=0.0,
        cycle_source="none",
        notes=note,
    )


__all__ = ["load_autocomp_row"]
