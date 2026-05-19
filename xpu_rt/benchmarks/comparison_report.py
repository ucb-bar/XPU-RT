"""Aggregate KB-vanilla + XPU-RT JSONL rows into a head-to-head report.

Reads ``<study_dir>/kb_vanilla.jsonl`` and ``<study_dir>/xpu_rt.jsonl``,
joins on ``contract.region_id``, and emits both a JSON snapshot
(``report.json``) and a human-readable markdown summary
(``report.md``) under the same directory.

Metrics surfaced per row:

* KB-vanilla side: ``compile``, ``intrinsic_use_rate``,
  ``shape_consistency``, ``rounds``, ``tokens_in/out``, ``cost_usd``,
  ``wall_s``.
* XPU-RT side: ``correct``, ``cycles``, ``speedup``, ``rounds``,
  ``tokens_in/out``, ``cost_usd``, ``wall_s``.

Aggregate totals at the bottom: per-backend cost, mean wall, and
contract-level "who won" tallies (compile-rate vs correctness-rate are
not directly comparable across backends — see ``out_of_scope`` in
plan 2 — so the markdown surfaces both side by side and lets the
reader judge).
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.observability import gemini_usage


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _join_by_region(
    kb_rows: list[dict[str, Any]],
    xr_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Return paired rows keyed by ``contract.region_id``."""
    kb_by_id = {r["contract"]["region_id"]: r for r in kb_rows}
    xr_by_id = {r["contract"]["region_id"]: r for r in xr_rows}
    ids = list(kb_by_id.keys())
    for rid in xr_by_id:
        if rid not in kb_by_id:
            ids.append(rid)
    return [(kb_by_id.get(rid), xr_by_id.get(rid)) for rid in ids]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class BackendStats:
    backend: str
    rows: int = 0
    compile_rate: float | None = None
    intrinsic_use_rate_mean: float | None = None
    intrinsic_use_rate_max: float | None = None
    shape_consistency_rate: float | None = None
    correctness_rate: float | None = None
    cycles_seen: int = 0
    cycles_geomean: float | None = None
    speedup_max: float | None = None
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_wall_s: float = 0.0
    rounds_mean: float | None = None


def _safe_mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _safe_geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x > 0]
    if not xs:
        return None
    return statistics.geometric_mean(xs)


def _stats_kb_vanilla(rows: list[dict[str, Any]]) -> BackendStats:
    s = BackendStats(backend="kb_vanilla", rows=len(rows))
    if not rows:
        return s
    s.compile_rate = sum(1 for r in rows if r.get("compile")) / len(rows)
    intrs = [float(r.get("intrinsic_use_rate", 0.0)) for r in rows]
    s.intrinsic_use_rate_mean = _safe_mean(intrs)
    s.intrinsic_use_rate_max = max(intrs) if intrs else None
    s.shape_consistency_rate = sum(
        1 for r in rows if r.get("shape_consistency")
    ) / len(rows)
    s.total_cost_usd = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    s.total_tokens_in = sum(int(r.get("tokens_in", 0)) for r in rows)
    s.total_tokens_out = sum(int(r.get("tokens_out", 0)) for r in rows)
    s.total_wall_s = sum(float(r.get("wall_s", 0.0)) for r in rows)
    s.rounds_mean = _safe_mean([int(r.get("rounds", 0)) for r in rows])
    return s


def _stats_xpu_rt(rows: list[dict[str, Any]]) -> BackendStats:
    s = BackendStats(backend="xpu_rt_kb_v2", rows=len(rows))
    if not rows:
        return s
    correct = [r for r in rows if r.get("correct")]
    s.correctness_rate = len(correct) / len(rows)
    cycles = [float(r["cycles"]) for r in correct if r.get("cycles")]
    s.cycles_seen = len(cycles)
    s.cycles_geomean = _safe_geomean(cycles)
    speedups = [float(r.get("speedup", 0.0)) for r in correct if r.get("speedup")]
    s.speedup_max = max(speedups) if speedups else None
    s.total_cost_usd = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    s.total_tokens_in = sum(int(r.get("tokens_in", 0)) for r in rows)
    s.total_tokens_out = sum(int(r.get("tokens_out", 0)) for r in rows)
    s.total_wall_s = sum(float(r.get("wall_s", 0.0)) for r in rows)
    s.rounds_mean = _safe_mean([int(r.get("rounds", 0)) for r in rows])
    return s


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    return f"{x*100:5.1f}%" if x is not None else "  —"


def _fmt_money(x: float | None) -> str:
    return f"${x:.4f}" if x is not None else "  —"


def _fmt_num(x: float | None, fmt: str = "{:.2f}") -> str:
    return fmt.format(x) if x is not None else "—"


def _fmt_int(x: int | None) -> str:
    return f"{x}" if x is not None else "—"


def _fmt_shape(contract: dict[str, Any]) -> str:
    A, B = contract["input_shapes"]
    return f"{A}×{B}"


def render_markdown(
    *,
    study_dir: Path,
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    kb_stats: BackendStats,
    xr_stats: BackendStats,
    pre_spend: dict[str, float],
    post_spend: dict[str, float],
) -> str:
    out: list[str] = []
    out.append("# SmolVLA-on-Gemmini comparison report")
    out.append("")
    out.append(f"Study dir: `{study_dir}`")
    out.append(
        f"Gemini cumulative spend before run: **${pre_spend['cumulative_usd']:.4f}** "
        f"({pre_spend['calls']} calls); after run: **${post_spend['cumulative_usd']:.4f}** "
        f"({post_spend['calls']} calls); "
        f"incremental: **${post_spend['cumulative_usd'] - pre_spend['cumulative_usd']:.4f}**."
    )
    out.append("")
    out.append("## Aggregate")
    out.append("")
    out.append("|              | KB-vanilla (prompt-injection) | XPU-RT / KB v2 (real eval) |")
    out.append("|--------------|-------------------------------|----------------------------|")
    out.append(
        f"| rows         | {kb_stats.rows:>5}                       | {xr_stats.rows:>5}                    |"
    )
    out.append(
        f"| compile rate | {_fmt_pct(kb_stats.compile_rate)}                       |        n/a                 |"
    )
    out.append(
        f"| intrinsic-use (mean) | {_fmt_pct(kb_stats.intrinsic_use_rate_mean)}               |        n/a                 |"
    )
    out.append(
        f"| intrinsic-use (max)  | {_fmt_pct(kb_stats.intrinsic_use_rate_max)}                |        n/a                 |"
    )
    out.append(
        f"| shape consistency | {_fmt_pct(kb_stats.shape_consistency_rate)}                  |        n/a                 |"
    )
    out.append(
        f"| correctness rate |        n/a                       | {_fmt_pct(xr_stats.correctness_rate)}                |"
    )
    out.append(
        f"| cycles (geo-mean correct)|        n/a               | {_fmt_num(xr_stats.cycles_geomean, '{:>10.0f}')}               |"
    )
    out.append(
        f"| best speedup |        n/a                       | {_fmt_num(xr_stats.speedup_max)}                       |"
    )
    out.append(
        f"| total $      | {_fmt_money(kb_stats.total_cost_usd)}                       | {_fmt_money(xr_stats.total_cost_usd)}                      |"
    )
    out.append(
        f"| total tokens in / out | {kb_stats.total_tokens_in} / {kb_stats.total_tokens_out}     | {xr_stats.total_tokens_in} / {xr_stats.total_tokens_out}        |"
    )
    out.append(
        f"| total wall (s) | {kb_stats.total_wall_s:>6.1f}                       | {xr_stats.total_wall_s:>6.1f}                       |"
    )
    out.append(
        f"| mean rounds  | {_fmt_num(kb_stats.rounds_mean)}                       | {_fmt_num(xr_stats.rounds_mean)}                       |"
    )

    out.append("")
    out.append("## Per-contract rows")
    out.append("")
    out.append(
        "| region_id (truncated) | shape | KB compile | KB intr-use | KB cost | XR correct | XR cycles | XR cost |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for kb, xr in pairs:
        region = (kb or xr or {}).get("contract", {}).get("region_id", "?")
        shape = _fmt_shape((kb or xr or {}).get("contract", {"input_shapes": [["?"], ["?"]]}))
        rid = region.split(".")[-2] + "." + region.split(".")[-1] if "." in region else region
        kb_compile = "✓" if kb and kb.get("compile") else "✗" if kb else "—"
        kb_intr = _fmt_pct(kb.get("intrinsic_use_rate") if kb else None)
        kb_cost = _fmt_money(kb.get("cost_usd") if kb else None)
        xr_correct = "✓" if xr and xr.get("correct") else "✗" if xr else "—"
        xr_cycles = _fmt_int(int(xr["cycles"]) if xr and xr.get("cycles") else None)
        xr_cost = _fmt_money(xr.get("cost_usd") if xr else None)
        out.append(f"| `{rid}` | {shape} | {kb_compile} | {kb_intr} | {kb_cost} | {xr_correct} | {xr_cycles} | {xr_cost} |")

    out.append("")
    out.append("## Notes")
    out.append("")
    out.append(
        "- **KB-vanilla** runs a 4-strategy round-robin Gemini prompt with the "
        "Gemmini Target Card injected as the 'optimization database'. Scoring "
        "is purely static (compile rate, intrinsic-name match against the card, "
        "shape literals present). No evaluator → no cycles or correctness."
    )
    out.append(
        "- **XPU-RT / KB v2** runs the contract-driven agent loop with the real "
        "Spike+gemmini evaluator. Cycles come from "
        "`MAIN_LD_ST_EX_CYCLES` counter via `spike --extension=gemmini`."
    )
    out.append(
        "- A 0% `intrinsic_use_rate` does **not** mean the LLM wrote zero "
        "`gemmini_*` calls — it means none of the calls match an entry in "
        "the Target Card by name. If the card under-covers primitives "
        "(e.g. omits `gemmini_mvin`/`gemmini_mvout` in favour of high-level "
        "`tiled_*` helpers), expect this rate to underestimate."
    )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate KB-vanilla + XPU-RT JSONL into a comparison report.")
    parser.add_argument("study_dir", type=Path)
    args = parser.parse_args(argv)

    kb_rows = _read_jsonl(args.study_dir / "kb_vanilla.jsonl")
    xr_rows = _read_jsonl(args.study_dir / "xpu_rt.jsonl")
    pairs = _join_by_region(kb_rows, xr_rows)

    kb_stats = _stats_kb_vanilla(kb_rows)
    xr_stats = _stats_xpu_rt(xr_rows)

    run_summary_path = args.study_dir / "run_summary.json"
    pre_post = {"pre_spend": {}, "post_spend": {}}
    if run_summary_path.exists():
        body = json.loads(run_summary_path.read_text())
        pre_post["pre_spend"] = body.get("pre_spend", {})
        pre_post["post_spend"] = body.get("post_spend", {})

    md = render_markdown(
        study_dir=args.study_dir,
        pairs=pairs,
        kb_stats=kb_stats,
        xr_stats=xr_stats,
        pre_spend=pre_post["pre_spend"] or {"cumulative_usd": 0.0, "calls": 0},
        post_spend=pre_post["post_spend"] or {"cumulative_usd": 0.0, "calls": 0},
    )
    md_path = args.study_dir / "report.md"
    md_path.write_text(md)

    json_path = args.study_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "study_dir": str(args.study_dir),
                "kb_vanilla": kb_stats.__dict__,
                "xpu_rt": xr_stats.__dict__,
                "pairs": len(pairs),
                "pre_spend": pre_post["pre_spend"],
                "post_spend": pre_post["post_spend"],
            },
            indent=2,
        )
    )
    print(md)
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
