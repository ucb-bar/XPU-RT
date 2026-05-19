"""Quick summary of a matrix run's per-cell samples.jsonl.

Reads ``<out_dir>/per_cell/*/samples.jsonl`` and prints:
  * total samples + correctness rate
  * per-cell correctness rate + median cycles (when correct)
  * cost spent / wall time

Use this between matrix runs to decide whether to expand from N=1
to N=3.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow, read_jsonl


def summarise(out_dir: Path) -> dict[str, Any]:
    cells_root = out_dir / "per_cell"
    by_cell: dict[tuple[str, str, str], list[CanonicalCellRow]] = defaultdict(list)
    total_cost = 0.0
    total_wall = 0.0

    for cell_dir in sorted(cells_root.iterdir()) if cells_root.is_dir() else ():
        rows = read_jsonl(cell_dir / "samples.jsonl")
        for r in rows:
            key = (r.backend, r.target, r.workload)
            by_cell[key].append(r)
            total_cost += r.cost_usd
            total_wall += r.wall_s

    total_samples = sum(len(rows) for rows in by_cell.values())
    total_correct = sum(
        1 for rows in by_cell.values() for r in rows if r.correctness
    )

    per_cell_summary: list[dict[str, Any]] = []
    for (backend, target, workload), rows in sorted(by_cell.items()):
        correct = [r for r in rows if r.correctness and r.cycles]
        cycles = sorted(r.cycles for r in correct if r.cycles is not None)
        per_cell_summary.append({
            "backend": backend,
            "target": target,
            "workload": workload,
            "n": len(rows),
            "n_correct": len(correct),
            "correctness_rate": len(correct) / len(rows) if rows else 0.0,
            "median_cycles": float(statistics.median(cycles)) if cycles else None,
            "min_cycles": int(cycles[0]) if cycles else None,
            "max_cycles": int(cycles[-1]) if cycles else None,
            "cost_usd": sum(r.cost_usd for r in rows),
            "wall_s": sum(r.wall_s for r in rows),
        })

    return {
        "out_dir": str(out_dir),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_correctness_rate": total_correct / total_samples if total_samples else 0.0,
        "total_cost_usd": total_cost,
        "total_wall_s": total_wall,
        "per_cell": per_cell_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    s = summarise(args.out_dir)
    print(f"== Matrix summary: {s['out_dir']} ==")
    print(f"  samples:      {s['total_samples']}")
    print(f"  correct:      {s['total_correct']}  ({s['overall_correctness_rate']:.0%})")
    print(f"  cost_usd:     ${s['total_cost_usd']:.4f}")
    print(f"  wall_s:       {s['total_wall_s']:.0f}s  ({s['total_wall_s']/60:.1f}m)")
    print(f"\n  per-cell (sorted by correctness desc):")
    print(f"  {'backend':<12} {'target':<18} {'workload':<22} {'n':>3} {'corr':>5} {'med_cyc':>10} {'min':>8} {'max':>8} {'cost':>9}")
    rows = sorted(s["per_cell"], key=lambda c: (-c["correctness_rate"], c["target"]))
    for c in rows:
        cyc = f"{c['median_cycles']:,.0f}" if c["median_cycles"] else "—"
        mn = f"{c['min_cycles']:,}" if c["min_cycles"] else "—"
        mx = f"{c['max_cycles']:,}" if c["max_cycles"] else "—"
        print(
            f"  {c['backend']:<12} {c['target']:<18} {c['workload']:<22} "
            f"{c['n']:>3} {c['n_correct']}/{c['n']:<3} {cyc:>10} {mn:>8} {mx:>8} ${c['cost_usd']:>7.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "summarise"]
