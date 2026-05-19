"""Experiment 3 — Z3/SMT obligation coverage vs deterministic ladder.

Walks the fault-injection corpus, runs each case through the
deterministic verification ladder (emulated probe) and through the
extended Z3 obligation harness, and writes a 2-D coverage table.

Usage:
    uv run python scripts/experiments/exp3_smt_coverage.py [--quick]

Outputs:
    build/experiments/exp3/results.jsonl
    build/experiments/exp3/summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))

from xpu_rt.audit.fault_injection_corpus import (  # noqa: E402
    FaultCase,
    build_corpus,
    run_case_z3,
)
from xpu_rt.solve.solver_types import (  # noqa: E402
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverStatus,
)

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp3"


def _probe() -> BackendProbeResult:
    return BackendProbeResult(
        backend=SolverBackendName.Z3,
        availability=BackendAvailabilityStatus.AVAILABLE,
        version="exp3",
    )


def _run_one(case: FaultCase, probe: BackendProbeResult) -> dict[str, object]:
    t0 = time.perf_counter()
    det_caught = case.run_deterministic_check()
    det_ms = (time.perf_counter() - t0) * 1000.0

    response = run_case_z3(case, probe=probe)
    z3_caught = response.status is SolverStatus.SAT_COUNTEREXAMPLE
    return {
        "name": case.name,
        "obligation_kind": case.obligation_kind,
        "is_clean": case.is_clean,
        "is_fault": case.expected_z3_status is SolverStatus.SAT_COUNTEREXAMPLE,
        "deterministic_caught": det_caught,
        "deterministic_check_ms": round(det_ms, 3),
        "z3_status": response.status.value,
        "z3_solve_time_ms": round(response.time_ms, 3),
        "z3_caught": z3_caught,
        "z3_expected_status": case.expected_z3_status.value,
        "z3_status_matches_expected": response.status is case.expected_z3_status,
        "deterministic_expected": case.expected_deterministic_caught,
        "deterministic_matches_expected": det_caught == case.expected_deterministic_caught,
    }


def _write_summary(results: list[dict[str, object]], path: Path) -> None:
    fault_rows = [r for r in results if r["z3_expected_status"] == SolverStatus.SAT_COUNTEREXAMPLE.value]
    n_faults = len(fault_rows)
    z3_only = sum(
        1 for r in fault_rows
        if r["z3_caught"] and not r["deterministic_caught"]
    )
    det_only = sum(
        1 for r in fault_rows
        if r["deterministic_caught"] and not r["z3_caught"]
    )
    both = sum(
        1 for r in fault_rows
        if r["deterministic_caught"] and r["z3_caught"]
    )
    neither = sum(
        1 for r in fault_rows
        if not r["deterministic_caught"] and not r["z3_caught"]
    )

    by_kind: dict[str, list[float]] = defaultdict(list)
    timeouts_by_kind: dict[str, int] = defaultdict(int)
    for r in results:
        by_kind[r["obligation_kind"]].append(float(r["z3_solve_time_ms"]))
        if r["z3_status"] == SolverStatus.TIMEOUT.value:
            timeouts_by_kind[r["obligation_kind"]] += 1

    lines: list[str] = []
    lines.append("# Experiment 3 — Z3 obligation coverage\n")
    lines.append(f"Total cases: **{len(results)}**  ")
    lines.append(f"Fault cases: **{n_faults}**\n")
    lines.append(
        f"## Headline: Z3 catches **{z3_only} / {n_faults}** faults the deterministic ladder misses.\n"
    )
    lines.append("| Subset | Count |")
    lines.append("|---|---|")
    lines.append(f"| Z3 only (silent in ladder) | {z3_only} |")
    lines.append(f"| Deterministic only (Z3 missed) | {det_only} |")
    lines.append(f"| Both caught | {both} |")
    lines.append(f"| Neither caught | {neither} |\n")

    lines.append("## Per-obligation solve cost\n")
    lines.append("| obligation_kind | n | mean ms | median ms | max ms | timeouts |")
    lines.append("|---|---|---|---|---|---|")
    for kind in sorted(by_kind):
        ts = by_kind[kind]
        lines.append(
            f"| {kind} | {len(ts)} | {statistics.mean(ts):.2f} | "
            f"{statistics.median(ts):.2f} | {max(ts):.2f} | {timeouts_by_kind[kind]} |"
        )
    lines.append("")

    lines.append("## Case table\n")
    lines.append("| case | obligation | det_caught | z3_status | z3_ms | z3_caught |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['obligation_kind']} | "
            f"{'Y' if r['deterministic_caught'] else '.'} | {r['z3_status']} | "
            f"{r['z3_solve_time_ms']:.2f} | {'Y' if r['z3_caught'] else '.'} |"
        )
    lines.append("")

    mismatches = [
        r for r in results
        if not r["z3_status_matches_expected"] or not r["deterministic_matches_expected"]
    ]
    if mismatches:
        lines.append("## Harness-expectation mismatches (investigate)\n")
        for r in mismatches:
            lines.append(
                f"- `{r['name']}`: z3 expected={r['z3_expected_status']} got={r['z3_status']}, "
                f"det expected={r['deterministic_expected']} got={r['deterministic_caught']}"
            )
        lines.append("")
    else:
        lines.append("All cases matched expected status. No harness bugs detected.\n")

    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the corpus (currently the only mode).",
    )
    args = parser.parse_args()
    _ = args.quick  # no large-mode yet — the corpus is the workload.

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.jsonl"
    summary_path = OUT_DIR / "summary.md"

    probe = _probe()
    corpus = build_corpus()
    results: list[dict[str, object]] = []

    with results_path.open("w") as fp:
        for case in corpus:
            row = _run_one(case, probe)
            fp.write(json.dumps(row) + "\n")
            results.append(row)

    _write_summary(results, summary_path)
    print(f"wrote {len(results)} rows -> {results_path}")
    print(f"wrote summary -> {summary_path}")

    n_faults = sum(1 for r in results if r["z3_expected_status"] == "sat_counterexample")
    z3_only = sum(
        1 for r in results
        if r["z3_expected_status"] == "sat_counterexample"
        and r["z3_caught"] and not r["deterministic_caught"]
    )
    print(f"headline: Z3 catches {z3_only}/{n_faults} faults the deterministic ladder misses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
