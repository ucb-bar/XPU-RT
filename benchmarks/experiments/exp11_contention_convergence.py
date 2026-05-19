"""Experiment 11 — Formal verification of QNN closed-loop contention convergence.

Loads ``xpu-rt/data/profiled/qnn_closed_loop/contention.jsonl`` and proves
(or refutes) five convergence-related obligations with Z3 over the actual
recorded factor values.

Obligations:
  1. Monotone delta decrease across rounds.
  2. Tolerance convergence at round 4 (max |delta| < 0.05).
  3. Factor stability past the converged round (within tolerance 0.05).
  4. Bounded factors (0.5 <= factor <= 2.0 for every (round, backend)).
  5. DSP-faster-under-CPU-contention invariant
     (factor[CPU] > 1.0 -> factor[DSP] < 1.0).

Outputs:
    build/experiments/exp11_contention/{results.jsonl,summary.md}

Usage:
    uv run python scripts/experiments/exp11_contention_convergence.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import z3

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_closed_loop" / "contention.jsonl"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp11_contention"

TOLERANCE = 0.05
LOWER_BOUND = 0.5
UPPER_BOUND = 2.0
CONVERGED_ROUND = 4
Z3_TIMEOUT_MS = 5000


def _load_rounds() -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    with DATA.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rounds.append(json.loads(line))
    rounds.sort(key=lambda r: r["round"])
    return rounds


def _max_abs_delta(row: dict[str, Any]) -> float:
    return max(abs(v) for v in row["last_delta"].values())


def _new_solver() -> z3.Solver:
    s = z3.Solver()
    s.set("timeout", Z3_TIMEOUT_MS)
    return s


# Each obligation is encoded as: build constraints stating the property HOLDS
# for the observed data; if the solver returns `sat`, the property is proved
# on the data. If it cannot satisfy the conjunction, we then re-check by
# searching for a single counterexample row and report it.


def obligation_1_monotone_delta(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """`max |delta[r]| >= max |delta[r+1]|` for r in 1..N-1."""
    violations: list[dict[str, Any]] = []
    for i in range(len(rounds) - 1):
        a = _max_abs_delta(rounds[i])
        b = _max_abs_delta(rounds[i + 1])
        solver = _new_solver()
        # Encode the pair as Z3 reals and ask whether `a >= b` holds.
        ra = z3.Real(f"max_abs_delta_r{rounds[i]['round']}")
        rb = z3.Real(f"max_abs_delta_r{rounds[i + 1]['round']}")
        solver.add(ra == a)
        solver.add(rb == b)
        solver.add(z3.Not(ra >= rb))  # search for counterexample
        if solver.check() == z3.sat:
            violations.append(
                {
                    "from_round": rounds[i]["round"],
                    "to_round": rounds[i + 1]["round"],
                    "from_max_abs_delta": a,
                    "to_max_abs_delta": b,
                    "growth": b - a,
                }
            )
    return {
        "obligation": "monotone_delta_decrease",
        "verdict": "proved" if not violations else "counterexample",
        "violations": violations,
        "n_round_pairs": max(0, len(rounds) - 1),
    }


def obligation_2_tol_at_round_4(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """`max |delta_at_round_4| < TOLERANCE`."""
    target = next((r for r in rounds if r["round"] == CONVERGED_ROUND), None)
    if target is None:
        return {
            "obligation": "tolerance_convergence_round_4",
            "verdict": "n/a",
            "reason": f"no row with round={CONVERGED_ROUND}",
        }
    m = _max_abs_delta(target)
    solver = _new_solver()
    rm = z3.Real("max_abs_delta_r4")
    tol = z3.Real("tol")
    solver.add(rm == m)
    solver.add(tol == TOLERANCE)
    solver.add(z3.Not(rm < tol))  # counterexample search
    sat = solver.check()
    proved = sat == z3.unsat
    return {
        "obligation": "tolerance_convergence_round_4",
        "verdict": "proved" if proved else "counterexample",
        "max_abs_delta_at_round_4": m,
        "tolerance": TOLERANCE,
        "per_backend_delta": target["last_delta"],
    }


def obligation_3_factor_stability(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """`|factor[r, b] - factor[converged_round, b]| < TOLERANCE` for r >= converged."""
    anchor = next((r for r in rounds if r["round"] == CONVERGED_ROUND), None)
    if anchor is None:
        return {
            "obligation": "factor_stability",
            "verdict": "n/a",
            "reason": f"no anchor round={CONVERGED_ROUND}",
        }
    later = [r for r in rounds if r["round"] >= CONVERGED_ROUND]
    backends = sorted(anchor["factors"].keys())
    violations: list[dict[str, Any]] = []
    for row in later:
        for b in backends:
            diff = abs(row["factors"][b] - anchor["factors"][b])
            solver = _new_solver()
            rd = z3.Real(f"diff_r{row['round']}_{b}")
            tol = z3.Real("tol")
            solver.add(rd == diff)
            solver.add(tol == TOLERANCE)
            solver.add(z3.Not(rd < tol))
            if solver.check() == z3.sat:
                violations.append(
                    {
                        "round": row["round"],
                        "backend": b,
                        "diff": diff,
                        "factor": row["factors"][b],
                        "anchor_factor": anchor["factors"][b],
                    }
                )
    # With only round 4 observed, this is vacuously true past convergence
    # (the anchor compares to itself with diff=0). Flag it.
    note: str | None = None
    if len(later) == 1:
        note = (
            "Only the converged round itself is observed; no post-convergence "
            "rounds exist. The obligation is vacuously true on the data but "
            "does not provide empirical evidence of stability."
        )
    return {
        "obligation": "factor_stability",
        "verdict": "proved" if not violations else "counterexample",
        "violations": violations,
        "tolerance": TOLERANCE,
        "converged_round": CONVERGED_ROUND,
        "post_convergence_rounds_observed": [r["round"] for r in later],
        "note": note,
    }


def obligation_4_bounded_factors(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """`LOWER_BOUND <= factor[r, b] <= UPPER_BOUND`."""
    violations: list[dict[str, Any]] = []
    for row in rounds:
        for b, val in row["factors"].items():
            solver = _new_solver()
            rv = z3.Real(f"factor_r{row['round']}_{b}")
            lo = z3.Real("lo")
            hi = z3.Real("hi")
            solver.add(rv == val)
            solver.add(lo == LOWER_BOUND)
            solver.add(hi == UPPER_BOUND)
            solver.add(z3.Not(z3.And(rv >= lo, rv <= hi)))
            if solver.check() == z3.sat:
                violations.append(
                    {
                        "round": row["round"],
                        "backend": b,
                        "factor": val,
                        "lower_bound": LOWER_BOUND,
                        "upper_bound": UPPER_BOUND,
                    }
                )
    return {
        "obligation": "bounded_factors",
        "verdict": "proved" if not violations else "counterexample",
        "violations": violations,
        "lower_bound": LOWER_BOUND,
        "upper_bound": UPPER_BOUND,
        "n_checks": sum(len(r["factors"]) for r in rounds),
    }


def obligation_5_dsp_speedup_under_cpu_contention(
    rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    """`factor[CPU] > 1.0 -> factor[DSP] < 1.0` for every round."""
    violations: list[dict[str, Any]] = []
    checked: list[int] = []
    for row in rounds:
        if "CPU" not in row["factors"] or "DSP" not in row["factors"]:
            continue
        checked.append(row["round"])
        cpu = row["factors"]["CPU"]
        dsp = row["factors"]["DSP"]
        solver = _new_solver()
        rcpu = z3.Real(f"cpu_r{row['round']}")
        rdsp = z3.Real(f"dsp_r{row['round']}")
        solver.add(rcpu == cpu)
        solver.add(rdsp == dsp)
        # Counterexample: CPU > 1.0 AND NOT(DSP < 1.0)
        solver.add(rcpu > 1.0)
        solver.add(rdsp >= 1.0)
        if solver.check() == z3.sat:
            violations.append(
                {
                    "round": row["round"],
                    "cpu_factor": cpu,
                    "dsp_factor": dsp,
                }
            )
    return {
        "obligation": "dsp_speedup_under_cpu_contention",
        "verdict": "proved" if not violations else "counterexample",
        "violations": violations,
        "rounds_checked": checked,
    }


OBLIGATIONS = [
    obligation_1_monotone_delta,
    obligation_2_tol_at_round_4,
    obligation_3_factor_stability,
    obligation_4_bounded_factors,
    obligation_5_dsp_speedup_under_cpu_contention,
]


def _emit_summary(
    rounds: list[dict[str, Any]],
    results: list[dict[str, Any]],
    elapsed: float,
    out_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Experiment 11 — QNN closed-loop contention convergence (Z3)\n")
    lines.append(
        f"Source: `xpu-rt/data/profiled/qnn_closed_loop/contention.jsonl` "
        f"({len(rounds)} rounds).\n"
    )
    lines.append(f"Z3 timeout per obligation: {Z3_TIMEOUT_MS} ms. Total runtime: {elapsed:.2f}s.\n")

    lines.append("## Per-round summary\n")
    lines.append("| round | CPU factor | DSP factor | max\\|delta\\| |")
    lines.append("|---:|---:|---:|---:|")
    for row in rounds:
        lines.append(
            f"| {row['round']} | {row['factors'].get('CPU', float('nan')):.6f} | "
            f"{row['factors'].get('DSP', float('nan')):.6f} | "
            f"{_max_abs_delta(row):.6f} |"
        )
    lines.append("")

    lines.append("## Obligations\n")
    lines.append("| # | obligation | verdict |")
    lines.append("|---:|---|---|")
    for idx, r in enumerate(results, start=1):
        lines.append(f"| {idx} | `{r['obligation']}` | **{r['verdict']}** |")
    lines.append("")

    for idx, r in enumerate(results, start=1):
        lines.append(f"### {idx}. `{r['obligation']}` — {r['verdict']}\n")
        if r["verdict"] == "counterexample":
            lines.append("Counterexamples / violating rows:")
            for v in r.get("violations", []):
                lines.append(f"- `{json.dumps(v, default=str)}`")
            lines.append("")
        else:
            for k, v in r.items():
                if k in ("obligation", "verdict", "violations"):
                    continue
                lines.append(f"- `{k}`: `{json.dumps(v, default=str)}`")
            lines.append("")

    out_dir.joinpath("summary.md").write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds = _load_rounds()
    if not rounds:
        print(f"[exp11] no rows in {DATA}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for fn in OBLIGATIONS:
        results.append(fn(rounds))
    elapsed = time.perf_counter() - t0

    with OUT_DIR.joinpath("results.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")

    _emit_summary(rounds, results, elapsed, OUT_DIR)

    verdicts = {r["obligation"]: r["verdict"] for r in results}
    print(f"[exp11] runtime={elapsed:.2f}s")
    for name, v in verdicts.items():
        print(f"[exp11] {name}: {v}")
    print(f"[exp11] results -> {OUT_DIR / 'results.jsonl'}")
    print(f"[exp11] summary -> {OUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
