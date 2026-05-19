"""Experiment 10 — Full QNN profiled cost-matrix audit.

Loads ``xpu-rt/data/profiled/qnn_cost_matrix.json`` (yolov8n and dronet
per-op CPU/GPU/DSP latencies) and runs five audits:

1. Cross-backend ordering stability (flip rate) — per workload.
2. Op-family ordering consistency (index-vs-cost rank correlation and
   inter-family backend specialty).
3. Cross-workload op consistency — does the same op family prefer the
   same backend across yolov8n and dronet?
4. Pathological cost ratios — flag ops with ``max/min > 100x``.
5. SMT formal pass — invoked only where the data fits the
   ``prove_cost_monotonicity`` (m, n, k) form. The per-op matrix does
   NOT embed shape, so this section is documented as not-applicable and
   intentionally skipped (mirroring Test #1's pragma).

Outputs:
    build/experiments/exp10_full_audit/results.jsonl
    build/experiments/exp10_full_audit/summary.md

Runtime: under 2 minutes (yolov8n has 271 rows with all 3 backends; the
``O(n^2)`` flip-rate pass is the dominant cost at ~36k pairs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import structlog

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))

from xpu_rt.audit.cost_table_audit import (  # noqa: E402
    Finding,
    cross_backend_flip_rate,
    cross_workload_consistency,
    family_backend_specialty,
    finding_to_dict,
    index_correlation,
    load_cost_matrix,
    pathological_ratios,
)

logger = structlog.get_logger(__name__)

COST_MATRIX = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp10_full_audit"
BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")


def _severity_for_flip_rate(rate: float, counted: int) -> str:
    """Severity ladder used for cross-backend ordering-stability findings.

    The well-calibrated benchmark in the task description is < 10%
    flips. Random would be 50%. > 30% on a backend pair is suspicious
    enough that a scheduler can't naively transfer orderings across it.
    """

    if counted < 10:
        return "low"
    if rate > 0.50:
        return "high"
    if rate > 0.30:
        return "medium"
    if rate > 0.10:
        return "low"
    return "low"


def run_audit() -> tuple[list[Finding], dict[str, Any]]:
    """Run all five audits over the per-op cost matrix."""

    matrices = load_cost_matrix(COST_MATRIX)
    findings: list[Finding] = []
    stats: dict[str, Any] = {
        "workloads": {},
        "cross_workload": {},
        "pathological": {},
    }

    for wl, rows in matrices.items():
        n_all = sum(1 for r in rows if r.has_all(BACKENDS))
        # 1. Cross-backend flip rate.
        flip_stats = cross_backend_flip_rate(rows, backends=BACKENDS)
        for pair in flip_stats:
            sev = _severity_for_flip_rate(
                pair["flip_rate"], pair["pairs_examined"] - pair["ties"]
            )
            findings.append(
                Finding(
                    check_kind="cross_backend_flip_rate",
                    severity=sev,
                    op_or_family=f"{wl}|{pair['backend_a']}_vs_{pair['backend_b']}",
                    backends=[pair["backend_a"], pair["backend_b"]],
                    detail={"workload": wl, **pair},
                )
            )

        # 2. Op-family specialty (per-backend argmin) and index-rank correlation.
        specialty = family_backend_specialty(rows, backends=BACKENDS)
        rank_corr = index_correlation(rows, backends=BACKENDS)
        # Index-vs-cost rank correlation findings (low severity — informational).
        for fam, per_be in rank_corr.items():
            for be, info in per_be.items():
                # |rho| > 0.7 with n >= 5 is a noteworthy structure-by-index trend.
                if info["n"] >= 5 and abs(info["rho"]) > 0.7:
                    findings.append(
                        Finding(
                            check_kind="family_index_rank_correlation",
                            severity="low",
                            op_or_family=f"{wl}|{fam}",
                            backends=[be],
                            detail={"workload": wl, "family": fam, "backend": be, **info},
                        )
                    )

        # 4. Pathological ratios — top 5.
        path = pathological_ratios(rows, backends=BACKENDS, top_n=5)
        for entry in path:
            ratio = entry["ratio"]
            if ratio > 1000.0:
                sev = "high"
            elif ratio > 100.0:
                sev = "medium"
            else:
                sev = "low"
            findings.append(
                Finding(
                    check_kind="pathological_cost_ratio",
                    severity=sev,
                    op_or_family=f"{wl}|{entry['family']}",
                    backends=list(entry["costs"].keys()),
                    detail=entry,
                )
            )

        stats["workloads"][wl] = {
            "n_ops": len(rows),
            "n_ops_with_all_backends": n_all,
            "flip_rate": flip_stats,
            "family_specialty": specialty,
            "index_rank_correlation": rank_corr,
            "top_pathological": path,
        }
        stats["pathological"][wl] = path

    # 3. Cross-workload consistency on canonical families.
    cw = cross_workload_consistency(matrices, backends=BACKENDS)
    stats["cross_workload"] = cw
    for fam_canon, per_wl in cw.items():
        # Compare fastest backend across workloads.
        fastest_set = {info["fastest"] for info in per_wl.values()}
        agree = len(fastest_set) == 1
        if not agree:
            findings.append(
                Finding(
                    check_kind="cross_workload_specialty_mismatch",
                    severity="medium",
                    op_or_family=fam_canon,
                    backends=sorted({b for info in per_wl.values() for b in info["argmin"].keys()}),
                    detail={
                        "canonical_family": fam_canon,
                        "per_workload": per_wl,
                    },
                )
            )
        else:
            findings.append(
                Finding(
                    check_kind="cross_workload_specialty_agree",
                    severity="low",
                    op_or_family=fam_canon,
                    backends=list(fastest_set),
                    detail={
                        "canonical_family": fam_canon,
                        "per_workload": per_wl,
                    },
                )
            )

    # 5. SMT formal pass — documented-skip. The per-op matrix has no
    # (m, n, k) shape parameters, so ``prove_cost_monotonicity`` cannot
    # be applied. Mirror Test #1's pragma and record as an informational
    # finding rather than silently skipping.
    findings.append(
        Finding(
            check_kind="z3_shape_monotonicity_not_applicable",
            severity="low",
            op_or_family="<matrix-wide>",
            backends=[],
            detail={
                "reason": (
                    "Per-op profile matrix has no embedded (m, n, k) shape; "
                    "prove_cost_monotonicity requires a shape-parameterized "
                    "cost expression. Honest skip — see Test #1 pragma."
                ),
            },
        )
    )

    return findings, stats


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _emit_summary(
    findings: list[Finding], stats: dict[str, Any], elapsed_s: float, out_dir: Path
) -> None:
    by_kind = Counter(f.check_kind for f in findings)
    by_sev = Counter(f.severity for f in findings)

    lines: list[str] = []
    lines.append("# Experiment 10 — Full QNN Cost-Matrix Audit\n")
    lines.append(f"- Runtime: {elapsed_s:.2f}s")
    lines.append("- Source: `xpu-rt/data/profiled/qnn_cost_matrix.json`")
    lines.append(f"- Backends audited: {list(BACKENDS)}\n")

    lines.append("## Data preamble\n")
    for wl, info in stats["workloads"].items():
        lines.append(
            f"- **{wl}**: {info['n_ops']} ops total, "
            f"{info['n_ops_with_all_backends']} with measurements on all 3 backends."
        )
    lines.append("")
    lines.append(
        "yolov8n uses QNN-normalized op IDs (`convolution_0`, `elementwiseneuron_5`, "
        "`pad_3`); dronet uses ONNX node paths (`/conv_modules.0/Conv`, `/Add_1`, "
        "`/relu_modules.0/Relu`). Op families are extracted from both flavors and "
        "ONNX names are folded onto QNN families (`Conv`->`convolution`, "
        "`Relu`/`Sigmoid`->`elementwiseneuron`, `Add`->`elementwise_sum`, "
        "`MaxPool`->`pool`) for cross-workload comparison.\n"
    )

    lines.append("## Headline counts\n")
    lines.append("| severity | count |")
    lines.append("|---|---|")
    for sev in ("high", "medium", "low"):
        lines.append(f"| {sev} | {by_sev.get(sev, 0)} |")
    lines.append("")
    lines.append("| check_kind | count |")
    lines.append("|---|---|")
    for kind, n in by_kind.most_common():
        lines.append(f"| {kind} | {n} |")
    lines.append("")

    # 1. Cross-backend flip rate.
    lines.append("## 1. Cross-backend ordering stability (flip rate)\n")
    lines.append(
        "For every op-pair (A, B) with measurements on both backends in a pair, "
        "we check whether the cost ordering on backend X agrees with backend Y. "
        "A flip rate near 50% means orderings are uncorrelated; <10% would mean "
        "a scheduler can safely transfer orderings; the boundary in between is "
        "the bulk of real-hardware behavior.\n"
    )
    lines.append("| workload | backend pair | ops compared | pairs | flips | ties | flip rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for wl, info in stats["workloads"].items():
        for p in info["flip_rate"]:
            lines.append(
                f"| {wl} | {p['backend_a']} vs {p['backend_b']} | {p['ops_compared']} | "
                f"{p['pairs_examined']} | {p['flips']} | {p['ties']} | {_fmt_pct(p['flip_rate'])} |"
            )
    lines.append("")

    # 2. Backend specialty per family.
    lines.append("## 2. Op-family backend specialty (argmin histogram)\n")
    lines.append("For each family, count which backend is fastest per op (n = ops with all 3 backends).\n")
    for wl, info in stats["workloads"].items():
        lines.append(f"### {wl}\n")
        lines.append("| family | n | CPU win | GPU win | DSP win | fastest overall |")
        lines.append("|---|---|---|---|---|---|")
        # Sort families by descending n.
        items = sorted(info["family_specialty"].items(), key=lambda kv: -kv[1]["n"])
        for fam, fam_info in items:
            argm = fam_info["argmin"]
            lines.append(
                f"| `{fam}` | {fam_info['n']} | {argm.get('CPU', 0)} | "
                f"{argm.get('GPU', 0)} | {argm.get('DSP', 0)} | **{fam_info['fastest']}** |"
            )
        lines.append("")

    # 2b. Index-rank correlation.
    lines.append("### Index-vs-cost rank correlation (Spearman rho)\n")
    lines.append(
        "Does the op-suffix index (`convolution_0`, `convolution_1`, ...) correlate "
        "with cost on each backend? Listed only where |rho| > 0.5 with n >= 5 — most "
        "families show no monotone trend, which is expected since suffix index "
        "tracks graph topology, not shape.\n"
    )
    lines.append("| workload | family | backend | n | rho |")
    lines.append("|---|---|---|---|---|")
    for wl, info in stats["workloads"].items():
        for fam, per_be in sorted(info["index_rank_correlation"].items()):
            for be, rho_info in sorted(per_be.items()):
                if rho_info["n"] >= 5 and abs(rho_info["rho"]) > 0.5:
                    lines.append(
                        f"| {wl} | `{fam}` | {be} | {rho_info['n']} | {rho_info['rho']:+.3f} |"
                    )
    lines.append("")

    # 3. Cross-workload consistency.
    lines.append("## 3. Cross-workload op consistency\n")
    lines.append(
        "Mapping ONNX op names to QNN families: `Conv`->`convolution`, "
        "`Relu`/`Sigmoid`->`elementwiseneuron`, `Add`->`elementwise_sum`, "
        "`MaxPool`->`pool`, `Gemm`->`fullyconnected`. Only families appearing "
        "in both workloads are shown.\n"
    )
    lines.append("| canonical family | yolov8n n / argmin / fastest | dronet n / argmin / fastest | agree? |")
    lines.append("|---|---|---|---|")
    for fam, per_wl in sorted(stats["cross_workload"].items()):
        y = per_wl.get("yolov8n")
        d = per_wl.get("dronet")

        def _fmt(info: dict[str, Any] | None) -> str:
            if info is None:
                return "—"
            return f"{info['n']} / {info['argmin']} / **{info['fastest']}**"

        agree = "yes" if (y and d and y["fastest"] == d["fastest"]) else "no"
        lines.append(f"| `{fam}` | {_fmt(y)} | {_fmt(d)} | {agree} |")
    lines.append("")

    # 4. Pathological ratios.
    lines.append("## 4. Pathological cost ratios (top 5 per workload)\n")
    lines.append(
        "Ops where `max(cost) / min(cost) > 100x` are either real backend "
        "specialties (e.g. DSP-accelerated quantized conv vs CPU fallback) "
        "or measurement artifacts. Listed for human triage.\n"
    )
    for wl, items in stats["pathological"].items():
        lines.append(f"### {wl}\n")
        if not items:
            lines.append("- (no ops met the min-cost threshold)\n")
            continue
        lines.append("| op_id | family | CPU us | GPU us | DSP us | ratio | fastest | slowest |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for it in items:
            c = it["costs"]
            lines.append(
                f"| `{it['op_id']}` | `{it['family']}` | "
                f"{c.get('CPU', '—')} | {c.get('GPU', '—')} | {c.get('DSP', '—')} | "
                f"{it['ratio']:.1f}x | {it['fastest_backend']} | {it['slowest_backend']} |"
            )
        lines.append("")

    # 5. SMT.
    lines.append("## 5. SMT formal pass\n")
    lines.append(
        "`prove_cost_monotonicity` requires a shape-parameterized "
        "cost expression of the form `cost(m, n, k)`. The per-op "
        "profile matrix does NOT embed shape — each row is a single "
        "measured op, not a shape family — so the Z3 obligation does "
        "not apply. Honest skip (matches Test #1's pragma: 12 of 12 "
        "shape-parameterized slices in qrb5165_costs.json were skipped "
        "for the same reason or fell outside the (m, n, k) grammar).\n"
    )

    out_dir.joinpath("summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    findings, stats = run_audit()
    elapsed = time.perf_counter() - t0

    with out_dir.joinpath("results.jsonl").open("w") as fh:
        for f in findings:
            fh.write(json.dumps(finding_to_dict(f), default=str) + "\n")

    _emit_summary(findings, stats, elapsed, out_dir)

    sev_counts = Counter(f.severity for f in findings)
    print(f"[exp10] runtime={elapsed:.2f}s findings={len(findings)} by_sev={dict(sev_counts)}")
    print(f"[exp10] results -> {out_dir / 'results.jsonl'}")
    print(f"[exp10] summary -> {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
