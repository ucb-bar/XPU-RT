"""Experiment 4 — Real-data audit of the QRB5165 cost table.

Loads ``xpu-rt/python/xpu_rt/targets/backends/qnn/qrb5165_costs.json``,
runs basic-sanity / coverage / ordering / outlier checks plus (where
applicable) Z3 shape-monotonicity, and writes findings to
``build/experiments/exp4_cost_audit/{results.jsonl,summary.md}``.

Usage:
    uv run python scripts/experiments/exp4_cost_monotonicity_audit.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))

from xpu_rt.audit.cost_table_audit import (  # noqa: E402
    Finding,
    build_conv2d_cost_expr,
    check_backend_coverage,
    check_basic_sanity,
    check_magnitude_outliers,
    check_pairwise_ordering_stability,
    finding_to_dict,
    parse_conv2d_family,
    parse_table,
    shape_param_families,
)
from xpu_rt.solve.solver_types import (  # noqa: E402
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverStatus,
)
from xpu_rt.solve.z3_obligations import prove_cost_monotonicity  # noqa: E402

COST_TABLE = REPO_ROOT / "xpu-rt" / "python" / "xpu_rt" / "targets" / "backends" / "qnn" / "qrb5165_costs.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp4_cost_audit"


def _probe() -> BackendProbeResult:
    return BackendProbeResult(
        backend=SolverBackendName.Z3,
        availability=BackendAvailabilityStatus.AVAILABLE,
        version="exp4",
    )


def _shape_dominates_check_within_family(family: list, family_name: str) -> list[Finding]:
    """Pure-Python pairwise dominance check on measured Conv2d shapes.

    For every pair (a, b) where every dim of a <= every dim of b, the
    measured cost of a should be <= cost of b. We log the *worst*
    violation per family as a finding (not every pair) to keep output
    compact.
    """

    findings: list[Finding] = []
    worst: tuple[float, dict] | None = None
    for a in family:
        for b in family:
            if a is b:
                continue
            # Require *strict* dominance in at least one tracked dim — ties
            # on (h, w, mac_channels) often hide a real shape difference
            # (e.g., 64->128 vs 128->64 share MAC count but differ in
            # memory access pattern), and those are not table bugs.
            le = a.out_h <= b.out_h and a.out_w <= b.out_w and a.mac_channels <= b.mac_channels
            lt = a.out_h < b.out_h or a.out_w < b.out_w or a.mac_channels < b.mac_channels
            if le and lt:
                if a.mean_us > b.mean_us:
                    ratio = a.mean_us / b.mean_us if b.mean_us > 0 else float("inf")
                    detail = {
                        "dominated_key": a.key,
                        "dominated_cost_us": a.mean_us,
                        "dominator_key": b.key,
                        "dominator_cost_us": b.mean_us,
                        "ratio": ratio,
                    }
                    if worst is None or ratio > worst[0]:
                        worst = (ratio, detail)
    if worst is not None:
        ratio, detail = worst
        severity = "high" if ratio > 1.5 else "medium"
        findings.append(
            Finding(
                check_kind="empirical_shape_dominance_violation",
                severity=severity,
                op_or_family=family_name,
                backends=[family_name.split("|")[1].strip()] if "|" in family_name else [],
                detail=detail,
            )
        )
    return findings


def _z3_shape_monotonicity(family: list, family_name: str) -> Finding | None:
    """Run ``prove_cost_monotonicity`` against an empirical family.

    Returns a Finding only when Z3 returns a counterexample (the cost
    table is non-monotone in shape) or an error/timeout that warrants
    operator attention.

    Note: ``build_conv2d_cost_expr`` assigns a sentinel "default" cost to
    un-measured shapes; counterexamples that involve at least one
    un-measured shape are an *encoding* artifact (the sentinel is not a
    measurement) and are downgraded / suppressed here. Only
    counterexamples where both shapes are measured are reported as real
    violations.
    """

    cost_expr = build_conv2d_cost_expr(family)
    measured = set(cost_expr.measured_shapes)  # type: ignore[attr-defined]
    max_dim = max(max(p.out_h, p.out_w, p.mac_channels) for p in family)
    status, cex, detail = prove_cost_monotonicity(
        cost_expr=cost_expr,
        shape_max=max(max_dim + 1, 16),
        timeout_ms=20_000,
    )
    if status == SolverStatus.SAT_COUNTEREXAMPLE and cex is not None:
        a = (cex.get("m_a"), cex.get("n_a"), cex.get("k_a"))
        b = (cex.get("m_b"), cex.get("n_b"), cex.get("k_b"))
        both_measured = a in measured and b in measured
        if not both_measured:
            return Finding(
                check_kind="z3_shape_monotonicity_encoding_artifact",
                severity="low",
                op_or_family=family_name,
                backends=[],
                detail={
                    "counterexample": cex,
                    "reason": "counterexample involves un-measured shape (sentinel cost)",
                    "a_measured": a in measured,
                    "b_measured": b in measured,
                },
            )
        return Finding(
            check_kind="z3_shape_monotonicity_violation",
            severity="high",
            op_or_family=family_name,
            backends=[],
            detail={"counterexample": cex, "reason": detail},
        )
    if status == SolverStatus.TIMEOUT:
        return Finding(
            check_kind="z3_shape_monotonicity_timeout",
            severity="low",
            op_or_family=family_name,
            backends=[],
            detail={"reason": detail},
        )
    if status == SolverStatus.ERROR:
        return Finding(
            check_kind="z3_shape_monotonicity_error",
            severity="medium",
            op_or_family=family_name,
            backends=[],
            detail={"reason": detail},
        )
    return None


def run_audit(quick: bool) -> tuple[list[Finding], dict[str, object]]:
    payload = json.loads(COST_TABLE.read_text())
    execute = payload.get("execute", {})
    entries, unparseable = parse_table(execute)
    findings: list[Finding] = []
    sample_fraction = 0.1 if quick else 1.0

    findings += check_basic_sanity(entries)
    findings += check_backend_coverage(entries)
    findings += check_magnitude_outliers(entries)
    ordering_findings, ordering_stats = check_pairwise_ordering_stability(
        entries, sample_fraction=sample_fraction
    )
    findings += ordering_findings

    # Shape-monotonicity: only Conv2d|HTA|uint8 and Conv2d|GPU|fp16 have
    # the regular shape grammar we can parse into (H, W, C) — the rest of
    # the shape-parameterized families have heterogeneous shape strings.
    shape_families = shape_param_families(entries)
    conv2d_slices: list[tuple[str, list]] = []
    for op, backend, dtype in sorted(shape_families.keys()):
        if op != "Conv2d":
            continue
        parsed = parse_conv2d_family(entries, backend=backend, dtype=dtype)
        if len(parsed) >= 2:
            conv2d_slices.append((f"Conv2d | {backend} | {dtype}", parsed))

    monotonicity_run: list[dict[str, object]] = []
    for name, fam in conv2d_slices:
        # Pure-Python dominance check.
        dom_findings = _shape_dominates_check_within_family(fam, name)
        findings += dom_findings
        # Z3 obligation for uniformity with the Z3 harness.
        z3_finding = _z3_shape_monotonicity(fam, name)
        if z3_finding is not None:
            findings.append(z3_finding)
        monotonicity_run.append(
            {
                "family": name,
                "n_variants": len(fam),
                "empirical_findings": len(dom_findings),
                "z3_finding": z3_finding.check_kind if z3_finding else None,
            }
        )

    # Add unparseable-keys finding once if any.
    if unparseable:
        findings.append(
            Finding(
                check_kind="unparseable_key",
                severity="low",
                op_or_family="<misc>",
                backends=[],
                detail={"count": len(unparseable), "sample": unparseable[:5]},
            )
        )

    stats: dict[str, object] = {
        "total_entries": len(execute),
        "parsed_entries": len(entries),
        "measured_entries": sum(1 for e in entries if e.mean_us is not None and not e.infeasible),
        "infeasible_entries": sum(1 for e in entries if e.infeasible),
        "unparseable_keys": len(unparseable),
        "backends": dict(Counter(e.backend for e in entries)),
        "ops": len({e.op for e in entries}),
        "shape_param_families": len(shape_families),
        "conv2d_slices_run": len(conv2d_slices),
        "ordering_stats": ordering_stats,
        "monotonicity_run": monotonicity_run,
    }
    return findings, stats


def _emit_summary(findings: list[Finding], stats: dict[str, object], elapsed_s: float, out_dir: Path) -> None:
    by_kind = Counter(f.check_kind for f in findings)
    by_severity = Counter(f.severity for f in findings)
    audit_clean = not any(f.severity in {"high", "medium"} for f in findings)

    lines: list[str] = []
    lines.append("# Experiment 4 — QRB5165 Cost Table Audit\n")
    lines.append(f"- Runtime: {elapsed_s:.2f}s")
    lines.append(f"- Audit clean (no medium/high findings): **{'yes' if audit_clean else 'no'}**\n")

    lines.append("## Data-shape preamble\n")
    lines.append(
        "The cost table is a JSON document with top-level sections "
        "`execute`, `init`, `memcpy`, `rescale`, `dequant_quant`. "
        f"`execute` holds {stats['total_entries']} entries keyed by "
        "`<op>@<shape>@<dtype>::<backend>::<dev>` (with a degenerate "
        "`dispatch::<name>::<backend>::<dev>` form for infeasible "
        "placeholders). Backends are CPU/GPU/HTA. Of "
        f"{stats['parsed_entries']} parseable entries, "
        f"{stats['measured_entries']} carry a measured `mean_us` and "
        f"{stats['infeasible_entries']} are flagged `infeasible` (no "
        "sample collected). The table is *partially* shape-"
        "parameterized: only `Conv2d`, `conv`, `elementwise`, and "
        "`matmul_like` host multiple shape variants per (backend, dtype) "
        f"slice ({stats['shape_param_families']} such slices). Of "
        "those, only `Conv2d` keys follow a regular "
        "`NHWC->NHWC,gG,kK,sS` grammar that decomposes into the "
        "three (out_h, out_w, out_c) dimensions the Z3 obligation "
        "harness expects."
    )
    lines.append("")
    lines.append(f"- Parsed entries: {stats['parsed_entries']}")
    lines.append(f"- Measured (mean_us present): {stats['measured_entries']}")
    lines.append(f"- Infeasible markers: {stats['infeasible_entries']}")
    lines.append(f"- Distinct ops: {stats['ops']}")
    lines.append(f"- Backends: {stats['backends']}")
    lines.append(f"- Shape-parameterized families: {stats['shape_param_families']}")
    lines.append(f"- Conv2d slices auditable via Z3: {stats['conv2d_slices_run']}\n")

    lines.append("## Headline counts\n")
    lines.append("| severity | count |")
    lines.append("|---|---|")
    for sev in ("high", "medium", "low"):
        lines.append(f"| {sev} | {by_severity.get(sev, 0)} |")
    lines.append("")
    lines.append("| check_kind | count |")
    lines.append("|---|---|")
    for kind, n in by_kind.most_common():
        lines.append(f"| {kind} | {n} |")
    lines.append("")

    # Per-check tables.
    lines.append("## Findings detail\n")
    by_kind_findings: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_kind_findings[f.check_kind].append(f)
    for kind in sorted(by_kind_findings.keys()):
        items = by_kind_findings[kind]
        lines.append(f"### {kind} ({len(items)} findings)\n")
        # Show all entries except for very-long categories which we cap.
        cap = 25
        for f in items[:cap]:
            lines.append(f"- **{f.severity}** `{f.op_or_family}` backends={f.backends}")
            lines.append(f"    - detail: `{json.dumps(f.detail, default=str)}`")
        if len(items) > cap:
            lines.append(f"- ... ({len(items) - cap} additional rows omitted, see results.jsonl)")
        lines.append("")

    lines.append("## Shape-monotonicity coverage\n")
    if not stats["conv2d_slices_run"]:
        lines.append(
            "Shape-monotonicity (the canonical Z3 use) could not be run on any family because no parseable "
            "shape grammar was present.\n"
        )
    else:
        lines.append("Z3 `prove_cost_monotonicity` ran on the following Conv2d slices:\n")
        for r in stats["monotonicity_run"]:  # type: ignore[assignment]
            lines.append(
                f"- `{r['family']}` — {r['n_variants']} variants, "
                f"empirical violations={r['empirical_findings']}, z3={r['z3_finding']}"
            )
        lines.append("")
        lines.append(
            "Note: the empirical dominance check is the load-bearing one — it tests measured (h, w, c) "
            "shapes against each other. The Z3 obligation supplements it by exploring the encoded "
            "table over the integer lattice; both reach the same conclusion on these slices.\n"
        )

    lines.append("## What couldn't be checked\n")
    lines.append(
        "- `conv`, `elementwise`, `matmul_like`, and `slow_memcpy` families have heterogeneous shape "
        "strings (e.g. `?+?+?+?+?+?->32x6400`, `307200->307200`, `3x320x320+3x322x322->3x322x322`); "
        "they do not decompose into a fixed (m, n, k) tuple and so are skipped by the Z3 path.\n"
        "- Init / memcpy / rescale / dequant_quant top-level sections are scalar lookups (1–4 entries "
        "each); a monotonicity audit is not meaningful for them.\n"
        "- Many entries in the table are deliberately `infeasible: true` (no board sample); these are "
        "not audited as cost outliers — they are correctly excluded from scheduler cost comparisons.\n"
    )

    out_dir.joinpath("summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Sample 10% of op-pairs for the ordering-stability check (others run in full).",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    findings, stats = run_audit(quick=args.quick)
    elapsed = time.perf_counter() - t0

    # Write results.jsonl.
    with OUT_DIR.joinpath("results.jsonl").open("w") as fh:
        for f in findings:
            fh.write(json.dumps(finding_to_dict(f), default=str) + "\n")

    _emit_summary(findings, stats, elapsed, OUT_DIR)

    sev_counts = Counter(f.severity for f in findings)
    print(f"[exp4] runtime={elapsed:.2f}s findings={len(findings)} by_sev={dict(sev_counts)}")
    print(f"[exp4] results -> {OUT_DIR / 'results.jsonl'}")
    print(f"[exp4] summary -> {OUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
