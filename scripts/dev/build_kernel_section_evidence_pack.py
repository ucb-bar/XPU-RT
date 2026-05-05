#!/usr/bin/env python3
"""Build the M-25 Kernel Section Evidence Pack.

Reads existing canonical and/or wide suite run directories and emits
the read-only evidence pack under
``results/graph_compilation/kernel_evidence_pack/`` (default).

Usage:

    python scripts/dev/build_kernel_section_evidence_pack.py \\
        --canonical-suite results/graph_compilation/canonical_m25_suite \\
        --wide-suite     results/graph_compilation/wide_m25_suite \\
        --out            results/graph_compilation/kernel_evidence_pack

Either suite path may be omitted; missing directories are skipped.

This script does NOT run the pipeline. To populate the suite roots
beforehand with kernels enabled:

    COMPGEN_RUN_KERNELS=1 \\
    python -m compgen.graph_compilation run-suite \\
        --suite configs/graph_compilation/always_test_models.yaml \\
        --out   results/graph_compilation/canonical_m25_suite \\
        --stop-after agent-decision-request \\
        --selection-mode greedy \\
        --continue-on-failure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compgen.graph_compilation.kernel_evidence_pack import (
    build_kernel_evidence_pack,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--canonical-suite", type=Path, default=None,
        help="canonical suite run-root (e.g. results/.../canonical_m25_suite)",
    )
    p.add_argument(
        "--wide-suite", type=Path, default=None,
        help="wide suite run-root",
    )
    p.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "graph_compilation"
        / "kernel_evidence_pack",
    )
    p.add_argument(
        "--skip-figures", action="store_true",
        help="skip matplotlib renderers",
    )
    args = p.parse_args()

    if args.canonical_suite is None and args.wide_suite is None:
        raise SystemExit(
            "must provide --canonical-suite and/or --wide-suite"
        )

    res = build_kernel_evidence_pack(
        canonical_suite=args.canonical_suite,
        wide_suite=args.wide_suite,
        out_dir=args.out,
        skip_figures=args.skip_figures,
    )
    print(f"models aggregated: {res.model_count}")
    print(f"summary:           {res.summary_md}")
    print(f"claim matrix:      {res.claim_matrix}")
    print(f"model matrix CSV:  {res.model_matrix_csv}")
    print(f"compiled coverage: {res.compiled_coverage_csv}")
    print(f"register pressure: {res.register_pressure_csv}")
    print(f"evidence tables:   {res.evidence_tables}")
    print(f"figures:           {res.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
