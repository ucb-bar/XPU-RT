#!/usr/bin/env python3
"""Build the M-17 Graph Section Evidence Pack.

Reads existing canonical and/or wide suite run directories and emits
the read-only evidence pack under
``results/graph_compilation/evidence_pack/`` (default).

Usage:

    python scripts/dev/build_graph_section_evidence_pack.py \\
        --canonical-suite results/graph_compilation/canonical_m17_suite \\
        --wide-suite results/graph_compilation/wide_m17_suite \\
        --out results/graph_compilation/evidence_pack

Either suite path may be omitted; missing directories are skipped.

This script does NOT run the pipeline. To populate the suite roots
beforehand:

    python -m compgen.graph_compilation run-suite \\
        --suite configs/graph_compilation/always_test_models.yaml \\
        --out results/graph_compilation/canonical_m17_suite \\
        --stop-after real-transform-differential \\
        --selection-mode greedy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--canonical-suite", type=Path, default=None,
        help="Path to a canonical suite run directory (one subdir per model).",
    )
    parser.add_argument(
        "--wide-suite", type=Path, default=None,
        help="Path to a wide suite run directory (one subdir per model).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/graph_compilation/evidence_pack"),
        help="Output directory for the evidence pack.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure rendering (useful for unit tests / no-matplotlib envs).",
    )
    args = parser.parse_args(argv)

    from compgen.graph_compilation.evidence_pack import build_evidence_pack

    if args.canonical_suite is None and args.wide_suite is None:
        parser.error(
            "must pass at least one of --canonical-suite or --wide-suite"
        )

    result = build_evidence_pack(
        canonical_suite_root=args.canonical_suite,
        wide_suite_root=args.wide_suite,
        out_dir=args.out,
        skip_figures=args.no_figures,
    )
    print(f"evidence pack: {result.out_dir}")
    print(f"  models: {len(result.rows)}")
    print(f"  bit_equality discharged: "
          f"{result.aggregates['bit_equality_discharged_count']}")
    print(f"  real transform families discharged: "
          f"{result.aggregates['real_transform_families_discharged_count']}")
    print(f"  agent changed from greedy: "
          f"{result.aggregates['agent_changed_from_greedy_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
