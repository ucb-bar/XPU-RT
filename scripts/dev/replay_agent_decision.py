#!/usr/bin/env python
"""CLI wrapper for trace replay (M-31A.3).

Usage::

    uv run python scripts/dev/replay_agent_decision.py \\
        --trace results/.../agent_decision_trace_0000.json \\
        --run-dir results/.../<run> \\
        [--promotion-library .compgen_cache/recipes/]

Exit 0 iff every hash in the trace re-derives from the run dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.audit.errors import ReplayHashMismatch
from compgen.audit.trace_replay import replay


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay an agent decision trace")
    p.add_argument("--trace", type=Path, required=True,
                   help="Path to a saved agent_decision_trace_<n>.json")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run dir to verify against (default: trace's parent)")
    p.add_argument("--promotion-library", type=Path, default=None,
                   help="Recipe library root (default: .compgen_cache/recipes)")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write the replay report as JSON")
    p.add_argument("--lenient", action="store_true",
                   help="Don't exit non-zero on mismatch; print the report")
    args = p.parse_args(argv)

    if not args.trace.exists():
        print(f"FAIL: trace {args.trace} does not exist", file=sys.stderr)
        return 2
    run_dir = args.run_dir or args.trace.parent

    try:
        report = replay(
            trace_path=args.trace,
            run_dir=run_dir,
            promotion_library=args.promotion_library,
            strict=not args.lenient,
        )
    except ReplayHashMismatch as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        f"replay {args.trace.name}: decision_id_match={report.decision_id_match} "
        f"input_hashes_match={report.input_hashes_match} "
        f"output_hashes_match={report.output_hashes_match}"
    )
    if report.input_deltas:
        for name, (exp, act) in report.input_deltas.items():
            print(f"  input delta {name}: expected={exp[:16]} actual={act[:16]}")
    if report.output_deltas:
        for name, (exp, act) in report.output_deltas.items():
            print(f"  output delta {name}: expected={exp[:16]} actual={act[:16]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")

    return 0 if report.all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
