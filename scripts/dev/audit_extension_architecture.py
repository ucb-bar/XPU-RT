#!/usr/bin/env python
"""architecture audit for the Phase F extension substrate.

Encodes the ten hard rules as deterministic checks::

  extension_card_completeness
  blocked_provider_not_paper_claimable
  pass_tool_no_direct_ir_mutation
  optional_provider_imports_quarantined
  provider_result_is_not_certificate

Exits 0 when all checks pass, 1 on any violation. Writes a typed
JSON report when ``--out`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.audit.extension_architecture import run_audit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON output path",
    )
    p.add_argument("--repo-root", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_audit(repo_root=args.repo_root)
    body = report.to_dict()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(body, indent=2, sort_keys=True))

    print(f"Extension architecture audit — {len(report.checks_run)} checks ran")
    for c in report.checks_run:
        print(f"  ✓ {c}")
    if report.passed:
        print(f"\nALL CHECKS PASSED ({body['summary']})")
        return 0
    print(f"\nFAILED: {len(report.violations)} violation(s)")
    for v in report.violations:
        print(f"  - {v.check}: {v.path} ({v.reason})")
        if v.detail:
            print(f"      detail: {v.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
