#!/usr/bin/env python
"""CLI wrapper for the realness scan (M-31A.2).

Usage::

    uv run python scripts/dev/audit_realness.py
    uv run python scripts/dev/audit_realness.py --include-tests
    uv run python scripts/dev/audit_realness.py --json out/scan.json

Exit code 0 only if every hit is allowlisted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.audit.realness_scan import (
    Allowlist,
    assert_clean,
    scan_repo,
)
from compgen.audit.errors import UnallowlistedStubError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Realness no-stub source scan")
    p.add_argument("--include-tests", action="store_true",
                   help="Scan tests/ as well (default: skip)")
    p.add_argument("--json", type=Path, default=None,
                   help="Write the scan report as JSON to this path")
    p.add_argument("--allowlist", type=Path, default=None,
                   help="Override the allowlist YAML path")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="Override repo root (default: package-resolved)")
    args = p.parse_args(argv)

    allowlist = Allowlist.load(args.allowlist) if args.allowlist else None
    report = scan_repo(
        repo_root=args.repo_root,
        include_tests=args.include_tests,
        allowlist=allowlist,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    print(
        f"realness scan: {report.files_scanned} files scanned, "
        f"{len(report.hits)} hits ({len(report.unallowlisted_hits)} unallowlisted)"
    )
    if report.unallowlisted_hits:
        for h in report.unallowlisted_hits[:50]:
            print(f"  FAIL {h.path}:{h.line_number}: [{h.marker}] {h.line_text}")
        if len(report.unallowlisted_hits) > 50:
            print(f"  ... and {len(report.unallowlisted_hits) - 50} more")

    try:
        assert_clean(report)
    except UnallowlistedStubError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
