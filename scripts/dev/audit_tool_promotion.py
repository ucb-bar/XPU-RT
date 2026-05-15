#!/usr/bin/env -S uv run python
"""CLI wrapper for the tool-promotion audit.

Usage::

    uv run python scripts/dev/audit_tool_promotion.py [--json] [--out PATH]

Exit code 0 if every audited card's declared maturity is verified by
its evidence; exit code 1 if any violation surfaces. The default
output is the Markdown rollup on stdout; ``--json`` switches to the
machine-readable shape used by the tool-evidence pack; ``--out``
writes the report to disk (also JSON).

Mirrors the convention of
``scripts/dev/audit_extension_architecture.py`` so CI gates can wire
the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.audit.tool_promotion import run_tool_promotion_audit


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards-root",
        type=Path,
        default=None,
        help="Override the ToolCard YAML directory (defaults to the shipped one).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the repo root (defaults to the CompGen checkout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of Markdown.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the JSON report to this file as well.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report = run_tool_promotion_audit(
        cards_root=args.cards_root,
        repo_root=args.repo_root,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    else:
        print(report.to_markdown())
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
