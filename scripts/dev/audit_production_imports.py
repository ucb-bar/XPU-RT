#!/usr/bin/env python
"""CLI wrapper for production-import provenance audit.

Reads ``<run_dir>/import_provenance.json`` and asserts no forbidden
modules were imported by the production run.

Usage::

    uv run python scripts/dev/audit_production_imports.py <run_dir>

Exit 0 iff the run is clean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compgen.audit.errors import ForbiddenImportError
from compgen.audit.import_provenance import (
    assert_no_forbidden,
    load_provenance,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Production import provenance audit")
    p.add_argument("run_dir", type=Path, help="Path to a graph_compilation run dir")
    p.add_argument(
        "--also-forbid",
        action="append",
        default=[],
        help="Additional module prefixes to forbid (repeatable)",
    )
    args = p.parse_args(argv)

    prov_path = args.run_dir / "import_provenance.json"
    if not prov_path.exists():
        print(f"FAIL: {prov_path} does not exist (was the run produced under M-31A?)",
              file=sys.stderr)
        return 2
    prov = load_provenance(prov_path)
    print(
        f"run_id={prov.run_id} selection_mode={prov.selection_mode} "
        f"cache_mode={prov.cache_mode} evidence_mode={prov.evidence_mode}"
    )
    print(f"  forbidden imports: {prov.forbidden_modules_imported or 'none'}")
    print(f"  mock imports     : {prov.mock_modules_imported or 'none'}")
    try:
        assert_no_forbidden(
            prov,
            additional_forbidden=tuple(args.also_forbid),
        )
    except ForbiddenImportError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    print("OK: no forbidden imports detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
