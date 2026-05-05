#!/usr/bin/env python
"""Build the CompGen trust report (M-31A.5).

Usage::

    uv run python scripts/dev/build_trust_report.py
    uv run python scripts/dev/build_trust_report.py --run-dir /path/to/run
    uv run python scripts/dev/build_trust_report.py --out results/audit/$(git rev-parse --short HEAD)

Exit 0 iff every gate is ``pass`` (skipped gates are tolerated; the
report records them honestly).
"""

from __future__ import annotations

from compgen.audit.trust_report import _cli_main

if __name__ == "__main__":
    raise SystemExit(_cli_main())
