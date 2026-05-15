#!/usr/bin/env python
"""probe every shipped provider / target / dialect card and
write the typed status reports + matrices into ``--out`` (default
``results/extension_provider_probe``).

Outputs::

    <out>/provider_status.json
    <out>/target_status.json
    <out>/dialect_status.json
    <out>/pass_tool_status.json
    <out>/provider_target_matrix.csv
    <out>/provider_contract_matrix.csv
    <out>/probe_summary.md

Hard rule 5: missing SDKs / hardware / licenses / packages always
produce a typed ``blocked`` status with a typed ``blocked_reason``
— never a crash, never a silent disappearance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compgen.providers.provider_reports import write_probe_reports


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/extension_provider_probe"),
        help="output directory (default: results/extension_provider_probe)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = write_probe_reports(args.out)
    print(f"Wrote probe report set to {args.out}")
    for label, rel in sorted(paths.items()):
        print(f"  {label}: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
