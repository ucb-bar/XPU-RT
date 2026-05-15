#!/usr/bin/env python3
"""Remove ephemeral and staging artifacts from the workspace."""
from __future__ import annotations

import shutil
from pathlib import Path

TO_DELETE = [
    ".compgen",
    "artifacts/runs",
    "artifacts/cache",
    "artifacts/traces",
    "artifacts/tmp",
    "generated/staging",
    "generated/scratch",
    "benchmarks/results",
    "benchmarks/tmp",
]


def main() -> None:
    for rel in TO_DELETE:
        p = Path(rel)
        if p.exists():
            print(f"Removing {p}")
            shutil.rmtree(p)


if __name__ == "__main__":
    main()
