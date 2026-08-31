#!/usr/bin/env python3
"""Make schedule profile provenance independent of the checkout directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "xpu-rt"))

from profile_loader import compute_pdb_hash  # noqa: E402


def _schedule_objects(value):
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict) and metadata.get("pdb_files"):
            yield value
        for child in value.values():
            yield from _schedule_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schedule_objects(child)


def _portable(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return os.path.normpath(path)
    try:
        return os.path.relpath(candidate.resolve(), _REPO)
    except ValueError:
        return path


def normalize(path: Path) -> bool:
    with path.open() as stream:
        payload = json.load(stream)
    changed = False
    for schedule in _schedule_objects(payload):
        metadata = schedule["metadata"]
        portable = [_portable(str(item)) for item in metadata["pdb_files"]]
        digest, used = compute_pdb_hash(portable, base_dir=str(_REPO))
        if len(used) != len(set(portable)):
            missing = sorted(set(portable) - set(used))
            raise SystemExit(f"{path}: missing profile files: {missing}")
        if portable != metadata["pdb_files"] or digest != metadata.get("pdb_hash"):
            metadata["pdb_files"] = portable
            metadata["pdb_hash"] = digest
            changed = True
    if changed:
        with path.open("w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    changed = [str(path) for path in files if normalize(path)]
    print(f"normalized {len(changed)} of {len(files)} JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
