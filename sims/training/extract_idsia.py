#!/usr/bin/env python3
"""Extract the IDSIA Forest Trails dataset into a flat per-segment directory.

The download from IDSIA is a zip-of-zips: a single ``files-archive`` containing
fifteen inner zips ``000.zip ... 014.zip``. Each inner zip extracts into a
``NNN/...`` directory tree of the form::

    NNN/
        info.txt
        videos/
            lc/<name>.mp4.frames/00000001.jpg ...   (left  camera)
            sc/<name>.mp4.frames/00000001.jpg ...   (centre camera)
            rc/<name>.mp4.frames/00000001.jpg ...   (right camera)

This script does both layers in one pass and skips entries that have already
been extracted, so re-runs are cheap.

Usage::

    python sims/training/extract_idsia.py
    python sims/training/extract_idsia.py \\
        --archive datasets/idsia/files-archive \\
        --out datasets/idsia/extracted
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO_ROOT / "datasets" / "idsia" / "files-archive"
DEFAULT_OUT = REPO_ROOT / "datasets" / "idsia" / "extracted"

# Junk emitted by macOS-zipped archives — never useful for training.
JUNK_PREFIXES = ("__MACOSX/", "__MACOSX\\")
JUNK_BASENAMES = (".DS_Store",)


def _is_junk(name: str) -> bool:
    if name.startswith(JUNK_PREFIXES):
        return True
    base = os.path.basename(name)
    return base in JUNK_BASENAMES or base.startswith("._")


def extract_inner_zip(inner_bytes: bytes, out_root: Path, label: str) -> tuple[int, int]:
    """Extract one inner ``NNN.zip`` into ``out_root``.

    Returns ``(extracted, skipped)`` counts.
    """
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir() or _is_junk(info.filename):
                continue
            target = out_root / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                # 1 MiB chunks — these are JPEGs, no point loading whole files.
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1
        if extracted % 500 == 0 and extracted > 0:
            print(f"  [{label}] extracted={extracted} skipped={skipped}", flush=True)
    return extracted, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                   help=f"Path to outer files-archive zip (default: {DEFAULT_ARCHIVE.relative_to(REPO_ROOT)}).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output root for extracted segments (default: {DEFAULT_OUT.relative_to(REPO_ROOT)}).")
    p.add_argument("--only", nargs="*", type=str, default=None,
                   help="Optional list of segment names to extract (e.g. 001 002). Default: all.")
    args = p.parse_args()

    if not args.archive.is_file():
        print(f"[error] archive not found: {args.archive}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[info] outer archive: {args.archive}")
    print(f"[info] out root:      {args.out}")

    total_extracted = 0
    total_skipped = 0
    with zipfile.ZipFile(args.archive) as outer:
        inner_names = sorted(n for n in outer.namelist() if n.lower().endswith(".zip"))
        if args.only:
            wanted = {f"{s}.zip" for s in args.only}
            inner_names = [n for n in inner_names if n in wanted]
            if not inner_names:
                print(f"[error] no inner zips matched --only={args.only}", file=sys.stderr)
                return 2
        print(f"[info] {len(inner_names)} inner zip(s) to process")

        for idx, inner in enumerate(inner_names, 1):
            label = inner.replace(".zip", "")
            print(f"[{idx}/{len(inner_names)}] {inner}", flush=True)
            with outer.open(inner) as f:
                buf = f.read()
            ex, sk = extract_inner_zip(buf, args.out, label)
            total_extracted += ex
            total_skipped += sk
            print(f"  done: extracted={ex} skipped={sk}", flush=True)

    print(f"[info] complete. total extracted={total_extracted} skipped={total_skipped}")
    print(f"[info] tree at: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
