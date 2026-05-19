"""Import the realtime_qnn bundle CSVs into the MeasurementCache.

Reads ``realtime_qnn/rt_yolo_f.csv`` (yolov8n@DSP) and
``realtime_qnn/rt_drone_f.csv`` (dronet@GPU), drops 5 warmup iters,
computes mean/p50/p99/stdev/n/deadline_met_rate from the post-warmup
window, and appends one ``CacheEntry`` per CSV under target
``qrb5165``. Idempotent — append-only JSONL, latest-wins semantics.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from xpu_rt.runtime.calibration import (
    TECHNIQUE_CACHED_CONTEXT,
    TECHNIQUE_FULL_BUFFER_REWRITE,
    TECHNIQUE_NO_FILE_IO,
    TECHNIQUE_PER_SENSOR_ROTATION,
    TECHNIQUE_PREALLOC_BUFFERS,
    TECHNIQUE_SCHED_FIFO,
    TECHNIQUE_TIMERFD_ABSTIME,
    _BUNDLE_CSV_REGISTRY,
)
from xpu_rt.runtime.measurement_cache import (
    DEFAULT_CACHE_DIR,
    CacheEntry,
    CacheKey,
    MeasuredStats,
    append_entry,
    load_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "realtime_qnn"
WARMUP_DROP = 5
TARGET_ID = "qrb5165"

# Per-CSV deployment-technique manifest. Cross-reference: REPLICATION.md §1.1.
_BASE_TECHNIQUES: frozenset[str] = frozenset(
    {
        TECHNIQUE_CACHED_CONTEXT,
        TECHNIQUE_PREALLOC_BUFFERS,
        TECHNIQUE_NO_FILE_IO,
        TECHNIQUE_SCHED_FIFO,
        TECHNIQUE_TIMERFD_ABSTIME,
    }
)
_PER_CSV_EXTRA: dict[str, frozenset[str]] = {
    "rt_yolo_f.csv": frozenset({TECHNIQUE_FULL_BUFFER_REWRITE}),
    "rt_drone_f.csv": frozenset(
        {TECHNIQUE_FULL_BUFFER_REWRITE, TECHNIQUE_PER_SENSOR_ROTATION}
    ),
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _stats_from_rows(rows: Iterable[dict[str, str]], *, source: str) -> MeasuredStats:
    rows = list(rows)
    exec_us = [float(r["exec_us"]) for r in rows]
    deadline_met = [int(r["deadline_met"]) for r in rows]
    n = len(exec_us)
    if n == 0:
        raise ValueError("no rows after warmup drop")
    mean = statistics.fmean(exec_us)
    p50 = statistics.median(exec_us)
    sorted_us = sorted(exec_us)
    # p99: index = ceil(0.99 * n) - 1, clamped.
    idx99 = max(0, min(n - 1, int(-(-99 * n // 100)) - 1))
    p99 = sorted_us[idx99]
    stdev = statistics.pstdev(exec_us) if n > 1 else 0.0
    rate = sum(deadline_met) / n if n > 0 else None
    return MeasuredStats(
        mean_us=mean,
        p50_us=p50,
        p99_us=p99,
        stdev_us=stdev,
        n_iters=n,
        deadline_met_rate=rate,
        captured_at=datetime.now(UTC).isoformat(),
        source=source,
    )


def import_bundle(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    bundle_dir: Path = BUNDLE_DIR,
    *,
    warmup_drop: int = WARMUP_DROP,
) -> list[CacheEntry]:
    """Append one CacheEntry per bundle CSV. Returns the entries written."""

    written: list[CacheEntry] = []
    for csv_name, (workload, lane) in _BUNDLE_CSV_REGISTRY.items():
        csv_path = bundle_dir / csv_name
        if not csv_path.is_file():
            print(f"[skip] {csv_path} not found")
            continue
        all_rows = _read_csv_rows(csv_path)
        post_warm = all_rows[warmup_drop:]
        stats = _stats_from_rows(post_warm, source=csv_name)
        techniques = sorted(_BASE_TECHNIQUES | _PER_CSV_EXTRA.get(csv_name, frozenset()))
        key = CacheKey.make(
            target_id=TARGET_ID,
            workload_id=workload,
            lane=lane,
            techniques=techniques,
            period_us=0,
        )
        entry = CacheEntry(key=key, stats=stats)
        append_entry(cache_dir, TARGET_ID, entry)
        written.append(entry)
    return written


def _print_summary(cache_dir: Path) -> None:
    cache = load_cache(cache_dir, TARGET_ID)
    entries = cache.all_for(TARGET_ID)
    print()
    print(f"MeasurementCache @ {cache_dir / (TARGET_ID + '.jsonl')}")
    print(f"  entries: {len(entries)}  (latest-by-key)")
    print()
    hdr = f"  {'workload':<10} {'lane':<5} {'n':>5} {'mean_us':>10} {'p50_us':>10} {'p99_us':>10} {'dl_met':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for e in entries:
        dl = "-" if e.stats.deadline_met_rate is None else f"{e.stats.deadline_met_rate:.3f}"
        print(
            f"  {e.key.workload_id:<10} {e.key.lane:<5} "
            f"{e.stats.n_iters:>5d} {e.stats.mean_us:>10.1f} "
            f"{e.stats.p50_us:>10.1f} {e.stats.p99_us:>10.1f} {dl:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Where to append JSONL. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
        help=f"Path to realtime_qnn bundle. Default: {BUNDLE_DIR}",
    )
    args = parser.parse_args()
    written = import_bundle(args.cache_dir, args.bundle_dir)
    print(f"Appended {len(written)} entries to cache.")
    _print_summary(args.cache_dir)


if __name__ == "__main__":
    main()
