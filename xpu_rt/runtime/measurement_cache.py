"""Append-only cache of on-board schedule measurements.

Decisions made from a cache hit are grounded in real wall time;
decisions made from a cache miss fall back to the v4 calibration's
prediction as a prior. The cache key includes deployment techniques
so cold-start and warm-loop measurements never get conflated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import structlog

log = structlog.get_logger(__name__)

MEASUREMENT_CACHE_SCHEMA_VERSION = "measurement_cache_v1"

# Canonical on-disk location. Per-target JSONL avoids cross-target lock
# contention and makes pruning a single target trivial.
DEFAULT_CACHE_DIR = Path("xpu-rt/data/measurement_cache")


@dataclass(frozen=True)
class CacheKey:
    """Identity of a measured schedule configuration.

    ``deployment_techniques`` is a sorted tuple so dict / set lookups are
    canonical. ``period_us = 0`` means ASAP (no period throttling).
    """

    target_id: str
    workload_id: str
    lane: str
    deployment_techniques: tuple[str, ...]
    period_us: int

    @staticmethod
    def make(
        target_id: str,
        workload_id: str,
        lane: str,
        techniques: Iterable[str],
        period_us: int = 0,
    ) -> CacheKey:
        return CacheKey(
            target_id=str(target_id),
            workload_id=str(workload_id),
            lane=str(lane),
            deployment_techniques=tuple(sorted(set(techniques))),
            period_us=int(period_us),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "workload_id": self.workload_id,
            "lane": self.lane,
            "deployment_techniques": list(self.deployment_techniques),
            "period_us": self.period_us,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CacheKey:
        return CacheKey.make(
            target_id=d["target_id"],
            workload_id=d["workload_id"],
            lane=d["lane"],
            techniques=tuple(d.get("deployment_techniques", ())),
            period_us=int(d.get("period_us", 0)),
        )


@dataclass(frozen=True)
class MeasuredStats:
    """Stats summarising a multi-iter measurement run."""

    mean_us: float
    p50_us: float
    p99_us: float
    stdev_us: float
    n_iters: int
    deadline_met_rate: float | None
    captured_at: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p99_us": self.p99_us,
            "stdev_us": self.stdev_us,
            "n_iters": self.n_iters,
            "deadline_met_rate": self.deadline_met_rate,
            "captured_at": self.captured_at,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> MeasuredStats:
        return MeasuredStats(
            mean_us=float(d["mean_us"]),
            p50_us=float(d["p50_us"]),
            p99_us=float(d["p99_us"]),
            stdev_us=float(d["stdev_us"]),
            n_iters=int(d["n_iters"]),
            deadline_met_rate=(
                None if d.get("deadline_met_rate") is None
                else float(d["deadline_met_rate"])
            ),
            captured_at=str(d["captured_at"]),
            source=str(d["source"]),
        )


@dataclass(frozen=True)
class CacheEntry:
    """One cached measurement."""

    key: CacheKey
    stats: MeasuredStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEASUREMENT_CACHE_SCHEMA_VERSION,
            "key": self.key.to_dict(),
            "stats": self.stats.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CacheEntry:
        return CacheEntry(
            key=CacheKey.from_dict(d["key"]),
            stats=MeasuredStats.from_dict(d["stats"]),
        )


@dataclass(frozen=True)
class MeasurementCache:
    """In-memory snapshot of the on-disk JSONL log.

    The on-disk format is append-only JSONL at
    ``<cache_dir>/<target_id>.jsonl`` so concurrent writers from different
    runs don't corrupt the file. When two entries share a key, the later
    one wins (interpretation: it superseded the earlier measurement).
    """

    entries: tuple[CacheEntry, ...] = ()

    def _latest_by_key(self) -> dict[CacheKey, CacheEntry]:
        latest: dict[CacheKey, CacheEntry] = {}
        for entry in self.entries:
            latest[entry.key] = entry
        return latest

    def get(self, key: CacheKey) -> CacheEntry | None:
        return self._latest_by_key().get(key)

    def get_best(
        self,
        target_id: str,
        workload_id: str,
        lane: str | None = None,
        prefer_techniques: tuple[str, ...] | None = None,
    ) -> CacheEntry | None:
        """Return the lowest-p50 entry matching the filter, or ``None``.

        When ``prefer_techniques`` is provided, entries whose technique
        set is a superset are preferred; if any such exists only those
        are considered, otherwise all matching entries are eligible.
        """

        candidates: list[CacheEntry] = []
        for entry in self._latest_by_key().values():
            if entry.key.target_id != target_id:
                continue
            if entry.key.workload_id != workload_id:
                continue
            if lane is not None and entry.key.lane != lane:
                continue
            candidates.append(entry)
        if not candidates:
            return None
        if prefer_techniques is not None:
            want = set(prefer_techniques)
            superset = [
                c for c in candidates
                if want.issubset(set(c.key.deployment_techniques))
            ]
            if superset:
                candidates = superset
        return min(candidates, key=lambda e: e.stats.p50_us)

    def all_for(
        self,
        target_id: str,
        workload_id: str | None = None,
    ) -> tuple[CacheEntry, ...]:
        out = []
        for entry in self._latest_by_key().values():
            if entry.key.target_id != target_id:
                continue
            if workload_id is not None and entry.key.workload_id != workload_id:
                continue
            out.append(entry)
        return tuple(out)


def cache_path_for(cache_dir: Path, target_id: str) -> Path:
    return Path(cache_dir) / f"{target_id}.jsonl"


def load_cache(cache_dir: Path, target_id: str) -> MeasurementCache:
    """Load all entries for ``target_id`` from the JSONL log.

    Returns an empty cache when the file does not exist; this is the
    expected first-run state and is not an error.
    """

    path = cache_path_for(cache_dir, target_id)
    if not path.is_file():
        return MeasurementCache(entries=())
    entries: list[CacheEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning(
                    "measurement_cache_skip_bad_line",
                    target=target_id,
                    line=ln,
                    error=type(exc).__name__,
                )
                continue
            try:
                entries.append(CacheEntry.from_dict(obj))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(
                    "measurement_cache_skip_bad_record",
                    target=target_id,
                    line=ln,
                    error=type(exc).__name__,
                )
                continue
    return MeasurementCache(entries=tuple(entries))


def append_entry(cache_dir: Path, target_id: str, entry: CacheEntry) -> None:
    """Append one entry to the per-target JSONL log.

    Creates the cache directory if missing. Append-only: callers reading
    the cache rely on "later entry wins" for the same key.
    """

    path = cache_path_for(cache_dir, target_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_dict(), sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    log.info(
        "measurement_cache_append",
        target=target_id,
        workload=entry.key.workload_id,
        lane=entry.key.lane,
        n_iters=entry.stats.n_iters,
        p50_us=entry.stats.p50_us,
    )
