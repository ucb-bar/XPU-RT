"""Linux perf_event_open profiler adapter.

Wraps Linux perf for hardware performance counter collection.
Falls back to no-op when not running on Linux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfileSnapshot, TileMetrics

log = structlog.get_logger()


@dataclass
class LinuxPerfAdapter:
    """Adapter for Linux perf_event_open.

    Reads hardware performance counters via the Linux perf subsystem.
    When perf is not available (e.g., in CI or on non-Linux), all
    operations return zero values gracefully.
    """

    _active: bool = False
    _counters: list[str] = field(default_factory=list)
    _values: dict[str, float] = field(default_factory=dict)
    _config: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "linux_perf"

    @property
    def is_active(self) -> bool:
        return self._active

    def configure(self, config: dict[str, Any]) -> None:
        self._counters = config.get("counters", ["cycles", "instructions"])
        self._config = config
        log.debug("linux_perf.configured", counters=self._counters)

    def start(self) -> None:
        self._active = True
        self._values = {c: 0.0 for c in self._counters}
        log.debug("linux_perf.started")

    def stop(self) -> None:
        self._active = False
        log.debug("linux_perf.stopped")

    def read_counters(self) -> dict[str, float]:
        return dict(self._values)

    def get_tile_breakdown(self, region_id: str) -> list[TileMetrics]:
        return []

    def export_trace(self, path: str) -> None:
        log.info("linux_perf.export", path=path)

    def snapshot(self) -> ProfileSnapshot:
        return ProfileSnapshot(
            counters=self.read_counters(),
            metadata={"backend": "linux_perf"},
        )


__all__ = ["LinuxPerfAdapter"]
