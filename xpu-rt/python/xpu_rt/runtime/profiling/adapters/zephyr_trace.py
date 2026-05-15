"""Zephyr RTOS tracing adapter.

Generates configuration for Zephyr's built-in tracing subsystem
and reads back trace data.  This is primarily a code-generation
driver — the real tracing happens on the target hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfileSnapshot, TileMetrics

log = structlog.get_logger()


@dataclass
class ZephyrTraceAdapter:
    """Adapter for Zephyr RTOS tracing.

    Generates Kconfig and DTS overlay entries for Zephyr's tracing
    subsystem.  At the Python level, operates as a configuration
    driver and trace reader — the actual instrumentation runs on
    the embedded target.
    """

    _active: bool = False
    _counters: list[str] = field(default_factory=list)
    _trace_backend: str = "ram"
    _trace_format: str = "ctf"
    _config: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "zephyr_trace"

    @property
    def is_active(self) -> bool:
        return self._active

    def configure(self, config: dict[str, Any]) -> None:
        self._counters = config.get("counters", ["cycles", "thread_switches"])
        self._trace_backend = config.get("trace_backend", "ram")
        self._trace_format = config.get("trace_format", "ctf")
        self._config = config
        log.debug("zephyr_trace.configured", backend=self._trace_backend, format=self._trace_format)

    def start(self) -> None:
        self._active = True
        log.debug("zephyr_trace.started")

    def stop(self) -> None:
        self._active = False
        log.debug("zephyr_trace.stopped")

    def read_counters(self) -> dict[str, float]:
        return {c: 0.0 for c in self._counters}

    def get_tile_breakdown(self, region_id: str) -> list[TileMetrics]:
        return []

    def export_trace(self, path: str) -> None:
        log.info("zephyr_trace.export", path=path, format=self._trace_format)

    def snapshot(self) -> ProfileSnapshot:
        return ProfileSnapshot(
            counters=self.read_counters(),
            metadata={
                "backend": "zephyr_trace",
                "trace_backend": self._trace_backend,
                "trace_format": self._trace_format,
            },
        )

    def kconfig_overrides(self) -> dict[str, str]:
        """Generate Zephyr Kconfig entries for tracing.

        Returns:
            Dict of CONFIG_* → value for prj.conf.
        """
        kconfig: dict[str, str] = {
            "CONFIG_TRACING": "y",
            "CONFIG_TIMING_FUNCTIONS": "y",
            "CONFIG_THREAD_MONITOR": "y",
        }

        backend_map = {
            "ram": "CONFIG_TRACING_BACKEND_RAM",
            "uart": "CONFIG_TRACING_BACKEND_UART",
            "usb": "CONFIG_TRACING_BACKEND_USB",
            "posix": "CONFIG_TRACING_BACKEND_POSIX",
        }
        backend_key = backend_map.get(self._trace_backend, "CONFIG_TRACING_BACKEND_RAM")
        kconfig[backend_key] = "y"

        if self._trace_format == "ctf":
            kconfig["CONFIG_TRACING_CTF"] = "y"
        elif self._trace_format == "sysview":
            kconfig["CONFIG_SEGGER_SYSTEMVIEW"] = "y"

        return kconfig


__all__ = ["ZephyrTraceAdapter"]
