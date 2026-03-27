"""CUDA profiler adapter (nsys / ncu / CUPTI).

Generates NVTX annotation code and launch wrappers for NVIDIA
profiling tools.  Actual profiling requires CUDA toolkit on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfileSnapshot, TileMetrics

log = structlog.get_logger()


@dataclass
class CudaProfilerAdapter:
    """Adapter for NVIDIA CUDA profiling tools.

    Supports:
        - Nsight Systems (nsys) — system-wide trace
        - Nsight Compute (ncu) — per-kernel analysis
        - CUPTI — programmatic counter collection
        - NVTX — annotation markers in generated code
    """

    _active: bool = False
    _tool: str = "nsys"
    _counters: list[str] = field(default_factory=list)
    _nvtx_ranges: list[str] = field(default_factory=list)
    _config: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "cuda_profiler"

    @property
    def is_active(self) -> bool:
        return self._active

    def configure(self, config: dict[str, Any]) -> None:
        self._tool = config.get("tool", "nsys")
        self._counters = config.get("counters", ["sm_active", "dram_read", "dram_write"])
        self._config = config
        log.debug("cuda_profiler.configured", tool=self._tool)

    def start(self) -> None:
        self._active = True
        log.debug("cuda_profiler.started", tool=self._tool)

    def stop(self) -> None:
        self._active = False
        log.debug("cuda_profiler.stopped")

    def read_counters(self) -> dict[str, float]:
        return {c: 0.0 for c in self._counters}

    def get_tile_breakdown(self, region_id: str) -> list[TileMetrics]:
        return []

    def export_trace(self, path: str) -> None:
        log.info("cuda_profiler.export", path=path, tool=self._tool)

    def snapshot(self) -> ProfileSnapshot:
        return ProfileSnapshot(
            counters=self.read_counters(),
            metadata={"backend": "cuda_profiler", "tool": self._tool},
        )

    def launch_command(self, executable: str) -> str:
        """Generate the shell command to launch with profiling.

        Args:
            executable: The binary to profile.

        Returns:
            Shell command string.
        """
        if self._tool == "nsys":
            return f"nsys profile --trace=cuda,nvtx --output=profile {executable}"
        if self._tool == "ncu":
            metrics = ",".join(self._counters) if self._counters else "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            return f"ncu --metrics {metrics} {executable}"
        return executable

    def nvtx_annotation_code(self, name: str) -> dict[str, str]:
        """Generate NVTX C annotation code for a named range.

        Args:
            name: Range name (e.g., kernel name).

        Returns:
            Dict with ``"begin"`` and ``"end"`` C code snippets.
        """
        return {
            "begin": f'nvtxRangePushA("{name}");',
            "end": "nvtxRangePop();",
            "include": "#include <nvToolsExt.h>",
        }


__all__ = ["CudaProfilerAdapter"]
