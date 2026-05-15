"""Profiler adapter protocol.

Defines the contract that every profiler backend must implement.
The registry discovers and instantiates adapters based on the
``ProfilingSpec`` declared in the hardware spec.

The agentic LLM can register custom adapters at runtime via
:func:`~compgen.runtime.profiling.registry.register_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TileMetrics:
    """Performance metrics for a single tile execution.

    Attributes:
        region_id: The region this tile belongs to.
        tile_index: Tile index within the region.
        latency_us: Execution latency in microseconds.
        compute_utilization: Fraction of peak compute used (0.0-1.0).
        memory_bandwidth_gbps: Achieved memory bandwidth in GB/s.
        counters: Raw hardware counter values.
    """

    region_id: str
    tile_index: int = 0
    latency_us: float = 0.0
    compute_utilization: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileSnapshot:
    """A point-in-time snapshot of profiling data.

    Attributes:
        timestamp_us: When this snapshot was taken.
        counters: Hardware counter values.
        tile_metrics: Per-tile metrics (if tile-level profiling is active).
        metadata: Additional backend-specific data.
    """

    timestamp_us: float = 0.0
    counters: dict[str, float] = field(default_factory=dict)
    tile_metrics: list[TileMetrics] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProfilerAdapter(Protocol):
    """Protocol for profiler backend adapters.

    Each target gets its own adapter implementation that knows how
    to interact with the target's profiling hardware/software.
    """

    @property
    def name(self) -> str:
        """Adapter name (e.g., ``"linux_perf"``, ``"zephyr_trace"``)."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether the adapter is currently collecting data."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the adapter.

        Args:
            config: Backend-specific configuration (counters to enable,
                sample rates, output paths, etc.).
        """
        ...

    def start(self) -> None:
        """Start collecting profiling data."""
        ...

    def stop(self) -> None:
        """Stop collecting and finalize data."""
        ...

    def read_counters(self) -> dict[str, float]:
        """Read current hardware counter values.

        Returns:
            Dict of counter name → value.
        """
        ...

    def get_tile_breakdown(self, region_id: str) -> list[TileMetrics]:
        """Get per-tile metrics for a region.

        Args:
            region_id: The region to query.

        Returns:
            List of tile metrics, one per tile.
        """
        ...

    def export_trace(self, path: str) -> None:
        """Export collected trace data to a file.

        Args:
            path: Output file path.
        """
        ...

    def snapshot(self) -> ProfileSnapshot:
        """Take a point-in-time snapshot of all profiling data."""
        ...


__all__ = [
    "ProfileSnapshot",
    "ProfilerAdapter",
    "TileMetrics",
]
