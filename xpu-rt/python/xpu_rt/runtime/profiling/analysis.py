"""Profile analysis — turn raw profiling data into actionable insights.

Reads collected data from profiler adapters and produces structured
analysis: per-tile latency breakdown, roofline plot data, bottleneck
identification, DMA/compute overlap analysis.

The agentic LLM uses this analysis to decide what to optimize next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfileSnapshot, TileMetrics

log = structlog.get_logger()


@dataclass(frozen=True)
class BottleneckInfo:
    """A detected performance bottleneck.

    Attributes:
        region_id: Region where the bottleneck occurs.
        kind: Bottleneck type (``"compute_bound"``, ``"memory_bound"``,
            ``"latency_bound"``, ``"idle"``, ``"dma_stall"``).
        severity: 0.0 (no issue) to 1.0 (critical).
        description: Human-readable description.
        suggested_action: What the LLM should consider doing.
    """

    region_id: str
    kind: str
    severity: float = 0.0
    description: str = ""
    suggested_action: str = ""


@dataclass(frozen=True)
class RooflinePoint:
    """A single point on the roofline model.

    Attributes:
        region_id: Which region this point represents.
        arithmetic_intensity: FLOP/byte ratio.
        achieved_gflops: Measured throughput in GFLOP/s.
        peak_gflops: Device peak throughput in GFLOP/s.
        peak_bandwidth_gbps: Device peak memory bandwidth in GB/s.
        is_compute_bound: Whether this op is compute-bound on the roofline.
    """

    region_id: str
    arithmetic_intensity: float = 0.0
    achieved_gflops: float = 0.0
    peak_gflops: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    is_compute_bound: bool = False


@dataclass
class ProfileAnalysis:
    """Complete analysis of profiling data.

    Attributes:
        bottlenecks: Detected bottlenecks sorted by severity.
        roofline_points: Per-region roofline model points.
        per_region_latency_us: Region ID → total latency in microseconds.
        per_tile_metrics: Region ID → list of tile metrics.
        compute_utilization: Overall compute utilization (0.0-1.0).
        memory_utilization: Overall memory bandwidth utilization (0.0-1.0).
        dma_compute_overlap: Fraction of DMA time overlapped with compute.
        idle_fraction: Fraction of time devices were idle.
        total_latency_us: Total measured wall-clock latency.
        metadata: Additional analysis data.
    """

    bottlenecks: list[BottleneckInfo] = field(default_factory=list)
    roofline_points: list[RooflinePoint] = field(default_factory=list)
    per_region_latency_us: dict[str, float] = field(default_factory=dict)
    per_tile_metrics: dict[str, list[TileMetrics]] = field(default_factory=dict)
    compute_utilization: float = 0.0
    memory_utilization: float = 0.0
    dma_compute_overlap: float = 0.0
    idle_fraction: float = 0.0
    total_latency_us: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_for_llm(self) -> dict[str, Any]:
        """Compact summary suitable for LLM prompt context.

        Returns a dict that can be serialized into a prompt for the
        agentic LLM to reason about what to optimize next.
        """
        return {
            "total_latency_us": round(self.total_latency_us, 2),
            "compute_utilization": round(self.compute_utilization, 3),
            "memory_utilization": round(self.memory_utilization, 3),
            "dma_compute_overlap": round(self.dma_compute_overlap, 3),
            "idle_fraction": round(self.idle_fraction, 3),
            "num_bottlenecks": len(self.bottlenecks),
            "top_bottlenecks": [
                {
                    "region": b.region_id,
                    "kind": b.kind,
                    "severity": round(b.severity, 2),
                    "suggestion": b.suggested_action,
                }
                for b in self.bottlenecks[:5]
            ],
            "hottest_regions": sorted(
                self.per_region_latency_us.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }


class ProfileAnalyzer:
    """Analyzes profiling data to produce actionable insights.

    Args:
        peak_gflops: Device peak compute throughput.
        peak_bandwidth_gbps: Device peak memory bandwidth.
    """

    def __init__(
        self,
        peak_gflops: float = 0.0,
        peak_bandwidth_gbps: float = 0.0,
    ) -> None:
        self._peak_gflops = peak_gflops
        self._peak_bw = peak_bandwidth_gbps

    def analyze(
        self,
        snapshots: list[ProfileSnapshot],
        *,
        region_flops: dict[str, float] | None = None,
        region_bytes: dict[str, float] | None = None,
    ) -> ProfileAnalysis:
        """Analyze a sequence of profiling snapshots.

        Args:
            snapshots: Time-ordered profiling snapshots.
            region_flops: Region ID → total FLOPs (for roofline).
            region_bytes: Region ID → total bytes transferred.

        Returns:
            A complete profile analysis.
        """
        region_flops = region_flops or {}
        region_bytes = region_bytes or {}

        # Aggregate per-tile metrics across snapshots
        per_tile: dict[str, list[TileMetrics]] = {}
        per_region_lat: dict[str, float] = {}

        for snap in snapshots:
            for tm in snap.tile_metrics:
                per_tile.setdefault(tm.region_id, []).append(tm)
                per_region_lat[tm.region_id] = per_region_lat.get(tm.region_id, 0.0) + tm.latency_us

        # Roofline points
        roofline: list[RooflinePoint] = []
        for rid, flops in region_flops.items():
            bytes_moved = region_bytes.get(rid, 1.0)
            ai = flops / bytes_moved if bytes_moved > 0 else 0.0
            latency = per_region_lat.get(rid, 1.0)
            achieved = (flops / 1e9) / (latency / 1e6) if latency > 0 else 0.0

            ridge_point = self._peak_gflops / self._peak_bw if self._peak_bw > 0 else float("inf")

            roofline.append(
                RooflinePoint(
                    region_id=rid,
                    arithmetic_intensity=ai,
                    achieved_gflops=achieved,
                    peak_gflops=self._peak_gflops,
                    peak_bandwidth_gbps=self._peak_bw,
                    is_compute_bound=ai > ridge_point,
                )
            )

        # Bottleneck detection
        bottlenecks = self._detect_bottlenecks(
            per_region_lat,
            roofline,
            per_tile,
        )

        total_lat = sum(per_region_lat.values())

        return ProfileAnalysis(
            bottlenecks=bottlenecks,
            roofline_points=roofline,
            per_region_latency_us=per_region_lat,
            per_tile_metrics=per_tile,
            total_latency_us=total_lat,
        )

    def _detect_bottlenecks(
        self,
        region_latencies: dict[str, float],
        roofline: list[RooflinePoint],
        per_tile: dict[str, list[TileMetrics]],
    ) -> list[BottleneckInfo]:
        """Detect performance bottlenecks from profiling data."""
        bottlenecks: list[BottleneckInfo] = []

        if not region_latencies:
            return bottlenecks

        total = sum(region_latencies.values())
        if total <= 0:
            return bottlenecks

        # Flag regions taking >20% of total as bottlenecks
        for rid, lat in region_latencies.items():
            frac = lat / total
            if frac > 0.2:
                # Check roofline to classify
                rp = next((r for r in roofline if r.region_id == rid), None)
                if rp and rp.is_compute_bound:
                    kind = "compute_bound"
                    suggestion = "Consider tiling or precision reduction"
                else:
                    kind = "memory_bound"
                    suggestion = "Consider fusion or layout optimization"

                bottlenecks.append(
                    BottleneckInfo(
                        region_id=rid,
                        kind=kind,
                        severity=min(frac * 2, 1.0),
                        description=f"Region {rid} takes {frac * 100:.1f}% of total latency",
                        suggested_action=suggestion,
                    )
                )

        bottlenecks.sort(key=lambda b: b.severity, reverse=True)
        return bottlenecks


__all__ = [
    "BottleneckInfo",
    "ProfileAnalysis",
    "ProfileAnalyzer",
    "RooflinePoint",
]
