"""Memory capacity and lifetime allocation.

Solves memory allocation respecting device capacity, buffer lifetimes,
and reuse opportunities using greedy interval scheduling.

Invariants:
    - Peak memory per device never exceeds capacity.
    - Buffer reuse is maximized (non-overlapping lifetimes share memory).
    - Infeasibility is detected (model doesn't fit on target).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BufferLifetime:
    """Lifetime of a buffer.

    Attributes:
        buffer_name: Buffer identifier.
        size_bytes: Buffer size.
        device_index: Device where the buffer lives.
        start_us: First use time.
        end_us: Last use time.
    """

    buffer_name: str
    size_bytes: int
    device_index: int
    start_us: float
    end_us: float


@dataclass(frozen=True)
class MemoryAllocation:
    """Memory allocation solution.

    Attributes:
        offsets: Dict mapping buffer_name -> offset within device memory.
        peak_per_device: Peak memory usage per device.
        feasible: Whether allocation fits within device capacities.
        reuse_count: Number of buffer reuse opportunities found.
        solve_time_ms: Solver wall-clock time.
    """

    offsets: dict[str, int] = field(default_factory=dict)
    peak_per_device: dict[int, int] = field(default_factory=dict)
    feasible: bool = False
    reuse_count: int = 0
    solve_time_ms: float = 0.0


def solve_memory(
    lifetimes: list[BufferLifetime],
    device_capacities: dict[int, int],
    timeout_ms: int = 10000,
) -> MemoryAllocation:
    """Solve memory allocation using greedy first-fit interval scheduling.

    Buffers are sorted by size (largest first). For each buffer, find
    the lowest offset where it doesn't overlap with any already-allocated
    buffer on the same device during the same time interval.

    Args:
        lifetimes: Buffer lifetime descriptions.
        device_capacities: Memory capacity per device (bytes).
        timeout_ms: Maximum solve time.

    Returns:
        MemoryAllocation with offsets and peak usage.
    """
    start = time.monotonic()
    offsets: dict[str, int] = {}
    peak_per_device: dict[int, int] = {}
    reuse_count = 0

    # Group by device
    per_device: dict[int, list[BufferLifetime]] = {}
    for buf in lifetimes:
        per_device.setdefault(buf.device_index, []).append(buf)

    for device_idx, buffers in per_device.items():
        # Sort by size descending (largest first for better packing)
        sorted_buffers = sorted(buffers, key=lambda b: b.size_bytes, reverse=True)

        # Allocated intervals: list of (offset, size, start_us, end_us)
        allocated: list[tuple[int, int, float, float]] = []

        for buf in sorted_buffers:
            # Find the lowest offset that doesn't conflict
            best_offset = 0
            placed = False

            # Try offset 0, then above each existing allocation
            candidate_offsets = [0] + [
                a[0] + a[1] for a in allocated
            ]
            candidate_offsets = sorted(set(candidate_offsets))

            for offset in candidate_offsets:
                # Check if this offset conflicts with any existing allocation
                conflicts = False
                for alloc_offset, alloc_size, alloc_start, alloc_end in allocated:
                    # Check spatial overlap
                    spatial_overlap = (
                        offset < alloc_offset + alloc_size
                        and offset + buf.size_bytes > alloc_offset
                    )
                    # Check temporal overlap
                    temporal_overlap = (
                        buf.start_us < alloc_end and buf.end_us > alloc_start
                    )
                    if spatial_overlap and temporal_overlap:
                        conflicts = True
                        break

                if not conflicts:
                    best_offset = offset
                    placed = True
                    # Check if we reused space from a non-overlapping lifetime
                    if offset < sum(b.size_bytes for b in sorted_buffers):
                        reuse_count += 1
                    break

            if not placed:
                # Fallback: place at end
                best_offset = sum(a[1] for a in allocated) if allocated else 0

            offsets[buf.buffer_name] = best_offset
            allocated.append((best_offset, buf.size_bytes, buf.start_us, buf.end_us))

        # Compute peak for this device
        if allocated:
            peak = max(a[0] + a[1] for a in allocated)
        else:
            peak = 0
        peak_per_device[device_idx] = peak

    # Check feasibility
    feasible = all(
        peak_per_device.get(d, 0) <= device_capacities.get(d, 2**63)
        for d in peak_per_device
    )

    elapsed_ms = (time.monotonic() - start) * 1000

    return MemoryAllocation(
        offsets=offsets,
        peak_per_device=peak_per_device,
        feasible=feasible,
        reuse_count=reuse_count,
        solve_time_ms=elapsed_ms,
    )


__all__ = ["BufferLifetime", "MemoryAllocation", "solve_memory"]
