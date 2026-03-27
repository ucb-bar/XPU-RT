"""Shared target profile extraction utilities.

Provides helper functions for extracting device memory capacities and
transfer cost matrices from TargetProfile objects. Used by both the
execution planner and the solver contracts module.
"""

from __future__ import annotations

from compgen.targets.schema import TargetProfile

_DEFAULT_MEMORY_BYTES = 8 * 1024**3  # 8 GB
_DEFAULT_TRANSFER_COST = 0.001  # us per byte


def extract_device_memory(target: TargetProfile) -> list[int]:
    """Extract per-device memory capacities from a target profile.

    Tries ``capacity_bytes`` first (for forward-compat), then falls
    back to ``size_bytes`` on each memory level.

    Args:
        target: Target hardware profile.

    Returns:
        List of memory capacities in bytes, one per device.
    """
    device_memory: list[int] = []
    for device in target.devices:
        if hasattr(device, "memory_hierarchy") and device.memory_hierarchy:
            total = sum(
                level.capacity_bytes
                for level in device.memory_hierarchy
                if hasattr(level, "capacity_bytes")
            )
            if total <= 0:
                total = sum(
                    level.size_bytes
                    for level in device.memory_hierarchy
                    if hasattr(level, "size_bytes")
                )
            device_memory.append(total if total > 0 else _DEFAULT_MEMORY_BYTES)
        else:
            device_memory.append(_DEFAULT_MEMORY_BYTES)
    return device_memory


def extract_transfer_cost_matrix(target: TargetProfile) -> dict[tuple[int, int], float]:
    """Build a transfer cost matrix from a target profile's interconnects.

    Returns:
        Mapping of ``(src_device, dst_device) -> cost_per_byte_us``.
        Self-transfers are 0.0; missing pairs default to ``_DEFAULT_TRANSFER_COST``.
    """
    num_devices = len(target.devices)
    matrix: dict[tuple[int, int], float] = {}
    for i in range(num_devices):
        for j in range(num_devices):
            matrix[(i, j)] = 0.0 if i == j else _DEFAULT_TRANSFER_COST

    for interconnect in target.interconnects:
        if hasattr(interconnect, "bandwidth_gbps") and interconnect.bandwidth_gbps > 0:
            cost_per_byte = 1.0 / (interconnect.bandwidth_gbps * 1e9) * 1e6
            if hasattr(interconnect, "devices") and interconnect.devices:
                i, j = interconnect.devices
                matrix[(i, j)] = cost_per_byte
                matrix[(j, i)] = cost_per_byte

    return matrix


__all__ = ["extract_device_memory", "extract_transfer_cost_matrix"]
