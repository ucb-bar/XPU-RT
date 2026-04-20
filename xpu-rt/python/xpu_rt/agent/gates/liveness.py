"""Liveness / memory-footprint gate — wraps solve_memory.

Context::

    ctx = {
        "lifetimes": list[BufferLifetime],    # required
        "device_capacities": dict[int, int],  # required
        "timeout_ms": 10000,                  # optional
    }

Returns ``accepted`` if the greedy allocator fits all buffers within
device capacities; otherwise ``rejected`` with the first overflowing
device reported.
"""

from __future__ import annotations

from typing import Any


def liveness_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    lifetimes = ctx.get("lifetimes")
    device_capacities = ctx.get("device_capacities")
    if lifetimes is None or device_capacities is None:
        return {
            "status": "deferred",
            "details": {
                "reason": "liveness_gate requires ctx.lifetimes + ctx.device_capacities",
            },
        }

    timeout_ms = int(ctx.get("timeout_ms", 10_000))

    try:
        from compgen.solve.memory import solve_memory
    except ImportError as e:  # pragma: no cover
        return {
            "status": "deferred",
            "details": {"reason": f"compgen.solve.memory unavailable: {e}"},
        }

    try:
        allocation = solve_memory(lifetimes, device_capacities, timeout_ms=timeout_ms)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "rejected",
            "details": {
                "reason": "solve_memory raised",
                "error": f"{type(e).__name__}: {e}",
            },
        }

    if not allocation.feasible:
        # Find the first device that overflowed.
        overflow: dict[int, dict[str, int]] = {}
        for dev, peak in allocation.peak_per_device.items():
            cap = device_capacities.get(dev, 0)
            if peak > cap:
                overflow[dev] = {"peak": int(peak), "capacity": int(cap)}
        return {
            "status": "rejected",
            "details": {
                "reason": "memory allocation infeasible",
                "overflow_by_device": overflow,
                "reuse_count": allocation.reuse_count,
                "solve_time_ms": allocation.solve_time_ms,
            },
        }

    return {
        "status": "accepted",
        "details": {
            "peak_per_device": dict(allocation.peak_per_device),
            "reuse_count": allocation.reuse_count,
            "solve_time_ms": allocation.solve_time_ms,
        },
    }


__all__ = ["liveness_gate"]
