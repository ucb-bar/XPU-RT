"""Placement group templates for common CompGen workloads.

These templates define resource bundles for Ray placement groups,
enabling co-location and hardware-aware scheduling.
"""

from __future__ import annotations

from typing import Any


def compile_placement(num_workers: int = 4) -> dict[str, Any]:
    """Placement template for parallel compilation jobs.

    Spreads workers across nodes for maximum parallelism.
    """
    return {
        "strategy": "SPREAD",
        "bundles": [{"CPU": 2}] * num_workers,
    }


def benchmark_gpu_placement() -> dict[str, Any]:
    """Placement template for GPU benchmark jobs.

    Packs onto a single node with GPU.
    """
    return {
        "strategy": "STRICT_PACK",
        "bundles": [{"CPU": 1, "GPU": 1}],
    }


def tune_search_placement(num_trials: int = 8) -> dict[str, Any]:
    """Placement template for Tune search experiments.

    Spreads trials across available workers.
    """
    return {
        "strategy": "SPREAD",
        "bundles": [{"CPU": 1}] * num_trials,
    }


def hardware_test_placement(custom_resource: str) -> dict[str, Any]:
    """Placement template for testing on specific hardware.

    Requires a node with the specified custom resource.

    Args:
        custom_resource: Custom resource name (e.g., ``"fpga_xilinx"``).
    """
    return {
        "strategy": "STRICT_PACK",
        "bundles": [{"CPU": 1, custom_resource: 1.0}],
    }


def distributed_compile_placement(
    num_cpu_workers: int = 4,
    num_gpu_workers: int = 1,
) -> dict[str, Any]:
    """Placement template for distributed compilation with GPU acceleration.

    CPU workers for IR transforms, GPU workers for kernel search.
    """
    bundles: list[dict[str, Any]] = []
    bundles.extend([{"CPU": 2}] * num_cpu_workers)
    bundles.extend([{"CPU": 1, "GPU": 1}] * num_gpu_workers)
    return {
        "strategy": "SPREAD",
        "bundles": bundles,
    }


__all__ = [
    "benchmark_gpu_placement",
    "compile_placement",
    "distributed_compile_placement",
    "hardware_test_placement",
    "tune_search_placement",
]
