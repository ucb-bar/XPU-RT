"""Ray Tune tile-size search.

Uses Tune's search algorithms to find optimal tile sizes for
compute-heavy operations based on target hardware characteristics.
"""

from __future__ import annotations

from typing import Any

import structlog

from infra.ray._require import require_ray, require_tune

ray = require_ray()
tune = require_tune()

log = structlog.get_logger()


def tile_search_space(
    op_type: str,
    max_tile_dim: int = 256,
) -> dict[str, Any]:
    """Generate Tune search space for tile sizes.

    Args:
        op_type: Operation type (e.g., ``"matmul"``, ``"conv2d"``).
        max_tile_dim: Maximum tile dimension to search.

    Returns:
        Dict suitable for ``tune.run(config=...)``.
    """
    powers = [2**i for i in range(3, 9) if 2**i <= max_tile_dim]

    if op_type in ("matmul", "batch_matmul"):
        return {
            "tile_m": tune.choice(powers),
            "tile_n": tune.choice(powers),
            "tile_k": tune.choice(powers),
            "op_type": op_type,
        }

    if op_type in ("conv2d",):
        return {
            "tile_oc": tune.choice(powers),
            "tile_oh": tune.choice([1, 2, 4, 8, 16]),
            "tile_ow": tune.choice([1, 2, 4, 8, 16]),
            "op_type": op_type,
        }

    # Generic: 2D tiling
    return {
        "tile_x": tune.choice(powers),
        "tile_y": tune.choice(powers),
        "op_type": op_type,
    }


def _tile_trial(config: dict[str, Any]) -> None:
    """Tune trainable: evaluate a tile configuration.

    Reports estimated latency based on a simple roofline model.
    """
    op_type = config.get("op_type", "matmul")

    # Extract tile sizes
    if op_type in ("matmul", "batch_matmul"):
        tile_m = config.get("tile_m", 64)
        tile_n = config.get("tile_n", 64)
        tile_k = config.get("tile_k", 64)
        tile_volume = tile_m * tile_n * tile_k
        tile_memory = (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) * 2  # FP16
    else:
        tile_x = config.get("tile_x", 64)
        tile_y = config.get("tile_y", 64)
        tile_volume = tile_x * tile_y
        tile_memory = tile_x * tile_y * 4  # FP32

    # Simple cost model: prefer tiles that fit in L1 cache
    l1_size = 65536  # 64 KB typical
    fits_l1 = tile_memory <= l1_size
    estimated_latency_us = tile_volume / 1e6  # simplistic

    if not fits_l1:
        estimated_latency_us *= 3.0  # penalty for L1 miss

    tune.report(
        latency_us=estimated_latency_us,
        tile_volume=tile_volume,
        tile_memory=tile_memory,
        fits_l1=fits_l1,
    )


def run_tile_search(
    op_type: str,
    num_samples: int = 20,
    max_concurrent: int = 4,
    max_tile_dim: int = 256,
) -> Any:
    """Run Tune-based tile-size search.

    Args:
        op_type: Operation type to search for.
        num_samples: Number of tile configs to try.
        max_concurrent: Max parallel trials.
        max_tile_dim: Maximum tile dimension.

    Returns:
        Tune ResultGrid.
    """
    search_space = tile_search_space(op_type, max_tile_dim)

    tuner = tune.Tuner(
        _tile_trial,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            metric="latency_us",
            mode="min",
        ),
    )

    results = tuner.fit()
    log.info("tile_search.done", op_type=op_type, num_trials=len(results))
    return results


__all__ = ["run_tile_search", "tile_search_space"]
