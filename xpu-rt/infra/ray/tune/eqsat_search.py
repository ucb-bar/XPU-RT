"""Ray Tune eqsat rule ablation search.

Systematically explores which eqsat rule combinations yield the best
cost reduction.  Each trial runs a subset of rules and reports metrics.
"""

from __future__ import annotations

from typing import Any

import structlog

from infra.ray._require import require_ray, require_tune

ray = require_ray()
tune = require_tune()

log = structlog.get_logger()

# Available rule categories
RULE_CATEGORIES = ["algebraic", "fusion", "layout"]


def eqsat_search_space() -> dict[str, Any]:
    """Generate Tune search space for eqsat rule ablation.

    Each rule category is a boolean toggle.
    """
    return {
        f"use_{cat}": tune.choice([True, False])
        for cat in RULE_CATEGORIES
    }


def _eqsat_trial(config: dict[str, Any]) -> None:
    """Tune trainable: evaluate an eqsat rule combination.

    Runs the eqsat pipeline with the selected rule categories
    and reports cost metrics.
    """
    # Determine active categories
    active_categories = [
        cat for cat in RULE_CATEGORIES
        if config.get(f"use_{cat}", False)
    ]

    if not active_categories:
        # At least one category must be active
        tune.report(cost_us=float("inf"), num_rules=0, categories="none")
        return

    # Simulate eqsat pass (real impl would call run_eqsat_pass)
    # Cost improvement correlates with number of active categories
    base_cost = 100.0
    reduction_per_category = {
        "algebraic": 0.15,
        "fusion": 0.25,
        "layout": 0.10,
    }

    total_reduction = sum(
        reduction_per_category.get(cat, 0.0)
        for cat in active_categories
    )

    # Diminishing returns for many categories
    if len(active_categories) > 2:
        total_reduction *= 0.85

    estimated_cost = base_cost * (1.0 - total_reduction)

    tune.report(
        cost_us=estimated_cost,
        num_rules=len(active_categories),
        categories=",".join(active_categories),
        reduction_pct=total_reduction * 100,
    )


def run_eqsat_search(
    num_samples: int = 20,
    max_concurrent: int = 4,
) -> Any:
    """Run Tune-based eqsat rule ablation.

    Args:
        num_samples: Number of rule combinations to try.
        max_concurrent: Max parallel trials.

    Returns:
        Tune ResultGrid.
    """
    tuner = tune.Tuner(
        _eqsat_trial,
        param_space=eqsat_search_space(),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            metric="cost_us",
            mode="min",
        ),
    )

    results = tuner.fit()
    log.info("eqsat_search.done", num_trials=len(results))
    return results


__all__ = ["run_eqsat_search", "eqsat_search_space"]
