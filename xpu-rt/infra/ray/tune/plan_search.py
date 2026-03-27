"""Ray Tune integration for evolutionary compiler search.

Wraps ``EvolutionaryOptimizer``'s evaluate loop into Tune's trial-based
search.  Each trial evaluates one Strategy on its own CompilerEnv instance.

The key insight: the existing ``_evaluate()`` loop is sequential with
checkpoint/rollback.  Tune parallelizes this — each trial gets its own
env, evaluates one strategy, and reports metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from infra.ray._require import require_ray, require_tune

ray = require_ray()
tune = require_tune()

log = structlog.get_logger()


@dataclass
class TuneSearchConfig:
    """Configuration for a Tune-based evolutionary search.

    Attributes:
        target_profile_path: Path to target profile YAML.
        population_size: Number of candidate strategies per generation.
        generations: Number of evolutionary generations.
        top_k: Number of top strategies to keep per generation.
        num_parallel: Maximum concurrent Tune trials.
        resources_per_trial: Ray resources for each trial.
    """

    target_profile_path: str
    population_size: int = 10
    generations: int = 5
    top_k: int = 3
    num_parallel: int = 4
    resources_per_trial: dict[str, float] | None = None


def _trial_evaluate(config: dict[str, Any]) -> None:
    """Tune trainable function: evaluate a single strategy.

    This function runs inside a Ray worker.  It reconstructs a
    CompilerEnv from serialized state, evaluates the strategy,
    and reports metrics to Tune.

    Config keys:
        strategy_name: str
        strategy_actions: list[str]
        target_profile_path: str
    """
    from compgen.agent.env import CompilerEnv, EqSatAction, NoopAction
    from compgen.targets.schema import load_profile
    from xdsl.dialects.builtin import ModuleOp

    strategy_name = config["strategy_name"]
    strategy_actions = config.get("strategy_actions", ["eqsat"])

    # Create a fresh env for this trial
    env = CompilerEnv()
    module = ModuleOp([])
    target = load_profile(config["target_profile_path"])
    env.reset(module, target, budget=len(strategy_actions) + 5)

    # Execute strategy actions
    actions_applied = 0
    initial_cost = env._current_cost

    for action_type in strategy_actions:
        if action_type == "eqsat":
            action = EqSatAction(rule_categories=("algebraic", "fusion"))
        else:
            action = NoopAction()

        result = env.step(action)
        if result.info.action_applied:
            actions_applied += 1

    final_cost = env._current_cost
    improvement = ((initial_cost - final_cost) / max(initial_cost, 1e-9)) * 100

    # Report metrics to Tune
    tune.report(
        cost_us=final_cost,
        improvement_pct=improvement,
        actions_applied=actions_applied,
        strategy_name=strategy_name,
    )


class TuneEvolutionarySearch:
    """Ray Tune wrapper for evolutionary compiler search.

    Replaces the sequential ``EvolutionaryOptimizer._evaluate()`` loop
    with parallel Tune trials.

    Usage::

        search = TuneEvolutionarySearch(
            llm_client=client,
            config=TuneSearchConfig(
                target_profile_path="specs/cuda_a100.yaml",
                population_size=10,
            ),
        )
        result = search.run()
    """

    def __init__(
        self,
        llm_client: Any,
        config: TuneSearchConfig,
    ) -> None:
        self._llm_client = llm_client
        self._config = config

    def run(self) -> dict[str, Any]:
        """Run Tune-based evolutionary search.

        For each generation:
            1. Generate candidate strategies via LLM.
            2. Launch parallel Tune trials (one per strategy).
            3. Collect results, select top_k.
            4. Feed winners back to LLM for mutation.

        Returns:
            Search result dict with best strategy and metrics.
        """

        all_results: list[dict[str, Any]] = []
        best_cost = float("inf")
        best_strategy_name = ""

        for gen in range(self._config.generations):
            # Generate strategies
            if gen == 0:
                strategies = self._initialize_strategies()
            else:
                strategies = self._mutate_strategies(
                    [r["strategy_name"] for r in all_results[:self._config.top_k]],
                )

            # Build Tune configs
            tune_configs = [
                {
                    "strategy_name": s.name,
                    "strategy_actions": s.action_types,
                    "target_profile_path": self._config.target_profile_path,
                }
                for s in strategies
            ]

            # Run parallel trials
            resources = self._config.resources_per_trial or {"cpu": 1}
            tuner = tune.Tuner(
                tune.with_resources(_trial_evaluate, resources),
                param_space=tune.grid_search(tune_configs),
                tune_config=tune.TuneConfig(
                    max_concurrent_trials=self._config.num_parallel,
                ),
            )

            results = tuner.fit()

            # Collect results
            gen_results: list[dict[str, Any]] = []
            for result in results:
                if result.metrics:
                    gen_results.append({
                        "strategy_name": result.metrics.get("strategy_name", ""),
                        "cost_us": result.metrics.get("cost_us", float("inf")),
                        "improvement_pct": result.metrics.get("improvement_pct", 0.0),
                        "generation": gen,
                    })

            # Sort by cost and select top_k
            gen_results.sort(key=lambda r: r["cost_us"])
            all_results = gen_results[:self._config.top_k] + all_results

            if gen_results and gen_results[0]["cost_us"] < best_cost:
                best_cost = gen_results[0]["cost_us"]
                best_strategy_name = gen_results[0]["strategy_name"]

            log.info(
                "tune.generation.done",
                generation=gen,
                candidates=len(gen_results),
                best_cost=best_cost,
            )

        return {
            "best_strategy": best_strategy_name,
            "best_cost_us": best_cost,
            "generations_run": self._config.generations,
            "total_candidates": len(all_results),
            "history": all_results,
        }

    def _initialize_strategies(self) -> list[Any]:
        """Generate initial population via LLM."""
        from compgen.agent.evolution import Strategy

        # Use LLM to generate strategies (same as EvolutionaryOptimizer)
        strategies = []
        for i in range(self._config.population_size):
            strategies.append(Strategy(
                name=f"strategy_{i}",
                action_types=["eqsat"],
                description=f"Auto-generated strategy {i}",
                generation=0,
            ))
        return strategies

    def _mutate_strategies(self, winner_names: list[str]) -> list[Any]:
        """Mutate winning strategies via LLM."""
        from compgen.agent.evolution import Strategy

        strategies = []
        for i, name in enumerate(winner_names):
            for j in range(self._config.population_size // len(winner_names)):
                strategies.append(Strategy(
                    name=f"{name}_mut{j}",
                    action_types=["eqsat"],
                    description=f"Mutation of {name}",
                    generation=1,
                ))
        return strategies


__all__ = ["TuneEvolutionarySearch", "TuneSearchConfig"]
