"""Evolution-driven optimization — population-based LLM search.

The LLM generates candidate optimization strategies. The best survive.
Each "strategy" is a sequence of agent actions. The env's checkpoint/rollback
enables parallel evaluation without interference.

Pipeline:
    1. Initialize: LLM generates N candidate strategies
    2. Evaluate: Run each through the pipeline, measure cost
    3. Select: Keep top-K performers
    4. Mutate: LLM refines winners with context from why they won
    5. Repeat for G generations
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any

import structlog

from compgen.agent.env import (
    Action,
    CheckpointAction,
    CompilerEnv,
    EqSatAction,
    NoopAction,
    RollbackAction,
)
from compgen.llm.base import CompGenLLMProtocol, GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass(frozen=True)
class Strategy:
    """A candidate optimization strategy = sequence of action types."""

    name: str
    action_types: list[str]
    description: str = ""
    generation: int = 0


@dataclass(frozen=True)
class ScoredStrategy:
    """A strategy with its measured performance."""

    strategy: Strategy
    cost_us: float
    improvement_pct: float
    actions_applied: int
    actions_failed: int


@dataclass(frozen=True)
class EvolutionResult:
    """Result of evolutionary optimization."""

    best_strategy: Strategy
    best_cost_us: float
    total_improvement_pct: float
    generations_run: int
    candidates_evaluated: int
    history: list[list[ScoredStrategy]]  # per-generation results


INIT_PROMPT = textwrap.dedent("""\
    You are designing optimization strategies for an ML compiler.

    Model has {op_count} ops, {total_flops:,} FLOPs, {num_devices} device(s).
    Target: {target_name}, objective: minimize {objective}.

    Generate {n} different optimization strategies. Each strategy is a sequence
    of action types to apply in order. Available actions:
    - "eqsat" — run equality saturation (algebraic rewrites)
    - "tile" — apply tiling to compute-heavy ops
    - "fuse" — fuse adjacent ops
    - "assign_device" — place ops on devices
    - "noop" — skip

    Respond as JSON array of strategies:
    [
      {{"name": "strategy_1", "actions": ["eqsat", "tile", "fuse"], "description": "..."}},
      ...
    ]
""")

MUTATE_PROMPT = textwrap.dedent("""\
    Refine these winning strategies for the next generation.

    ## Winners from generation {gen}:
    {winners}

    ## Task
    Create {n} refined variants. Keep what worked, change what didn't.
    Available actions: eqsat, tile, fuse, assign_device, noop.

    Respond as JSON array:
    [
      {{"name": "...", "actions": ["..."], "description": "..."}},
      ...
    ]
""")


@dataclass
class EvolutionaryOptimizer:
    """Population-based optimization where LLM generates candidates."""

    llm_client: CompGenLLMProtocol
    env: CompilerEnv
    population_size: int = 5
    top_k: int = 2
    generations: int = 3

    def evolve(self, target: TargetProfile) -> EvolutionResult:
        """Run evolutionary optimization. Env must be reset first."""
        obs = self.env._make_observation()
        initial_cost = obs.estimated_total_latency_us
        history: list[list[ScoredStrategy]] = []
        best_scored: ScoredStrategy | None = None

        # Generation 0: initialize
        strategies = self._initialize_population(obs, target)

        for gen in range(self.generations):
            # Evaluate all candidates
            scored = self._evaluate(strategies)
            history.append(scored)

            # Track best
            gen_best = min(scored, key=lambda s: s.cost_us)
            if best_scored is None or gen_best.cost_us < best_scored.cost_us:
                best_scored = gen_best

            log.info(
                "evolution.generation",
                gen=gen,
                best_cost=gen_best.cost_us,
                best_name=gen_best.strategy.name,
            )

            if gen < self.generations - 1:
                # Select winners and mutate
                winners = sorted(scored, key=lambda s: s.cost_us)[:self.top_k]
                strategies = self._mutate(winners, gen, target)

        if best_scored is None:
            best_scored = ScoredStrategy(
                strategy=Strategy(name="none", action_types=[]),
                cost_us=initial_cost,
                improvement_pct=0.0,
                actions_applied=0,
                actions_failed=0,
            )

        total_improvement = ((initial_cost - best_scored.cost_us) / max(initial_cost, 1e-9)) * 100
        total_candidates = sum(len(gen) for gen in history)

        return EvolutionResult(
            best_strategy=best_scored.strategy,
            best_cost_us=best_scored.cost_us,
            total_improvement_pct=total_improvement,
            generations_run=len(history),
            candidates_evaluated=total_candidates,
            history=history,
        )

    def _initialize_population(self, obs: Any, target: TargetProfile) -> list[Strategy]:
        """Ask LLM to generate initial population."""
        prompt = INIT_PROMPT.format(
            op_count=len(obs.regions),
            total_flops=obs.total_flops,
            num_devices=obs.num_devices,
            target_name=target.name,
            objective="latency",
            n=self.population_size,
        )

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="", target_profile_summary=target.name,
                    available_transforms=[], kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.7),
            )
            response = self.llm_client.generate(request)
            return self._parse_strategies(response.raw_text, generation=0)
        except Exception:
            # Fallback: default strategies
            return [
                Strategy("eqsat_only", ["eqsat"], "Just eqsat", 0),
                Strategy("eqsat_then_tile", ["eqsat", "tile"], "Eqsat + tile", 0),
            ]

    def _evaluate(self, strategies: list[Strategy]) -> list[ScoredStrategy]:
        """Evaluate each strategy using checkpoint/rollback."""
        results: list[ScoredStrategy] = []
        obs = self.env._make_observation()
        baseline_cost = obs.estimated_total_latency_us

        for strategy in strategies:
            # Checkpoint before evaluation
            self.env.step(CheckpointAction())

            applied = 0
            failed = 0
            for action_type in strategy.action_types:
                action = self._action_type_to_action(action_type)
                result = self.env.step(action)
                if result.info.action_applied:
                    applied += 1
                else:
                    failed += 1

            obs_after = self.env._make_observation()
            cost = obs_after.estimated_total_latency_us
            improvement = ((baseline_cost - cost) / max(baseline_cost, 1e-9)) * 100

            results.append(ScoredStrategy(
                strategy=strategy,
                cost_us=cost,
                improvement_pct=improvement,
                actions_applied=applied,
                actions_failed=failed,
            ))

            # Rollback to try next strategy
            self.env.step(RollbackAction())

        return results

    def _mutate(self, winners: list[ScoredStrategy], gen: int, target: TargetProfile) -> list[Strategy]:
        """Ask LLM to refine winning strategies."""
        winner_desc = "\n".join(
            f"  {w.strategy.name}: {w.strategy.action_types} → {w.improvement_pct:+.1f}%"
            for w in winners
        )
        prompt = MUTATE_PROMPT.format(
            gen=gen, winners=winner_desc, n=self.population_size,
        )

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="", target_profile_summary=target.name,
                    available_transforms=[], kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.7),
            )
            response = self.llm_client.generate(request)
            return self._parse_strategies(response.raw_text, generation=gen + 1)
        except Exception:
            # Keep winners as-is
            return [w.strategy for w in winners]

    def _parse_strategies(self, text: str, generation: int) -> list[Strategy]:
        """Parse strategy JSON from LLM response."""
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(text[start:end])
                return [
                    Strategy(
                        name=item.get("name", f"gen{generation}_{i}"),
                        action_types=item.get("actions", ["noop"]),
                        description=item.get("description", ""),
                        generation=generation,
                    )
                    for i, item in enumerate(items)
                ]
        except (json.JSONDecodeError, ValueError):
            pass
        return [Strategy(f"fallback_gen{generation}", ["eqsat"], "fallback", generation)]

    def _action_type_to_action(self, action_type: str) -> Action:
        """Convert action type string to concrete Action."""
        if action_type == "eqsat":
            return EqSatAction(rule_categories=("algebraic", "fusion"))
        return NoopAction()
