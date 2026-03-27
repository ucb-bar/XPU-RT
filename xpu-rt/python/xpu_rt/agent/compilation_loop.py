"""Agentic compilation loop — LLM-driven iterative optimization.

The core thesis: an LLM agent iteratively optimizes a model by analyzing,
proposing, applying, verifying, profiling, and refining compilation decisions.

Pipeline:
    1. Analyze: NetworkAnalyzer → pattern clusters + bottlenecks
    2. Propose: LLM suggests optimization (via prompts/)
    3. Apply: Execute in CompilerEnv (validated before execution)
    4. Verify: Check correctness (structural + differential)
    5. Profile: Benchmark on real hardware (optional)
    6. Decide: Accept/reject based on cost improvement
    7. Refine: Feed results back to LLM for next iteration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.agent.env import (
    Action,
    CompilerEnv,
    ConfigureDispatchAction,
    ConfigureProfilingAction,
    EqSatAction,
    GenerateRuntimeHooksAction,
    NoopAction,
    Observation,
)
from compgen.agent.prompts.analyze import AnalysisContext, ProposedOptimization
from compgen.agent.prompts.analyze import format_prompt as fmt_analyze
from compgen.agent.prompts.analyze import parse_response as parse_analyze
from compgen.agent.prompts.refine import RefinementAction, RefinementContext
from compgen.agent.prompts.refine import format_prompt as fmt_refine
from compgen.agent.prompts.refine import parse_response as parse_refine
from compgen.llm.base import CompGenLLMProtocol, GenerationRequest, LLMConfig, Objective, PromptContext
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass(frozen=True)
class IterationRecord:
    """Record of one optimization iteration."""

    iteration: int
    action_type: str
    target: str
    applied: bool
    cost_before_us: float
    cost_after_us: float
    improvement_pct: float
    reasoning: str


@dataclass(frozen=True)
class CompilationResult:
    """Result of the full agentic compilation loop."""

    initial_cost_us: float
    final_cost_us: float
    total_improvement_pct: float
    iterations_run: int
    iterations_improved: int
    history: list[IterationRecord]
    best_observation: Observation | None = None
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticCompilationLoop:
    """LLM-driven compilation with iterative refinement.

    Attributes:
        llm_client: LLM client (real or mock).
        env: CompilerEnv instance (must be reset before run()).
        budget: Maximum optimization iterations.
        min_improvement_pct: Stop if improvement below this threshold.
    """

    llm_client: CompGenLLMProtocol
    env: CompilerEnv
    budget: int = 10
    min_improvement_pct: float = 0.1

    def run(self, target: TargetProfile) -> CompilationResult:
        """Run the full agentic compilation loop.

        The env must already be reset with a module + target before calling.
        """
        history: list[IterationRecord] = []
        obs = self.env._make_observation()
        initial_cost = obs.estimated_total_latency_us
        best_cost = initial_cost
        best_obs = obs
        no_improvement_count = 0

        log.info("agentic.start", initial_cost=initial_cost, budget=self.budget)

        # Step 1: Initial analysis
        proposals = self._analyze(obs, target)

        for iteration in range(self.budget):
            # Step 2: Get next action
            if iteration < len(proposals):
                action = self._proposal_to_action(proposals[iteration])
            else:
                action = self._ask_llm_for_refinement(obs, history, target)

            if action is None or isinstance(action, NoopAction):
                log.info("agentic.stop", reason="LLM suggested noop", iteration=iteration)
                break

            # Step 3: Apply action
            cost_before = obs.estimated_total_latency_us
            result = self.env.step(action)
            obs = result.observation
            cost_after = obs.estimated_total_latency_us

            improvement = ((cost_before - cost_after) / max(cost_before, 1e-9)) * 100

            record = IterationRecord(
                iteration=iteration,
                action_type=action.action_type,
                target=action.region_id,
                applied=result.info.action_applied,
                cost_before_us=cost_before,
                cost_after_us=cost_after,
                improvement_pct=improvement,
                reasoning=getattr(action, "description", ""),
            )
            history.append(record)

            if cost_after < best_cost:
                best_cost = cost_after
                best_obs = obs
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            log.info(
                "agentic.iteration",
                iteration=iteration,
                action=action.action_type,
                improvement=f"{improvement:+.1f}%",
                cost=cost_after,
            )

            # Early stop if no improvement for 3 consecutive iterations
            if no_improvement_count >= 3:
                log.info("agentic.stop", reason="no improvement for 3 iterations")
                break

        total_improvement = ((initial_cost - best_cost) / max(initial_cost, 1e-9)) * 100
        iterations_improved = sum(1 for r in history if r.improvement_pct > 0)

        log.info(
            "agentic.complete",
            iterations=len(history),
            total_improvement=f"{total_improvement:+.1f}%",
            initial_cost=initial_cost,
            final_cost=best_cost,
        )

        # Phase 2: Runtime orchestration (after compile-time optimization converges)
        runtime_artifacts = self._orchestrate_runtime(best_obs, target)

        return CompilationResult(
            initial_cost_us=initial_cost,
            final_cost_us=best_cost,
            total_improvement_pct=total_improvement,
            iterations_run=len(history),
            iterations_improved=iterations_improved,
            history=history,
            best_observation=best_obs,
            runtime_artifacts=runtime_artifacts,
        )

    def run_with_recipe(self, target: TargetProfile) -> CompilationResult:
        """Run compilation with Recipe IR tracking, validation, and lowering.

        Same as ``run()``, but also:
            1. Enables recipe tracking (seed generation from payload).
            2. Runs the normal optimization loop.
            3. Validates and lowers the accumulated Recipe IR.
            4. Attaches lowered outputs to the result.
        """
        # Enable recipe tracking
        self.env.enable_recipe_tracking()

        # Run normal optimization loop
        result = self.run(target)

        # Validate and lower recipe
        recipe_module = self.env.recipe
        if recipe_module is not None:
            from compgen.ir.recipe.lower import lower_recipe
            from compgen.ir.recipe.validate import validate_recipe_module

            validation = validate_recipe_module(recipe_module)
            lowered = lower_recipe(recipe_module)
            result.runtime_artifacts["recipe_validation"] = {
                "valid": validation.valid,
                "errors": [e.message for e in validation.errors],
            }
            result.runtime_artifacts["recipe_lowered"] = {
                "transform_scripts": len(lowered.transform_scripts),
                "kernel_jobs": len(lowered.kernel_jobs),
                "plan_fragments": len(lowered.plan_fragments),
                "verification_obligations": len(lowered.verification_obligations),
                "eqsat_jobs": len(lowered.eqsat_jobs),
            }

        return result

    def _analyze(self, obs: Observation, target: TargetProfile) -> list[ProposedOptimization]:
        """Ask LLM to analyze the model and propose optimizations."""
        ctx = AnalysisContext(
            model_name="model",
            op_count=len(obs.regions),
            op_summary={r.op_type: 1 for r in obs.regions},
            total_flops=obs.total_flops,
            total_bytes=obs.total_bytes,
            num_devices=obs.num_devices,
            device_names=list(obs.device_names),
            bottleneck_ops=[r.region_id for r in obs.regions if r.is_compute_bound],
        )
        prompt = fmt_analyze(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary=target.name,
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.3),
            )
            response = self.llm_client.generate(request)
            return parse_analyze(response.raw_text)
        except Exception as e:
            log.warning("agentic.analyze_failed", error=str(e))
            return []

    def _ask_llm_for_refinement(
        self, obs: Observation, history: list[IterationRecord], target: TargetProfile,
    ) -> Action | None:
        """Ask LLM what to try next based on history."""
        ctx = RefinementContext(
            iteration=len(history),
            total_budget=self.budget,
            best_latency_us=min((r.cost_after_us for r in history), default=obs.estimated_total_latency_us),
            current_latency_us=obs.estimated_total_latency_us,
            improvement_so_far_pct=sum(r.improvement_pct for r in history),
            actions_tried=[
                {"action_type": r.action_type, "target": r.target, "result": f"{r.improvement_pct:+.1f}%"}
                for r in history[-5:]
            ],
            last_action_result=history[-1].improvement_pct if history else 0.0,
            remaining_bottlenecks=[r.region_id for r in obs.regions if r.is_compute_bound][:5],
        )
        prompt = fmt_refine(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary=target.name,
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.3),
            )
            response = self.llm_client.generate(request)
            action = parse_refine(response.raw_text)
            if action is None or action.action_type == "noop":
                return NoopAction()
            return self._refinement_to_action(action)
        except Exception:
            return NoopAction()

    def _proposal_to_action(self, proposal: ProposedOptimization) -> Action:
        """Convert an LLM proposal into a concrete env action."""
        if proposal.action_type == "eqsat":
            return EqSatAction(rule_categories=("algebraic", "fusion"))
        return NoopAction()

    def _refinement_to_action(self, refinement: RefinementAction) -> Action:
        """Convert a refinement suggestion into a concrete env action."""
        if refinement.action_type == "eqsat":
            return EqSatAction(rule_categories=("algebraic", "fusion"))
        return NoopAction()

    # ------------------------------------------------------------------
    # Phase 2: Runtime orchestration
    # ------------------------------------------------------------------

    def _orchestrate_runtime(
        self, obs: Observation | None, target: TargetProfile,
    ) -> dict[str, Any]:
        """Generate runtime artifacts after optimization converges.

        Asks the LLM to configure:
            1. Profiling — which counters, instrumentation level, hooks.
            2. Dispatch — strategy selection, transport config.
            3. Hooks — target-specific C code for instrumentation.

        Args:
            obs: Best observation from the optimization phase.
            target: Hardware target profile.

        Returns:
            Dict of generated runtime artifacts.
        """
        if obs is None:
            return {}

        log.info("agentic.runtime_orchestration.start")

        # 1. Configure profiling
        profiling_action = self._ask_llm_for_profiling(obs, target)
        if profiling_action is not None:
            result = self.env.step(profiling_action)
            log.info(
                "agentic.runtime.profiling",
                applied=result.info.action_applied,
                level=profiling_action.instrumentation_level,
            )

        # 2. Configure dispatch strategy
        dispatch_action = self._ask_llm_for_dispatch(obs, target)
        if dispatch_action is not None:
            result = self.env.step(dispatch_action)
            log.info(
                "agentic.runtime.dispatch",
                applied=result.info.action_applied,
                strategy=dispatch_action.strategy,
            )

        # 3. Generate runtime hooks
        hooks_action = self._ask_llm_for_hooks(obs, target)
        if hooks_action is not None:
            result = self.env.step(hooks_action)
            log.info(
                "agentic.runtime.hooks",
                applied=result.info.action_applied,
                num_hooks=len(hooks_action.hook_code),
            )

        artifacts = self.env.runtime_artifacts
        log.info("agentic.runtime_orchestration.done", num_artifacts=len(artifacts))
        return artifacts

    def _ask_llm_for_profiling(
        self, obs: Observation, target: TargetProfile,
    ) -> ConfigureProfilingAction | None:
        """Ask LLM to configure profiling."""
        from compgen.agent.prompts.runtime_profile import ProfileHookContext
        from compgen.agent.prompts.runtime_profile import format_prompt as fmt_profile
        from compgen.agent.prompts.runtime_profile import parse_response as parse_profile

        bottlenecks = [
            {"region": r.region_id, "kind": "compute_bound" if r.is_compute_bound else "memory_bound",
             "severity": r.estimated_latency_us / max(obs.estimated_total_latency_us, 1e-9),
             "suggestion": "tile" if r.is_compute_bound else "fuse"}
            for r in obs.regions
            if r.estimated_latency_us > obs.estimated_total_latency_us * 0.1
        ]

        ctx = ProfileHookContext(
            target_name=target.name,
            current_bottlenecks=bottlenecks,
            current_level="NONE",
        )
        prompt = fmt_profile(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary=target.name,
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.2),
            )
            response = self.llm_client.generate(request)
            config = parse_profile(response.raw_text)

            return ConfigureProfilingAction(
                instrumentation_level=config.instrumentation_level.lower(),
                counters=config.counters_to_enable,
                custom_hooks=config.custom_hooks,
                analysis_focus=config.analysis_focus,
            )
        except Exception as e:
            log.warning("agentic.runtime.profiling_failed", error=str(e))
            return None

    def _ask_llm_for_dispatch(
        self, obs: Observation, target: TargetProfile,
    ) -> ConfigureDispatchAction | None:
        """Ask LLM to select dispatch strategy."""
        from compgen.agent.prompts.runtime_dispatch import DispatchContext
        from compgen.agent.prompts.runtime_dispatch import format_prompt as fmt_dispatch
        from compgen.agent.prompts.runtime_dispatch import parse_response as parse_dispatch

        ctx = DispatchContext(
            target_name=target.name,
            device_utilization={
                name: 0.0 for name in obs.device_names
            },
            current_strategy="bulk_sync",
        )
        prompt = fmt_dispatch(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary=target.name,
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model="gemini-2.5-flash", temperature=0.2),
            )
            response = self.llm_client.generate(request)
            config = parse_dispatch(response.raw_text)

            return ConfigureDispatchAction(
                strategy=config.strategy,
                transport_overrides=config.transport_overrides,
                thread_config=config.thread_config,
                double_buffer=config.double_buffer,
            )
        except Exception as e:
            log.warning("agentic.runtime.dispatch_failed", error=str(e))
            return None

    def _ask_llm_for_hooks(
        self, obs: Observation, target: TargetProfile,
    ) -> GenerateRuntimeHooksAction | None:
        """Ask LLM to generate runtime hooks.

        For now, generates standard hooks.  With a real LLM call,
        the LLM would produce target-specific C code.
        """
        # Default hooks that the LLM would typically propose
        default_hooks = {
            "pre_dispatch": 'CG_TRACE_BEGIN("dispatch", kernel_name);',
            "post_dispatch": "CG_TRACE_END();",
        }

        try:
            return GenerateRuntimeHooksAction(
                hook_type="profiling",
                target_language="c",
                hook_code=default_hooks,
            )
        except Exception as e:
            log.warning("agentic.runtime.hooks_failed", error=str(e))
            return None
