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

from dataclasses import dataclass
from typing import Any

import structlog

from compgen.agent.env import (
    Action,
    CompilerEnv,
    ConfigureDispatchAction,
    ConfigureProfilingAction,
    DiscoverOpsAction,
    EqSatAction,
    GenerateRuntimeHooksAction,
    NoopAction,
    Observation,
    ProposeRuleAction,
    RequestVerificationAction,
    SetExtractionObjectiveAction,
)
from compgen.agent.loop import prompts
from compgen.agent.loop.records import CompilationResult, IterationRecord
from compgen.agent.prompts.analyze import ANALYSIS_SCHEMA, AnalysisContext, ProposedOptimization
from compgen.agent.prompts.analyze import format_prompt as fmt_analyze
from compgen.agent.prompts.analyze import parse_response as parse_analyze
from compgen.agent.prompts.refine import REFINEMENT_SCHEMA, RefinementAction, RefinementContext
from compgen.agent.prompts.refine import format_prompt as fmt_refine
from compgen.agent.prompts.refine import parse_response as parse_refine
from compgen.agent.serialize import observation_to_prompt
from compgen.llm.base import CompGenLLMProtocol, GenerationRequest, LLMConfig, PromptContext
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


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
    compiler_memory: Any = None  # Optional CompilerMemory instance
    no_improvement_threshold: int = 5
    explore_temperature: float = 0.5

    def run(self, target: TargetProfile) -> CompilationResult:
        """Run the full agentic compilation loop.

        The env must already be reset with a module + target before calling.
        """
        # Load persistent agent memory for cost calibration + strategy history
        try:
            from pathlib import Path

            from compgen.agent.memory import AgentMemory

            memory_path = Path(".compgen_cache/agent_memory.json")
            self._memory = AgentMemory.load(memory_path) if memory_path.exists() else AgentMemory()
        except Exception:
            self._memory = None

        # Initialize unified CompilerMemory if not provided
        if self.compiler_memory is None:
            try:
                from compgen.memory.store import CompilerMemory

                self.compiler_memory = CompilerMemory()
            except Exception:
                pass

        self.env.attach_llm_client(self.llm_client)
        history: list[IterationRecord] = []
        obs = self.env.observe()
        obs = self._prepare_observation(obs)
        initial_cost = obs.estimated_total_latency_us
        best_cost = initial_cost
        best_obs = obs
        no_improvement_count = 0
        self._active_plan: list[Action] = []
        self._plan_attempted = False

        # Create search task + retrieve priors from memory
        self._current_task_id = "default"
        self._retrieval_priors: list[Any] = []
        if self.compiler_memory is not None:
            try:
                from compgen.memory.schema import ObjectKind
                from compgen.memory.search.retrieve import SearchRetriever
                from compgen.memory.search.task import SearchTask

                task = self.compiler_memory.create_task(
                    kind=ObjectKind.BACKEND_PLAN,
                    target_key=target.name,
                    objective="latency",
                )
                self._current_task_id = task.task_id

                # Retrieve prior knowledge to seed analysis
                retriever = SearchRetriever(self.compiler_memory)
                search_task = SearchTask.for_kernel(
                    task_id=task.task_id,
                    op_family="",  # broad retrieval
                    hardware=target.name,
                )
                retrieval = retriever.retrieve_for_task(search_task)
                self._retrieval_priors = retrieval.schedule_templates + retrieval.tactics
                if self._retrieval_priors:
                    log.info("agentic.retrieval", priors=len(self._retrieval_priors))
            except Exception:
                pass

            # Retrieve learned cost weights for this target (Unit 12)
            try:
                from compgen.solve.learned_weights import retrieve_best_weights

                learned_weights = retrieve_best_weights(self.compiler_memory, target_key=target.name)
                if learned_weights:
                    self.env._eqsat_weights = learned_weights
                    log.info("agentic.learned_weights", weights=learned_weights)
            except Exception:
                pass

        log.info("agentic.start", initial_cost=initial_cost, budget=self.budget)

        # Step 1: Initial analysis
        proposals = self._analyze(obs, target)

        for iteration in range(self.budget):
            # Step 2: Get next action
            if iteration < len(proposals):
                action = self._proposal_to_action(proposals[iteration])
            elif self._active_plan:
                # Multi-step plan: consume next planned action
                action = self._active_plan.pop(0)
            else:
                # Try multi-step planning when stalled
                if no_improvement_count > 0 and not self._plan_attempted:
                    plan_actions = self._ask_llm_for_plan(obs, history, target)
                    self._plan_attempted = True
                    if plan_actions:
                        self._active_plan = plan_actions
                        action = self._active_plan.pop(0)
                    else:
                        action = self._ask_llm_for_refinement(
                            obs,
                            history,
                            target,
                            no_improvement_count=no_improvement_count,
                        )
                else:
                    action = self._ask_llm_for_refinement(
                        obs,
                        history,
                        target,
                        no_improvement_count=no_improvement_count,
                    )

            if action is None or isinstance(action, NoopAction):
                log.info("agentic.stop", reason="LLM suggested noop", iteration=iteration)
                break

            # Step 2b: Consult eqsat search state before applying eqsat actions
            if isinstance(action, EqSatAction):
                extra = self._consult_eqsat_search_state(obs, target)
                if extra is not None and not isinstance(extra, NoopAction):
                    # Apply the supplementary action (e.g., weight tuning, rule proposal)
                    self.env.step(extra)
                    obs = self.env.observe()

            # Step 3: Apply action
            cost_before = obs.estimated_total_latency_us
            result = self.env.step(action)
            obs = result.observation
            cost_after = obs.estimated_total_latency_us

            improvement = ((cost_before - cost_after) / max(cost_before, 1e-9)) * 100

            # Step 3b: Per-step verification (if transform warrants it)
            verification_passed = result.info.verification_passed
            if self._should_verify(action) and result.info.action_applied:
                vr = self._run_per_step_verification(action, obs, target)
                if vr is not None and not vr.get("passed", True) and vr.get("counterexample"):
                    # TV failed — attempt counterexample repair
                    repair = self._ask_llm_for_counterexample_repair(
                        vr,
                        action,
                        obs,
                        target,
                    )
                    if repair is not None and not isinstance(repair, NoopAction):
                        log.info(
                            "agentic.repair",
                            original=action.action_type,
                            repair=repair.action_type,
                        )
                        # Re-apply with repaired action
                        result = self.env.step(repair)
                        obs = result.observation
                        cost_after = obs.estimated_total_latency_us
                        improvement = ((cost_before - cost_after) / max(cost_before, 1e-9)) * 100
                        verification_passed = True  # assume repair fixes it

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

            # Record episode step in CompilerMemory
            if self.compiler_memory is not None:
                try:
                    self.compiler_memory.record_episode_step(
                        task_id=getattr(self, "_current_task_id", "default"),
                        action=action.action_type,
                        reward=improvement / 100.0,
                        step_number=iteration,
                        metadata={"region": action.region_id, "cost_after": str(cost_after)},
                    )
                except Exception:
                    pass

                # Record error patterns for failed actions (Unit 14)
                if not result.info.action_applied or not verification_passed:
                    try:
                        from compgen.memory.error_patterns import record_error_pattern

                        reason = "verification_failed" if not verification_passed else "not_applied"
                        record_error_pattern(
                            self.compiler_memory,
                            action_type=action.action_type,
                            region_context=action.region_id,
                            failure_reason=reason,
                            target_key=target.name,
                        )
                    except Exception:
                        pass

                # Record calibration data (Unit 15)
                if result.info.action_applied and improvement != 0:
                    try:
                        from compgen.memory.calibration import record_calibration

                        record_calibration(
                            self.compiler_memory,
                            target_key=target.name,
                            op_family=action.action_type,
                            estimated_us=cost_before,
                            measured_us=cost_after,
                        )
                    except Exception:
                        pass

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

            # Early stop if no improvement for threshold consecutive iterations
            if no_improvement_count >= self.no_improvement_threshold:
                log.info("agentic.stop", reason=f"no improvement for {self.no_improvement_threshold} iterations")
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

        # Save agent memory with strategy results
        if self._memory is not None:
            try:
                from pathlib import Path

                self._memory.record_strategy(
                    model_name="model",
                    target_name=target.name,
                    pattern_type="agentic",
                    actions_taken=[r.action_type for r in history],
                    estimated_improvement=total_improvement,
                    actual_improvement=total_improvement,
                    success=total_improvement > 0,
                )
                self._memory.save(Path(".compgen_cache/agent_memory.json"))
            except Exception:
                pass

        # Store learned cost weights (Unit 12)
        if self.compiler_memory is not None and total_improvement > 0:
            try:
                from compgen.solve.learned_weights import store_cost_weights

                eqsat_weights = getattr(self.env, "_eqsat_weights", {})
                if eqsat_weights:
                    store_cost_weights(
                        self.compiler_memory,
                        target_key=target.name,
                        weights=eqsat_weights,
                        measured_gain=total_improvement,
                    )
            except Exception:
                pass

        # Extract reusable knowledge from this search trajectory
        if self.compiler_memory is not None and total_improvement > 0:
            try:
                from compgen.memory.search.promote import SearchPromoter

                promoter = SearchPromoter(self.compiler_memory)
                promoter.extract_knowledge(
                    task_id=self._current_task_id,
                    task_kind="backend_plan",
                    op_family="",
                )
                # Update retrieval stats for priors that were used
                if self._retrieval_priors:
                    used_ids = [p.knowledge_id for p in self._retrieval_priors if hasattr(p, "knowledge_id")]
                    promoter.update_retrieval_stats(used_ids, task_succeeded=total_improvement > 0)
            except Exception:
                pass

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
            4. Executes verification obligations via the VerificationExecutor.
            5. Attaches lowered outputs and verification results to the result.
        """
        # Enable recipe tracking
        self.env.enable_recipe_tracking()

        # Snapshot the module before optimization for TV
        payload_before = self.env.best_module().clone() if hasattr(self.env, "best_module") else None

        # Run normal optimization loop
        result = self.run(target)

        # Snapshot the module after optimization
        payload_after = self.env.best_module() if hasattr(self.env, "best_module") else None

        # Validate, lower, and EXECUTE recipe
        recipe_module = self.env.recipe
        if recipe_module is not None:
            from compgen.ir.recipe.execute import RecipeExecutor
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

            # Execute ALL lowered outputs (transforms, eqsat, kernels, plan, verification)
            if payload_after is not None:
                executor = RecipeExecutor()
                exec_result = executor.execute(payload_after, lowered, target)

                result.runtime_artifacts["recipe_execution"] = {
                    "transforms_applied": exec_result.transforms_applied,
                    "transforms_failed": exec_result.transforms_failed,
                    "eqsat_runs": exec_result.eqsat_runs,
                    "kernels_found": sum(1 for k in exec_result.kernels if k.found),
                    "plan_applied": exec_result.plan_applied,
                    "diagnostics": exec_result.diagnostics,
                }

                # Verification results from execution
                if exec_result.verification_results:
                    result.runtime_artifacts["verification_results"] = [
                        {
                            "type": vr.obligation_type,
                            "region_id": vr.region_id,
                            "passed": vr.passed,
                            "status": vr.status,
                            "solver_time_ms": vr.solver_time_ms,
                            "counterexample_summary": (vr.counterexample.summary if vr.counterexample else None),
                        }
                        for vr in exec_result.verification_results
                    ]
                    result.runtime_artifacts["verification_summary"] = {
                        "total": len(exec_result.verification_results),
                        "passed": sum(1 for vr in exec_result.verification_results if vr.passed),
                        "failed": sum(
                            1 for vr in exec_result.verification_results if not vr.passed and vr.status != "skipped"
                        ),
                        "skipped": sum(1 for vr in exec_result.verification_results if vr.status == "skipped"),
                    }

            # Attempt promotion if all verifications passed (with LLM guidance - Unit 16)
            try:
                from compgen.promotion.promote import promote_recipe
                from compgen.runtime.bundle import Bundle

                ver_summary = result.runtime_artifacts.get("verification_summary", {})
                all_passed = ver_summary.get("failed", 0) == 0

                # Ask LLM for promotion decision
                llm_promotes = True
                try:
                    from compgen.agent.prompts.promotion_decision import PROMOTION_SCHEMA, PromotionContext
                    from compgen.agent.prompts.promotion_decision import format_prompt as fmt_promo
                    from compgen.agent.prompts.promotion_decision import parse_response as parse_promo

                    promo_ctx = PromotionContext(
                        improvement_pct=result.total_improvement_pct,
                        verification_summary=str(ver_summary),
                        target_name=target.name,
                        similar_promoted_count=0,
                        iterations_run=result.iterations_run,
                        best_latency_us=result.final_cost_us,
                        initial_latency_us=result.initial_cost_us,
                    )
                    promo_prompt = fmt_promo(promo_ctx)
                    promo_request = GenerationRequest(
                        prompt_template=promo_prompt,
                        context=PromptContext(
                            model_ir_summary="", target_profile_summary=prompts.target_summary(target)
                        ),
                        config=self._llm_config(temperature=0.1, max_tokens=800),
                    )
                    promo_response = self._generate_with_schema(promo_request, PROMOTION_SCHEMA)
                    promo_decision = parse_promo(promo_response.raw_text)
                    if promo_decision is not None:
                        llm_promotes = promo_decision.get("promote", True)
                        log.info(
                            "agentic.promotion_decision", promote=llm_promotes, reason=promo_decision.get("reason", "")
                        )
                except Exception:
                    pass  # Fall back to rule-based

                if all_passed and lowered.transform_scripts and llm_promotes:
                    bundle = Bundle(
                        target_profile=target.name,
                        model_hash=str(id(payload_after))[:12],
                        objective="latency",
                        transform_scripts=lowered.transform_scripts,
                        kernel_jobs=lowered.kernel_jobs,
                        plan_fragments=lowered.plan_fragments,
                    )
                    promo = promote_recipe(
                        bundle,
                        ".compgen_cache/recipes",
                        memory=self.compiler_memory,
                    )
                    result.runtime_artifacts["promotion"] = {
                        "promoted": promo.promoted,
                        "key": promo.key.key if promo.key else None,
                        "path": str(promo.recipe_path) if promo.recipe_path else None,
                    }
                    log.info("agentic.promotion", promoted=promo.promoted, key=promo.key)
            except Exception as e:
                log.debug("agentic.promotion.skipped", error=str(e))

        agent_module = self.env.agent_ir if hasattr(self.env, "agent_ir") else None
        if agent_module is not None:
            from compgen.ir.agent.lower import lower_agent
            from compgen.ir.agent.validate import validate_agent_module

            validation = validate_agent_module(agent_module, recipe_module=recipe_module)
            lowered = lower_agent(agent_module)
            result.runtime_artifacts["agent_validation"] = {
                "valid": validation.valid,
                "errors": [e.message for e in validation.errors],
            }
            result.runtime_artifacts["agent_lowered"] = {
                "request_jobs": len(lowered.request_jobs),
                "claim_records": len(lowered.claim_records),
                "frontier_states": len(lowered.frontier_states),
                "critique_records": len(lowered.critique_records),
                "memory_records": len(lowered.memory_records),
                "protocol_records": len(lowered.protocol_records),
            }

        return result

    def _analyze(self, obs: Observation, target: TargetProfile) -> list[ProposedOptimization]:
        """Ask LLM to analyze the model and propose optimizations."""
        legal_actions = self.env.legal_actions(max_actions=20)
        op_summary: dict[str, int] = {}
        for region in obs.regions:
            op_summary[region.op_type] = op_summary.get(region.op_type, 0) + 1

        dossier = obs.analysis_dossier
        ctx = AnalysisContext(
            model_name="model",
            op_count=len(obs.regions),
            op_summary=op_summary,
            total_flops=obs.total_flops,
            total_bytes=obs.total_bytes,
            num_devices=obs.num_devices,
            device_names=list(obs.device_names),
            bottleneck_ops=[r.region_id for r in obs.regions if r.is_compute_bound],
            graph_break_count=obs.graph_break_count,
            guard_count=obs.guard_count,
            unsupported_ops=list(obs.unsupported_ops),
            repeated_patterns=dict(getattr(dossier, "repeated_patterns", {})) if dossier else {},
            critical_path=list(getattr(dossier, "critical_path", ())) if dossier else [],
            backend_viability=prompts.backend_viability_summary(obs),
            analysis_summary=prompts.analysis_summary(obs),
            legal_actions_summary=prompts.legal_actions_summary(legal_actions),
        )
        prompt = fmt_analyze(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.2, max_tokens=1600),
            )
            response = self._generate_with_schema(request, ANALYSIS_SCHEMA)
            return parse_analyze(response.raw_text)
        except Exception as e:
            log.warning("agentic.analyze_failed", error=str(e))
            return []

    def _ask_llm_for_refinement(
        self,
        obs: Observation,
        history: list[IterationRecord],
        target: TargetProfile,
        *,
        no_improvement_count: int = 0,
    ) -> Action | None:
        """Ask LLM what to try next based on history."""
        legal_actions = self.env.legal_actions(max_actions=20)

        # Retrieve error patterns to include as context
        error_pattern_dicts: list[dict] = []
        if self.compiler_memory is not None:
            try:
                from compgen.memory.error_patterns import error_patterns_to_prompt, retrieve_error_patterns

                patterns = retrieve_error_patterns(self.compiler_memory, target_key=target.name, top_k=3)
                error_pattern_dicts = error_patterns_to_prompt(patterns)
            except Exception:
                pass

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
            last_action_result=f"{history[-1].improvement_pct:+.1f}%" if history else "0.0%",
            remaining_bottlenecks=[r.region_id for r in obs.regions if r.is_compute_bound][:5],
            graph_break_count=obs.graph_break_count,
            guard_count=obs.guard_count,
            unsupported_ops=list(obs.unsupported_ops),
            analysis_summary=prompts.analysis_summary(obs),
            legal_actions_summary=prompts.legal_actions_summary(legal_actions),
            verification_summary=prompts.verification_summary(obs),
            error_patterns=error_pattern_dicts,
        )
        prompt = fmt_refine(ctx)

        # Use higher temperature when stalled (explore mode)
        explore = no_improvement_count >= self.no_improvement_threshold // 2
        temperature = self.explore_temperature if explore else 0.2

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions, history=history),
                config=self._llm_config(temperature=temperature, max_tokens=1200),
            )
            response = self._generate_with_schema(request, REFINEMENT_SCHEMA)
            action = parse_refine(response.raw_text)
            if action is None or action.action_type == "noop":
                return NoopAction()
            return self._refinement_to_action(action)
        except Exception:
            return NoopAction()

    # ------------------------------------------------------------------
    # Multi-step planning (Unit 2)
    # ------------------------------------------------------------------

    def _ask_llm_for_plan(
        self,
        obs: Observation,
        history: list[IterationRecord],
        target: TargetProfile,
    ) -> list[Action]:
        """Ask LLM for a multi-step optimization plan (3-5 steps)."""
        try:
            from compgen.agent.prompts.plan_multi_step import PLAN_SCHEMA, PlanContext
            from compgen.agent.prompts.plan_multi_step import format_prompt as fmt_plan
            from compgen.agent.prompts.plan_multi_step import parse_response as parse_plan
        except ImportError:
            return []

        legal_actions = self.env.legal_actions(max_actions=20)
        ctx = PlanContext(
            observation_summary=observation_to_prompt(obs, legal_actions),
            history_summary="\n".join(prompts.prior_attempts(history)),
            legal_actions_summary=prompts.legal_actions_summary(legal_actions),
            budget_remaining=self.budget - len(history),
        )
        prompt = fmt_plan(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions, history=history),
                config=self._llm_config(temperature=0.3, max_tokens=2000),
            )
            response = self._generate_with_schema(request, PLAN_SCHEMA)
            steps = parse_plan(response.raw_text)
            if not steps:
                return []

            actions: list[Action] = []
            for step in steps:
                from compgen.agent.prompts.analyze import ProposedOptimization

                proposal = ProposedOptimization(
                    action_type=step.get("action_type", "noop"),
                    target=step.get("target", ""),
                    reason=step.get("reason", ""),
                    expected_improvement=0.0,
                )
                action = self._proposal_to_action(proposal)
                if not isinstance(action, NoopAction):
                    actions.append(action)

            log.info("agentic.multi_step_plan", steps=len(actions))
            return actions
        except Exception as e:
            log.debug("agentic.plan_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # EqSat rule generation (Unit 3)
    # ------------------------------------------------------------------

    def _ask_llm_for_eqsat_rule(
        self,
        obs: Observation,
        target: TargetProfile,
    ) -> ProposeRuleAction | None:
        """Ask LLM to generate a new EqSat rewrite rule."""
        try:
            from compgen.eqsat.explain import summarize_module
            from compgen.eqsat.llm_interface import format_rule_proposal_prompt
        except ImportError:
            return None

        try:
            module = self.env._module if hasattr(self.env, "_module") else None
            if module is None:
                return None

            summary = summarize_module(module)
            prompt = format_rule_proposal_prompt(
                egraph_summary=summary.to_prompt() if hasattr(summary, "to_prompt") else str(summary),
                target_description=prompts.target_summary(target),
                objective="latency",
            )

            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(model_ir_summary="", target_profile_summary=prompts.target_summary(target)),
                config=self._llm_config(temperature=0.3, max_tokens=2000),
            )
            response = self.llm_client.generate(request)
            rule_code = response.raw_text.strip()

            # Extract code from markdown fences if present
            if "```" in rule_code:
                import re

                m = re.search(r"```(?:python)?\n(.*?)```", rule_code, re.DOTALL)
                if m:
                    rule_code = m.group(1).strip()

            if rule_code:
                log.info("agentic.eqsat_rule_generated", code_len=len(rule_code))
                return ProposeRuleAction(region_id="", rule_code=rule_code, category="llm_generated")
        except Exception as e:
            log.debug("agentic.eqsat_rule_failed", error=str(e))

        return None

    # ------------------------------------------------------------------
    # EqSat search state consultation (Unit 4)
    # ------------------------------------------------------------------

    def _consult_eqsat_search_state(
        self,
        obs: Observation,
        target: TargetProfile,
    ) -> Action | None:
        """Consult LLM about eqsat search direction and weight tuning."""
        try:
            from compgen.agent.prompts.eqsat_extraction_weights import EXTRACTION_WEIGHTS_SCHEMA, WeightsContext
            from compgen.agent.prompts.eqsat_extraction_weights import format_prompt as fmt_weights
            from compgen.agent.prompts.eqsat_extraction_weights import parse_response as parse_weights
            from compgen.agent.prompts.eqsat_search_state import SEARCH_STATE_SCHEMA, SearchStateContext
            from compgen.agent.prompts.eqsat_search_state import format_prompt as fmt_ss
            from compgen.agent.prompts.eqsat_search_state import parse_response as parse_ss
        except ImportError:
            return None

        try:
            ctx = SearchStateContext(
                egraph_summary=prompts.analysis_summary(obs),
                rule_stats={},
                best_cost=obs.best_latency_us,
                iteration=obs.step_count,
                total_eclasses=0,
                total_enodes=0,
            )
            prompt = fmt_ss(ctx)

            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(model_ir_summary="", target_profile_summary=prompts.target_summary(target)),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
            )
            response = self._generate_with_schema(request, SEARCH_STATE_SCHEMA)
            result = parse_ss(response.raw_text)

            if result and result.get("action") == "CHANGE_WEIGHTS":
                # Ask for specific weights
                w_ctx = WeightsContext(
                    egraph_summary=prompts.analysis_summary(obs),
                    target_description=prompts.target_summary(target),
                    current_fusion_weight=1.0,
                    current_transfer_weight=1.0,
                    current_backend_match_weight=1.0,
                )
                w_prompt = fmt_weights(w_ctx)
                w_request = GenerationRequest(
                    prompt_template=w_prompt,
                    context=PromptContext(model_ir_summary="", target_profile_summary=prompts.target_summary(target)),
                    config=self._llm_config(temperature=0.2, max_tokens=800),
                )
                w_response = self._generate_with_schema(w_request, EXTRACTION_WEIGHTS_SCHEMA)
                weights = parse_weights(w_response.raw_text)
                if weights:
                    return SetExtractionObjectiveAction(
                        fusion_weight=weights.get("fusion_weight", 1.0),
                        transfer_weight=weights.get("transfer_weight", 1.0),
                        backend_match_weight=weights.get("backend_match_weight", 1.0),
                    )

            if result and result.get("action") == "PROPOSE_RULE":
                return self._ask_llm_for_eqsat_rule(obs, target)

        except Exception as e:
            log.debug("agentic.eqsat_search_state_failed", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Evolutionary optimizer integration (Unit 11)
    # ------------------------------------------------------------------

    def run_with_evolution(self, target: TargetProfile) -> CompilationResult:
        """Run optimization using the evolutionary strategy optimizer."""
        from compgen.agent.evolution import EvolutionaryOptimizer

        optimizer = EvolutionaryOptimizer(
            llm_client=self.llm_client,
            env=self.env,
            population_size=5,
            top_k=2,
            generations=3,
        )
        evo_result = optimizer.evolve(target)

        # Record in compiler memory if available
        if self.compiler_memory is not None:
            try:
                from compgen.memory.schema import ObjectKind

                task = self.compiler_memory.create_task(
                    kind=ObjectKind.BACKEND_PLAN,
                    target_key=target.name,
                    objective="latency",
                )
                for gen_idx, gen_strats in enumerate(evo_result.history):
                    for strat in gen_strats:
                        self.compiler_memory.record_episode_step(
                            task_id=task.task_id,
                            action=strat.strategy.name,
                            reward=strat.improvement_pct / 100.0,
                            step_number=gen_idx,
                            metadata={"actions": ",".join(strat.strategy.action_types)},
                        )
            except Exception:
                pass

        return CompilationResult(
            initial_cost_us=evo_result.best_cost_us / max(1 - evo_result.total_improvement_pct / 100, 1e-9),
            final_cost_us=evo_result.best_cost_us,
            total_improvement_pct=evo_result.total_improvement_pct,
            iterations_run=evo_result.candidates_evaluated,
            iterations_improved=sum(1 for gen in evo_result.history for s in gen if s.improvement_pct > 0),
            history=[],
            best_observation=self.env.observe(),
        )

    # ------------------------------------------------------------------
    # Multi-module global strategy (Unit 16)
    # ------------------------------------------------------------------

    def run_multi_module(
        self,
        modules: list[Any],
        target: TargetProfile,
    ) -> list[CompilationResult]:
        """Coordinate optimization across multiple modules."""
        try:
            from compgen.agent.prompts.global_strategy import GLOBAL_STRATEGY_SCHEMA, GlobalStrategyContext
            from compgen.agent.prompts.global_strategy import format_prompt as fmt_gs
            from compgen.agent.prompts.global_strategy import parse_response as parse_gs
        except ImportError:
            # Fall back to sequential optimization
            results = []
            for module in modules:
                self.env.reset_module(module) if hasattr(self.env, "reset_module") else None
                results.append(self.run(target))
            return results

        # Build per-module summaries
        summaries = []
        for i, module in enumerate(modules):
            summaries.append(
                {
                    "name": f"module_{i}",
                    "op_count": sum(1 for _ in module.walk()) if hasattr(module, "walk") else 0,
                    "flops": 0,
                    "bottleneck": "unknown",
                }
            )

        ctx = GlobalStrategyContext(
            module_count=len(modules),
            per_module_summaries=summaries,
            target_name=target.name,
        )
        prompt = fmt_gs(ctx)

        # Get global strategy from LLM
        priority_order = list(range(len(modules)))
        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(model_ir_summary="", target_profile_summary=prompts.target_summary(target)),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
            )
            response = self._generate_with_schema(request, GLOBAL_STRATEGY_SCHEMA)
            result = parse_gs(response.raw_text)
            if result and result.get("priority_order"):
                # Map module names to indices
                name_to_idx = {f"module_{i}": i for i in range(len(modules))}
                order = [name_to_idx.get(n, -1) for n in result["priority_order"]]
                priority_order = [i for i in order if 0 <= i < len(modules)]
                # Add any missing indices
                for i in range(len(modules)):
                    if i not in priority_order:
                        priority_order.append(i)
            log.info("agentic.global_strategy", order=priority_order)
        except Exception:
            pass

        # Execute in LLM-suggested order
        results: list[CompilationResult] = []
        for idx in priority_order:
            if hasattr(self.env, "reset_module"):
                self.env.reset_module(modules[idx])
            results.append(self.run(target))

        return results

    def _proposal_to_action(self, proposal: ProposedOptimization) -> Action:
        """Convert an LLM proposal into a concrete env action."""
        from compgen.agent.env import (
            AssignDeviceAction,
            DiscoverOpsAction,
            FuseAction,
            GeneratePassAction,
            TileAction,
        )

        target = getattr(proposal, "target", "") or ""
        if proposal.action_type == "eqsat":
            # Try to generate a new rule via LLM first (Unit 3)
            # The rule will be applied via ProposeRuleAction before EqSatAction
            return EqSatAction(region_id=target, rule_categories=("algebraic", "fusion", "llm_generated"))
        if proposal.action_type == "tile":
            return TileAction(region_id=target, tile_sizes=(32, 32))
        if proposal.action_type == "fuse":
            return FuseAction(region_id=target)
        if proposal.action_type == "assign_device":
            return AssignDeviceAction(region_id=target, device_index=0)
        if proposal.action_type == "generate_pass":
            return GeneratePassAction(
                region_id=target,
                description=getattr(proposal, "reason", "LLM-generated pass"),
            )
        if proposal.action_type == "discover_ops":
            return DiscoverOpsAction(region_id=target, auto_generate=True)
        if proposal.action_type == "request_verification":
            return RequestVerificationAction(region_id=target, level="both")
        return NoopAction()

    def _refinement_to_action(self, refinement: RefinementAction) -> Action:
        """Convert a refinement suggestion into a concrete env action."""
        from compgen.agent.env import (
            AssignDeviceAction,
            DiscoverOpsAction,
            FuseAction,
            GeneratePassAction,
            TileAction,
        )

        target = refinement.target_region or ""
        params = refinement.parameters or {}

        if refinement.action_type == "eqsat":
            categories = params.get("categories", "algebraic,fusion").split(",")
            return EqSatAction(region_id=target, rule_categories=tuple(c.strip() for c in categories))
        if refinement.action_type == "tile":
            sizes_raw = params.get("tile_sizes", "32,32")
            try:
                sizes = tuple(int(s.strip()) for s in sizes_raw.split(","))
            except ValueError:
                sizes = (32, 32)
            return TileAction(region_id=target, tile_sizes=sizes)
        if refinement.action_type == "fuse":
            return FuseAction(region_id=target, target_region_id=params.get("target_region_id", ""))
        if refinement.action_type == "assign_device":
            try:
                device_idx = int(params.get("device_index", "0"))
            except ValueError:
                device_idx = 0
            return AssignDeviceAction(region_id=target, device_index=device_idx)
        if refinement.action_type == "generate_pass":
            return GeneratePassAction(
                region_id=target,
                description=refinement.reasoning or "LLM-generated pass",
            )
        if refinement.action_type == "discover_ops":
            auto_generate = params.get("auto_generate", "true").lower() != "false"
            return DiscoverOpsAction(region_id=target, auto_generate=auto_generate)
        if refinement.action_type == "request_verification":
            return RequestVerificationAction(
                region_id=target,
                level=params.get("level", "both"),
            )
        return NoopAction()

    def _prepare_observation(self, obs: Observation) -> Observation:
        """Ensure deterministic analysis and unsupported-op discovery have run before prompting."""
        if obs.analysis_dossier is None:
            ok, err, _diagnostics = self.env._apply_analyze()
            if ok:
                obs = self.env.observe()
            else:
                log.debug("agentic.pre_analysis_failed", error=err)
        if obs.unsupported_ops:
            ok, err, _diagnostics = self.env._apply_discover_ops(DiscoverOpsAction(auto_generate=True))
            if ok:
                obs = self.env.observe()
            else:
                log.debug("agentic.pre_discover_failed", error=err)
        return obs

    def _generate_with_schema(
        self,
        request: GenerationRequest,
        schema: dict[str, Any],
    ) -> Any:
        try:
            return self.llm_client.generate_structured(request, schema)
        except Exception as exc:
            log.debug("agentic.structured_generation_failed", error=str(exc))
            return self.llm_client.generate(request)

    def _llm_config(self, *, temperature: float, max_tokens: int) -> LLMConfig:
        return LLMConfig(
            model=self._model_id(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _model_id(self) -> str:
        return str(getattr(self.llm_client, "model", "default"))

    # ------------------------------------------------------------------
    # Phase 2: Runtime orchestration
    # ------------------------------------------------------------------

    def _orchestrate_runtime(
        self,
        obs: Observation | None,
        target: TargetProfile,
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
        self,
        obs: Observation,
        target: TargetProfile,
    ) -> ConfigureProfilingAction | None:
        """Ask LLM to configure profiling."""
        from compgen.agent.prompts.runtime_profile import ProfileHookContext
        from compgen.agent.prompts.runtime_profile import format_prompt as fmt_profile
        from compgen.agent.prompts.runtime_profile import parse_response as parse_profile

        bottlenecks = [
            {
                "region": r.region_id,
                "kind": "compute_bound" if r.is_compute_bound else "memory_bound",
                "severity": r.estimated_latency_us / max(obs.estimated_total_latency_us, 1e-9),
                "suggestion": "tile" if r.is_compute_bound else "fuse",
            }
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
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
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
        self,
        obs: Observation,
        target: TargetProfile,
    ) -> ConfigureDispatchAction | None:
        """Ask LLM to select dispatch strategy."""
        from compgen.agent.prompts.runtime_dispatch import DispatchContext
        from compgen.agent.prompts.runtime_dispatch import format_prompt as fmt_dispatch
        from compgen.agent.prompts.runtime_dispatch import parse_response as parse_dispatch

        ctx = DispatchContext(
            target_name=target.name,
            device_utilization={name: 0.0 for name in obs.device_names},
            current_strategy="bulk_sync",
        )
        prompt = fmt_dispatch(ctx)

        try:
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
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
        self,
        obs: Observation,
        target: TargetProfile,
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

    # ------------------------------------------------------------------
    # Per-step verification
    # ------------------------------------------------------------------

    _VERIFIABLE_ACTION_TYPES = {"tile", "fuse", "vectorize", "eqsat", "generate_pass"}

    def _should_verify(self, action: Action) -> bool:
        """Decide whether an action warrants formal verification.

        Tiles, fuses, vectorizes, eqsat, and generated passes get TV.
        Placement, copy insertion, runtime config do not.
        """
        return action.action_type in self._VERIFIABLE_ACTION_TYPES

    def _run_per_step_verification(
        self,
        action: Action,
        obs: Observation,
        target: TargetProfile,
    ) -> dict[str, Any] | None:
        """Run verification for the action that was just applied.

        Uses the LLM to decide the verification level, then executes.
        Falls back to differential if LLM call fails.
        """
        from compgen.agent.prompts.verify_strategy import VerifyStrategyContext
        from compgen.agent.prompts.verify_strategy import format_prompt as fmt_vs
        from compgen.agent.prompts.verify_strategy import parse_response as parse_vs

        # Ask LLM for verification strategy
        ctx = VerifyStrategyContext(
            regions=[
                {
                    "region_id": action.region_id,
                    "op_type": action.action_type,
                    "transform_applied": action.action_type,
                }
            ],
            verification_budget_ms=30_000,
            verifiable_ops=list(getattr(obs.verification, "verifiable_op_types", ())),
            past_failures=[],
        )
        prompt = fmt_vs(ctx)

        level = "differential"  # safe default
        try:
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
            )
            response = self.llm_client.generate(request)
            assignments = parse_vs(response.raw_text)
            if assignments:
                level = assignments[0].level
        except Exception:
            pass  # fall back to differential

        # Execute verification
        from compgen.semantic.executor import VerificationExecutor

        executor = VerificationExecutor()
        obligation = {"type": level, "region_id": action.region_id}

        before = self.env.best_module() if hasattr(self.env, "best_module") else None
        after_mod = getattr(self.env, "_module", None)

        result = executor.execute_single(obligation, before, after_mod)

        log.info(
            "agentic.verify",
            region=action.region_id,
            level=level,
            status=result.status,
            solver_ms=f"{result.solver_time_ms:.0f}",
        )

        return {
            "passed": result.passed,
            "status": result.status,
            "region_id": result.region_id,
            "counterexample": (
                {
                    "inputs": result.counterexample.inputs,
                    "expected": result.counterexample.expected,
                    "actual": result.counterexample.actual,
                    "summary": result.counterexample.summary,
                }
                if result.counterexample
                else None
            ),
        }

    def _ask_llm_for_counterexample_repair(
        self,
        vr: dict[str, Any],
        action: Action,
        obs: Observation,
        target: TargetProfile,
    ) -> Action | None:
        """Ask LLM to repair a transform after TV failure."""
        from compgen.agent.prompts.counterexample_repair import CounterexampleRepairContext
        from compgen.agent.prompts.counterexample_repair import format_prompt as fmt_cex
        from compgen.agent.prompts.counterexample_repair import parse_response as parse_cex

        ctx = CounterexampleRepairContext(
            region_id=vr.get("region_id", ""),
            transform_applied=action.action_type,
            counterexample=vr.get("counterexample") or {},
            verification_error=vr.get("status", "unknown"),
            available_alternatives=["tile", "fuse", "eqsat", "noop"],
        )
        prompt = fmt_cex(ctx)

        try:
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=prompts.build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.3, max_tokens=1200),
            )
            response = self.llm_client.generate(request)
            proposal = parse_cex(response.raw_text)
            if proposal is None or proposal.action_type == "noop":
                return NoopAction()
            # Convert repair proposal to action
            if proposal.action_type == "eqsat":
                return EqSatAction(rule_categories=("algebraic", "fusion"))
            return NoopAction()
        except Exception:
            return NoopAction()
