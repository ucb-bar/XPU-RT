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

import json
from dataclasses import dataclass, field
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
    RequestVerificationAction,
)
from compgen.agent.prompts.analyze import ANALYSIS_SCHEMA, AnalysisContext, ProposedOptimization
from compgen.agent.prompts.analyze import format_prompt as fmt_analyze
from compgen.agent.prompts.analyze import parse_response as parse_analyze
from compgen.agent.prompts.refine import REFINEMENT_SCHEMA, RefinementAction, RefinementContext
from compgen.agent.prompts.refine import format_prompt as fmt_refine
from compgen.agent.prompts.refine import parse_response as parse_refine
from compgen.agent.serialize import legal_actions_to_dict, observation_to_dict, observation_to_prompt
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
    compiler_memory: Any = None  # Optional CompilerMemory instance

    def run(self, target: TargetProfile) -> CompilationResult:
        """Run the full agentic compilation loop.

        The env must already be reset with a module + target before calling.
        """
        # Load persistent agent memory for cost calibration + strategy history
        try:
            from compgen.agent.memory import AgentMemory
            from pathlib import Path

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

        # Create search task + retrieve priors from memory
        self._current_task_id = "default"
        self._retrieval_priors: list[Any] = []
        if self.compiler_memory is not None:
            try:
                from compgen.memory.schema import ObjectKind
                from compgen.search.retrieve import SearchRetriever
                from compgen.search.task import SearchTask

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
                self._retrieval_priors = (
                    retrieval.schedule_templates + retrieval.tactics
                )
                if self._retrieval_priors:
                    log.info("agentic.retrieval", priors=len(self._retrieval_priors))
            except Exception:
                pass

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

            # Step 3b: Per-step verification (if transform warrants it)
            verification_passed = result.info.verification_passed
            if self._should_verify(action) and result.info.action_applied:
                vr = self._run_per_step_verification(action, obs, target)
                if vr is not None and not vr.get("passed", True) and vr.get("counterexample"):
                    # TV failed — attempt counterexample repair
                    repair = self._ask_llm_for_counterexample_repair(
                        vr, action, obs, target,
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

        # Extract reusable knowledge from this search trajectory
        if self.compiler_memory is not None and total_improvement > 0:
            try:
                from compgen.search.promote import SearchPromoter

                promoter = SearchPromoter(self.compiler_memory)
                promoter.extract_knowledge(
                    task_id=self._current_task_id,
                    task_kind="backend_plan",
                    op_family="",
                )
                # Update retrieval stats for priors that were used
                if self._retrieval_priors:
                    used_ids = [
                        p.knowledge_id for p in self._retrieval_priors
                        if hasattr(p, "knowledge_id")
                    ]
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
                            "counterexample_summary": (
                                vr.counterexample.summary if vr.counterexample else None
                            ),
                        }
                        for vr in exec_result.verification_results
                    ]
                    result.runtime_artifacts["verification_summary"] = {
                        "total": len(exec_result.verification_results),
                        "passed": sum(1 for vr in exec_result.verification_results if vr.passed),
                        "failed": sum(
                            1 for vr in exec_result.verification_results
                            if not vr.passed and vr.status != "skipped"
                        ),
                        "skipped": sum(
                            1 for vr in exec_result.verification_results if vr.status == "skipped"
                        ),
                    }

            # Attempt promotion if all verifications passed
            try:
                from compgen.promotion.promote import promote_recipe
                from compgen.runtime.bundle import Bundle

                ver_summary = result.runtime_artifacts.get("verification_summary", {})
                all_passed = ver_summary.get("failed", 0) == 0

                if all_passed and lowered.transform_scripts:
                    bundle = Bundle(
                        target_profile=target.name,
                        model_hash=str(id(module))[:12],
                        objective="latency",
                        transform_scripts=lowered.transform_scripts,
                        kernel_jobs=lowered.kernel_jobs,
                        plan_fragments=lowered.plan_fragments,
                    )
                    promo = promote_recipe(
                        bundle, ".compgen_cache/recipes",
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
            backend_viability=self._backend_viability_summary(obs),
            analysis_summary=self._analysis_summary(obs),
            legal_actions_summary=self._legal_actions_summary(legal_actions),
        )
        prompt = fmt_analyze(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=self._build_prompt_context(obs, target, legal_actions),
                config=self._llm_config(temperature=0.2, max_tokens=1600),
            )
            response = self._generate_with_schema(request, ANALYSIS_SCHEMA)
            return parse_analyze(response.raw_text)
        except Exception as e:
            log.warning("agentic.analyze_failed", error=str(e))
            return []

    def _ask_llm_for_refinement(
        self, obs: Observation, history: list[IterationRecord], target: TargetProfile,
    ) -> Action | None:
        """Ask LLM what to try next based on history."""
        legal_actions = self.env.legal_actions(max_actions=20)
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
            analysis_summary=self._analysis_summary(obs),
            legal_actions_summary=self._legal_actions_summary(legal_actions),
            verification_summary=self._verification_summary(obs),
        )
        prompt = fmt_refine(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=self._build_prompt_context(obs, target, legal_actions, history=history),
                config=self._llm_config(temperature=0.2, max_tokens=1200),
            )
            response = self._generate_with_schema(request, REFINEMENT_SCHEMA)
            action = parse_refine(response.raw_text)
            if action is None or action.action_type == "noop":
                return NoopAction()
            return self._refinement_to_action(action)
        except Exception:
            return NoopAction()

    def _proposal_to_action(self, proposal: ProposedOptimization) -> Action:
        """Convert an LLM proposal into a concrete env action."""
        from compgen.agent.env import (
            AssignDeviceAction,
            DiscoverOpsAction,
            FuseAction,
            GeneratePassAction,
            RequestVerificationAction,
            TileAction,
        )

        target = getattr(proposal, "target", "") or ""
        if proposal.action_type == "eqsat":
            return EqSatAction(region_id=target, rule_categories=("algebraic", "fusion"))
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
            RequestVerificationAction,
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

    def _build_prompt_context(
        self,
        obs: Observation,
        target: TargetProfile,
        legal_actions: list[Any],
        history: list[IterationRecord] | None = None,
    ) -> PromptContext:
        return PromptContext(
            model_ir_summary=observation_to_prompt(obs, legal_actions),
            target_profile_summary=self._target_summary(target),
            available_transforms=sorted({entry["type"] for entry in legal_actions_to_dict(legal_actions)}),
            kernel_contracts=self._kernel_contracts(obs),
            objective=Objective.LATENCY,
            prior_attempts=self._prior_attempts(history or []),
            hardware_feedback=self._verification_summary(obs),
            frontend_diagnostics_summary=self._frontend_summary(obs),
            analysis_dossier_summary=self._analysis_summary(obs),
            unsupported_operator_summary=self._unsupported_summary(obs),
            pack_summary=self._pack_summary(obs),
            integration_branch_summary=self._integration_branch_summary(obs),
            frontier_summary=self._frontier_summary(obs, history or []),
            legal_action_summary=self._legal_actions_summary(legal_actions),
            evidence_json=self._evidence_json(obs, legal_actions),
        )

    def _target_summary(self, target: TargetProfile) -> str:
        device_parts = []
        for device in target.devices:
            max_mem = max((level.size_bytes for level in device.memory_hierarchy), default=0)
            device_parts.append(f"{device.name}: memory={max_mem}B")
        return f"{target.name}\n" + "\n".join(device_parts)

    def _frontend_summary(self, obs: Observation) -> str:
        lines = [
            f"graph_breaks={obs.graph_break_count}",
            f"guards={obs.guard_count}",
            f"unsupported_ops={len(obs.unsupported_ops)}",
        ]
        if obs.unsupported_ops:
            lines.append("targets=" + ", ".join(obs.unsupported_ops[:10]))
        return "\n".join(lines)

    def _analysis_summary(self, obs: Observation) -> str:
        dossier = obs.analysis_dossier
        if dossier is None:
            return "analysis unavailable"
        repeated = ", ".join(
            f"{name}:{count}" for name, count in sorted(
                dossier.repeated_patterns.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        ) or "(none)"
        lines = [
            f"regions={dossier.total_regions}",
            f"critical_path={list(dossier.critical_path[:8])}",
            f"dynamic_shapes={list(dossier.dynamic_shape_regions[:8])}",
            f"unsupported_targets={list(dossier.unsupported_targets[:8])}",
            f"repeated_patterns={repeated}",
        ]
        for region in dossier.regions[:8]:
            lines.append(
                f"{region.region_id}: kind={region.kind} ai={region.arithmetic_intensity:.2f} "
                f"backends={list(region.backend_viability)} layouts={list(region.layout_candidates)} "
                f"parallel={list(region.parallelizable_with[:5])}"
            )
        return "\n".join(lines)

    def _unsupported_summary(self, obs: Observation) -> str:
        if not obs.unsupported_ops:
            return "no unsupported operators"
        return "\n".join(f"- {target}" for target in obs.unsupported_ops)

    def _pack_summary(self, obs: Observation) -> str:
        if not obs.active_packs:
            return "no active extension packs"
        lines = [
            f"active={list(obs.active_packs)}",
            f"sealed_surfaces={list(obs.sealed_surfaces[:12])}",
            f"generation_apertures={list(obs.generation_apertures[:12])}",
            f"available_profilers={list(obs.available_profilers[:12])}",
            f"benchmark_targets={list(obs.pack_benchmark_targets[:12])}",
        ]
        return "\n".join(lines)

    def _integration_branch_summary(self, obs: Observation) -> str:
        return obs.integration_branch or "no integration branch"

    def _frontier_summary(self, obs: Observation, history: list[IterationRecord]) -> str:
        last_action = history[-1].action_type if history else "none"
        return (
            f"step={obs.step_count} budget_remaining={obs.budget_remaining}\n"
            f"best_latency_us={obs.best_latency_us:.3f}\n"
            f"current_latency_us={obs.estimated_total_latency_us:.3f}\n"
            f"last_action={last_action}"
        )

    def _verification_summary(self, obs: Observation) -> str:
        if obs.verification is None:
            return "verification unavailable"
        summary = (
            f"tv_passed={obs.verification.tv_passed} "
            f"tv_failed={obs.verification.tv_failed} "
            f"tv_pending={obs.verification.tv_pending}"
        )
        if obs.verification.last_failure_region:
            summary += (
                f"\nlast_failure={obs.verification.last_failure_region}: "
                f"{obs.verification.last_counterexample_summary}"
            )
        return summary

    def _legal_actions_summary(self, legal_actions: list[Any]) -> str:
        entries = []
        for item in legal_actions[:12]:
            delta = f"{item.estimated_cost_delta_us:+.2f}us"
            entries.append(f"{item.rank}. {item.action.action_type} {item.action.region_id} {delta} [{item.risk}]")
        return "\n".join(entries) or "(none)"

    def _kernel_contracts(self, obs: Observation) -> list[str]:
        dossier = obs.analysis_dossier
        if dossier is None:
            return []
        contracts: list[str] = []
        for region in dossier.regions[:12]:
            contracts.append(
                f"{region.region_id}: backends={','.join(region.backend_viability)} "
                f"layouts={','.join(region.layout_candidates)} "
                f"local_mem_fit={region.local_memory_fit}"
            )
        return contracts

    def _backend_viability_summary(self, obs: Observation) -> list[str]:
        dossier = obs.analysis_dossier
        if dossier is None:
            return []
        seen: list[str] = []
        for region in dossier.regions:
            for backend in region.backend_viability:
                if backend not in seen:
                    seen.append(backend)
        return seen

    def _prior_attempts(self, history: list[IterationRecord]) -> list[str]:
        return [
            f"iter={record.iteration} action={record.action_type} target={record.target} "
            f"improvement={record.improvement_pct:+.2f}% applied={record.applied}"
            for record in history[-8:]
        ]

    def _evidence_json(self, obs: Observation, legal_actions: list[Any]) -> str:
        payload = observation_to_dict(obs)
        payload["legal_actions"] = legal_actions_to_dict(legal_actions[:20])
        return json.dumps(payload, sort_keys=True)

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
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=self._build_prompt_context(obs, target, legal_actions),
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
            legal_actions = self.env.legal_actions(max_actions=12)
            request = GenerationRequest(
                prompt_template=prompt,
                context=self._build_prompt_context(obs, target, legal_actions),
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
        self, action: Action, obs: Observation, target: TargetProfile,
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
            regions=[{
                "region_id": action.region_id,
                "op_type": action.action_type,
                "transform_applied": action.action_type,
            }],
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
                context=self._build_prompt_context(obs, target, legal_actions),
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
                context=self._build_prompt_context(obs, target, legal_actions),
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
