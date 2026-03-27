"""Stage registry and pipeline runner.

The registry holds all available stages and target dialect stacks.
Each target declares its own stage sequence (variable depth).
The pipeline runner executes stages in order, enforcing contracts
between adjacent stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.stages.base import CompilationStage, StageResult, TargetStagePlugin
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass
class TargetDialectStack:
    """Declares which stages a target needs and in what order.

    Different targets have different stack depths:
      - GPU (Triton): encoding → dispatch → codegen → bundle (4 stages)
      - NPU: encoding → dispatch → kernel → schedule → isa → memory → bundle (7 stages)
      - CPU: encoding → dispatch → tiling → codegen → bundle (5 stages)

    Attributes:
        target_name: Target identifier.
        stages: Ordered list of compilation stages.
        plugins: Dict mapping stage_name → target-specific plugin.
    """

    target_name: str
    stages: list[CompilationStage] = field(default_factory=list)
    plugins: dict[str, TargetStagePlugin] = field(default_factory=dict)

    def bind_plugins(self) -> None:
        """Bind all plugins to their corresponding stages."""
        for stage in self.stages:
            if stage.name in self.plugins:
                stage.register_plugin(self.plugins[stage.name])


@dataclass(frozen=True)
class PipelineResult:
    """Result of running the full compilation pipeline."""

    stage_results: list[StageResult] = field(default_factory=list)
    final_module: ModuleOp | None = None
    all_artifacts: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    first_failure: str | None = None
    stages_run: int = 0


class StageRegistry:
    """Registry of all stages and target dialect stacks.

    Stages are registered globally.  Target stacks reference stages by
    instance.  The pipeline runner sequences stages and enforces contracts.
    """

    def __init__(self) -> None:
        self._shared_stages: dict[str, CompilationStage] = {}
        self._target_stacks: dict[str, TargetDialectStack] = {}

    def register_shared_stage(self, stage: CompilationStage) -> None:
        """Register a shared (target-agnostic) stage."""
        self._shared_stages[stage.name] = stage

    def register_target_stack(self, stack: TargetDialectStack) -> None:
        """Register a target dialect stack."""
        self._target_stacks[stack.target_name] = stack

    def get_shared_stage(self, name: str) -> CompilationStage | None:
        """Look up a shared stage by name."""
        return self._shared_stages.get(name)

    def get_target_stack(self, target_name: str) -> TargetDialectStack | None:
        """Look up a target dialect stack."""
        return self._target_stacks.get(target_name)

    def list_targets(self) -> list[str]:
        """List all registered target names."""
        return list(self._target_stacks.keys())

    def run_pipeline(
        self,
        module: ModuleOp,
        target: TargetProfile,
        capabilities: CapabilitySpec,
        config: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Run the full compilation pipeline for a target.

        Looks up the target's dialect stack, binds plugins, and executes
        stages in order.  Stops on first failure.

        Args:
            module: Input IR (after capture + global optimizations).
            target: Target profile.
            capabilities: Target capability spec.
            config: Per-stage configuration overrides keyed by stage name.

        Returns:
            PipelineResult with per-stage results and final module.
        """
        stack = self._target_stacks.get(target.name)
        if stack is None:
            return PipelineResult(
                passed=False,
                first_failure=f"No dialect stack registered for target '{target.name}'",
            )

        # Bind plugins to stages
        stack.bind_plugins()

        stage_results: list[StageResult] = []
        all_artifacts: dict[str, Any] = {}
        current_module = module

        for stage in stack.stages:
            stage_config = (config or {}).get(stage.name)
            result = stage.run(current_module, target, capabilities, stage_config)
            stage_results.append(result)
            all_artifacts.update(result.artifacts)

            if not result.passed:
                log.error(
                    "pipeline.stage_failed",
                    stage=stage.name,
                    violations=result.contract_violations,
                )
                return PipelineResult(
                    stage_results=stage_results,
                    final_module=current_module,
                    all_artifacts=all_artifacts,
                    passed=False,
                    first_failure=stage.name,
                    stages_run=len(stage_results),
                )

            if result.module is not None:
                current_module = result.module

            log.info("pipeline.stage_complete", stage=stage.name)

        return PipelineResult(
            stage_results=stage_results,
            final_module=current_module,
            all_artifacts=all_artifacts,
            passed=True,
            stages_run=len(stage_results),
        )

    def run_single_stage(
        self,
        stage_name: str,
        module: ModuleOp,
        target: TargetProfile,
        capabilities: CapabilitySpec,
    ) -> StageResult:
        """Run a single named stage (for debugging / agent interaction)."""
        # Check target stack first
        stack = self._target_stacks.get(target.name)
        if stack is not None:
            stack.bind_plugins()
            for stage in stack.stages:
                if stage.name == stage_name:
                    return stage.run(module, target, capabilities)

        # Fall back to shared stages
        stage = self._shared_stages.get(stage_name)
        if stage is None:
            return StageResult(
                stage_name=stage_name,
                passed=False,
                contract_violations=[f"Stage '{stage_name}' not found"],
            )
        return stage.run(module, target, capabilities)
