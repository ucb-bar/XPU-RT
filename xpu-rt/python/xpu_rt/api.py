"""Top-level Python control plane API for CompGen.

Provides two entry points:

- ``device(spec_path)`` -- load a HardwareSpec YAML file, generate a target,
  and return a :class:`CompGenDevice` encapsulating the spec, profile,
  capabilities, and dialect stack.

- ``compile_model(model, device, objective)`` -- capture a PyTorch model,
  convert to xDSL Payload IR, run equality-saturation optimisation, execute
  the stage pipeline for the target, and return a :class:`CompiledModel`
  callable that benchmarks on the local executor.

Example::

    import compgen

    dev = compgen.device("hardware_specs/my_chip.yaml")
    compiled = compgen.compile_model(model, dev, objective="latency")
    result = compiled(sample_input)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from xdsl.dialects.builtin import ModuleOp

from compgen.agent.analyzer import GraphAnalysisDossier, NetworkAnalyzer
from compgen.agent.loop import AgenticCompilationLoop, CompilationResult
from compgen.agent.env import CompilerEnv
from compgen.capture.torch_export import CaptureArtifact, capture_frontend_artifact
from compgen.eqsat.pipeline import EqSatResult, run_eqsat_pass
from compgen.ir.payload.import_fx import ImportDiagnostic, fx_to_xdsl
from compgen.llm.base import CompGenLLMProtocol
from compgen.runtime.local_executor import BenchmarkResult, LocalExecutor
from compgen.stages.registry import PipelineResult, StageRegistry, TargetDialectStack
from compgen.targetgen.generate import GeneratedTarget, generate_target
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.package import TargetPackage, generate_target_package
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass(frozen=True)
class CompGenDevice:
    """Handle representing a target device ready for compilation.

    Created by :func:`device`.  Holds everything the compilation pipeline
    needs to know about the hardware target: the full hardware spec, the
    extracted :class:`~compgen.targets.schema.TargetProfile`, inferred
    :class:`~compgen.targets.capability.CapabilitySpec`, and the generated
    :class:`~compgen.stages.registry.TargetDialectStack`.

    Attributes:
        spec: The loaded HardwareSpec.
        profile: Extracted TargetProfile (backward-compat view).
        capabilities: Inferred CapabilitySpec (what the target can do).
        dialect_stack: Generated TargetDialectStack for the stage pipeline.
        generated_target: Full GeneratedTarget result from targetgen.
        target_package: Optional composed target package with extension packs.
    """

    spec: HardwareSpec
    profile: TargetProfile
    capabilities: CapabilitySpec
    dialect_stack: TargetDialectStack
    generated_target: GeneratedTarget
    target_package: TargetPackage | None = None


@dataclass(frozen=True)
class CompiledModel:
    """A compiled model ready for local benchmarking.

    Created by :func:`compile_model`.  Calling an instance runs the
    original PyTorch model through :class:`~compgen.runtime.local_executor.LocalExecutor`
    and returns a :class:`~compgen.runtime.local_executor.BenchmarkResult`.

    Attributes:
        model: The original PyTorch model.
        device: The CompGenDevice used for compilation.
        objective: Optimisation objective (e.g. ``"latency"``).
        capture_artifact: Canonical frontend boundary data.
        analysis_dossier: Deterministic graph analysis from the prepared export.
        pipeline_result: Result from the stage pipeline.
        eqsat_result: Result from the equality-saturation pass.
        import_diagnostics: Diagnostics from FX-to-xDSL import.
    """

    model: nn.Module
    device: CompGenDevice
    objective: str
    capture_artifact: CaptureArtifact
    analysis_dossier: GraphAnalysisDossier | None
    payload_module: ModuleOp
    pipeline_result: PipelineResult
    eqsat_result: EqSatResult
    sample_inputs: tuple[Any, ...]
    import_diagnostics: list[ImportDiagnostic] = field(default_factory=list)
    #: P2 completion — when ``compile_model`` is invoked with a
    #: ``drive_loop=PhasedDriveLoop(...)`` argument, the result of that
    #: drive-loop run is captured here. ``None`` otherwise.
    drive_loop_result: Any = None
    #: Populated when ``compile_model`` is called with
    #: ``recover_unsupported=True``. Gives per-op recovery decisions.
    recovery_plan: Any = None

    def __call__(
        self,
        *args: Any,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> BenchmarkResult:
        """Benchmark the model on the local executor.

        Args:
            *args: Input tensors for the model.
            num_iterations: Number of timed iterations.
            warmup: Warmup iterations before timing.

        Returns:
            BenchmarkResult with real hardware measurements.
        """
        executor = LocalExecutor()
        return executor.benchmark(
            model=self.model,
            sample_inputs=args,
            device="cpu",
            mode="eager",
            num_iterations=num_iterations,
            warmup=warmup,
        )

    def create_agent_env(self, *, budget: int = 50) -> CompilerEnv:
        """Create a pack-aware agent environment from this compiled model."""

        env = CompilerEnv()
        target_package = self.device.target_package
        pack_context = target_package.pack_context() if target_package is not None else None
        loaded_packs = target_package.extension_packs if target_package is not None else ()
        env.reset(
            self.payload_module.clone(),
            self.device.profile,
            objective=self.objective,
            budget=budget,
            exported_program=self.capture_artifact.exported_program,
            capture_artifact=self.capture_artifact,
            pack_context=pack_context,
            loaded_packs=loaded_packs,
            pytorch_model=self.model,
            sample_inputs=self.sample_inputs,
        )
        return env

    def run_agentic(
        self,
        llm_client: CompGenLLMProtocol,
        *,
        budget: int = 10,
        with_recipe: bool = True,
    ) -> CompilationResult:
        """Run the agentic loop starting from this compiled model's payload IR."""

        env = self.create_agent_env(budget=budget)
        loop = AgenticCompilationLoop(llm_client=llm_client, env=env, budget=budget)
        if with_recipe:
            return loop.run_with_recipe(self.device.profile)
        return loop.run(self.device.profile)


def device(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    packs: tuple[str | Path, ...] | None = None,
) -> CompGenDevice:
    """Load a HardwareSpec YAML and generate a target device.

    Args:
        spec_path: Path to a HardwareSpec YAML file.
        output_dir: Directory for generated artifacts.  Defaults to
            ``compgen_output/<spec_name>`` relative to ``spec_path``'s parent.

    Returns:
        A :class:`CompGenDevice` ready for use with :func:`compile_model`.

    Raises:
        FileNotFoundError: If ``spec_path`` does not exist.
        ValueError: If the hardware spec fails validation.
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Hardware spec not found: {spec_path}")

    if output_dir is None:
        output_dir = spec_path.parent / "compgen_output" / spec_path.stem

    log.info("api.device.loading", spec_path=str(spec_path))

    generated = generate_target(spec_path, output_dir)

    log.info(
        "api.device.ready",
        name=generated.profile.name,
        target_class=generated.capabilities.target_class.value,
        stages=len(generated.dialect_stack.stages),
    )

    target_package = None
    if packs:
        package_dir = Path(output_dir) / "package"
        target_package = generate_target_package(
            generated.profile,
            package_dir,
            extension_packs=packs,
        )

    return CompGenDevice(
        spec=generated.spec,
        profile=generated.profile,
        capabilities=generated.capabilities,
        dialect_stack=generated.dialect_stack,
        generated_target=generated,
        target_package=target_package,
    )


def compile_model(
    model: nn.Module,
    target_device: CompGenDevice,
    objective: str = "latency",
    sample_inputs: tuple[Any, ...] | None = None,
    drive_loop: Any = None,
    drive_loop_phases: tuple[int, ...] = (2, 3),
    recover_unsupported: bool = False,
    recovery_llm_client: Any = None,
) -> CompiledModel:
    """Capture, optimise, and compile a PyTorch model for a target device.

    Pipeline:
        1. Capture via ``torch.export``
        2. Convert FX graph to xDSL Payload IR
        2.5. (Optional) Run the LLM-driven :class:`PhasedDriveLoop` —
             tools + invent-slots registered into
             :func:`compgen.llm.registry.get_registry`.
        3. Run equality-saturation optimisation
        4. Run the target's stage pipeline

    Args:
        model: A PyTorch ``nn.Module``.
        target_device: A :class:`CompGenDevice` from :func:`device`.
        objective: Optimisation objective (``"latency"`` or ``"throughput"``).
        sample_inputs: Sample inputs for export.  If ``None``, a default
            ``(torch.randn(1, 64),)`` tensor is used.
        drive_loop: **P2 completion** (see
            ``user_perspective/reports/stage_b_fourth_wave_status.md``).
            When not ``None`` (typically a
            :class:`compgen.agent.loop.PhasedDriveLoop`), the drive loop
            runs between the FX→xDSL import and equality saturation.
            Backward-compatible: ``None`` preserves legacy behaviour.
        drive_loop_phases: Which phases the drive loop iterates over
            (default: ``(2, 3)`` — semantic opt + placement/layout).
            Ignored when ``drive_loop`` is ``None``.

    Returns:
        A :class:`CompiledModel` callable for benchmarking.
    """
    if sample_inputs is None:
        sample_inputs = (torch.randn(1, 64),)

    log.info(
        "api.compile.start",
        model=type(model).__name__,
        target=target_device.profile.name,
        objective=objective,
    )

    # Stage 0: Capture
    log.info("api.compile.capture")
    capture_artifact = capture_frontend_artifact(model, sample_inputs)

    # Stage 0.5: Optional LLM-driven unsupported-op recovery.
    recovery_plan_obj: Any = None
    if recover_unsupported and capture_artifact.unsupported_resolutions:
        from compgen.agent.llm_driver_recovery import plan_recovery

        log.info(
            "api.compile.recover_unsupported",
            num_issues=len(capture_artifact.unsupported_resolutions),
            have_llm=recovery_llm_client is not None,
        )
        recovery_plan_obj = plan_recovery(
            capture_artifact, llm_client=recovery_llm_client,
        )
        log.info(
            "api.compile.recover_unsupported.done",
            ok=recovery_plan_obj.ok(),
            llm_consulted=recovery_plan_obj.llm_consulted,
            by_strategy={k: len(v) for k, v in recovery_plan_obj.by_strategy().items()},
        )

    # Stage 1: FX -> xDSL Payload IR
    log.info("api.compile.import_fx")
    module, diagnostics = fx_to_xdsl(
        capture_artifact.exported_program,
        **capture_artifact.strict_import_options(),
    )

    # Stage 1.5: Graph dossier
    log.info("api.compile.analyze")
    analysis = NetworkAnalyzer().analyze(
        capture_artifact.exported_program,
        target_device.profile,
        model_name=type(model).__name__,
    )

    # Stage 1.75: Annotate ops with ukernel matches
    log.info("api.compile.ukernel_annotate")
    from compgen.ir.ukernel.builtins import build_default_registry
    from compgen.ir.ukernel.annotate import annotate_ukernel_ops

    ukernel_registry = build_default_registry()
    # Extract target features for ukernel matching
    _target_features: set[str] = set()
    for _dev in target_device.profile.devices:
        for _cu in _dev.compute_units:
            _target_features.add(f"has_{_cu.name}")
        for _feat in getattr(_dev, "features", []):
            _target_features.add(f"has_{_feat}")
    _device_type = target_device.profile.devices[0].device_type if target_device.profile.devices else ""

    ukernel_annotations = annotate_ukernel_ops(
        module,
        ukernel_registry,
        target_features=frozenset(_target_features),
        device_type=_device_type,
    )
    log.info("api.compile.ukernel_annotate.done", annotated=ukernel_annotations)

    # Stage 1.9: Optional LLM drive loop (P2 completion).
    drive_loop_result: Any = None
    if drive_loop is not None:
        log.info("api.compile.drive_loop", phases=list(drive_loop_phases))
        # The caller owns the policy. We accept either an object with a
        # ``.run(phases=, policy=)`` method (PhasedDriveLoop) or a plain
        # callable ``drive_loop(module, phases) -> result``.
        if hasattr(drive_loop, "run"):
            # Default policy: no-op (empty steps per phase). Real callers
            # set ``drive_loop.context["policy"]`` or subclass.
            def _default_policy(_phase: int, _registry: Any, _ctx: dict[str, Any]) -> list:
                return []
            policy = (drive_loop.context or {}).get("policy", _default_policy)
            drive_loop_result = drive_loop.run(
                phases=list(drive_loop_phases), policy=policy
            )
        else:
            drive_loop_result = drive_loop(module, drive_loop_phases)
        log.info("api.compile.drive_loop.done", result_summary=type(drive_loop_result).__name__)

    # Stage 2: Equality saturation
    log.info("api.compile.eqsat")
    eqsat_result = run_eqsat_pass(module)

    # Stage 3+: Target pipeline
    log.info("api.compile.pipeline")
    registry = StageRegistry()
    stack = target_device.dialect_stack
    stack.target_name = target_device.profile.name
    registry.register_target_stack(stack)

    pipeline_result = registry.run_pipeline(
        module,
        target_device.profile,
        target_device.capabilities,
    )

    log.info(
        "api.compile.done",
        pipeline_passed=pipeline_result.passed,
        stages_run=pipeline_result.stages_run,
        eqsat_changed=eqsat_result.changed,
    )

    return CompiledModel(
        model=model,
        device=target_device,
        objective=objective,
        capture_artifact=capture_artifact,
        analysis_dossier=analysis.dossier,
        payload_module=module.clone(),
        pipeline_result=pipeline_result,
        eqsat_result=eqsat_result,
        sample_inputs=sample_inputs,
        import_diagnostics=diagnostics,
        drive_loop_result=drive_loop_result,
        recovery_plan=recovery_plan_obj,
    )


__all__ = [
    "CompGenDevice",
    "CompiledModel",
    "compile_model",
    "device",
]
