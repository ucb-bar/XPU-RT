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

from compgen.agent.analyzer import GraphAnalysisDossier, NetworkAnalyzer
from compgen.capture.torch_export import CaptureArtifact, capture_frontend_artifact
from compgen.eqsat.pipeline import EqSatResult, run_eqsat_pass
from compgen.ir.payload.import_fx import ImportDiagnostic, fx_to_xdsl
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
    pipeline_result: PipelineResult
    eqsat_result: EqSatResult
    import_diagnostics: list[ImportDiagnostic] = field(default_factory=list)

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
) -> CompiledModel:
    """Capture, optimise, and compile a PyTorch model for a target device.

    Pipeline:
        1. Capture via ``torch.export``
        2. Convert FX graph to xDSL Payload IR
        3. Run equality-saturation optimisation
        4. Run the target's stage pipeline

    Args:
        model: A PyTorch ``nn.Module``.
        target_device: A :class:`CompGenDevice` from :func:`device`.
        objective: Optimisation objective (``"latency"`` or ``"throughput"``).
        sample_inputs: Sample inputs for export.  If ``None``, a default
            ``(torch.randn(1, 64),)`` tensor is used.

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
        pipeline_result=pipeline_result,
        eqsat_result=eqsat_result,
        import_diagnostics=diagnostics,
    )


__all__ = [
    "CompGenDevice",
    "CompiledModel",
    "compile_model",
    "device",
]
