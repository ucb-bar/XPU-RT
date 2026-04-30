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
from compgen.agent.env import CompilerEnv
from compgen.agent.loop import AgenticCompilationLoop, CompilationResult
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


# Phase-7 sentinel — read by
# :func:`compgen.testing.etc_conformance._check_etc_routing_ready`.
# True when the conformance harness's ETC dispatch path is wired
# (workload factory + ``_compile_and_evaluate`` populated). The set
# of currently-wired workloads is enumerated by
# :data:`compgen.testing.workloads.WORKLOAD_FACTORIES`; workloads
# absent there fail loud rather than silently route through a stub.
_ETC_DISPATCH_READY: bool = True


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
        mode: str = "eager",
        device: str = "cpu",
    ) -> BenchmarkResult:
        """Benchmark the model on the local executor.

        Args:
            *args: Input tensors for the model.
            num_iterations: Number of timed iterations.
            warmup: Warmup iterations before timing.
            mode: Execution mode.  ``"eager"`` (default — original
                PyTorch model), ``"compiled"`` (via ``torch.compile``),
                or ``"compgen_ir"`` (runs the compiled xDSL payload
                through :func:`compgen.runtime.cpu_executor.execute`).
            device: ``"cpu"`` or ``"cuda"``.  ``mode="compgen_ir"`` is
                CPU-only today; it is silently routed to CPU if ``"cuda"``
                is requested.

        Returns:
            BenchmarkResult with real hardware measurements. When
            ``mode="compgen_ir"``, the result's ``sample_output`` can be
            diffed against an ``mode="eager"`` run for correctness.
        """
        executor = LocalExecutor()
        return executor.benchmark(
            model=self.model,
            sample_inputs=args,
            device=device,
            mode=mode,
            num_iterations=num_iterations,
            warmup=warmup,
            payload_module=self.payload_module if mode == "compgen_ir" else None,
            exported_program=(self.capture_artifact.exported_program if mode == "compgen_ir" else None),
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


@dataclass(frozen=True)
class MegakernelBundle:
    """A compiled Event Tensor Compiler bundle ready for dispatch.

    Returned by :func:`compile_to_megakernel`. Wraps the
    bundle directory + decision metadata + a ``dispatch`` callable.

    The agentic-compilation contract: a PyPI user (or their agent)
    calls ``compile_to_megakernel(model, sample_inputs)`` with no
    flags and gets a callable bundle back. The agent doesn't need
    to know about cuBLASDx, cu13 NVRTC, or SM tags — those are
    resolved by :func:`compgen.runtime.autotune.probe_device` and
    surfaced in :attr:`backend_choice` for audit.

    Attributes:
        bundle_dir: Path to the on-disk bundle. Layout matches the
            artifact contract — ``megakernel/source.cu``,
            ``megakernel/manifest.yaml``, ``compile_context.json``.
        decision: :class:`compgen.runtime.lowering.LoweringDecision`
            output (pattern, per-op backend, schedule hints).
        backend_choice: Resolved
            :class:`compgen.runtime.autotune.BackendChoice` snapshot
            for the audit-via-MCP story. ``None`` when the user
            passed explicit flags rather than ``backend="auto"``.
        kernel_name: NVRTC entry-point symbol.
        manifest: Emitted manifest dict (queue tables, event-tensor
            allocs, launch config).
        elapsed_ms: Wall-clock cost of the compile.
    """

    bundle_dir: Path
    decision: dict[str, Any]
    backend_choice: dict[str, Any] | None
    kernel_name: str
    manifest: dict[str, Any]
    elapsed_ms: float
    #: Roofline cost prediction (etc_us / eager_us / passes_gate /
    #: per-component breakdown) emitted at compile time by
    #: :func:`compgen.kernels.cost.predict_etc_dispatch`. The agent
    #: queries ``cost_prediction["passes_gate"]`` to decide whether
    #: to dispatch through ETC or fall through to eager. ``None``
    #: when the prediction couldn't be computed.
    cost_prediction: dict[str, Any] | None = None

    def dispatch(self, *args: torch.Tensor) -> Any:
        """Run the compiled bundle on real input tensors.

        Wraps :func:`compgen.mcp.tools.compile.compgen_run_compiled_bundle`
        so Python callers don't need to go through the MCP tool's
        base64/pickle JSON interface. GPU-only — raises
        :class:`compgen.runtime.native.cuda.CudaUnavailableError`
        on hosts without ``libcompgen_rt-cuda.so`` reachable.

        Args:
            *args: Input tensors. Must match the shape the bundle
                was compiled for (the bundle's ``compile_context.json``
                pins ``sample_input_shape``).

        Returns:
            Whatever ``compgen_run_compiled_bundle`` returns —
            ``{"status": "ok", "outputs_pickle_b64": ..., "etc_us":
            ..., "eager_us": ..., "speedup_vs_eager": ..., ...}``.
            Errors land in ``status`` rather than raising.
        """
        import base64
        import pickle

        from compgen.mcp.tools.compile import compgen_run_compiled_bundle

        return compgen_run_compiled_bundle(
            bundle_dir=str(self.bundle_dir),
            input_pickle_b64=base64.b64encode(pickle.dumps(args)).decode(),
        )


def compile_to_megakernel(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    target: str = "auto",
    output_dir: str | Path | None = None,
    backend_overrides: dict[str, Any] | None = None,
    fail_when_wont_win: bool = False,
    perf_threshold: float = 1.2,
) -> MegakernelBundle:
    """Compile a torch ``nn.Module`` to an Event Tensor Compiler bundle.

    The flagless agentic-compilation entry. With default arguments
    everything is auto-detected:

    - **Backend selection** (NVRTC version, cuBLASDx availability,
      precision, SM tag, tile shape) via
      :func:`compgen.runtime.autotune.probe_device`.
    - **Pattern matching** (diamond, FFN, etc.) via
      :func:`compgen.runtime.lowering.lower_torch_to_megakernel`.
    - **Bundle layout** under ``output_dir`` per the artifact contract.

    The agent doesn't need to know about cuBLASDx, cu13 NVRTC, or
    SM tags — the probe handles those, and the resolved choice
    lands in :attr:`MegakernelBundle.backend_choice` for audit.

    Args:
        model: PyTorch ``nn.Module`` to compile. Must match a
            registered pattern (diamond, FFN); generic fallback is
            Wave 2 work.
        sample_inputs: Concrete input tensors. Their shape pins
            the bundle (no symbolic-shape support yet).
        target: ``"auto"`` (default) → probe the local device.
            Any other string is passed through as the NVRTC arch
            (``"sm_100"``, ``"sm_90"``, etc.) for cross-compilation.
        output_dir: Where to write the bundle. ``None`` (default)
            → ``./compgen_output/megakernel_<short-uuid>``.
        backend_overrides: Optional dict of per-flag overrides on
            top of the probe (e.g. ``{"cublasdx_precision": "fp32"}``
            to force fp32 even on Blackwell). Each non-None key
            replaces the probe's value; None / missing keys keep
            the probe's choice. The agent uses this when it needs
            to pin one knob without giving up the rest of
            auto-detection.
        fail_when_wont_win: When True, raise
            :class:`compgen.kernels.cost.WontWinError` if the
            roofline cost model predicts ETC won't beat eager by
            ``perf_threshold``. Default False — the bundle is still
            emitted with the prediction stamped in
            :attr:`MegakernelBundle.cost_prediction` so the agent
            can decide whether to dispatch ETC or fall through to
            eager.
        perf_threshold: Speedup gate (eager_us / etc_us) for the
            cost-model decision. Default 1.2× per the conformance
            harness.

    Returns:
        :class:`MegakernelBundle` callable for ``.dispatch(*args)``.

    Raises:
        UnsupportedShape: No registered pattern matched the model.
            The exception message lists every matcher's reason. (A
            generic FX→megakernel fallback that handles any model
            is Wave 2.2.)
    """
    import time
    import uuid
    from pathlib import Path as _Path

    from compgen.runtime.autotune import probe_device
    from compgen.runtime.lowering import (
        lower_torch_to_megakernel,
    )
    from compgen.transforms.emit_cuda_megakernel import emit_cuda_megakernel
    from compgen.transforms.event_static_schedule import compute_static_schedule

    t0 = time.perf_counter()

    if output_dir is None:
        out = _Path("compgen_output") / f"megakernel_{uuid.uuid4().hex[:8]}"
    else:
        out = _Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Probe → BackendChoice.
    choice = probe_device(target=target)
    if backend_overrides:
        import dataclasses

        # Translate user-facing override keys to BackendChoice fields.
        # We accept the same keys the MCP tool accepts so callers can
        # use one mental model.
        normalised: dict[str, Any] = {}
        if "prefer_cublasdx_for_linears" in backend_overrides:
            normalised["use_cublasdx_for_linears"] = bool(backend_overrides["prefer_cublasdx_for_linears"])
        for key in ("cublasdx_precision", "use_cu13_nvrtc"):
            if key in backend_overrides:
                normalised[key] = backend_overrides[key]
        if "target_arch" in backend_overrides:
            normalised["target_arch"] = backend_overrides["target_arch"]
        if normalised:
            choice = dataclasses.replace(choice, **normalised)

    # 2. Lower torch → MegakernelGraph.
    result = lower_torch_to_megakernel(
        model,
        sample_inputs,
        backend_choice=choice,
    )

    # Wave 1.8 — when the matcher accepted via the submodule
    # fallback (decision.submodule_path is set), the bundle must
    # store the SUBMODULE not the wrapper. Otherwise dispatch's
    # weight-extraction (``effective_model.up.weight`` etc.) blows
    # up because the wrapper doesn't have ``up`` directly.
    # Substitute now, before we pickle into compile_context.json.
    effective_model: nn.Module = model
    submodule_path = getattr(result.decision, "submodule_path", "") or ""
    if submodule_path:
        effective_model = model.get_submodule(submodule_path)

    # 3. Schedule + emit + write bundle.
    sm_count = _resolve_sm_count_for_target(choice.target_arch)
    # Wave 1.6 — cluster-launch wiring. When the probe set
    # ``supports_clusters=True`` (Blackwell), pass the chosen
    # cluster shape to the static schedule. Multi-block-per-task
    # cooperation is the structural fix bridge #108 identified for
    # the cooperative-grid-sync overhead. ``None`` for non-Blackwell
    # keeps the legacy single-block-per-task path.
    cluster_dim: tuple[int, int, int] | None
    if choice.supports_clusters and choice.cluster_dim_x is not None:
        cluster_dim = (
            choice.cluster_dim_x,
            choice.cluster_dim_y or 1,
            choice.cluster_dim_z or 1,
        )
    else:
        cluster_dim = None

    schedule = compute_static_schedule(
        result.megakernel_graph,
        sm_count=sm_count,
        block_dim=(32, 32, 1),
        supports_clusters=choice.supports_clusters,
        cluster_dim=cluster_dim,
    )
    emit = emit_cuda_megakernel(
        schedule,
        device_function_sources=result.device_function_sources,
        user_buffer_count=len(result.user_buffer_layout),
    )

    bundle_dir = out / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    emit.write_to_bundle(bundle_dir / "megakernel")

    import base64
    import json
    import pickle

    (bundle_dir / "compile_context.json").write_text(
        json.dumps(
            {
                "user_buffer_layout": list(result.user_buffer_layout),
                "target_arch": choice.target_arch,
                "sample_input_shape": list(sample_inputs[0].shape),
                "decision": result.decision.to_dict(),
                "kernel_name": emit.kernel_name,
                "model_pickle_b64": base64.b64encode(pickle.dumps(effective_model)).decode(),
                "submodule_path": submodule_path,
                "wrapper_class": type(model).__name__,
                "nvrtc_include_paths": list(result.decision.nvrtc_include_paths),
                "nvrtc_extra_options": list(result.decision.nvrtc_extra_options),
                "prefer_cublasdx_for_linears": choice.use_cublasdx_for_linears,
                "cublasdx_precision": choice.cublasdx_precision,
                "use_cu13_nvrtc": choice.use_cu13_nvrtc,
                "backend_choice": choice.to_dict(),
                "backend_mode": "auto" if not backend_overrides else "auto+overrides",
            },
            indent=2,
        )
    )

    # 4. Cost prediction (Wave 1.3) — predict whether ETC will beat
    # eager. Stamp the prediction into the bundle so the agent's
    # dispatch decision is auditable. When fail_when_wont_win=True,
    # raise so the caller can fall through to eager.
    from compgen.kernels.cost import (
        WontWinError,
        predict_etc_dispatch,
    )

    # Model dtype drives the eager-rate lookup (per bridge #121):
    # ``cublasdx_precision`` is the compgen path's compute precision
    # (bf16+fp32-acc on Blackwell), but eager runs the user's model
    # in its parameter dtype. Sniff the dominant linear-weight dtype
    # so eager_us reflects what cuBLAS actually delivers for THIS
    # model. Falls back to fp32 (torch's default) when the model has
    # no parameters or the sample input drives the choice.
    def _sniff_model_dtype(m: nn.Module, samples: tuple[torch.Tensor, ...]) -> str:
        try:
            params = list(m.parameters())
            if params:
                dt = params[0].dtype
            else:
                dt = samples[0].dtype if samples else torch.float32
        except Exception:  # noqa: BLE001
            dt = torch.float32
        if dt == torch.float32:
            return "fp32"
        if dt in (torch.bfloat16,):
            return "bf16"
        if dt in (torch.float16,):
            return "fp16"
        if dt == torch.float64:
            return "fp32"  # cuBLAS routes fp64 GEMM via fp32 path for our purposes
        if hasattr(torch, "float8_e4m3fn") and dt == getattr(torch, "float8_e4m3fn"):
            return "fp8"
        if hasattr(torch, "float8_e5m2") and dt == getattr(torch, "float8_e5m2"):
            return "fp8"
        return "fp32"

    model_dtype = _sniff_model_dtype(model, sample_inputs)

    cost_prediction_dict: dict[str, Any] | None = None
    try:
        prediction = predict_etc_dispatch(
            sample_input_shape=tuple(sample_inputs[0].shape),
            decision=result.decision.to_dict(),
            backend_choice=choice.to_dict(),
            threshold=perf_threshold,
            model_dtype=model_dtype,
        )
        cost_prediction_dict = {
            "etc_us": prediction.etc_us,
            "eager_us": prediction.eager_us,
            "speedup": prediction.speedup,
            "threshold": prediction.threshold,
            "passes_gate": prediction.passes_gate,
            "components": prediction.components,
            "reason": prediction.reason,
        }
        # Stamp the prediction into the bundle's verification report
        # so compgen_run_compiled_bundle + the agent's audit query
        # see it without re-running the predictor.
        (bundle_dir / "verification_report.json").write_text(
            json.dumps({"cost_prediction": cost_prediction_dict}, indent=2)
        )
        # Per bridge #129: also surface the prediction inside
        # ``compile_context.json`` so the dispatch path
        # (``compgen_run_compiled_bundle``) and any audit-via-MCP query
        # can read components like ``intra_cluster_edge_fraction``,
        # ``num_linear_waves``, ``num_pointwise_waves`` without having
        # to re-run the predictor or load a second JSON file. Read +
        # update + write so a predictor crash earlier above doesn't
        # leave the context file half-written.
        ctx_path = bundle_dir / "compile_context.json"
        try:
            ctx = json.loads(ctx_path.read_text())
            ctx["cost_prediction"] = cost_prediction_dict
            ctx_path.write_text(json.dumps(ctx, indent=2))
        except (OSError, ValueError) as ctx_exc:  # noqa: BLE001
            log.warning(
                "compgen.compile_to_megakernel.cost_predict_ctx_update_failed",
                error=repr(ctx_exc),
            )
        if fail_when_wont_win and not prediction.passes_gate:
            raise WontWinError(prediction)
    except WontWinError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Cost prediction is best-effort. Never fail the compile
        # because the predictor couldn't compute (e.g. shape decoder
        # has gaps). Log and continue.
        log.warning(
            "compgen.compile_to_megakernel.cost_predict_failed",
            error=repr(exc),
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "compgen.compile_to_megakernel.ok",
        pattern=result.decision.pattern_name,
        target_arch=choice.target_arch,
        use_cublasdx=choice.use_cublasdx_for_linears,
        bundle_dir=str(bundle_dir),
        elapsed_ms=elapsed_ms,
        passes_perf_gate=(cost_prediction_dict["passes_gate"] if cost_prediction_dict else None),
    )

    return MegakernelBundle(
        bundle_dir=bundle_dir,
        decision=result.decision.to_dict(),
        backend_choice=choice.to_dict(),
        kernel_name=emit.kernel_name,
        manifest=emit.manifest,
        elapsed_ms=elapsed_ms,
        cost_prediction=cost_prediction_dict,
    )


def _resolve_sm_count_for_target(target_arch: str) -> int:
    """SM count default for the static scheduler. Production code
    will probe live; for compile-time scheduling we use a per-arch
    default that's correct for the canonical hardware.

    - sm_100 (B100/B200 datacenter): 132 SMs.
    - sm_120 (workstation Blackwell): 188 SMs.
    - sm_90 (H100): 132 SMs.
    - else: 80 (conservative).
    """
    a = target_arch.lower().lstrip("sm_").rstrip("a")
    return {
        "100": 132,
        "120": 188,
        "90": 132,
        "89": 128,
        "86": 84,
        "80": 108,
    }.get(a, 80)


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


def _run_inline_verification(
    *,
    module: ModuleOp,
    target_profile: TargetProfile,
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    bundle_dir: Path,
    exported_program: Any | None = None,
) -> dict[str, Any]:
    """Run the verify stage on the compiled module and return its payload.

    ``compile_model(verify=True)`` calls this so ``verification_report.json``
    is always part of a production bundle. The returned dict is what
    gets serialised into ``verification_report.json``; shape mirrors
    ``TransformVerificationResult`` but adds the per-level strings from
    the ``details`` dict so the JSON is self-describing.

    Failure of any verify level does not raise — the caller decides
    whether a failed verify is fatal. But every level that ran is
    reported; nothing is silently "PASSED".
    """
    from compgen.transforms.verify import (
        TransformVerifier,
        VerificationLevel,
    )

    verifier = TransformVerifier(
        levels=[
            VerificationLevel.STRUCTURAL,
            VerificationLevel.DIFFERENTIAL,
            VerificationLevel.NUMERIC,
        ]
    )
    result = verifier.verify(
        module,
        module,
        sample_inputs,
        model=model,
        exported_program=exported_program,
    )
    return {
        "target": target_profile.name,
        "bundle_dir": str(bundle_dir),
        "passed": bool(result.passed),
        "max_abs_error": float(result.max_abs_error) if result.max_abs_error is not None else None,
        "levels_run": [lvl.value for lvl in result.levels_run],
        "levels_passed": [lvl.value for lvl in result.levels_passed],
        "details": dict(result.details),
    }


def compile_model(
    model: nn.Module,
    target_device: CompGenDevice,
    objective: str = "latency",
    sample_inputs: tuple[Any, ...] | None = None,
    drive_loop: Any = None,
    drive_loop_phases: tuple[int, ...] = (2, 3),
    recover_unsupported: bool = False,
    recovery_llm_client: Any = None,
    *,
    output_dir: str | Path | None = None,
    dump_ir: bool | None = None,
    session_id: str | None = None,
    verify: bool = True,
    strict_artifacts: bool = True,
    run_compile_baseline: bool = True,
    memory: Any = None,
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

    # Resolve output_dir + session_id and install the trace bus + IR-dump
    # writer before doing anything else, so capture / import / pipeline all
    # emit into the same linked trace. Local imports avoid a module-load
    # cycle with :mod:`compgen.trace.adapters`.
    from compgen.mcp.transcript import default_session_root
    from compgen.trace import (
        IRDumpWriter,
        build_session_id,
        dump_enabled_from_env,
        install_bus,
        install_ir_dump_writer,
    )

    if session_id is None:
        # Human-readable id: YYYYMMDD-HHMMSS_<model>_<target>_<short>.
        # Sorts by wall clock, identifies the run at a glance, stays
        # unique via the random suffix.
        session_id = build_session_id(model=model, target_device=target_device)
    resolved_out_dir: Path
    if output_dir is None:
        resolved_out_dir = Path("compgen_output") / session_id
    else:
        resolved_out_dir = Path(output_dir).expanduser()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    session_mirror = default_session_root() / session_id / "trace.jsonl"
    # Always install a fresh bus for this compile. The process-level
    # fallback used by MCP async handlers would otherwise cause a
    # previous compile's bus (from session N-1) to capture this
    # compile's events, dropping trace files for every compile after
    # the first. ``install_bus`` both sets the ContextVar and updates
    # the process-level bus, so subsequent MCP calls in this same
    # process find the latest bus.
    install_bus(
        output_dir=resolved_out_dir,
        session_id=session_id,
        session_mirror=session_mirror,
    )
    dump_enabled = bool(dump_ir) if dump_ir is not None else dump_enabled_from_env()
    ir_dump_writer = IRDumpWriter(resolved_out_dir, enabled=dump_enabled)
    install_ir_dump_writer(ir_dump_writer)

    # Install a decision registry for this compile so stage plugins can
    # enqueue their choices and the agent can read/override them via MCP
    # tools. Re-use any registry already installed for this context
    # (e.g. from an MCP session that pre-loaded the model).
    from compgen.agent.decisions import (
        DecisionRegistry,
        get_active_registry,
        install_registry,
    )

    if get_active_registry() is None:
        install_registry(DecisionRegistry())

    log.info(
        "api.compile.start",
        model=type(model).__name__,
        target=target_device.profile.name,
        objective=objective,
        output_dir=str(resolved_out_dir),
        session_id=session_id,
        dump_ir=dump_enabled,
    )

    # Every top-level compile step fires a ``pass_run`` span so the
    # trace carries true per-pass granularity (gap #2). Each span also
    # measures its own duration and records before/after IR dumps for
    # operations that mutate the module.
    from compgen.trace import (
        AnalysisPublisher,
        DecisionPublisher,
        PassPublisher,
        get_ir_dump_writer,
    )

    def _traced_step(
        name: str,
        fn,
        *,
        module_before: ModuleOp | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Run ``fn`` inside a ``pass_run`` span with IR before/after dumps.

        ``module_before`` is optional — when supplied and ``fn`` mutates
        (returns) a module, we dump both sides and record the hash delta
        in the span's end payload. Measured duration is real.
        """
        import time as _time

        dumper = get_ir_dump_writer()
        hash_before = ""
        with PassPublisher.span(
            payload={"name": name, "source": "api.compile_model", **(extra or {})},
        ) as span_id:
            t0 = _time.time()
            if dumper is not None and module_before is not None:
                _, hash_before = dumper.dump(
                    name=name, phase="before", module=module_before, trace_event_id=span_id or ""
                )
            result = fn()
            result_module = result if isinstance(result, ModuleOp) else module_before
            elapsed_ms = (_time.time() - t0) * 1000.0
            hash_after = ""
            if dumper is not None and result_module is not None:
                _, hash_after = dumper.dump(
                    name=name,
                    phase="after",
                    module=result_module,
                    trace_event_id=span_id or "",
                    duration_ms=elapsed_ms,
                )
            # Emit a terminal point event so consumers can grep for
            # "pass_run" + "phase: point" without reconstructing start/end pairs.
            from compgen.trace import EventKind, Phase, get_active_bus

            bus = get_active_bus()
            if bus is not None:
                bus.publish(
                    kind=EventKind.PASS_RUN.value,
                    phase=Phase.POINT.value,
                    parent_event_id=span_id or "",
                    elapsed_ms=elapsed_ms,
                    payload={
                        "name": name,
                        "source": "api.compile_model",
                        "ir_hash_before": hash_before,
                        "ir_hash_after": hash_after,
                        "duration_ms": elapsed_ms,
                        "span_id": span_id or "",
                    },
                )
            return result

    # Stage 0: Capture
    log.info("api.compile.capture")
    capture_artifact = _traced_step(
        "capture_frontend",
        lambda: capture_frontend_artifact(model, sample_inputs),
    )

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
            capture_artifact,
            llm_client=recovery_llm_client,
        )
        log.info(
            "api.compile.recover_unsupported.done",
            ok=recovery_plan_obj.ok(),
            llm_consulted=recovery_plan_obj.llm_consulted,
            by_strategy={k: len(v) for k, v in recovery_plan_obj.by_strategy().items()},
        )

    # Stage 1: FX -> xDSL Payload IR
    log.info("api.compile.import_fx")

    def _run_fx_to_xdsl():
        return fx_to_xdsl(
            capture_artifact.exported_program,
            **capture_artifact.strict_import_options(),
        )

    module, diagnostics = _traced_step("fx_to_xdsl", _run_fx_to_xdsl)

    # Stage 1.5: Graph dossier
    log.info("api.compile.analyze")
    with AnalysisPublisher.span(payload={"analysis": "NetworkAnalyzer", "target": target_device.profile.name}):
        analysis = NetworkAnalyzer().analyze(
            capture_artifact.exported_program,
            target_device.profile,
            model_name=type(model).__name__,
        )
    AnalysisPublisher.emit(
        analysis="NetworkAnalyzer",
        clusters=len(getattr(analysis, "clusters", []) or []),
        unclustered=len(getattr(analysis, "unclustered_ops", []) or []),
        opportunities=list(getattr(analysis, "optimization_opportunities", []) or []),
    )
    # First real decision the compiler makes: the target class (drives
    # whether triton/baremetal/ukernel backends are used). Surface it
    # to the trace so the chain "target_class -> stage plugin -> kernel
    # output" is auditable. Guarded against test stubs where
    # ``capabilities`` is None.
    _cap = getattr(target_device, "capabilities", None)
    if _cap is not None and getattr(_cap, "target_class", None) is not None:
        DecisionPublisher.emit(
            decision_type="target_class",
            chosen=_cap.target_class.value,
            candidates=[c.value for c in type(_cap.target_class)],
            rationale=f"capabilities derived from {target_device.profile.name!r}",
        )

    # Stage 1.75: Annotate ops with ukernel matches
    log.info("api.compile.ukernel_annotate")
    from compgen.ir.ukernel.annotate import annotate_ukernel_ops
    from compgen.ir.ukernel.builtins import build_default_registry

    ukernel_registry = build_default_registry()
    # Extract target features for ukernel matching
    _target_features: set[str] = set()
    for _dev in target_device.profile.devices:
        for _cu in _dev.compute_units:
            _target_features.add(f"has_{_cu.name}")
        for _feat in getattr(_dev, "features", []):
            _target_features.add(f"has_{_feat}")
    _device_type = target_device.profile.devices[0].device_type if target_device.profile.devices else ""

    ukernel_annotations = _traced_step(
        "ukernel_annotate",
        lambda: annotate_ukernel_ops(
            module,
            ukernel_registry,
            target_features=frozenset(_target_features),
            device_type=_device_type,
        ),
        module_before=module,
        extra={"target_features": sorted(_target_features), "device_type": _device_type},
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
            drive_loop_result = drive_loop.run(phases=list(drive_loop_phases), policy=policy)
        else:
            drive_loop_result = drive_loop(module, drive_loop_phases)
        log.info("api.compile.drive_loop.done", result_summary=type(drive_loop_result).__name__)

    # Stage 2: Equality saturation
    log.info("api.compile.eqsat")
    eqsat_result = _traced_step(
        "eqsat",
        lambda: run_eqsat_pass(module),
        module_before=module,
    )

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

    # Emit the final glued module (kernels remain separate under
    # ``generated_kernels/``). No-op when ``dump_ir`` is False.
    ir_dump_writer.write_final(module)
    # Detach so subsequent compiles get a fresh writer.
    install_ir_dump_writer(None)

    # Extended artefact emission — every failure surfaces.
    # The bundle stage writes payload.mlir + manifest.json; the call
    # below fills in the rest of the 14-artifact contract and returns
    # a BundleEmissionReport. We honestly propagate per-artifact
    # failures via manifest.json::extended_artifacts + the trace bus,
    # and raise BundleEmissionError if any artifact is in "failed"
    # status (unless the caller opts out via strict_artifacts=False).
    bundle_dir_str = pipeline_result.all_artifacts.get("bundle_dir") if pipeline_result else None
    emission_report = None
    if bundle_dir_str:
        from compgen.runtime.bundle_emit import emit_extended_artefacts
        from compgen.runtime.errors import BundleEmissionError

        # Gather optional inputs for the three slots that need upstream
        # data threaded through: transforms, generated_kernels, verify.
        pipeline_artifacts = pipeline_result.all_artifacts if pipeline_result is not None else None

        # When the auto-generated codegen stage couldn't ship a native
        # kernel (Triton declined / target_class isn't triton-friendly),
        # dispatch to any registered KernelProvider whose
        # accepts_contract() is True. The result feeds straight into
        # the ``generated_kernels`` artifact slot. Provider knowledge
        # is persisted into ``memory`` if the caller supplied one;
        # contract feedback is always surfaced via ``pipeline_artifacts``
        # so callers without memory can still consume it.
        if pipeline_artifacts is not None and not pipeline_artifacts.get("generated_kernels"):
            try:
                from compgen.kernels.codegen_fallback import run_provider_fallback

                feedback_buf: list[Any] = []
                fallback_kernels = run_provider_fallback(
                    module,
                    target_device.profile,
                    sample_inputs=sample_inputs,
                    memory=memory,
                    feedback_out=feedback_buf,
                )
                if fallback_kernels:
                    pipeline_artifacts["generated_kernels"] = fallback_kernels
                if feedback_buf:
                    pipeline_artifacts["provider_contract_feedback"] = [
                        {
                            "field": fb.field,
                            "current_value": fb.current_value,
                            "suggested_value": fb.suggested_value,
                            "reason": fb.reason,
                            "measured_gain": fb.measured_gain,
                        }
                        for fb in feedback_buf
                    ]
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "api.compile.codegen_fallback_failed",
                    error=str(exc),
                )

        verification_payload: dict[str, Any] | None = None
        if verify:
            verification_payload = _run_inline_verification(
                module=module,
                target_profile=target_device.profile,
                model=model,
                sample_inputs=sample_inputs,
                bundle_dir=Path(bundle_dir_str),
                exported_program=getattr(capture_artifact, "exported_program", None),
            )

        emission_report = emit_extended_artefacts(
            bundle_dir_str,
            capture_artifact=capture_artifact,
            sample_inputs=sample_inputs,
            model=model,
            run_compile_baseline=run_compile_baseline,
            payload_module=module,
            target_profile=target_device.profile,
            analysis=analysis,
            pipeline_artifacts=pipeline_artifacts,
            verification_report=verification_payload,
        )
        log.info(
            "api.compile.extended_artefact_report",
            ok=[s.name for s in emission_report.ok],
            failed=[s.name for s in emission_report.failed],
            skipped=[s.name for s in emission_report.skipped],
        )
        if strict_artifacts and emission_report.failed:
            raise BundleEmissionError(emission_report)
    else:
        log.warning(
            "api.compile.no_bundle_dir",
            hint="bundle stage did not expose a bundle_dir; extended artefacts skipped",
        )

    # Render the human-readable companion ``trace.log`` next to the
    # canonical NDJSON ``trace.jsonl``. Pure function of the JSONL —
    # nothing else in the pipeline reads it, so rendering once at the
    # end is enough and keeps the hot path free of string formatting.
    try:
        from compgen.trace import get_active_bus, render_trace

        bus_ref = get_active_bus()
        if bus_ref is not None:
            render_trace(bus_ref.trace_path)
    except Exception as _render_exc:  # noqa: BLE001
        log.debug("api.compile.trace_render_failed", error=str(_render_exc))

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


def compile_with_vendor(
    model: nn.Module,
    vendor_adapter: Any,
    *,
    sample_inputs: tuple[Any, ...] | None = None,
    output_dir: str | Path | None = None,
    options: dict[str, Any] | None = None,
) -> Any:
    """Drive end-to-end compilation of a PyTorch model onto a vendor dialect.

    ``vendor_adapter`` is an instance of
    :class:`compgen.extensions.vendor_dialect.VendorDialectAdapter` —
    typically obtained via
    :func:`compgen.extensions.vendor_dialect.get_adapter` after the
    user-space package has been ``pip install``-ed. The helper performs:

    1. Stage 0 capture (``torch.export``) and Stage 1 import (FX → Payload IR)
       using the same code paths as :func:`compile_model`.
    2. Serializes the Payload IR to text.
    3. Hands the text to ``vendor_adapter.compile(...)`` which drives
       the vendor-specific lowering + bundling.

    Args:
        model: PyTorch module.
        vendor_adapter: The registered vendor adapter.
        sample_inputs: Inputs for ``torch.export`` (default: random 1x64).
        output_dir: Where vendor artifacts are written. Defaults to a
            ``/tmp/compgen_<vendor>_<n>`` directory.
        options: Adapter-specific options forwarded to ``compile()``.

    Returns:
        A :class:`compgen.targets.backend.CompiledArtifact` with the
        vendor output and metadata.
    """
    from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter

    if not isinstance(vendor_adapter, VendorDialectAdapter):
        raise TypeError(f"vendor_adapter must be a VendorDialectAdapter, got {type(vendor_adapter).__name__}")
    if sample_inputs is None:
        sample_inputs = (torch.randn(1, 64),)

    log.info(
        "api.compile_with_vendor.start",
        model=type(model).__name__,
        vendor=vendor_adapter.name,
        target=vendor_adapter.target,
    )

    capture_artifact = capture_frontend_artifact(model, sample_inputs)
    module, _ = fx_to_xdsl(
        capture_artifact.exported_program,
        **capture_artifact.strict_import_options(),
    )
    payload_mlir = _module_to_text(module)

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(f"/tmp/compgen_{vendor_adapter.name}_{id(model)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "payload.mlir").write_text(payload_mlir)

    artifact = vendor_adapter.compile(payload_mlir, output_dir=out_dir, options=options or {})
    log.info(
        "api.compile_with_vendor.done",
        vendor=vendor_adapter.name,
        format=artifact.format,
        output_dir=str(out_dir),
    )
    return artifact


def _module_to_text(module: ModuleOp) -> str:
    """Best-effort xDSL ModuleOp → MLIR text."""
    from io import StringIO

    from xdsl.printer import Printer

    buf = StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


__all__ = [
    "CompGenDevice",
    "CompiledModel",
    "compile_model",
    "compile_with_vendor",
    "device",
]
