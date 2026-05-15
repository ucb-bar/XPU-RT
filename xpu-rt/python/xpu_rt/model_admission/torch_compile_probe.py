"""The admission probe: eager -> dynamo -> torch.compile.

Wraps :func:`compgen.capture.dynamo_baseline.compile_baseline` and
:func:`compgen.capture.dynamo_baseline.collect_diagnostics` from the
existing capture subsystem; it does not duplicate torch.compile plumbing.

The probe boundary never raises out: every loader / eager / dynamo /
compile failure is captured into a typed
:class:`~compgen.model_admission.schemas.AdmissionStatus` and a
human-readable reason. The only exit path is one of the eight admission
statuses.
"""

from __future__ import annotations

import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

from compgen.capture.dynamo_baseline import collect_diagnostics, compile_baseline
from compgen.model_admission import report
from compgen.model_admission.loaders import LoadedModel, LoaderUnavailable, load
from compgen.model_admission.schemas import (
    AdmissionReport,
    AdmissionStatus,
    DynamoCaptureReport,
    EagerReport,
    ExportReport,
    FxReport,
    ModelConfig,
    SliceConfig,
    StageStatus,
    TorchCompileReport,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """Bundle returned to suite runners. ``out_dir`` contains the JSON files."""

    admission: AdmissionReport
    eager: EagerReport
    fx: FxReport
    export: ExportReport
    dynamo: DynamoCaptureReport
    compile: TorchCompileReport
    out_dir: Path


def run_admission(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    out_dir: Path,
) -> ProbeResult:
    """Run the full eager -> dynamo -> torch.compile probe and write reports.

    Args:
        model_cfg: Parsed model config.
        slice_cfg: Optional slice config; if absent the full model is probed.
        out_dir: Output directory (created if missing).

    Returns:
        :class:`ProbeResult`. Always returns; never raises out of the probe
        boundary.
    """

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = model_cfg.model_id
    slice_id = slice_cfg.slice_id if slice_cfg is not None else ""

    eager = EagerReport(model_id=model_id, slice_id=slice_id, status=StageStatus.SKIPPED.value)
    fx = FxReport(model_id=model_id, slice_id=slice_id, status=StageStatus.SKIPPED.value)
    export = ExportReport(model_id=model_id, slice_id=slice_id, status=StageStatus.SKIPPED.value)
    dynamo = DynamoCaptureReport(model_id=model_id, slice_id=slice_id, status=StageStatus.SKIPPED.value)
    compile_rep = TorchCompileReport(
        model_id=model_id,
        slice_id=slice_id,
        attempted=False,
        status=StageStatus.SKIPPED.value,
        backend=model_cfg.compile.backend,
        fullgraph=model_cfg.compile.fullgraph,
        dynamic=model_cfg.compile.dynamic,
    )
    error_path: Path | None = None

    # --- environment snapshot.
    env_path = out_dir / "environment.json"
    report.write_environment(env_path)

    # --- input summary placeholder; overwritten on success.
    input_summary_path = out_dir / "input_summary.json"
    report.write_json(input_summary_path, {"model_id": model_id, "slice_id": slice_id})

    # --- 1. load.
    loaded: LoadedModel | None = None
    try:
        loaded = load(model_cfg, slice_cfg)
    except LoaderUnavailable as exc:
        log.info(
            "loader_unavailable",
            model_id=model_id,
            slice_id=slice_id,
            status=exc.status.value,
            reason=exc.reason,
        )
        if exc.error:
            error_path = out_dir / "error.txt"
            error_path.write_text(f"{exc.reason}\n\n{exc.error}\n", encoding="utf-8")
        admission = AdmissionReport(
            model_id=model_id,
            slice_id=slice_id,
            status=exc.status.value,
            reason=exc.reason,
            eager_report_path=str((out_dir / "eager_report.json").relative_to(out_dir)),
            torch_compile_report_path=str((out_dir / "torch_compile_report.json").relative_to(out_dir)),
            dynamo_report_path=str((out_dir / "dynamo_report.json").relative_to(out_dir)),
            environment_path=str(env_path.relative_to(out_dir)),
            error_path=str(error_path.relative_to(out_dir)) if error_path else None,
            recommended_next_step=_next_step_for_unavailable(exc.status),
            hardware_requirements=getattr(exc, "hardware_requirements", None),
        )
        report.write_admission(out_dir, admission, eager, fx, export, dynamo, compile_rep)
        return ProbeResult(admission=admission, eager=eager, fx=fx, export=export, dynamo=dynamo, compile=compile_rep, out_dir=out_dir)

    assert loaded is not None
    # Move model + inputs to GPU when available (TITAN RTX class works for
    # 7B-13B BF16, AWQ kernels actually run, FP8 dequantizes to BF16 cleanly).
    device = _resolve_device()
    moved_model, moved_args, moved_kwargs = _move_to_device(
        loaded.model, loaded.sample_inputs, loaded.sample_kwargs, device
    )
    loaded = LoadedModel(model=moved_model, sample_inputs=moved_args, sample_kwargs=moved_kwargs)
    report.write_json(
        input_summary_path,
        {
            "model_id": model_id,
            "slice_id": slice_id,
            "positional_inputs": [_summarise_value(v) for v in loaded.sample_inputs],
            "kwarg_inputs": {k: _summarise_value(v) for k, v in loaded.sample_kwargs.items()},
        },
    )

    # --- 2. eager.
    eager = _run_eager(loaded, model_id, slice_id)
    if eager.status != StageStatus.PASS.value:
        admission = AdmissionReport(
            model_id=model_id,
            slice_id=slice_id,
            status=AdmissionStatus.FAILED_EAGER.value,
            reason=eager.error or "eager forward pass failed",
            eager_report_path="eager_report.json",
            torch_compile_report_path="torch_compile_report.json",
            dynamo_report_path="dynamo_report.json",
            environment_path="environment.json",
            recommended_next_step="Investigate eager-mode failure before retrying admission.",
        )
        report.write_admission(out_dir, admission, eager, fx, export, dynamo, compile_rep)
        _free_gpu(loaded)
        return ProbeResult(admission=admission, eager=eager, fx=fx, export=export, dynamo=dynamo, compile=compile_rep, out_dir=out_dir)

    # --- 3. fx symbolic trace.
    fx = _run_fx(loaded, model_id, slice_id)

    # --- 4. torch.export.
    export = _run_export(loaded, model_id, slice_id)

    # --- 5. dynamo capture (graph breaks, op coverage).
    dynamo = _run_dynamo(loaded, model_id, slice_id)

    # --- 6. torch.compile.
    compile_rep = _run_torch_compile(loaded, model_cfg, model_id, slice_id)

    # --- 7. classify.
    if compile_rep.status == StageStatus.PASS.value:
        admission_status = AdmissionStatus.AVAILABLE
        reason = ""
        next_step = "Promote to admission baseline; record compile_time_s for future runs."
    elif compile_rep.status == StageStatus.FAIL.value:
        admission_status = AdmissionStatus.FAILED_TORCH_COMPILE
        reason = compile_rep.error or "torch.compile failed"
        next_step = "Inspect graph breaks and torch.compile error; consider fullgraph=False or dynamic=False."
    else:
        admission_status = AdmissionStatus.AVAILABLE_SLICE_ONLY if slice_cfg else AdmissionStatus.AVAILABLE
        reason = "torch.compile skipped"
        next_step = "torch.compile skipped; eager and dynamo passed."

    admission = AdmissionReport(
        model_id=model_id,
        slice_id=slice_id,
        status=admission_status.value,
        reason=reason,
        eager_report_path="eager_report.json",
        fx_report_path="fx_report.json",
        export_report_path="export_report.json",
        dynamo_report_path="dynamo_report.json",
        torch_compile_report_path="torch_compile_report.json",
        environment_path="environment.json",
        error_path=None,
        recommended_next_step=next_step,
    )
    report.write_admission(out_dir, admission, eager, fx, export, dynamo, compile_rep)
    _free_gpu(loaded)
    return ProbeResult(admission=admission, eager=eager, fx=fx, export=export, dynamo=dynamo, compile=compile_rep, out_dir=out_dir)


def _free_gpu(loaded: LoadedModel | None) -> None:
    """Release GPU memory between probes -- critical when multiple multi-GB
    models run back-to-back through the suite runner.
    """

    if loaded is None:
        return
    try:
        del loaded
    except Exception:
        pass
    import gc

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Stage runners.
# --------------------------------------------------------------------------- #


def _run_eager(loaded: LoadedModel, model_id: str, slice_id: str) -> EagerReport:
    t0 = time.perf_counter()
    args, kwargs = _align_input_dtypes(loaded.model, loaded.sample_inputs, loaded.sample_kwargs)
    try:
        with torch.no_grad():
            output = loaded.model(*args, **kwargs)
        wall = time.perf_counter() - t0
        return EagerReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.PASS.value,
            wall_time_s=wall,
            output_summary={"summary": _summarise_value(output)},
        )
    except Exception as exc:
        return EagerReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            wall_time_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _run_fx(loaded: LoadedModel, model_id: str, slice_id: str) -> FxReport:
    """Try ``torch.fx.symbolic_trace`` on the loaded model.

    FX is intentionally restrictive: it accepts only models with a static
    graph and no data-dependent control flow. A passing trace is a strong
    "graph is friendly to ahead-of-time tooling" signal.
    """

    try:
        import torch.fx as fx  # noqa: PLC0415
    except Exception as exc:
        return FxReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        # FX symbolic_trace doesn't accept kwargs; we have to pass concrete
        # kwarg defaults for everything we want to keep.
        concrete_args = {k: v for k, v in loaded.sample_kwargs.items() if not isinstance(v, torch.Tensor)}
        traced = fx.symbolic_trace(loaded.model, concrete_args=concrete_args)
    except Exception as exc:
        return FxReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    histogram: dict[str, int] = {}
    node_count = 0
    for node in traced.graph.nodes:
        node_count += 1
        op = node.op if not callable(node.target) else f"{node.op}:{getattr(node.target, '__name__', str(node.target))}"
        histogram[op] = histogram.get(op, 0) + 1
    return FxReport(
        model_id=model_id,
        slice_id=slice_id,
        status=StageStatus.PASS.value,
        node_count=node_count,
        op_histogram=histogram,
    )


def _run_export(loaded: LoadedModel, model_id: str, slice_id: str) -> ExportReport:
    """Try ``torch.export.export`` (the dynamo-export ATen graph path).

    ``torch.export`` is the modern AOT graph capture: stricter than
    ``torch.compile`` (no graph breaks allowed) but produces a stable
    serialisable ATen-IR graph. A passing export is "this model is
    deployable via ExportedProgram".
    """

    try:
        from torch.export import export as torch_export  # noqa: PLC0415
    except Exception as exc:
        return ExportReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            error=f"{type(exc).__name__}: {exc}",
        )
    args, kwargs = _align_input_dtypes(loaded.model, loaded.sample_inputs, loaded.sample_kwargs)
    try:
        with torch.no_grad():
            ep = torch_export(loaded.model, args=tuple(args), kwargs=dict(kwargs), strict=False)
    except Exception as exc:
        return ExportReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
    histogram: dict[str, int] = {}
    node_count = 0
    try:
        for node in ep.graph.nodes:
            node_count += 1
            tgt = getattr(node.target, "__name__", str(node.target))
            key = f"{node.op}:{tgt}"
            histogram[key] = histogram.get(key, 0) + 1
    except Exception:
        pass
    return ExportReport(
        model_id=model_id,
        slice_id=slice_id,
        status=StageStatus.PASS.value,
        graph_node_count=node_count,
        op_histogram=histogram,
        has_dynamic_shapes=False,
    )


def _run_dynamo(loaded: LoadedModel, model_id: str, slice_id: str) -> DynamoCaptureReport:
    try:
        args, kwargs = _align_input_dtypes(loaded.model, loaded.sample_inputs, loaded.sample_kwargs)
        positional = _flatten_for_dynamo(args, kwargs)
        diag = collect_diagnostics(loaded.model, positional)
    except Exception as exc:
        return DynamoCaptureReport(
            model_id=model_id,
            slice_id=slice_id,
            status=StageStatus.FAIL.value,
            error=f"{type(exc).__name__}: {exc}",
        )
    return DynamoCaptureReport(
        model_id=model_id,
        slice_id=slice_id,
        status=StageStatus.PASS.value,
        graph_count=int(diag.graph_count),
        op_count=int(diag.op_count),
        graph_break_count=len(diag.graph_breaks),
        graph_breaks=[{"location": loc, "reason": rsn} for (loc, rsn) in diag.graph_breaks],
        warnings=list(diag.warnings),
    )


def _run_torch_compile(
    loaded: LoadedModel,
    model_cfg: ModelConfig,
    model_id: str,
    slice_id: str,
) -> TorchCompileReport:
    backend = model_cfg.compile.backend
    fullgraph = model_cfg.compile.fullgraph
    dynamic = model_cfg.compile.dynamic
    args, kwargs = _align_input_dtypes(loaded.model, loaded.sample_inputs, loaded.sample_kwargs)
    positional = _flatten_for_dynamo(args, kwargs)
    try:
        baseline = compile_baseline(
            loaded.model,
            positional,
            backend=backend,
            num_warmup=1,
            num_runs=2,
        )
    except Exception as exc:
        return TorchCompileReport(
            model_id=model_id,
            slice_id=slice_id,
            attempted=True,
            status=StageStatus.FAIL.value,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Re-time a single warm run separately so first_run_time_s is real (not the
    # cold-compile wall-time, which is captured in compile_time_s).
    first_run = 0.0
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            torch.compile(loaded.model, backend=backend)(*positional)
            first_run = time.perf_counter() - t0
    except Exception:
        first_run = 0.0

    return TorchCompileReport(
        model_id=model_id,
        slice_id=slice_id,
        attempted=True,
        status=StageStatus.PASS.value,
        backend=baseline.backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
        compile_time_s=baseline.cold_compile_ms / 1000.0,
        first_run_time_s=first_run,
        second_run_time_s=baseline.warm_run_ms / 1000.0,
        graph_break_count=int(baseline.num_graph_breaks),
    )


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _resolve_device() -> torch.device:
    """Probe target device. CUDA when available -- VLM/AWQ inference is much
    more reliable on GPU than CPU. Falls back to CPU otherwise.
    """

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _move_to_device(model: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], device: torch.device) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any]]:
    """Move model + every tensor in (args, kwargs) to ``device``."""

    if device.type == "cpu":
        return model, args, kwargs

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        return value

    try:
        model = model.to(device)
    except Exception:
        return model, args, kwargs
    return model, tuple(move(a) for a in args), {k: move(v) for k, v in kwargs.items()}


def _model_compute_dtype(model: nn.Module) -> torch.dtype | None:
    """Best-effort: infer the floating-point dtype the model expects.

    Returns the dtype of the first floating-point parameter or buffer found.
    Quantized models often have mixed dtypes; in that case we still return the
    first float dtype so synthetic inputs can be cast to it instead of float32.
    """

    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    for b in model.buffers():
        if b.is_floating_point():
            return b.dtype
    return None


def _cast_floats(value: Any, target: torch.dtype) -> Any:
    """Recursively cast every floating-point tensor in a tree to ``target``.

    Integer / bool tensors and non-tensor values are passed through unchanged.
    """

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != target:
            return value.to(target)
        return value
    if isinstance(value, dict):
        return {k: _cast_floats(v, target) for k, v in value.items()}
    if isinstance(value, list):
        return [_cast_floats(v, target) for v in value]
    if isinstance(value, tuple):
        return tuple(_cast_floats(v, target) for v in value)
    return value


def _align_input_dtypes(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Cast float tensors in (args, kwargs) to match the model's compute dtype.

    Required for quantized / bf16 / fp8 models loaded via transformers, where
    the AutoProcessor returns float32 pixel_values but the model's matmul
    expects bf16 / fp16. Without this, Qwen3-VL-FP8 and Qwen2.5-VL-AWQ raise
    ``RuntimeError: expected m1 and m2 to have the same dtype``.
    """

    target = _model_compute_dtype(model)
    if target is None or target == torch.float32:
        return args, kwargs
    return tuple(_cast_floats(a, target) for a in args), {k: _cast_floats(v, target) for k, v in kwargs.items()}


def _flatten_for_dynamo(
    sample_inputs: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """``compile_baseline`` accepts only positional args. Append kwargs by value."""

    if not sample_kwargs:
        return tuple(sample_inputs)
    return tuple(sample_inputs) + tuple(sample_kwargs.values())


def _summarise_value(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "len": len(value), "head": [_summarise_value(v) for v in list(value)[:4]]}
    if isinstance(value, dict):
        return {"kind": "mapping", "keys": sorted(str(k) for k in value)}
    return {"kind": type(value).__name__}


def _next_step_for_unavailable(status: AdmissionStatus) -> str:
    if status == AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS:
        return "Provision the model weights into the local HF cache and rerun."
    if status == AdmissionStatus.UNAVAILABLE_GATED_ACCESS:
        return "Resolve gating (HF token / repo agreement) and rerun."
    if status == AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY:
        return "Install the missing python dependency named in error.txt and rerun."
    if status == AdmissionStatus.UNAVAILABLE_TOO_LARGE:
        return "Use a slice config (configs/slices/) to probe a representative substructure."
    if status == AdmissionStatus.UNAVAILABLE_HARDWARE_CONSTRAINT:
        return "Re-run admission on hardware that meets the model's hardware_requirements."
    return "Investigate failure and rerun."


def _platform_summary() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
    }


# Re-export for ``report.write_environment``.
def platform_summary() -> dict[str, str]:
    return _platform_summary()


__all__ = ["ProbeResult", "platform_summary", "run_admission"]
