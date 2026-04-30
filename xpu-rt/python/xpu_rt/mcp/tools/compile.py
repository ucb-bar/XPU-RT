"""MCP tools for the agent-driven compile loop — Phase 10a.

Exposes two tools:

- ``compgen_compile_torch_model``: takes a base64-pickled torch
  ``nn.Module`` + sample inputs, runs the FX→MegakernelGraph
  lowering, emits the megakernel + manifest into the bundle dir,
  and returns a bundle handle + the lowering decision log.
- ``compgen_run_compiled_bundle``: takes a bundle handle + a
  base64-pickled input tensor, runs the bundle on the GPU, and
  returns the output (or just timing stats for benchmarking).

The remote agent on a Blackwell box uses these instead of
``compgen-run-conformance``: pip install + MCP register → ask
Claude "compile this and run it" → Claude calls these tools →
the architecture takes over without the agent ever seeing the
source.

Decision log: every compile records its ``LoweringDecision``
(pattern matched, backend per body, tile shape, rationale). The
log is part of the response so the agent can audit "why was this
op routed to backend X?" without source-reading.
"""

from __future__ import annotations

import base64
import ctypes
import io
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def _override_backend_choice(
    probed: Any,
    *,
    prefer_cublasdx_for_linears: bool | None = None,
    cublasdx_precision: str | None = None,
    use_cu13_nvrtc: bool | None = None,
) -> Any:
    """Apply per-flag overrides on top of a probe-derived
    :class:`BackendChoice`. Each ``None`` keeps the probe's value;
    each non-None replaces it. The agent typically passes nothing
    (full auto), but a sophisticated caller can pin one knob (e.g.
    force fp32 to match a baseline) without giving up the rest of
    the auto-detection."""
    import dataclasses

    overrides: dict[str, Any] = {}
    if prefer_cublasdx_for_linears is not None:
        overrides["use_cublasdx_for_linears"] = bool(prefer_cublasdx_for_linears)
    if cublasdx_precision is not None:
        overrides["cublasdx_precision"] = cublasdx_precision
    if use_cu13_nvrtc is not None:
        overrides["use_cu13_nvrtc"] = bool(use_cu13_nvrtc)
    if not overrides:
        return probed
    return dataclasses.replace(probed, **overrides)


# ---------------------------------------------------------------------------
# compgen_compile_torch_model
# ---------------------------------------------------------------------------


def compgen_compile_torch_model(
    *,
    model_pickle_b64: str,
    sample_input_pickle_b64: str,
    output_dir: str,
    backend: str = "auto",
    target_arch: str | None = None,
    prefer_cublasdx_for_linears: bool | None = None,
    cublasdx_precision: str | None = None,
    use_cu13_nvrtc: bool | None = None,
    fuse_epilogue: bool = False,
) -> dict[str, Any]:
    """Compile a torch nn.Module via the ETC dispatch path.

    The agentic-compilation entry point. With default arguments —
    ``compgen_compile_torch_model(model_pickle_b64=..., sample_input_pickle_b64=..., output_dir=...)``
    — the matcher probes the local device + reachable libraries via
    :func:`compgen.runtime.autotune.probe_device` and picks every
    backend knob (NVRTC version, cuBLASDx precision, SM tag, tile
    shape) automatically. The agent doesn't need to know about
    cuBLASDx / cu13 / sm_100 / bf16+fp32-acc — those are
    implementation details the probe handles.

    Args:
        model_pickle_b64: base64-encoded ``pickle.dumps(model)``.
        sample_input_pickle_b64: base64-encoded ``pickle.dumps(x,)``
            — the sample-inputs tuple.
        output_dir: filesystem path; the bundle lands at
            ``<output_dir>/bundle/megakernel/{source.cu,manifest.yaml}``.
        backend: ``"auto"`` (default) → probe-and-pick everything.
            Any other value disables the probe and falls through to
            the explicit flags below.
        target_arch: explicit override for the NVRTC ``--gpu-architecture``
            flag. ``None`` (default) → probe picks ("sm_100" on
            Blackwell, "sm_90" on Hopper, fallback "sm_100" on CPU
            host). Set explicitly for cross-compilation.
        prefer_cublasdx_for_linears: explicit override. ``None``
            (default) → probe picks based on library availability.
        cublasdx_precision: explicit override. ``None`` (default) →
            probe picks ``"bf16_fp32"`` on Blackwell, ``"fp32"``
            elsewhere.
        use_cu13_nvrtc: explicit override. ``None`` (default) →
            probe picks True on Blackwell when cu13 NVRTC is
            reachable.

    Returns:
        ``{
            "status": "ok" | "unsupported_shape" | "compile_failed",
            "bundle_dir": <path or None>,
            "kernel_name": <str>,
            "decision": LoweringDecision.to_dict(),
            "manifest": <dict from emit>,
            "elapsed_ms": <float>,
        }``

    Raises:
        Never. Failures land in the ``status`` field with structured
        error context.
    """
    import pickle

    from compgen.runtime.lowering import (
        UnsupportedShape,
        lower_torch_to_megakernel,
    )
    from compgen.transforms.emit_cuda_megakernel import emit_cuda_megakernel
    from compgen.transforms.event_static_schedule import compute_static_schedule

    t0 = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        model = pickle.loads(base64.b64decode(model_pickle_b64))
        sample_inputs = pickle.loads(base64.b64decode(sample_input_pickle_b64))
    except Exception as exc:
        return {
            "status": "compile_failed",
            "stage": "deserialize",
            "error": repr(exc),
            "bundle_dir": None,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    # Resolve backend choice. Three modes:
    # 1. backend="auto" + no explicit flags → probe_device picks all.
    # 2. backend="auto" + some explicit flags → probe + per-flag override.
    # 3. backend!="auto" → bypass probe, use explicit flags (legacy).
    backend_choice = None
    if backend == "auto":
        from compgen.runtime.autotune import probe_device

        probed = probe_device(target=target_arch or "auto")
        # Per-flag override: explicit user values win over the probe.
        backend_choice = _override_backend_choice(
            probed,
            prefer_cublasdx_for_linears=prefer_cublasdx_for_linears,
            cublasdx_precision=cublasdx_precision,
            use_cu13_nvrtc=use_cu13_nvrtc,
        )

    try:
        if backend_choice is not None:
            result = lower_torch_to_megakernel(
                model,
                sample_inputs,
                backend_choice=backend_choice,
                fuse_epilogue=fuse_epilogue,
            )
        else:
            # Legacy path — explicit flags only.
            result = lower_torch_to_megakernel(
                model,
                sample_inputs,
                prefer_cublasdx_for_linears=prefer_cublasdx_for_linears or False,
                cublasdx_precision=cublasdx_precision or "fp32",
                target_arch=target_arch or "sm_100",
                fuse_epilogue=fuse_epilogue,
            )
    except UnsupportedShape as exc:
        return {
            "status": "unsupported_shape",
            "error": str(exc),
            "bundle_dir": None,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
    except Exception as exc:
        return {
            "status": "compile_failed",
            "stage": "fx_to_megakernel",
            "error": repr(exc),
            "bundle_dir": None,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    # Schedule + emit. The remote agent doesn't see the source —
    # the megakernel CUDA source lands on disk for inspection via
    # the ``etc_megakernel_inspect`` tool.
    try:
        # Wave 1.6 — cluster-launch wiring. Pull the chosen cluster
        # dim from BackendChoice (set by the probe for Blackwell)
        # and pass to the schedule. Non-Blackwell paths keep
        # cluster_dim=None and stay on single-block tasks.
        if (
            backend_choice is not None
            and getattr(backend_choice, "supports_clusters", False)
            and getattr(backend_choice, "cluster_dim_x", None) is not None
        ):
            cluster_dim_tuple: tuple[int, int, int] | None = (
                backend_choice.cluster_dim_x,
                backend_choice.cluster_dim_y or 1,
                backend_choice.cluster_dim_z or 1,
            )
            supports_clusters_flag = True
        else:
            cluster_dim_tuple = None
            supports_clusters_flag = False

        schedule = compute_static_schedule(
            result.megakernel_graph,
            sm_count=_resolve_sm_count(),
            block_dim=(32, 32, 1),
            supports_clusters=supports_clusters_flag,
            cluster_dim=cluster_dim_tuple,
        )
        emit = emit_cuda_megakernel(
            schedule,
            device_function_sources=result.device_function_sources,
            user_buffer_count=len(result.user_buffer_layout),
        )
        bundle_dir = output_path / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        emit.write_to_bundle(bundle_dir / "megakernel")
        # Stash the layout + arch + sample-input shape so run_bundle
        # has enough context without re-pickling.
        import json

        # Resolve the effective values that ended up in the bodies.
        # When backend="auto" was used, these come from the probe;
        # otherwise from the user's explicit flags. Either way the
        # bundle is self-contained — run_compiled_bundle uses these
        # to re-compile against identical settings.
        if backend_choice is not None:
            eff_target_arch = backend_choice.target_arch
            eff_prefer_cublasdx = backend_choice.use_cublasdx_for_linears
            eff_precision = backend_choice.cublasdx_precision
            eff_use_cu13 = backend_choice.use_cu13_nvrtc
            backend_choice_dict = backend_choice.to_dict()
        else:
            eff_target_arch = target_arch or "sm_100"
            eff_prefer_cublasdx = prefer_cublasdx_for_linears or False
            eff_precision = cublasdx_precision or "fp32"
            eff_use_cu13 = use_cu13_nvrtc or False
            backend_choice_dict = None

        # Wave 1.8 — when the matcher accepted via the submodule
        # fallback, the bundle must store the SUBMODULE (not the
        # wrapper) so dispatch's weight-extraction works. Otherwise
        # ``model.up.weight`` blows up — the wrapper doesn't have
        # ``up`` directly. Per bridge #108: the AttributeError fix.
        submodule_path = getattr(result.decision, "submodule_path", "") or ""
        if submodule_path:
            effective_model = model.get_submodule(submodule_path)
            effective_model_pickle_b64 = base64.b64encode(pickle.dumps(effective_model)).decode()
        else:
            effective_model_pickle_b64 = model_pickle_b64

        (bundle_dir / "compile_context.json").write_text(
            json.dumps(
                {
                    "user_buffer_layout": list(result.user_buffer_layout),
                    "target_arch": eff_target_arch,
                    "sample_input_shape": list(sample_inputs[0].shape),
                    "decision": result.decision.to_dict(),
                    "kernel_name": emit.kernel_name,
                    "model_pickle_b64": effective_model_pickle_b64,  # for run-time weight reload
                    "submodule_path": submodule_path,
                    "wrapper_class": type(model).__name__,
                    # NVRTC -I paths the matcher said are needed for the
                    # bodies' #includes to resolve (cuBLASDx headers, etc).
                    # ``run_compiled_bundle`` re-passes these to CudaModule
                    # so the recompile path stays self-contained.
                    "nvrtc_include_paths": list(result.decision.nvrtc_include_paths),
                    # NVRTC compiler options the matcher said the bodies
                    # need (e.g. ``-default-device`` when cuBLASDx is in the
                    # mix). Same self-contained re-pass story as the include
                    # paths.
                    "nvrtc_extra_options": list(result.decision.nvrtc_extra_options),
                    "prefer_cublasdx_for_linears": eff_prefer_cublasdx,
                    "cublasdx_precision": eff_precision,
                    "use_cu13_nvrtc": eff_use_cu13,
                    "fuse_epilogue": fuse_epilogue,
                    # Full BackendChoice snapshot when backend="auto" was
                    # used — surfaces the probe's rationale for agent audit.
                    "backend_choice": backend_choice_dict,
                    "backend_mode": backend,
                },
                indent=2,
            )
        )
    except Exception as exc:
        return {
            "status": "compile_failed",
            "stage": "schedule_or_emit",
            "error": repr(exc),
            "bundle_dir": None,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "compgen_compile_torch_model.ok",
        pattern=result.decision.pattern_name,
        kernel_name=emit.kernel_name,
        bundle_dir=str(bundle_dir),
        elapsed_ms=elapsed_ms,
    )
    return {
        "status": "ok",
        "bundle_dir": str(bundle_dir),
        "kernel_name": emit.kernel_name,
        "decision": result.decision.to_dict(),
        # Surface the auto-resolved BackendChoice in the response so
        # the agent's audit query ("why was X picked?") gets the
        # answer without needing to read compile_context.json off
        # disk. None when backend!="auto".
        "backend_choice": backend_choice_dict,
        "backend_mode": backend,
        "manifest": emit.manifest,
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# compgen_run_compiled_bundle
# ---------------------------------------------------------------------------


def compgen_run_compiled_bundle(
    *,
    bundle_dir: str,
    input_pickle_b64: str,
    num_iters: int = 10,
    return_output: bool = False,
) -> dict[str, Any]:
    """Run a previously-compiled bundle on a real input.

    Args:
        bundle_dir: path produced by ``compgen_compile_torch_model``.
        input_pickle_b64: base64-encoded ``pickle.dumps((x,))`` — a
            tuple matching the original sample_inputs shape.
        num_iters: timed iterations after a 3-iteration warmup.
        return_output: when True, the response includes the
            ``output_pickle_b64`` field (base64-encoded
            ``pickle.dumps(y)``). Default False — the agent typically
            wants timing + correctness summary, not the raw tensor.

    Returns:
        ``{
            "status": "ok" | "load_failed" | "launch_failed",
            "etc_us": <float>,
            "eager_us": <float>,
            "speedup_vs_eager": <float>,
            "max_abs_err": <float>,
            "max_rel_err": <float>,
            "output_pickle_b64": <str | absent>,
        }``

    Failures (no GPU, missing .so, NVRTC errors) populate ``status`` +
    ``error`` rather than raising.
    """
    import json
    import pickle

    from compgen.runtime.native.cuda import (
        CudaMegakernelLauncher,
        CudaModule,
        CudaUnavailableError,
    )
    from compgen.runtime.native.device import Device
    from compgen.testing.etc_dispatch import (
        _allocate_etc_state,
        _launch_and_readback,
    )

    t0 = time.perf_counter()
    bundle_path = Path(bundle_dir)
    ctx_path = bundle_path / "compile_context.json"
    if not ctx_path.is_file():
        return {
            "status": "load_failed",
            "error": f"compile_context.json missing under {bundle_path}",
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    try:
        ctx = json.loads(ctx_path.read_text())
        model = pickle.loads(base64.b64decode(ctx["model_pickle_b64"]))
        x = pickle.loads(base64.b64decode(input_pickle_b64))[0]
        source = (bundle_path / "megakernel" / "source.cu").read_text()
    except Exception as exc:
        return {
            "status": "load_failed",
            "error": repr(exc),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    # Recompile from cached source (fast — same NVRTC inputs).
    try:
        cumod = CudaModule(
            cuda_source=source,
            kernel_name=ctx["kernel_name"],
            arch=ctx.get("target_arch", "sm_100"),
            extra_include_paths=tuple(ctx.get("nvrtc_include_paths", [])),
            extra_options=tuple(ctx.get("nvrtc_extra_options", [])),
            use_cu13_nvrtc=ctx.get("use_cu13_nvrtc", False),
        )
    except (CudaUnavailableError, RuntimeError) as exc:
        return {
            "status": "load_failed",
            "stage": "nvrtc",
            "error": repr(exc),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    # Reconstruct the schedule for state alloc — same model + inputs
    # → same graph + schedule.
    from compgen.runtime.lowering import lower_torch_to_megakernel
    from compgen.transforms.event_static_schedule import compute_static_schedule

    sample_inputs = (x,)
    result = lower_torch_to_megakernel(
        model,
        sample_inputs,
        prefer_cublasdx_for_linears=ctx.get("prefer_cublasdx_for_linears", False),
        cublasdx_precision=ctx.get("cublasdx_precision", "fp32"),
        target_arch=ctx.get("target_arch", "sm_100"),
        fuse_epilogue=ctx.get("fuse_epilogue", False),
    )
    schedule = compute_static_schedule(
        result.megakernel_graph,
        sm_count=_resolve_sm_count(),
        block_dim=(32, 32, 1),
    )

    # When the matcher landed on a submodule (composition-aware path
    # — pattern_name ``"ffn@ffn"`` etc.), the dispatch helpers
    # (``_workload_buffers``) expect the matched module's attributes
    # (e.g. ``model.up.weight``) at the top level. Resolve the
    # submodule by walking the @<path> suffix once here, so the rest
    # of the dispatch path is composition-agnostic. Per bridge #108.
    dispatch_model = model
    pattern_name = result.decision.pattern_name
    if "@" in pattern_name:
        sub_path = pattern_name.split("@", 1)[1]
        for attr in sub_path.split("."):
            dispatch_model = getattr(dispatch_model, attr)

    # Build a "Workload-like" object the existing dispatch helpers expect.
    class _Workload:
        pass

    workload = _Workload()
    workload.model = dispatch_model
    workload.sample_inputs = sample_inputs
    workload.user_buffer_layout = tuple(ctx["user_buffer_layout"])

    try:
        device = Device.create("cuda:0")
        launcher = CudaMegakernelLauncher(device.handle.value or 0, device_index=0)
        state = _allocate_etc_state(
            workload=workload,
            schedule=schedule,
            sample_x=x,
            device_index=0,
        )
    except (CudaUnavailableError, RuntimeError) as exc:
        return {
            "status": "load_failed",
            "stage": "device_state",
            "error": repr(exc),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    import torch

    eager_device = torch.device("cuda:0")
    eager_model = model.to(eager_device).eval()
    x_dev = x.to(eager_device, dtype=torch.float32)

    try:
        # warmup
        for _ in range(3):
            with torch.no_grad():
                eager_model(x_dev)
            _launch_and_readback(
                workload=workload,
                schedule=schedule,
                launcher=launcher,
                cumod=cumod,
                emit=None,
                x=x,
                state=state,
            )
        torch.cuda.synchronize(eager_device)

        torch.cuda.synchronize(eager_device)
        e0 = time.perf_counter()
        for _ in range(num_iters):
            with torch.no_grad():
                eager_model(x_dev)
        torch.cuda.synchronize(eager_device)
        eager_us = (time.perf_counter() - e0) * 1e6 / num_iters

        c0 = time.perf_counter()
        for _ in range(num_iters):
            y_etc = _launch_and_readback(
                workload=workload,
                schedule=schedule,
                launcher=launcher,
                cumod=cumod,
                emit=None,
                x=x,
                state=state,
            )
        etc_us = (time.perf_counter() - c0) * 1e6 / num_iters

        with torch.no_grad():
            y_eager = eager_model(x_dev).detach().to(torch.float32).cpu()
        # Eager preserves ND shape (torch.nn.Linear's leading-dim
        # broadcast); the matcher flattened to (batch_flat, in) for
        # the tile graph so y_etc is 2D. Reshape y_etc back to the
        # eager output's ND shape before the abs-diff so the agent's
        # broadcast comparison succeeds. Per bridge #118.
        if y_etc.shape != y_eager.shape:
            try:
                y_etc = y_etc.reshape(y_eager.shape)
            except RuntimeError:
                pass  # let the diff surface the real shape mismatch
        diff = (y_etc - y_eager).abs()
        max_abs = float(diff.max().item())
        denom = y_eager.abs().clamp_min(1e-6)
        max_rel = float((diff / denom).max().item())
    except Exception as exc:
        return {
            "status": "launch_failed",
            "error": repr(exc),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
    finally:
        try:
            cumod.close()
            device.close()
        except Exception:
            pass

    # Dump the actual cuLaunchKernelEx params per bridge #102 Option C.
    # Surfaces gridDim/blockDim/cluster/shared/cooperative directly so
    # the agent can audit "is parallelism actually engaging across SMs,
    # or are we serializing to gridDim=(1,1,1)?". Counts visible
    # tasks too — sm_count vs scheduled-task ratio.
    lc = schedule.launch_config
    sm_queues = schedule.sm_queues
    sms_used = sum(1 for q in sm_queues if q.tasks)
    total_tasks = sum(len(q.tasks) for q in sm_queues)
    launch_params = {
        "grid_dim": list(lc.grid_dim),
        "block_dim": list(lc.block_dim),
        "cluster_dim": list(lc.cluster_dim) if lc.cluster_dim is not None else None,
        "shared_mem_bytes": int(lc.shared_mem_bytes),
        "cooperative": bool(lc.cooperative),
        "sm_count_probed": _resolve_sm_count(),
        "sms_with_tasks": sms_used,
        "total_tasks": total_tasks,
        "tasks_per_sm_max": max((len(q.tasks) for q in sm_queues), default=0),
        "tasks_per_sm_mean": (total_tasks / max(len(sm_queues), 1)),
    }

    out = {
        "status": "ok",
        "etc_us": etc_us,
        "eager_us": eager_us,
        "speedup_vs_eager": eager_us / max(etc_us, 1e-6),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
        "launch_params": launch_params,
    }
    if return_output:
        buf = io.BytesIO()
        pickle.dump(y_etc, buf)
        out["output_pickle_b64"] = base64.b64encode(buf.getvalue()).decode()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sm_count() -> int:
    """Probe SM count; fall back to 132 (B200) if probe fails."""
    try:
        from compgen.runtime.probe import probe_cuda_device

        probe = probe_cuda_device(0)
        sm = probe.get("sm_count") or probe.get("multi_processor_count") or 132
        return int(sm)
    except Exception:
        return 132


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


COMPILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "compgen_compile_torch_model",
        "description": (
            "Compile a torch nn.Module via CompGen's Event Tensor Compiler "
            "dispatch path. Lowers the FX-graph shape to a MegakernelGraph, "
            "schedules across SMs, emits the persistent megakernel CUDA "
            "source, and writes the bundle to output_dir. Returns a bundle "
            "handle + decision log (pattern matched, backend per body, "
            "tile shape, rationale). Round 1 supports the diamond-DAG shape "
            "(linear_a(x) + linear_b(x)).relu(); other shapes return "
            "status='unsupported_shape' with a typed reason."
        ),
        "phase": "compile",
        "handler": compgen_compile_torch_model,
        "input_schema": {
            "type": "object",
            "properties": {
                "model_pickle_b64": {
                    "type": "string",
                    "description": "base64-encoded pickle.dumps(nn.Module).",
                },
                "sample_input_pickle_b64": {
                    "type": "string",
                    "description": (
                        "base64-encoded pickle.dumps((x,)) — a single-element tuple of the sample input tensor."
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": "Where to write the bundle.",
                },
                "backend": {
                    "type": "string",
                    "default": "auto",
                    "description": (
                        "Backend selection mode. 'auto' (default) → "
                        "compgen.runtime.autotune.probe_device picks "
                        "every backend knob (NVRTC version, cuBLASDx "
                        "precision, SM tag, tile shape) automatically. "
                        "Any other value disables the probe and uses "
                        "the explicit flags below. Pass 'auto' for "
                        "the agentic-compilation flow — the agent "
                        "doesn't need to know any of the lower-level "
                        "backend details."
                    ),
                },
                "target_arch": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": (
                        "Optional override for NVRTC --gpu-architecture. "
                        "When backend='auto', None means probe-and-pick "
                        "(sm_100 on Blackwell, sm_90 on Hopper, fallback "
                        "sm_100 on CPU). Set explicitly for cross-"
                        "compilation."
                    ),
                },
                "prefer_cublasdx_for_linears": {
                    "type": ["boolean", "null"],
                    "default": None,
                    "description": (
                        "Optional override. None (default) → probe "
                        "picks based on library availability. True → "
                        "force cuBLASDx bodies for linear ops. False → "
                        "force hand_rolled_fmaf."
                    ),
                },
                "cublasdx_precision": {
                    "type": ["string", "null"],
                    "default": None,
                    "enum": ["fp32", "bf16_fp32", None],
                    "description": (
                        "Optional override. None (default) → probe "
                        "picks 'bf16_fp32' on Blackwell, 'fp32' "
                        "elsewhere. 'fp32' is fp32 SIMT (no tensor "
                        "cores), 'bf16_fp32' is bf16 inputs + fp32 "
                        "accumulator (engages Blackwell tensor cores)."
                    ),
                },
                "use_cu13_nvrtc": {
                    "type": ["boolean", "null"],
                    "default": None,
                    "description": (
                        "Optional override. None (default) → probe "
                        "picks True on Blackwell when cu13 NVRTC is "
                        "reachable. Required for SM<1000> tcgen05.mma "
                        "(cu12 NVRTC max sm_90 silently SIMTs cuBLASDx)."
                    ),
                },
                "fuse_epilogue": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Wave 2.5 — when True, FFN matcher folds "
                        "relu_up into linear_up's MMA epilogue (post-"
                        "MMA cast + threshold + write to next stage's "
                        "input buffer). Eliminates the bipartite "
                        "linear→relu→linear edge wave at MLP-1 "
                        "(predicted to lift cluster-locality from "
                        "5% intra-cluster → ≥30%). Default False to "
                        "preserve the unfused tile-graph structure."
                    ),
                },
            },
            "required": ["model_pickle_b64", "sample_input_pickle_b64", "output_dir"],
        },
    },
    {
        "name": "compgen_run_compiled_bundle",
        "description": (
            "Run a previously-compiled bundle on real input. Returns timing "
            "(etc_us, eager_us, speedup_vs_eager) + correctness "
            "(max_abs_err, max_rel_err) vs the original eager torch model. "
            "Designed for the agent-driven loop: compile_torch_model → "
            "run_compiled_bundle → ask 'did we beat eager? if not, why?' "
            "and inspect the decision log."
        ),
        "phase": "verify",
        "handler": compgen_run_compiled_bundle,
        "input_schema": {
            "type": "object",
            "properties": {
                "bundle_dir": {
                    "type": "string",
                    "description": "Path produced by compgen_compile_torch_model.",
                },
                "input_pickle_b64": {
                    "type": "string",
                    "description": "base64-encoded pickle.dumps((x,)).",
                },
                "num_iters": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                },
                "return_output": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, response includes output_pickle_b64 "
                        "(the etc-path output tensor). Default false — "
                        "agents typically want timing + correctness only."
                    ),
                },
            },
            "required": ["bundle_dir", "input_pickle_b64"],
        },
    },
]


# ---------------------------------------------------------------------------
# compgen_cublasdx_header_smoke — round 2b
# ---------------------------------------------------------------------------


def compgen_cublasdx_header_smoke(*, target_arch: str = "sm_90") -> dict[str, Any]:
    """NVRTC-compile a minimal kernel that ``#include <cublasdx.hpp>``.

    Round-2b risk-de-confliction tool. Before writing a cuBLASDx-
    templated GEMM body in the matcher, we want to know whether
    NVRTC can even parse cuBLASDx's header + transitive dependencies
    (CUTLASS, CuTe, CUDA toolkit pieces) with the discovered include
    path. This tool answers that without committing to body-emission
    code that depends on the API specifics.

    Args:
        target_arch: NVRTC ``--gpu-architecture`` flag.

    Returns:
        ``{
            "status": "ok" | "missing" | "compile_failed",
            "include_path": <str or None>,
            "ptx_size": <int>,           # only on ok
            "compile_ms": <float>,
            "log": <str>,                # NVRTC's log on failure
        }``

    The remote agent uses this to decide whether the round-2c
    body emission is unblocked. ``status == "ok"`` means cuBLASDx
    is reachable from NVRTC — proceed. Anything else surfaces the
    diagnostic so we can fix the include graph or pin a header
    version.
    """
    from compgen.runtime.native.cuda import (
        CudaUnavailableError,
        discover_cublasdx_include,
        discover_cutlass_include,
        discover_libcudacxx_include,
    )

    t0 = time.perf_counter()
    include_path = discover_cublasdx_include()
    if include_path is None:
        return {
            "status": "missing",
            "include_path": None,
            "include_paths": [],
            "log": (
                "cuBLASDx not discoverable. Install with "
                "`pip install nvidia-mathdx`, or set "
                "$CUBLASDX_INCLUDE_PATH to the directory containing "
                "cublasdx.hpp."
            ),
            "compile_ms": (time.perf_counter() - t0) * 1000,
        }

    # cuBLASDx's commondx layer pulls in <cuda/std/*> from
    # libcudacxx, which NVRTC's built-in header set doesn't ship on
    # the cu12 toolkit. cuBLASDx itself pulls in <cutlass/*> from
    # the CUTLASS sublibrary that nvidia-mathdx vendors under
    # ``external/cutlass/include``. Discover all three and pass
    # them as a -I list to NVRTC.
    libcudacxx_path = discover_libcudacxx_include()
    cutlass_path = discover_cutlass_include()
    nvrtc_include_paths: list[str] = [include_path]
    if libcudacxx_path is not None:
        nvrtc_include_paths.append(libcudacxx_path)
    if cutlass_path is not None:
        nvrtc_include_paths.append(cutlass_path)

    # Minimal header-only smoke. The empty kernel body ensures NVRTC
    # parses cuBLASDx's headers + transitive deps without exercising
    # the GEMM template — a separate round 2c step. If THIS fails,
    # the body smoke would also fail; isolating the diagnostic.
    smoke_source = """
#include <cublasdx.hpp>

extern "C" __global__ void compgen_cublasdx_header_smoke_kernel() {
    // Empty — exists so NVRTC links a translation unit. The header
    // include is what we want to compile-test.
}
"""
    try:
        from compgen.runtime.native.cuda import CudaModule

        # cuBLASDx headers contain ``static constexpr`` member
        # functions without explicit ``__host__``/``__device__``
        # annotations; NVRTC's default JIT mode treats unannotated
        # functions as host-only and rejects them. ``-default-device``
        # makes NVRTC treat unannotated functions as ``__device__``,
        # which matches cuBLASDx's header-only design. REMOTE bridge
        # probe #074 surfaced this as the second blocker.
        module = CudaModule(
            cuda_source=smoke_source,
            kernel_name="compgen_cublasdx_header_smoke_kernel",
            arch=target_arch,
            extra_include_paths=tuple(nvrtc_include_paths),
            extra_options=("-default-device",),
        )
        ptx_size = len(module.ptx)
        module.close()
        return {
            "status": "ok",
            "include_path": include_path,
            "include_paths": nvrtc_include_paths,
            "libcudacxx_path": libcudacxx_path,
            "cutlass_path": cutlass_path,
            "ptx_size": ptx_size,
            "compile_ms": (time.perf_counter() - t0) * 1000,
            "log": (
                f"cuBLASDx header reachable; NVRTC compiled "
                f"a kernel using `#include <cublasdx.hpp>` to {ptx_size}-byte "
                f"PTX in {(time.perf_counter() - t0) * 1000:.0f}ms."
            ),
        }
    except CudaUnavailableError as exc:
        return {
            "status": "compile_failed",
            "stage": "cuda_unavailable",
            "include_path": include_path,
            "include_paths": nvrtc_include_paths,
            "libcudacxx_path": libcudacxx_path,
            "cutlass_path": cutlass_path,
            "log": repr(exc),
            "compile_ms": (time.perf_counter() - t0) * 1000,
        }
    except RuntimeError as exc:
        # NVRTC compile failure surfaces here — its log is in the
        # exception message. The ``include_paths`` + ``*_path`` fields
        # tell the agent which paths the compile saw, so a missing
        # header is easy to attribute.
        return {
            "status": "compile_failed",
            "stage": "nvrtc",
            "include_path": include_path,
            "include_paths": nvrtc_include_paths,
            "libcudacxx_path": libcudacxx_path,
            "cutlass_path": cutlass_path,
            "log": str(exc),
            "compile_ms": (time.perf_counter() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# compgen_run_cuda_source — round 2c agent-driven kernel iteration
# ---------------------------------------------------------------------------


def compgen_run_cuda_source(
    *,
    cuda_source: str,
    kernel_name: str,
    grid_dim: tuple[int, int, int] | list[int] = (1, 1, 1),
    block_dim: tuple[int, int, int] | list[int] = (32, 32, 1),
    shared_mem_bytes: int = 0,
    inputs_pickle_b64: str = "",
    output_shapes_b64: str = "",
    target_arch: str = "sm_90",
    extra_options: tuple[str, ...] | list[str] = (),
    use_cublasdx_includes: bool = True,
    num_warmup: int = 3,
    num_timed: int = 10,
) -> dict[str, Any]:
    """Generic NVRTC compile + cooperative launch of an arbitrary
    CUDA source string. Round-2c agent-driven harness for iterating
    on cuBLASDx-templated kernels against the live header.

    The agent on the GPU host has read access to the installed
    nvidia-mathdx wheel's cuBLASDx examples + headers, so they can
    construct a kernel source string from the canonical patterns,
    pass it through this tool, and see the NVRTC log + run output
    for each iteration. This shifts cuBLASDx authorship from
    Garden-side guesswork (without the live header) to the
    actually-running GPU host.

    Args:
        cuda_source: full ``__global__ void <name>(...)`` source.
        kernel_name: matches the ``__global__`` symbol.
        grid_dim, block_dim: launch dims.
        shared_mem_bytes: dynamic smem the kernel needs (cuBLASDx
            requires a real value here for its smem tiles).
        inputs_pickle_b64: base64 ``pickle.dumps(tuple_of_torch_tensors)``.
            Each tensor is uploaded to GPU; the kernel's signature
            is ``void(T1*, T2*, ..., Tk*, U1*, U2*, ..., Um*)`` —
            inputs first, then outputs.
        output_shapes_b64: base64 ``pickle.dumps([(shape, dtype), ...])``.
            One entry per output buffer; tool allocates accordingly.
        target_arch: NVRTC --gpu-architecture flag.
        extra_options: forwarded to NVRTC. ``-default-device`` is
            already added when ``use_cublasdx_includes=True``.
        use_cublasdx_includes: when True (default), threads cuBLASDx
            + libcudacxx + CUTLASS include paths into the NVRTC
            compile + adds ``-default-device``. Set False to skip if
            you're testing a vanilla CUDA kernel.
        num_warmup, num_timed: bench iterations.

    Returns:
        ``{
            "status": "ok" | "compile_failed" | "launch_failed" | "missing_includes",
            "log": "<NVRTC log on failure / success summary>",
            "ptx_size": <int>,
            "compile_ms": <float>,
            "etc_us": <float>,
            "outputs_pickle_b64": "<str>",   # only on ok
        }``

    Failures stay in ``status``/``log``. Use the same iteration
    pattern as ``compgen_cublasdx_header_smoke``: each failure log
    points at the next thing to fix in the kernel source.
    """
    import pickle

    import torch

    from compgen.runtime.native.cuda import (
        CudaUnavailableError,
        _ensure_cuda_driver_context,
        discover_cublasdx_include,
        discover_cutlass_include,
        discover_libcudacxx_include,
    )

    t0 = time.perf_counter()

    # Resolve includes when requested.
    nvrtc_includes: list[str] = []
    options = list(extra_options)
    if use_cublasdx_includes:
        cublasdx_path = discover_cublasdx_include()
        libcudacxx_path = discover_libcudacxx_include()
        cutlass_path = discover_cutlass_include()
        if cublasdx_path is None:
            return {
                "status": "missing_includes",
                "log": (
                    "cuBLASDx not discoverable. Install with "
                    "`pip install nvidia-mathdx`, or pass "
                    "use_cublasdx_includes=False if you're "
                    "compiling a vanilla CUDA kernel."
                ),
                "compile_ms": (time.perf_counter() - t0) * 1000,
            }
        nvrtc_includes.append(cublasdx_path)
        if libcudacxx_path:
            nvrtc_includes.append(libcudacxx_path)
        if cutlass_path:
            nvrtc_includes.append(cutlass_path)
        if "-default-device" not in options:
            options.append("-default-device")

    # NVRTC compile.
    try:
        from compgen.runtime.native.cuda import CudaModule

        compile_t0 = time.perf_counter()
        module = CudaModule(
            cuda_source=cuda_source,
            kernel_name=kernel_name,
            arch=target_arch,
            extra_include_paths=tuple(nvrtc_includes),
            extra_options=tuple(options),
        )
        compile_ms = (time.perf_counter() - compile_t0) * 1000
    except CudaUnavailableError as exc:
        return {
            "status": "missing_includes",
            "log": repr(exc),
            "include_paths": nvrtc_includes,
            "compile_ms": (time.perf_counter() - t0) * 1000,
        }
    except RuntimeError as exc:
        # NVRTC compile error; the message contains the full log.
        return {
            "status": "compile_failed",
            "log": str(exc),
            "include_paths": nvrtc_includes,
            "options": options,
            "compile_ms": (time.perf_counter() - t0) * 1000,
        }

    # Decode inputs + output shapes. Errors here surface as
    # ``bad_input`` so the agent can fix the encoding without
    # confusing it for an NVRTC / launch failure.
    try:
        inputs: tuple = pickle.loads(base64.b64decode(inputs_pickle_b64)) if inputs_pickle_b64 else ()
        output_specs: list = pickle.loads(base64.b64decode(output_shapes_b64)) if output_shapes_b64 else []
        # Canonical encoding: list of ``(shape_tuple, dtype_str)``
        # pairs, where dtype_str is one of "float32" / "float16" /
        # "bfloat16" / "int32" / "int64". The shape entries must be
        # plain Python ints (not torch.Size / numpy types).
        normalised_specs: list[tuple[tuple[int, ...], str]] = []
        for i, spec in enumerate(output_specs):
            try:
                shape_raw, dtype_raw = spec
            except (TypeError, ValueError) as exc:
                module.close()
                return {
                    "status": "bad_input",
                    "log": (
                        f"output_shapes_b64[{i}]={spec!r} doesn't unpack as "
                        "(shape, dtype). Canonical encoding: "
                        "[((dim0, dim1, ...), 'float32'), ...]. "
                        f"Underlying error: {exc!r}."
                    ),
                    "compile_ms": compile_ms,
                }
            shape_tuple = tuple(int(d) for d in shape_raw)
            dtype_str = str(dtype_raw)
            if dtype_str not in {"float32", "float16", "bfloat16", "int32", "int64"}:
                module.close()
                return {
                    "status": "bad_input",
                    "log": (
                        f"output_shapes_b64[{i}] dtype={dtype_raw!r} not in "
                        "supported set: float32 / float16 / bfloat16 / "
                        "int32 / int64. Pass as a string."
                    ),
                    "compile_ms": compile_ms,
                }
            normalised_specs.append((shape_tuple, dtype_str))
    except Exception as exc:  # noqa: BLE001
        module.close()
        return {
            "status": "bad_input",
            "log": (
                f"Failed to decode inputs/output_shapes: {exc!r}. "
                "Canonical encoding: "
                "inputs_pickle_b64=base64(pickle.dumps((tensor1, tensor2, ...))); "
                "output_shapes_b64=base64(pickle.dumps([((d0, d1), 'float32'), ...]))."
            ),
            "compile_ms": compile_ms,
        }

    # Allocate device buffers + upload inputs + run. cuda-bindings
    # 13.x is strict about argument types — pass plain Python ints,
    # not ctypes.c_void_p, for host/device pointers.
    try:
        from cuda.bindings import driver as cu_driver  # type: ignore

        from compgen.runtime.native.cuda import _cu_check

        _dtype_bytes = {
            "float32": 4,
            "int32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int64": 8,
        }

        _ensure_cuda_driver_context(0)
        # Inputs: contiguous-cast each torch tensor + cuMemAlloc + HtoD.
        input_dev_ptrs: list[int] = []
        input_buffers = []
        for t in inputs:
            t_c = t.contiguous().to(torch.float32)
            nbytes = int(t_c.numel() * 4)
            ptr = _cu_check(cu_driver.cuMemAlloc(nbytes))
            # ``data_ptr()`` returns int; cuMemcpyHtoD expects int
            # for srcHost. Pass without the c_void_p wrapper —
            # cuda-bindings 13.x rejects that.
            _cu_check(cu_driver.cuMemcpyHtoD(ptr, int(t_c.data_ptr()), nbytes))
            input_dev_ptrs.append(int(ptr))
            input_buffers.append(t_c)  # keep host tensor alive

        # Outputs: alloc empty.
        output_dev_ptrs: list[int] = []
        output_metas: list[tuple[tuple[int, ...], str, int]] = []
        for shape_tuple, dtype_str in normalised_specs:
            n = 1
            for d in shape_tuple:
                n *= d
            elem_bytes = _dtype_bytes[dtype_str]
            nbytes = int(n * elem_bytes)
            ptr = _cu_check(cu_driver.cuMemAlloc(nbytes))
            output_dev_ptrs.append(int(ptr))
            output_metas.append((shape_tuple, dtype_str, nbytes))

        # Build the void** kernel-args (input ptrs followed by output
        # ptrs). cuda-bindings' ``cuLaunchKernel`` accepts a
        # ``(int *)`` of kernel-argument addresses. Cleanest pattern
        # is a ctypes-allocated array of c_void_p whose entries point
        # at c_void_p containers holding the actual pointer values.
        all_arg_ptrs = input_dev_ptrs + output_dev_ptrs
        n_args = len(all_arg_ptrs)
        arg_values = (ctypes.c_void_p * n_args)(*[ctypes.c_void_p(p) for p in all_arg_ptrs])
        arg_ptrs = (ctypes.c_void_p * n_args)(
            *[ctypes.addressof(arg_values) + i * ctypes.sizeof(ctypes.c_void_p) for i in range(n_args)]
        )

        # Launch via cuda-bindings. cuLaunchKernel signature:
        # cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, stream,
        #                kernelParams, extra)
        # ``stream=0`` selects the default stream; ``extra=0``
        # (NULL) means no extra config.
        gd = tuple(int(x) for x in grid_dim)
        bd = tuple(int(x) for x in block_dim)
        smem = int(shared_mem_bytes)
        kernel_params_addr = ctypes.addressof(arg_ptrs)
        # Warmup.
        for _ in range(num_warmup):
            _cu_check(
                cu_driver.cuLaunchKernel(
                    module._kernel,
                    gd[0],
                    gd[1],
                    gd[2],
                    bd[0],
                    bd[1],
                    bd[2],
                    smem,
                    0,  # default stream
                    kernel_params_addr,
                    0,  # extras NULL
                )
            )
        _cu_check(cu_driver.cuCtxSynchronize())

        # Timed.
        run_t0 = time.perf_counter()
        for _ in range(num_timed):
            _cu_check(
                cu_driver.cuLaunchKernel(
                    module._kernel,
                    gd[0],
                    gd[1],
                    gd[2],
                    bd[0],
                    bd[1],
                    bd[2],
                    smem,
                    0,
                    kernel_params_addr,
                    0,
                )
            )
        _cu_check(cu_driver.cuCtxSynchronize())
        etc_us = (time.perf_counter() - run_t0) * 1e6 / num_timed

        # Read outputs back.
        outputs: list[torch.Tensor] = []
        _torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        for ptr, (shape, dtype_str, nbytes) in zip(output_dev_ptrs, output_metas, strict=True):
            host_t = torch.empty(shape, dtype=_torch_dtype[dtype_str])
            _cu_check(cu_driver.cuMemcpyDtoH(int(host_t.data_ptr()), ptr, nbytes))
            outputs.append(host_t)

        # Free device buffers.
        for ptr in input_dev_ptrs + output_dev_ptrs:
            _cu_check(cu_driver.cuMemFree(ptr))

        del arg_values, arg_ptrs  # keep alive across launch
        module.close()

        out_buf = io.BytesIO()
        pickle.dump(tuple(outputs), out_buf)
        return {
            "status": "ok",
            "log": (
                f"NVRTC compiled to {len(module.ptx)}-byte PTX in "
                f"{compile_ms:.0f}ms; ran {num_timed} iterations of "
                f"<<{gd}, {bd}>> with {shared_mem_bytes} smem in "
                f"{etc_us:.1f}us mean."
            ),
            "include_paths": nvrtc_includes,
            "options": options,
            "ptx_size": len(module.ptx),
            "compile_ms": compile_ms,
            "etc_us": etc_us,
            "outputs_pickle_b64": base64.b64encode(out_buf.getvalue()).decode(),
        }
    except Exception as exc:  # noqa: BLE001
        try:
            module.close()
        except Exception:
            pass
        return {
            "status": "launch_failed",
            "log": repr(exc),
            "compile_ms": compile_ms,
        }


# Append the cuBLASDx smoke tool descriptor after the function is
# defined. (The other two are referenced in the literal above
# because they're defined earlier in the file; keeping the order
# in the list stable is purely cosmetic.)
COMPILE_TOOLS.append(
    {
        "name": "compgen_run_cuda_source",
        "description": (
            "NVRTC-compile + run an arbitrary CUDA source string. "
            "Round-2c agent-driven harness for iterating on cuBLASDx-"
            "templated kernels against the live header. Threads "
            "cuBLASDx + libcudacxx + CUTLASS include paths through "
            "NVRTC by default + adds -default-device. Inputs/outputs "
            "are passed as base64-pickled torch tensors; failures land "
            "in status/log without raising."
        ),
        "phase": "compile",
        "handler": compgen_run_cuda_source,
        "input_schema": {
            "type": "object",
            "properties": {
                "cuda_source": {"type": "string"},
                "kernel_name": {"type": "string"},
                "grid_dim": {"type": "array", "items": {"type": "integer"}},
                "block_dim": {"type": "array", "items": {"type": "integer"}},
                "shared_mem_bytes": {"type": "integer", "default": 0},
                "inputs_pickle_b64": {"type": "string", "default": ""},
                "output_shapes_b64": {"type": "string", "default": ""},
                "target_arch": {"type": "string", "default": "sm_90"},
                "extra_options": {"type": "array", "items": {"type": "string"}},
                "use_cublasdx_includes": {"type": "boolean", "default": True},
                "num_warmup": {"type": "integer", "default": 3},
                "num_timed": {"type": "integer", "default": 10},
            },
            "required": ["cuda_source", "kernel_name"],
        },
    }
)


COMPILE_TOOLS.append(
    {
        "name": "compgen_cublasdx_header_smoke",
        "description": (
            "NVRTC-compile a minimal kernel that #includes <cublasdx.hpp> "
            "to verify the cuBLASDx header + its transitive deps "
            "(CUTLASS, CuTe, CUDA toolkit pieces) are reachable through "
            "the discovered include path. Returns status='ok' if NVRTC "
            "produces PTX, 'missing' if cuBLASDx isn't installed, or "
            "'compile_failed' with the NVRTC log on header-include errors. "
            "Round-2b risk-de-confliction tool: tells the agent whether "
            "to proceed with cuBLASDx body emission (round 2c) or to fix "
            "the include graph first."
        ),
        "phase": "diagnose",
        "handler": compgen_cublasdx_header_smoke,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_arch": {
                    "type": "string",
                    "default": "sm_90",
                    "description": (
                        "NVRTC --gpu-architecture flag. sm_90 covers "
                        "Blackwell workstation via JIT; sm_100 / sm_120 "
                        "for native datacenter / workstation Blackwell."
                    ),
                },
            },
            "required": [],
        },
    }
)


__all__ = [
    "COMPILE_TOOLS",
    "compgen_compile_torch_model",
    "compgen_cublasdx_header_smoke",
    "compgen_run_compiled_bundle",
    "compgen_run_cuda_source",
]
