"""MCP tools for the merlin compiler flow.

Seven typed handlers, all routing through ``MerlinBridge`` (Python-API
first, subprocess fallback). Mirrors the shape of ``qnn_flow.py`` and
slots into ``ALL_TOOLS`` exactly the same way: append
``MERLIN_FLOW_TOOLS`` in ``mcp/tools/__init__.py``.

Tools:

* ``xpu_rt_merlin_list_targets``       — enumerate ``target_specs/examples/``.
* ``xpu_rt_merlin_describe_target``    — typed projection of one capability.yaml.
* ``xpu_rt_merlin_onnx_to_mlir``       — ONNX → payload MLIR (delegates to
                                         backends/merlin/onnx_bridge.py).
* ``xpu_rt_merlin_compile``            — one (target, source) → vmfb.
* ``xpu_rt_merlin_compile_dispatch_matrix`` — multi-target sweep.
* ``xpu_rt_merlin_chipyard_build``     — chipyard image build.
* ``xpu_rt_merlin_profile``            — profile a matrix with a runner.

All handlers return ``{"ok": bool, ...}`` and never raise — bridge
failures land as ``{"ok": False, "error": ..., "via": ...}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xpu_rt.mcp.session import SessionManager
from xpu_rt.targets.backends.merlin import (
    MerlinBridge,
    MerlinUnavailableError,
    build_chipyard_image,
    compile_dispatch_matrix,
    compile_program,
    list_targets,
    load_target_spec,
    onnx_to_payload_mlir,
    profile_dispatch_matrix,
)
from xpu_rt.targets.backends.merlin.runners import (
    FiresimRunner,
    HostRunner,
    NoopRunner,
    Runner,
    SpikeRunner,
)


def _bridge(merlin_root: str | None) -> MerlinBridge:
    return MerlinBridge(
        merlin_root=Path(merlin_root) if merlin_root
        else MerlinBridge().merlin_root,
    )


def _err(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc), "exception_type": type(exc).__name__}


# --------------------------------------------------------------------------- #
# Tool: list targets
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_list_targets(
    sm: SessionManager,  # noqa: ARG001
    *,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """List every target under ``$MERLIN_ROOT/target_specs/examples/``."""
    bridge = _bridge(merlin_root)
    targets = list_targets(bridge.merlin_root)
    return {
        "ok": True,
        "merlin_root": str(bridge.merlin_root),
        "merlin_root_available": bridge.available(),
        "targets": targets,
        "count": len(targets),
    }


# --------------------------------------------------------------------------- #
# Tool: describe target
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_describe_target(
    sm: SessionManager,  # noqa: ARG001
    *,
    name: str,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """Return a typed projection of one ``capability.yaml``."""
    bridge = _bridge(merlin_root)
    try:
        spec = load_target_spec(name, bridge.merlin_root)
    except FileNotFoundError as exc:
        return _err(exc)
    return {
        "ok": True,
        "name": spec.name,
        "display_name": spec.display_name,
        "vendor": spec.vendor,
        "maturity": spec.maturity,
        "host_isa": spec.host_isa,
        "environments": list(spec.environments),
        "execution_kind": spec.execution_kind,
        "isa_features": list(spec.isa_features),
        "runtime_executable_format": spec.runtime_executable_format,
        "has_simulator": spec.has_simulator,
        "simulator_kind": spec.simulator_kind,
        "spec_dir": str(spec.spec_dir),
    }


# --------------------------------------------------------------------------- #
# Tool: onnx → mlir
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_onnx_to_mlir(
    sm: SessionManager,  # noqa: ARG001
    *,
    onnx_path: str,
    out_mlir: str,
    opset_check: int = 18,
    workload_id: str | None = None,
    allow_stub: bool = True,
) -> dict[str, Any]:
    """Convert ONNX → payload MLIR via merlin's importer (or fall back)."""
    try:
        result = onnx_to_payload_mlir(
            Path(onnx_path),
            Path(out_mlir),
            opset_check=opset_check,
            workload_id=workload_id,
            allow_stub=allow_stub,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        return _err(exc)
    return {
        "ok": True,
        "mlir_path": str(result.mlir_path),
        "importer": result.importer,
        "cache_hit": result.cache_hit,
        "sha256": result.sha256,
    }


# --------------------------------------------------------------------------- #
# Tool: compile one (target, source)
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_compile(
    sm: SessionManager,  # noqa: ARG001
    *,
    target: str,
    source: str,
    out_dir: str,
    hw: str | None = None,
    quantized: bool = False,
    extra_args: list[str] | None = None,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """Compile ``source`` for one ``target``. Wraps merlin's ``compile.py``."""
    bridge = _bridge(merlin_root)
    try:
        result = compile_program(
            bridge,
            target=target,
            source=Path(source),
            out_dir=Path(out_dir),
            hw=hw,
            quantized=quantized,
            extra_args=extra_args,
        )
    except MerlinUnavailableError as exc:
        return _err(exc)
    return {
        "ok": result.returncode == 0,
        "target": result.target,
        "source": str(result.source),
        "out_dir": str(result.out_dir),
        "returncode": result.returncode,
        "vmfb_path": str(result.vmfb_path) if result.vmfb_path else None,
        "via": result.call_result.via,
        "stderr_tail": result.call_result.stderr[-1000:],
    }


# --------------------------------------------------------------------------- #
# Tool: compile multi-target dispatch matrix
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_compile_dispatch_matrix(
    sm: SessionManager,  # noqa: ARG001
    *,
    source: str,
    targets: list[str],
    out_dir: str,
    extra_args: list[str] | None = None,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """Sweep ``targets`` for one ``source``; return ``matrix.json`` shape."""
    bridge = _bridge(merlin_root)
    try:
        result = compile_dispatch_matrix(
            bridge,
            source=Path(source),
            targets=targets,
            out_dir=Path(out_dir),
            extra_args=extra_args,
        )
    except MerlinUnavailableError as exc:
        return _err(exc)
    return {
        "ok": result.returncode == 0,
        "matrix_path": str(result.matrix_path),
        "targets": list(result.targets),
        "out_dir": str(result.out_dir),
        "returncode": result.returncode,
        "per_target_dispatches": result.per_target_dispatches,
        "via": result.call_result.via,
        "stderr_tail": result.call_result.stderr[-1000:],
    }


# --------------------------------------------------------------------------- #
# Tool: chipyard build
# --------------------------------------------------------------------------- #


def xpu_rt_merlin_chipyard_build(
    sm: SessionManager,  # noqa: ARG001
    *,
    hardware: str,
    out_dir: str,
    subcommand: str = "build",
    extra_args: list[str] | None = None,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """Run merlin's ``tools/chipyard.py <subcommand> --hardware <hardware>``."""
    bridge = _bridge(merlin_root)
    try:
        result = build_chipyard_image(
            bridge,
            hardware=hardware,
            out_dir=Path(out_dir),
            subcommand=subcommand,
            extra_args=extra_args,
        )
    except MerlinUnavailableError as exc:
        return _err(exc)
    return {
        "ok": result.returncode == 0,
        "hardware": result.hardware,
        "out_dir": str(result.out_dir),
        "returncode": result.returncode,
        "image_path": str(result.image_path) if result.image_path else None,
        "via": result.call_result.via,
        "stderr_tail": result.call_result.stderr[-1000:],
    }


# --------------------------------------------------------------------------- #
# Tool: profile dispatch matrix
# --------------------------------------------------------------------------- #


_RUNNERS: dict[str, type[Runner]] = {
    "host": HostRunner,
    "spike": SpikeRunner,
    "firesim": FiresimRunner,
    "noop": NoopRunner,
}


def _build_runner(kind: str | None, table: dict[str, float] | None) -> Runner | None:
    if kind is None:
        return None
    if kind == "noop":
        return NoopRunner(table=dict(table or {}))
    cls = _RUNNERS.get(kind)
    if cls is None:
        return None
    return cls()


def xpu_rt_merlin_profile(
    sm: SessionManager,  # noqa: ARG001
    *,
    matrix_path: str,
    out_path: str,
    runner: str | None = None,
    noop_table: dict[str, float] | None = None,
    ssh_host: str | None = None,
    ssh_identity: str | None = None,
    board_bench: str | None = None,
    iterations: int = 10,
    extra_args: list[str] | None = None,
    merlin_root: str | None = None,
) -> dict[str, Any]:
    """Profile a previously-compiled ``matrix.json``.

    Pass ``runner="host"|"spike"|"firesim"|"noop"`` for the local
    runner path; pass ``ssh_host`` + ``board_bench`` (and leave
    ``runner=None``) for merlin's on-board profiler.
    """
    bridge = _bridge(merlin_root)
    runner_obj = _build_runner(runner, noop_table)
    try:
        result = profile_dispatch_matrix(
            bridge,
            matrix_path=Path(matrix_path),
            out_path=Path(out_path),
            runner=runner_obj,
            ssh_host=ssh_host,
            ssh_identity=Path(ssh_identity) if ssh_identity else None,
            board_bench=board_bench,
            iterations=iterations,
            extra_args=extra_args,
        )
    except (MerlinUnavailableError, ValueError) as exc:
        return _err(exc)
    return {
        "ok": result.returncode == 0,
        "manifest_path": str(result.manifest_path),
        "via": result.via,
        "returncode": result.returncode,
        "mean_us_by_dispatch": result.mean_us_by_dispatch,
    }


# --------------------------------------------------------------------------- #
# Tool list (consumed by mcp/tools/__init__.py)
# --------------------------------------------------------------------------- #


MERLIN_FLOW_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_merlin_list_targets",
        "description": (
            "Enumerate every merlin target spec under "
            "$MERLIN_ROOT/target_specs/examples/. Returns sorted target "
            "names and a flag for whether merlin_root is reachable on disk."
        ),
        "phase": "inspect",
        "handler": xpu_rt_merlin_list_targets,
        "input_schema": {
            "type": "object",
            "properties": {"merlin_root": {"type": "string"}},
        },
    },
    {
        "name": "xpu_rt_merlin_describe_target",
        "description": (
            "Read one capability.yaml under "
            "$MERLIN_ROOT/target_specs/examples/<name>/ and return a "
            "typed projection (vendor, host ISA, ISA features, "
            "execution kind, runtime format, simulator info)."
        ),
        "phase": "inspect",
        "handler": xpu_rt_merlin_describe_target,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "merlin_root": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "xpu_rt_merlin_onnx_to_mlir",
        "description": (
            "Convert ONNX → payload MLIR via merlin's onnx_to_mlir.py, "
            "with torch-mlir-import-onnx and a stub-MLIR fallback. "
            "Result is cached on (sha256(onnx), opset, importer)."
        ),
        "phase": "transform",
        "handler": xpu_rt_merlin_onnx_to_mlir,
        "input_schema": {
            "type": "object",
            "properties": {
                "onnx_path": {"type": "string"},
                "out_mlir": {"type": "string"},
                "opset_check": {"type": "integer", "default": 18},
                "workload_id": {"type": "string"},
                "allow_stub": {"type": "boolean", "default": True},
            },
            "required": ["onnx_path", "out_mlir"],
        },
    },
    {
        "name": "xpu_rt_merlin_compile",
        "description": (
            "Compile a single (target, source) via merlin's compile.py. "
            "Honours --hw and --quantized; routes through MerlinBridge "
            "(Python-API → subprocess fallback). Returns the discovered "
            ".vmfb path (best-effort) plus the stderr tail on failure."
        ),
        "phase": "transform",
        "handler": xpu_rt_merlin_compile,
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "source": {"type": "string"},
                "out_dir": {"type": "string"},
                "hw": {"type": "string"},
                "quantized": {"type": "boolean", "default": False},
                "extra_args": {"type": "array", "items": {"type": "string"}},
                "merlin_root": {"type": "string"},
            },
            "required": ["target", "source", "out_dir"],
        },
    },
    {
        "name": "xpu_rt_merlin_compile_dispatch_matrix",
        "description": (
            "Run merlin's compile_dispatch_matrix.py over N targets for "
            "one source MLIR. Returns the matrix.json contents plus a "
            "{target: [dispatch_id, ...]} projection."
        ),
        "phase": "transform",
        "handler": xpu_rt_merlin_compile_dispatch_matrix,
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "out_dir": {"type": "string"},
                "extra_args": {"type": "array", "items": {"type": "string"}},
                "merlin_root": {"type": "string"},
            },
            "required": ["source", "targets", "out_dir"],
        },
    },
    {
        "name": "xpu_rt_merlin_chipyard_build",
        "description": (
            "Drive merlin's tools/chipyard.py — build a simulator image, "
            "configure firesim, or validate a hardware recipe. "
            "Subcommand defaults to 'build'."
        ),
        "phase": "job",
        "handler": xpu_rt_merlin_chipyard_build,
        "input_schema": {
            "type": "object",
            "properties": {
                "hardware": {"type": "string"},
                "out_dir": {"type": "string"},
                "subcommand": {"type": "string", "default": "build"},
                "extra_args": {"type": "array", "items": {"type": "string"}},
                "merlin_root": {"type": "string"},
            },
            "required": ["hardware", "out_dir"],
        },
    },
    {
        "name": "xpu_rt_merlin_profile",
        "description": (
            "Profile a compiled dispatch matrix. Pass runner=host/spike/"
            "firesim/noop for the local path, or ssh_host+board_bench "
            "(leaving runner empty) for merlin's on-board profiler. "
            "Writes profiled_manifest.json at out_path."
        ),
        "phase": "job",
        "handler": xpu_rt_merlin_profile,
        "input_schema": {
            "type": "object",
            "properties": {
                "matrix_path": {"type": "string"},
                "out_path": {"type": "string"},
                "runner": {
                    "type": "string",
                    "enum": ["host", "spike", "firesim", "noop"],
                },
                "noop_table": {"type": "object"},
                "ssh_host": {"type": "string"},
                "ssh_identity": {"type": "string"},
                "board_bench": {"type": "string"},
                "iterations": {"type": "integer", "default": 10},
                "extra_args": {"type": "array", "items": {"type": "string"}},
                "merlin_root": {"type": "string"},
            },
            "required": ["matrix_path", "out_path"],
        },
    },
]
