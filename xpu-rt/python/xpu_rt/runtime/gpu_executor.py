"""GPU runtime hookup: compile Triton sources + launch on CUDA.

Takes a compiled CompGen module + the Triton emitter's artifact
directory (`kernels/*.py` + `emission_manifest.json`) and:

1. Imports each kernel's module via ``importlib``.
2. Extracts the ``@triton.jit`` function by name.
3. Launches the kernel on CUDA with a caller-supplied argument set.

Graceful fallback:

- If ``triton`` is not importable **or** ``torch.cuda.is_available()``
  is ``False``, :func:`launch_triton_kernel` raises
  :class:`GPUNotAvailable` with a clear message.
- :func:`gpu_available` is the single source of truth for
  "should we try the GPU path".

The executor is deliberately **one-kernel-at-a-time**. Full
multi-kernel orchestration lands when the pipeline driver's Phase
5 ``ExecutionPlan`` queue is wired through the GPU launcher;
today the contract is: emit kernel → import → launch → return
tensor.

Usage (on a GPU host):

    from pathlib import Path
    from compgen.runtime.gpu_executor import (
        gpu_available, launch_triton_kernel,
    )
    assert gpu_available()
    out = launch_triton_kernel(
        artifact_dir=Path("/tmp/compgen_triton"),
        kernel_name="compgen_matmul_0",
        args=[a_cuda, b_cuda, c_cuda, M, N, K, ...],
        grid=(grid_m, grid_n),
    )
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


class GPUNotAvailable(RuntimeError):
    """Raised when the GPU path is unavailable in the current env."""


def gpu_available() -> bool:
    """Whether we can run Triton kernels on CUDA right now."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _require_gpu() -> None:
    if not gpu_available():
        try:
            import torch

            have_cuda = torch.cuda.is_available()
        except ImportError:
            have_cuda = False
        try:
            import triton  # noqa: F401

            have_triton = True
        except ImportError:
            have_triton = False
        raise GPUNotAvailable(f"GPU path unavailable: cuda={have_cuda}, triton={have_triton}")


@dataclass
class LaunchResult:
    kernel_name: str
    grid: tuple[int, ...]
    launch_ms: float = 0.0


def _load_kernel_module(path: Path, kernel_name: str) -> Any:
    """Import a single Triton kernel file and return the ``@triton.jit`` fn."""
    spec = importlib.util.spec_from_file_location(f"compgen_kernel_{kernel_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, kernel_name, None)
    if fn is None:
        raise AttributeError(f"kernel file {path} does not export {kernel_name!r}")
    return fn


def load_emission_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Read ``emission_manifest.json`` written by the Triton emitter."""
    path = Path(artifact_dir) / "emission_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"emission_manifest.json missing from {artifact_dir}; run ``emit_triton_kernels`` first"
        )
    return json.loads(path.read_text())


def launch_triton_kernel(
    artifact_dir: str | Path,
    kernel_name: str,
    *,
    args: list[Any] | tuple[Any, ...],
    grid: tuple[int, ...],
    require_gpu: bool = True,
) -> LaunchResult:
    """Compile + launch one Triton kernel.

    ``args`` is the positional arg list passed to the ``@triton.jit``
    function (usually tensor pointers + shapes + strides + constexpr
    block sizes).

    ``grid`` is the launch grid tuple passed via ``kernel[grid](*args)``.

    Raises :class:`GPUNotAvailable` when ``require_gpu=True`` and
    CUDA / Triton are missing.
    """
    if require_gpu:
        _require_gpu()

    artifact = Path(artifact_dir)
    manifest = load_emission_manifest(artifact)
    entry = manifest.get(kernel_name)
    if entry is None:
        raise KeyError(f"{kernel_name!r} not in emission manifest; known kernels: {sorted(manifest)}")
    kernel_path = Path(entry["source_path"])
    if not kernel_path.exists():
        raise FileNotFoundError(f"kernel source missing: {kernel_path}")

    fn = _load_kernel_module(kernel_path, kernel_name)

    # Launch.
    import time

    t0 = time.perf_counter()
    fn[grid](*args)
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    launch_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "gpu_executor.launched",
        kernel=kernel_name,
        grid=grid,
        launch_ms=launch_ms,
    )
    return LaunchResult(kernel_name=kernel_name, grid=grid, launch_ms=launch_ms)


def launch_all_from_manifest(
    artifact_dir: str | Path,
    args_by_kernel: dict[str, list[Any]],
    grids_by_kernel: dict[str, tuple[int, ...]],
    *,
    require_gpu: bool = True,
) -> list[LaunchResult]:
    """Convenience: iterate through every kernel in the manifest and launch it."""
    manifest = load_emission_manifest(artifact_dir)
    out: list[LaunchResult] = []
    for name in manifest:
        if name not in args_by_kernel or name not in grids_by_kernel:
            log.warning(
                "gpu_executor.skip_missing_args",
                kernel=name,
            )
            continue
        out.append(
            launch_triton_kernel(
                artifact_dir,
                name,
                args=args_by_kernel[name],
                grid=grids_by_kernel[name],
                require_gpu=require_gpu,
            )
        )
    return out


__all__ = [
    "GPUNotAvailable",
    "LaunchResult",
    "gpu_available",
    "launch_all_from_manifest",
    "launch_triton_kernel",
    "load_emission_manifest",
]
