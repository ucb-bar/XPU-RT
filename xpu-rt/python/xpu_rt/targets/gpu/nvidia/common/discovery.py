"""Discover NVIDIA header-only library include paths.

cuBLASDx, libcudacxx, and CUTLASS are all header-only libraries
shipped under PyPI packages (``nvidia-mathdx``,
``nvidia-cuda-cccl``, etc.). NVRTC needs ``-I`` paths pointing at
the headers; the agentic-compilation flow picks them up via
:func:`compgen.runtime.autotune.probe_device`.

Wave 1.14 moves these helpers here from ``runtime/native/cuda.py``.
The original file re-exports for backward compatibility — existing
imports keep working.

Owned by ``common/`` because they apply to every NVIDIA arch —
the cu13 NVRTC discovery (which IS Blackwell-specific because it
gates ``__CUDA_ARCH__ == 1000``) lives separately under
``blackwell/cu13_nvrtc.py``.

The search order for each helper is the result of bridge rounds
#070-#091 — see the per-function docstrings for the rationale.
"""

from __future__ import annotations

import os
from pathlib import Path


def discover_cublasdx_include() -> str | None:
    """Locate cuBLASDx headers via the ``nvidia-mathdx`` pip wheel.

    Returns the absolute path to the include directory containing
    ``cublasdx.hpp`` (typically
    ``<site>/nvidia/mathdx/<ver>/include``), or ``None`` if mathdx
    isn't importable / installed.

    Search order:

    1. ``$CUBLASDX_INCLUDE_PATH`` env var (lets users point at a
       system-wide install or vendored copy).
    2. The ``nvidia.mathdx`` Python package's ``include/`` directory.
    3. ``None`` — caller decides whether to fall back to a
       hand-rolled body.

    PEP 420 namespace package handling (per bridge #070):
    ``nvidia.mathdx.__file__`` is None; use ``__path__``.
    """
    env = os.environ.get("CUBLASDX_INCLUDE_PATH")
    if env and Path(env).is_dir():
        if (Path(env) / "cublasdx.hpp").is_file():
            return env

    try:
        import nvidia.mathdx as _mathdx  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    paths = list(getattr(_mathdx, "__path__", []) or [])
    if not paths:
        return None
    pkg_root = Path(paths[0]).resolve()
    for include_dir in pkg_root.rglob("include"):
        if (include_dir / "cublasdx.hpp").is_file():
            return str(include_dir)
    return None


def cublasdx_available() -> bool:
    """Cheap probe — is cuBLASDx reachable for an NVRTC compile?"""
    return discover_cublasdx_include() is not None


def discover_libcudacxx_include() -> str | None:
    """Locate libcudacxx (``cuda/std/*``) headers.

    cuBLASDx's commondx layer pulls ``<cuda/std/type_traits>`` and
    friends; NVRTC's built-in header set on the cu12 toolkit doesn't
    ship libcudacxx (per bridge #072).

    Search order:

    1. ``$LIBCUDACXX_INCLUDE_PATH`` env override.
    2. The ``nvidia.cuda_cccl`` Python package's include dir.
    3. System CUDA toolkit include dirs (``$CUDA_HOME``,
       ``/usr/local/cuda*/include``).
    4. ``None``.

    Sentinel: ``cuda/std/type_traits`` must exist under the chosen
    dir.
    """
    sentinel = "cuda/std/type_traits"

    env = os.environ.get("LIBCUDACXX_INCLUDE_PATH")
    if env and (Path(env) / sentinel).exists():
        return env

    try:
        import nvidia.cuda_cccl as _cccl  # type: ignore

        for p in list(getattr(_cccl, "__path__", []) or []):
            cand = Path(p) / "include"
            if (cand / sentinel).exists():
                return str(cand)
    except Exception:  # noqa: BLE001
        pass

    candidates: list[str] = []
    if cuda_home := os.environ.get("CUDA_HOME"):
        candidates.append(f"{cuda_home}/include")
    candidates.extend(
        [
            "/usr/local/cuda/include",
            "/usr/local/cuda-12.6/include",
            "/usr/local/cuda-12.4/include",
            "/usr/local/cuda-12.0/include",
            "/usr/local/cuda-13.0/include",
        ]
    )
    try:
        for entry in Path("/usr/local").glob("cuda-*"):
            inc = entry / "include"
            if inc.is_dir():
                candidates.append(str(inc))
    except OSError:
        pass

    for cand in candidates:
        if cand and (Path(cand) / sentinel).is_file():
            return cand
    return None


def libcudacxx_available() -> bool:
    return discover_libcudacxx_include() is not None


def discover_cutlass_include() -> str | None:
    """Locate CUTLASS headers (``cutlass/numeric_types.h``).

    cuBLASDx pulls in CUTLASS sub-headers; the ``nvidia-mathdx``
    wheel vendors CUTLASS under
    ``nvidia/mathdx/external/cutlass/include`` (per bridge #074).

    Search order:

    1. ``$CUTLASS_INCLUDE_PATH`` env var.
    2. ``nvidia.mathdx`` wheel's vendored CUTLASS.
    3. System CUDA toolkit (some distributions ship CUTLASS).
    4. ``None``.

    Sentinel: ``cutlass/numeric_types.h``.
    """
    sentinel = "cutlass/numeric_types.h"

    env = os.environ.get("CUTLASS_INCLUDE_PATH")
    if env and (Path(env) / sentinel).is_file():
        return env

    try:
        import nvidia.mathdx as _mathdx  # type: ignore

        for p in list(getattr(_mathdx, "__path__", []) or []):
            cand = Path(p) / "external" / "cutlass" / "include"
            if (cand / sentinel).is_file():
                return str(cand)
    except Exception:  # noqa: BLE001
        pass

    candidates: list[str] = []
    if cuda_home := os.environ.get("CUDA_HOME"):
        candidates.append(f"{cuda_home}/include")
    candidates.extend(
        [
            "/usr/local/cuda/include",
            "/usr/local/cuda-12.6/include",
        ]
    )
    try:
        for entry in Path("/usr/local").glob("cuda-*"):
            inc = entry / "include"
            if inc.is_dir():
                candidates.append(str(inc))
    except OSError:
        pass

    for cand in candidates:
        if cand and (Path(cand) / sentinel).is_file():
            return cand
    return None


def cutlass_available() -> bool:
    return discover_cutlass_include() is not None
