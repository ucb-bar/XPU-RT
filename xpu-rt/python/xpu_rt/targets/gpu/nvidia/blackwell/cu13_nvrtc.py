"""cu13 NVRTC ctypes wrapper — Blackwell-specific JIT path.

Per bridge #089: cuBLASDx's SM<1000> dispatcher needs
``CUTE_ARCH_FFMA2_SM100_ENABLED`` which CUTLASS gates on
``__CUDA_ARCH__ == 1000``. cu12 NVRTC (what cuda-python's bundled
wrapper uses) maxes at sm_90 → ``__CUDA_ARCH__ == 900`` → tcgen05.mma
silently falls back to SIMT.

The cu13 NVRTC shipped under ``nvidia.cu13`` (a torch≥2.6 dep on
every Blackwell venv) accepts ``sm_100`` / ``sm_120`` and emits the
right ``__CUDA_ARCH__``. We dlopen it via ctypes directly — no
version-bumping cuda-python needed.

Why this lives in ``blackwell/`` (not ``common/``): cu13 NVRTC is
ONLY required for Blackwell tcgen05.mma. Hopper / Ampere / older
arches use the cu12 NVRTC bundled with cuda-python. So the cu13
path is an arch-specific specialization on top of NVIDIA's
vendor-common JIT story.

Wave 1.14 moves this from ``runtime/native/cuda.py``. The original
location re-exports for backward compatibility — old callers keep
working, but new code should import from here.

Search order, library loading + builtins handling, etc. are
unchanged from the original — see the per-function docstrings for
the bridge-round rationale.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

# Avoid an import cycle — the original location re-exports BACK to
# us, so we can't import its CudaUnavailableError without going
# through the old module path. Define our own local alias instead;
# downstream callers receive the same exception type.
from compgen.runtime.native.cuda import CudaUnavailableError

_CU13_NVRTC_LIB: Any | None = None


def _resolve_cu13_nvrtc_lib_path() -> str | None:
    """Locate ``libnvrtc.so.13`` shipped by the cu13 NVRTC wheel.

    Search order (per bridge #091 — bwell ships the unified
    ``nvidia.cu13/`` layout, not the older split
    ``nvidia.cuda_nvrtc/``):

    1. ``$COMPGEN_CU13_NVRTC_LIB_PATH`` env override (if it points
       at an existing file).
    2. ``nvidia.cu13/lib/libnvrtc.so.13*`` — the unified torch≥2.6
       meta-wheel layout.
    3. ``nvidia.cuda_nvrtc/lib/libnvrtc.so.13*`` and at the package
       root — the older split-package layout.

    Returns the full path to the .so or ``None`` if none reachable.
    """
    env_path = os.environ.get("COMPGEN_CU13_NVRTC_LIB_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    candidate_packages = ("nvidia.cu13", "nvidia.cuda_nvrtc")
    for pkg_name in candidate_packages:
        try:
            pkg = __import__(pkg_name, fromlist=["__path__"])
        except ImportError:
            continue
        paths = list(getattr(pkg, "__path__", []) or [])
        if not paths:
            continue
        pkg_root = Path(paths[0]).resolve()
        candidates = list(pkg_root.glob("lib/libnvrtc.so.13*")) + list(pkg_root.glob("libnvrtc.so.13*"))
        for cand in candidates:
            if cand.is_file():
                return str(cand)
    return None


def cu13_nvrtc_available() -> bool:
    """Cheap probe — is the cu13 NVRTC reachable?"""
    return _resolve_cu13_nvrtc_lib_path() is not None


def _load_cu13_nvrtc() -> Any:
    """Load + cache the cu13 NVRTC ctypes wrapper.

    Loads ``libnvrtc.so.13`` once and binds the prototypes for the
    seven functions the megakernel compile needs. Subsequent calls
    return the cached handle.

    Per bridge #095: dlopens ``libnvrtc-builtins.so.13.0`` first
    with ``RTLD_GLOBAL`` so its symbols are visible when
    libnvrtc.so.13's internal lookup happens during compile.
    Without this, the compile fails standalone (only works when
    torch's bootstrap already set up LD_LIBRARY_PATH).
    """
    global _CU13_NVRTC_LIB
    if _CU13_NVRTC_LIB is not None:
        return _CU13_NVRTC_LIB

    libpath = _resolve_cu13_nvrtc_lib_path()
    if libpath is None:
        raise CudaUnavailableError(
            "cu13 NVRTC not reachable — needed for "
            "use_cu13_nvrtc=True. Install with `pip install "
            "nvidia-cu13` (or `nvidia-cuda-nvrtc-cu13`; both are "
            "torch>=2.6 deps on Blackwell hosts) or set "
            "$COMPGEN_CU13_NVRTC_LIB_PATH to a libnvrtc.so.13. "
            "Set use_cu13_nvrtc=False to fall back to the cu12 "
            "NVRTC bundled with cuda-python."
        )

    nvrtc_dir = os.path.dirname(libpath)
    builtins_candidates = list(Path(nvrtc_dir).glob("libnvrtc-builtins.so.13*")) + list(
        Path(nvrtc_dir).glob("libnvrtc-builtins.so*")
    )
    for cand in builtins_candidates:
        if cand.is_file():
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
    # Belt-and-suspenders: also prepend to LD_LIBRARY_PATH for any
    # subprocess we might spawn (the Python-side dlopen above is
    # what actually unblocks the in-process compile).
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if nvrtc_dir not in existing_ld.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = nvrtc_dir + os.pathsep + existing_ld if existing_ld else nvrtc_dir

    try:
        lib = ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise CudaUnavailableError(f"failed to load cu13 NVRTC at {libpath}: {exc!r}") from exc

    # Bind the seven NVRTC functions the megakernel compile path uses.
    lib.nvrtcCreateProgram.restype = ctypes.c_int
    lib.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCompileProgram.restype = ctypes.c_int
    lib.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
    lib.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetProgramLog.restype = ctypes.c_int
    lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcGetPTXSize.restype = ctypes.c_int
    lib.nvrtcGetPTXSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetPTX.restype = ctypes.c_int
    lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.restype = ctypes.c_int
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    _CU13_NVRTC_LIB = lib
    return lib


def _compile_via_cu13_nvrtc(
    *,
    cuda_source: str,
    kernel_name: str,
    opts: list[bytes],
) -> bytes:
    """ctypes-direct NVRTC compile via cu13's libnvrtc.so.13.

    Bypasses ``cuda.bindings.nvrtc`` so the megakernel's PTX is
    emitted by the cu13 NVRTC (knows sm_100/sm_120) rather than
    whatever the cuda-python wrapper would have picked.

    Raises:
        CudaUnavailableError: cu13 NVRTC not present.
        RuntimeError: NVRTC compile failure (full log surfaced).
    """
    NVRTC_SUCCESS = 0
    lib = _load_cu13_nvrtc()

    src_b = cuda_source.encode("utf-8")
    name_b = f"{kernel_name}.cu".encode()

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(
        ctypes.byref(prog),
        src_b,
        name_b,
        0,
        None,
        None,
    )
    if rc != NVRTC_SUCCESS:
        raise RuntimeError(f"cu13 NVRTC nvrtcCreateProgram failed for {kernel_name}: status {rc}")

    opts_arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), opts_arr)
    if rc != NVRTC_SUCCESS:
        log_size = ctypes.c_size_t(0)
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_buf = ctypes.create_string_buffer(log_size.value)
        lib.nvrtcGetProgramLog(prog, log_buf)
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(
            f"cu13 NVRTC compile failed for {kernel_name} "
            f"(opts={[o.decode() for o in opts]}):\n" + log_buf.value.decode("utf-8", errors="replace")
        )

    ptx_size = ctypes.c_size_t(0)
    rc = lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    if rc != NVRTC_SUCCESS:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"cu13 NVRTC nvrtcGetPTXSize failed: status {rc}")
    ptx_buf = ctypes.create_string_buffer(ptx_size.value)
    rc = lib.nvrtcGetPTX(prog, ptx_buf)
    if rc != NVRTC_SUCCESS:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"cu13 NVRTC nvrtcGetPTX failed: status {rc}")

    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx_buf.raw[: ptx_size.value]
