"""x86 CPU probe — detect AVX-512 / AVX2 / scalar fallback.

Reads ``/proc/cpuinfo`` flags on Linux, falls back to a pure-Python
``platform`` check elsewhere. The probe never raises — every
host has SOME x86 capability (or isn't x86, in which case
``is_available`` returns False).

This satisfies a subset of :class:`compgen.targets.gpu.contracts.GpuProbe`
adapted for CPU semantics — no clusters, no library paths in the
GPU sense (the C++ compiler is the runtime dependency).
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


class X86Probe:
    """Detect x86 CPU capability + JIT-toolchain availability.

    Public surface:
    - :meth:`is_available` — host is x86 + a C++ compiler is on PATH.
    - :meth:`device_arch` — ``"x86_avx512"`` | ``"x86_avx2"`` | ``"x86_scalar"``.
    - :meth:`supports_tensor_cores` — False (CPUs don't have them).
    - :meth:`supports_clusters` — False (no GPU-style cluster concept).
    - :meth:`library_paths` — placeholder; CPU has no header-only libs to discover.
    - :meth:`vendor_extras` — surfaces the detected SIMD width + compiler.
    """

    def is_available(self) -> bool:
        """Host runs x86 AND has a working C++ compiler reachable
        (clang or gcc on PATH). Both must be true for the runtime
        to JIT bodies."""
        if not self._is_x86_host():
            return False
        return self._find_cxx_compiler() is not None

    def device_arch(self) -> str:
        flags = self._read_cpu_flags()
        if "avx512f" in flags:
            return "x86_avx512"
        if "avx2" in flags:
            return "x86_avx2"
        return "x86_scalar"

    def supports_clusters(self) -> bool:
        return False

    def supports_tensor_cores(self) -> bool:
        return False

    def library_paths(self) -> dict[str, str | None]:
        compiler = self._find_cxx_compiler()
        return {"cxx_compiler": compiler}

    def vendor_extras(self) -> dict[str, Any]:
        flags = self._read_cpu_flags()
        return {
            "simd_width_bits": (
                512 if "avx512f" in flags else 256 if "avx2" in flags else 128 if "sse2" in flags else 64
            ),
            "cxx_compiler": self._find_cxx_compiler(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }

    # ----- helpers -----

    def _is_x86_host(self) -> bool:
        m = platform.machine().lower()
        return m in ("x86_64", "amd64", "i386", "i686")

    def _read_cpu_flags(self) -> set[str]:
        """Linux cpuinfo flags. Empty on macOS/Windows; the SIMD
        detection there falls back to runtime-checking via ``__builtin_cpu_supports``
        in a future iteration."""
        cpuinfo = Path("/proc/cpuinfo")
        if not cpuinfo.is_file():
            return set()
        try:
            text = cpuinfo.read_text()
        except OSError:
            return set()
        for line in text.splitlines():
            if line.startswith("flags") and ":" in line:
                return set(line.split(":", 1)[1].strip().split())
        return set()

    def _find_cxx_compiler(self) -> str | None:
        """Best clang/gcc on PATH. Preference: clang > clang++ > gcc > g++.
        Returns the absolute path or None."""
        import shutil

        for name in ("clang++", "clang", "g++", "gcc"):
            path = shutil.which(name)
            if path is not None:
                return path
        return None
