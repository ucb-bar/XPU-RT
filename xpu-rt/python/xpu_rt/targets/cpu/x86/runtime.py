"""x86 CPU runtime — JIT compile via clang/gcc + ``ctypes.CDLL`` dispatch.

The agentic-compilation contract for CPU: the runtime takes a
universal source string (multiple ``DeviceFunctionSource`` bodies
glued together with a serial dispatcher), invokes the system C++
compiler to build a ``.so``, and exposes ``dispatch()`` that calls
into the symbol via ctypes.

No event tensors over PCIe means the megakernel collapses to a
serial task chain — the dispatcher is just a sequence of function
calls in topological order, with no inter-task synchronization
needed. Intra-CPU sync (between two function calls) is effectively
free on coherent caches.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class X86Runtime:
    """Compile + load + dispatch CPU bodies."""

    def compile_source(
        self,
        *,
        source: str,
        symbol_name: str,
        compile_flags: tuple[str, ...] = (),
    ) -> Any:
        """Build ``source`` with the system C++ compiler and load
        the resulting shared object. Returns a ``ctypes.CDLL``
        handle.

        Args:
            source: Full C++ source string. Must define a function
                named ``symbol_name`` with signature
                ``void <symbol_name>(int task_id, int sm_id, void **buffers)``.
            symbol_name: Symbol the universal dispatch path looks
                up after load. Doesn't need to be unique across
                bundles — each bundle gets its own .so.
            compile_flags: Extra flags to forward to the compiler.
                Defaults already include ``-O2 -fPIC -shared``.

        Raises:
            RuntimeError: compiler invocation failed. The full
                stderr is included in the message so the agent's
                audit query can read it.
        """
        cxx = self._find_cxx()
        if cxx is None:
            raise RuntimeError(
                "x86 runtime: no C++ compiler reachable on PATH. "
                "Install clang or gcc; the runtime needs one to "
                "JIT compile bodies."
            )

        # Source goes to a temp file; .so lands next to it. Both
        # cleaned up when the bundle goes out of scope (the user
        # holds the CDLL handle which keeps the .so alive on disk).
        #
        # Compile as C (``-x c``) rather than C++. Bodies don't use
        # any STL/C++ runtime; treating them as C avoids needing
        # libstdc++ on minimal Linux hosts (caught in Wave 1.15
        # tests on a clang-without-libstdc++ box).
        tmpdir = Path(tempfile.mkdtemp(prefix="compgen_cpu_"))
        src_path = tmpdir / f"{symbol_name}.c"
        so_path = tmpdir / f"{symbol_name}.so"
        src_path.write_text(source)

        cmd = [cxx, "-x", "c", "-O2", "-fPIC", "-shared", "-o", str(so_path), str(src_path), *compile_flags]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"x86 runtime: C++ compile failed:\n"
                f"  command: {' '.join(cmd)}\n"
                f"  stderr:\n{result.stderr}\n"
                f"  source ({len(source)} bytes) preserved at: {src_path}"
            )

        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError as exc:
            raise RuntimeError(f"x86 runtime: ctypes.CDLL load failed for {so_path}: {exc!r}") from exc

        return lib

    def dispatch(
        self,
        *,
        library_handle: Any,
        kernel_params: Any,
    ) -> None:
        """Call into the loaded symbol. ``kernel_params`` is a
        :class:`ctypes.Array` of ``ctypes.c_void_p`` (the buffer
        pointers); the universal dispatch layer marshals torch
        tensors → ``data_ptr()`` ints → ``c_void_p`` array.

        Synchronous — the call returns after the kernel completes."""
        # ``kernel_params`` carries (symbol_name, *buffer_ptrs).
        # The universal dispatch layer hands us a tuple of those.
        if not isinstance(kernel_params, tuple) or len(kernel_params) < 1:
            raise ValueError("x86 runtime: kernel_params must be (symbol_name, *buffer_ptrs)")
        symbol_name, *buffer_ptrs = kernel_params
        sym = getattr(library_handle, symbol_name, None)
        if sym is None:
            raise RuntimeError(f"x86 runtime: symbol {symbol_name!r} not found in loaded .so")
        # void <sym>(int task_id, int sm_id, void **buffers).
        sym.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        sym.restype = None
        n = len(buffer_ptrs)
        BufArrT = ctypes.c_void_p * n
        bufarr = BufArrT(*[ctypes.c_void_p(int(p)) for p in buffer_ptrs])
        # task_id=0, sm_id=0 — CPU has no per-task / per-SM concept;
        # the bodies ignore them on this path.
        sym(0, 0, bufarr)

    def _find_cxx(self) -> str | None:
        """Prefer C compilers (clang, gcc) over C++ ones to avoid
        the libstdc++ link dep — bodies are pure C compiled with
        ``-x c``, so a C-only toolchain is sufficient."""
        import shutil

        # Allow override for hosts with non-standard toolchains.
        env = os.environ.get("COMPGEN_CXX")
        if env:
            return env

        # C compilers first — they don't require libstdc++.
        for name in ("clang", "gcc", "clang++", "g++"):
            path = shutil.which(name)
            if path is not None:
                return path
        return None
