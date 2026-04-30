"""Python bindings for ``libcompgen_rt``.

This package is the Python side of CompGen's new native HAL
(:mod:`runtime/native/libcompgen_rt`).  It wraps the C11 shared library
via :mod:`ctypes` in a thin, idiomatic Python surface.

If the shared library is not built, importing submodules still works
but :func:`load_library` raises :class:`RuntimeError`.  Higher layers
can catch and fall back to the pure-Python runtime.
"""

from __future__ import annotations

from compgen.runtime.native.library import (
    CgRtError,
    available,
    load_library,
)

__all__ = [
    "CgRtError",
    "available",
    "load_library",
]
