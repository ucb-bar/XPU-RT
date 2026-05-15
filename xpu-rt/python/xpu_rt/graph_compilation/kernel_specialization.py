"""DEPRECATED — superseded (``compgen.graph_compilation.kernel_codegen``).

This module shipped with as the data-only kernel-specialization
request emitter. (Phase C) replaced its schema
(``kernel_specialization_request_v1``) with the leaner
``kernel_codegen_request_v1`` that points at materialised contract
files instead of embedding shape/tile/layout/dtype inline. The
``04_kernel_specialization/`` directory is no longer written by the
pipeline; everything lives under ``04_kernel_codegen/`` now.

This file is preserved as a thin shim only to avoid breaking imports
in third-party code. New callers must use
``compgen.graph_compilation.kernel_codegen.run_kernel_codegen_request``.

The legacy symbols below raise typed ``DeprecationError`` (a subclass
of ``RuntimeError``) so any rogue caller fails loud rather than
silently writing to the deprecated directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DeprecationError(RuntimeError):
    """Raised when a deprecated entry point is invoked."""


def build_kernel_specialization_request(*args: Any, **kwargs: Any) -> Any:
    raise DeprecationError(
        "build_kernel_specialization_request is retired in M-42. Use "
        "compgen.graph_compilation.kernel_codegen.build_kernel_codegen_request "
        "instead. The new schema is kernel_codegen_request_v1; the new "
        "directory is 04_kernel_codegen/."
    )


def run_kernel_specialization_request(*args: Any, **kwargs: Any) -> Any:
    raise DeprecationError(
        "run_kernel_specialization_request is retired in M-42. Use "
        "compgen.graph_compilation.kernel_codegen.run_kernel_codegen_request "
        "instead. The new schema is kernel_codegen_request_v1; the new "
        "directory is 04_kernel_codegen/."
    )


# Legacy schema-name shim — keep importable for any third-party code
# that referenced the schema string.
KERNEL_SPECIALIZATION_REQUEST_SCHEMA = "kernel_specialization_request_v1"
KERNEL_CODEGEN_REQUEST_SCHEMA = "kernel_codegen_request_v1"
