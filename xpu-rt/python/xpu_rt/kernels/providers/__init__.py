"""Kernel provider implementations.

Each module wraps an existing kernel search backend as a
``KernelProvider`` that communicates bidirectionally with
CompGen's unified memory.
"""

from __future__ import annotations

__all__: list[str] = []

# Extension point marker — enables programmatic discovery
__extension_point__ = True
__extension_type__ = "kernel_provider"
__extension_protocol__ = "compgen.kernels.provider.KernelProvider"
__extension_template__ = "compgen.kernels.providers._template"
