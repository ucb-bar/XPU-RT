"""Kernel provider implementations.

Each module wraps an existing kernel search backend as a
``KernelProvider`` that communicates bidirectionally with
CompGen's unified memory.

In-tree providers are imported lazily through ``__getattr__`` so that
``from compgen.kernels.providers import KernelBlasterProvider`` works
without forcing every downstream backend (autocomp, KB, Exo, …) to load
at package-import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compgen.kernels.providers.autocomp import (  # noqa: F401
        AutocompProvider,
        ExoProvider,
    )
    from compgen.kernels.providers.kernelblaster import (  # noqa: F401
        KernelBlasterProvider,
    )


_PROVIDER_MAP = {
    "AutocompProvider": "compgen.kernels.providers.autocomp",
    "ExoProvider": "compgen.kernels.providers.autocomp",
    "ExoRiscvOpuProvider": "compgen.kernels.providers.exo_riscv_opu",
    "KernelBlasterProvider": "compgen.kernels.providers.kernelblaster",
}


def __getattr__(name: str) -> Any:
    module_path = _PROVIDER_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module 'compgen.kernels.providers' has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, name)


__all__ = list(_PROVIDER_MAP)

# Extension point marker — enables programmatic discovery
__extension_point__ = True
__extension_type__ = "kernel_provider"
__extension_protocol__ = "compgen.kernels.provider.KernelProvider"
__extension_template__ = "compgen.kernels.providers._template"
