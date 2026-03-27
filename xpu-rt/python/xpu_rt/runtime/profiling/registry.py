"""Profiler adapter registry.

Discovers and instantiates profiler adapters based on the
``ProfilingSpec`` declared in the hardware spec.  The agentic LLM
can register custom adapters at runtime.
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.runtime.profiling.adapter import ProfilerAdapter
from compgen.targetgen.hardware_spec import ProfilerBackend, ProfilingSpec

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str, cls: type) -> None:
    """Register a profiler adapter class.

    Args:
        name: Adapter name (matches ``ProfilerBackend.name``).
        cls: Adapter class (must implement :class:`ProfilerAdapter`).
    """
    _ADAPTER_REGISTRY[name] = cls
    log.info("profiler_registry.registered", name=name, cls=cls.__name__)


def get_adapter_class(name: str) -> type | None:
    """Look up a registered adapter class by name."""
    return _ADAPTER_REGISTRY.get(name)


def list_adapters() -> list[str]:
    """List all registered adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())


def create_adapter(backend: ProfilerBackend, **kwargs: Any) -> ProfilerAdapter | None:
    """Create an adapter instance for a profiler backend.

    Args:
        backend: The profiler backend spec.
        **kwargs: Passed to the adapter constructor.

    Returns:
        An adapter instance, or ``None`` if no adapter is registered
        for this backend name.
    """
    cls = _ADAPTER_REGISTRY.get(backend.name)
    if cls is None:
        log.warning("profiler_registry.no_adapter", backend=backend.name)
        return None
    return cls(**kwargs)


def create_adapters_for_spec(spec: ProfilingSpec) -> list[ProfilerAdapter]:
    """Create adapters for all backends in a profiling spec.

    Args:
        spec: The hardware profiling capabilities.

    Returns:
        List of adapter instances (only for backends with registered
        adapters).
    """
    adapters: list[ProfilerAdapter] = []
    for backend in spec.backends:
        adapter = create_adapter(backend)
        if adapter is not None:
            adapters.append(adapter)
    return adapters


# ---------------------------------------------------------------------------
# Auto-registration of built-in adapters
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register the built-in profiler adapters."""
    from compgen.runtime.profiling.adapters.bare_metal_pmu import BareMetalPMUAdapter
    from compgen.runtime.profiling.adapters.cuda_profiler import CudaProfilerAdapter
    from compgen.runtime.profiling.adapters.linux_perf import LinuxPerfAdapter
    from compgen.runtime.profiling.adapters.zephyr_trace import ZephyrTraceAdapter

    register_adapter("perf", LinuxPerfAdapter)
    register_adapter("linux_perf", LinuxPerfAdapter)
    register_adapter("zephyr_trace", ZephyrTraceAdapter)
    register_adapter("cuda_cupti", CudaProfilerAdapter)
    register_adapter("nsight_systems", CudaProfilerAdapter)
    register_adapter("riscv_csr", BareMetalPMUAdapter)
    register_adapter("bare_metal_pmu", BareMetalPMUAdapter)
    register_adapter("etm", BareMetalPMUAdapter)


_register_builtins()


__all__ = [
    "create_adapter",
    "create_adapters_for_spec",
    "get_adapter_class",
    "list_adapters",
    "register_adapter",
]
