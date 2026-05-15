"""Instrumentation configuration for dual-mode runtime.

Defines what to measure, at what granularity, and which hardware
counters to use.  This configuration drives:

    1. **C codegen flags** — ``CG_TRACE_ENABLED``, ``CG_PERF_BACKEND``,
       ``CG_INSTRUMENTATION_LEVEL`` in ``CMakeLists.txt``.
    2. **HAL codegen** — whether to emit ``CG_TRACE_BEGIN/END`` calls
       around alloc/dispatch/sync in generated HAL drivers.
    3. **Zephyr codegen** — whether to enable ``CONFIG_TRACING``,
       select tracing backend, emit ``sys_trace_*`` calls.
    4. **Profiler adapters** — which adapters to activate and which
       counters to read.

The agentic LLM configures this via :class:`ConfigureProfilingAction`
in the compilation loop.

Invariants:
    - ``InstrumentationLevel.NONE`` guarantees zero runtime overhead.
    - Counter names reference those declared in
      ``ProfilingSpec.backends[*].counters``.
    - The configuration is serializable to YAML for the artifact bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import structlog

from compgen.targetgen.hardware_spec import ProfilingSpec

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Instrumentation level
# ---------------------------------------------------------------------------


class InstrumentationLevel(IntEnum):
    """Granularity of runtime instrumentation.

    Ordered by overhead: ``NONE`` < ``OP_LEVEL`` < ``TILE_LEVEL`` < ``FULL``.
    """

    NONE = 0
    OP_LEVEL = 1
    TILE_LEVEL = 2
    FULL = 3


# ---------------------------------------------------------------------------
# Perf backend enum
# ---------------------------------------------------------------------------


class PerfBackend(Enum):
    """Hardware performance counter backend.

    Selected at C compile time via ``CG_PERF_BACKEND``.
    """

    NONE = "none"
    LINUX_PERF = "linux_perf"
    ZEPHYR_TIMING = "zephyr_timing"
    BARE_METAL_CSR = "bare_metal_csr"
    CUDA_CUPTI = "cuda_cupti"


# ---------------------------------------------------------------------------
# Trace backend enum
# ---------------------------------------------------------------------------


class TraceBackend(Enum):
    """Trace output backend.

    Determines how trace data is collected and exported.
    """

    NONE = "none"
    RING_BUFFER = "ring_buffer"
    CHROME_TRACE = "chrome_trace"
    ZEPHYR_TRACING = "zephyr_tracing"
    ZEPHYR_CTF = "zephyr_ctf"
    ZEPHYR_SYSVIEW = "zephyr_sysview"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Counter group
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterGroup:
    """A group of hardware performance counters to read together.

    Attributes:
        name: Group name (e.g., ``"compute"``, ``"memory"``,
            ``"energy"``).
        counters: Counter names to enable (must match those declared
            in ``ProfilingSpec.backends[*].counters``).
        sample_every_n: Read counters every N dispatches (1 = every
            dispatch, higher = less overhead).
    """

    name: str
    counters: list[str] = field(default_factory=list)
    sample_every_n: int = 1


# ---------------------------------------------------------------------------
# Trace point filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceFilter:
    """Controls which trace points are active.

    Attributes:
        categories: Active trace categories (e.g., ``["engine",
            "dispatch", "dma", "sync"]``).  Empty = all.
        region_ids: Only trace these region IDs.  Empty = all.
        min_duration_us: Suppress trace events shorter than this.
    """

    categories: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)
    min_duration_us: float = 0.0


# ---------------------------------------------------------------------------
# Instrumentation config
# ---------------------------------------------------------------------------


@dataclass
class InstrumentationConfig:
    """Complete instrumentation configuration.

    This is the single source of truth for what instrumentation is
    active.  Produced by the agentic LLM (via
    ``ConfigureProfilingAction``) or constructed manually.

    Attributes:
        level: Instrumentation granularity.
        perf_backend: Hardware counter backend.
        trace_backend: Trace output backend.
        counter_groups: Performance counter groups to enable.
        trace_filter: Trace point filter.
        trace_buffer_size: Trace ring buffer size in bytes.
        output_dir: Where to write trace/counter output files.
        zephyr_trace_backend: Zephyr-specific tracing backend
            (``"uart"``, ``"usb"``, ``"ram"``, ``"posix"``,
            ``"semihosting"``).
        metadata: Additional LLM-tunable parameters.
    """

    level: InstrumentationLevel = InstrumentationLevel.NONE
    perf_backend: PerfBackend = PerfBackend.NONE
    trace_backend: TraceBackend = TraceBackend.NONE
    counter_groups: list[CounterGroup] = field(default_factory=list)
    trace_filter: TraceFilter = field(default_factory=TraceFilter)
    trace_buffer_size: int = 1024 * 1024  # 1 MB default
    output_dir: str = ""
    zephyr_trace_backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_enabled(self) -> bool:
        """Whether any instrumentation is active."""
        return self.level > InstrumentationLevel.NONE

    @property
    def has_counters(self) -> bool:
        """Whether hardware performance counters are active."""
        return self.perf_backend != PerfBackend.NONE and bool(self.counter_groups)

    @property
    def has_tracing(self) -> bool:
        """Whether tracing is active."""
        return self.trace_backend != TraceBackend.NONE

    def cmake_defines(self) -> dict[str, str]:
        """Generate CMake cache variables for the C build.

        Returns:
            Dict of ``-D`` variable name → value.
        """
        defines: dict[str, str] = {}

        if self.level > InstrumentationLevel.NONE:
            defines["CG_TRACE_ENABLED"] = "ON"
            defines["CG_INSTRUMENTATION_LEVEL"] = str(self.level.value)

        if self.perf_backend != PerfBackend.NONE:
            defines["CG_PERF_BACKEND"] = self.perf_backend.value

        if self.trace_buffer_size != 1024 * 1024:
            defines["CG_TRACE_BUFFER_SIZE"] = str(self.trace_buffer_size)

        return defines

    def zephyr_kconfig(self) -> dict[str, str]:
        """Generate Zephyr Kconfig overrides for instrumented builds.

        Returns:
            Dict of ``CONFIG_*`` → value for ``prj.conf``.
        """
        kconfig: dict[str, str] = {}

        if not self.is_enabled:
            return kconfig

        # Core tracing
        kconfig["CONFIG_TRACING"] = "y"
        kconfig["CONFIG_SCHED_THREAD_USAGE_ANALYSIS"] = "y"

        # Timing functions for cycle counting
        kconfig["CONFIG_TIMING_FUNCTIONS"] = "y"

        # Backend selection
        backend = self.zephyr_trace_backend or "ram"
        backend_configs = {
            "uart": "CONFIG_TRACING_BACKEND_UART",
            "usb": "CONFIG_TRACING_BACKEND_USB",
            "ram": "CONFIG_TRACING_BACKEND_RAM",
            "posix": "CONFIG_TRACING_BACKEND_POSIX",
            "semihosting": "CONFIG_TRACING_BACKEND_SEMIHOSTING",
        }
        config_key = backend_configs.get(backend, "CONFIG_TRACING_BACKEND_RAM")
        kconfig[config_key] = "y"

        # Tracing format
        if self.trace_backend == TraceBackend.ZEPHYR_CTF:
            kconfig["CONFIG_TRACING_CTF"] = "y"
        elif self.trace_backend == TraceBackend.ZEPHYR_SYSVIEW:
            kconfig["CONFIG_SEGGER_SYSTEMVIEW"] = "y"

        # Buffer size
        if self.trace_buffer_size > 0:
            kconfig["CONFIG_TRACING_BUFFER_SIZE"] = str(
                min(self.trace_buffer_size, 65536)  # Zephyr max
            )

        # Thread monitoring for per-thread profiling
        kconfig["CONFIG_THREAD_MONITOR"] = "y"
        kconfig["CONFIG_THREAD_NAME"] = "y"
        kconfig["CONFIG_THREAD_STACK_INFO"] = "y"

        return kconfig

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML output in artifact bundle."""
        return {
            "level": self.level.name,
            "perf_backend": self.perf_backend.value,
            "trace_backend": self.trace_backend.value,
            "counter_groups": [
                {"name": cg.name, "counters": cg.counters, "sample_every_n": cg.sample_every_n}
                for cg in self.counter_groups
            ],
            "trace_filter": {
                "categories": self.trace_filter.categories,
                "region_ids": self.trace_filter.region_ids,
                "min_duration_us": self.trace_filter.min_duration_us,
            },
            "trace_buffer_size": self.trace_buffer_size,
            "output_dir": self.output_dir,
            "zephyr_trace_backend": self.zephyr_trace_backend,
        }


# ---------------------------------------------------------------------------
# Config inference from ProfilingSpec
# ---------------------------------------------------------------------------


def infer_instrumentation(
    profiling_spec: ProfilingSpec,
    *,
    level: InstrumentationLevel = InstrumentationLevel.OP_LEVEL,
    runtime_env: str = "linux_userspace",
) -> InstrumentationConfig:
    """Infer an instrumentation config from a profiling spec.

    Reads the ``ProfilingSpec`` to determine which backends and counters
    are available, then builds a config at the requested level.

    Args:
        profiling_spec: Hardware profiling capabilities.
        level: Desired instrumentation level.
        runtime_env: Runtime environment (``"linux_userspace"``,
            ``"zephyr_rtos"``, ``"bare_metal"``).

    Returns:
        An ``InstrumentationConfig`` populated from the spec.
    """
    if level == InstrumentationLevel.NONE:
        return InstrumentationConfig()

    # Select perf backend based on runtime environment
    perf_backend = _select_perf_backend(profiling_spec, runtime_env)

    # Select trace backend
    trace_backend = _select_trace_backend(profiling_spec, runtime_env)

    # Build counter groups from available counters
    counter_groups = _build_counter_groups(profiling_spec, level)

    # Zephyr-specific trace backend
    zephyr_trace = ""
    if runtime_env == "zephyr_rtos":
        zephyr_trace = _select_zephyr_trace_backend(profiling_spec)

    return InstrumentationConfig(
        level=level,
        perf_backend=perf_backend,
        trace_backend=trace_backend,
        counter_groups=counter_groups,
        trace_buffer_size=1024 * 1024 if level <= InstrumentationLevel.OP_LEVEL else 4 * 1024 * 1024,
        zephyr_trace_backend=zephyr_trace,
    )


def _select_perf_backend(spec: ProfilingSpec, runtime_env: str) -> PerfBackend:
    """Select the appropriate perf counter backend."""
    env_to_backend = {
        "linux_userspace": PerfBackend.LINUX_PERF,
        "zephyr_rtos": PerfBackend.ZEPHYR_TIMING,
        "bare_metal": PerfBackend.BARE_METAL_CSR,
    }

    # Check if spec has a CUPTI-style backend
    for be in spec.backends:
        if "cupti" in be.name.lower() or "cuda" in be.name.lower():
            return PerfBackend.CUDA_CUPTI

    return env_to_backend.get(runtime_env, PerfBackend.LINUX_PERF)


def _select_trace_backend(spec: ProfilingSpec, runtime_env: str) -> TraceBackend:
    """Select the appropriate trace backend."""
    if runtime_env == "zephyr_rtos":
        # Check if spec prefers CTF or SystemView
        for be in spec.backends:
            if "sysview" in be.name.lower() or "systemview" in be.name.lower():
                return TraceBackend.ZEPHYR_SYSVIEW
            if "ctf" in be.name.lower():
                return TraceBackend.ZEPHYR_CTF
        return TraceBackend.ZEPHYR_TRACING

    return TraceBackend.CHROME_TRACE


def _select_zephyr_trace_backend(spec: ProfilingSpec) -> str:
    """Select Zephyr tracing output backend."""
    for be in spec.backends:
        name = be.name.lower()
        if "uart" in name:
            return "uart"
        if "usb" in name:
            return "usb"
        if "semihosting" in name:
            return "semihosting"
    return "ram"  # safe default


def _build_counter_groups(
    spec: ProfilingSpec,
    level: InstrumentationLevel,
) -> list[CounterGroup]:
    """Build counter groups from available profiler backends."""
    groups: list[CounterGroup] = []

    all_counters: list[str] = []
    for be in spec.backends:
        all_counters.extend(be.counters)

    if not all_counters:
        return groups

    # Classify counters into groups
    compute_counters = [
        c for c in all_counters if any(k in c.lower() for k in ("cycle", "instruction", "flop", "compute", "warp"))
    ]
    memory_counters = [
        c for c in all_counters if any(k in c.lower() for k in ("cache", "memory", "bandwidth", "dram", "hbm", "l2"))
    ]
    other_counters = [c for c in all_counters if c not in compute_counters and c not in memory_counters]

    sample_rate = 1 if level >= InstrumentationLevel.TILE_LEVEL else 10

    if compute_counters:
        groups.append(
            CounterGroup(
                name="compute",
                counters=compute_counters,
                sample_every_n=sample_rate,
            )
        )
    if memory_counters:
        groups.append(
            CounterGroup(
                name="memory",
                counters=memory_counters,
                sample_every_n=sample_rate,
            )
        )
    if other_counters:
        groups.append(
            CounterGroup(
                name="other",
                counters=other_counters,
                sample_every_n=sample_rate * 10,
            )
        )

    return groups


__all__ = [
    "CounterGroup",
    "InstrumentationConfig",
    "InstrumentationLevel",
    "PerfBackend",
    "TraceBackend",
    "TraceFilter",
    "infer_instrumentation",
]
