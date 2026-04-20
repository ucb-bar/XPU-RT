"""Tests for runtime/instrumentation.py -- dual-mode runtime config."""

from __future__ import annotations

from compgen.runtime.instrumentation import (
    CounterGroup,
    InstrumentationConfig,
    InstrumentationLevel,
    PerfBackend,
    TraceBackend,
    TraceFilter,
    infer_instrumentation,
)
from compgen.targetgen.hardware_spec import ProfilerBackend, ProfilingSpec

# ---- InstrumentationLevel ----


class TestInstrumentationLevel:
    def test_ordering(self) -> None:
        assert InstrumentationLevel.NONE < InstrumentationLevel.OP_LEVEL
        assert InstrumentationLevel.OP_LEVEL < InstrumentationLevel.TILE_LEVEL
        assert InstrumentationLevel.TILE_LEVEL < InstrumentationLevel.FULL

    def test_values(self) -> None:
        assert InstrumentationLevel.NONE == 0
        assert InstrumentationLevel.FULL == 3


# ---- InstrumentationConfig ----


class TestInstrumentationConfig:
    def test_defaults_disabled(self) -> None:
        cfg = InstrumentationConfig()
        assert cfg.level == InstrumentationLevel.NONE
        assert cfg.is_enabled is False
        assert cfg.has_counters is False
        assert cfg.has_tracing is False

    def test_enabled(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.OP_LEVEL,
            perf_backend=PerfBackend.LINUX_PERF,
            trace_backend=TraceBackend.CHROME_TRACE,
            counter_groups=[CounterGroup(name="compute", counters=["cycles"])],
        )
        assert cfg.is_enabled is True
        assert cfg.has_counters is True
        assert cfg.has_tracing is True

    def test_cmake_defines_disabled(self) -> None:
        cfg = InstrumentationConfig()
        defines = cfg.cmake_defines()
        assert "CG_TRACE_ENABLED" not in defines

    def test_cmake_defines_enabled(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.TILE_LEVEL,
            perf_backend=PerfBackend.LINUX_PERF,
            trace_buffer_size=4 * 1024 * 1024,
        )
        defines = cfg.cmake_defines()
        assert defines["CG_TRACE_ENABLED"] == "ON"
        assert defines["CG_INSTRUMENTATION_LEVEL"] == "2"
        assert defines["CG_PERF_BACKEND"] == "linux_perf"
        assert defines["CG_TRACE_BUFFER_SIZE"] == str(4 * 1024 * 1024)

    def test_zephyr_kconfig_disabled(self) -> None:
        cfg = InstrumentationConfig()
        kconfig = cfg.zephyr_kconfig()
        assert kconfig == {}

    def test_zephyr_kconfig_enabled(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.OP_LEVEL,
            trace_backend=TraceBackend.ZEPHYR_TRACING,
            zephyr_trace_backend="uart",
        )
        kconfig = cfg.zephyr_kconfig()
        assert kconfig["CONFIG_TRACING"] == "y"
        assert kconfig["CONFIG_TIMING_FUNCTIONS"] == "y"
        assert kconfig["CONFIG_TRACING_BACKEND_UART"] == "y"
        assert kconfig["CONFIG_THREAD_MONITOR"] == "y"

    def test_zephyr_kconfig_ctf(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.OP_LEVEL,
            trace_backend=TraceBackend.ZEPHYR_CTF,
        )
        kconfig = cfg.zephyr_kconfig()
        assert kconfig["CONFIG_TRACING_CTF"] == "y"

    def test_zephyr_kconfig_sysview(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.OP_LEVEL,
            trace_backend=TraceBackend.ZEPHYR_SYSVIEW,
        )
        kconfig = cfg.zephyr_kconfig()
        assert kconfig["CONFIG_SEGGER_SYSTEMVIEW"] == "y"

    def test_zephyr_kconfig_buffer_capped(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.OP_LEVEL,
            trace_buffer_size=1024 * 1024,  # 1 MB
        )
        kconfig = cfg.zephyr_kconfig()
        # Zephyr max is 65536
        assert int(kconfig["CONFIG_TRACING_BUFFER_SIZE"]) <= 65536

    def test_to_dict(self) -> None:
        cfg = InstrumentationConfig(
            level=InstrumentationLevel.TILE_LEVEL,
            perf_backend=PerfBackend.ZEPHYR_TIMING,
            trace_backend=TraceBackend.ZEPHYR_CTF,
            counter_groups=[
                CounterGroup(name="compute", counters=["cycles", "instructions"]),
            ],
            trace_filter=TraceFilter(categories=["dispatch", "dma"]),
            zephyr_trace_backend="ram",
        )
        d = cfg.to_dict()
        assert d["level"] == "TILE_LEVEL"
        assert d["perf_backend"] == "zephyr_timing"
        assert d["trace_backend"] == "zephyr_ctf"
        assert len(d["counter_groups"]) == 1
        assert d["counter_groups"][0]["name"] == "compute"
        assert d["trace_filter"]["categories"] == ["dispatch", "dma"]


# ---- CounterGroup ----


class TestCounterGroup:
    def test_defaults(self) -> None:
        cg = CounterGroup(name="test")
        assert cg.counters == []
        assert cg.sample_every_n == 1

    def test_custom(self) -> None:
        cg = CounterGroup(
            name="memory",
            counters=["cache_misses", "dram_reads"],
            sample_every_n=10,
        )
        assert len(cg.counters) == 2
        assert cg.sample_every_n == 10


# ---- TraceFilter ----


class TestTraceFilter:
    def test_defaults(self) -> None:
        tf = TraceFilter()
        assert tf.categories == []
        assert tf.region_ids == []
        assert tf.min_duration_us == 0.0


# ---- infer_instrumentation ----


class TestInferInstrumentation:
    def test_none_level(self) -> None:
        spec = ProfilingSpec()
        cfg = infer_instrumentation(spec, level=InstrumentationLevel.NONE)
        assert cfg.is_enabled is False

    def test_linux_perf(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(
                    name="perf",
                    counters=["cycles", "instructions", "cache_misses"],
                ),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
            runtime_env="linux_userspace",
        )
        assert cfg.perf_backend == PerfBackend.LINUX_PERF
        assert cfg.trace_backend == TraceBackend.CHROME_TRACE
        assert len(cfg.counter_groups) >= 1

    def test_zephyr_env(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(
                    name="zephyr_trace",
                    counters=["cycles", "thread_switches"],
                ),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
            runtime_env="zephyr_rtos",
        )
        assert cfg.perf_backend == PerfBackend.ZEPHYR_TIMING
        assert cfg.trace_backend == TraceBackend.ZEPHYR_TRACING
        assert cfg.zephyr_trace_backend == "ram"

    def test_cuda_cupti(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(name="cuda_cupti", counters=["sm_active"]),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.TILE_LEVEL,
            runtime_env="linux_userspace",
        )
        assert cfg.perf_backend == PerfBackend.CUDA_CUPTI

    def test_bare_metal(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(
                    name="riscv_csr",
                    counters=["mcycle", "minstret"],
                ),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
            runtime_env="bare_metal",
        )
        assert cfg.perf_backend == PerfBackend.BARE_METAL_CSR

    def test_counter_classification(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(
                    name="full",
                    counters=[
                        "cycles",
                        "instructions",  # compute
                        "cache_misses",
                        "dram_reads",  # memory
                        "power_watts",  # other
                    ],
                ),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.TILE_LEVEL,
            runtime_env="linux_userspace",
        )
        group_names = {g.name for g in cfg.counter_groups}
        assert "compute" in group_names
        assert "memory" in group_names

    def test_tile_level_larger_buffer(self) -> None:
        spec = ProfilingSpec()
        cfg_op = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
        )
        cfg_tile = infer_instrumentation(
            spec,
            level=InstrumentationLevel.TILE_LEVEL,
        )
        assert cfg_tile.trace_buffer_size > cfg_op.trace_buffer_size

    def test_zephyr_uart_backend_selection(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(name="uart_trace"),
            ],
        )
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
            runtime_env="zephyr_rtos",
        )
        assert cfg.zephyr_trace_backend == "uart"

    def test_empty_spec(self) -> None:
        spec = ProfilingSpec()
        cfg = infer_instrumentation(
            spec,
            level=InstrumentationLevel.OP_LEVEL,
            runtime_env="linux_userspace",
        )
        assert cfg.is_enabled is True
        assert cfg.counter_groups == []  # no counters declared
