"""Tests for the profiling subpackage — adapters, registry, analysis, hooks."""

from __future__ import annotations

from compgen.runtime.profiling.adapter import (
    ProfileSnapshot,
    ProfilerAdapter,
    TileMetrics,
)
from compgen.runtime.profiling.adapters.bare_metal_pmu import BareMetalPMUAdapter
from compgen.runtime.profiling.adapters.cuda_profiler import CudaProfilerAdapter
from compgen.runtime.profiling.adapters.linux_perf import LinuxPerfAdapter
from compgen.runtime.profiling.adapters.zephyr_trace import ZephyrTraceAdapter
from compgen.runtime.profiling.analysis import (
    BottleneckInfo,
    ProfileAnalysis,
    ProfileAnalyzer,
    RooflinePoint,
)
from compgen.runtime.profiling.hooks_codegen import (
    GeneratedHook,
    HookCodeGenerator,
)
from compgen.runtime.profiling.registry import (
    create_adapter,
    create_adapters_for_spec,
    list_adapters,
    register_adapter,
)
from compgen.runtime.instrumentation import (
    InstrumentationConfig,
    InstrumentationLevel,
)
from compgen.targetgen.hardware_spec import ProfilerBackend, ProfilingSpec


# ---- Adapter Protocol ----


class TestAdapterProtocol:
    def test_linux_perf_is_adapter(self) -> None:
        a = LinuxPerfAdapter()
        assert isinstance(a, ProfilerAdapter)

    def test_zephyr_trace_is_adapter(self) -> None:
        a = ZephyrTraceAdapter()
        assert isinstance(a, ProfilerAdapter)

    def test_cuda_profiler_is_adapter(self) -> None:
        a = CudaProfilerAdapter()
        assert isinstance(a, ProfilerAdapter)

    def test_bare_metal_is_adapter(self) -> None:
        a = BareMetalPMUAdapter()
        assert isinstance(a, ProfilerAdapter)


# ---- LinuxPerfAdapter ----


class TestLinuxPerfAdapter:
    def test_lifecycle(self) -> None:
        a = LinuxPerfAdapter()
        assert a.name == "linux_perf"
        assert a.is_active is False
        a.configure({"counters": ["cycles", "cache_misses"]})
        a.start()
        assert a.is_active is True
        counters = a.read_counters()
        assert "cycles" in counters
        a.stop()
        assert a.is_active is False

    def test_snapshot(self) -> None:
        a = LinuxPerfAdapter()
        a.configure({"counters": ["instructions"]})
        a.start()
        snap = a.snapshot()
        assert snap.metadata["backend"] == "linux_perf"
        a.stop()


# ---- ZephyrTraceAdapter ----


class TestZephyrTraceAdapter:
    def test_kconfig(self) -> None:
        a = ZephyrTraceAdapter()
        a.configure({"trace_backend": "uart", "trace_format": "ctf"})
        kc = a.kconfig_overrides()
        assert kc["CONFIG_TRACING"] == "y"
        assert kc["CONFIG_TRACING_BACKEND_UART"] == "y"
        assert kc["CONFIG_TRACING_CTF"] == "y"

    def test_kconfig_sysview(self) -> None:
        a = ZephyrTraceAdapter()
        a.configure({"trace_format": "sysview"})
        kc = a.kconfig_overrides()
        assert kc["CONFIG_SEGGER_SYSTEMVIEW"] == "y"


# ---- CudaProfilerAdapter ----


class TestCudaProfilerAdapter:
    def test_launch_command_nsys(self) -> None:
        a = CudaProfilerAdapter()
        a.configure({"tool": "nsys"})
        cmd = a.launch_command("./my_kernel")
        assert "nsys profile" in cmd
        assert "./my_kernel" in cmd

    def test_launch_command_ncu(self) -> None:
        a = CudaProfilerAdapter()
        a.configure({"tool": "ncu", "counters": ["sm_active"]})
        cmd = a.launch_command("./my_kernel")
        assert "ncu --metrics" in cmd

    def test_nvtx_annotation(self) -> None:
        a = CudaProfilerAdapter()
        ann = a.nvtx_annotation_code("matmul_0")
        assert "nvtxRangePushA" in ann["begin"]
        assert "nvtxRangePop" in ann["end"]
        assert "nvToolsExt.h" in ann["include"]


# ---- BareMetalPMUAdapter ----


class TestBareMetalPMUAdapter:
    def test_csr_read_code_cycles(self) -> None:
        a = BareMetalPMUAdapter()
        a.configure({"counters": ["cycles"]})
        code = a.csr_read_code("cycles")
        assert "rdcycle" in code

    def test_csr_read_code_instructions(self) -> None:
        a = BareMetalPMUAdapter()
        code = a.csr_read_code("instructions")
        assert "rdinstret" in code

    def test_instrumentation_code(self) -> None:
        a = BareMetalPMUAdapter()
        a.configure({"counters": ["cycles", "instructions"]})
        ic = a.instrumentation_code()
        assert "declarations" in ic
        assert "start" in ic
        assert "stop" in ic
        assert "read" in ic
        assert "_start_cycles" in ic["declarations"]


# ---- Registry ----


class TestRegistry:
    def test_builtins_registered(self) -> None:
        names = list_adapters()
        assert "perf" in names
        assert "linux_perf" in names
        assert "zephyr_trace" in names
        assert "cuda_cupti" in names
        assert "riscv_csr" in names

    def test_create_adapter(self) -> None:
        backend = ProfilerBackend(name="perf")
        adapter = create_adapter(backend)
        assert adapter is not None
        assert adapter.name == "linux_perf"

    def test_create_adapter_unknown(self) -> None:
        backend = ProfilerBackend(name="unknown_profiler")
        adapter = create_adapter(backend)
        assert adapter is None

    def test_create_adapters_for_spec(self) -> None:
        spec = ProfilingSpec(
            backends=[
                ProfilerBackend(name="perf"),
                ProfilerBackend(name="zephyr_trace"),
            ],
        )
        adapters = create_adapters_for_spec(spec)
        assert len(adapters) == 2

    def test_register_custom(self) -> None:
        class MyAdapter:
            @property
            def name(self) -> str:
                return "my_custom"

            @property
            def is_active(self) -> bool:
                return False

            def configure(self, config: dict) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def read_counters(self) -> dict[str, float]:
                return {}

            def get_tile_breakdown(self, region_id: str) -> list:
                return []

            def export_trace(self, path: str) -> None:
                pass

            def snapshot(self) -> ProfileSnapshot:
                return ProfileSnapshot()

        register_adapter("my_custom_test", MyAdapter)
        assert "my_custom_test" in list_adapters()


# ---- ProfileAnalyzer ----


class TestProfileAnalyzer:
    def test_empty_analysis(self) -> None:
        analyzer = ProfileAnalyzer(peak_gflops=100.0, peak_bandwidth_gbps=1000.0)
        result = analyzer.analyze([])
        assert result.total_latency_us == 0.0
        assert result.bottlenecks == []

    def test_bottleneck_detection(self) -> None:
        analyzer = ProfileAnalyzer(peak_gflops=100.0, peak_bandwidth_gbps=1000.0)
        snapshots = [
            ProfileSnapshot(
                tile_metrics=[
                    TileMetrics(region_id="matmul_0", latency_us=80.0),
                    TileMetrics(region_id="relu_0", latency_us=5.0),
                    TileMetrics(region_id="add_0", latency_us=2.0),
                ],
            ),
        ]
        result = analyzer.analyze(snapshots)
        assert result.total_latency_us == 87.0
        assert len(result.bottlenecks) >= 1
        assert result.bottlenecks[0].region_id == "matmul_0"

    def test_roofline(self) -> None:
        analyzer = ProfileAnalyzer(peak_gflops=100.0, peak_bandwidth_gbps=1000.0)
        snapshots = [
            ProfileSnapshot(
                tile_metrics=[
                    TileMetrics(region_id="matmul_0", latency_us=100.0),
                ],
            ),
        ]
        result = analyzer.analyze(
            snapshots,
            region_flops={"matmul_0": 1e9},
            region_bytes={"matmul_0": 1e6},
        )
        assert len(result.roofline_points) == 1
        rp = result.roofline_points[0]
        assert rp.arithmetic_intensity == 1000.0  # 1e9/1e6

    def test_summary_for_llm(self) -> None:
        analysis = ProfileAnalysis(
            total_latency_us=100.0,
            compute_utilization=0.75,
            per_region_latency_us={"matmul_0": 80.0, "relu_0": 20.0},
            bottlenecks=[
                BottleneckInfo(
                    region_id="matmul_0",
                    kind="compute_bound",
                    severity=0.8,
                    suggested_action="tile",
                ),
            ],
        )
        s = analysis.summary_for_llm()
        assert s["total_latency_us"] == 100.0
        assert s["num_bottlenecks"] == 1
        assert len(s["top_bottlenecks"]) == 1
        assert len(s["hottest_regions"]) == 2


# ---- HookCodeGenerator ----


class TestHookCodeGenerator:
    def test_disabled(self) -> None:
        spec = ProfilingSpec()
        instr = InstrumentationConfig(level=InstrumentationLevel.NONE)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        assert len(result.hooks) == 0

    def test_op_level(self) -> None:
        spec = ProfilingSpec()
        instr = InstrumentationConfig(level=InstrumentationLevel.OP_LEVEL)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        assert "pre_dispatch" in result.hooks
        assert "post_dispatch" in result.hooks
        assert "pre_dma" in result.hooks
        assert "CG_TRACE_BEGIN" in result.hooks["pre_dispatch"].code

    def test_tile_level_has_counter_hooks(self) -> None:
        spec = ProfilingSpec()
        instr = InstrumentationConfig(level=InstrumentationLevel.TILE_LEVEL)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        assert "pre_tile" in result.hooks
        assert "post_tile" in result.hooks
        assert "cg_perf_start" in result.hooks["pre_tile"].code

    def test_custom_hooks_merged(self) -> None:
        spec = ProfilingSpec(
            custom_hooks={
                "pre_dispatch": "my_custom_trace_begin();",
                "post_kernel": "my_custom_kernel_end();",
            },
        )
        instr = InstrumentationConfig(level=InstrumentationLevel.OP_LEVEL)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        # Custom hook overrides standard pre_dispatch
        assert result.hooks["pre_dispatch"].code == "my_custom_trace_begin();"
        # Custom hook adds new hook point
        assert "post_kernel" in result.hooks

    def test_header_includes(self) -> None:
        spec = ProfilingSpec()
        instr = InstrumentationConfig(level=InstrumentationLevel.OP_LEVEL)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        assert "compgen/trace.h" in result.header_code

    def test_metadata(self) -> None:
        spec = ProfilingSpec(custom_hooks={"x": "y();"})
        instr = InstrumentationConfig(level=InstrumentationLevel.FULL)
        gen = HookCodeGenerator(spec, instr)
        result = gen.generate()
        assert result.metadata["level"] == "FULL"
        assert result.metadata["custom_hooks"] == 1
