"""Tests for ProfilingSpec, TopologySpec, and related types in hardware_spec."""

from __future__ import annotations

from compgen.targetgen.hardware_spec import (
    DeploymentTopology,
    HardwareSpec,
    PlatformSpec,
    ProfilerBackend,
    ProfilingSpec,
    TopologyLink,
    TopologyNode,
    TopologySpec,
)


# ---- ProfilerBackend ----


class TestProfilerBackend:
    def test_defaults(self) -> None:
        pb = ProfilerBackend(name="perf")
        assert pb.name == "perf"
        assert pb.kind == "pmu_counters"
        assert pb.counters == []
        assert pb.tile_level is False
        assert pb.integration == "external"
        assert pb.setup_command == ""
        assert pb.sdk_header == ""
        assert pb.requires_root is False

    def test_full_construction(self) -> None:
        pb = ProfilerBackend(
            name="nsight_systems",
            kind="device_trace",
            counters=["sm_active", "dram_read", "dram_write"],
            tile_level=True,
            integration="sdk",
            setup_command="nsys profile --trace=cuda,nvtx",
            sdk_header="nvToolsExt.h",
            requires_root=False,
        )
        assert pb.kind == "device_trace"
        assert len(pb.counters) == 3
        assert pb.tile_level is True
        assert pb.integration == "sdk"

    def test_zephyr_backend(self) -> None:
        pb = ProfilerBackend(
            name="zephyr_trace",
            kind="device_trace",
            counters=["cycles", "thread_switches"],
            integration="embedded",
            sdk_header="zephyr/tracing/tracing.h",
        )
        assert pb.integration == "embedded"
        assert "cycles" in pb.counters


# ---- ProfilingSpec ----


class TestProfilingSpec:
    def test_defaults(self) -> None:
        ps = ProfilingSpec()
        assert ps.backends == []
        assert ps.default_backend == ""
        assert ps.tile_profiling is False
        assert ps.memory_traffic_counters is False
        assert ps.power_monitoring is False
        assert ps.thermal_monitoring is False
        assert ps.custom_hooks == {}

    def test_with_backends(self) -> None:
        ps = ProfilingSpec(
            backends=[
                ProfilerBackend(name="perf", counters=["cycles", "instructions"]),
                ProfilerBackend(name="etm", kind="hardware_trace"),
            ],
            default_backend="perf",
            tile_profiling=True,
            memory_traffic_counters=True,
        )
        assert len(ps.backends) == 2
        assert ps.default_backend == "perf"
        assert ps.tile_profiling is True

    def test_custom_hooks(self) -> None:
        ps = ProfilingSpec(
            custom_hooks={
                "pre_dispatch": "CG_TRACE_BEGIN(\"dispatch\", kernel_name);",
                "post_dispatch": "CG_TRACE_END();",
            },
        )
        assert "pre_dispatch" in ps.custom_hooks
        assert "CG_TRACE_BEGIN" in ps.custom_hooks["pre_dispatch"]


# ---- DeploymentTopology ----


class TestDeploymentTopology:
    def test_all_values(self) -> None:
        assert len(DeploymentTopology) == 4
        assert DeploymentTopology.SINGLE_DEVICE.value == "single_device"
        assert DeploymentTopology.MULTI_DEVICE.value == "multi_device"
        assert DeploymentTopology.MULTI_DOMAIN_SOC.value == "multi_domain_soc"
        assert DeploymentTopology.DISTRIBUTED.value == "distributed"


# ---- TopologyNode ----


class TestTopologyNode:
    def test_defaults(self) -> None:
        tn = TopologyNode(name="host")
        assert tn.name == "host"
        assert tn.device_indices == []
        assert tn.role == "worker"
        assert tn.runtime_env == "linux_userspace"
        assert tn.properties == {}

    def test_soc_node(self) -> None:
        tn = TopologyNode(
            name="npu_domain",
            device_indices=[1, 2],
            role="accelerator",
            runtime_env="zephyr_rtos",
            properties={"stack_size": 8192, "priority": 3},
        )
        assert tn.role == "accelerator"
        assert tn.runtime_env == "zephyr_rtos"
        assert tn.properties["stack_size"] == 8192


# ---- TopologyLink ----


class TestTopologyLink:
    def test_defaults(self) -> None:
        tl = TopologyLink(src_node="host", dst_node="npu")
        assert tl.transport == "local"
        assert tl.bandwidth_gbps == 0.0
        assert tl.latency_us == 0.0
        assert tl.bidirectional is True

    def test_full_construction(self) -> None:
        tl = TopologyLink(
            src_node="host",
            dst_node="npu",
            transport="zephyr_ipc",
            bandwidth_gbps=10.0,
            latency_us=0.5,
            bidirectional=False,
            properties={"queue_depth": 32, "msg_size": 128},
        )
        assert tl.transport == "zephyr_ipc"
        assert tl.properties["queue_depth"] == 32
        assert tl.bidirectional is False


# ---- TopologySpec ----


class TestTopologySpec:
    def test_defaults(self) -> None:
        ts = TopologySpec()
        assert ts.deployment == DeploymentTopology.SINGLE_DEVICE
        assert ts.nodes == []
        assert ts.links == []

    def test_heterogeneous_soc(self) -> None:
        ts = TopologySpec(
            deployment=DeploymentTopology.MULTI_DOMAIN_SOC,
            nodes=[
                TopologyNode(name="host", device_indices=[0], role="host",
                             runtime_env="linux_userspace"),
                TopologyNode(name="npu", device_indices=[1], role="accelerator",
                             runtime_env="zephyr_rtos"),
            ],
            links=[
                TopologyLink(src_node="host", dst_node="npu",
                             transport="zephyr_ipc", bandwidth_gbps=5.0),
            ],
        )
        assert ts.deployment == DeploymentTopology.MULTI_DOMAIN_SOC
        assert len(ts.nodes) == 2
        assert len(ts.links) == 1


# ---- HardwareSpec integration ----


class TestHardwareSpecNewFields:
    def test_has_profiling(self) -> None:
        hs = HardwareSpec(name="test")
        assert isinstance(hs.profiling, ProfilingSpec)
        assert hs.profiling.backends == []

    def test_has_topology(self) -> None:
        hs = HardwareSpec(name="test")
        assert isinstance(hs.topology, TopologySpec)
        assert hs.topology.deployment == DeploymentTopology.SINGLE_DEVICE

    def test_full_spec_with_profiling_and_topology(self) -> None:
        hs = HardwareSpec(
            name="test-soc",
            platform=PlatformSpec(
                vendor="acme", family="soc", chip_name="acme-soc-1",
                deployment_model="zephyr_rtos",
            ),
            profiling=ProfilingSpec(
                backends=[
                    ProfilerBackend(
                        name="zephyr_trace",
                        kind="device_trace",
                        counters=["cycles", "thread_switches"],
                        integration="embedded",
                    ),
                ],
                default_backend="zephyr_trace",
                tile_profiling=True,
            ),
            topology=TopologySpec(
                deployment=DeploymentTopology.MULTI_DOMAIN_SOC,
                nodes=[
                    TopologyNode(name="host_cpu", device_indices=[0]),
                    TopologyNode(name="npu", device_indices=[1],
                                 runtime_env="zephyr_rtos"),
                ],
                links=[
                    TopologyLink(src_node="host_cpu", dst_node="npu",
                                 transport="dma"),
                ],
            ),
        )
        assert hs.profiling.tile_profiling is True
        assert hs.topology.deployment == DeploymentTopology.MULTI_DOMAIN_SOC
        assert len(hs.topology.nodes) == 2
