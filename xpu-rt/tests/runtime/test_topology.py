"""Tests for runtime/topology.py -- target-agnostic topology graph."""

from __future__ import annotations

import pytest

from compgen.runtime.topology import (
    DeviceRole,
    RuntimeDevice,
    RuntimeLink,
    RuntimeNode,
    RuntimeTopology,
    infer_topology,
    topology_from_spec,
)
from compgen.targetgen.hardware_spec import (
    DeploymentTopology,
    TopologyLink,
    TopologyNode,
    TopologySpec,
)
from compgen.targets.schema import DeviceSpec, Interconnect, TargetProfile


# ---- RuntimeDevice ----


class TestRuntimeDevice:
    def test_defaults(self) -> None:
        d = RuntimeDevice(device_index=0)
        assert d.device_type == "cpu"
        assert d.role == DeviceRole.COMPUTE

    def test_custom(self) -> None:
        d = RuntimeDevice(
            device_index=1,
            device_type="npu",
            name="acme-npu-v2",
            role=DeviceRole.COMPUTE,
            properties={"max_batch": 16},
        )
        assert d.device_type == "npu"
        assert d.properties["max_batch"] == 16


# ---- RuntimeNode ----


class TestRuntimeNode:
    def test_device_indices(self) -> None:
        node = RuntimeNode(
            name="host",
            devices=[
                RuntimeDevice(device_index=0),
                RuntimeDevice(device_index=1),
            ],
        )
        assert node.device_indices == [0, 1]

    def test_empty_node(self) -> None:
        node = RuntimeNode(name="empty")
        assert node.device_indices == []


# ---- RuntimeTopology ----


def _make_simple_topology() -> RuntimeTopology:
    """Single-host with one GPU."""
    return RuntimeTopology(
        deployment=DeploymentTopology.SINGLE_DEVICE,
        nodes=[
            RuntimeNode(
                name="host",
                devices=[RuntimeDevice(device_index=0, device_type="gpu", name="A100")],
                role="host",
            ),
        ],
        links=[],
    )


def _make_soc_topology() -> RuntimeTopology:
    """Heterogeneous SoC: host CPU + NPU + DSP."""
    return RuntimeTopology(
        deployment=DeploymentTopology.MULTI_DOMAIN_SOC,
        nodes=[
            RuntimeNode(
                name="host_cpu",
                devices=[RuntimeDevice(device_index=0, device_type="cpu")],
                role="host",
                runtime_env="linux_userspace",
            ),
            RuntimeNode(
                name="npu_domain",
                devices=[RuntimeDevice(device_index=1, device_type="npu")],
                role="accelerator",
                runtime_env="zephyr_rtos",
            ),
            RuntimeNode(
                name="dsp_domain",
                devices=[RuntimeDevice(device_index=2, device_type="dsp")],
                role="accelerator",
                runtime_env="bare_metal",
            ),
        ],
        links=[
            RuntimeLink(src_node="host_cpu", dst_node="npu_domain",
                        transport="zephyr_ipc", bandwidth_gbps=5.0, latency_us=0.5),
            RuntimeLink(src_node="host_cpu", dst_node="dsp_domain",
                        transport="dma", bandwidth_gbps=2.0, latency_us=1.0),
        ],
    )


class TestRuntimeTopology:
    def test_get_node(self) -> None:
        topo = _make_soc_topology()
        node = topo.get_node("npu_domain")
        assert node is not None
        assert node.runtime_env == "zephyr_rtos"
        assert topo.get_node("nonexistent") is None

    def test_get_node_for_device(self) -> None:
        topo = _make_soc_topology()
        node = topo.get_node_for_device(1)
        assert node is not None
        assert node.name == "npu_domain"
        assert topo.get_node_for_device(99) is None

    def test_get_links_from(self) -> None:
        topo = _make_soc_topology()
        links = topo.get_links_from("host_cpu")
        assert len(links) == 2

    def test_get_links_from_bidirectional(self) -> None:
        topo = _make_soc_topology()
        # npu_domain has no outbound links, but host_cpu→npu is bidirectional
        links = topo.get_links_from("npu_domain")
        assert len(links) == 1
        assert links[0].src_node == "host_cpu"

    def test_get_link_between(self) -> None:
        topo = _make_soc_topology()
        link = topo.get_link_between("host_cpu", "npu_domain")
        assert link is not None
        assert link.transport == "zephyr_ipc"

        # Bidirectional: reverse lookup works
        link_rev = topo.get_link_between("npu_domain", "host_cpu")
        assert link_rev is not None

    def test_all_device_indices(self) -> None:
        topo = _make_soc_topology()
        assert sorted(topo.all_device_indices) == [0, 1, 2]

    def test_is_distributed(self) -> None:
        assert _make_simple_topology().is_distributed is False
        assert _make_soc_topology().is_distributed is True

    def test_is_heterogeneous(self) -> None:
        assert _make_simple_topology().is_heterogeneous is False
        assert _make_soc_topology().is_heterogeneous is True

    def test_validate_valid(self) -> None:
        errors = _make_soc_topology().validate()
        assert errors == []

    def test_validate_duplicate_node(self) -> None:
        topo = RuntimeTopology(
            nodes=[
                RuntimeNode(name="host"),
                RuntimeNode(name="host"),
            ],
        )
        errors = topo.validate()
        assert any("duplicate" in e for e in errors)

    def test_validate_bad_link(self) -> None:
        topo = RuntimeTopology(
            nodes=[RuntimeNode(name="host")],
            links=[RuntimeLink(src_node="host", dst_node="nonexistent")],
        )
        errors = topo.validate()
        assert any("nonexistent" in e for e in errors)

    def test_validate_duplicate_device(self) -> None:
        topo = RuntimeTopology(
            nodes=[
                RuntimeNode(name="a", devices=[RuntimeDevice(device_index=0)]),
                RuntimeNode(name="b", devices=[RuntimeDevice(device_index=0)]),
            ],
        )
        errors = topo.validate()
        assert any("device_index 0" in e for e in errors)

    def test_summary(self) -> None:
        topo = _make_soc_topology()
        s = topo.summary()
        assert s["deployment"] == "multi_domain_soc"
        assert s["num_nodes"] == 3
        assert s["num_links"] == 2
        assert s["is_distributed"] is True
        assert s["is_heterogeneous"] is True


# ---- topology_from_spec ----


class TestTopologyFromSpec:
    def test_simple_spec(self) -> None:
        spec = TopologySpec(
            deployment=DeploymentTopology.SINGLE_DEVICE,
            nodes=[TopologyNode(name="host", device_indices=[0])],
        )
        profile = TargetProfile(
            name="test",
            devices=[DeviceSpec(device_type="gpu", name="TestGPU")],
        )
        topo = topology_from_spec(spec, profile)
        assert len(topo.nodes) == 1
        assert topo.nodes[0].devices[0].device_type == "gpu"
        assert topo.nodes[0].devices[0].name == "TestGPU"

    def test_soc_spec(self) -> None:
        spec = TopologySpec(
            deployment=DeploymentTopology.MULTI_DOMAIN_SOC,
            nodes=[
                TopologyNode(name="cpu", device_indices=[0], runtime_env="linux_userspace"),
                TopologyNode(name="npu", device_indices=[1], runtime_env="zephyr_rtos"),
            ],
            links=[
                TopologyLink(src_node="cpu", dst_node="npu", transport="zephyr_ipc"),
            ],
        )
        profile = TargetProfile(
            name="test-soc",
            devices=[
                DeviceSpec(device_type="cpu", name="HostCPU"),
                DeviceSpec(device_type="npu", name="AccelNPU"),
            ],
        )
        topo = topology_from_spec(spec, profile)
        assert len(topo.nodes) == 2
        assert topo.nodes[1].runtime_env == "zephyr_rtos"
        assert len(topo.links) == 1
        assert topo.links[0].transport == "zephyr_ipc"


# ---- infer_topology ----


class TestInferTopology:
    def test_single_device(self) -> None:
        profile = TargetProfile(
            name="single",
            devices=[DeviceSpec(device_type="gpu", name="A100")],
        )
        topo = infer_topology(profile)
        assert topo.deployment == DeploymentTopology.SINGLE_DEVICE
        assert len(topo.nodes) == 1
        assert topo.nodes[0].name == "host"

    def test_no_devices(self) -> None:
        profile = TargetProfile(name="empty")
        topo = infer_topology(profile)
        assert topo.deployment == DeploymentTopology.SINGLE_DEVICE
        assert len(topo.nodes) == 1
        assert topo.nodes[0].devices == []

    def test_homogeneous_multi_gpu(self) -> None:
        profile = TargetProfile(
            name="multi-gpu",
            devices=[
                DeviceSpec(device_type="gpu", name="A100-0"),
                DeviceSpec(device_type="gpu", name="A100-1"),
                DeviceSpec(device_type="gpu", name="A100-2"),
            ],
        )
        topo = infer_topology(profile)
        assert topo.deployment == DeploymentTopology.MULTI_DEVICE
        assert len(topo.nodes) == 1
        assert len(topo.nodes[0].devices) == 3

    def test_heterogeneous_cpu_gpu(self) -> None:
        profile = TargetProfile(
            name="cpu-gpu",
            devices=[
                DeviceSpec(device_type="cpu", name="Host"),
                DeviceSpec(device_type="gpu", name="A100"),
            ],
            interconnects=[
                Interconnect(topology="pcie", bandwidth_gbps=32.0,
                             devices=(0, 1)),
            ],
        )
        topo = infer_topology(profile)
        assert topo.deployment == DeploymentTopology.MULTI_DEVICE
        assert len(topo.nodes) == 2
        # cpu group and gpu group
        node_names = {n.name for n in topo.nodes}
        assert "cpu_group" in node_names
        assert "gpu_group" in node_names
        # Link inferred from interconnect
        assert len(topo.links) == 1
        assert topo.links[0].transport == "pcie"
