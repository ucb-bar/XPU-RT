"""Target-agnostic runtime topology graph.

Models any deployment topology through the same abstractions:
    - **Single host GPU**: 1 node, N devices.
    - **Heterogeneous SoC**: 1 node per domain (host CPU, NPU, DSP)
      connected by IPC/DMA links.
    - **Distributed cluster**: K nodes connected by network links.

The planner, executor, and profiling framework all operate on this
same graph — target-specific details (transport selection, thread
priorities, buffer sizes) are filled in by codegen or the agentic LLM.

Invariants:
    - Node names are unique within a topology.
    - Device indices reference the ``TargetProfile.devices`` list.
    - Links are directional but may be marked bidirectional.
    - A topology with zero nodes is valid (degenerate single-device case).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from compgen.targetgen.hardware_spec import (
    DeploymentTopology,
    TopologySpec,
)
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Runtime device descriptor
# ---------------------------------------------------------------------------


class DeviceRole(Enum):
    """Role a device plays in the runtime topology."""

    COMPUTE = "compute"
    COORDINATOR = "coordinator"
    STORAGE = "storage"
    SENSOR = "sensor"


@dataclass
class RuntimeDevice:
    """A device within a runtime node.

    Attributes:
        device_index: Index into ``TargetProfile.devices``.
        device_type: Device kind (``"cpu"``, ``"gpu"``, ``"npu"``, ...).
        name: Human-readable name.
        role: Role in the runtime (compute, coordinator, ...).
        properties: Target-specific properties (LLM-tunable).
    """

    device_index: int
    device_type: str = "cpu"
    name: str = ""
    role: DeviceRole = DeviceRole.COMPUTE
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime node
# ---------------------------------------------------------------------------


@dataclass
class RuntimeNode:
    """A compute node — a unit of independent execution.

    Can be a host process, a Zephyr RTOS domain on an SoC, or a remote
    machine in a cluster.

    Attributes:
        name: Unique node identifier.
        devices: Devices belonging to this node.
        role: Node role (``"host"``, ``"accelerator"``, ``"worker"``,
            ``"coordinator"``).
        runtime_env: Execution environment (``"linux_userspace"``,
            ``"zephyr_rtos"``, ``"bare_metal"``, ``"firmware"``).
        properties: Target-specific node properties (stack sizes,
            thread priorities, etc.).  LLM-tunable.
    """

    name: str
    devices: list[RuntimeDevice] = field(default_factory=list)
    role: str = "worker"
    runtime_env: str = "linux_userspace"
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def device_indices(self) -> list[int]:
        """Device indices owned by this node."""
        return [d.device_index for d in self.devices]


# ---------------------------------------------------------------------------
# Runtime link
# ---------------------------------------------------------------------------


@dataclass
class RuntimeLink:
    """A communication link between two runtime nodes.

    Attributes:
        src_node: Name of the source node.
        dst_node: Name of the destination node.
        transport: Transport mechanism name (``"local"``,
            ``"shared_memory"``, ``"zephyr_ipc"``, ``"dma"``,
            ``"network"``, ``"pcie"``, ``"custom"``).
        bandwidth_gbps: Link bandwidth in GB/s.
        latency_us: One-way latency in microseconds.
        bidirectional: Whether the link supports full-duplex.
        properties: Transport-specific configuration (buffer sizes,
            queue depths, etc.).  LLM-tunable.
    """

    src_node: str
    dst_node: str
    transport: str = "local"
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    bidirectional: bool = True
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runtime topology graph
# ---------------------------------------------------------------------------


@dataclass
class RuntimeTopology:
    """Target-agnostic topology graph.

    This is the scaffold — it works for any deployment model.  The LLM
    fills in transport selection, buffer sizes, thread priorities, and
    other target-specific parameters via ``RuntimeLink.properties`` and
    ``RuntimeNode.properties``.

    Attributes:
        deployment: The deployment topology class.
        nodes: Compute nodes in the system.
        links: Communication links between nodes.
    """

    deployment: DeploymentTopology = DeploymentTopology.SINGLE_DEVICE
    nodes: list[RuntimeNode] = field(default_factory=list)
    links: list[RuntimeLink] = field(default_factory=list)

    # -- Lookup helpers ----------------------------------------------------

    def get_node(self, name: str) -> RuntimeNode | None:
        """Look up a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_node_for_device(self, device_index: int) -> RuntimeNode | None:
        """Find which node owns a given device index."""
        for node in self.nodes:
            if device_index in node.device_indices:
                return node
        return None

    def get_links_from(self, node_name: str) -> list[RuntimeLink]:
        """Get all outbound links from a node."""
        result = [lk for lk in self.links if lk.src_node == node_name]
        # Include reverse direction for bidirectional links
        result.extend(lk for lk in self.links if lk.dst_node == node_name and lk.bidirectional)
        return result

    def get_link_between(self, src: str, dst: str) -> RuntimeLink | None:
        """Find the link between two nodes (checking bidirectional)."""
        for lk in self.links:
            if lk.src_node == src and lk.dst_node == dst:
                return lk
            if lk.bidirectional and lk.src_node == dst and lk.dst_node == src:
                return lk
        return None

    @property
    def all_device_indices(self) -> list[int]:
        """All device indices across all nodes."""
        indices: list[int] = []
        for node in self.nodes:
            indices.extend(node.device_indices)
        return indices

    @property
    def is_distributed(self) -> bool:
        """Whether this topology spans multiple independent nodes."""
        return self.deployment in (
            DeploymentTopology.DISTRIBUTED,
            DeploymentTopology.MULTI_DOMAIN_SOC,
        )

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the topology has nodes with different runtime environments."""
        envs = {node.runtime_env for node in self.nodes}
        return len(envs) > 1

    def validate(self) -> list[str]:
        """Validate topology consistency.

        Returns:
            List of error messages (empty = valid).
        """
        errors: list[str] = []
        node_names = {n.name for n in self.nodes}

        # Check for duplicate node names
        if len(node_names) != len(self.nodes):
            seen: set[str] = set()
            for n in self.nodes:
                if n.name in seen:
                    errors.append(f"duplicate node name: {n.name!r}")
                seen.add(n.name)

        # Check link endpoints reference valid nodes
        for lk in self.links:
            if lk.src_node not in node_names:
                errors.append(f"link src_node {lk.src_node!r} not in topology")
            if lk.dst_node not in node_names:
                errors.append(f"link dst_node {lk.dst_node!r} not in topology")

        # Check device indices are unique across nodes
        all_devs: list[int] = []
        for node in self.nodes:
            for dev in node.devices:
                if dev.device_index in all_devs:
                    errors.append(f"device_index {dev.device_index} claimed by multiple nodes")
                all_devs.append(dev.device_index)

        return errors

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for LLM prompts."""
        return {
            "deployment": self.deployment.value,
            "num_nodes": len(self.nodes),
            "num_links": len(self.links),
            "num_devices": len(self.all_device_indices),
            "is_distributed": self.is_distributed,
            "is_heterogeneous": self.is_heterogeneous,
            "nodes": [
                {
                    "name": n.name,
                    "role": n.role,
                    "runtime_env": n.runtime_env,
                    "num_devices": len(n.devices),
                }
                for n in self.nodes
            ],
            "links": [
                {
                    "src": lk.src_node,
                    "dst": lk.dst_node,
                    "transport": lk.transport,
                    "bandwidth_gbps": lk.bandwidth_gbps,
                }
                for lk in self.links
            ],
        }


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def topology_from_spec(spec: TopologySpec, profile: TargetProfile) -> RuntimeTopology:
    """Build a ``RuntimeTopology`` from a ``TopologySpec`` and ``TargetProfile``.

    Args:
        spec: The topology specification from the hardware spec.
        profile: The target profile (for device info).

    Returns:
        A fully populated ``RuntimeTopology``.
    """
    nodes = [
        RuntimeNode(
            name=tn.name,
            devices=[
                RuntimeDevice(
                    device_index=di,
                    device_type=profile.devices[di].device_type if di < len(profile.devices) else "unknown",
                    name=profile.devices[di].name if di < len(profile.devices) else f"device_{di}",
                )
                for di in tn.device_indices
            ],
            role=tn.role,
            runtime_env=tn.runtime_env,
            properties=dict(tn.properties),
        )
        for tn in spec.nodes
    ]

    links = [
        RuntimeLink(
            src_node=tl.src_node,
            dst_node=tl.dst_node,
            transport=tl.transport,
            bandwidth_gbps=tl.bandwidth_gbps,
            latency_us=tl.latency_us,
            bidirectional=tl.bidirectional,
            properties=dict(tl.properties),
        )
        for tl in spec.links
    ]

    topo = RuntimeTopology(
        deployment=spec.deployment,
        nodes=nodes,
        links=links,
    )

    errors = topo.validate()
    if errors:
        log.warning("topology.validation_errors", errors=errors)

    return topo


def infer_topology(profile: TargetProfile) -> RuntimeTopology:
    """Infer a default topology from a ``TargetProfile``.

    When no explicit ``TopologySpec`` is provided, infer from the profile:
        - 0-1 devices → single_device, one node.
        - 2+ devices, same type → multi_device, one node.
        - 2+ devices, mixed types → multi_device, one node per type group.

    Args:
        profile: The target profile.

    Returns:
        A ``RuntimeTopology`` inferred from device layout.
    """
    if len(profile.devices) <= 1:
        devices = (
            [
                RuntimeDevice(
                    device_index=0,
                    device_type=profile.devices[0].device_type if profile.devices else "cpu",
                    name=profile.devices[0].name if profile.devices else "cpu0",
                )
            ]
            if profile.devices
            else []
        )

        return RuntimeTopology(
            deployment=DeploymentTopology.SINGLE_DEVICE,
            nodes=[RuntimeNode(name="host", devices=devices, role="host")],
            links=[],
        )

    # Group devices by type
    type_groups: dict[str, list[int]] = {}
    for i, dev in enumerate(profile.devices):
        type_groups.setdefault(dev.device_type, []).append(i)

    if len(type_groups) == 1:
        # Homogeneous multi-device (e.g., 4x A100)
        devices = [
            RuntimeDevice(
                device_index=i,
                device_type=profile.devices[i].device_type,
                name=profile.devices[i].name,
            )
            for i in range(len(profile.devices))
        ]
        return RuntimeTopology(
            deployment=DeploymentTopology.MULTI_DEVICE,
            nodes=[RuntimeNode(name="host", devices=devices, role="host")],
            links=[],
        )

    # Heterogeneous — one node per device type group
    nodes: list[RuntimeNode] = []
    for dev_type, indices in type_groups.items():
        devices = [
            RuntimeDevice(
                device_index=i,
                device_type=dev_type,
                name=profile.devices[i].name,
            )
            for i in indices
        ]
        role = "host" if dev_type == "cpu" else "accelerator"
        nodes.append(RuntimeNode(name=f"{dev_type}_group", devices=devices, role=role))

    # Infer links from interconnects
    links: list[RuntimeLink] = []
    node_for_dev: dict[int, str] = {}
    for node in nodes:
        for dev in node.devices:
            node_for_dev[dev.device_index] = node.name

    for ic in profile.interconnects:
        src_node = node_for_dev.get(ic.devices[0])
        dst_node = node_for_dev.get(ic.devices[1])
        if src_node and dst_node and src_node != dst_node:
            links.append(
                RuntimeLink(
                    src_node=src_node,
                    dst_node=dst_node,
                    transport=_map_interconnect_to_transport(ic.topology),
                    bandwidth_gbps=ic.bandwidth_gbps,
                    latency_us=ic.latency_us or 0.0,
                )
            )

    return RuntimeTopology(
        deployment=DeploymentTopology.MULTI_DEVICE,
        nodes=nodes,
        links=links,
    )


def _map_interconnect_to_transport(topology: str) -> str:
    """Map a TargetProfile interconnect topology to a transport name."""
    mapping = {
        "nvlink": "pcie",
        "pcie": "pcie",
        "network": "network",
        "shared_memory": "shared_memory",
        "dma": "dma",
        "on_chip": "shared_memory",
    }
    return mapping.get(topology.lower(), "local")


__all__ = [
    "DeviceRole",
    "RuntimeDevice",
    "RuntimeLink",
    "RuntimeNode",
    "RuntimeTopology",
    "infer_topology",
    "topology_from_spec",
]
