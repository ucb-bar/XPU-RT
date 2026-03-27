"""Target profile schema definitions.

Defines the ``TargetProfile`` data model and its nested structures.
Profiles are loaded from YAML files and validated against a JSON schema.

The schema is designed to support:
- Single-device profiles (one GPU, one accelerator)
- Multi-device heterogeneous profiles (CPU + GPU, multi-GPU, mixed accelerators)
- Interconnect topology (NVLink, PCIe, network)
- Cost model data (from docs or calibration)

Invariants:
    - All fields have explicit types and defaults.
    - Profiles are serializable to/from YAML without loss.
    - The schema is versioned (schema_version field).

TODO: Implement load_profile() from YAML with validation.
TODO: Implement to_yaml() serialization.
TODO: Add JSON schema generation for external validation tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryLevel:
    """A level in the memory hierarchy.

    Attributes:
        name: Level name (e.g., "registers", "shared_memory", "l2_cache", "hbm").
        size_bytes: Size in bytes.
        bandwidth_gbps: Bandwidth in GB/s (if measurable).
        latency_ns: Access latency in nanoseconds (if known).
    """

    name: str
    size_bytes: int
    bandwidth_gbps: float | None = None
    latency_ns: float | None = None


@dataclass(frozen=True)
class ComputeUnit:
    """A compute unit on a device.

    Attributes:
        name: Unit name (e.g., "tensor_core", "cuda_core", "vector_unit").
        count: Number of units.
        supported_dtypes: Supported data types (e.g., {"fp16", "bf16", "tf32"}).
        peak_tflops: Peak throughput in TFLOPS (if known).
    """

    name: str
    count: int
    supported_dtypes: set[str] = field(default_factory=lambda: {"float32"})
    peak_tflops: float | None = None


@dataclass(frozen=True)
class DeviceSpec:
    """Specification for a single device.

    Attributes:
        device_type: "cpu", "gpu", "accelerator", "npu", etc.
        name: Device name (e.g., "A100-SXM4-80GB").
        vendor: Device vendor (e.g., "nvidia", "amd", "aws").
        compute_units: List of compute units.
        memory_hierarchy: Memory levels from fastest to slowest.
        supported_ops: List of natively supported op names.
        features: Hardware features (e.g., "sparsity_2_4", "tf32").
        kernel_backends: Supported kernel backends (e.g., ["triton", "cutlass"]).
    """

    device_type: str
    name: str
    vendor: str = ""
    compute_units: list[ComputeUnit] = field(default_factory=list)
    memory_hierarchy: list[MemoryLevel] = field(default_factory=list)
    supported_ops: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    kernel_backends: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Interconnect:
    """Interconnect between devices.

    Attributes:
        topology: "nvlink", "pcie", "network", "shared_memory", etc.
        bandwidth_gbps: Bandwidth in GB/s.
        latency_us: Latency in microseconds.
        devices: Pair of device indices connected by this interconnect.
    """

    topology: str
    bandwidth_gbps: float
    latency_us: float | None = None
    devices: tuple[int, int] = (0, 1)


@dataclass(frozen=True)
class TargetProfile:
    """Complete target profile for a deployment target.

    Attributes:
        name: Profile name (e.g., "cuda-a100", "trainium1-x4").
        schema_version: Schema version string.
        devices: List of device specifications.
        interconnects: Interconnects between devices.
        constraints: System-level constraints.
        cost_model: Op-level cost data (latencies, bandwidth).
        calibration_data: Measured calibration data (from hardware).
        metadata: Additional profile metadata.
    """

    name: str
    schema_version: str = "1.0"
    devices: list[DeviceSpec] = field(default_factory=list)
    interconnects: list[Interconnect] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)
    calibration_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_memory_level(data: dict[str, Any]) -> MemoryLevel:
    return MemoryLevel(
        name=data["name"],
        size_bytes=int(data["size_bytes"]),
        bandwidth_gbps=data.get("bandwidth_gbps"),
        latency_ns=data.get("latency_ns"),
    )


def _build_compute_unit(data: dict[str, Any]) -> ComputeUnit:
    dtypes = data.get("supported_dtypes", ["float32"])
    return ComputeUnit(
        name=data["name"],
        count=int(data["count"]),
        supported_dtypes=set(dtypes) if isinstance(dtypes, list) else {dtypes},
        peak_tflops=data.get("peak_tflops"),
    )


def _build_device_spec(data: dict[str, Any]) -> DeviceSpec:
    return DeviceSpec(
        device_type=data["device_type"],
        name=data["name"],
        vendor=data.get("vendor", ""),
        compute_units=[_build_compute_unit(cu) for cu in data.get("compute_units", [])],
        memory_hierarchy=[_build_memory_level(ml) for ml in data.get("memory_hierarchy", [])],
        supported_ops=data.get("supported_ops", []),
        features=data.get("features", []),
        kernel_backends=data.get("kernel_backends", []),
    )


def _build_interconnect(data: dict[str, Any]) -> Interconnect:
    devices_raw = data.get("devices", [0, 1])
    return Interconnect(
        topology=data["topology"],
        bandwidth_gbps=float(data["bandwidth_gbps"]),
        latency_us=data.get("latency_us"),
        devices=(int(devices_raw[0]), int(devices_raw[1])),
    )


def load_profile(path: str | Path) -> TargetProfile:
    """Load a target profile from a YAML file.

    Args:
        path: Path to the target_profile.yaml file.

    Returns:
        A TargetProfile instance.
    """
    import yaml

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile YAML must be a mapping, got {type(data).__name__}")

    return TargetProfile(
        name=data["name"],
        schema_version=data.get("schema_version", "1.0"),
        devices=[_build_device_spec(d) for d in data.get("devices", [])],
        interconnects=[_build_interconnect(ic) for ic in data.get("interconnects", [])],
        constraints=data.get("constraints", {}),
        cost_model=data.get("cost_model", {}),
        calibration_data=data.get("calibration_data", {}),
        metadata=data.get("metadata", {}),
    )


__all__ = [
    "ComputeUnit",
    "DeviceSpec",
    "Interconnect",
    "MemoryLevel",
    "TargetProfile",
    "load_profile",
]
