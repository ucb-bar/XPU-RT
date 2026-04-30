"""Tests for runtime/traits.py — DeviceTraits derivation from TargetProfile."""

from __future__ import annotations

from compgen.runtime.traits import DeviceTraits
from compgen.targets.schema import ComputeUnit, DeviceSpec, MemoryLevel, TargetProfile


def _cuda_a100_profile() -> TargetProfile:
    """A minimal CUDA A100-shaped profile."""
    return TargetProfile(
        name="cuda-a100",
        devices=[
            DeviceSpec(
                device_type="gpu",
                name="A100",
                vendor="nvidia",
                compute_units=[
                    ComputeUnit(name="tensor_core", count=432, supported_dtypes={"bf16", "fp16", "tf32"}),
                    ComputeUnit(name="cuda_core", count=6912),
                ],
                memory_hierarchy=[
                    MemoryLevel(name="registers", size_bytes=128 * 1024, bandwidth_gbps=None),
                    MemoryLevel(name="shared_memory", size_bytes=164 * 1024, bandwidth_gbps=None),
                    MemoryLevel(name="l2_cache", size_bytes=40 * 1024 * 1024, bandwidth_gbps=None),
                    MemoryLevel(name="hbm", size_bytes=80 * 1024 * 1024 * 1024, bandwidth_gbps=2039.0),
                ],
                features=["tf32", "cp_async", "cooperative_launch", "graph_capture"],
                kernel_backends=["triton", "cutlass", "cublas"],
            )
        ],
    )


def _cpu_host_profile() -> TargetProfile:
    return TargetProfile(
        name="cpu-host",
        devices=[
            DeviceSpec(
                device_type="cpu",
                name="host",
                vendor="intel",
                memory_hierarchy=[MemoryLevel(name="ram", size_bytes=64 * 1024 * 1024 * 1024)],
            )
        ],
    )


def _saturn_opu_profile() -> TargetProfile:
    """Minimal RISC-V NPU (Saturn OPU) profile with bare-metal metadata."""
    return TargetProfile(
        name="saturn-opu",
        devices=[
            DeviceSpec(
                device_type="npu",
                name="saturn",
                vendor="custom",
                features=["bare_metal", "zephyr", "atomics"],
                memory_hierarchy=[MemoryLevel(name="sram", size_bytes=4 * 1024 * 1024)],
            )
        ],
        metadata={
            "is_bare_metal": True,
            "has_rtos_support": True,
            "runtime_memory_budget_bytes": 2 * 1024 * 1024,
        },
    )


# --- derivation tests -------------------------------------------------------


def test_cuda_a100_traits() -> None:
    traits = DeviceTraits.from_target_profile(_cuda_a100_profile())
    assert traits.device_class == "gpu"
    assert traits.vendor == "nvidia"
    assert traits.has_native_timeline_semaphores
    assert traits.has_global_atomics
    assert traits.has_shared_memory_atomics
    assert traits.supports_persistent_kernels
    assert traits.supports_cooperative_launch
    assert traits.supports_command_buffers
    assert traits.supports_graph_capture
    assert "hbm" in traits.memory_spaces
    assert traits.max_device_memory_bytes > 0
    assert traits.supports_host_pinned
    assert not traits.supports_peer_access  # single device
    assert traits.max_concurrent_queues == 32
    assert traits.max_workgroup_size == 1024
    assert not traits.is_bare_metal
    assert not traits.has_rtos_support
    assert traits.runtime_memory_budget_bytes == 0
    # Derived
    assert traits.supports_event_tensors
    assert traits.supports_task_grid
    # Features round-trip
    assert "tf32" in traits.features
    assert "cp_async" in traits.features


def test_cpu_host_traits() -> None:
    traits = DeviceTraits.from_target_profile(_cpu_host_profile())
    assert traits.device_class == "cpu"
    assert traits.vendor == "intel"
    # CPU: no native timeline, no command buffers, no graph capture
    assert not traits.has_native_timeline_semaphores
    assert not traits.has_shared_memory_atomics
    assert not traits.supports_command_buffers
    assert not traits.supports_graph_capture
    assert not traits.supports_host_pinned
    # But: atomics + persistent kernels -> event tensors emulable
    assert traits.has_global_atomics
    assert traits.supports_persistent_kernels
    assert traits.supports_event_tensors
    # Parallelism: queues per core
    assert traits.max_concurrent_queues >= 1


def test_saturn_opu_traits() -> None:
    traits = DeviceTraits.from_target_profile(_saturn_opu_profile())
    assert traits.device_class == "npu"
    assert traits.vendor == "custom"
    # Accel / NPU: no native timeline + no command buffers + no graph
    assert not traits.has_native_timeline_semaphores
    assert not traits.supports_command_buffers
    assert not traits.supports_graph_capture
    # But: features declare atomics + persistent_kernels are implied
    assert traits.has_global_atomics  # "atomics" in features
    assert traits.is_bare_metal
    assert traits.has_rtos_support
    assert traits.runtime_memory_budget_bytes == 2 * 1024 * 1024


def test_empty_profile_falls_back_to_cpu() -> None:
    profile = TargetProfile(name="empty")
    traits = DeviceTraits.from_target_profile(profile)
    assert traits.device_class == "cpu"
    assert traits.vendor == "unknown"
    assert traits.supports_event_tensors  # CPU-emulated path
    assert traits.supports_persistent_kernels


def test_metadata_override_wins() -> None:
    """Target authors can override any derived field via metadata."""
    profile = _cuda_a100_profile()
    # Override via metadata
    profile = TargetProfile(
        name=profile.name,
        devices=profile.devices,
        metadata={
            "device_traits_overrides": {
                "supports_graph_capture": False,  # override: this build lacks cuGraph
                "runtime_memory_budget_bytes": 64 * 1024 * 1024,
            }
        },
    )
    traits = DeviceTraits.from_target_profile(profile)
    assert not traits.supports_graph_capture  # override applied
    assert traits.runtime_memory_budget_bytes == 64 * 1024 * 1024


def test_metadata_override_rejects_unknown_fields() -> None:
    profile = _cuda_a100_profile()
    profile = TargetProfile(
        name=profile.name,
        devices=profile.devices,
        metadata={"device_traits_overrides": {"not_a_field": 42}},
    )
    traits = DeviceTraits.from_target_profile(profile)
    # Should not raise; unknown fields silently filtered.
    assert traits.device_class == "gpu"


def test_to_dict_roundtrip_shape() -> None:
    """to_dict() returns JSON-friendly primitives for trace + bundles."""
    traits = DeviceTraits.from_target_profile(_cuda_a100_profile())
    d = traits.to_dict()
    assert d["device_class"] == "gpu"
    assert d["vendor"] == "nvidia"
    assert isinstance(d["memory_spaces"], list)
    assert isinstance(d["features"], list)
    assert sorted(d["features"]) == d["features"]  # stable ordering
    # All booleans are real booleans (not truthy ints)
    for k in ("has_global_atomics", "is_bare_metal", "supports_event_tensors"):
        assert isinstance(d[k], bool)


def test_supports_event_tensors_implication() -> None:
    """supports_event_tensors == has_global_atomics AND
    supports_persistent_kernels. No other configuration should flip
    this."""
    for profile in (_cuda_a100_profile(), _cpu_host_profile(), _saturn_opu_profile()):
        traits = DeviceTraits.from_target_profile(profile)
        assert traits.supports_event_tensors == (traits.has_global_atomics and traits.supports_persistent_kernels)
