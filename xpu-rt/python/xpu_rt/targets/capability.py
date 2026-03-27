"""Target capability specification and classification.

Separates "what the hardware IS" (TargetProfile) from "what it CAN DO"
(CapabilitySpec). The capability spec maps each op to its best backend
lane and classifies the target into one of four classes.

Target classes:
    TRITON_FRIENDLY  -- programmable GPU-like target; Triton covers most ops
    ACCEL_NATIVE     -- custom accelerator; needs accel dialect lowering
    UKERNEL_RUNTIME  -- firmware/NPU; driven by runtime API calls
    HYBRID           -- mixed system; different lanes for different regions

The classification step happens early (before compilation) and determines
which compiler paths CompGen scaffolds for this target.

Invariants:
    - Every op in the workload gets a capability entry.
    - Classification is deterministic given the same profile.
    - Unknown ops default to FALLBACK lane.

TODO: Implement classify_target() from TargetProfile features + kernel_backends.
TODO: Implement infer_capabilities() from profile + op list.
TODO: Support user overrides (force an op to a specific lane).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from compgen.targets.schema import TargetProfile


class TargetClass(Enum):
    """Classification of a target's primary compilation strategy."""

    TRITON_FRIENDLY = "triton_friendly"
    ACCEL_NATIVE = "accel_native"
    UKERNEL_RUNTIME = "ukernel_runtime"
    HYBRID = "hybrid"


class BackendLane(Enum):
    """Backend lane for a specific op on a specific target."""

    TRITON = "triton"
    ACCEL_DIALECT = "accel_dialect"
    UKERNEL = "ukernel"
    EXO = "exo"
    VENDOR_LIBRARY = "vendor_library"
    NATIVE_LOWERING = "native_lowering"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class OpCapability:
    """Capability entry for a single op on this target.

    Attributes:
        op_name: Operation name (e.g., "matmul", "conv2d", "layernorm").
        preferred_lane: Best backend lane for this op.
        alternative_lanes: Other viable lanes (fallback order).
        supported_dtypes: Dtypes this op supports on this target.
        estimated_speedup: Estimated speedup vs fallback (if known).
    """

    op_name: str
    preferred_lane: BackendLane = BackendLane.FALLBACK
    alternative_lanes: list[BackendLane] = field(default_factory=list)
    supported_dtypes: set[str] = field(default_factory=lambda: {"float32"})
    estimated_speedup: float | None = None


@dataclass(frozen=True)
class CapabilitySpec:
    """Complete capability specification for a target.

    Attributes:
        target_class: Overall target classification.
        op_capabilities: Per-op capability entries.
        default_lane: Default lane for unrecognized ops.
        notes: Human-readable notes about the classification.
    """

    target_class: TargetClass
    op_capabilities: dict[str, OpCapability] = field(default_factory=dict)
    default_lane: BackendLane = BackendLane.FALLBACK
    notes: str = ""


def classify_target(profile: TargetProfile) -> TargetClass:
    """Classify a target into one of the four target classes.

    Classification logic:
        - If kernel_backends includes "triton" -> TRITON_FRIENDLY
        - If device_type is "accelerator" with custom features -> ACCEL_NATIVE
        - If device_type is "npu" or features include firmware-driven -> UKERNEL_RUNTIME
        - If multiple devices with different types -> HYBRID
        - Default: TRITON_FRIENDLY

    """
    # Check for explicit override in metadata
    override = profile.metadata.get("target_class")
    if override:
        try:
            return TargetClass(override)
        except ValueError:
            pass

    if not profile.devices:
        return TargetClass.TRITON_FRIENDLY

    # Multiple device types -> HYBRID
    device_types = {d.device_type for d in profile.devices}
    if len(device_types) > 1:
        return TargetClass.HYBRID

    # Single device type
    dtype = next(iter(device_types))
    all_backends = set()
    for d in profile.devices:
        all_backends.update(d.kernel_backends)

    if dtype == "npu":
        return TargetClass.UKERNEL_RUNTIME
    if dtype == "accelerator":
        return TargetClass.ACCEL_NATIVE
    if "triton" in all_backends:
        return TargetClass.TRITON_FRIENDLY

    return TargetClass.TRITON_FRIENDLY


def infer_capabilities(
    profile: TargetProfile,
    op_list: list[str] | None = None,
) -> CapabilitySpec:
    """Infer capability spec from a target profile."""
    target_class = classify_target(profile)

    # Collect all supported ops from devices if no explicit list
    if op_list is None:
        ops: set[str] = set()
        for d in profile.devices:
            ops.update(d.supported_ops)
        op_list = sorted(ops)

    # Collect all supported dtypes across devices
    all_dtypes: set[str] = set()
    for d in profile.devices:
        for cu in d.compute_units:
            all_dtypes.update(cu.supported_dtypes)
    if not all_dtypes:
        all_dtypes = {"float32"}

    # Determine default lane from target class
    default_lane_map = {
        TargetClass.TRITON_FRIENDLY: BackendLane.TRITON,
        TargetClass.ACCEL_NATIVE: BackendLane.ACCEL_DIALECT,
        TargetClass.UKERNEL_RUNTIME: BackendLane.UKERNEL,
        TargetClass.HYBRID: BackendLane.TRITON,
    }
    primary_lane = default_lane_map.get(target_class, BackendLane.FALLBACK)

    # Build per-op capabilities
    op_capabilities: dict[str, OpCapability] = {}
    for op_name in op_list:
        op_capabilities[op_name] = OpCapability(
            op_name=op_name,
            preferred_lane=primary_lane,
            alternative_lanes=[BackendLane.FALLBACK],
            supported_dtypes=all_dtypes,
        )

    return CapabilitySpec(
        target_class=target_class,
        op_capabilities=op_capabilities,
        default_lane=BackendLane.FALLBACK,
    )


__all__ = [
    "BackendLane",
    "CapabilitySpec",
    "OpCapability",
    "TargetClass",
    "classify_target",
    "infer_capabilities",
]
