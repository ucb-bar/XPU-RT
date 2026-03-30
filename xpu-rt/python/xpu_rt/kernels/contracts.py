"""Kernel contract definitions.

Kernel contracts specify what each op/subgraph needs from a kernel
implementation: shapes, dtypes, layouts, performance targets. These
contracts bridge the IR layer and the kernel generation layer.

Contracts are used to:
- Drive kernel strategy selection (native/library/autocomp/fallback)
- Provide context to the LLM for kernel generation
- Define acceptance criteria for generated kernels

Invariants:
    - Every kernel contract references a specific op or subgraph in the IR.
    - Contracts are serializable to YAML for LLM context injection.
    - Performance targets are relative to the target profile's cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.ir.payload.contracts import KernelContract, extract_contracts
from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class KernelSpec:
    """Full specification for a kernel that needs to be generated or selected.

    Extends KernelContract with generation-specific metadata.

    Attributes:
        contract: The underlying kernel contract from the IR.
        input_shapes: Concrete input shapes (from sample inputs).
        output_shapes: Concrete output shapes.
        reference_code: Reference implementation (for correctness testing).
        perf_target_us: Performance target in microseconds (from cost model).
        priority: Generation priority (higher = more important to optimize).
    """

    contract: KernelContract
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    reference_code: str = ""
    perf_target_us: float | None = None
    priority: int = 0


@dataclass(frozen=True)
class KernelSearchPlan:
    """Plan for searching/generating a kernel.

    Attributes:
        spec: The kernel specification.
        strategy: Selected strategy ("autocomp", "triton_template", "library", "native").
        search_budget: Max iterations for the search loop.
        backends: Which kernel backends to try (e.g., ["triton", "cuda"]).
        constraints: Additional constraints for the search.
    """

    spec: KernelSpec
    strategy: str
    search_budget: int = 50
    backends: list[str] = field(default_factory=lambda: ["triton"])
    constraints: dict[str, Any] = field(default_factory=dict)


def _estimate_perf_target(contract: KernelContract, target: TargetProfile) -> float | None:
    """Estimate a performance target in microseconds from the roofline model."""
    if not target.devices:
        return None

    device = target.devices[0]
    compute_tops = device.compute_tops if hasattr(device, "compute_tops") else 1.0
    bw_gbps = device.memory_bandwidth_gbps if hasattr(device, "memory_bandwidth_gbps") else 50.0

    flops = contract.cost.flops
    total_bytes = contract.cost.bytes_read + contract.cost.bytes_written

    if flops == 0 and total_bytes == 0:
        return None

    # Roofline: max(compute_time, memory_time)
    compute_us = (flops / (compute_tops * 1e12)) * 1e6 if compute_tops > 0 else 0
    memory_us = (total_bytes / (bw_gbps * 1e9)) * 1e6 if bw_gbps > 0 else 0

    return max(compute_us, memory_us) if (compute_us + memory_us) > 0 else None


def build_kernel_contracts(
    module: ModuleOp,
    target_profile: TargetProfile,
    sample_inputs: Any = None,
) -> list[KernelSpec]:
    """Build kernel specifications from IR and target profile.

    Args:
        module: Canonical xDSL module.
        target_profile: TargetProfile instance.
        sample_inputs: Optional sample inputs for shape concretization.

    Returns:
        List of KernelSpec, one per op/subgraph that needs a kernel.
    """
    contracts = extract_contracts(module)
    specs: list[KernelSpec] = []

    for i, contract in enumerate(contracts):
        perf_target = _estimate_perf_target(contract, target_profile)

        # Priority: higher FLOPs = higher priority
        priority = contract.cost.flops

        specs.append(KernelSpec(
            contract=contract,
            perf_target_us=perf_target,
            priority=priority,
        ))

    # Sort by priority (highest first)
    specs.sort(key=lambda s: s.priority, reverse=True)
    return specs


def spec_to_provider_contract(
    spec: KernelSpec,
    region_id: str,
    target: TargetProfile,
) -> Any:
    """Bridge IR-level KernelSpec to provider-level KernelContract.

    Converts the IR kernel spec into the provider protocol's contract type
    so that ``ProviderRegistry.search()`` can be called.
    """
    from compgen.kernels.provider import KernelContract as ProviderContract

    ir_contract = spec.contract
    op_name = ir_contract.op_name
    op_family = op_name.split(".")[-1] if "." in op_name else op_name

    # Map IR op_family names to template-compatible names
    _aliases = {
        "softmax": "softmax",
        "layer_norm": "layer_norm",
        "batch_norm": "layer_norm",
    }
    op_family = _aliases.get(op_family, op_family)

    input_shapes = tuple(tuple(s) for s in spec.input_shapes) if spec.input_shapes else ()
    output_shapes = tuple(tuple(s) for s in spec.output_shapes) if spec.output_shapes else ()

    # Synthesize minimal shapes when IR contracts don't carry concrete shapes
    # so that constraint evaluators (e.g. M>=1) can fire.
    if not input_shapes and ir_contract.cost.flops > 0:
        if op_family in ("matmul", "batch_matmul"):
            input_shapes = ((1, 64), (64, 64))
        elif op_family in ("conv_2d_nhwc_hwcf",):
            input_shapes = ((1, 8, 8, 3), (3, 3, 3, 16))
        else:
            input_shapes = ((1, 64),)
    dtypes = tuple(ir_contract.supported_dtypes) if ir_contract.supported_dtypes else ("float32",)

    target_name = target.name
    hardware_key = target.devices[0].name if target.devices else ""

    return ProviderContract(
        region_id=region_id,
        op_family=op_family,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        dtypes=dtypes,
        target_name=target_name,
        hardware_key=hardware_key,
        objective="latency",
    )


__all__ = ["KernelSearchPlan", "KernelSpec", "build_kernel_contracts", "spec_to_provider_contract"]
