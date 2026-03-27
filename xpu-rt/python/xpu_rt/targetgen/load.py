"""YAML loading and TargetProfile extraction for HardwareSpec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from compgen.targetgen.hardware_spec import (
    AddressSpace,
    DtypeSupport,
    EngineGeometrySpec,
    ExecutionModel,
    ExecutionModelSpec,
    HardwareSpec,
    ISAExtension,
    ISASpec,
    MemoryModelSpec,
    NativeOpFamily,
    NativeOpsSpec,
    NumericContractSpec,
    PatchRequirement,
    PatchSpec,
    PlatformSpec,
    RuntimeContractSpec,
    TileGeometry,
    VerificationSurfaceSpec,
)
from compgen.targets.schema import (
    ComputeUnit,
    DeviceSpec,
    MemoryLevel,
    TargetProfile,
)


def _build_platform(d: dict[str, Any]) -> PlatformSpec:
    return PlatformSpec(**{k: v for k, v in d.items() if k in PlatformSpec.__dataclass_fields__})


def _build_execution_model(d: dict[str, Any]) -> ExecutionModelSpec:
    model = ExecutionModel(d.get("model", "simd_vector"))
    fields = {k: v for k, v in d.items() if k in ExecutionModelSpec.__dataclass_fields__ and k != "model"}
    return ExecutionModelSpec(model=model, **fields)


def _build_isa(d: dict[str, Any]) -> ISASpec:
    extensions = [ISAExtension(**ext) for ext in d.get("extensions", [])]
    fields = {k: v for k, v in d.items() if k in ISASpec.__dataclass_fields__ and k != "extensions"}
    return ISASpec(extensions=extensions, **fields)


def _build_native_ops(d: dict[str, Any]) -> NativeOpsSpec:
    families = [NativeOpFamily(**fam) for fam in d.get("families", [])]
    return NativeOpsSpec(
        families=families,
        custom_ops=d.get("custom_ops", {}),
        decomposition_rules=d.get("decomposition_rules", {}),
    )


def _build_geometry(d: dict[str, Any]) -> EngineGeometrySpec:
    tiles = [TileGeometry(**t) for t in d.get("tiles", [])]
    fields = {k: v for k, v in d.items() if k in EngineGeometrySpec.__dataclass_fields__ and k != "tiles"}
    return EngineGeometrySpec(tiles=tiles, **fields)


def _build_memory(d: dict[str, Any]) -> MemoryModelSpec:
    spaces = [AddressSpace(**s) for s in d.get("address_spaces", [])]
    fields = {k: v for k, v in d.items() if k in MemoryModelSpec.__dataclass_fields__ and k != "address_spaces"}
    return MemoryModelSpec(address_spaces=spaces, **fields)


def _build_numeric(d: dict[str, Any]) -> NumericContractSpec:
    dtypes = [DtypeSupport(**dt) for dt in d.get("supported_dtypes", [])]
    pairs = [tuple(p) for p in d.get("mixed_precision_pairs", [])]
    return NumericContractSpec(
        supported_dtypes=dtypes,
        mixed_precision_pairs=pairs,
        denormal_handling=d.get("denormal_handling", "ieee"),
        nan_handling=d.get("nan_handling", "ieee"),
        max_ulp_error=d.get("max_ulp_error", {}),
    )


def _build_patches(d: dict[str, Any]) -> PatchSpec:
    reqs = [PatchRequirement(**r) for r in d.get("requirements", [])]
    return PatchSpec(
        requirements=reqs,
        new_dialects_needed=d.get("new_dialects_needed", []),
        new_stages_needed=d.get("new_stages_needed", []),
        existing_backend_integration=d.get("existing_backend_integration", ""),
    )


def load_hardware_spec(path: str | Path) -> HardwareSpec:
    """Load a HardwareSpec from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return HardwareSpec(
        name=data["name"],
        schema_version=data.get("schema_version", "2.0"),
        platform=_build_platform(data.get("platform", {})),
        execution_model=_build_execution_model(data.get("execution_model", {})),
        isa=_build_isa(data.get("isa", {})),
        native_ops=_build_native_ops(data.get("native_ops", {})),
        engine_geometry=_build_geometry(data.get("engine_geometry", {})),
        memory_model=_build_memory(data.get("memory_model", {})),
        numeric_contract=_build_numeric(data.get("numeric_contract", {})),
        runtime_contract=RuntimeContractSpec(**data.get("runtime_contract", {})),
        verification_surface=VerificationSurfaceSpec(**data.get("verification_surface", {})),
        patches=_build_patches(data.get("patches", {})),
        constraints=data.get("constraints", {}),
        cost_model=data.get("cost_model", {}),
        metadata=data.get("metadata", {}),
    )


def extract_target_profile(spec: HardwareSpec) -> TargetProfile:
    """Extract a TargetProfile from a HardwareSpec for backward compatibility."""
    # Map execution model to device_type
    model_to_type = {
        ExecutionModel.SIMT_GPU: "gpu",
        ExecutionModel.ROCC_COPROCESSOR: "accelerator",
        ExecutionModel.TEXT_ISA_NPU: "npu",
        ExecutionModel.DATAFLOW: "accelerator",
        ExecutionModel.FIRMWARE_DRIVEN: "accelerator",
        ExecutionModel.SIMD_VECTOR: "cpu",
        ExecutionModel.DECOUPLED_MATRIX: "cpu",
    }
    device_type = model_to_type.get(spec.execution_model.model, "cpu")

    # Build compute units from geometry
    compute_units: list[ComputeUnit] = []
    if spec.engine_geometry.systolic_array_dim:
        compute_units.append(ComputeUnit(
            name="systolic_array",
            count=1,
            supported_dtypes={d.name for d in spec.numeric_contract.supported_dtypes if d.native},
        ))
    if spec.engine_geometry.vector_length_bits > 0:
        compute_units.append(ComputeUnit(
            name="vector_unit",
            count=1,
            supported_dtypes={d.name for d in spec.numeric_contract.supported_dtypes if d.native},
        ))
    if spec.engine_geometry.max_warp_size > 0:
        compute_units.append(ComputeUnit(
            name="simt_core",
            count=1,
            supported_dtypes={d.name for d in spec.numeric_contract.supported_dtypes if d.native},
        ))
    if not compute_units:
        compute_units.append(ComputeUnit(name="default", count=1))

    # Build memory hierarchy from address spaces
    memory_hierarchy = [
        MemoryLevel(name=space.name, size_bytes=space.size_bytes)
        for space in spec.memory_model.address_spaces
    ]

    # Collect supported ops from native op families
    supported_ops: list[str] = []
    for fam in spec.native_ops.families:
        supported_ops.extend(fam.ops)

    # Infer kernel backends
    kernel_backends: list[str] = []
    if spec.execution_model.model == ExecutionModel.SIMT_GPU:
        kernel_backends = ["triton"]
    elif spec.isa.compiler_intrinsics:
        kernel_backends = ["llvm"]

    device = DeviceSpec(
        device_type=device_type,
        name=spec.platform.chip_name,
        vendor=spec.platform.vendor,
        compute_units=compute_units,
        memory_hierarchy=memory_hierarchy,
        supported_ops=supported_ops,
        features=[ext.name for ext in spec.isa.extensions],
        kernel_backends=kernel_backends,
    )

    return TargetProfile(
        name=spec.name,
        schema_version=spec.schema_version,
        devices=[device],
        interconnects=[],
        constraints=spec.constraints,
        cost_model=spec.cost_model,
        metadata=spec.metadata,
    )


def load_spec_with_profile(path: str | Path) -> tuple[HardwareSpec, TargetProfile]:
    """Load both the full spec and the extracted profile."""
    spec = load_hardware_spec(path)
    profile = extract_target_profile(spec)
    return spec, profile
