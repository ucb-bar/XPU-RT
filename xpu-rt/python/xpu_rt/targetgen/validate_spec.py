"""Strict validation of HardwareSpec for completeness and consistency."""

from __future__ import annotations

from compgen.targetgen.hardware_spec import ExecutionModel, HardwareSpec
from compgen.targets.validate import ValidationError, ValidationResult


def validate_hardware_spec(spec: HardwareSpec) -> ValidationResult:
    """Validate a HardwareSpec for completeness and cross-field consistency.

    Checks:
      - Required fields non-empty
      - Execution model is valid
      - At least one native op family
      - Cross-field consistency (geometry matches execution model)
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Required fields
    if not spec.name:
        errors.append(ValidationError("name", "name is required"))
    if not spec.platform.vendor:
        errors.append(ValidationError("platform.vendor", "vendor is required"))
    if not spec.platform.chip_name:
        errors.append(ValidationError("platform.chip_name", "chip_name is required"))
    if not spec.isa.base_isa or spec.isa.base_isa == "unknown":
        warnings.append(ValidationError("isa.base_isa", "base_isa should be specified", level="warning"))

    # Execution model consistency
    model = spec.execution_model.model

    if model in (ExecutionModel.ROCC_COPROCESSOR, ExecutionModel.DECOUPLED_MATRIX):
        if not spec.engine_geometry.systolic_array_dim and not spec.engine_geometry.tiles:
            warnings.append(ValidationError(
                "engine_geometry",
                f"{model.value} targets should specify systolic_array_dim or tiles",
                level="warning",
            ))

    if model == ExecutionModel.SIMD_VECTOR:
        if spec.engine_geometry.vector_length_bits == 0:
            warnings.append(ValidationError(
                "engine_geometry.vector_length_bits",
                "SIMD_VECTOR targets should specify vector_length_bits",
                level="warning",
            ))

    if model == ExecutionModel.SIMT_GPU:
        if spec.engine_geometry.max_warp_size == 0:
            warnings.append(ValidationError(
                "engine_geometry.max_warp_size",
                "SIMT_GPU targets should specify max_warp_size",
                level="warning",
            ))

    # Native ops check
    if not spec.native_ops.families:
        warnings.append(ValidationError(
            "native_ops.families",
            "No native operation families specified",
            level="warning",
        ))

    # Memory model check
    if not spec.memory_model.address_spaces:
        warnings.append(ValidationError(
            "memory_model.address_spaces",
            "No address spaces specified",
            level="warning",
        ))

    # Numeric contract check
    if not spec.numeric_contract.supported_dtypes:
        warnings.append(ValidationError(
            "numeric_contract.supported_dtypes",
            "No supported dtypes specified",
            level="warning",
        ))

    # Positive geometry dimensions
    for tile in spec.engine_geometry.tiles:
        if any(d <= 0 for d in tile.dimensions):
            errors.append(ValidationError(
                f"engine_geometry.tiles.{tile.name}",
                f"Tile dimensions must be positive, got {tile.dimensions}",
            ))

    for dim in spec.engine_geometry.systolic_array_dim:
        if dim <= 0:
            errors.append(ValidationError(
                "engine_geometry.systolic_array_dim",
                f"Systolic array dimensions must be positive, got {dim}",
            ))

    # Address space sizes
    for space in spec.memory_model.address_spaces:
        if space.size_bytes < 0:
            errors.append(ValidationError(
                f"memory_model.address_spaces.{space.name}",
                f"Address space size must be non-negative, got {space.size_bytes}",
            ))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
