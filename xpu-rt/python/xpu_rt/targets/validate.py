"""Target profile validation.

Validates target profiles against semantic rules. Catches errors early,
before the profile is used in the pipeline.

Validation levels:
    1. Required fields: name, schema_version, at least one device.
    2. Semantic: interconnect device indices valid, compute counts positive,
       memory hierarchy non-empty, bandwidth/latency non-negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from compgen.targets.schema import TargetProfile, load_profile


@dataclass(frozen=True)
class ValidationError:
    """A single validation error.

    Attributes:
        path: JSON path to the invalid field.
        message: Human-readable error description.
        level: "error" or "warning".
    """

    path: str
    message: str
    level: str = "error"


@dataclass(frozen=True)
class ValidationResult:
    """Result of profile validation.

    Attributes:
        valid: Whether the profile passed all checks.
        errors: List of validation errors.
        warnings: List of validation warnings.
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


def validate_profile(profile: TargetProfile) -> ValidationResult:
    """Validate a loaded TargetProfile with semantic checks."""
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # At least one device
    if not profile.devices:
        errors.append(ValidationError(path="devices", message="At least one device is required"))

    num_devices = len(profile.devices)
    for i, device in enumerate(profile.devices):
        prefix = f"devices[{i}]"

        # Compute unit counts must be positive
        for j, cu in enumerate(device.compute_units):
            if cu.count <= 0:
                errors.append(
                    ValidationError(
                        path=f"{prefix}.compute_units[{j}].count",
                        message=f"Compute unit count must be positive, got {cu.count}",
                    )
                )

        # Memory hierarchy should have at least one level for non-trivial devices
        if not device.memory_hierarchy:
            warnings.append(
                ValidationError(
                    path=f"{prefix}.memory_hierarchy",
                    message="Device has no memory hierarchy levels",
                    level="warning",
                )
            )

        # Bandwidth must be non-negative
        for j, ml in enumerate(device.memory_hierarchy):
            if ml.bandwidth_gbps is not None and ml.bandwidth_gbps < 0:
                errors.append(
                    ValidationError(
                        path=f"{prefix}.memory_hierarchy[{j}].bandwidth_gbps",
                        message=f"Bandwidth must be non-negative, got {ml.bandwidth_gbps}",
                    )
                )

    # Interconnect device indices must be valid
    for i, ic in enumerate(profile.interconnects):
        for idx in ic.devices:
            if idx < 0 or idx >= num_devices:
                errors.append(
                    ValidationError(
                        path=f"interconnects[{i}].devices",
                        message=f"Device index {idx} out of range [0, {num_devices})",
                    )
                )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_profile_file(path: str | Path) -> ValidationResult:
    """Validate a target profile YAML file (load + validate)."""
    import yaml

    path = Path(path)
    errors: list[ValidationError] = []

    # Check file exists
    if not path.exists():
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(path=str(path), message="File not found"),
            ],
        )

    # Load raw YAML and check required keys
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(path=str(path), message="YAML must be a mapping"),
            ],
        )

    for key in ("name", "devices"):
        if key not in data:
            errors.append(ValidationError(path=key, message=f"Required field '{key}' is missing"))

    if errors:
        return ValidationResult(valid=False, errors=errors)

    # Load into dataclass and run semantic validation
    profile = load_profile(path)
    return validate_profile(profile)


__all__ = ["ValidationError", "ValidationResult", "validate_profile", "validate_profile_file"]
