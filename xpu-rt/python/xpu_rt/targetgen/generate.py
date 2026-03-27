"""Top-level target generator.

Given a hardware spec YAML, generates a complete target:
  1. Load + validate spec
  2. Extract TargetProfile
  3. Classify into family
  4. Generate support plan
  5. Create TargetDialectStack with family-specific stages + plugins
  6. Generate verification manifest
  7. Write all artifacts to output_dir (GITIGNORED)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from compgen.stages.registry import TargetDialectStack
from compgen.targetgen.classify import Classification, TargetFamily, classify_hardware
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targetgen.load import extract_target_profile, load_hardware_spec
from compgen.targetgen.plan import SupportPlan, generate_support_plan
from compgen.targetgen.validate_spec import validate_hardware_spec
from compgen.targetgen.verification_ladder import VerificationManifest, generate_verification_manifest
from compgen.targets.capability import CapabilitySpec, infer_capabilities
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass
class GeneratedTarget:
    """Result of target generation."""

    output_dir: Path
    spec: HardwareSpec
    profile: TargetProfile
    capabilities: CapabilitySpec
    classification: Classification
    plan: SupportPlan
    dialect_stack: TargetDialectStack
    verification_manifest: VerificationManifest


# Family → stack constructor
def _get_family_constructor(family: TargetFamily):  # noqa: ANN202
    """Import and return the stack constructor for a family."""
    if family == TargetFamily.SIMT_GPU_HAL:
        from compgen.targetgen.families.simt_gpu_hal import create_gpu_stack
        return create_gpu_stack
    if family == TargetFamily.RVV_CPU_EXTENSION:
        from compgen.targetgen.families.rvv_cpu_extension import create_rvv_cpu_stack
        return create_rvv_cpu_stack
    if family == TargetFamily.ROCC_ACCELERATOR:
        from compgen.targetgen.families.rocc_accelerator import create_rocc_stack
        return create_rocc_stack
    if family == TargetFamily.RISCV_VENDOR_MATRIX:
        from compgen.targetgen.families.riscv_vendor_matrix import create_vendor_matrix_stack
        return create_vendor_matrix_stack
    if family == TargetFamily.STRUCTURED_NPU_TEXT_ISA:
        from compgen.targetgen.families.structured_npu import create_npu_stack
        return create_npu_stack
    msg = f"No stack constructor for family {family}"
    raise ValueError(msg)


def generate_target(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    emit_hal_driver: bool = False,
) -> GeneratedTarget:
    """Generate a complete target from a hardware spec YAML.

    Args:
        spec_path: Path to the hardware spec YAML.
        output_dir: Directory for generated artifacts (NOT in-tree, GITIGNORED).
        emit_hal_driver: When True, also generate C HAL driver source files
            into ``<output_dir>/hal/``.

    All output goes to output_dir (NOT in-tree, GITIGNORED).
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    spec = load_hardware_spec(spec_path)
    validation = validate_hardware_spec(spec)
    if not validation.valid:
        error_msgs = [e.message for e in validation.errors]
        raise ValueError(f"Hardware spec validation failed: {error_msgs}")

    log.info("targetgen.loaded", name=spec.name)

    # 2. Extract TargetProfile
    profile = extract_target_profile(spec)

    # 3. Classify
    classification = classify_hardware(spec)
    log.info(
        "targetgen.classified",
        family=classification.family.value,
        confidence=classification.confidence,
    )

    # 4. Generate support plan
    plan = generate_support_plan(spec, classification)
    log.info("targetgen.planned", stages=len(plan.required_stages), backend=plan.kernel_backend)

    # 5. Create dialect stack
    constructor = _get_family_constructor(classification.family)
    dialect_stack = constructor(spec, output_dir=str(output / "bundle"))

    # 6. Generate verification manifest
    capabilities = infer_capabilities(profile)
    manifest = generate_verification_manifest(spec, classification, plan)

    # 7. Write artifacts
    _write_artifacts(output, spec, profile, classification, plan, manifest)

    # 8. Optionally generate C HAL driver
    if emit_hal_driver:
        from compgen.targetgen.hal_codegen import generate_hal_driver

        hal_dir = output / "hal"
        hal_files = generate_hal_driver(spec, hal_dir)
        log.info("targetgen.hal_generated", files=[str(f) for f in hal_files])

    log.info("targetgen.complete", output_dir=str(output), tests=len(manifest.tests))

    return GeneratedTarget(
        output_dir=output,
        spec=spec,
        profile=profile,
        capabilities=capabilities,
        classification=classification,
        plan=plan,
        dialect_stack=dialect_stack,
        verification_manifest=manifest,
    )


def _write_artifacts(
    output: Path,
    spec: HardwareSpec,
    profile: TargetProfile,
    classification: Classification,
    plan: SupportPlan,
    manifest: VerificationManifest,
) -> None:
    """Write all generated artifacts to the output directory."""
    # Classification
    (output / "classification.json").write_text(json.dumps({
        "family": classification.family.value,
        "target_class": classification.target_class.value,
        "integration_style": classification.integration_style.value,
        "lowering_surface": classification.lowering_surface.value,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning,
    }, indent=2))

    # Support plan
    (output / "support_plan.json").write_text(json.dumps({
        "target_name": plan.target_name,
        "family": plan.family.value,
        "kernel_backend": plan.kernel_backend,
        "needs_accel_dialect": plan.needs_accel_dialect,
        "needs_ukernel_dialect": plan.needs_ukernel_dialect,
        "llvm_patches_needed": plan.llvm_patches_needed,
        "estimated_effort": plan.estimated_effort,
        "required_stages": [
            {"name": s.stage_name, "class": s.stage_class, "needs_plugin": s.needs_plugin}
            for s in plan.required_stages
        ],
        "required_dialects": plan.required_dialects,
        "runtime_template": plan.runtime_template,
        "threading_model": plan.threading_model,
        "memory_strategy": plan.memory_strategy,
    }, indent=2))

    # Verification manifest
    (output / "verification_manifest.json").write_text(json.dumps({
        "target_name": manifest.target_name,
        "highest_achievable": manifest.highest_achievable.name,
        "maturity": manifest.maturity.name,
        "test_count": len(manifest.tests),
        "tests": [
            {"level": t.level.name, "name": t.name, "description": t.description,
             "requires_hardware": t.requires_hardware}
            for t in manifest.tests
        ],
    }, indent=2))
