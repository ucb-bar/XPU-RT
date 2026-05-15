"""Support plan generation — maps classification to required stages and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.targetgen.classify import Classification, TargetFamily
from compgen.targetgen.hardware_spec import HardwareSpec


@dataclass(frozen=True)
class StageRequirement:
    """A required stage in the compilation pipeline.

    Attributes:
        stage_name: Stage identifier.
        stage_class: Which CompilationStage class to use.
        needs_plugin: Whether a target-specific plugin is needed.
        plugin_complexity: Effort estimate for the plugin.
        description: What this stage does for this target.
    """

    stage_name: str
    stage_class: str
    needs_plugin: bool = True
    plugin_complexity: str = "simple"
    description: str = ""


@dataclass(frozen=True)
class SupportPlan:
    """What CompGen needs to generate for this target.

    Attributes:
        target_name: Target identifier.
        family: Classified target family.
        classification: Full classification result.
        required_stages: Ordered list of stages needed.
        required_dialects: New IR dialects needed.
        kernel_backend: Primary kernel backend.
        needs_accel_dialect: Whether the accel dialect is needed.
        needs_ukernel_dialect: Whether the ukernel dialect is needed.
        llvm_patches_needed: Whether LLVM modifications are needed.
        estimated_effort: Overall effort estimate.
        notes: Human-readable notes.
        runtime_template: Which runtime template to use.
        threading_model: Threading model — pthreads, k_thread, polling, or none.
        memory_strategy: Memory strategy — dynamic (malloc), static (arena), or firmware.
    """

    target_name: str
    family: TargetFamily
    classification: Classification
    required_stages: list[StageRequirement] = field(default_factory=list)
    required_dialects: list[str] = field(default_factory=list)
    kernel_backend: str = "triton"
    needs_accel_dialect: bool = False
    needs_ukernel_dialect: bool = False
    llvm_patches_needed: bool = False
    estimated_effort: str = "medium"
    notes: str = ""
    runtime_template: str = "linux_userspace"
    threading_model: str = "pthreads"
    memory_strategy: str = "dynamic"


# Stage sequences per family
_SHARED_ENCODING = StageRequirement("encoding", "EncodingStage", True, "simple", "Data layout resolution")
_SHARED_DISPATCH = StageRequirement("dispatch", "DispatchStage", True, "simple", "Graph partitioning")
_SHARED_BUNDLE = StageRequirement("bundle", "BundleStage", True, "trivial", "Artifact packaging")
_TILING = StageRequirement("tiling", "TilingStage", True, "moderate", "Data tiling")
_CODEGEN = StageRequirement("codegen", "CodegenStage", True, "moderate", "Backend selection")
_SCHEDULING = StageRequirement("scheduling", "SchedulingStage", True, "moderate", "Device scheduling")
_MEMORY_PLAN = StageRequirement("memory_plan", "MemoryPlanStage", True, "complex", "Memory allocation")


def _rvv_cpu_stages() -> list[StageRequirement]:
    return [_SHARED_ENCODING, _SHARED_DISPATCH, _TILING, _CODEGEN, _SHARED_BUNDLE]


def _vendor_matrix_stages() -> list[StageRequirement]:
    return [
        _SHARED_ENCODING,
        _SHARED_DISPATCH,
        _TILING,
        StageRequirement("matrix_lowering", "LoweringStage", True, "complex", "Vendor matrix extension lowering"),
        _CODEGEN,
        _SHARED_BUNDLE,
    ]


def _rocc_accelerator_stages() -> list[StageRequirement]:
    return [
        _SHARED_ENCODING,
        _SHARED_DISPATCH,
        _TILING,
        StageRequirement("accel_lowering", "LoweringStage", True, "complex", "Accelerator dialect lowering"),
        _MEMORY_PLAN,
        _SCHEDULING,
        _SHARED_BUNDLE,
    ]


def _npu_text_isa_stages() -> list[StageRequirement]:
    return [
        _SHARED_ENCODING,
        _SHARED_DISPATCH,
        StageRequirement("kernel_contract", "LoweringStage", True, "complex", "Kernel→schedule lowering"),
        StageRequirement("isa_lowering", "LoweringStage", True, "complex", "Schedule→ISA lowering"),
        _MEMORY_PLAN,
        _SCHEDULING,
        _SHARED_BUNDLE,
    ]


def _simt_gpu_stages() -> list[StageRequirement]:
    return [_SHARED_ENCODING, _SHARED_DISPATCH, _TILING, _CODEGEN, _SHARED_BUNDLE]


_FAMILY_STAGES: dict[TargetFamily, list[StageRequirement]] = {
    TargetFamily.RVV_CPU_EXTENSION: _rvv_cpu_stages(),
    TargetFamily.RISCV_VENDOR_MATRIX: _vendor_matrix_stages(),
    TargetFamily.ROCC_ACCELERATOR: _rocc_accelerator_stages(),
    TargetFamily.STRUCTURED_NPU_TEXT_ISA: _npu_text_isa_stages(),
    TargetFamily.SIMT_GPU_HAL: _simt_gpu_stages(),
}

_FAMILY_BACKENDS: dict[TargetFamily, str] = {
    TargetFamily.RVV_CPU_EXTENSION: "llvm",
    TargetFamily.RISCV_VENDOR_MATRIX: "llvm",
    TargetFamily.ROCC_ACCELERATOR: "accel",
    TargetFamily.STRUCTURED_NPU_TEXT_ISA: "ukernel",
    TargetFamily.SIMT_GPU_HAL: "triton",
}

# Deployment model → threading model
_DEPLOYMENT_THREADING: dict[str, str] = {
    "linux_userspace": "pthreads",
    "linux_embedded": "pthreads",
    "zephyr_rtos": "k_thread",
    "bare_metal": "polling",
    "firmware": "none",
}

# Deployment model → memory strategy
_DEPLOYMENT_MEMORY: dict[str, str] = {
    "linux_userspace": "dynamic",
    "linux_embedded": "dynamic",
    "zephyr_rtos": "static",
    "bare_metal": "static",
    "firmware": "firmware",
}


def generate_support_plan(
    spec: HardwareSpec,
    classification: Classification,
) -> SupportPlan:
    """Generate a support plan from a hardware spec and classification.

    Maps each family to its required stages, dialects, and patches.
    """
    family = classification.family
    stages = list(_FAMILY_STAGES.get(family, _rvv_cpu_stages()))
    backend = _FAMILY_BACKENDS.get(family, "fallback")

    # Determine dialect and patch needs
    needs_accel = family in (TargetFamily.ROCC_ACCELERATOR, TargetFamily.RISCV_VENDOR_MATRIX)
    needs_ukernel = family == TargetFamily.STRUCTURED_NPU_TEXT_ISA
    required_dialects = list(spec.patches.new_dialects_needed)

    # Check if LLVM patches are needed
    llvm_patches = any(r.component == "llvm" or "llvm" in r.description.lower() for r in spec.patches.requirements)
    # Also check ISA exposure
    if spec.isa.compiler_intrinsics and family in (TargetFamily.RVV_CPU_EXTENSION, TargetFamily.RISCV_VENDOR_MATRIX):
        if spec.isa.custom_instructions:
            llvm_patches = True

    # Effort estimate
    stage_count = len(stages)
    if stage_count <= 5:
        effort = "small"
    elif stage_count <= 7:
        effort = "medium"
    else:
        effort = "large"

    if needs_accel or needs_ukernel:
        effort = "large"

    # Runtime fields from deployment model
    deployment = spec.platform.deployment_model
    runtime_template = deployment
    threading_model = _DEPLOYMENT_THREADING.get(deployment, "pthreads")
    memory_strategy = _DEPLOYMENT_MEMORY.get(deployment, "dynamic")

    return SupportPlan(
        target_name=spec.name,
        family=family,
        classification=classification,
        required_stages=stages,
        required_dialects=required_dialects,
        kernel_backend=backend,
        needs_accel_dialect=needs_accel,
        needs_ukernel_dialect=needs_ukernel,
        llvm_patches_needed=llvm_patches,
        estimated_effort=effort,
        notes=f"Family: {family.value}, {len(stages)} stages, backend: {backend}",
        runtime_template=runtime_template,
        threading_model=threading_model,
        memory_strategy=memory_strategy,
    )
