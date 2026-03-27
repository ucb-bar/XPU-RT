"""5-family decision engine for target classification.

Given a HardwareSpec, deterministically classifies the target into a
family, integration style, and lowering surface.  The primary classifier
is ``ExecutionModel``; secondary classifiers refine the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from compgen.targetgen.hardware_spec import ExecutionModel, HardwareSpec
from compgen.targets.capability import TargetClass


class TargetFamily(Enum):
    """Five target families from Merlin analysis."""

    RVV_CPU_EXTENSION = "rvv_cpu_extension"
    RISCV_VENDOR_MATRIX = "riscv_vendor_matrix_extension"
    ROCC_ACCELERATOR = "rocc_accelerator"
    STRUCTURED_NPU_TEXT_ISA = "structured_npu_text_isa"
    SIMT_GPU_HAL = "simt_gpu_hal"


class IntegrationStyle(Enum):
    """How CompGen integrates with this target."""

    LLVM_BACKEND = "llvm_backend"
    CUSTOM_LOWERING = "custom_lowering"
    RUNTIME_API = "runtime_api"
    HAL_DRIVER = "hal_driver"


class LoweringSurface(Enum):
    """Where CompGen's IR lowers to."""

    LLVM_IR = "llvm_ir"
    CUSTOM_DIALECT = "custom_dialect"
    UKERNEL_CALLS = "ukernel_calls"
    TRITON_IR = "triton_ir"


@dataclass(frozen=True)
class Classification:
    """Result of classifying a hardware spec."""

    family: TargetFamily
    target_class: TargetClass
    integration_style: IntegrationStyle
    lowering_surface: LoweringSurface
    confidence: float
    reasoning: str


def classify_hardware(spec: HardwareSpec) -> Classification:
    """Classify a hardware spec into a target family.

    Decision rules (deterministic):
      SIMD_VECTOR + rv* ISA → RVV_CPU_EXTENSION
      DECOUPLED_MATRIX + rv* → RISCV_VENDOR_MATRIX
      ROCC_COPROCESSOR → ROCC_ACCELERATOR
      TEXT_ISA_NPU / FIRMWARE_DRIVEN → STRUCTURED_NPU_TEXT_ISA
      SIMT_GPU → SIMT_GPU_HAL
      DATAFLOW → ROCC_ACCELERATOR (closest, medium confidence)
    """
    model = spec.execution_model.model
    base_isa = spec.isa.base_isa.lower()
    is_riscv = "rv" in base_isa or "riscv" in base_isa

    if model == ExecutionModel.SIMD_VECTOR:
        if is_riscv:
            return Classification(
                family=TargetFamily.RVV_CPU_EXTENSION,
                target_class=TargetClass.TRITON_FRIENDLY,
                integration_style=IntegrationStyle.LLVM_BACKEND,
                lowering_surface=LoweringSurface.LLVM_IR,
                confidence=0.95,
                reasoning="SIMD vector on RISC-V → RVV CPU extension path",
            )
        return Classification(
            family=TargetFamily.RVV_CPU_EXTENSION,
            target_class=TargetClass.TRITON_FRIENDLY,
            integration_style=IntegrationStyle.LLVM_BACKEND,
            lowering_surface=LoweringSurface.LLVM_IR,
            confidence=0.80,
            reasoning=f"SIMD vector on {base_isa} → LLVM backend (non-RV)",
        )

    if model == ExecutionModel.DECOUPLED_MATRIX:
        return Classification(
            family=TargetFamily.RISCV_VENDOR_MATRIX,
            target_class=TargetClass.ACCEL_NATIVE,
            integration_style=IntegrationStyle.CUSTOM_LOWERING,
            lowering_surface=LoweringSurface.CUSTOM_DIALECT,
            confidence=0.90 if is_riscv else 0.75,
            reasoning="Decoupled matrix extension → vendor matrix path",
        )

    if model == ExecutionModel.ROCC_COPROCESSOR:
        return Classification(
            family=TargetFamily.ROCC_ACCELERATOR,
            target_class=TargetClass.ACCEL_NATIVE,
            integration_style=IntegrationStyle.CUSTOM_LOWERING,
            lowering_surface=LoweringSurface.CUSTOM_DIALECT,
            confidence=0.95,
            reasoning="RoCC coprocessor → accelerator dialect recovery path",
        )

    if model in (ExecutionModel.TEXT_ISA_NPU, ExecutionModel.FIRMWARE_DRIVEN):
        return Classification(
            family=TargetFamily.STRUCTURED_NPU_TEXT_ISA,
            target_class=TargetClass.UKERNEL_RUNTIME,
            integration_style=IntegrationStyle.RUNTIME_API,
            lowering_surface=LoweringSurface.UKERNEL_CALLS,
            confidence=0.90,
            reasoning="NPU/firmware → structured text ISA path",
        )

    if model == ExecutionModel.SIMT_GPU:
        return Classification(
            family=TargetFamily.SIMT_GPU_HAL,
            target_class=TargetClass.TRITON_FRIENDLY,
            integration_style=IntegrationStyle.HAL_DRIVER,
            lowering_surface=LoweringSurface.TRITON_IR,
            confidence=0.95,
            reasoning="SIMT GPU → Triton/HAL driver path",
        )

    if model == ExecutionModel.DATAFLOW:
        return Classification(
            family=TargetFamily.ROCC_ACCELERATOR,
            target_class=TargetClass.ACCEL_NATIVE,
            integration_style=IntegrationStyle.CUSTOM_LOWERING,
            lowering_surface=LoweringSurface.CUSTOM_DIALECT,
            confidence=0.60,
            reasoning="Dataflow → closest match is accelerator path (review needed)",
        )

    # Fallback
    return Classification(
        family=TargetFamily.RVV_CPU_EXTENSION,
        target_class=TargetClass.HYBRID,
        integration_style=IntegrationStyle.LLVM_BACKEND,
        lowering_surface=LoweringSurface.LLVM_IR,
        confidence=0.30,
        reasoning=f"Unknown model {model.value} → fallback to CPU extension",
    )
