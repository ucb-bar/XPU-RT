"""Saturn OPU target backend.

Saturn OPU (UCB-BAR ``chipyard/generators/saturn``, ``OPUV128D64ShuttleConfig``)
is an RV64 Shuttle core with a 128-bit vector unit and a custom
outer-product accumulator (``+xopu``). The vector half uses the standard
RVV CPU extension path; the outer-product half dispatches to Exo-generated
``mmt4d_s8s8s32_16x16x128`` ukernels.

This module glues the HardwareSpec at
``examples/hardware_specs/saturn_opu.yaml`` into CompGen's target backend
protocol. Compile stages are delegated to the RVV CPU extension family
(``compgen.targetgen.families.rvv_cpu_extension``); the only Saturn-specific
logic here is recognizing the target name and exposing tuned
:class:`SaturnOPUOptions` that the Exo provider and the embedded runtime
emitter inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

from compgen.targets.backend import BaseTargetBackend, CompilationStageResult
from compgen.targets.options import TargetOptions


@dataclass(frozen=True)
class SaturnOPUOptions(TargetOptions):
    """Saturn OPU compilation options.

    Attributes mirror :class:`TargetOptions` with Saturn-specific fields
    appended. Frozen so :class:`SaturnOPUBackend` can share a single
    default options instance.
    """

    target_name: str = "saturn-opu-v128d64"
    vector_length_bits: int = 128
    dlen_bits: int = 64
    opu_tile_m: int = 16
    opu_tile_n: int = 16
    opu_tile_k: int = 128
    opu_num_matrix_regs: int = 4
    mcpu_features: str = "+m,+a,+f,+d,+c,+v,+zvl128b,+xopu"
    target_triple: str = "riscv64-unknown-elf"
    target_abi: str = "lp64d"
    chipyard_config: str = "OPUV128D64ShuttleConfig"
    dram_base: int = 0x80000000


class SaturnOPUBackend(BaseTargetBackend):
    """Saturn OPU backend; delegates lowering to the RVV CPU extension family."""

    def __init__(self) -> None:
        super().__init__(SaturnOPUOptions())

    def supports_target(self, target_name: str) -> bool:
        return target_name in {
            "saturn-opu-v128d64",
            "saturn_opu",
            "saturn-opu",
        }

    def compile_stage(
        self,
        stage_name: str,
        ir_text: str,
        options: TargetOptions,
    ) -> CompilationStageResult:
        # The RVV family's stage stack (Encoding/Dispatch/Tiling/Codegen/Bundle)
        # is registered when api.device() extracts the dialect stack. This
        # backend is a carrier for SaturnOPUOptions; real lowering happens
        # through the stage pipeline.
        return CompilationStageResult(stage_name=stage_name, success=True, ir_text=ir_text)


__all__ = ["SaturnOPUBackend", "SaturnOPUOptions"]
