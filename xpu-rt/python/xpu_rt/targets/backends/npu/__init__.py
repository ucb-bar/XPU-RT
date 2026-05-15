"""NPU target backend — extension point implementation.

Provides NPU-specific compilation, program emission, and memory mapping
for the 32x32 tile MXU + BF16 VPU architecture.

This is an extension point implementation. The scaffold is in
``compgen.targets.backend`` (TargetBackendProtocol).
"""

from compgen.targets.backends.npu.memory_map import (
    NPU_DRAM_BASE,
    NPU_DRAM_SIZE,
    NPU_VMEM_BASE,
    NPU_VMEM_SIZE,
    npu_memory_regions,
)
from compgen.targets.backends.npu.program_emitter import NpuProgramEmitter

__all__ = [
    "NPU_DRAM_BASE",
    "NPU_DRAM_SIZE",
    "NPU_VMEM_BASE",
    "NPU_VMEM_SIZE",
    "NpuProgramEmitter",
    "npu_memory_regions",
]
