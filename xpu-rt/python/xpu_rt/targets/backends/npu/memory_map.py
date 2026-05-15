"""NPU memory map and address space definitions.

Defines the physical memory layout for the NPU target:
- DRAM: 16 GiB at 0x80000000 (shared with CPU host)
- VMEM: 1 MiB at 0x20000000 (on-chip scratchpad, fast)
- IMEM: 64 KiB at 0x00020000 (instruction memory)

These match the hardware spec in ``third_party/npu_model/``.
"""

from __future__ import annotations

from compgen.runtime.memory_layout import MemoryRegion

# NPU address spaces (from npu_model/configs/hardware/default.py)
NPU_DRAM_BASE = 0x80000000
NPU_DRAM_SIZE = 16 * 1024 * 1024 * 1024  # 16 GiB

NPU_VMEM_BASE = 0x20000000
NPU_VMEM_SIZE = 1 * 1024 * 1024  # 1 MiB

NPU_IMEM_BASE = 0x00020000
NPU_IMEM_SIZE = 64 * 1024  # 64 KiB

# MXU tile geometry
NPU_MXU_TILE_M = 32
NPU_MXU_TILE_N = 32
NPU_MXU_TILE_K = 32

# DMA
NPU_DMA_CHANNELS = 8
NPU_DMA_ALIGNMENT = 32  # bytes

# Tensor register file
NPU_TENSOR_REGISTERS = 64
NPU_TENSOR_REG_SIZE = 1024  # bytes (32 rows x 32 bytes)

# Scale registers (for FP8 po2 scaling)
NPU_SCALE_REGISTERS = 32


def npu_memory_regions() -> list[MemoryRegion]:
    """Return the NPU's standard memory regions for memory planning."""
    return [
        MemoryRegion(
            name="dram",
            base_addr=NPU_DRAM_BASE,
            size_bytes=16 * 1024 * 1024,  # Use 16 MiB for simulation
            device="npu",
            address_space=0,
        ),
        MemoryRegion(
            name="vmem",
            base_addr=NPU_VMEM_BASE,
            size_bytes=NPU_VMEM_SIZE,
            device="npu",
            address_space=1,
        ),
    ]


__all__ = [
    "NPU_DMA_ALIGNMENT",
    "NPU_DMA_CHANNELS",
    "NPU_DRAM_BASE",
    "NPU_DRAM_SIZE",
    "NPU_IMEM_BASE",
    "NPU_IMEM_SIZE",
    "NPU_MXU_TILE_K",
    "NPU_MXU_TILE_M",
    "NPU_MXU_TILE_N",
    "NPU_SCALE_REGISTERS",
    "NPU_TENSOR_REG_SIZE",
    "NPU_TENSOR_REGISTERS",
    "NPU_VMEM_BASE",
    "NPU_VMEM_SIZE",
    "npu_memory_regions",
]
