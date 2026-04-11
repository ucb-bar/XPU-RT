"""Memory layout planning for multi-device programs.

Assigns tensor buffers to physical memory regions across devices (DRAM,
scratchpad, host RAM).  Supports heterogeneous systems where a CPU host
and one or more accelerators share a memory hierarchy.

The planner produces deterministic, zero-fragmentation allocations suitable
for baremetal deployment (no dynamic allocation at runtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryRegion:
    """A contiguous physical memory region on a device.

    Attributes:
        name: Region identifier (``"dram"``, ``"vmem"``, ``"host_stack"``).
        base_addr: Physical base address.
        size_bytes: Total capacity.
        device: Which device owns this region (``"cpu"``, ``"npu"``).
        address_space: MLIR address space index (0=global, 1=scratchpad).
    """

    name: str
    base_addr: int
    size_bytes: int
    device: str = "cpu"
    address_space: int = 0


@dataclass(frozen=True)
class BufferRef:
    """Reference to a data buffer in the program.

    Attributes:
        name: Buffer identifier (``"weight_q_proj"``, ``"activation_0"``).
        shape: Tensor shape.
        dtype: Element data type string (``"bf16"``, ``"fp8_e4m3"``, ``"f32"``).
        size_bytes: Total size in bytes.
        persistent: Whether the buffer persists across kernel invocations
            (weights=True, intermediates=False).
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    persistent: bool = False


@dataclass
class BufferAllocation:
    """A buffer assigned to a specific location in a memory region.

    Attributes:
        buffer: The buffer being allocated.
        region: Which memory region (by name).
        offset: Byte offset within the region.
        addr: Absolute physical address.
    """

    buffer: BufferRef
    region: str
    offset: int
    addr: int


@dataclass
class MemoryLayout:
    """Complete memory layout for a multi-device program.

    Attributes:
        regions: Available memory regions.
        allocations: Buffer-to-region assignments.
        peak_usage: Peak bytes used per region.
    """

    regions: dict[str, MemoryRegion] = field(default_factory=dict)
    allocations: dict[str, BufferAllocation] = field(default_factory=dict)
    peak_usage: dict[str, int] = field(default_factory=dict)

    def get_addr(self, buffer_name: str) -> int:
        """Get the absolute address for a buffer."""
        alloc = self.allocations.get(buffer_name)
        if alloc is None:
            raise KeyError(f"Buffer not allocated: {buffer_name}")
        return alloc.addr

    def to_c_defines(self) -> str:
        """Generate C #define statements for all buffer addresses."""
        lines = ["/* Auto-generated memory map */"]
        for name, alloc in sorted(self.allocations.items()):
            c_name = name.upper().replace(".", "_")
            lines.append(f"#define {c_name}_ADDR  0x{alloc.addr:08X}")
            lines.append(f"#define {c_name}_SIZE  {alloc.buffer.size_bytes}")
        return "\n".join(lines)


class MemoryPlanner:
    """Assigns buffers to memory regions using a simple bump allocator.

    For baremetal deployment: no dynamic allocation, deterministic layout.
    Weights are placed first (persistent), then intermediates (reusable).

    Args:
        regions: Available memory regions.
        alignment: Byte alignment for each allocation.
    """

    def __init__(
        self,
        regions: list[MemoryRegion],
        alignment: int = 64,
    ) -> None:
        self._regions = {r.name: r for r in regions}
        self._alignment = alignment

    def plan(
        self,
        buffers: list[BufferRef],
        region_assignment: dict[str, str] | None = None,
    ) -> MemoryLayout:
        """Plan buffer allocations across memory regions.

        Args:
            buffers: Buffers to allocate.
            region_assignment: Optional explicit buffer→region mapping.
                If not provided, persistent buffers go to the largest region,
                intermediates to the fastest (smallest) region.

        Returns:
            Complete ``MemoryLayout`` with all allocations.
        """
        layout = MemoryLayout(regions=dict(self._regions))
        cursors: dict[str, int] = {name: 0 for name in self._regions}

        # Sort: persistent first (weights), then by size descending
        sorted_buffers = sorted(
            buffers,
            key=lambda b: (not b.persistent, -b.size_bytes),
        )

        for buf in sorted_buffers:
            # Determine target region
            if region_assignment and buf.name in region_assignment:
                region_name = region_assignment[buf.name]
            elif buf.persistent:
                # Weights → largest region (typically DRAM)
                region_name = max(self._regions, key=lambda r: self._regions[r].size_bytes)
            else:
                # Intermediates → smallest region (typically scratchpad)
                region_name = min(self._regions, key=lambda r: self._regions[r].size_bytes)

            region = self._regions[region_name]

            # Align cursor
            cursor = cursors[region_name]
            aligned = (cursor + self._alignment - 1) & ~(self._alignment - 1)

            # Check capacity
            if aligned + buf.size_bytes > region.size_bytes:
                # Fall back to largest region
                region_name = max(self._regions, key=lambda r: self._regions[r].size_bytes)
                region = self._regions[region_name]
                cursor = cursors[region_name]
                aligned = (cursor + self._alignment - 1) & ~(self._alignment - 1)

            addr = region.base_addr + aligned
            layout.allocations[buf.name] = BufferAllocation(
                buffer=buf,
                region=region_name,
                offset=aligned,
                addr=addr,
            )
            cursors[region_name] = aligned + buf.size_bytes

        # Record peak usage
        layout.peak_usage = dict(cursors)
        return layout


__all__ = [
    "BufferAllocation",
    "BufferRef",
    "MemoryLayout",
    "MemoryPlanner",
    "MemoryRegion",
]
