"""Multi-device program assembly.

Stitches generated kernels into a complete program for heterogeneous
execution (CPU host + accelerator device).  Handles:

- Kernel-to-device assignment (which ops run on CPU vs accelerator)
- Memory layout planning (buffer → address mapping)
- Data transfer sequencing (DMA / memcpy between devices)
- Weight initialization ordering

Target-agnostic: delegates device-specific code emission to the target
backend (``TargetBackendProtocol``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.runtime.memory_layout import BufferRef, MemoryLayout, MemoryPlanner, MemoryRegion


@dataclass
class DeviceKernel:
    """A kernel assigned to a specific device.

    Attributes:
        kernel_id: Unique identifier (``"matmul_0"``, ``"softmax_3"``).
        pattern_id: Which pattern this implements (``"matmul"``, ``"softmax"``).
        device: Target device (``"cpu"``, ``"npu"``, ``"gpu"``).
        code: Generated kernel code (ISA, C, Python).
        language: Code language (``"npu_asm"``, ``"c"``, ``"python"``).
        inputs: Buffers this kernel reads.
        outputs: Buffers this kernel writes.
        estimated_cycles: Estimated execution cycles (for scheduling).
    """

    kernel_id: str = ""
    pattern_id: str = ""
    device: str = "npu"
    code: str = ""
    language: str = "c"
    inputs: list[BufferRef] = field(default_factory=list)
    outputs: list[BufferRef] = field(default_factory=list)
    estimated_cycles: int = 0


@dataclass
class DataTransfer:
    """Data movement between devices or address spaces.

    Attributes:
        src: Source buffer.
        dst: Destination buffer.
        transfer_type: How the transfer is performed.
        size_bytes: Number of bytes to transfer.
    """

    src: BufferRef
    dst: BufferRef
    transfer_type: str = "dma"  # "dma", "memcpy", "mmio"
    size_bytes: int = 0


@dataclass
class BufferInit:
    """Weight/constant initialization for a buffer.

    Attributes:
        buffer: Which buffer to initialize.
        data_source: Where the data comes from (``"embedded"``, ``"file"``).
        data_key: Key in the weight_data dict or file path.
    """

    buffer: BufferRef
    data_source: str = "embedded"  # "embedded", "file", "zero"
    data_key: str = ""


@dataclass
class ModelProgram:
    """A complete multi-device program ready for emission.

    Attributes:
        name: Program identifier.
        host_kernels: Ops running on the CPU host.
        device_kernels: Ops running on the accelerator.
        execution_order: Ordered list of kernel_ids (interleaves host + device).
        data_transfers: Transfers between kernels.
        memory_layout: Buffer → physical address mapping.
        initialization: Weight loading sequence.
        weight_data: Serialized weight tensors (name → bytes).
    """

    name: str = "model"
    host_kernels: list[DeviceKernel] = field(default_factory=list)
    device_kernels: list[DeviceKernel] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    data_transfers: list[DataTransfer] = field(default_factory=list)
    memory_layout: MemoryLayout | None = None
    initialization: list[BufferInit] = field(default_factory=list)
    weight_data: dict[str, bytes] = field(default_factory=dict)

    @property
    def all_kernels(self) -> list[DeviceKernel]:
        """All kernels in execution order."""
        by_id = {k.kernel_id: k for k in self.host_kernels + self.device_kernels}
        return [by_id[kid] for kid in self.execution_order if kid in by_id]

    @property
    def kernel_count(self) -> dict[str, int]:
        """Count of kernels per device."""
        counts: dict[str, int] = {}
        for k in self.host_kernels + self.device_kernels:
            counts[k.device] = counts.get(k.device, 0) + 1
        return counts


class ProgramBuilder:
    """Assembles kernels + execution plan into a multi-device program.

    Usage::

        builder = ProgramBuilder(name="smolvla_fp8")
        builder.add_kernel(DeviceKernel(kernel_id="matmul_0", device="npu", ...))
        builder.add_kernel(DeviceKernel(kernel_id="embedding_0", device="cpu", ...))
        builder.set_memory_regions([
            MemoryRegion("dram", 0x80000000, 16*1024*1024, "npu"),
            MemoryRegion("vmem", 0x20000000, 1*1024*1024, "npu", address_space=1),
        ])
        program = builder.build()

    Args:
        name: Program identifier.
    """

    def __init__(self, name: str = "model") -> None:
        self._name = name
        self._kernels: list[DeviceKernel] = []
        self._execution_order: list[str] = []
        self._regions: list[MemoryRegion] = []
        self._weight_data: dict[str, bytes] = {}

    def add_kernel(self, kernel: DeviceKernel) -> ProgramBuilder:
        """Add a kernel to the program."""
        self._kernels.append(kernel)
        self._execution_order.append(kernel.kernel_id)
        return self

    def set_execution_order(self, order: list[str]) -> ProgramBuilder:
        """Override the default execution order."""
        self._execution_order = order
        return self

    def set_memory_regions(self, regions: list[MemoryRegion]) -> ProgramBuilder:
        """Set the available memory regions."""
        self._regions = regions
        return self

    def add_weight(self, name: str, data: bytes) -> ProgramBuilder:
        """Add serialized weight data."""
        self._weight_data[name] = data
        return self

    def build(self) -> ModelProgram:
        """Assemble the complete program.

        Performs memory planning and data transfer sequencing.
        """
        # Partition kernels by device
        host_kernels = [k for k in self._kernels if k.device == "cpu"]
        device_kernels = [k for k in self._kernels if k.device != "cpu"]

        # Collect all buffers
        all_buffers: dict[str, BufferRef] = {}
        for k in self._kernels:
            for buf in k.inputs + k.outputs:
                all_buffers[buf.name] = buf

        # Plan memory layout
        layout = None
        if self._regions:
            planner = MemoryPlanner(self._regions)
            layout = planner.plan(list(all_buffers.values()))

        # Build initialization list (persistent buffers = weights)
        initialization = [
            BufferInit(
                buffer=buf,
                data_source="embedded" if buf.name in self._weight_data else "zero",
                data_key=buf.name,
            )
            for buf in all_buffers.values()
            if buf.persistent
        ]

        # Sequence data transfers (between consecutive kernels on different devices)
        transfers = self._plan_transfers(self._kernels, self._execution_order)

        return ModelProgram(
            name=self._name,
            host_kernels=host_kernels,
            device_kernels=device_kernels,
            execution_order=self._execution_order,
            data_transfers=transfers,
            memory_layout=layout,
            initialization=initialization,
            weight_data=self._weight_data,
        )

    def _plan_transfers(
        self,
        kernels: list[DeviceKernel],
        order: list[str],
    ) -> list[DataTransfer]:
        """Plan data transfers between kernels on different devices."""
        transfers: list[DataTransfer] = []
        by_id = {k.kernel_id: k for k in kernels}

        for i in range(len(order) - 1):
            curr = by_id.get(order[i])
            next_k = by_id.get(order[i + 1])
            if curr is None or next_k is None:
                continue

            # If consecutive kernels are on different devices,
            # their shared buffers need transfers
            if curr.device != next_k.device:
                shared = set(b.name for b in curr.outputs) & set(b.name for b in next_k.inputs)
                for buf_name in shared:
                    src_buf = next(b for b in curr.outputs if b.name == buf_name)
                    dst_buf = next(b for b in next_k.inputs if b.name == buf_name)
                    transfers.append(
                        DataTransfer(
                            src=src_buf,
                            dst=dst_buf,
                            transfer_type="dma",
                            size_bytes=src_buf.size_bytes,
                        )
                    )

        return transfers


__all__ = [
    "BufferInit",
    "DataTransfer",
    "DeviceKernel",
    "ModelProgram",
    "ProgramBuilder",
]
