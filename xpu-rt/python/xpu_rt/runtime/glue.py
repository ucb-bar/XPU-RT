"""Unified runtime adapter — one protocol every backend satisfies.

Today each backend (cuda_gpu, cpu, baremetal) is a bespoke executor.
The agent has to know which one to call. ``RuntimeAdapter`` collapses
that into a single interface every backend implements. Routes
per-kernel by ``KernelContractV3.orchestration.dispatch.model``:

  * ``SYNC``       — host blocks until completion
  * ``ASYNC``      — fire-and-forget, completion via events
  * ``PERSISTENT`` — long-lived worker (megakernel)
  * ``INLINE``     — not dispatched at all (MICRO ukernel — emitted into caller)

The adapter doesn't *generate* kernels — that's the kernel-provider
layer (already built). It launches them, manages buffers, captures
graphs when supported, and synchronises.

Implementations:
  * ``CudaRuntimeAdapter`` — wraps ``CudaGraphCaptureWrapper`` +
    ``torch.cuda`` allocator + the existing ``gpu_executor``.
  * ``CpuRuntimeAdapter`` — wraps the existing ``cpu_executor.execute``.
  * ``BaremetalRuntimeAdapter`` — wraps the C-codegen path
    (``runtime/baremetal/c_codegen.py``); skeleton for now.
  * ``RocmRuntimeAdapter`` — Wave 5 skeleton.

A single ``select_adapter(target_name) → RuntimeAdapter`` factory
walks the existing target-name → backend mapping (mirrors the
knowledge-store scope rules so adapter selection and lesson lookup
share the same target taxonomy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from compgen.kernels.contract_v3 import (
    DispatchModel,
    KernelContractV3,
    MemoryTier,
)


# ---------------------------------------------------------------------------
# Buffer + dispatch primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferSpec:
    """What the planner is asking the runtime to allocate."""

    nbytes: int
    dtype: str
    tier: MemoryTier = MemoryTier.DEVICE_DRAM
    alignment_bytes: int = 16


@dataclass
class Buffer:
    """An adapter-allocated buffer. Opaque to the caller; carries an
    adapter-specific handle the dispatcher passes to the kernel."""

    spec: BufferSpec
    handle: Any            # adapter-specific (a torch.Tensor, an mmap, …)
    adapter_name: str = ""


@dataclass
class CapturedGraph:
    """A captured execution graph the adapter can replay cheaply."""

    backend: str
    handle: Any            # adapter-specific (cudaGraph_t, …)
    input_shapes: tuple[tuple, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchResult:
    """What ``dispatch`` returns."""

    output: Any                                     # kernel result(s)
    elapsed_us: float | None = None                 # when the adapter measured it
    adapter_name: str = ""
    used_graph_replay: bool = False


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RuntimeAdapter(Protocol):
    """One interface, all backends. The agent calls these without
    knowing which backend it's on."""

    @property
    def name(self) -> str: ...

    def supports(self, contract: KernelContractV3) -> bool:
        """Whether this adapter can host this kernel's dispatch model."""
        ...

    def dispatch(
        self,
        contract: KernelContractV3,
        callable_kernel: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> DispatchResult:
        """Launch one invocation of ``callable_kernel`` per the contract."""
        ...

    def capture_graph(
        self,
        model_fn: Callable[..., Any],
        sample_inputs: tuple,
    ) -> CapturedGraph | None:
        """Capture ``model_fn`` for one-shot replay. Return ``None`` if
        graph capture isn't supported on this backend."""
        ...

    def replay(self, graph: CapturedGraph, inputs: tuple) -> Any:
        """Replay a previously captured graph with new inputs."""
        ...

    def synchronize(self) -> None:
        """Block until all outstanding async dispatches complete."""
        ...

    def allocate_buffer(self, spec: BufferSpec) -> Buffer:
        """Allocate a buffer at the requested tier."""
        ...


# ---------------------------------------------------------------------------
# CUDA adapter — wraps cuda_graph_wrapper + torch.cuda
# ---------------------------------------------------------------------------


@dataclass
class CudaRuntimeAdapter:
    name_str: str = "cuda"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        # CUDA can host any of the four dispatch models; INLINE means the
        # kernel is spliced into a parent at codegen time, not dispatched.
        return True

    def dispatch(
        self,
        contract: KernelContractV3,
        callable_kernel: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> DispatchResult:
        import torch

        model = contract.orchestration.dispatch.model
        if model is DispatchModel.INLINE:
            raise ValueError(
                f"{self.name!r}: INLINE-dispatch contract {contract.op_name!r} "
                "is not launched as a separate kernel — it's emitted into a "
                "parent kernel at codegen time."
            )

        if model is DispatchModel.SYNC:
            out = callable_kernel(*args, **kwargs)
            torch.cuda.synchronize()
            return DispatchResult(output=out, adapter_name=self.name)

        # ASYNC + PERSISTENT both fire-and-forget; caller relies on
        # subsequent .synchronize() or event-tensor completion.
        out = callable_kernel(*args, **kwargs)
        return DispatchResult(output=out, adapter_name=self.name)

    def capture_graph(
        self,
        model_fn: Callable[..., Any],
        sample_inputs: tuple,
    ) -> CapturedGraph | None:
        import torch
        if not torch.cuda.is_available():
            return None

        from compgen.runtime.cuda_graph_wrapper import CudaGraphCaptureWrapper
        wrapper = CudaGraphCaptureWrapper(
            model_fn=lambda x: model_fn(x), warmup_iters=3,
        )
        # Trigger capture for the sample shape
        if sample_inputs:
            wrapper(sample_inputs[0])
        return CapturedGraph(
            backend="cuda",
            handle=wrapper,
            input_shapes=tuple(tuple(t.shape) for t in sample_inputs if hasattr(t, "shape")),
        )

    def replay(self, graph: CapturedGraph, inputs: tuple) -> Any:
        wrapper = graph.handle
        return wrapper(inputs[0])

    def synchronize(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def allocate_buffer(self, spec: BufferSpec) -> Buffer:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Map dtype string → torch dtype
        dtype_map = {
            "f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32,
            "f64": torch.float64, "i8": torch.int8, "i32": torch.int32, "i64": torch.int64,
        }
        td = dtype_map.get(spec.dtype, torch.float32)
        elements = max(spec.nbytes // td.itemsize, 1)
        t = torch.empty(elements, device=device, dtype=td)
        return Buffer(spec=spec, handle=t, adapter_name=self.name)


# ---------------------------------------------------------------------------
# CPU adapter — wraps cpu_executor for SYNC dispatches
# ---------------------------------------------------------------------------


@dataclass
class CpuRuntimeAdapter:
    name_str: str = "cpu"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        # CPU doesn't do PERSISTENT (no host-side persistent worker model)
        # or INLINE (no codegen-time inlining for dispatched ops).
        return contract.orchestration.dispatch.model in (
            DispatchModel.SYNC, DispatchModel.ASYNC,
        )

    def dispatch(
        self, contract, callable_kernel, args, kwargs,
    ) -> DispatchResult:
        out = callable_kernel(*args, **kwargs)
        return DispatchResult(output=out, adapter_name=self.name)

    def capture_graph(self, model_fn, sample_inputs) -> CapturedGraph | None:
        # CPU has no native graph capture (torch.compile aside, which is
        # handled by the existing torch_backend).
        return None

    def replay(self, graph: CapturedGraph, inputs: tuple) -> Any:
        raise NotImplementedError("CPU adapter does not capture graphs")

    def synchronize(self) -> None:
        # CPU is synchronous by construction.
        return

    def allocate_buffer(self, spec: BufferSpec) -> Buffer:
        import torch
        dtype_map = {
            "f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32,
            "f64": torch.float64, "i8": torch.int8, "i32": torch.int32, "i64": torch.int64,
        }
        td = dtype_map.get(spec.dtype, torch.float32)
        elements = max(spec.nbytes // td.itemsize, 1)
        t = torch.empty(elements, device="cpu", dtype=td)
        return Buffer(spec=spec, handle=t, adapter_name=self.name)


# ---------------------------------------------------------------------------
# Baremetal adapter — for ukernel_runtime targets (Hexagon C codegen path)
# ---------------------------------------------------------------------------


@dataclass
class BaremetalRuntimeAdapter:
    """Skeleton — wraps the C-codegen path that already exists at
    ``runtime/baremetal/c_codegen.py``. Dispatch is "compile + simulate"
    today; production deployment loads the bundle on real NPU silicon."""

    name_str: str = "baremetal"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        # Baremetal targets typically use SYNC + ASYNC; INLINE for ukernels.
        return contract.orchestration.dispatch.model in (
            DispatchModel.SYNC, DispatchModel.ASYNC, DispatchModel.INLINE,
        )

    def dispatch(self, contract, callable_kernel, args, kwargs) -> DispatchResult:
        # In the bundle-export flow the "callable_kernel" is the C source
        # function. Real dispatch lives off-host on NPU. For agent loop
        # bench/correctness, we use the cpu_executor as a stand-in.
        out = callable_kernel(*args, **kwargs)
        return DispatchResult(output=out, adapter_name=self.name)

    def capture_graph(self, model_fn, sample_inputs) -> CapturedGraph | None:
        return None     # baremetal models the entire program as one C bundle

    def replay(self, graph: CapturedGraph, inputs: tuple) -> Any:
        raise NotImplementedError

    def synchronize(self) -> None:
        return

    def allocate_buffer(self, spec: BufferSpec) -> Buffer:
        import torch
        # On host we use torch tensors; in real deployment this maps to
        # the NPU's allocator (the bundle's memory_plan.yaml drives it).
        t = torch.empty(max(spec.nbytes // 4, 1), device="cpu", dtype=torch.float32)
        return Buffer(spec=spec, handle=t, adapter_name=self.name)


# ---------------------------------------------------------------------------
# ROCm skeleton — wave 5 will flesh this out
# ---------------------------------------------------------------------------


@dataclass
class RocmRuntimeAdapter:
    """Skeleton. Wraps HIP / ROCm-Triton stack; same protocol as CUDA
    but routes to AMD's runtime. Wave 5 fills in dispatch + capture."""

    name_str: str = "rocm"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        return False    # not implemented yet — falls through to CPU adapter

    def dispatch(self, contract, callable_kernel, args, kwargs):
        raise NotImplementedError("ROCm adapter is W5 placeholder")

    def capture_graph(self, model_fn, sample_inputs):
        return None

    def replay(self, graph, inputs):
        raise NotImplementedError("ROCm adapter is W5 placeholder")

    def synchronize(self) -> None:
        return

    def allocate_buffer(self, spec: BufferSpec) -> Buffer:
        raise NotImplementedError("ROCm adapter is W5 placeholder")


# ---------------------------------------------------------------------------
# Factory — pick adapter from target name
# ---------------------------------------------------------------------------


def select_adapter(target_name: str) -> RuntimeAdapter:
    """Return the right ``RuntimeAdapter`` for ``target_name``.

    Mirrors the knowledge-store target-name → backend mapping so
    adapter selection and lesson lookup share the same target taxonomy.
    """
    n = (target_name or "").lower()
    if n.startswith("cuda") or "titan-rtx" in n or "test-gpu-simt" in n:
        return CudaRuntimeAdapter()
    if n.startswith("rocm") or n.startswith("mi"):
        return RocmRuntimeAdapter()
    if n.startswith("hexagon") or n.startswith("openq") or n.startswith("trainium"):
        return BaremetalRuntimeAdapter()
    if n.startswith("cpu") or "host" in n or "riscv" in n:
        return CpuRuntimeAdapter()
    return CpuRuntimeAdapter()         # safest fall-back


__all__ = [
    "BaremetalRuntimeAdapter",
    "Buffer",
    "BufferSpec",
    "CapturedGraph",
    "CpuRuntimeAdapter",
    "CudaRuntimeAdapter",
    "DispatchResult",
    "RocmRuntimeAdapter",
    "RuntimeAdapter",
    "select_adapter",
]
