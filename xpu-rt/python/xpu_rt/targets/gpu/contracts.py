"""GPU target-class Protocols — the contract every vendor implements.

These are the abstractions the universal compile/dispatch path
calls into. The matcher (``runtime/lowering/fx_to_megakernel.py``),
the autotune probe (``runtime/autotune/``), and the dispatch
runtime (``mcp/tools/compile.py::compgen_run_compiled_bundle``)
import only from here — never from vendor-specific modules.

Every GPU vendor package (``targets/gpu/nvidia/``,
``targets/gpu/amd/``, ``targets/gpu/intel/``) must provide
implementations of:

- :class:`GpuProbe` — detect device capability at compile time.
- :class:`GpuBodyEmitter` — emit per-op kernel-body source for
  the chosen tile shape + precision.
- :class:`GpuRuntime` — JIT compile bodies, load module, dispatch.
- :class:`GpuCostModel` — TFLOPS / bandwidth / launch-overhead
  numbers for the roofline predictor.
- :class:`Device` — per-GPU handle (open / close / sync / ...).

Each Protocol is a runtime-checkable interface with a small,
stable surface. Adding a new vendor means writing a package that
satisfies these — no edits to the universal modules.

Scope notes (per ``target-hierarchy-inventory.md``):

- Things that look NVIDIA-specific in the names (cuBLASDx,
  cu13_nvrtc) are NOT in these Protocols. Vendor packages choose
  whatever GEMM library + JIT they want; the Protocol only sees
  ``DeviceFunctionSource`` (universal IR) and bytes of compiled
  PTX/SASS/AMDGPU/SPIR-V.
- The class-level scheduling-overhead constant lives here as a
  default; vendors override via :meth:`GpuCostModel.scheduling_overhead_us`.
- ``Device`` is the abstract handle. NVIDIA's CUcontext, AMD's
  hipContext_t, and Intel's level_zero handles all satisfy it.

The full migration plan is in
``docs/architecture/target-hierarchy-inventory.md``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

# ---------------------------------------------------------------------------
# Class-level constants — vendors can override
# ---------------------------------------------------------------------------


# Empirical scheduling overhead per task in the megakernel's
# per-SM dispatch loop. Includes event-tensor wait + body invoke
# + event-tensor notify cycle.
#
# Per bridge #099/#108: measured ~1.0 µs on Blackwell sm_120 in
# the static-schedule path. AMD/Intel may differ; vendors override
# via :meth:`GpuCostModel.scheduling_overhead_us`.
DEFAULT_SCHEDULING_OVERHEAD_US: float = 1.0


# ---------------------------------------------------------------------------
# Probes — compile-time hardware + library detection
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuProbe(Protocol):
    """Detect what's available on the local GPU + vendor's libraries.

    The autotune layer calls into this once per process to populate
    a :class:`compgen.runtime.autotune.BackendChoice`. The vendor
    package returns vendor-shaped data; the Protocol only requires
    the universal subset.
    """

    def is_available(self) -> bool:
        """Cheap probe — is this vendor's runtime reachable on this
        host? CPU-only hosts return False without raising."""
        ...

    def device_arch(self) -> str:
        """Vendor-specific arch tag. NVIDIA returns ``"sm_100"``,
        AMD returns ``"gfx942"``, Intel returns ``"xe_hpc"``. The
        autotune layer doesn't interpret the string — it passes it
        back to the vendor's other adapters."""
        ...

    def supports_clusters(self) -> bool:
        """True when the GPU exposes a multi-block-per-task
        cooperative primitive (NVIDIA cluster-launch on sm_90+,
        AMD has a similar concept on CDNA3+, etc.). The static
        schedule sets ``cluster_dim`` accordingly when this is
        True."""
        ...

    def supports_tensor_cores(self) -> bool:
        """True when the GPU has a tensor-core class MMA unit
        (NVIDIA sm_70+, AMD CDNA, Intel Xe-HPG+). The body emitter
        chooses tensor-core paths when this is True."""
        ...

    def library_paths(self) -> dict[str, str | None]:
        """Vendor-specific library include / .so paths the
        compiler / NVRTC / hipcc needs. Universal modules pass
        these through to the body emitter; they don't interpret
        the keys."""
        ...

    def vendor_extras(self) -> dict[str, Any]:
        """Anything else the vendor wants surfaced for audit. Lands
        in ``BackendChoice.vendor_extras[vendor_id]``. The agent's
        audit query reads this through the universal probe; the
        keys are vendor-defined."""
        ...


# ---------------------------------------------------------------------------
# Body emitter — per-op kernel sources
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuBodyEmitter(Protocol):
    """Emit per-op kernel-body source for the lowering matcher.

    The matcher walks an ``nn.Module``, identifies a pattern
    (Diamond, FFN, ...), and asks the body emitter for sources for
    each op. The emitter chooses tile shape + precision + library
    backend; the matcher only sees :class:`DeviceFunctionSource`.

    Vendor implementations:

    - NVIDIA (Blackwell): cuBLASDx-backed bf16+fp32-acc GEMM at
      64×64×16 + tile-aware elementwise.
    - NVIDIA (Hopper): cuBLASDx fp32 at 32×32×32 (no tcgen05).
    - NVIDIA (Ampere): hand-rolled fmaf at 32×32×32.
    - AMD (CDNA3): rocBLAS-ish path, mfma atoms.
    - CPU x86: fmaf-equivalent C++ AVX-512 loop.

    All of these satisfy this Protocol with the same signatures.
    """

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """Tile-shape hint the matcher uses for divisibility checks
        + body emission. NVIDIA Blackwell returns ``(64, 64, 16)``
        for cuBLASDx; older arches return ``(32, 32, 32)``; CPU
        returns ``(32, 32, 32)`` for cache-line alignment."""
        ...

    def gemm(
        self,
        *,
        b_dim: int,
        k_dim: int,
        n_dim: int,
        n_tiles_per_row: int,
        x_buf: int,
        w_buf: int,
        out_buf: int,
        precision: str,
        tile_m: int,
        tile_n: int,
        tile_k: int,
    ) -> DeviceFunctionSource:
        """Emit a GEMM body for one task. The matcher decides
        which buffer indices the bodies use; the emitter just
        consumes them."""
        ...

    def elementwise(
        self,
        *,
        op: str,  # "relu" | "add" | "gelu" | ...
        total_elems: int,
        in_bufs: tuple[int, ...],
        out_buf: int,
        tile_m: int,
        tile_n: int,
    ) -> DeviceFunctionSource:
        """Emit a tile-aware elementwise body. The Wave 1.7
        coverage-loop pattern (``for p in range(...)``) is the
        responsibility of each vendor's emitter."""
        ...


# ---------------------------------------------------------------------------
# Runtime — JIT compile + load + dispatch
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuRuntime(Protocol):
    """JIT compile a megakernel source, load the module, dispatch.

    Vendor-specific concerns the universal dispatch path pushes
    into here:

    - JIT toolchain (NVRTC, hipcc, ocloc).
    - Module load primitive (``cuModuleLoadData``, ``hipModuleLoadData``,
      ``zeModuleCreate``).
    - Cooperative-launch primitive (``cuLaunchCooperativeKernel``,
      ``hipLaunchCooperativeKernel``).
    - Device-context management.
    """

    def compile_source(
        self,
        *,
        cuda_source: str,  # ← named for legacy; really vendor-agnostic source
        kernel_name: str,
        arch: str,
        extra_options: tuple[str, ...] = (),
        extra_include_paths: tuple[str, ...] = (),
    ) -> Any:
        """Returns a vendor-defined module handle. The universal
        path doesn't introspect — it just hands the handle back to
        :meth:`launch`."""
        ...

    def launch(
        self,
        *,
        module_handle: Any,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        cluster_dim: tuple[int, int, int] | None,
        shared_mem_bytes: int,
        kernel_params: Any,
        cooperative: bool,
    ) -> None:
        """Synchronous launch + completion. Vendor-specific
        cluster_dim handling: NVIDIA uses ``cuLaunchKernelEx`` with
        attributes; AMD/Intel use their respective primitives."""
        ...

    def synchronize(self) -> None:
        """Block until all in-flight launches complete on the
        current device."""
        ...


# ---------------------------------------------------------------------------
# Cost model — vendor-supplied perf numbers for the roofline predictor
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuCostModel(Protocol):
    """Vendor-specific perf coefficients for the universal
    roofline + ETC-vs-eager predictor.

    The predictor (``compgen.kernels.cost.predict_etc_dispatch``)
    is vendor-blind; it asks the cost model for numbers and
    composes them. NVIDIA's per-arch TFLOPS table, AMD's CDNA3
    BF16 throughput, Intel's Xe-HPG numbers — all surface through
    here.
    """

    def peak_tflops_per_sm(self, *, dtype: str, tensor_core: bool) -> float:
        """Per-SM peak throughput at the given dtype. NVIDIA
        Blackwell at bf16 + tensor-cores ≈ 50 TFLOPS/SM."""
        ...

    def sm_count(self) -> int:
        """Number of SMs (or analogous compute units) on the
        device."""
        ...

    def scheduling_overhead_us(self) -> float:
        """Per-task megakernel scheduling cost on this vendor's
        cooperative-launch path. Defaults to
        :data:`DEFAULT_SCHEDULING_OVERHEAD_US` (1.0 µs); vendors
        override when they have measured data."""
        ...

    def eager_launch_overhead_us(self) -> float:
        """One-shot kernel-launch overhead for the vendor's eager
        BLAS library (cuBLAS, rocBLAS, oneMKL). Used as the eager
        side of the ETC-vs-eager comparison."""
        ...


# ---------------------------------------------------------------------------
# Device handle — per-vendor opaque
# ---------------------------------------------------------------------------


@runtime_checkable
class Device(Protocol):
    """Per-GPU handle. Vendor-defined under the hood (NVIDIA's
    CUcontext, AMD's hipCtx_t, Intel's ze_context_handle_t) — the
    Protocol only exposes what every GPU has."""

    def index(self) -> int:
        """Device ordinal (which GPU on a multi-GPU host)."""
        ...

    def name(self) -> str:
        """Human-readable device name (``"NVIDIA RTX PRO 6000 Blackwell"``,
        ``"AMD MI300X"``, ...) for the audit log."""
        ...

    def synchronize(self) -> None:
        """Block until all queued work completes on this device."""
        ...


# ---------------------------------------------------------------------------
# Event timer — for measuring etc_us / eager_us
# ---------------------------------------------------------------------------


@runtime_checkable
class EventTimer(Protocol):
    """Timing primitive for the conformance harness. NVIDIA wraps
    ``cudaEventRecord`` + ``cudaEventElapsedTime``; AMD wraps
    ``hipEventRecord``; CPU wraps ``time.perf_counter_ns``."""

    def __enter__(self) -> EventTimer: ...
    def __exit__(self, *exc: Any) -> bool: ...
    def elapsed_us(self) -> float:
        """Microseconds between ``__enter__`` and ``__exit__``."""
        ...
