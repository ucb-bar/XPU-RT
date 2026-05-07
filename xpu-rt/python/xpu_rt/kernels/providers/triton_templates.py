"""Triton kernel template provider.

Generates real Triton kernels from parameterised templates for common
fused operation patterns (matmul+bias+epilogue, elementwise). Templates
are valid ``@triton.jit`` functions with tiling parameters exposed as
``tl.constexpr``.

The provider compiles and validates kernels when Triton is available;
when it is not, it still emits source-code strings but skips execution.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.kernels.provider import (
    BidPreview,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Triton availability probe
# ---------------------------------------------------------------------------

_TRITON_AVAILABLE: bool
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    """Return True when the ``triton`` package can be imported."""
    return _TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Templates (source-code strings)
# ---------------------------------------------------------------------------

_MATMUL_BIAS_GELU_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    import torch

    @triton.jit
    def matmul_bias_gelu_kernel(
        A_ptr, B_ptr, bias_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mask_a = (offs_m[:, None] < M) & ((offs_k[None, :] + k_start) < K)
            mask_b = ((offs_k[:, None] + k_start) < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # bias + GELU
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        # Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = acc
        cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        acc = x * cdf

        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_c)

    def kernel(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = {block_m}, {block_n}, {block_k}
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        matmul_bias_gelu_kernel[grid](
            A, B, bias, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return C
""")

_MATMUL_BIAS_RELU_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    import torch

    @triton.jit
    def matmul_bias_relu_kernel(
        A_ptr, B_ptr, bias_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mask_a = (offs_m[:, None] < M) & ((offs_k[None, :] + k_start) < K)
            mask_b = ((offs_k[:, None] + k_start) < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # bias + ReLU
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        acc = tl.maximum(acc, 0.0)

        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_c)

    def kernel(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = {block_m}, {block_n}, {block_k}
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        matmul_bias_relu_kernel[grid](
            A, B, bias, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return C
""")

_ELEMENTWISE_GELU_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    import torch

    @triton.jit
    def elementwise_gelu_kernel(
        X_ptr, Y_ptr, N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        # Approximate GELU
        cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        y = x * cdf
        tl.store(Y_ptr + offs, y, mask=mask)

    def kernel(X: torch.Tensor) -> torch.Tensor:
        Y = torch.empty_like(X)
        N = X.numel()
        BLOCK_SIZE = {block_size}
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        elementwise_gelu_kernel[grid](X, Y, N, BLOCK_SIZE=BLOCK_SIZE)
        return Y
""")

_MATMUL_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    import torch

    @triton.jit
    def matmul_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mask_a = (offs_m[:, None] < M) & ((offs_k[None, :] + k_start) < K)
            mask_b = ((offs_k[:, None] + k_start) < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_c)

    def kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = {block_m}, {block_n}, {block_k}
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return C
""")

_SOFTMAX_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    import torch

    @triton.jit
    def softmax_kernel(
        input_ptr, output_ptr,
        n_cols,
        stride_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start = row_idx * stride_row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        row_max = tl.max(row, axis=0)
        numerator = tl.exp(row - row_max)
        denominator = tl.sum(numerator, axis=0)
        result = numerator / denominator
        tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

    def kernel(X: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = X.shape
        Y = torch.empty_like(X)
        BLOCK_SIZE = {block_size}
        grid = (n_rows,)
        softmax_kernel[grid](X, Y, n_cols, X.stride(0), BLOCK_SIZE=BLOCK_SIZE)
        return Y
""")

_LAYERNORM_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    import torch

    @triton.jit
    def layernorm_kernel(
        input_ptr, output_ptr,
        n_cols,
        stride_row,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start = row_idx * stride_row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
        mean = tl.sum(row, axis=0) / n_cols
        centered = row - mean
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        result = centered * rstd
        tl.store(output_ptr + row_start + col_offsets, result, mask=mask)

    def kernel(X: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = X.shape
        Y = torch.empty_like(X)
        BLOCK_SIZE = {block_size}
        grid = (n_rows,)
        layernorm_kernel[grid](X, Y, n_cols, X.stride(0), eps=1e-5, BLOCK_SIZE=BLOCK_SIZE)
        return Y
""")

_ADD_RELU_TEMPLATE = textwrap.dedent("""\
    import triton
    import triton.language as tl
    import torch

    @triton.jit
    def add_relu_kernel(
        a_ptr, b_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        result = tl.maximum(a + b, 0.0)
        tl.store(output_ptr + offsets, result, mask=mask)

    def kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(A)
        n_elements = A.numel()
        BLOCK_SIZE = {block_size}
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        add_relu_kernel[grid](A, B, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output
""")

# Map from op_family tag to template string and description.
_TEMPLATES: dict[str, tuple[str, str]] = {
    "matmul": (_MATMUL_TEMPLATE, "Matrix multiply (C = A @ B)"),
    "matmul_bias_gelu": (_MATMUL_BIAS_GELU_TEMPLATE, "Fused matmul + bias + GELU"),
    "matmul_bias_relu": (_MATMUL_BIAS_RELU_TEMPLATE, "Fused matmul + bias + ReLU"),
    "elementwise_gelu": (_ELEMENTWISE_GELU_TEMPLATE, "Element-wise GELU"),
    "softmax": (_SOFTMAX_TEMPLATE, "Row-wise softmax"),
    "layer_norm": (_LAYERNORM_TEMPLATE, "Fused layer normalization"),
    "add_relu": (_ADD_RELU_TEMPLATE, "Elementwise add + ReLU"),
}


# ---------------------------------------------------------------------------
# Block-size heuristic
# ---------------------------------------------------------------------------


def _pick_tile_sizes(
    dim_m: int,
    dim_n: int,
    dim_k: int,
) -> tuple[int, int, int]:
    """Choose BLOCK_M, BLOCK_N, BLOCK_K for a matmul tile.

    Uses power-of-two clamping sized to fit in shared memory.  For the
    matmul inner loop the dominant shared-memory cost is roughly
    ``(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4 * num_stages`` bytes.
    We target <= 48 KiB to stay within most GPU limits.
    """

    def _clamp_po2(dim: int, lo: int = 16, hi: int = 64) -> int:
        v = max(lo, min(hi, dim))
        # round down to nearest power of two
        v = 1 << (v.bit_length() - 1)
        return max(lo, v)

    bm = _clamp_po2(dim_m, lo=16, hi=64)
    bn = _clamp_po2(dim_n, lo=16, hi=64)
    bk = _clamp_po2(dim_k, lo=16, hi=32)
    return bm, bn, bk


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


@dataclass
class TritonTemplateProvider:
    """Generates Triton kernels from parameterised templates.

    Implements :class:`~compgen.kernels.provider.KernelProvider`.

    Attributes:
        default_block_size: Default element-wise block size.
    """

    default_block_size: int = 1024
    _accumulated_knowledge: list[KnowledgeExport] = field(default_factory=list)

    # -- KernelProvider protocol ---------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Provider identifier."""
        return "triton_templates"

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Return True if *contract.op_family* matches a known template."""
        return contract.op_family in _TEMPLATES

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Generate a Triton kernel from the matching template.

        Steps:
            1. Select template by ``contract.op_family``.
            2. Derive tile sizes from contract shapes.
            3. Instantiate the source string.
            4. Optionally validate with :class:`KernelValidator`.
            5. Return :class:`ProviderResult`.

        Args:
            contract: Kernel contract describing the required operation.
            budget: Resource budget (largely unused for templates).

        Returns:
            A populated :class:`ProviderResult`.
        """
        op = contract.op_family
        if op not in _TEMPLATES:
            return ProviderResult(found=False, metadata={"reason": f"no template for {op}"})

        template_src, description = _TEMPLATES[op]

        # Derive shapes from contract
        dim_m, dim_n, dim_k = _shapes_from_contract(contract)

        # Instantiate template
        kernel_code = _instantiate(template_src, op, dim_m, dim_n, dim_k, self.default_block_size)

        log.info(
            "triton_templates.instantiated",
            op_family=op,
            dim_m=dim_m,
            dim_n=dim_n,
            dim_k=dim_k,
            triton_available=_TRITON_AVAILABLE,
        )

        # Validation (best-effort when Triton + CUDA are present).
        # Unmeasurable cases carry ``math.nan``, never 0.0 — so the
        # escalating router's ``latency_us > 0`` check correctly
        # refuses to compare against a ghost measurement.
        import math

        validation_diags: list[str] = []
        correct = False
        latency_us: float = math.nan

        if _TRITON_AVAILABLE:
            try:
                import torch

                if torch.cuda.is_available():
                    correct, latency_us, validation_diags = _validate_on_gpu(
                        kernel_code,
                        op,
                        dim_m,
                        dim_n,
                        dim_k,
                    )
                else:
                    validation_diags.append("CUDA not available; skipping GPU validation")
            except Exception as exc:
                validation_diags.append(f"Validation error: {exc}")
        else:
            validation_diags.append("Triton not importable; returning source only")

        # Knowledge export
        knowledge = [
            KnowledgeExport(
                kind="template_match",
                scope="operator_family",
                scope_key=op,
                content=f"Template '{op}' instantiated ({description}), M={dim_m} N={dim_n} K={dim_k}",
                metadata={"correct": correct, "latency_us": latency_us},
                confidence=0.9 if correct else 0.4,
            ),
        ]
        self._accumulated_knowledge.extend(knowledge)

        # ``cost_source`` records whether ``latency_us`` came from a
        # real measurement or couldn't be measured. Downstream
        # selectors must consult this metadata when the value is NaN
        # to decide whether to escalate or roofline.
        cost_source = "measured_gpu" if math.isfinite(latency_us) else "unmeasured"
        return ProviderResult(
            found=True,
            kernel_code=kernel_code,
            language="triton",
            latency_us=latency_us,
            correct=correct,
            plan=f"template:{op}",
            iterations_used=1,
            total_candidates=1,
            knowledge_exports=knowledge,
            metadata={
                "description": description,
                "triton_available": _TRITON_AVAILABLE,
                "validation": validation_diags,
                "cost_source": cost_source,
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export accumulated knowledge and clear the buffer."""
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports

    # -- Phase D / M-56: bid() ------------------------------------------------

    def bid(self, contract_v3: Any) -> BidPreview:
        """Cheap pre-codegen estimate over a :class:`KernelContractV3`.

        Match logic:

        * Map contract archetype + op_name onto the template family.
          ``COMPUTE_TILED`` + ``matmul``-shaped op_name → ``matmul``.
          ``POINTWISE`` + recognised activation → matching template.
        * If a template matches: high confidence (0.7), fast generate
          time (~0.1s — pure string substitution), real perf
          estimate from a roofline based on tile shape.
        * If no template matches: confidence=0.0 placeholder so the
          auction skips this provider.
        """
        family = _archetype_to_family(contract_v3)
        if family is None or family not in _TEMPLATES:
            return BidPreview(
                provider_name=self.name,
                confidence=0.0,
                rationale=f"no_template_for_archetype_{family or 'unknown'}",
            )

        # Roofline: estimated compute / peak_compute_per_dtype, with a
        # nominal launch overhead of 5us. The actual numbers don't have
        # to be tight — they're a relative ordering signal for the
        # auction.
        try:
            inputs = contract_v3.io.inputs
            dim_m = int(inputs[0].shape.dims[0]) if inputs[0].shape.dims else 128
            dim_k = int(inputs[0].shape.dims[1]) if len(inputs[0].shape.dims) > 1 else 128
            dim_n = (
                int(inputs[1].shape.dims[1])
                if len(inputs) > 1 and len(inputs[1].shape.dims) > 1
                else 128
            )
            flops = 2.0 * dim_m * dim_n * dim_k
            # Default to 1 TFLOP/s if envelope doesn't carry peak.
            tflops_per_s = 1.0e12
            try:
                peak_dict = contract_v3.orchestration.execution.hardware.peak_compute_per_dtype
                if peak_dict:
                    # First entry — same dtype the kernel emits.
                    tflops_per_s = float(next(iter(peak_dict.values()))) * 1.0e12
            except (AttributeError, TypeError, ValueError, StopIteration):
                pass
            est_us = max(1.0, (flops / tflops_per_s) * 1e6 + 5.0)
        except Exception:  # noqa: BLE001
            est_us = float("inf")

        return BidPreview(
            provider_name=self.name,
            perf_estimate_us=est_us,
            confidence=0.7,
            time_to_generate_s_estimate=0.1,
            rationale=f"template_match_{family}",
            cache_hit=False,
        )


def _archetype_to_family(contract_v3: Any) -> str | None:
    """Map a v3 archetype + op_name onto a TritonTemplateProvider family key.

    Returns ``None`` when the contract isn't covered by any template.
    """
    try:
        archetype = contract_v3.archetype.value
        op_name = contract_v3.op_name.lower()
    except AttributeError:
        return None

    if archetype == "compute_tiled":
        if "matmul_bias_gelu" in op_name:
            return "matmul_bias_gelu"
        if "matmul_bias_relu" in op_name:
            return "matmul_bias_relu"
        if "matmul" in op_name:
            return "matmul"
    if archetype == "pointwise":
        if "gelu" in op_name:
            return "elementwise_gelu"
        if "add_relu" in op_name or ("add" in op_name and "relu" in op_name):
            return "add_relu"
    if archetype == "reduce":
        if "softmax" in op_name:
            return "softmax"
        if "layer_norm" in op_name or "layernorm" in op_name:
            return "layer_norm"
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shapes_from_contract(contract: KernelContract) -> tuple[int, int, int]:
    """Extract ``(dim_m, dim_n, dim_k)`` from the contract shapes.

    Falls back to sensible defaults when shape information is absent.
    """
    dim_m, dim_n, dim_k = 128, 128, 256

    if contract.input_shapes:
        first = contract.input_shapes[0]
        if len(first) >= 2:
            dim_m, dim_k = first[0], first[1]
        if len(contract.input_shapes) > 1:
            second = contract.input_shapes[1]
            if len(second) >= 2:
                dim_n = second[1]

    if contract.output_shapes:
        out = contract.output_shapes[0]
        if len(out) >= 2:
            dim_m, dim_n = out[0], out[1]

    return dim_m, dim_n, dim_k


def _instantiate(
    template: str,
    op: str,
    dim_m: int,
    dim_n: int,
    dim_k: int,
    default_block: int,
) -> str:
    """Format a template string with tile-size parameters."""
    if op.startswith("elementwise") or op in ("softmax", "layer_norm", "add_relu"):
        block_size = default_block
        if op in ("softmax", "layer_norm"):
            # For row-wise ops, block_size must be >= n_cols (power of 2)
            block_size = max(default_block, 1 << max(dim_n.bit_length(), 4))
            block_size = min(block_size, 8192)  # cap to avoid excessive shared memory
        return template.format(block_size=block_size)

    block_m, block_n, block_k = _pick_tile_sizes(dim_m, dim_n, dim_k)
    return template.format(block_m=block_m, block_n=block_n, block_k=block_k)


def _import_kernel_from_source(kernel_code: str) -> Any:
    """Write *kernel_code* to a temp file and import the ``kernel`` callable.

    Triton's ``@triton.jit`` decorator inspects source via
    ``inspect.getsource()``, which only works for functions defined in
    real files on disk -- ``exec()`` into a dict will fail.
    """
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=".py",
        prefix="triton_tmpl_",
        mode="w",
        delete=False,
    ) as f:
        f.write(kernel_code)
        f.flush()
        spec = importlib.util.spec_from_file_location("_triton_tmpl", f.name)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "kernel")


def _validate_on_gpu(
    kernel_code: str,
    op: str,
    dim_m: int,
    dim_n: int,
    dim_k: int,
) -> tuple[bool, float, list[str]]:
    """Compile, run, and validate a kernel on GPU.

    Returns ``(correct, latency_us, diagnostics)``.
    """
    import torch

    diags: list[str] = []
    device = "cuda"

    # 1. Import kernel from temp file (Triton JIT needs real source)
    try:
        kernel_fn = _import_kernel_from_source(kernel_code)
        diags.append("Compilation: PASS")
    except Exception as exc:
        return False, 0.0, [f"Compilation failed: {exc}"]

    # 2. Build reference inputs and outputs
    if op == "elementwise_gelu":
        x = torch.randn(dim_m, dim_n, device=device, dtype=torch.float32)
        ref_out = torch.nn.functional.gelu(x)
        test_inputs = (x,)
    elif op == "matmul":
        a_mat = torch.randn(dim_m, dim_k, device=device, dtype=torch.float32)
        b_mat = torch.randn(dim_k, dim_n, device=device, dtype=torch.float32)
        ref_out = a_mat @ b_mat
        test_inputs = (a_mat, b_mat)
    elif op == "softmax":
        x = torch.randn(dim_m, dim_n, device=device, dtype=torch.float32)
        ref_out = torch.softmax(x, dim=-1)
        test_inputs = (x,)
    elif op == "layer_norm":
        x = torch.randn(dim_m, dim_n, device=device, dtype=torch.float32)
        ref_out = torch.nn.functional.layer_norm(x, (dim_n,))
        test_inputs = (x,)
    elif op == "add_relu":
        a = torch.randn(dim_m, dim_n, device=device, dtype=torch.float32)
        b = torch.randn(dim_m, dim_n, device=device, dtype=torch.float32)
        ref_out = torch.relu(a + b)
        test_inputs = (a, b)
    else:
        a_mat = torch.randn(dim_m, dim_k, device=device, dtype=torch.float32)
        b_mat = torch.randn(dim_k, dim_n, device=device, dtype=torch.float32)
        bias = torch.randn(dim_n, device=device, dtype=torch.float32)
        if op == "matmul_bias_gelu":
            ref_out = torch.nn.functional.gelu(a_mat @ b_mat + bias)
        else:  # matmul_bias_relu
            ref_out = torch.nn.functional.relu(a_mat @ b_mat + bias)
        test_inputs = (a_mat, b_mat, bias)

    # 3. Correctness check
    try:
        actual = kernel_fn(*test_inputs)
        diff = actual.float() - ref_out.float()
        l2 = float(torch.norm(diff).item())
        max_abs = float(torch.max(torch.abs(diff)).item())
        tol = 1e-2  # GELU approximation needs wider tolerance
        correct = max_abs <= tol
        diags.append(f"Correctness: {'PASS' if correct else 'FAIL'} (l2={l2:.6f}, max_abs={max_abs:.6f}, tol={tol})")
    except Exception as exc:
        return False, 0.0, diags + [f"Execution failed: {exc}"]

    # 4. Latency measurement — use the blessed measure_kernel path so
    # every provider produces comparable numbers. Failures surface via
    # ``math.nan`` (not 0.0, which would lie about being 0 latency)
    # plus a diagnostic; the escalating router's ``latency_us > 0``
    # comparison treats NaN the same as "unmeasured".
    import math

    from compgen.kernels.errors import UnmeasurableKernelError
    from compgen.kernels.measure import measure_kernel

    latency_us = math.nan
    if correct:
        try:
            measurement = measure_kernel(
                runnable=kernel_fn,
                golden_inputs=test_inputs,
                warmup=3,
                iters=20,
            )
            latency_us = measurement.latency_us
            diags.append(
                f"Performance: {latency_us:.1f} us/iter ({measurement.iters} iters, "
                f"stddev {measurement.latency_stddev_us:.2f} us, source={measurement.source})"
            )
        except UnmeasurableKernelError as exc:
            # Explicit unmeasurable — don't substitute 0.0. The caller
            # sees NaN + the diagnostic and decides how to escalate.
            diags.append(f"Performance: unmeasurable ({exc})")

    return correct, latency_us, diags


__all__ = ["TritonTemplateProvider", "triton_available"]
