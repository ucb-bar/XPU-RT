"""Exo seed proc generation from kernel specifications.

Translates operation names and shapes into unscheduled Exo @proc
definitions that serve as starting points for schedule search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class ExoSeedProc:
    """An unscheduled Exo proc definition.

    Attributes:
        name: Proc function name.
        proc_source: Valid Exo @proc Python source.
        c_skeleton: Corresponding C skeleton.
        input_types: Exo type strings for inputs.
        output_types: Exo type strings for outputs.
    """

    name: str
    proc_source: str
    c_skeleton: str
    input_types: list[str]
    output_types: list[str]


def generate_seed_proc(
    op_name: str,
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    dtype: str = "f32",
) -> ExoSeedProc | None:
    """Generate an unscheduled Exo proc from an operation specification.

    Args:
        op_name: Operation name (e.g., "matmul", "conv2d").
        input_shapes: Input tensor shapes.
        output_shapes: Output tensor shapes.
        dtype: Data type string.

    Returns:
        ExoSeedProc or None if op not supported.
    """
    generators: dict[str, Any] = {
        "matmul": _gen_matmul,
        "linalg.matmul": _gen_matmul,
        "conv2d": _gen_conv2d,
        "linalg.conv_2d_nhwc_hwcf": _gen_conv2d,
        "reduction": _gen_reduction,
        "elementwise": _gen_elementwise,
    }

    gen = generators.get(op_name)
    if gen is None:
        log.warning("exo.unsupported_op", op_name=op_name)
        return None

    return gen(input_shapes, output_shapes, dtype)


def _exo_dtype(dtype: str) -> str:
    """Map CompGen dtype string to Exo type."""
    mapping = {"f32": "f32", "f16": "f16", "f64": "f64", "i32": "i32", "i8": "i8"}
    return mapping.get(dtype, "f32")


def _gen_matmul(
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    dtype: str,
) -> ExoSeedProc:
    """Generate seed proc for matrix multiplication."""
    # Default shapes if not provided
    if len(input_shapes) >= 2:
        dim_m = input_shapes[0][0] if len(input_shapes[0]) >= 1 else 128
        dim_k = input_shapes[0][1] if len(input_shapes[0]) >= 2 else 128
        dim_n = input_shapes[1][1] if len(input_shapes[1]) >= 2 else 128
    else:
        dim_m, dim_k, dim_n = 128, 128, 128

    exo_t = _exo_dtype(dtype)
    proc_name = f"matmul_{dim_m}x{dim_k}x{dim_n}"

    proc_source = (
        f"@proc\n"
        f"def {proc_name}(\n"
        f"    M: size, K: size, N: size,\n"
        f"    A: {exo_t}[M, K] @ DRAM,\n"
        f"    B: {exo_t}[K, N] @ DRAM,\n"
        f"    C: {exo_t}[M, N] @ DRAM,\n"
        f"):\n"
        f"    for i in seq(0, M):\n"
        f"        for j in seq(0, N):\n"
        f"            for k in seq(0, K):\n"
        f"                C[i, j] += A[i, k] * B[k, j]\n"
    )

    c_skeleton = (
        f"// Exo-generated matmul: {dim_m}x{dim_k}x{dim_n} ({exo_t})\n"
        f"void {proc_name}(int M, int K, int N,\n"
        f"    const float* A, const float* B, float* C) {{\n"
        f"  for (int i = 0; i < M; i++)\n"
        f"    for (int j = 0; j < N; j++)\n"
        f"      for (int k = 0; k < K; k++)\n"
        f"        C[i*N + j] += A[i*K + k] * B[k*N + j];\n"
        f"}}\n"
    )

    return ExoSeedProc(
        name=proc_name,
        proc_source=proc_source,
        c_skeleton=c_skeleton,
        input_types=[f"{exo_t}[{dim_m},{dim_k}]", f"{exo_t}[{dim_k},{dim_n}]"],
        output_types=[f"{exo_t}[{dim_m},{dim_n}]"],
    )


def _gen_conv2d(
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    dtype: str,
) -> ExoSeedProc:
    """Generate seed proc for 2D convolution."""
    exo_t = _exo_dtype(dtype)
    proc_name = "conv2d_seed"

    proc_source = (
        f"@proc\n"
        f"def {proc_name}(\n"
        f"    N: size, H: size, W: size, C_in: size, C_out: size,\n"
        f"    KH: size, KW: size,\n"
        f"    inp: {exo_t}[N, H, W, C_in] @ DRAM,\n"
        f"    weights: {exo_t}[KH, KW, C_in, C_out] @ DRAM,\n"
        f"    out: {exo_t}[N, H-KH+1, W-KW+1, C_out] @ DRAM,\n"
        f"):\n"
        f"    for n in seq(0, N):\n"
        f"        for oh in seq(0, H-KH+1):\n"
        f"            for ow in seq(0, W-KW+1):\n"
        f"                for co in seq(0, C_out):\n"
        f"                    for kh in seq(0, KH):\n"
        f"                        for kw in seq(0, KW):\n"
        f"                            for ci in seq(0, C_in):\n"
        f"                                out[n,oh,ow,co] += inp[n,oh+kh,ow+kw,ci] * weights[kh,kw,ci,co]\n"
    )

    c_skeleton = (
        f"// Exo-generated conv2d ({exo_t})\n"
        f"void {proc_name}(...) {{ /* conv2d loop nest */ }}\n"
    )

    return ExoSeedProc(
        name=proc_name,
        proc_source=proc_source,
        c_skeleton=c_skeleton,
        input_types=[f"{exo_t}[N,H,W,C_in]", f"{exo_t}[KH,KW,C_in,C_out]"],
        output_types=[f"{exo_t}[N,OH,OW,C_out]"],
    )


def _gen_reduction(
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    dtype: str,
) -> ExoSeedProc:
    """Generate seed proc for reduction."""
    exo_t = _exo_dtype(dtype)
    dim_n = input_shapes[0][0] if input_shapes else 1024

    proc_source = (
        f"@proc\n"
        f"def reduce_sum(N: size, x: {exo_t}[N] @ DRAM, result: {exo_t}[1] @ DRAM):\n"
        f"    for i in seq(0, N):\n"
        f"        result[0] += x[i]\n"
    )

    c_skeleton = (
        f"// Exo-generated reduction ({exo_t})\n"
        f"void reduce_sum(int N, const float* x, float* result) {{ ... }}\n"
    )

    return ExoSeedProc(
        name="reduce_sum",
        proc_source=proc_source,
        c_skeleton=c_skeleton,
        input_types=[f"{exo_t}[{dim_n}]"],
        output_types=[f"{exo_t}[1]"],
    )


def _gen_elementwise(
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    dtype: str,
) -> ExoSeedProc:
    """Generate seed proc for elementwise operation."""
    exo_t = _exo_dtype(dtype)
    dim_n = input_shapes[0][0] if input_shapes else 1024

    proc_source = (
        f"@proc\n"
        f"def elementwise_add(N: size, a: {exo_t}[N] @ DRAM, b: {exo_t}[N] @ DRAM, c: {exo_t}[N] @ DRAM):\n"
        f"    for i in seq(0, N):\n"
        f"        c[i] = a[i] + b[i]\n"
    )

    c_skeleton = (
        f"// Exo-generated elementwise ({exo_t})\n"
        f"void elementwise_add(int N, ...) {{ ... }}\n"
    )

    return ExoSeedProc(
        name="elementwise_add",
        proc_source=proc_source,
        c_skeleton=c_skeleton,
        input_types=[f"{exo_t}[{dim_n}]", f"{exo_t}[{dim_n}]"],
        output_types=[f"{exo_t}[{dim_n}]"],
    )


__all__ = ["ExoSeedProc", "generate_seed_proc"]
