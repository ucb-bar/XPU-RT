"""Triton kernel emitter skeleton.

Walks a compiled xDSL module and, for each op tagged with
``compgen.library_dispatch = "triton"`` (set by
``match_library_call`` in Wave 5), emits a Triton kernel source
file + a launch-grid record.

The emitter is **source-only** today: no Triton compile + no CUDA
launch. This keeps the skeleton runnable on CPU-only CI while the
actual compile/execute hookup lives behind a gate that fires when
``triton`` is importable + CUDA is available.

Output artifact layout:

    <out_dir>/
        kernels/
            matmul_0.py           # Triton source
            softmax_3.py
        launch_grid.json          # {op_name: (grid_dim, block_size)}
        emission_manifest.json    # {op_name: {source_path, callee, attrs}}

Usage::

    from compgen.runtime.triton_emitter import emit_triton_kernels
    report = emit_triton_kernels(module, out_dir=Path("/tmp/compgen_triton"))
    assert report.kernels_emitted >= 1
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg import MatmulOp

log = structlog.get_logger()


@dataclass
class TritonEmitterReport:
    kernels_emitted: int = 0
    skipped_no_dispatch_tag: int = 0
    skipped_unsupported_op: int = 0
    manifest: dict[str, dict[str, str]] = field(default_factory=dict)
    out_dir: Path | None = None


# --- templates --------------------------------------------------------------


_MATMUL_TEMPLATE = textwrap.dedent(
    """\
    import triton
    import triton.language as tl


    @triton.jit
    def {name}(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 64,
        BLOCK_K: tl.constexpr = 32,
    ):
        '''Auto-emitted by CompGen's triton_emitter for ``linalg.matmul``.'''
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc, mask=mask,
        )
    """
)


_SOFTMAX_TEMPLATE = textwrap.dedent(
    """\
    import triton
    import triton.language as tl


    @triton.jit
    def {name}(
        input_ptr, output_ptr,
        n_rows, n_cols,
        input_row_stride, output_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        '''Auto-emitted softmax (row-wise, last axis).'''
        row_idx = tl.program_id(0)
        row_start = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))
        x_stable = x - tl.max(x, axis=0)
        exp_x = tl.exp(x_stable)
        denom = tl.sum(exp_x, axis=0)
        y = exp_x / denom
        tl.store(
            output_ptr + row_idx * output_row_stride + col_offsets,
            y, mask=mask,
        )
    """
)


# --- emitter ---------------------------------------------------------------


def _safe_kernel_name(op_kind: str, index: int) -> str:
    return f"compgen_{op_kind}_{index}"


def _library_dispatch(op: Any) -> str | None:
    attr = op.attributes.get("compgen.library_dispatch")
    return attr.data if isinstance(attr, StringAttr) else None


def emit_triton_kernels(
    module: Any,
    *,
    out_dir: str | Path,
) -> TritonEmitterReport:
    """Walk ``module`` and write Triton source files for every
    Triton-tagged op.

    Returns a :class:`TritonEmitterReport` recording what got
    emitted. Idempotent per ``out_dir`` -- re-emitting overwrites
    the source files.
    """
    out_path = Path(out_dir)
    kernels_dir = out_path / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    report = TritonEmitterReport(out_dir=out_path)
    manifest: dict[str, dict[str, str]] = {}
    matmul_count = 0
    softmax_count = 0

    for op in module.walk():
        lib = _library_dispatch(op)
        if lib != "triton":
            report.skipped_no_dispatch_tag += 1
            continue

        if isinstance(op, MatmulOp):
            name = _safe_kernel_name("matmul", matmul_count)
            matmul_count += 1
            src = _MATMUL_TEMPLATE.format(name=name)
            path = kernels_dir / f"{name}.py"
            path.write_text(src)
            manifest[name] = {
                "op_name": op.name,
                "kernel": name,
                "source_path": str(path),
                "template": "matmul",
            }
            report.kernels_emitted += 1
            continue

        # compgen.linalg_ext.softmax already carries a Triton source
        # on the ``compgen.triton_source`` attr when
        # ``fuse_softmax_to_triton`` has run. Emit whatever's there.
        if op.name == "compgen.linalg_ext.softmax":
            source_attr = op.attributes.get("compgen.triton_source")
            kname_attr = op.attributes.get("compgen.triton_kernel_call")
            if isinstance(source_attr, StringAttr) and isinstance(kname_attr, StringAttr):
                name = kname_attr.data
                path = kernels_dir / f"{name}.py"
                path.write_text(source_attr.data)
            else:
                name = _safe_kernel_name("softmax", softmax_count)
                softmax_count += 1
                src = _SOFTMAX_TEMPLATE.format(name=name)
                path = kernels_dir / f"{name}.py"
                path.write_text(src)
            manifest[name] = {
                "op_name": op.name,
                "kernel": name,
                "source_path": str(path),
                "template": "softmax",
            }
            report.kernels_emitted += 1
            continue

        report.skipped_unsupported_op += 1

    # Write the manifest.
    (out_path / "emission_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    report.manifest = manifest
    log.info(
        "triton_emitter.done",
        kernels_emitted=report.kernels_emitted,
        out_dir=str(out_path),
    )
    return report


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = [
    "TritonEmitterReport",
    "emit_triton_kernels",
    "triton_available",
]
