"""Payload IR → cuda_tile MLIR lowering (single-tile FFN reference).

Emits a ``cuda_tile.module`` containing a single entry that performs
matmul-relu-matmul (the FFN forward pattern). The structural shape
matches bridge #144's bytecode-validated template:

* Rank-0 ``tile<ptr<f32>>`` entry args.
* ``make_tensor_view`` + ``make_partition_view`` per pointer.
* ``load_view_tko`` to materialize 2D tiles.
* ``mmaf`` + ``maxf`` + ``mmaf`` for the FFN body.
* ``store_view_tko`` to write the output.

Default shapes ``(M=8, K=16, N=32, M_out=16)`` are the smallest valid
``mmaf`` combo from the cuda-tile test suite. Callers override via
``options={"shapes": {"M": ..., "K": ..., "N": ..., "M_out": ...}}``.

The lowering is **deterministic Python string formatting** — no LLM,
no toolchain. That means it can be regression-unit-tested without
``cuda-tile-translate`` on PATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.extensions.vendor_dialect.adapter import LoweringResult
from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor
from compgen.kernels.provider import KernelProvider


@dataclass(frozen=True)
class FfnShapes:
    """Tile shapes for the FFN single-tile template."""

    M: int = 8
    K: int = 16
    N: int = 32
    M_out: int = 16

    def validate(self) -> None:
        for name, val in (("M", self.M), ("K", self.K), ("N", self.N), ("M_out", self.M_out)):
            if val <= 0:
                raise ValueError(f"FfnShapes.{name} must be positive, got {val}")


_DEFAULT_SHAPES = FfnShapes()


def _shapes_from_options(options: dict[str, Any]) -> FfnShapes:
    raw = options.get("shapes")
    if raw is None:
        return _DEFAULT_SHAPES
    if not isinstance(raw, dict):
        raise TypeError(f"options['shapes'] must be a dict, got {type(raw).__name__!r}")
    s = FfnShapes(
        M=int(raw.get("M", _DEFAULT_SHAPES.M)),
        K=int(raw.get("K", _DEFAULT_SHAPES.K)),
        N=int(raw.get("N", _DEFAULT_SHAPES.N)),
        M_out=int(raw.get("M_out", _DEFAULT_SHAPES.M_out)),
    )
    s.validate()
    return s


def emit_ffn_single_tile_mlir(shapes: FfnShapes = _DEFAULT_SHAPES) -> str:
    """Render the FFN single-tile MLIR text for the given tile shapes.

    The output is deterministic for fixed shapes. The structure mirrors
    the bridge #144 template that ``cuda-tile-translate`` accepted and
    bytecode-roundtripped successfully.
    """
    shapes.validate()
    M, K, N, M_out = shapes.M, shapes.K, shapes.N, shapes.M_out

    return f"""cuda_tile.module @ffn_kernels {{
  cuda_tile.entry @ffn_matmul_relu_matmul(
      %x_ptr:      !cuda_tile.tile<!cuda_tile.ptr<f32>>,
      %w_up_ptr:   !cuda_tile.tile<!cuda_tile.ptr<f32>>,
      %w_down_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
      %y_ptr:      !cuda_tile.tile<!cuda_tile.ptr<f32>>) {{
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>

    // Materialize x[{M}x{K}].
    %x_tv = cuda_tile.make_tensor_view %x_ptr,
        shape=[{M}, {K}], strides=[{K}, 1]
        : !cuda_tile.tensor_view<{M}x{K}xf32, strides=[{K},1]>
    %x_pv = cuda_tile.make_partition_view %x_tv, tile=({M}, {K})
        : !cuda_tile.partition_view<tile=({M}x{K}), tensor_view<{M}x{K}xf32>>
    %x_tile = cuda_tile.load_view_tko %x_pv[%c0, %c0]
        : !cuda_tile.tile<{M}x{K}xf32>

    // Materialize w_up[{K}x{N}].
    %w_up_tv = cuda_tile.make_tensor_view %w_up_ptr,
        shape=[{K}, {N}], strides=[{N}, 1]
        : !cuda_tile.tensor_view<{K}x{N}xf32, strides=[{N},1]>
    %w_up_pv = cuda_tile.make_partition_view %w_up_tv, tile=({K}, {N})
        : !cuda_tile.partition_view<tile=({K}x{N}), tensor_view<{K}x{N}xf32>>
    %w_up_tile = cuda_tile.load_view_tko %w_up_pv[%c0, %c0]
        : !cuda_tile.tile<{K}x{N}xf32>

    // y_up = x @ w_up
    %acc0 = cuda_tile.constant <f32: 0.0>
        : !cuda_tile.tile<{M}x{N}xf32>
    %y_up = cuda_tile.mmaf %x_tile, %w_up_tile, %acc0
        : !cuda_tile.tile<{M}x{K}xf32>, !cuda_tile.tile<{K}x{N}xf32>, !cuda_tile.tile<{M}x{N}xf32>

    // y_relu = max(y_up, 0)  ← fused relu in the MMA tail
    %zero = cuda_tile.constant <f32: 0.0>
        : !cuda_tile.tile<{M}x{N}xf32>
    %y_relu = cuda_tile.maxf %y_up, %zero
        : !cuda_tile.tile<{M}x{N}xf32>

    // Materialize w_down[{N}x{M_out}].
    %w_down_tv = cuda_tile.make_tensor_view %w_down_ptr,
        shape=[{N}, {M_out}], strides=[{M_out}, 1]
        : !cuda_tile.tensor_view<{N}x{M_out}xf32, strides=[{M_out},1]>
    %w_down_pv = cuda_tile.make_partition_view %w_down_tv, tile=({N}, {M_out})
        : !cuda_tile.partition_view<tile=({N}x{M_out}), tensor_view<{N}x{M_out}xf32>>
    %w_down_tile = cuda_tile.load_view_tko %w_down_pv[%c0, %c0]
        : !cuda_tile.tile<{N}x{M_out}xf32>

    // y = y_relu @ w_down
    %acc1 = cuda_tile.constant <f32: 0.0>
        : !cuda_tile.tile<{M}x{M_out}xf32>
    %y = cuda_tile.mmaf %y_relu, %w_down_tile, %acc1
        : !cuda_tile.tile<{M}x{N}xf32>, !cuda_tile.tile<{N}x{M_out}xf32>, !cuda_tile.tile<{M}x{M_out}xf32>

    // Store y[{M}x{M_out}].
    %y_tv = cuda_tile.make_tensor_view %y_ptr,
        shape=[{M}, {M_out}], strides=[{M_out}, 1]
        : !cuda_tile.tensor_view<{M}x{M_out}xf32, strides=[{M_out},1]>
    %y_pv = cuda_tile.make_partition_view %y_tv, tile=({M}, {M_out})
        : !cuda_tile.partition_view<tile=({M}x{M_out}), tensor_view<{M}x{M_out}xf32>>
    cuda_tile.store_view_tko %y, %y_pv[%c0, %c0]
        : !cuda_tile.tile<{M}x{M_out}xf32>

    cuda_tile.return
  }}
}}
"""


_OPS_USED = (
    "make_tensor_view",
    "make_partition_view",
    "load_view_tko",
    "mmaf",
    "maxf",
    "constant",
    "store_view_tko",
)


def lower_to_cuda_tile(
    payload_mlir: str,
    *,
    descriptor: VendorDialectDescriptor,
    kernel_provider: KernelProvider | None,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> LoweringResult:
    """Lower Payload IR to a single-tile FFN cuda_tile module.

    The current reference adapter ignores ``payload_mlir`` content and
    always emits the FFN matmul-relu-matmul template — it's a *reference*
    for the dispatch contract, not a general-purpose lowering. Real
    lowerings (the bwell-side ``compgen_cuda_tile`` package) inspect
    the payload to pick op-family templates per region.

    The lowering is recorded in metadata so consumers can audit which
    op-families and shapes were used.
    """
    del kernel_provider, descriptor  # reference adapter has no kernel provider
    opts = options or {}
    shapes = _shapes_from_options(opts)
    vendor_mlir = emit_ffn_single_tile_mlir(shapes)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = output_dir / "ffn_kernel.mlir"
    mlir_path.write_text(vendor_mlir)

    return LoweringResult(
        vendor_mlir=vendor_mlir,
        kernels={},
        metadata={
            "lowering_mode": "ffn_matmul_relu_template",
            "ops_used": list(_OPS_USED),
            "shapes": {
                "M": shapes.M,
                "K": shapes.K,
                "N": shapes.N,
                "M_out": shapes.M_out,
            },
            "vendor_mlir_path": str(mlir_path),
            "in_tree_reference": True,
            "validated_against": "bridge#144",
            "payload_mlir_consumed": False,
        },
    )


__all__ = [
    "FfnShapes",
    "emit_ffn_single_tile_mlir",
    "lower_to_cuda_tile",
]
