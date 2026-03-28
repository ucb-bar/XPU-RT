"""Built-in ukernel library.

Provides a default set of ukernel declarations, match constraints, and
bodies that CompGen ships out of the box. Extension packs can add more
via the registry.

The built-in library is target-agnostic: the same declarations work for
any target. Target-specific behavior comes from match constraints and
bodies, not from the declarations themselves.
"""

from __future__ import annotations

from compgen.ir.ukernel.ops import UkernelBodyOp, UkernelDeclOp, UkernelMatchOp
from compgen.ir.ukernel.registry import UkernelRegistry


def _register_matmul_family(reg: UkernelRegistry) -> None:
    """Register matmul ukernel variants."""
    # Transparent generic matmul (works on any target with f32)
    reg.register_ukernel(
        decl=UkernelDeclOp(
            kernel_name="compgen_matmul_f32",
            input_types=["tensor<?x?xf32>", "tensor<?x?xf32>"],
            output_types=["tensor<?x?xf32>"],
            side_effects="none",
            calling_convention="c",
            transparency="transparent",
            body_kind="mlir",
            accepted_layouts=("rowmajor", "colmajor", "tiled"),
            preferred_layouts=("tiled",),
            output_layout="rowmajor",
            supports_prepacked_rhs=True,
            supports_transpose_absorption=True,
            tile_family="generic_tile",
        ),
        matches=[
            UkernelMatchOp(
                kernel_name="compgen_matmul_f32",
                op_family="matmul",
                dtype_constraints=("dtype_in(float32,float64,f32,f64)",),
                shape_constraints=("M>=1", "N>=1", "K>=1"),
                target_constraints=(),  # any target
                priority=1,  # low priority — specific targets override
            ),
        ],
        bodies=[
            UkernelBodyOp(
                kernel_name="compgen_matmul_f32",
                body_kind="mlir",
                transparency="transparent",
                inline_body="linalg.matmul",
                target_family="any",
            ),
        ],
    )

    # Opaque vendor matmul (GPU targets with tensor cores)
    reg.register_ukernel(
        decl=UkernelDeclOp(
            kernel_name="vendor_matmul_gpu_f32",
            input_types=["tensor<?x?xf32>", "tensor<?x?xf32>"],
            output_types=["tensor<?x?xf32>"],
            side_effects="none",
            calling_convention="cuda",
            transparency="opaque",
            body_kind="library",
            accepted_layouts=("rowmajor", "tiled_128x64"),
            preferred_layouts=("tiled_128x64",),
            output_layout="rowmajor",
            supports_prepacked_rhs=True,
            supports_transpose_absorption=True,
            tile_family="mma",
        ),
        matches=[
            UkernelMatchOp(
                kernel_name="vendor_matmul_gpu_f32",
                op_family="matmul",
                dtype_constraints=("dtype_in(float32,float16,f32,f16)",),
                shape_constraints=("M%16==0", "N%16==0"),
                target_constraints=("has_tensor_core",),
                priority=20,  # high priority for GPU with tensor cores
            ),
        ],
        bodies=[
            UkernelBodyOp(
                kernel_name="vendor_matmul_gpu_f32",
                body_kind="library",
                transparency="opaque",
                source_ref="cublas_sgemm",
                target_family="cuda",
            ),
        ],
    )

    # Transparent matmul for RVV targets
    reg.register_ukernel(
        decl=UkernelDeclOp(
            kernel_name="rvv_matmul_f32",
            input_types=["tensor<?x?xf32>", "tensor<?x?xf32>"],
            output_types=["tensor<?x?xf32>"],
            side_effects="none",
            calling_convention="c",
            transparency="transparent",
            body_kind="mlir",
            accepted_layouts=("rowmajor",),
            preferred_layouts=("rowmajor",),
            output_layout="rowmajor",
            supports_prepacked_rhs=True,
            supports_transpose_absorption=False,
            tile_family="rvv_vlmul",
        ),
        matches=[
            UkernelMatchOp(
                kernel_name="rvv_matmul_f32",
                op_family="matmul",
                dtype_constraints=("dtype_in(float32,float64,f32,f64)",),
                shape_constraints=("M>=1",),
                target_constraints=("has_rvv",),
                priority=10,  # higher than generic, lower than vendor
            ),
        ],
        bodies=[
            UkernelBodyOp(
                kernel_name="rvv_matmul_f32",
                body_kind="mlir",
                transparency="transparent",
                inline_body="linalg.matmul",
                target_family="rvv",
            ),
            UkernelBodyOp(
                kernel_name="rvv_matmul_f32",
                body_kind="mlir",
                transparency="transparent",
                inline_body="linalg.matmul",
                target_family="any",
            ),
        ],
    )


def _register_elementwise_family(reg: UkernelRegistry) -> None:
    """Register elementwise ukernel variants (relu, gelu, etc.)."""
    for op_name, inline in [
        ("relu", "arith.maximumf(%x, 0.0)"),
        ("gelu", "math.erf + arith.mulf"),
        ("sigmoid", "math.exp + arith.divf"),
    ]:
        reg.register_ukernel(
            decl=UkernelDeclOp(
                kernel_name=f"compgen_{op_name}_f32",
                input_types=["tensor<?xf32>"],
                output_types=["tensor<?xf32>"],
                side_effects="none",
                transparency="transparent",
                body_kind="mlir",
                accepted_layouts=("rowmajor", "colmajor", "tiled"),
                preferred_layouts=(),
                supports_transpose_absorption=True,
            ),
            matches=[
                UkernelMatchOp(
                    kernel_name=f"compgen_{op_name}_f32",
                    op_family=op_name,
                    dtype_constraints=("dtype_in(float32,float16,bfloat16,f32,f16,bf16)",),
                    priority=1,
                ),
            ],
            bodies=[
                UkernelBodyOp(
                    kernel_name=f"compgen_{op_name}_f32",
                    body_kind="mlir",
                    transparency="transparent",
                    inline_body=inline,
                    target_family="any",
                ),
            ],
        )


def build_default_registry() -> UkernelRegistry:
    """Build the default built-in ukernel registry.

    Returns a registry pre-populated with CompGen's built-in ukernel
    families. Extension packs can add more via register_ukernel().
    """
    reg = UkernelRegistry()
    _register_matmul_family(reg)
    _register_elementwise_family(reg)
    return reg


__all__ = ["build_default_registry"]
