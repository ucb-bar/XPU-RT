"""Frozen descriptor for the reference cuda_tile adapter.

Mirrors the descriptor that bwell's ``compgen_cuda_tile`` package
declared after running ``scaffold_vendor_package`` on the cuda-tile
repo (bridge #138 onward). Kept in code rather than YAML so the
reference adapter has zero on-disk dependencies — any caller can
``make_adapter()`` in a fresh interpreter without filesystem state.
"""

from __future__ import annotations

from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    OpEntry,
    VendorDialectDescriptor,
    VerificationPlan,
)

_OP_REGISTRY: tuple[OpEntry, ...] = (
    OpEntry(name="cuda_tile.module", summary="Top-level kernel module."),
    OpEntry(name="cuda_tile.entry", summary="Kernel entry point; rank-0 tile<ptr<*>> args only."),
    OpEntry(name="cuda_tile.make_tensor_view", summary="Materialize a tensor view from a pointer."),
    OpEntry(
        name="cuda_tile.make_partition_view",
        summary="Partition a tensor view into tiles.",
    ),
    OpEntry(name="cuda_tile.load_view_tko", summary="Load a tile from a partition view at indices."),
    OpEntry(name="cuda_tile.store_view_tko", summary="Store a tile into a partition view at indices."),
    OpEntry(name="cuda_tile.mmaf", summary="Matrix-multiply-accumulate (float)."),
    OpEntry(name="cuda_tile.maxf", summary="Element-wise max (float) — used for relu."),
    OpEntry(name="cuda_tile.constant", summary="Tile-shaped constant."),
    OpEntry(name="cuda_tile.return", summary="Entry return."),
)


def build_descriptor() -> VendorDialectDescriptor:
    """Return the canonical reference descriptor."""
    return VendorDialectDescriptor(
        name="cuda_tile",
        package_name="compgen.extensions.vendor_dialect.builtins.cuda_tile",
        repo_path="<builtin>",
        target="nvidia-blackwell",
        input_ir=("payload-mlir",),
        output_format="cuda-tile-bitcode",
        compile_entry=CompileEntry(
            cli_tools=("cuda-tile-translate",),
            python_module="cuda_tile._mlir",
        ),
        td_files=(),
        op_registry=_OP_REGISTRY,
        lowering=LoweringStrategy(
            mode="kernel_authoring",
            op_families=("matmul", "relu", "ffn"),
            template_ops=("mmaf", "maxf", "load_view_tko", "store_view_tko"),
            notes=(
                "Single-tile FFN(matmul-relu-matmul) template. Rank-0 "
                "tile<ptr<f32>> entry args; view ops materialize 2D tiles "
                "inside the body. Validated against cuda-tile-translate "
                "via bridge #144."
            ),
        ),
        bundle=BundlePlan(
            steps=(
                "lower_payload_to_cuda_tile_mlir",
                "cuda-tile-translate -mlir-to-cudatilebc",
            ),
            output_format="cuda-tile-bitcode",
            runtime_entry="ffn_matmul_relu_matmul",
        ),
        verification=VerificationPlan(
            structural=True,
            matmul_diff_test=True,
            workload_diff_test=False,
            workloads=("ffn_64_128_64",),
            tolerance_rtol=1e-3,
            tolerance_atol=1e-3,
        ),
        kernel_authoring_required=False,
        dependencies=("cuda-tile-translate",),
        license="Apache-2.0",
        extras={
            "validated_against": "bridge#144",
            "bytecode_magic": "7f54696c654952000d01",
            "in_tree_reference": True,
        },
    )


__all__ = ["build_descriptor"]
