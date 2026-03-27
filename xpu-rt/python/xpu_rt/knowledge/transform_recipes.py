"""Curated transform sequences that work well for common (op, target) pairs.

Each recipe encodes a known-good sequence of scheduling / lowering transforms
extracted from production compilers and kernel libraries.  The LLM proposal
engine can reference these as starting points.
"""

from __future__ import annotations

from compgen.knowledge.base import (
    Confidence,
    TransformRecipe,
    TransformStep,
)


def build_default_recipes() -> list[TransformRecipe]:
    """Return the default catalogue of curated transform recipes.

    Returns:
        List of ``TransformRecipe`` entries covering GPU, CPU, and NPU
        workloads for matmul, conv2d, attention, reduction, and
        elementwise ops.
    """
    return [
        _matmul_gpu_tensor_core(),
        _matmul_cpu_avx2(),
        _matmul_npu_systolic(),
        _conv2d_gpu_implicit_gemm(),
        _attention_flash(),
        _reduction_gpu_tree(),
        _elementwise_fusion_chain(),
        _depthwise_conv2d_gpu(),
        _layernorm_gpu_fused(),
        _softmax_gpu_online(),
        _batched_matmul_gpu(),
    ]


# ---------------------------------------------------------------------------
# Individual recipe builders
# ---------------------------------------------------------------------------


def _matmul_gpu_tensor_core() -> TransformRecipe:
    """GPU matmul recipe targeting tensor cores."""
    return TransformRecipe(
        name="matmul_gpu_tensor_core",
        op_family="matmul",
        target_class="gpu",
        steps=[
            TransformStep(
                action="tile",
                parameters={"tile_sizes": [128, 128, 32]},
                rationale="Match tensor core tile shape and shared memory capacity",
            ),
            TransformStep(
                action="interchange",
                parameters={"permutation": [2, 0, 1]},
                rationale="Move K to outermost for output-stationary dataflow",
            ),
            TransformStep(
                action="stage_mem",
                parameters={"operands": ["A", "B"], "target": "shared"},
                rationale="Exploit data reuse across thread block",
            ),
            TransformStep(
                action="tile",
                parameters={"tile_sizes": [16, 16, 16], "level": "inner"},
                rationale="Match warp-level MMA shape",
            ),
            TransformStep(
                action="vectorize",
                parameters={"target": "loads", "width_bits": 128},
                rationale="Use 128-bit loads for coalesced access",
            ),
        ],
        expected_speedup="10-50x over naive, within 80% of CUTLASS",
        source="CUTLASS/Triton patterns",
        confidence=Confidence.HIGH,
    )


def _matmul_cpu_avx2() -> TransformRecipe:
    """CPU matmul recipe targeting AVX2 (Exo-style)."""
    return TransformRecipe(
        name="matmul_cpu_avx2",
        op_family="matmul",
        target_class="cpu",
        steps=[
            TransformStep(
                action="tile",
                parameters={"tile_sizes": [256, 256, 512], "labels": ["MC", "NC", "KC"]},
                rationale="L2 cache blocking",
            ),
            TransformStep(
                action="tile",
                parameters={"tile_sizes": [6, 16], "labels": ["MR", "NR"], "level": "inner"},
                rationale="Register blocking for AVX2 (6 rows x 2 YMM regs)",
            ),
            TransformStep(
                action="interchange",
                parameters={"move_to_outermost": "K"},
                rationale="Output-stationary accumulation in registers",
            ),
            TransformStep(
                action="stage_mem",
                parameters={"operand": "C_reg", "target": "registers"},
                rationale="Keep accumulator in YMM registers",
            ),
            TransformStep(
                action="divide_dim",
                parameters={"operand": "C_reg", "factor": 8},
                rationale="Map to individual YMM registers (8 floats each)",
            ),
            TransformStep(
                action="set_memory",
                parameters={"operand": "C_reg", "memory": "AVX2"},
                rationale="Pin to vector registers",
            ),
            TransformStep(
                action="replace_intrinsics",
                parameters={
                    "instructions": [
                        "mm256_loadu_ps",
                        "mm256_fmadd_ps",
                        "mm256_broadcast_ss",
                    ],
                },
                rationale="Replace loads/stores/FMA with AVX2 intrinsics",
            ),
        ],
        expected_speedup="80-95% of MKL",
        source="Exo AVX2 matmul tutorial",
        confidence=Confidence.HIGH,
    )


def _matmul_npu_systolic() -> TransformRecipe:
    """NPU matmul recipe targeting systolic arrays (Gemmini-style)."""
    return TransformRecipe(
        name="matmul_npu_systolic",
        op_family="matmul",
        target_class="npu",
        steps=[
            TransformStep(
                action="tile",
                parameters={"tile_sizes": ["DIM", "DIM", "DIM"]},
                rationale="Match PE array geometry",
            ),
            TransformStep(
                action="configure",
                parameters={"mode": "matmul_accumulate"},
                rationale="Set config for matmul+accumulate",
            ),
            TransformStep(
                action="dma_load",
                parameters={"operands": ["A", "B"], "target": "scratchpad"},
                rationale="Pre-stage data for systolic execution",
            ),
            TransformStep(
                action="invoke_engine",
                parameters={"engine": "systolic_matmul"},
                rationale="Single instruction for tile matmul",
            ),
            TransformStep(
                action="dma_store",
                parameters={"operand": "result", "source": "accumulator"},
                rationale="Move result from accumulator to main memory",
            ),
        ],
        expected_speedup="Near-peak hardware utilization",
        source="Gemmini/Exo Gemmini schedule library",
        confidence=Confidence.HIGH,
    )


def _conv2d_gpu_implicit_gemm() -> TransformRecipe:
    """GPU conv2d recipe using implicit GEMM lowering."""
    return TransformRecipe(
        name="conv2d_gpu_implicit_gemm",
        op_family="conv2d",
        target_class="gpu",
        steps=[
            TransformStep(
                action="reshape",
                parameters={"strategy": "implicit_gemm"},
                rationale="Reshape as matmul without materializing im2col",
            ),
            TransformStep(
                action="tile",
                parameters={"tile_sizes": ["N_tile", "K_tile", "PQ_tile"]},
                rationale="Map to matmul tile",
            ),
            TransformStep(
                action="use_tensor_cores",
                parameters={"dtype": "fp16", "accumulator": "fp32"},
                rationale="fp16 accumulation via tensor cores",
            ),
            TransformStep(
                action="fuse",
                parameters={"ops": ["batch_norm", "relu"]},
                rationale="Epilogue fusion",
            ),
        ],
        expected_speedup="Within 90% of cuDNN for common shapes",
        source="cuDNN implicit GEMM / CUTLASS conv",
        confidence=Confidence.HIGH,
    )


def _attention_flash() -> TransformRecipe:
    """GPU attention recipe based on FlashAttention tiling."""
    return TransformRecipe(
        name="attention_flash",
        op_family="attention",
        target_class="gpu",
        steps=[
            TransformStep(
                action="tile",
                parameters={"dim": "sequence_length", "block_size": [64, 128]},
                rationale="Process attention in blocks",
            ),
            TransformStep(
                action="fuse",
                parameters={"ops": ["qk_matmul", "softmax"]},
                rationale="Avoid materializing N x N attention matrix",
            ),
            TransformStep(
                action="online_softmax",
                parameters={"method": "incremental"},
                rationale="Compute softmax incrementally",
            ),
            TransformStep(
                action="accumulate",
                parameters={"op": "attn_v_matmul", "strategy": "output_stationary"},
                rationale="Output-stationary accumulation within tile",
            ),
        ],
        expected_speedup="2-4x over naive, O(N) memory vs O(N^2)",
        source="FlashAttention paper",
        confidence=Confidence.HIGH,
    )


def _reduction_gpu_tree() -> TransformRecipe:
    """GPU reduction recipe using tree reduction."""
    return TransformRecipe(
        name="reduction_gpu_tree",
        op_family="reduction",
        target_class="gpu",
        steps=[
            TransformStep(
                action="parallel_reduce",
                parameters={"scope": "thread_block"},
                rationale="Each block reduces a tile",
            ),
            TransformStep(
                action="tree_reduce",
                parameters={"method": "shfl_down", "steps": "log2"},
                rationale="Log2 steps with __shfl_down within warp",
            ),
            TransformStep(
                action="cross_block_reduce",
                parameters={"method": "atomic_or_multipass"},
                rationale="Final reduction across blocks",
            ),
        ],
        expected_speedup="Near bandwidth-bound peak",
        source="CUDA reduction best practices",
        confidence=Confidence.HIGH,
    )


def _elementwise_fusion_chain() -> TransformRecipe:
    """GPU elementwise fusion recipe."""
    return TransformRecipe(
        name="elementwise_fusion_chain",
        op_family="elementwise",
        target_class="gpu",
        steps=[
            TransformStep(
                action="identify_chain",
                parameters={"ops": ["relu", "add", "mul", "sigmoid"]},
                rationale="Identify chain of elementwise ops",
            ),
            TransformStep(
                action="fuse",
                parameters={"strategy": "single_kernel"},
                rationale="Avoid intermediate materializations",
            ),
            TransformStep(
                action="vectorize",
                parameters={"dtype": "float4", "target": "loads_stores"},
                rationale="Maximize memory bandwidth with vectorized access",
            ),
            TransformStep(
                action="loop_strategy",
                parameters={"method": "grid_stride"},
                rationale="Handle arbitrary tensor sizes",
            ),
        ],
        expected_speedup="2-5x over separate kernels (memory-bound)",
        source="Triton/CUDA fusion patterns",
        confidence=Confidence.HIGH,
    )


def _depthwise_conv2d_gpu() -> TransformRecipe:
    """GPU depthwise conv2d recipe."""
    return TransformRecipe(
        name="depthwise_conv2d_gpu",
        op_family="conv2d",
        target_class="gpu",
        steps=[
            TransformStep(
                action="map_channels",
                parameters={"strategy": "one_channel_per_block"},
                rationale="Each thread block handles one output channel",
            ),
            TransformStep(
                action="tile",
                parameters={"tile_sizes": ["H_tile", "W_tile"], "dims": ["height", "width"]},
                rationale="Tile spatial dimensions for shared memory reuse",
            ),
            TransformStep(
                action="stage_mem",
                parameters={"operand": "input_tile", "target": "shared", "halo": True},
                rationale="Load input patch with halo into shared memory",
            ),
            TransformStep(
                action="fuse",
                parameters={"ops": ["bias", "relu"]},
                rationale="Epilogue fusion avoids extra kernel launch",
            ),
        ],
        expected_speedup="3-10x over naive, within 85% of cuDNN",
        source="EfficientNet / MobileNet optimization guides",
        confidence=Confidence.MEDIUM,
    )


def _layernorm_gpu_fused() -> TransformRecipe:
    """GPU fused layer normalization recipe."""
    return TransformRecipe(
        name="layernorm_gpu_fused",
        op_family="layernorm",
        target_class="gpu",
        steps=[
            TransformStep(
                action="parallel_reduce",
                parameters={"ops": ["mean", "variance"], "scope": "row"},
                rationale="Compute mean and variance in a single pass per row",
            ),
            TransformStep(
                action="warp_reduce",
                parameters={"method": "shfl_down"},
                rationale="Use warp shuffle for fast intra-warp reduction",
            ),
            TransformStep(
                action="normalize_and_scale",
                parameters={"fused": True, "ops": ["subtract_mean", "divide_std", "scale", "bias"]},
                rationale="Fuse normalization and affine transform to avoid extra reads",
            ),
            TransformStep(
                action="vectorize",
                parameters={"width_bits": 128},
                rationale="Coalesced 128-bit loads and stores",
            ),
        ],
        expected_speedup="2-3x over separate mean/var/normalize kernels",
        source="Apex/Triton fused LayerNorm",
        confidence=Confidence.HIGH,
    )


def _softmax_gpu_online() -> TransformRecipe:
    """GPU numerically stable softmax recipe using online algorithm."""
    return TransformRecipe(
        name="softmax_gpu_online",
        op_family="softmax",
        target_class="gpu",
        steps=[
            TransformStep(
                action="online_max",
                parameters={"method": "running_max"},
                rationale="Track running max to avoid separate max-reduction pass",
            ),
            TransformStep(
                action="online_sum",
                parameters={"method": "rescaled_accumulation"},
                rationale="Accumulate exp(x - running_max) with correction factors",
            ),
            TransformStep(
                action="normalize",
                parameters={"fused": True},
                rationale="Divide by final sum in same kernel",
            ),
            TransformStep(
                action="vectorize",
                parameters={"width_bits": 128},
                rationale="Coalesced access for memory-bound workload",
            ),
        ],
        expected_speedup="1.5-2x over two-pass (max then softmax)",
        source="Online softmax / FlashAttention",
        confidence=Confidence.HIGH,
    )


def _batched_matmul_gpu() -> TransformRecipe:
    """GPU batched matmul recipe for transformer workloads."""
    return TransformRecipe(
        name="batched_matmul_gpu",
        op_family="matmul",
        target_class="gpu",
        steps=[
            TransformStep(
                action="batch_map",
                parameters={"strategy": "batch_per_block_cluster"},
                rationale="Map batch dimension to block grid z-axis",
            ),
            TransformStep(
                action="tile",
                parameters={"tile_sizes": [128, 128, 32]},
                rationale="Standard tensor core tile for M, N, K",
            ),
            TransformStep(
                action="stage_mem",
                parameters={"operands": ["A", "B"], "target": "shared", "stages": 3},
                rationale="Multi-stage async pipeline to hide global memory latency",
            ),
            TransformStep(
                action="use_tensor_cores",
                parameters={"dtype": "fp16", "accumulator": "fp32"},
                rationale="Leverage tensor cores for throughput",
            ),
            TransformStep(
                action="swizzle",
                parameters={"target": "shared_memory", "pattern": "128B"},
                rationale="Avoid shared memory bank conflicts",
            ),
        ],
        expected_speedup="Within 85% of cuBLAS batched GEMM",
        source="CUTLASS batched GEMM / Triton batched matmul",
        confidence=Confidence.HIGH,
    )
