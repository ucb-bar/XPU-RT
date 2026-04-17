"""Per-op-family optimization knowledge.

Encodes real optimization lessons distilled from production compilers and
kernel libraries (CUTLASS, cuDNN, Triton, Exo, MKL, oneDNN, TVM, XLA,
FlashAttention, BLIS, etc.) as structured ``OpWisdom`` entries.
"""

from __future__ import annotations

import structlog

from compgen.llm.knowledge.base import (
    BackendGuidance,
    Confidence,
    FusionOpportunity,
    LayoutPreference,
    OpWisdom,
    TilingGuidance,
)

log = structlog.get_logger()


def build_default_op_wisdom() -> dict[str, OpWisdom]:
    """Build the default library of per-op optimization wisdom.

    Returns:
        Dictionary mapping op family names to their ``OpWisdom`` entries.
    """
    entries: dict[str, OpWisdom] = {}

    # -----------------------------------------------------------------
    # matmul
    # -----------------------------------------------------------------
    entries["matmul"] = OpWisdom(
        op_family="matmul",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[128, 128, 32],
                rationale=(
                    "Maximises tensor core utilisation on NVIDIA Ampere/Hopper; "
                    "128x128 output tile saturates the warp scheduler and 32-deep K "
                    "tile fits comfortably in shared memory (~128 KB budget at fp16)."
                ),
                source="CUTLASS",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[64, 64, 32],
                rationale=(
                    "Better choice for medium-sized matrices (M,N < 2048) where 128x128 "
                    "tiles cause wave quantisation waste.  Still fills tensor cores."
                ),
                source="Triton",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="cpu",
                tile_sizes=[256, 256, 512],
                rationale=(
                    "Outer blocking targets L2 cache (256 KB - 1 MB per core).  The 256x256 "
                    "output tile and 512-deep K panel stream through L1 in a GEBP micro-kernel "
                    "pattern, matching MKL/BLIS designs."
                ),
                source="Exo/MKL",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="cpu",
                tile_sizes=[6, 16],
                rationale=(
                    "Inner register block for AVX-512: 6 rows x 16 columns keeps 6 accumulators "
                    "in ZMM registers (6 x 16 fp32 = 6 ZMMs) with headroom for A and B panels."
                ),
                source="Exo/MKL",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="npu",
                tile_sizes=[16, 16, 16],
                rationale=(
                    "Matches common systolic array dimensions (16x16 PE grid).  "
                    "Hardware accumulates a 16x16 output tile per cycle."
                ),
                source="Gemmini/Exo",
                confidence=Confidence.MEDIUM,
            ),
            TilingGuidance(
                target_class="npu",
                tile_sizes=[32, 32, 32],
                rationale=(
                    "Larger systolic arrays (32x32 PE grid) use this tile size; "
                    "common on custom TPU-style accelerators."
                ),
                source="Gemmini/Exo",
                confidence=Confidence.MEDIUM,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="matmul+bias+relu",
                description=(
                    "Epilogue fusion: bias addition and ReLU activation are folded into the "
                    "matmul kernel, avoiding a separate memory-bound kernel launch and an "
                    "extra round-trip to global memory for the intermediate result."
                ),
                conditions=[
                    "Bias is broadcastable along the N dimension.",
                    "ReLU (or other pointwise) immediately follows bias addition.",
                ],
                source="CUTLASS",
                confidence=Confidence.HIGH,
            ),
            FusionOpportunity(
                pattern="matmul+softmax",
                description=(
                    "Attention pattern: when matmul feeds directly into softmax (as in "
                    "Q@K^T -> softmax), fusing avoids materialising the full N x N "
                    "attention matrix in global memory."
                ),
                conditions=[
                    "Second operand is attention scores (Q @ K^T result).",
                    "Softmax operates along the last dimension.",
                ],
                source="cuDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="NT (row-major A, col-major B)",
                rationale=(
                    "CUTLASS NT layout enables coalesced loads for both A (row-major, "
                    "consecutive elements in K) and B (col-major, consecutive elements in K), "
                    "maximising global memory bandwidth utilisation."
                ),
                source="CUTLASS",
            ),
            LayoutPreference(
                target_class="cpu",
                preferred_layout="row-major A, row-major B (packed)",
                rationale=(
                    "Row-major storage with BLIS-style packing: B is packed into column-panels "
                    "for sequential access in the micro-kernel.  Avoids TLB misses on large matrices."
                ),
                source="BLIS/MKL",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton/CUTLASS",
                conditions="fp16/bf16/tf32 operands, M*N*K > 2^18",
                rationale=(
                    "Tensor core utilisation via mma.sync; CUTLASS for peak throughput, "
                    "Triton for faster iteration and fused epilogues."
                ),
                source="CUTLASS/Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="Exo/MKL",
                conditions="fp32/int8 operands, server-class CPU with AVX-512 or AMX",
                rationale=(
                    "Exo generates cache-tuned micro-kernels; MKL provides hand-optimised "
                    "fallback with AMX int8/bf16 support."
                ),
                source="Exo/MKL",
            ),
            BackendGuidance(
                target_class="npu",
                recommended_backend="accel dialect / Exo",
                conditions="Systolic array target with custom ISA",
                rationale=(
                    "Accel dialect maps matmul to hardware matrix engine instructions; "
                    "Exo generates explicit DMA + compute schedules."
                ),
                source="Gemmini/Exo",
            ),
        ],
        pitfalls=[
            "Bank conflicts in shared memory when tile sizes are not multiples of 32 (GPU): "
            "use padding or swizzled layouts to avoid 32-bank conflicts.",
            "Register pressure with large tile sizes (e.g., 256x256 on GPU) causes spills "
            "to local memory, negating the benefit of tiling.",
            "Wave quantisation waste: if grid_dim * block_size does not evenly divide the "
            "number of SMs, some waves run with partially-filled SMs.  Choose tile sizes so "
            "ceil(M/tile_M) * ceil(N/tile_N) is a multiple of the SM count.",
            "Incorrect K-tile alignment for tensor cores: K tile must be a multiple of 8 "
            "(fp16) or 4 (tf32) to use mma.sync instructions.",
        ],
        performance_bounds=[
            "Compute-bound when arithmetic intensity (2*M*N*K / (M*K + K*N + M*N) * elem_size) "
            "exceeds the device's ops:byte ratio (e.g., ~200 ops/byte on A100 fp16).",
            "Memory-bound for tall-skinny or short-fat matrices where one dimension is < 128.",
            "Theoretical peak: 312 TFLOPS fp16 on A100, 989 TFLOPS fp8 on H100.",
        ],
    )

    # -----------------------------------------------------------------
    # conv2d
    # -----------------------------------------------------------------
    entries["conv2d"] = OpWisdom(
        op_family="conv2d",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[1, 64, 16, 16],
                rationale=(
                    "Tile over batch (1), output channels (64), and spatial dims (16x16).  "
                    "Balances occupancy and shared memory usage for implicit GEMM on GPU."
                ),
                source="cuDNN",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="cpu",
                tile_sizes=[1, 16, 14, 14],
                rationale=(
                    "Smaller spatial tiles fit L1 cache; 16 output channels match AVX-512 "
                    "vector width for fp32."
                ),
                source="oneDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="conv2d+batchnorm+relu",
                description=(
                    "Canonical CNN fusion: batch normalisation (scale + shift) and ReLU are "
                    "folded into the convolution epilogue, eliminating two intermediate tensors."
                ),
                conditions=[
                    "BatchNorm is in inference mode (running mean/var, no gradient).",
                    "ReLU (or ReLU6 / clamp) immediately follows BatchNorm.",
                ],
                source="cuDNN/TensorRT",
                confidence=Confidence.HIGH,
            ),
            FusionOpportunity(
                pattern="conv2d+add (residual)",
                description=(
                    "ResNet-style residual addition fused into convolution output, "
                    "avoiding an extra elementwise kernel."
                ),
                conditions=["Residual tensor has same shape as conv output."],
                source="cuDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="NHWC",
                rationale=(
                    "cuDNN's fastest convolution algorithms (implicit GEMM, Winograd) "
                    "require NHWC.  Tensor cores expect channel-last layout."
                ),
                source="cuDNN",
            ),
            LayoutPreference(
                target_class="cpu",
                preferred_layout="NCHW (blocked: nChw16c)",
                rationale=(
                    "oneDNN uses blocked layout (nChw16c for AVX-512) for SIMD-friendly "
                    "channel access while keeping spatial locality."
                ),
                source="oneDNN",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="cuDNN / Triton",
                conditions="Standard kernel sizes (1x1, 3x3, 5x5, 7x7), NHWC layout",
                rationale=(
                    "cuDNN auto-tunes algorithm selection (implicit GEMM, Winograd, FFT).  "
                    "Triton for custom fused convolutions."
                ),
                source="cuDNN/Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="oneDNN",
                conditions="x86 CPU with AVX-512 or AMX",
                rationale="oneDNN provides optimised JIT convolution kernels for Intel CPUs.",
                source="oneDNN",
            ),
            BackendGuidance(
                target_class="npu",
                recommended_backend="accel dialect",
                conditions="Accelerator with conv/matmul engine",
                rationale="Map to hardware convolution instructions via accel dialect.",
                source="CompGen",
            ),
        ],
        pitfalls=[
            "Winograd convolution is only worthwhile for 3x3 and 5x5 kernels with stride 1; "
            "for other sizes the transform overhead exceeds the FLOPs saving.",
            "im2col (explicit GEMM) has significant memory overhead: the im2col buffer is "
            "C_in * K_h * K_w times larger than the input spatial dims.",
            "Algorithm selection matters enormously: cuDNN's auto-tuner can find 2-5x "
            "differences between algorithms for the same problem size.",
            "Depthwise convolution needs a completely different kernel strategy (per-channel); "
            "do not use the same tiling as standard convolution.",
        ],
        performance_bounds=[
            "Compute-bound for large batch sizes and high channel counts.",
            "Memory-bound for depthwise and 1x1 pointwise convolutions.",
            "Winograd reduces FLOPs by up to 2.25x for 3x3 (F(4x4, 3x3) transform) "
            "but adds transform overhead.",
        ],
    )

    # -----------------------------------------------------------------
    # attention (Q@K -> scale -> softmax -> @V)
    # -----------------------------------------------------------------
    entries["attention"] = OpWisdom(
        op_family="attention",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[64, 64],
                rationale=(
                    "FlashAttention tiles over sequence length in blocks of 64.  Each thread "
                    "block computes a 64-token slice of Q against 64-token blocks of K/V, "
                    "accumulating the softmax statistics in registers (online softmax)."
                ),
                source="FlashAttention",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[128, 128],
                rationale=(
                    "FlashAttention-2 uses larger tiles (128) for better tensor core "
                    "utilisation on Hopper GPUs with larger shared memory."
                ),
                source="FlashAttention-2",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="Q@K^T -> scale -> softmax -> @V",
                description=(
                    "FlashAttention pattern: the entire multi-head attention computation is "
                    "fused into a single kernel that never materialises the N x N attention "
                    "matrix in global memory.  Uses online softmax (log-sum-exp tracking) to "
                    "compute exact softmax in a single pass."
                ),
                conditions=[
                    "Standard multi-head attention (Q, K, V with compatible shapes).",
                    "Causal or non-causal masking (both supported).",
                    "Head dimension is a power of 2 and <= 256.",
                ],
                source="FlashAttention/Triton",
                confidence=Confidence.HIGH,
            ),
            FusionOpportunity(
                pattern="attention+dropout",
                description=(
                    "Dropout mask is generated and applied inside the fused attention kernel, "
                    "avoiding materialisation of the dropout mask tensor."
                ),
                conditions=["Training mode with dropout probability > 0."],
                source="FlashAttention",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="(batch, heads, seq_len, head_dim) contiguous",
                rationale=(
                    "FlashAttention expects Q, K, V in (B, H, S, D) layout with the last "
                    "dimension contiguous for coalesced loads."
                ),
                source="FlashAttention",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton (FlashAttention)",
                conditions="Sequence length > 128, head_dim in {64, 128}",
                rationale=(
                    "Triton FlashAttention kernel achieves near-peak memory bandwidth and "
                    "avoids O(N^2) memory for the attention matrix."
                ),
                source="FlashAttention/Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="fused attention kernel",
                conditions="Inference with moderate sequence lengths",
                rationale=(
                    "CPU fused attention avoids materialising the attention matrix; "
                    "block-wise computation fits in L2 cache."
                ),
                source="oneDNN/PyTorch",
            ),
        ],
        pitfalls=[
            "Naive attention materialises an O(N^2) intermediate matrix (seq_len x seq_len), "
            "causing OOM for long sequences (e.g., 8192+ tokens on 40 GB GPU).",
            "Memory-bound for long sequences even with FlashAttention: total I/O scales as "
            "O(N^2 * d / M) where M is SRAM size.",
            "Causal masking requires careful handling in the tiled kernel to avoid wasted "
            "computation on masked-out tiles (skip fully-masked blocks).",
            "Numerical precision: online softmax must track running max and log-sum-exp "
            "carefully to match the numerics of the unfused version.",
        ],
        performance_bounds=[
            "Memory-bound for seq_len > 512 on most GPUs when not using FlashAttention.",
            "FlashAttention is IO-aware: complexity is O(N^2 * d^2 / M) HBM accesses vs "
            "O(N^2 * d + N^2) for standard attention.",
            "Compute-bound with FlashAttention when head_dim is large (128+) and seq_len "
            "is moderate (< 2048).",
        ],
    )

    # -----------------------------------------------------------------
    # reduction (sum, max, softmax, mean)
    # -----------------------------------------------------------------
    entries["reduction"] = OpWisdom(
        op_family="reduction",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[256],
                rationale=(
                    "256-thread block for intra-block tree reduction using shared memory.  "
                    "Warp-level reductions via __shfl_down_sync for the final 32 elements, "
                    "then shared memory for cross-warp combination."
                ),
                source="Triton/CUDA handbook",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="cpu",
                tile_sizes=[16],
                rationale=(
                    "SIMD horizontal reduction: reduce 16 fp32 elements (AVX-512) per "
                    "instruction, then scalar reduction across lanes."
                ),
                source="oneDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="reduction+broadcast+elementwise",
                description=(
                    "Softmax pattern: reduce(max) -> subtract -> exp -> reduce(sum) -> divide.  "
                    "Fusing the entire chain avoids multiple kernel launches and intermediate "
                    "tensor allocations."
                ),
                conditions=[
                    "Reduction and broadcast are along the same dimension.",
                    "Elementwise ops are between reductions.",
                ],
                source="Triton/TVM",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="Reduce along contiguous (innermost) dimension",
                rationale=(
                    "Coalesced reads when reducing along the last dimension.  Reducing along "
                    "non-contiguous dimensions requires a transpose or strided access."
                ),
                source="CUDA handbook",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton",
                conditions="Any reduction axis, any dtype",
                rationale=(
                    "Triton's tl.reduce primitives generate efficient warp-shuffle and "
                    "shared-memory reduction code."
                ),
                source="Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="native vectorisation",
                conditions="Reduction over contiguous axis",
                rationale="LLVM auto-vectorises simple reductions effectively with -O3.",
                source="LLVM/oneDNN",
            ),
        ],
        pitfalls=[
            "Cross-warp reduction on GPU requires __shfl instructions (warp shuffle); "
            "naive shared memory atomic approach causes serialisation.",
            "Global atomics have severe contention for large grids; prefer a two-pass "
            "approach (per-block partial reduction, then final reduction kernel).",
            "Softmax numerical stability: always subtract the row maximum before exp() "
            "to avoid overflow in fp16/bf16.",
            "Reduction along non-contiguous dimensions is much slower; consider transposing "
            "first if the reduction is the bottleneck.",
        ],
        performance_bounds=[
            "Reduction is almost always memory-bound: one pass over input data with minimal "
            "compute per element.",
            "Theoretical throughput limited by memory bandwidth / element_size.",
        ],
    )

    # -----------------------------------------------------------------
    # elementwise (relu, gelu, sigmoid, add, mul, etc.)
    # -----------------------------------------------------------------
    entries["elementwise"] = OpWisdom(
        op_family="elementwise",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[1024],
                rationale=(
                    "Block size of 1024 threads with vectorised loads (float4 / 128-bit).  "
                    "Grid-stride loop over the full tensor.  Maximises memory bandwidth "
                    "utilisation by keeping all SMs busy."
                ),
                source="Triton/CUDA",
                confidence=Confidence.HIGH,
            ),
            TilingGuidance(
                target_class="cpu",
                tile_sizes=[64],
                rationale=(
                    "Process 64 elements per inner loop iteration (4 AVX-512 vectors of "
                    "16 fp32 elements) to hide instruction latency."
                ),
                source="oneDNN",
                confidence=Confidence.MEDIUM,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="elementwise chain",
                description=(
                    "Always fuse consecutive elementwise operations (e.g., add -> relu, "
                    "mul -> sigmoid, gelu = x * 0.5 * (1 + erf(x/sqrt(2)))) into a single "
                    "kernel to avoid redundant global memory round-trips."
                ),
                conditions=["All ops have compatible shapes (broadcasting allowed)."],
                source="TVM/XLA/Triton",
                confidence=Confidence.HIGH,
            ),
            FusionOpportunity(
                pattern="matmul epilogue",
                description=(
                    "Fuse elementwise ops (bias, activation, residual add) into the matmul "
                    "epilogue rather than launching separate kernels."
                ),
                conditions=[
                    "Elementwise op immediately consumes matmul output.",
                    "No other consumer of the intermediate matmul result.",
                ],
                source="CUTLASS/Triton",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="contiguous (any order, as long as innermost dim is contiguous)",
                rationale="Elementwise ops only need coalesced access; layout order is irrelevant.",
                source="Triton",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton or fused into matmul epilogue",
                conditions="Any pointwise operation",
                rationale=(
                    "Standalone elementwise kernels in Triton are trivial to write; "
                    "but prefer fusing into a producer kernel's epilogue when possible."
                ),
                source="Triton/CUTLASS",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="native vectorisation",
                conditions="Any pointwise operation",
                rationale="LLVM vectorises elementwise loops automatically with proper pragmas.",
                source="LLVM",
            ),
        ],
        pitfalls=[
            "Do NOT launch a separate GPU kernel for a single elementwise op if it can be "
            "fused with its producer or consumer.  Kernel launch overhead (~5 us) dominates "
            "for small tensors.",
            "Broadcasting can cause non-coalesced access if the broadcast dimension is the "
            "innermost one.  Materialise the broadcast explicitly for large tensors.",
            "GELU and SiLU have expensive transcendental functions (erf, sigmoid); use "
            "polynomial approximations when fp16 precision is acceptable.",
        ],
        performance_bounds=[
            "Always memory-bound: arithmetic intensity is O(1) -- one op per element loaded.",
            "Peak throughput = memory_bandwidth / element_size.",
            "Fusion is the ONLY lever for improving elementwise performance (turn memory-bound "
            "elementwise into part of a compute-bound fused kernel).",
        ],
    )

    # -----------------------------------------------------------------
    # transpose / permute
    # -----------------------------------------------------------------
    entries["transpose"] = OpWisdom(
        op_family="transpose",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[32, 32],
                rationale=(
                    "32x32 tile in shared memory: read a 32x32 tile with coalesced loads, "
                    "transpose in shared memory, write back with coalesced stores.  Padding "
                    "shared memory to 33 columns avoids bank conflicts."
                ),
                source="CUDA handbook",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="transpose+matmul",
                description=(
                    "Avoid explicit transpose by using transposed GEMM variant (e.g., "
                    "CUTLASS TN/NT layout).  The matmul kernel reads the matrix in "
                    "transposed order directly."
                ),
                conditions=["Transpose feeds directly into a matmul operand."],
                source="CUTLASS",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton or vendor kernel",
                conditions="Standalone transpose required",
                rationale=(
                    "Triton generates shared-memory-based transpose kernels.  "
                    "Prefer fusing into consumer when possible."
                ),
                source="Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="native (loop interchange)",
                conditions="Any transpose",
                rationale="Cache-oblivious recursive transpose or tiled copy.",
                source="BLIS",
            ),
        ],
        pitfalls=[
            "Naive GPU transpose causes completely uncoalesced writes (or reads), reducing "
            "effective bandwidth by up to 10x.  Always use shared memory staging.",
            "Permutations of more than 2 dimensions may require multiple passes or "
            "generalised copy kernels; check if a view/stride change suffices.",
        ],
        performance_bounds=[
            "Memory-bound: 2 * tensor_size bytes transferred (read + write).",
            "Achievable bandwidth: 80-90% of peak HBM bandwidth with shared memory staging.",
        ],
    )

    # -----------------------------------------------------------------
    # gather / scatter
    # -----------------------------------------------------------------
    entries["gather"] = OpWisdom(
        op_family="gather",
        tiling_guidance=[],
        fusion_opportunities=[],
        layout_preferences=[],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="custom Triton kernel",
                conditions="Irregular index patterns",
                rationale=(
                    "Triton's tl.load with mask handles irregular gather patterns.  "
                    "No vendor library covers the general case well."
                ),
                source="Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="native loops",
                conditions="Any gather/scatter",
                rationale=(
                    "AVX-512 has vpgatherdd/vpscatterdd but throughput is limited; "
                    "scalar loops may be faster for irregular patterns."
                ),
                source="Intel intrinsics guide",
            ),
        ],
        pitfalls=[
            "Irregular access patterns defeat caching and vectorisation.  Pre-sort indices "
            "when possible to improve spatial locality.",
            "Scatter with duplicate indices has undefined order; use atomic operations or "
            "segment-based reduction for correctness.",
            "GPU gather/scatter is limited by L2 cache hit rate on the index pattern.  "
            "Random indices achieve only 10-20% of peak bandwidth.",
        ],
        performance_bounds=[
            "Throughput depends entirely on index pattern locality.",
            "Best case (sequential indices): approaches memcpy bandwidth.",
            "Worst case (random indices): limited by cache line fetch rate.",
        ],
    )

    # -----------------------------------------------------------------
    # batch_norm / layer_norm
    # -----------------------------------------------------------------
    entries["batch_norm"] = OpWisdom(
        op_family="batch_norm",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[256],
                rationale=(
                    "Reduce over spatial dimensions (H, W) and batch (N) per channel.  "
                    "One thread block per channel with 256 threads for the reduction."
                ),
                source="cuDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="conv2d+batch_norm",
                description=(
                    "Fold BatchNorm into preceding convolution weights at inference time: "
                    "W_fused = W * gamma / sqrt(var + eps), b_fused = (b - mean) * gamma / "
                    "sqrt(var + eps) + beta.  Zero runtime cost."
                ),
                conditions=[
                    "Inference mode (running statistics, no gradient).",
                    "Preceding op is conv2d or linear.",
                ],
                source="cuDNN/TensorRT/oneDNN",
                confidence=Confidence.HIGH,
            ),
            FusionOpportunity(
                pattern="batch_norm+relu",
                description="Fuse activation into the BatchNorm output pass.",
                conditions=["ReLU or clamp immediately follows BatchNorm."],
                source="cuDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="NHWC",
                rationale="Channel-last layout enables coalesced reduction over spatial dims.",
                source="cuDNN",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="cuDNN",
                conditions="Standard batch normalisation",
                rationale="cuDNN's fused BN kernels handle forward, backward, and inference.",
                source="cuDNN",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="oneDNN",
                conditions="Standard batch normalisation",
                rationale="oneDNN provides optimised BN with SIMD reduction.",
                source="oneDNN",
            ),
        ],
        pitfalls=[
            "Training BN requires two passes (mean, then variance) unless using Welford's "
            "online algorithm in a single pass.",
            "Small batch sizes make BN statistics noisy; consider GroupNorm or LayerNorm.",
            "BN is often the fusion boundary: always try to fold into preceding conv/linear.",
        ],
        performance_bounds=[
            "Memory-bound: two passes over the activation tensor (compute stats, then normalise).",
            "Inference: can be completely eliminated by folding into preceding linear op.",
        ],
    )

    entries["layer_norm"] = OpWisdom(
        op_family="layer_norm",
        tiling_guidance=[
            TilingGuidance(
                target_class="gpu",
                tile_sizes=[256],
                rationale=(
                    "Reduce over the normalisation axis (typically the last dimension).  "
                    "One thread block per token/sample with 256 threads for the reduction."
                ),
                source="Triton/cuDNN",
                confidence=Confidence.HIGH,
            ),
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern="layer_norm+linear",
                description=(
                    "Fuse LayerNorm into the subsequent linear layer's prologue to avoid "
                    "writing the normalised intermediate to global memory."
                ),
                conditions=["Linear immediately follows LayerNorm."],
                source="Triton/Megatron-LM",
                confidence=Confidence.MEDIUM,
            ),
        ],
        layout_preferences=[
            LayoutPreference(
                target_class="gpu",
                preferred_layout="(..., D) with D contiguous",
                rationale="Reduction over the last (contiguous) dimension is fastest.",
                source="Triton",
            ),
        ],
        backend_guidance=[
            BackendGuidance(
                target_class="gpu",
                recommended_backend="Triton",
                conditions="Standard layer normalisation",
                rationale=(
                    "Triton LayerNorm kernel with online Welford reduction.  "
                    "Often fused with adjacent ops."
                ),
                source="Triton",
            ),
            BackendGuidance(
                target_class="cpu",
                recommended_backend="oneDNN",
                conditions="Standard layer normalisation",
                rationale="oneDNN layer_normalization primitive with SIMD.",
                source="oneDNN",
            ),
        ],
        pitfalls=[
            "LayerNorm with large hidden dimensions (e.g., 12288 in GPT-3) benefits from "
            "warp-level reduction; ensure enough threads per block.",
            "RMSNorm (used in LLaMA) is cheaper: skip mean subtraction, only compute "
            "root-mean-square.  Do not use full LayerNorm when RMSNorm suffices.",
        ],
        performance_bounds=[
            "Memory-bound: single pass over the normalisation dimension.",
            "Throughput limited by reduction bandwidth, not compute.",
        ],
    )

    log.info("built_default_op_wisdom", op_count=len(entries))
    return entries
