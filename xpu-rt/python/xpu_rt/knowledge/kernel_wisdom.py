"""Lessons distilled from production kernel libraries.

Each entry captures a single actionable insight from CUTLASS, cuDNN, oneDNN,
Triton, Exo, or another kernel library, along with the conditions under which
the insight applies.
"""

from __future__ import annotations

from compgen.knowledge.base import (
    Confidence,
    KernelLibraryWisdom,
)


def build_default_kernel_wisdom() -> list[KernelLibraryWisdom]:
    """Return the default catalogue of kernel library wisdom.

    Returns:
        List of ``KernelLibraryWisdom`` entries (25+) covering CUTLASS,
        cuDNN, oneDNN, Triton, and Exo.
    """
    return [
        # -- CUTLASS ---------------------------------------------------------
        KernelLibraryWisdom(
            library="cutlass",
            topic="threadblock_tile_sizes",
            insight=(
                "Use 128x128, 128x256, or 256x128 threadblock tiles for large GEMMs. "
                "Fall back to 64x64 for small or skinny GEMMs. Warp tiles of 32x32 or "
                "64x32 map well to tensor core shapes."
            ),
            conditions=["GEMM problem size >= 512 in each dimension for large tiles"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="epilogue_fusion",
            insight=(
                "Fuse bias addition, activation functions (relu, gelu, silu), and "
                "residual add into the GEMM epilogue to avoid separate kernel launches "
                "and redundant global memory traffic."
            ),
            conditions=["Epilogue ops are element-wise or row-broadcast"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="persistent_kernels",
            insight=(
                "Persistent kernels keep thread blocks alive across multiple output "
                "tiles, improving SM utilization by amortizing launch overhead and "
                "enabling software-managed scheduling. Crucial for small problem sizes "
                "that would otherwise under-occupy the GPU."
            ),
            conditions=["CUTLASS 3.x or later", "Problem size small enough to under-occupy SMs"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="swizzle_patterns",
            insight=(
                "Swizzle patterns remap thread IDs to data elements to avoid shared "
                "memory bank conflicts. A 128B swizzle aligns 128-byte access patterns "
                "across threads in a warp."
            ),
            conditions=["Shared memory used for tile staging"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="pipeline_stages",
            insight=(
                "Use 2-4 asynchronous copy pipeline stages (cp.async / TMA) to overlap "
                "global-to-shared memory transfers with MMA computation. More stages "
                "hide more latency but consume more shared memory."
            ),
            conditions=["Ampere or later GPU architecture"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="mixed_precision",
            insight=(
                "Use fp16 or bf16 for compute operands with fp32 accumulator to prevent "
                "error accumulation in large reductions. The tensor core natively supports "
                "this mixed-precision mode."
            ),
            conditions=["Tensor core capable GPU", "Acceptable accuracy trade-off"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="warp_specialization",
            insight=(
                "In CUTLASS 3.x, warps within a CTA can be specialized for different "
                "roles: producer warps issue TMA loads while consumer warps execute MMA, "
                "enabling a pipelined producer-consumer overlap."
            ),
            conditions=["Hopper architecture (sm_90+)", "CUTLASS 3.x"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cutlass",
            topic="split_k",
            insight=(
                "For skinny GEMMs (small M or N, large K), split the K dimension across "
                "multiple thread blocks and accumulate partial results, trading extra "
                "memory for higher parallelism."
            ),
            conditions=["M or N < ~256 and K >> M*N"],
            confidence=Confidence.MEDIUM,
        ),
        # -- cuDNN -----------------------------------------------------------
        KernelLibraryWisdom(
            library="cudnn",
            topic="algorithm_selection",
            insight=(
                "cuDNN provides multiple algorithms for each operation "
                "(IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, GEMM, FFT, WINOGRAD, "
                "WINOGRAD_NONFUSED). Profile all and pick the fastest for each "
                "specific input shape -- there is no single best algorithm."
            ),
            conditions=["Shape known at compile time or amortized across iterations"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cudnn",
            topic="workspace_management",
            insight=(
                "Allocate the workspace buffer once and reuse across layers. "
                "Some algorithms require O(N^2) workspace; pre-query max workspace "
                "to avoid repeated allocations."
            ),
            conditions=["Static graph with known shapes"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cudnn",
            topic="tensor_core_enforcement",
            insight=(
                "Set CUDNN_TENSOR_OP_MATH explicitly to enable tensor core paths. "
                "Default math mode may not use tensor cores even when available."
            ),
            conditions=["Volta or later GPU", "fp16 or bf16 data"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cudnn",
            topic="graph_api",
            insight=(
                "cuDNN 8.x graph API builds an execution plan for an entire subgraph, "
                "enabling inter-kernel fusion and global optimization across multiple "
                "operations (e.g., conv + bias + relu + residual)."
            ),
            conditions=["cuDNN 8.x or later", "Subgraph of supported operations"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="cudnn",
            topic="nhwc_preference",
            insight=(
                "cuDNN tensor core kernels strongly prefer NHWC layout. Using NCHW "
                "forces a layout transpose or falls back to non-tensor-core paths."
            ),
            conditions=["Tensor core capable GPU"],
            confidence=Confidence.HIGH,
        ),
        # -- oneDNN ----------------------------------------------------------
        KernelLibraryWisdom(
            library="onednn",
            topic="jit_code_generation",
            insight=(
                "oneDNN JIT-compiles optimized kernels at runtime for specific tensor "
                "shapes and CPU features. This eliminates the need to ship pre-compiled "
                "variants for every shape."
            ),
            conditions=["x86 CPU with AVX2 or later"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="onednn",
            topic="brgemm_primitive",
            insight=(
                "The batch-reduce GEMM (BRGEMM) is oneDNN's universal building block. "
                "Convolutions, attention, and other ops are lowered to BRGEMM calls, "
                "amortizing optimization effort."
            ),
            conditions=["x86 CPU", "AMX or AVX-512 available"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="onednn",
            topic="memory_format_propagation",
            insight=(
                "Let oneDNN choose optimal blocked formats (nChw16c, nChw8c, etc.) "
                "and propagate them through the graph. Inserting manual reorders at "
                "every op boundary destroys performance."
            ),
            conditions=["Graph of oneDNN-supported ops"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="onednn",
            topic="post_ops_fusion",
            insight=(
                "oneDNN supports appending post-ops (eltwise, sum, binary) to "
                "primitives. Fusing relu or residual add into convolution avoids "
                "separate passes over memory."
            ),
            conditions=["Post-op is element-wise or broadcast-compatible"],
            confidence=Confidence.HIGH,
        ),
        # -- Triton ----------------------------------------------------------
        KernelLibraryWisdom(
            library="triton",
            topic="block_pointers",
            insight=(
                "Block pointers (tl.make_block_ptr) enable the compiler to reason about "
                "structured memory access patterns and emit TMA instructions on Hopper. "
                "Prefer them over manual pointer arithmetic."
            ),
            conditions=["Hopper GPU for TMA", "Regular access patterns"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="triton",
            topic="dot_operand_placement",
            insight=(
                "tl.dot requires both operands to reside in shared memory (or registers) "
                "on pre-Hopper architectures. Failing to stage operands to shared memory "
                "before tl.dot causes correctness issues or poor performance."
            ),
            conditions=["Pre-Hopper GPU (sm_80, sm_86)"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="triton",
            topic="accumulator_dtype",
            insight=(
                "Always use fp32 accumulators with fp16/bf16 inputs in tl.dot. "
                "fp16 accumulation over large K dimensions leads to catastrophic "
                "numerical errors."
            ),
            conditions=["K dimension > 256", "fp16 or bf16 operands"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="triton",
            topic="autotuning_knobs",
            insight=(
                "Key autotuning parameters are BLOCK_M, BLOCK_N, BLOCK_K, num_warps, "
                "and num_stages. Sweep a grid of these; optimal values vary dramatically "
                "across shapes and GPUs."
            ),
            conditions=["Performance-critical kernel"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="triton",
            topic="mask_handling",
            insight=(
                "Use boolean masks for boundary conditions and pass other=0.0 to "
                "tl.load for safe out-of-bounds padding. Incorrect masking is the "
                "most common source of Triton kernel bugs."
            ),
            conditions=["Tensor dimensions not divisible by block size"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="triton",
            topic="persistent_reduction",
            insight=(
                "For reductions, keep partial results in registers across loop "
                "iterations instead of writing to shared memory. This turns a "
                "multi-pass algorithm into a single efficient pass."
            ),
            conditions=["Reduction dimension fits in register budget"],
            confidence=Confidence.HIGH,
        ),
        # -- Exo -------------------------------------------------------------
        KernelLibraryWisdom(
            library="exo",
            topic="cursor_navigation",
            insight=(
                "Use cursor-based navigation (proc.find(...) and cursor forwarding) "
                "for stable references to program points across transforms. Name-based "
                "references break when surrounding code changes."
            ),
            conditions=["Exo 2 or later"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="exo",
            topic="schedule_libraries",
            insight=(
                "Compose fine-grained scheduling primitives (split, reorder, "
                "stage_mem, set_memory, replace) into reusable scheduling functions "
                "that encapsulate target-specific patterns."
            ),
            conditions=["Reusable schedule across similar ops"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="exo",
            topic="config_hoisting",
            insight=(
                "Hoist stateful accelerator configuration instructions (e.g., Gemmini "
                "config_ld, config_matmul) out of hot loops to avoid redundant "
                "reconfiguration on every iteration."
            ),
            conditions=["Accelerator with stateful configuration registers"],
            confidence=Confidence.HIGH,
        ),
        KernelLibraryWisdom(
            library="exo",
            topic="replace_instruction",
            insight=(
                "Define hardware instructions with semantic bodies (EXO instruction "
                "definitions), then use replace_all to match equivalent loop nest "
                "patterns in the scheduled code. This ensures the mapping from loops "
                "to instructions is semantics-preserving."
            ),
            conditions=["Target has custom instructions with known semantics"],
            confidence=Confidence.HIGH,
        ),
    ]
