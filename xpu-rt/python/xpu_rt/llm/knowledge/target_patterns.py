"""Per-target-class optimization patterns.

Encodes hardware-specific optimization strategies distilled from production
compilers and kernel libraries (CUTLASS, cuDNN, oneDNN, Exo, Gemmini, BLIS,
MKL, Zephyr RTOS, etc.) as structured ``TargetPattern`` entries.
"""

from __future__ import annotations

import structlog

from compgen.llm.knowledge.base import Confidence, TargetPattern

log = structlog.get_logger()


def build_default_target_patterns() -> dict[str, list[TargetPattern]]:
    """Build the default library of per-target optimization patterns.

    Returns:
        Dictionary mapping target class names to their ``TargetPattern`` lists.
    """
    patterns: dict[str, list[TargetPattern]] = {}

    # =================================================================
    # GPU
    # =================================================================
    patterns["gpu"] = [
        # -- memory_hierarchy -------------------------------------------
        TargetPattern(
            target_class="gpu",
            category="memory_hierarchy",
            pattern_name="shared_memory_staging",
            description=(
                "Stage data from global memory into shared memory (scratchpad) for "
                "reuse across threads within a thread block.  Amortises the cost of "
                "high-latency HBM reads over many compute operations."
            ),
            implementation_notes=[
                "Allocate shared memory tile: __shared__ float tile[TILE_M][TILE_K].",
                "Cooperative load: each thread loads one or more elements from global memory.",
                "__syncthreads() before consuming the tile.",
                "Pad shared memory to TILE_K+1 columns to avoid 32-bank conflicts.",
                "Double-buffer: overlap next tile's load with current tile's compute.",
            ],
            source="CUTLASS/Triton",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="memory_hierarchy",
            pattern_name="l2_persistent_kernels",
            description=(
                "Persistent kernel design: launch exactly as many thread blocks as the "
                "GPU has SMs, and loop over work items within each block.  Improves L2 "
                "cache residency by keeping the working set in L2 across iterations."
            ),
            implementation_notes=[
                "Set grid_dim = num_SMs (or 2 * num_SMs for latency hiding).",
                "Each block loops: while (tile_idx = atomicAdd(&counter, 1)) < total_tiles.",
                "Tile ordering (Hilbert curve or row-major) affects L2 hit rate.",
                "Use cudaFuncSetAttribute to set max shared memory if needed.",
            ],
            source="CUTLASS",
            confidence=Confidence.MEDIUM,
        ),
        TargetPattern(
            target_class="gpu",
            category="memory_hierarchy",
            pattern_name="register_tiling",
            description=(
                "Tile the innermost compute loop into registers.  Each thread computes "
                "a small output sub-tile (e.g., 8x8 fp16 elements) entirely in registers, "
                "maximising arithmetic intensity before writing back."
            ),
            implementation_notes=[
                "Declare register arrays: float acc[TM][TN] for the output sub-tile.",
                "Load A and B fragments into registers from shared memory.",
                "Compute outer product: acc[i][j] += a[i] * b[j] in the inner loop.",
                "Monitor register usage: NVIDIA GPUs have 256 32-bit registers per thread; "
                "exceeding this causes spills to local memory.",
            ],
            source="CUTLASS/Triton",
            confidence=Confidence.HIGH,
        ),
        # -- parallelism ------------------------------------------------
        TargetPattern(
            target_class="gpu",
            category="parallelism",
            pattern_name="output_tile_mapping",
            description=(
                "Map each thread block to one output tile.  The 2D grid covers the "
                "output matrix (or tensor), and each block computes its tile by "
                "iterating over the reduction (K) dimension."
            ),
            implementation_notes=[
                "grid_dim = (ceil(M/TILE_M), ceil(N/TILE_N)).",
                "block_dim = (WARP_M * 32, WARP_N) where warps tile the output tile.",
                "K-loop is sequential within each block.",
                "For batched ops, add a third grid dimension for the batch.",
            ],
            source="CUDA programming guide",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="parallelism",
            pattern_name="warp_cooperative_ops",
            description=(
                "Use warp-level matrix operations (mma.sync / WMMA) to exploit tensor "
                "cores.  Multiple threads in a warp cooperatively compute a small "
                "matrix multiply (e.g., 16x16x16)."
            ),
            implementation_notes=[
                "Use nvcuda::wmma or inline PTX mma.sync.aligned.m16n8k16.",
                "Fragment types: matrix_a (row_major), matrix_b (col_major), accumulator.",
                "Each warp computes one fragment-sized output; warps tile the thread block's output.",
                "Supported types: fp16, bf16, tf32, int8, fp8 (Hopper).",
            ],
            source="NVIDIA PTX guide",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="parallelism",
            pattern_name="wave_quantisation",
            description=(
                "Choose grid dimensions so that the total number of thread blocks is a "
                "multiple of the number of SMs.  Partial waves (the last wave with fewer "
                "blocks than SMs) waste GPU resources."
            ),
            implementation_notes=[
                "Compute num_blocks = ceil(M/TILE_M) * ceil(N/TILE_N).",
                "Adjust TILE_M, TILE_N so num_blocks % num_SMs == 0 (or close to 0).",
                "For small problems, consider reducing tile sizes to fill more SMs.",
                "A100 has 108 SMs; H100 has 132 SMs.  Target multiples of these.",
            ],
            source="CUDA programming guide",
            confidence=Confidence.HIGH,
        ),
        # -- data_movement ----------------------------------------------
        TargetPattern(
            target_class="gpu",
            category="data_movement",
            pattern_name="async_copy",
            description=(
                "Use cp.async (Ampere+) to overlap global-to-shared-memory copies with "
                "computation.  The copy bypasses L1 cache and runs in the background "
                "while the SM executes math instructions."
            ),
            implementation_notes=[
                "Use cp.async.cg.shared.global for 16-byte copies.",
                "Group copies with cp.async.commit_group and cp.async.wait_group.",
                "Combine with double buffering: compute on buffer 0 while loading buffer 1.",
                "Hopper TMA (Tensor Memory Accelerator) further automates this.",
            ],
            source="CUTLASS/CUDA toolkit",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="data_movement",
            pattern_name="double_buffering",
            description=(
                "Allocate two shared memory buffers.  While the kernel computes on one "
                "buffer, the next tile is being loaded into the other.  Hides memory "
                "latency behind compute."
            ),
            implementation_notes=[
                "Declare smem[2][TILE_M][TILE_K] for double buffering.",
                "Iteration i: compute on smem[i % 2], load into smem[(i+1) % 2].",
                "Requires careful sync: __syncthreads() between compute and next load.",
                "Increases shared memory usage by 2x; ensure it fits in the SM budget.",
            ],
            source="CUTLASS",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="data_movement",
            pattern_name="l1_prefetch",
            description=(
                "Prefetch data into L1 cache for subsequent access.  Useful when access "
                "patterns are predictable but shared memory staging is not needed."
            ),
            implementation_notes=[
                "Use __ldg() for read-only data (uses texture cache path on some arches).",
                "LDG.128 (128-bit loads) maximise bandwidth utilisation.",
                "Cache hints: prefetch.global.L1 for data needed soon.",
            ],
            source="CUDA toolkit",
            confidence=Confidence.MEDIUM,
        ),
        # -- instruction_selection --------------------------------------
        TargetPattern(
            target_class="gpu",
            category="instruction_selection",
            pattern_name="tensor_core_mma",
            description=(
                "Select tensor core instructions (mma.sync) for matrix multiply-accumulate "
                "operations in fp16, bf16, tf32, int8, or fp8.  Delivers 2-16x higher "
                "throughput than CUDA cores for supported types."
            ),
            implementation_notes=[
                "fp16 mma.sync.aligned.m16n8k16: 256 FLOPs per instruction per warp.",
                "tf32 mma: use float inputs truncated to tf32 for 8x throughput with ~fp16 accuracy.",
                "int8 mma: for quantised inference, 2x throughput over fp16.",
                "Hopper fp8 (e4m3/e5m2): 2x throughput over fp16 mma.",
            ],
            source="NVIDIA PTX guide",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="instruction_selection",
            pattern_name="vector_loads",
            description=(
                "Use 128-bit vector loads (LDG.128) to maximise memory bandwidth.  "
                "Each load fetches 4 fp32 or 8 fp16 elements in one transaction."
            ),
            implementation_notes=[
                "Align data to 16 bytes for LDG.128.",
                "Use float4 or half8 types for 128-bit loads.",
                "Ensure consecutive threads access consecutive 128-bit words for coalescing.",
            ],
            source="NVIDIA PTX guide",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="gpu",
            category="instruction_selection",
            pattern_name="fma_instructions",
            description=(
                "Use fused multiply-add (FMA) instructions for better accuracy and "
                "throughput.  One FMA = 2 FLOPs with a single rounding."
            ),
            implementation_notes=[
                "CUDA FMA: fmaf(a, b, c) or __fmaf_rn(a, b, c).",
                "Most compilers emit FMA automatically; verify with SASS disassembly.",
                "FMA count is the basis for roofline peak FLOPS calculations.",
            ],
            source="NVIDIA PTX guide",
            confidence=Confidence.HIGH,
        ),
    ]

    # =================================================================
    # CPU
    # =================================================================
    patterns["cpu"] = [
        # -- memory_hierarchy -------------------------------------------
        TargetPattern(
            target_class="cpu",
            category="memory_hierarchy",
            pattern_name="l1_tiling",
            description=(
                "Tile innermost loops so the working set fits in L1 cache (32-64 KB per core).  "
                "For matmul, this means the micro-kernel's A panel and B panel must fit in L1."
            ),
            implementation_notes=[
                "Typical L1 tile: 6 rows of A (6 * K * 4B) + one column-panel of B (K * 16 * 4B).",
                "Total: ~(6 + 16) * K * 4B; for K=512, this is ~44 KB -- fits in 48 KB L1d.",
                "Access pattern must be sequential to benefit from hardware prefetcher.",
                "Align tiles to cache line boundaries (64 bytes).",
            ],
            source="oneDNN/BLIS",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="memory_hierarchy",
            pattern_name="l2_tiling",
            description=(
                "Tile outer loops so the working set fits in L2 cache (256 KB - 1 MB per core).  "
                "The GEBP (General Block Panel) pattern keeps a block of A in L2 while streaming "
                "panels of B through."
            ),
            implementation_notes=[
                "GEBP: A block is MC x KC, B panel is KC x NC.",
                "MC * KC * elem_size should fit in L2 (e.g., 256 * 512 * 4B = 512 KB).",
                "Iterate over NC-wide panels of B, reusing A block from L2.",
                "Pack A and B into contiguous buffers for sequential access.",
            ],
            source="BLIS/Exo",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="memory_hierarchy",
            pattern_name="register_blocking",
            description=(
                "Block the micro-kernel to use all available SIMD registers as accumulators.  "
                "For AVX-512, this means filling 32 ZMM registers with output tiles."
            ),
            implementation_notes=[
                "AVX-512: 32 ZMM registers, each holding 16 fp32 or 32 bf16 elements.",
                "Typical micro-kernel: 6x16 (6 rows x 16 cols = 6 ZMM accumulators), "
                "leaving registers for A and B fragments.",
                "AMX: 8 tile registers of 16x16 bytes each; use for int8/bf16 matmul.",
                "Register blocking is the innermost tiling level.",
            ],
            source="Exo/oneDNN",
            confidence=Confidence.HIGH,
        ),
        # -- parallelism ------------------------------------------------
        TargetPattern(
            target_class="cpu",
            category="parallelism",
            pattern_name="openmp_outer_parallel",
            description=(
                "Parallelise outer loops with OpenMP.  Each thread gets a chunk of the "
                "output to compute independently, minimising synchronisation."
            ),
            implementation_notes=[
                "#pragma omp parallel for schedule(static) on the outermost loop.",
                "Chunk size should give each thread at least one L2 tile.",
                "Avoid false sharing: ensure thread-local output tiles are on separate cache lines.",
                "Thread count = number of physical cores (not hyperthreads) for compute-bound work.",
            ],
            source="oneDNN/MKL",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="parallelism",
            pattern_name="simd_vectorisation",
            description=(
                "Vectorise inner loops with SIMD instructions (AVX-512, AVX2, NEON).  "
                "Process multiple data elements per instruction cycle."
            ),
            implementation_notes=[
                "AVX-512: 16 fp32 elements per instruction (512 bits).",
                "AVX2: 8 fp32 elements per instruction (256 bits).",
                "ARM NEON/SVE: 4-16 fp32 elements depending on vector length.",
                "Use intrinsics or let the compiler auto-vectorise with -O3 -march=native.",
                "Ensure loop trip count is a multiple of vector width (or add remainder loop).",
            ],
            source="oneDNN/MKL",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="parallelism",
            pattern_name="numa_aware_allocation",
            description=(
                "Allocate data on the NUMA node where it will be consumed.  Cross-node "
                "memory access has 2-3x higher latency and lower bandwidth."
            ),
            implementation_notes=[
                "Use numactl --membind or libnuma's numa_alloc_onnode().",
                "First-touch policy: initialise data on the thread that will use it.",
                "For multi-socket systems, partition work so each socket operates on local memory.",
                "Check topology with numactl -H or hwloc.",
            ],
            source="oneDNN",
            confidence=Confidence.HIGH,
        ),
        # -- data_movement ----------------------------------------------
        TargetPattern(
            target_class="cpu",
            category="data_movement",
            pattern_name="software_prefetch",
            description=(
                "Issue software prefetch instructions to bring data into cache before "
                "it is needed, hiding memory latency for predictable access patterns."
            ),
            implementation_notes=[
                "_mm_prefetch(ptr, _MM_HINT_T0) for L1 prefetch.",
                "_mm_prefetch(ptr, _MM_HINT_T1) for L2 prefetch.",
                "Prefetch 2-4 cache lines ahead of the current access.",
                "Effective for sequential and strided access; useless for random access.",
            ],
            source="BLIS/MKL",
            confidence=Confidence.MEDIUM,
        ),
        TargetPattern(
            target_class="cpu",
            category="data_movement",
            pattern_name="cache_line_alignment",
            description=(
                "Align data structures and allocation boundaries to cache line size "
                "(typically 64 bytes) to avoid false sharing and partial cache line loads."
            ),
            implementation_notes=[
                "Use aligned_alloc(64, size) or posix_memalign(&ptr, 64, size).",
                "Pad structure members to 64-byte boundaries for thread-local data.",
                "Matrix leading dimensions should be multiples of 16 (64B / 4B per float).",
            ],
            source="BLIS/oneDNN",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="data_movement",
            pattern_name="matrix_packing",
            description=(
                "Pack matrices into contiguous buffers with the micro-kernel's access "
                "pattern.  Converts arbitrary stride layouts into sequential access, "
                "eliminating TLB misses on large matrices."
            ),
            implementation_notes=[
                "Pack B into column-panels of width NR (e.g., 16 for AVX-512).",
                "Pack A into row-panels of height MR (e.g., 6 for the 6x16 micro-kernel).",
                "Packing is O(N^2) for an NxN matrix; amortised over the O(N^3) compute.",
                "Pack into pre-allocated buffers to avoid allocation overhead.",
            ],
            source="BLIS/MKL",
            confidence=Confidence.HIGH,
        ),
        # -- instruction_selection --------------------------------------
        TargetPattern(
            target_class="cpu",
            category="instruction_selection",
            pattern_name="avx512_fp32",
            description=(
                "Use AVX-512 instructions for fp32 computation.  Process 16 fp32 "
                "elements per instruction with ZMM registers."
            ),
            implementation_notes=[
                "vfmadd231ps: fused multiply-add, 16 FLOPs per instruction.",
                "vmovups/vmovaps for 512-bit loads/stores.",
                "vbroadcastss for scalar broadcast into ZMM register.",
                "Throughput: 2 FMA per cycle on Skylake-X and later (64 FLOPs/cycle/core).",
            ],
            source="Exo/oneDNN",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="cpu",
            category="instruction_selection",
            pattern_name="amx_matmul",
            description=(
                "Use Intel AMX (Advanced Matrix Extensions) for int8 and bf16 matrix "
                "multiply.  Each TDPB* instruction computes a 16x16 tile in one cycle."
            ),
            implementation_notes=[
                "8 tile registers (tmm0-tmm7), each 16 rows x 64 bytes (16x16 int32 accumulator).",
                "TDPBSSD: int8 matmul with int32 accumulation.",
                "TDPBF16PS: bf16 matmul with fp32 accumulation.",
                "Requires explicit tile configuration via LDTILECFG.",
                "Available on Sapphire Rapids and later.",
            ],
            source="Exo/oneDNN",
            confidence=Confidence.HIGH,
        ),
    ]

    # =================================================================
    # NPU / Accelerator
    # =================================================================
    patterns["npu"] = [
        # -- memory_hierarchy -------------------------------------------
        TargetPattern(
            target_class="npu",
            category="memory_hierarchy",
            pattern_name="explicit_scratchpad",
            description=(
                "NPU/accelerator local memory is a software-managed scratchpad, not a "
                "hardware cache.  All data movement must be explicitly programmed via "
                "DMA or load/store instructions."
            ),
            implementation_notes=[
                "Scratchpad sizes are typically 64 KB - 512 KB per PE or cluster.",
                "Partition scratchpad into input buffer, output buffer, and weight buffer.",
                "Size each buffer to hold exactly one tile of the computation.",
                "No cache coherence: software must manage consistency.",
            ],
            source="Gemmini/Exo",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="memory_hierarchy",
            pattern_name="dma_data_movement",
            description=(
                "Use DMA engines for bulk data transfer between main memory and "
                "scratchpad.  DMA runs independently of the compute units, enabling "
                "overlap of data movement and computation."
            ),
            implementation_notes=[
                "Configure DMA: src_addr, dst_addr, transfer_size, stride, padding.",
                "2D DMA for strided access (e.g., extracting a tile from a larger matrix).",
                "DMA completion is signalled via interrupt or polling a status register.",
                "Tile sizes must account for DMA setup overhead (~100-1000 cycles).",
            ],
            source="Gemmini/Exo",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="memory_hierarchy",
            pattern_name="software_managed_cache",
            description=(
                "Some accelerators provide a software-managed cache (SMC) that behaves "
                "like a scratchpad with cache-like lookup.  Requires explicit "
                "allocate/invalidate but supports flexible reuse patterns."
            ),
            implementation_notes=[
                "Allocate cache lines explicitly for tiles that will be reused.",
                "Invalidate after use to free space for new tiles.",
                "Useful for irregular access patterns where pure scratchpad is wasteful.",
            ],
            source="Exo",
            confidence=Confidence.LOW,
        ),
        # -- parallelism ------------------------------------------------
        TargetPattern(
            target_class="npu",
            category="parallelism",
            pattern_name="systolic_array_mapping",
            description=(
                "Map matrix multiply onto a systolic array: data flows through a 2D "
                "grid of processing elements (PEs), with each PE performing a "
                "multiply-accumulate and forwarding data to its neighbour."
            ),
            implementation_notes=[
                "Output-stationary: each PE accumulates one output element.",
                "Weight-stationary: each PE holds one weight, inputs flow through.",
                "Row-stationary: each PE processes one row of the output.",
                "Tile sizes must match the PE array dimensions (e.g., 16x16 or 32x32).",
                "Underutilisation when problem dimensions do not divide array dimensions.",
            ],
            source="Gemmini",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="parallelism",
            pattern_name="pe_utilisation",
            description=(
                "Maximise PE array utilisation by choosing tile sizes that fill the array.  "
                "Partial tiles waste compute: a 10x10 problem on a 16x16 array uses only "
                "39% of PEs."
            ),
            implementation_notes=[
                "Pad matrices to multiples of the array dimension when possible.",
                "For non-square problems, consider remapping to better fill the array.",
                "Monitor utilisation: effective_ops / peak_ops ratio.",
                "Pipeline multiple tiles through the array for throughput.",
            ],
            source="Gemmini",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="parallelism",
            pattern_name="pipeline_stages",
            description=(
                "Pipeline the computation into stages: load, compute, store.  "
                "Each stage operates on a different tile, achieving steady-state "
                "throughput where all units are busy."
            ),
            implementation_notes=[
                "Three-stage pipeline: DMA_in[i+1] | Compute[i] | DMA_out[i-1].",
                "Requires at least 3 tile buffers in scratchpad (triple buffering).",
                "Steady-state throughput = max(DMA_time, compute_time) per tile.",
                "Pipeline fill and drain add latency for small workloads.",
            ],
            source="Gemmini/Exo",
            confidence=Confidence.HIGH,
        ),
        # -- data_movement ----------------------------------------------
        TargetPattern(
            target_class="npu",
            category="data_movement",
            pattern_name="dma_double_buffering",
            description=(
                "Overlap DMA transfers with computation using double buffering.  While "
                "the accelerator computes on buffer A, DMA loads the next tile into "
                "buffer B, and vice versa."
            ),
            implementation_notes=[
                "Allocate 2x scratchpad space: buffer[0] and buffer[1].",
                "Iteration i: compute(buffer[i%2]), DMA_load(buffer[(i+1)%2]).",
                "Synchronise: wait for DMA completion before computing on new buffer.",
                "Effective when compute_time >= DMA_time (compute-bound regime).",
            ],
            source="Exo/Gemmini",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="data_movement",
            pattern_name="explicit_address_computation",
            description=(
                "Compute scratchpad addresses explicitly in software.  There is no "
                "hardware address translation (no TLB, no virtual memory)."
            ),
            implementation_notes=[
                "Base address + row * stride + col * element_size.",
                "Account for padding/alignment requirements of the DMA engine.",
                "Tile layout in scratchpad may differ from layout in main memory.",
                "Exo's memory-aware scheduling handles this automatically.",
            ],
            source="Exo",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="data_movement",
            pattern_name="blocking_for_local_memory",
            description=(
                "Choose tile sizes so each tile fits in the accelerator's local memory.  "
                "The tile must include input, output, and any intermediate buffers."
            ),
            implementation_notes=[
                "Available = scratchpad_size - firmware_reserved.",
                "Tile memory = input_tile + output_tile + weight_tile (for matmul).",
                "If double buffering: 2 * tile_memory <= available.",
                "Iterate over tiles in an order that maximises data reuse.",
            ],
            source="Exo/Gemmini",
            confidence=Confidence.HIGH,
        ),
        # -- instruction_selection --------------------------------------
        TargetPattern(
            target_class="npu",
            category="instruction_selection",
            pattern_name="matrix_engine_instructions",
            description=(
                "Use the accelerator's custom matrix engine instructions (e.g., Gemmini's "
                "mvin/mvout/compute, Google TPU's MXU ops) for matrix multiply."
            ),
            implementation_notes=[
                "Instructions are typically: load_to_accumulator, matmul_accumulate, store_from_accumulator.",
                "Operand sizes must match the hardware array dimensions.",
                "Accumulator precision is usually higher than input precision (e.g., int8 -> int32).",
                "Some engines require explicit accumulator initialisation.",
            ],
            source="Gemmini/Exo",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="npu",
            category="instruction_selection",
            pattern_name="config_state_management",
            description=(
                "Many accelerators require explicit configuration of compute mode, "
                "precision, and layout before issuing compute instructions."
            ),
            implementation_notes=[
                "Config instruction sets: dataflow mode, activation function, precision.",
                "Configuration changes have latency; batch operations with the same config.",
                "Save/restore config state when switching between op types.",
                "Exo platform definitions encode legal configurations.",
            ],
            source="Exo/Gemmini",
            confidence=Confidence.MEDIUM,
        ),
    ]

    # Also register under "accelerator" as an alias
    patterns["accelerator"] = list(patterns["npu"])

    # =================================================================
    # SoC (Heterogeneous)
    # =================================================================
    patterns["soc"] = [
        # -- data_movement ----------------------------------------------
        TargetPattern(
            target_class="soc",
            category="data_movement",
            pattern_name="minimise_cross_domain_transfers",
            description=(
                "Minimise data transfers between heterogeneous domains (e.g., host CPU "
                "to accelerator, CPU to GPU, FPGA to CPU).  Cross-domain transfers incur "
                "high latency (PCIe: ~5-15 us, on-chip bus: ~100-500 ns) and limited bandwidth."
            ),
            implementation_notes=[
                "Batch multiple ops on the accelerator before transferring results back.",
                "Use device-resident tensors: keep data on the accelerator across op boundaries.",
                "Profile transfer costs with the runtime profiler; compare to compute savings.",
                "Consider computation duplication on both domains if transfer cost exceeds recompute.",
            ],
            source="Zephyr RTOS patterns",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="soc",
            category="data_movement",
            pattern_name="dma_bulk_transfer",
            description=(
                "Use DMA engines for bulk data transfers between domains.  DMA avoids "
                "tying up the CPU for data movement and can achieve higher bandwidth "
                "than programmed I/O."
            ),
            implementation_notes=[
                "Configure DMA channel: source, destination, size, callback.",
                "Use scatter-gather DMA for non-contiguous buffers.",
                "Double-buffer: CPU fills buffer A while DMA transfers buffer B.",
                "Align buffers to DMA burst size (typically 32-128 bytes).",
            ],
            source="Zephyr RTOS patterns",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="soc",
            category="data_movement",
            pattern_name="latency_sensitive_on_coordinator",
            description=(
                "Keep latency-sensitive operations (control flow, branching, small "
                "scalar computations) on the host CPU (coordinator).  Only offload "
                "compute-intensive parallel work to accelerators."
            ),
            implementation_notes=[
                "Control flow (if/else, loop bounds) on CPU; data-parallel compute on accelerator.",
                "Scalar reductions: reduce on accelerator, transfer single scalar to CPU.",
                "Dynamic shapes: compute on CPU, compile/dispatch static tiles to accelerator.",
                "Runtime decision logic (scheduling, memory allocation) always on CPU.",
            ],
            source="CompGen runtime",
            confidence=Confidence.HIGH,
        ),
        # -- parallelism ------------------------------------------------
        TargetPattern(
            target_class="soc",
            category="parallelism",
            pattern_name="heterogeneous_assignment",
            description=(
                "Assign operations to the most appropriate domain: compute-heavy parallel "
                "work to the accelerator (GPU/NPU), sequential control to the host CPU, "
                "and I/O-bound work to DMA engines."
            ),
            implementation_notes=[
                "Profile each op: FLOPs, memory access, control flow complexity.",
                "Ops with high arithmetic intensity -> accelerator.",
                "Ops with high branch divergence -> CPU.",
                "Ops with bulk sequential I/O -> DMA.",
                "Use the solver (CP-SAT) to optimise the assignment globally.",
            ],
            source="CompGen runtime",
            confidence=Confidence.HIGH,
        ),
        TargetPattern(
            target_class="soc",
            category="parallelism",
            pattern_name="cross_domain_pipeline",
            description=(
                "Pipeline work across domains: CPU prepares inputs while accelerator "
                "computes previous batch, and DMA transfers results from the batch before that."
            ),
            implementation_notes=[
                "Three-stage pipeline: CPU_prepare[i+1] | Accel_compute[i] | DMA_writeback[i-1].",
                "Use event/semaphore signalling between domains for synchronisation.",
                "Pipeline depth trades latency for throughput.",
                "Monitor pipeline bubbles: if one stage is much slower, it bottlenecks the pipeline.",
            ],
            source="CompGen runtime",
            confidence=Confidence.MEDIUM,
        ),
    ]

    log.info("built_default_target_patterns", target_count=len(patterns))
    return patterns
