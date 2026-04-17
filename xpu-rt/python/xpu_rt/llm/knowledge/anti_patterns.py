"""Common optimization anti-patterns to avoid.

Each entry describes a known-bad pattern, the symptoms it produces, and how to
fix it.  The LLM proposal engine can reference these to avoid generating recipes
that repeat well-known mistakes.
"""

from __future__ import annotations

from compgen.llm.knowledge.base import AntiPattern


def build_default_anti_patterns() -> list[AntiPattern]:
    """Return the default catalogue of optimization anti-patterns.

    Returns:
        List of ``AntiPattern`` entries (15+) covering tiling, fusion,
        memory access, numerics, and multi-device pitfalls.
    """
    return [
        AntiPattern(
            name="tile_too_small",
            description=(
                "Choosing tile sizes that are too small causes launch overhead and "
                "loop control overhead to dominate useful computation. Each tile "
                "carries fixed costs (kernel launch, barrier setup, index math) that "
                "must be amortized over enough work."
            ),
            symptoms=[
                "Kernel launch overhead dominates runtime profile",
                "Low arithmetic intensity per tile",
                "GPU occupancy is high but throughput is low",
            ],
            fix=(
                "Increase tile sizes until compute-to-overhead ratio is favorable. "
                "For GPU, aim for at least 64x64 output tiles. Profile to confirm "
                "launch overhead is negligible."
            ),
            source="CUTLASS tuning guide / CUDA best practices",
        ),
        AntiPattern(
            name="tile_too_large",
            description=(
                "Choosing tile sizes that are too large causes register spilling to "
                "local memory and shared memory overflow, both of which destroy "
                "performance. On CPUs, overly large tiles cause L1/L2 cache thrashing."
            ),
            symptoms=[
                "High local memory (lmem) traffic in profiler",
                "Shared memory allocation exceeds hardware limit (compilation failure)",
                "Register spill count is non-zero in PTX/SASS analysis",
                "L1 cache miss rate spikes on CPU",
            ],
            fix=(
                "Reduce tile sizes to fit within shared memory budget and register "
                "file. Use occupancy calculator to verify. For CPU, tile to L1/L2 "
                "cache size."
            ),
            source="CUTLASS tuning guide / Triton autotuning",
        ),
        AntiPattern(
            name="fuse_across_reduction_without_sync",
            description=(
                "Fusing operations across a reduction boundary without proper "
                "synchronization produces incorrect results. A reduction changes "
                "the parallelism structure; downstream ops depend on the fully "
                "reduced value."
            ),
            symptoms=[
                "Numerical mismatches vs reference implementation",
                "Results change with different thread counts or block sizes",
                "Race conditions detected by compute-sanitizer",
            ],
            fix=(
                "Insert a barrier or kernel boundary at reduction boundaries. "
                "Only fuse across a reduction if the fused kernel correctly "
                "implements the two-phase reduction pattern (partial + final)."
            ),
            source="TVM fusion rules / XLA fusion pass",
        ),
        AntiPattern(
            name="naive_transpose",
            description=(
                "Transposing a matrix by directly swapping indices causes "
                "uncoalesced global memory access on GPUs. One of the two accesses "
                "(read or write) will stride across memory, reducing effective "
                "bandwidth by 8-32x."
            ),
            symptoms=[
                "Global memory throughput far below peak bandwidth",
                "High L2 sector-miss rate",
                "Transpose kernel slower than expected for a memory-bound op",
            ],
            fix=(
                "Use shared memory as a staging buffer: read a tile coalesced into "
                "shared memory, apply the transpose in shared memory (with padding "
                "to avoid bank conflicts), then write the tile coalesced to global."
            ),
            source="CUDA transpose optimization guide",
        ),
        AntiPattern(
            name="materialize_attention_matrix",
            description=(
                "Computing the full N x N attention matrix before applying softmax "
                "and V multiplication consumes O(N^2) memory, making it infeasible "
                "for long sequences and wasting memory bandwidth."
            ),
            symptoms=[
                "Out-of-memory errors on long sequences (N > 2048)",
                "Memory usage grows quadratically with sequence length",
                "Attention computation dominates both time and memory",
            ],
            fix=(
                "Use tiled attention (FlashAttention) to compute attention in blocks "
                "with O(N) memory. Fuse Q@K, softmax, and @V within each tile using "
                "online softmax."
            ),
            source="FlashAttention paper",
        ),
        AntiPattern(
            name="separate_elementwise_kernels",
            description=(
                "Launching a separate GPU kernel for each elementwise operation "
                "(relu, add, mul, etc.) wastes kernel launch overhead and forces "
                "each intermediate result to round-trip through global memory."
            ),
            symptoms=[
                "Many small kernels in profiler trace",
                "Kernel launch overhead visible in timeline",
                "Low compute utilization despite high SM activity",
                "Memory bandwidth is the bottleneck for trivially compute-bound ops",
            ],
            fix=(
                "Fuse chains of elementwise ops into a single kernel. Most compiler "
                "frameworks (TVM, XLA, Triton) do this automatically. For manual "
                "kernels, write a single kernel with a grid-stride loop."
            ),
            source="XLA fusion / TVM operator fusion",
        ),
        AntiPattern(
            name="unaligned_memory_access",
            description=(
                "Issuing loads and stores at addresses that are not aligned to the "
                "access width (e.g., 128-bit loads at non-16-byte-aligned addresses) "
                "forces multiple memory transactions and can be 2-8x slower."
            ),
            symptoms=[
                "Low global memory throughput despite high access volume",
                "Excessive memory transactions per request in profiler",
                "Performance cliff when tensor dimensions change slightly",
            ],
            fix=(
                "Pad tensor dimensions to alignment boundaries (typically 128 bits / "
                "16 bytes). Ensure base pointers are aligned. Use aligned load/store "
                "intrinsics when available."
            ),
            source="CUDA C++ Programming Guide / Triton best practices",
        ),
        AntiPattern(
            name="wrong_memory_space",
            description=(
                "Keeping data in global memory when it could reside in shared memory "
                "or registers wastes bandwidth. Conversely, staging data to shared "
                "memory with no reuse just adds overhead."
            ),
            symptoms=[
                "High global memory traffic for data that is reused within a block",
                "Shared memory allocation for data accessed only once",
                "Performance does not improve with shared memory staging",
            ],
            fix=(
                "Stage to shared memory only when reuse factor > 1 across threads "
                "in a block. Keep scalars and per-thread accumulators in registers. "
                "Use the roofline model to determine whether you are bandwidth-bound."
            ),
            source="CUDA best practices / CUTLASS design",
        ),
        AntiPattern(
            name="missing_barrier",
            description=(
                "Omitting __syncthreads() (or equivalent barrier) between shared "
                "memory writes and reads causes race conditions. Some threads read "
                "stale or partially written data."
            ),
            symptoms=[
                "Non-deterministic numerical results across runs",
                "Results differ between debug and release builds",
                "compute-sanitizer reports hazards",
            ],
            fix=(
                "Insert __syncthreads() after all threads have written to shared "
                "memory and before any thread reads from it. Minimize barriers by "
                "restructuring the algorithm to reduce shared memory communication "
                "points."
            ),
            source="CUDA C++ Programming Guide",
        ),
        AntiPattern(
            name="wave_quantization_waste",
            description=(
                "If the grid (total thread blocks) is not a multiple of the number "
                "of SMs, the last wave has idle SMs. For small grids, this causes "
                "significant utilization loss."
            ),
            symptoms=[
                "Achieved occupancy much lower than theoretical occupancy",
                "GPU utilization drops for certain problem sizes",
                "Performance cliff at specific batch sizes or tile configurations",
            ],
            fix=(
                "Choose tile sizes so that the total number of blocks is a multiple "
                "of the SM count, or use persistent kernels that keep all SMs busy. "
                "Pad the grid if needed."
            ),
            source="CUDA occupancy analysis / CUTLASS persistent kernels",
        ),
        AntiPattern(
            name="over_parallelization",
            description=(
                "Launching more threads than there is useful work creates idle warps "
                "that still consume scheduling resources. On CPUs, over-subscribing "
                "cores with threads adds context-switch overhead."
            ),
            symptoms=[
                "Many threads with zero or trivial work in profiler",
                "High warp stall rate due to insufficient work per warp",
                "CPU performance degrades beyond optimal thread count",
            ],
            fix=(
                "Size the grid to match the work: one thread block per output tile, "
                "one thread per element within the tile. On CPU, use thread count "
                "equal to physical core count, not logical."
            ),
            source="CUDA best practices / oneDNN threading guide",
        ),
        AntiPattern(
            name="fp16_accumulation_instability",
            description=(
                "Accumulating many fp16 values (e.g., a dot product over K > 256) "
                "in fp16 precision causes catastrophic loss of significance. The "
                "limited dynamic range and mantissa bits of fp16 make large sums "
                "inaccurate."
            ),
            symptoms=[
                "Large numerical errors vs fp32 reference",
                "Errors grow with K dimension or sequence length",
                "Training diverges or evaluation metrics degrade",
            ],
            fix=(
                "Use fp32 accumulators with fp16/bf16 operands. Tensor cores "
                "natively support this mixed-precision mode. For Triton, set "
                "acc_dtype=tl.float32 in tl.dot."
            ),
            source="NVIDIA mixed-precision guide / Triton best practices",
        ),
        AntiPattern(
            name="cache_thrashing_poor_tiling",
            description=(
                "Tile sizes that do not align with cache line sizes or cache "
                "capacity cause repeated eviction and re-fetch of the same data "
                "(thrashing). This is especially severe for matrix operations with "
                "strided access."
            ),
            symptoms=[
                "Cache miss rate far above expected for working set size",
                "Performance varies non-monotonically with tile size",
                "Hardware performance counters show high LLC misses",
            ],
            fix=(
                "Tile to fit within the target cache level (L1: 32-64KB, L2: 256KB-1MB). "
                "Align tile boundaries to cache line size (64 bytes). For matrices, "
                "tile both dimensions to keep the working set within cache."
            ),
            source="Halide scheduling guide / oneDNN tuning",
        ),
        AntiPattern(
            name="dma_without_double_buffering",
            description=(
                "Issuing DMA transfers synchronously (load, wait, compute, store) "
                "stalls the compute pipeline while waiting for data. Without double "
                "buffering or pipelining, compute and memory transfer cannot overlap."
            ),
            symptoms=[
                "Compute unit idle time visible in profiler",
                "Achieved throughput well below roofline for both compute and memory",
                "Performance does not improve with faster compute",
            ],
            fix=(
                "Use double or multi-buffering: while computing on buffer A, "
                "prefetch the next tile into buffer B. On GPUs, use cp.async with "
                "multiple pipeline stages. On NPUs, overlap DMA with systolic "
                "execution."
            ),
            source="CUTLASS async pipeline / Gemmini double buffering",
        ),
        AntiPattern(
            name="single_device_placement",
            description=(
                "Placing all operations on a single device in a multi-device system "
                "leaves other devices idle and may cause memory pressure on the "
                "overloaded device."
            ),
            symptoms=[
                "Only one device shows utilization in system monitor",
                "Out-of-memory on one device while others have free memory",
                "Training/inference time scales linearly despite multiple devices",
            ],
            fix=(
                "Partition the computation across devices using pipeline parallelism, "
                "tensor parallelism, or data parallelism. Use the solver-backed "
                "placement planner to minimize cross-device communication."
            ),
            source="XLA GSPMD / IREE multi-device",
        ),
        AntiPattern(
            name="excessive_synchronization",
            description=(
                "Inserting unnecessary barriers or device synchronizations (e.g., "
                "cudaDeviceSynchronize after every kernel) serializes execution and "
                "prevents the hardware from overlapping independent work."
            ),
            symptoms=[
                "Profiler shows large gaps between kernel launches",
                "GPU utilization is low despite many kernels",
                "Removing sync points speeds up execution significantly",
            ],
            fix=(
                "Use fine-grained synchronization: CUDA events or stream ordering "
                "instead of device-wide sync. Only synchronize when there is a true "
                "data dependency between operations."
            ),
            source="CUDA best practices / IREE stream dialect",
        ),
    ]
