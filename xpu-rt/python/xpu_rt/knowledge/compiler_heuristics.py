"""Heuristics extracted from production compiler frameworks.

Each entry captures one actionable rule-of-thumb from TVM, Halide, IREE, XLA,
or another compiler, together with the conditions under which it applies.
"""

from __future__ import annotations

from compgen.knowledge.base import (
    CompilerHeuristic,
    Confidence,
)


def build_default_compiler_heuristics() -> list[CompilerHeuristic]:
    """Return the default catalogue of compiler heuristics.

    Returns:
        List of ``CompilerHeuristic`` entries (20+) covering TVM, Halide,
        IREE, and XLA.
    """
    return [
        # -- TVM (6) ---------------------------------------------------------
        CompilerHeuristic(
            compiler="tvm",
            topic="auto_scheduling",
            heuristic=(
                "Start from a naive loop nest and apply a fixed sequence of sketch "
                "rules: multi-level tiling, fusion, cache read/write insertion, "
                "vectorization, and parallelization. Each rule generates candidate "
                "programs; a cost model prunes the search space."
            ),
            conditions=["Using Ansor auto-scheduler"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="tvm",
            topic="cost_model",
            heuristic=(
                "Train an XGBoost model on profiled program features (loop structure, "
                "memory access patterns, arithmetic intensity) to predict kernel "
                "runtime. Transfer-learning from similar workloads reduces measurement "
                "budget on new targets."
            ),
            conditions=["AutoTVM or Ansor with profiling data"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="tvm",
            topic="operator_fusion",
            heuristic=(
                "Classify ops as injective, reduction, or complex. Fuse injective ops "
                "greedily into their consumers. Fuse elementwise ops into the epilogue "
                "of reductions. Never fuse two reductions unless they share the same "
                "reduction axis."
            ),
            conditions=["Relay or Relax graph-level optimization"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="tvm",
            topic="layout_transformation",
            heuristic=(
                "Insert explicit layout_transform ops at boundaries between subgraphs "
                "with different preferred layouts (e.g., NCHW for compute, NCHWc for "
                "vectorized inner kernel). Run a global pass to minimize total "
                "transformation cost."
            ),
            conditions=["Heterogeneous layout preferences across ops"],
            confidence=Confidence.MEDIUM,
        ),
        CompilerHeuristic(
            compiler="tvm",
            topic="dynamic_shapes",
            heuristic=(
                "Use symbolic shape variables with declared upper bounds. Pad tensors "
                "to the upper bound at kernel boundaries and mask out padding. This "
                "keeps the kernel code static while handling dynamic input sizes."
            ),
            conditions=["Dynamic batch size or sequence length"],
            confidence=Confidence.MEDIUM,
        ),
        CompilerHeuristic(
            compiler="tvm",
            topic="schedule_templates",
            heuristic=(
                "Define a schedule template that encodes the high-level structure "
                "(number of tile levels, parallelism strategy) and leave tile sizes, "
                "unroll factors, and vectorization widths as tunable parameters. "
                "Search fills in the concrete values."
            ),
            conditions=["AutoTVM template-based tuning"],
            confidence=Confidence.HIGH,
        ),
        # -- Halide (5) ------------------------------------------------------
        CompilerHeuristic(
            compiler="halide",
            topic="compute_at_vs_store_at",
            heuristic=(
                "compute_at places a producer's computation inside a consumer's loop "
                "for fusion. store_at controls where the intermediate buffer is "
                "allocated. Use compute_at for fusion, and store_at at a coarser loop "
                "level when the producer's values are reused across consumer iterations."
            ),
            conditions=["Producer-consumer pair in a Halide pipeline"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="halide",
            topic="bound_inference",
            heuristic=(
                "Halide automatically infers the required input bounds for each stage "
                "by backward-propagating output bounds through the dependency graph. "
                "The compiler inserts boundary checks or clamping. Trust the inference "
                "rather than manually computing bounds."
            ),
            conditions=["Standard Halide pipeline"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="halide",
            topic="sliding_window",
            heuristic=(
                "For stencil computations, compute only the newly required rows of the "
                "producer as the consumer slides over the output. This reduces redundant "
                "computation from O(filter_size * output) to O(output)."
            ),
            conditions=["Stencil/convolution with spatial reuse", "Consumer iterates in order"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="halide",
            topic="unroll_and_vectorize",
            heuristic=(
                "Unroll small inner loops to expose instruction-level parallelism, then "
                "vectorize over the fastest-varying dimension (innermost contiguous). "
                "Unrolling the vectorized dimension itself wastes registers."
            ),
            conditions=["Inner loop trip count small and known at compile time"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="halide",
            topic="split_reorder_parallelize",
            heuristic=(
                "Split the outermost loop to create parallel tiles, reorder loops to "
                "place the parallel dimension outermost and the vectorizable dimension "
                "innermost. This is the canonical parallelization pattern."
            ),
            conditions=["Sufficient output size for parallelism"],
            confidence=Confidence.HIGH,
        ),
        # -- IREE (5) --------------------------------------------------------
        CompilerHeuristic(
            compiler="iree",
            topic="flow_stream_hal",
            heuristic=(
                "IREE decomposes execution into three levels: Flow (coarse-grained "
                "device placement and partitioning), Stream (fine-grained asynchronous "
                "ordering within a device), and HAL (hardware dispatch and resource "
                "management). Map high-level decisions to the appropriate level."
            ),
            conditions=["IREE-based compilation pipeline"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="iree",
            topic="tiling_heuristics",
            heuristic=(
                "Tile to workgroup level first (mapping to GPU thread blocks or CPU "
                "threads), then tile again within each workgroup to distribute work "
                "across invocations. Workgroup tile sizes should match shared memory "
                "capacity; invocation tiles should match register file size."
            ),
            conditions=["GPU or multi-core CPU target"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="iree",
            topic="buffer_planning",
            heuristic=(
                "Perform lifetime analysis on stream-ordered commands and reuse buffers "
                "whose lifetimes do not overlap. This reduces peak memory without "
                "inserting unnecessary synchronization."
            ),
            conditions=["Static execution plan or bounded dynamic shapes"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="iree",
            topic="dispatch_region_formation",
            heuristic=(
                "Group ops into dispatch regions that execute together on one device. "
                "The grouping respects data dependencies and device affinity. Minimize "
                "the number of dispatches to reduce launch overhead."
            ),
            conditions=["IREE Flow dialect"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="iree",
            topic="executable_caching",
            heuristic=(
                "Compile each dispatch executable once and cache the resulting VMFB. "
                "On re-execution, load from cache to skip compilation. Keyed by "
                "source hash, target triple, and compilation flags."
            ),
            conditions=["Repeated execution of the same model"],
            confidence=Confidence.HIGH,
        ),
        # -- XLA (5) ---------------------------------------------------------
        CompilerHeuristic(
            compiler="xla",
            topic="operator_fusion",
            heuristic=(
                "Fuse element-wise ops greedily into their consumers. Stop at reduction "
                "boundaries (reduce, reduce-window) because reductions change the "
                "parallelism structure. After fusion, each fused computation becomes "
                "a single HLO fusion instruction."
            ),
            conditions=["XLA HLO optimization pipeline"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="xla",
            topic="spmd_partitioning",
            heuristic=(
                "Annotate tensors with sharding specs over a device mesh. The GSPMD "
                "partitioner propagates sharding through the graph and inserts "
                "all-gather, all-reduce, and reduce-scatter collectives at boundaries. "
                "Minimize cross-device communication by aligning sharding with "
                "computation locality."
            ),
            conditions=["Multi-device execution", "TPU or multi-GPU"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="xla",
            topic="memory_optimization",
            heuristic=(
                "Schedule HLO instructions to minimize peak memory by using liveness "
                "analysis: free buffers as soon as their last consumer executes. "
                "Recomputation (rematerialization) trades compute for memory when peak "
                "is too high."
            ),
            conditions=["Memory-constrained device (GPU, TPU)"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="xla",
            topic="buffer_assignment",
            heuristic=(
                "Share physical buffers between non-interfering logical buffers using "
                "an interference graph. Color the graph to minimize total memory. "
                "Respect alignment constraints and aliasing rules."
            ),
            conditions=["Static shape graph"],
            confidence=Confidence.HIGH,
        ),
        CompilerHeuristic(
            compiler="xla",
            topic="algebraic_simplification",
            heuristic=(
                "Run algebraic simplifications early: constant folding, strength "
                "reduction (x * 2 -> x + x), dead computation elimination, and "
                "identity removal (add 0, multiply 1). These reduce graph size before "
                "expensive optimization passes."
            ),
            conditions=["Any HLO graph"],
            confidence=Confidence.HIGH,
        ),
    ]
