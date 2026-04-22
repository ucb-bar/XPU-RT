"""Seed the knowledge store with lessons accumulated during the
foundational sessions (TinyLlama-1.1B e2e, FlashAttention MEGA, CUDA
graph eval, Triton autotune persistence).

Idempotent — safe to call multiple times; lessons are matched by
``(scope, summary)`` so duplicates aren't appended. Run on first
session start; the agent reads the resulting lessons via
``KnowledgeStore.context_brief(target, stage, op_family, topic)``.

Each lesson is tagged with `stage`, `op_family`, `topic` so the agent
can ask narrow questions ("what do I know about *kernel-gen for matmul
on Turing*?") and only get the relevant subset.
"""

from __future__ import annotations

from compgen.memory.knowledge import KnowledgeStore, Lesson

SEED_LESSONS: list[Lesson] = [
    # ----- General compiler design -----
    Lesson(
        scope="general",
        category="design",
        stage="instrumentation",
        topic="profiling",
        summary=(
            "Instrumentation must be opt-in for production paths — per-op "
            "torch.cuda.synchronize() in a profiler can add ~25 ms to a "
            "100 ms forward (measurement noise > most actual perf wins)."
        ),
        evidence={"tinyllama_S32_with_profiler_ms": 92, "tinyllama_S32_no_profiler_ms": 67},
        tags=("profiling", "instrumentation", "noise"),
        applicability="any e2e bench harness with per-op timing",
        next_action="default to NoOpProfiler in production paths; opt-in to KernelProfiler for analysis only",
    ),
    Lesson(
        scope="general",
        category="design",
        stage="instrumentation",
        topic="profiling",
        summary=(
            "Hot-path identification should use WARM steady-state numbers, "
            "not cold-start. Cold-start matmul shows 75% of wall-time; warm "
            "shows the same matmul at ~40% — autotune sweep dominated cold."
        ),
        evidence={"cold_matmul_share_pct": 75, "warm_matmul_share_pct": 40},
        tags=("autotune", "cold-start", "profiling"),
        applicability="any model with autotune-driven kernels",
    ),
    Lesson(
        scope="general",
        category="design",
        stage="kernel-gen",
        topic="cost-model",
        summary=(
            "Refinement-driven kernel iteration costs ~$0.10-0.15 per kernel "
            "(Claude Code in-session × 2-3 attempts) vs autocomp's ~$13-20. "
            "Hybrid (Claude default + autocomp escalation on the long-tail "
            "5%) is 30-100× cheaper than autocomp-everything."
        ),
        evidence={"per_kernel_claude_usd": 0.15, "per_kernel_autocomp_usd": 15},
        tags=("cost", "autocomp", "refinement"),
        applicability="any kernel-generation pipeline",
        next_action="default to Claude Code; escalate only when best-of-3 still > 3× eager",
    ),
    Lesson(
        scope="general",
        category="recipe",
        stage="deployment",
        topic="autotune-cache",
        summary=(
            "Triton autotune picks should persist to disk across processes. "
            "Without it, every fresh process pays a 1-3s sweep per unique "
            "(M,N,K) tuple — for TinyLlama's 7 unique matmul shapes that's "
            "9.5s of cold-start vs ~500ms with persisted picks."
        ),
        evidence={"cold_start_s": 9.5, "warm_after_persist_ms": 500},
        tags=("autotune", "deployment", "persistence"),
        applicability="any Triton kernel with autotune",
        next_action="auto-load autotune cache on module import",
    ),
    # ----- Any GPU -----
    Lesson(
        scope="backends/gpu/general",
        category="design",
        stage="dispatch",
        topic="launch-overhead",
        summary=(
            "Per-launch overhead is ~10-20 μs on consumer GPUs. For models "
            "with > 200 small kernel launches per forward, CUDA-graph "
            "capture is structurally the right answer; for compute-bound "
            "models, the saving is 5-10% (small)."
        ),
        evidence={"per_launch_us": 15, "tinyllama_total_launches": 270, "graph_save_pct_at_S32": 6},
        tags=("cuda-graph", "launch-overhead", "small-batch"),
        applicability="any GPU with > 100 kernels per forward",
    ),
    Lesson(
        scope="backends/gpu/general",
        category="recipe",
        stage="kernel-gen",
        op_family="attention",
        topic="fusion-decision",
        summary=(
            "FlashAttention's win scales with attention's share of total "
            "compute. At TinyLlama S=32 with hidden=2048, attention is "
            "only ~3% of warm wall-time — FA's 2.27× microbench win = "
            "small e2e win. FA is necessary architecture; not always the "
            "biggest lever."
        ),
        evidence={"fa_microbench_speedup": 2.27, "fa_e2e_speedup_at_S32": 1.04},
        tags=("flash-attention", "attention", "perf-analysis"),
        applicability="small batch + small sequence + large hidden_dim",
    ),
    # ----- NVIDIA-general -----
    Lesson(
        scope="backends/gpu/nvidia/general",
        category="limit",
        stage="kernel-gen",
        op_family="matmul",
        topic="perf-ceiling",
        summary=(
            "Hand-written Triton matmul lands ~5-8× behind cuBLAS for small "
            "fp16 matmuls on consumer NVIDIA GPUs. Closing requires either "
            "autocomp (beam search over tile / split-K / register-tile "
            "shapes) or hand-tuned PTX."
        ),
        evidence={"matmul_512x1024x512_fp16_ours_us": 195, "cublas_us": 36, "ratio": 5.4},
        tags=("matmul", "perf-ceiling", "triton"),
        applicability="any naive Triton matmul without autocomp on NVIDIA",
        next_action="route matmul through autocomp escalation when perf budget < 2× cuBLAS",
    ),
    Lesson(
        scope="backends/gpu/nvidia/general",
        category="recipe",
        stage="kernel-gen",
        op_family="matmul",
        topic="tile-selection",
        summary=(
            "GROUP_M swizzle (linearise pid_m,pid_n into groups of GROUP_M "
            "rows) is a near-free 1.3× win on Triton matmul because it "
            "improves L2 re-use across consecutive output tiles."
        ),
        evidence={"v1_us": 195, "v2_with_swizzle_us": 147, "speedup": 1.32},
        tags=("matmul", "swizzle", "L2-cache"),
        applicability="any Triton matmul on NVIDIA",
    ),
    # ----- Turing (sm_75 — TITAN RTX) -----
    Lesson(
        scope="backends/gpu/nvidia/turing",
        category="limit",
        stage="kernel-gen",
        topic="hardware-envelope",
        summary=(
            "sm_75 has fp16 tensor cores (HMMA) but NO bf16 TC, NO TF32, "
            "NO async copy (cp.async). Codegen for Turing must avoid bf16 "
            "matmul + cp.async-style pipelining; FlashAttention-2's "
            "Ampere-tuned tricks don't transfer cleanly."
        ),
        evidence={"sm": "7.5", "smem_kb_per_block": 48, "vector_lanes_TITAN": 72},
        tags=("turing", "sm_75", "tensor-core", "limits"),
        applicability="all Turing GPUs (TITAN RTX, T4, RTX 20-series)",
        next_action="prefer fp16 IO + fp32 accumulator; avoid Ampere-only intrinsics",
    ),
    Lesson(
        scope="backends/gpu/nvidia/turing",
        category="recipe",
        stage="kernel-gen",
        op_family="matmul",
        topic="scheduling",
        summary=(
            "Persistent CTA scheduling pays only when num_output_tiles ≥ "
            "4 × num_SMs. At small shapes (output 64 tiles, 72 SMs) the "
            "launch overhead was already minimal; at 1024 tiles it gives "
            "1.6×."
        ),
        evidence={"small_shape_speedup": 1.0, "large_shape_speedup": 1.62},
        tags=("persistent-cta", "scheduling", "matmul"),
        applicability="Triton matmul on Turing/Ampere with > 1024 output tiles",
    ),
    # ----- Ampere (sm_80) -----
    Lesson(
        scope="backends/gpu/nvidia/ampere",
        category="recipe",
        stage="kernel-gen",
        op_family="attention",
        topic="hardware-envelope",
        summary=(
            "Ampere has bf16 TC, TF32, and cp.async (async copy). FlashAttention-2's "
            "full speedup (~3-5×) is achievable here. cp.async with num_stages=3 "
            "gives software-pipelined matmul that Turing can't match."
        ),
        evidence={"bf16_tc_supported": True, "cp_async_supported": True, "smem_kb_per_block": 164},
        tags=("ampere", "sm_80", "flash-attention", "tensor-core"),
        applicability="A100, H100, RTX 30+ series",
    ),
    # ----- CUDA driver -----
    Lesson(
        scope="drivers/cuda/general",
        category="limit",
        stage="dispatch",
        topic="cuda-graph",
        summary=(
            "torch.cuda.CUDAGraph capture forbids torch.cuda.synchronize() "
            "inside the captured region. Any profiler that syncs per op "
            "must be opt-out for the captured path."
        ),
        evidence={"error": "cudaErrorStreamCaptureUnsupported"},
        tags=("cuda-graph", "capture", "sync"),
        applicability="any CUDA-graph capture in PyTorch",
    ),
    Lesson(
        scope="drivers/cuda/general",
        category="recipe",
        stage="dispatch",
        topic="cuda-graph",
        summary=(
            "CUDA-graph capture must run on a non-default stream. "
            "Pattern: launch warmup on side stream, current_stream.wait_stream(side), "
            "then with torch.cuda.graph(graph): captured_out = model_fn(static_input)."
        ),
        evidence={},
        tags=("cuda-graph", "capture", "stream"),
        applicability="any CUDA-graph capture in PyTorch",
    ),
]


def install(store: KnowledgeStore | None = None) -> int:
    """Install seed lessons, skipping any (scope, summary) already present.

    Returns number of new lessons added.
    """
    from compgen.memory.knowledge import shared_store

    s = store or shared_store()

    added = 0
    for lesson in SEED_LESSONS:
        existing = s._read_lessons(s.lessons_file(lesson.scope))
        if any(e.summary == lesson.summary for e in existing):
            continue
        s.add(lesson)
        added += 1
    return added


__all__ = ["SEED_LESSONS", "install"]
