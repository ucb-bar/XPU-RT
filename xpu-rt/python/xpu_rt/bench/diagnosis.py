"""Turn a raw ``BenchResult`` + ``KernelContractV3`` into a distilled
``KernelDiagnosis`` — the signal we feed into the *next* codegen attempt.

Design principle: **distillation, not dumping**. A full ncu report is
~80 metrics; pasting them into a refinement prompt drowns the LLM. The
valuable move is to condense everything into:

  * one **primary_bottleneck** label (``bandwidth_bound`` /
    ``compute_bound`` / ``latency_bound`` / ``launch_bound`` /
    ``correctness_bound``)
  * a **roofline_efficiency** fraction — how close we are to the roof
    the kernel is hitting
  * up to **3 prioritised hypotheses** the next codegen should try,
    ordered by expected impact
  * 5–10 **supporting_metrics** that justify the hypotheses

Hypotheses are archetype-aware: a COMPUTE_TILED matmul and a POINTWISE
addf get different advice for the same latency overshoot.

This module is **profiler-free** — everything here is derived from
the latency + contract + hardware envelope we already have. A deeper
signal pass (ncu-backed SMEM conflicts / occupancy / tensor-core
utilisation) is a separate module that only fires when the distilled
diagnosis here says "needs deeper signal".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compgen.bench.kernel_bench import BenchResult
from compgen.kernels.contract_v3 import (
    HardwareEnvelope,
    KernelArchetype,
    KernelContractV3,
)


# ---------------------------------------------------------------------------
# Bottleneck taxonomy
# ---------------------------------------------------------------------------


class Bottleneck(Enum):
    BANDWIDTH_BOUND = "bandwidth_bound"
    COMPUTE_BOUND = "compute_bound"
    LATENCY_BOUND = "latency_bound"         # launch + sync overhead dominates
    CORRECTNESS_BOUND = "correctness_bound"  # refinement should fix math, not perf
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


@dataclass
class KernelDiagnosis:
    """Distilled refinement signal."""

    kernel_name: str
    archetype: KernelArchetype
    primary_bottleneck: Bottleneck
    # roofline_efficiency ∈ [0, 1]: achieved / peak on whichever roof the kernel hit.
    roofline_efficiency: float
    hypotheses: tuple[str, ...]
    supporting_metrics: dict[str, float] = field(default_factory=dict)
    # Raw pointers that the refinement-prompt builder uses verbatim.
    compared_to: str = ""       # e.g. "cuBLAS 36.8μs = 85% peak compute"
    previous_attempt_summary: str = ""


# ---------------------------------------------------------------------------
# FLOPS / byte-traffic estimation per archetype
# ---------------------------------------------------------------------------


def _tensor_size_bytes(shape: list[int], dtype_bytes: int) -> int:
    if not shape:
        return 0
    n = 1
    for d in shape:
        n *= d if d > 0 else 1
    return n * dtype_bytes


def _dtype_bytes(dt: str) -> int:
    return {"f64": 8, "f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4,
            "i16": 2, "i8": 1, "i1": 1, "tf32": 4}.get(dt, 4)


def _estimate_flops_and_bytes(
    contract: KernelContractV3,
    input_shapes: list[list[int]],
    output_shapes: list[list[int]] | None = None,
) -> tuple[int, int]:
    """Return ``(total_flops, total_bytes)`` for one invocation.

    ``total_bytes`` = ∑ input_sizes + ∑ output_sizes (bytes moved through DRAM
    in the simplest — no-reuse — case).  The actual bandwidth demand can
    be less (caches) or more (no tile re-use), but this is the right
    first-order number for the roofline.
    """
    archetype = contract.archetype
    # Bytes — input + output traffic
    in_shapes = input_shapes
    out_shapes = output_shapes or []
    input_bytes = sum(
        _tensor_size_bytes(s, _dtype_bytes(t.dtype_class[0] if t.dtype_class else "f32"))
        for s, t in zip(in_shapes, contract.io.inputs, strict=False)
    )
    output_bytes = sum(
        _tensor_size_bytes(s, _dtype_bytes(t.dtype_class[0] if t.dtype_class else "f32"))
        for s, t in zip(out_shapes, contract.io.outputs, strict=False)
    ) if out_shapes else sum(
        _tensor_size_bytes(s, _dtype_bytes(t.dtype_class[0] if t.dtype_class else "f32"))
        for s, t in zip(in_shapes[:len(contract.io.outputs)], contract.io.outputs, strict=False)
    )

    # FLOPs — archetype-specific.
    flops = 0
    if archetype is KernelArchetype.COMPUTE_TILED and len(in_shapes) >= 2:
        # Assume matmul shape: (M,K) × (K,N) → 2*M*N*K FLOPs.
        a, b = in_shapes[0], in_shapes[1]
        if len(a) >= 2 and len(b) >= 2:
            M = a[-2] if a[-2] > 0 else 1
            K = a[-1] if a[-1] > 0 else 1
            N = b[-1] if b[-1] > 0 else 1
            flops = 2 * M * N * K
    elif archetype is KernelArchetype.REDUCE and in_shapes:
        # Softmax-like: each element takes ~3 flops (exp + div + sub) plus log(N) reduction.
        n = 1
        for d in in_shapes[0]:
            n *= d if d > 0 else 1
        flops = 5 * n
    elif archetype is KernelArchetype.POINTWISE and in_shapes:
        n = 1
        for d in in_shapes[0]:
            n *= d if d > 0 else 1
        flops = n
    elif archetype is KernelArchetype.ACTIVATION and in_shapes:
        n = 1
        for d in in_shapes[0]:
            n *= d if d > 0 else 1
        flops = 4 * n  # exp + mul + add + div
    # MEMORY / TYPE_CONV_INDEX: no meaningful FLOP count; stays 0.

    return flops, input_bytes + output_bytes


# ---------------------------------------------------------------------------
# Distillation — the core heuristics
# ---------------------------------------------------------------------------


def _roofline(
    flops: int, bytes_: int, elapsed_us: float, hw: HardwareEnvelope,
) -> tuple[Bottleneck, float, dict[str, float]]:
    """Decide compute vs bandwidth bound + return achieved/peak ratios.

    Returns ``(bottleneck, efficiency, metrics)``. ``efficiency`` ∈ [0,1]
    is the ratio of achieved-to-peak on whichever roof we hit.
    """
    if elapsed_us <= 0:
        return Bottleneck.UNKNOWN, 0.0, {}

    sec = elapsed_us * 1e-6
    achieved_gflops = flops / sec / 1e9 if flops > 0 else 0.0
    achieved_gbps = bytes_ / sec / 1e9 if bytes_ > 0 else 0.0

    # Peak compute is a rough derivation from HardwareEnvelope.
    # We don't have peak_tflops on the envelope yet (lives on ComputeUnit);
    # conservative guess: 15 TFLOPS (cuda_core fp32) if nothing declared.
    peak_tflops = 15.0  # fp32 baseline
    if hw.native_dtypes:
        if "bf16" in hw.native_dtypes or "f16" in hw.native_dtypes:
            peak_tflops = 125.0  # fp16 tensor-core baseline (sm_75 / Ampere)
    peak_gflops = peak_tflops * 1000
    peak_gbps = hw.peak_bandwidth_gbps if hw.peak_bandwidth_gbps > 0 else 500.0

    compute_eff = achieved_gflops / peak_gflops if peak_gflops > 0 else 0.0
    bandwidth_eff = achieved_gbps / peak_gbps if peak_gbps > 0 else 0.0

    metrics = {
        "achieved_gflops": achieved_gflops,
        "achieved_gbps": achieved_gbps,
        "peak_gflops": float(peak_gflops),
        "peak_gbps": float(peak_gbps),
        "compute_efficiency": compute_eff,
        "bandwidth_efficiency": bandwidth_eff,
        "arithmetic_intensity_flops_per_byte": flops / bytes_ if bytes_ > 0 else 0.0,
    }

    # Pick bottleneck: the roof we're closer to.
    if compute_eff > bandwidth_eff and compute_eff > 0.05:
        return Bottleneck.COMPUTE_BOUND, compute_eff, metrics
    if bandwidth_eff > 0.05:
        return Bottleneck.BANDWIDTH_BOUND, bandwidth_eff, metrics
    # Both low → likely launch-overhead dominated (tiny kernel).
    return Bottleneck.LATENCY_BOUND, max(compute_eff, bandwidth_eff), metrics


def _archetype_hypotheses(
    archetype: KernelArchetype,
    bottleneck: Bottleneck,
    eff: float,
    metrics: dict[str, float],
    vs_eager_ratio: float,
) -> tuple[str, ...]:
    """Generate ≤3 prioritised hypotheses for the next codegen attempt."""

    hypos: list[str] = []

    if archetype is KernelArchetype.COMPUTE_TILED:
        if bottleneck is Bottleneck.COMPUTE_BOUND and eff < 0.30:
            hypos.append(
                "Tensor-core path likely missed — verify dtype passes through "
                "`tl.dot` (bf16/f16 on Turing/Ampere); check BLOCK_K ≥ 16 so "
                "the MMA instruction issues."
            )
        if bottleneck is Bottleneck.BANDWIDTH_BOUND or eff < 0.5:
            hypos.append(
                "Tiles aren't re-using lhs/rhs across the K-loop. Try "
                "BLOCK_M×BLOCK_N ∈ {128×128, 128×64} and `num_stages=3` "
                "for pipelined async loads."
            )
        if vs_eager_ratio > 2.0:
            hypos.append(
                "Consider `triton.autotune` over (BLOCK_M, BLOCK_N, BLOCK_K, "
                "num_warps, num_stages) — first-cut tile is leaving perf "
                "on the table; autotune typically closes 40–60% of the gap."
            )

    elif archetype is KernelArchetype.REDUCE:
        if eff < 0.4:
            hypos.append(
                "Reduction is likely reading row-at-a-time — batch multiple "
                "rows per CTA and use warp-level reductions to amortise launch."
            )
        if metrics.get("arithmetic_intensity_flops_per_byte", 0) < 1.0:
            hypos.append(
                "Bandwidth-bound by nature. Check `BLOCK_N` is a power of 2 "
                "next_power_of_2(N); pick `num_warps=8` for BLOCK_N ≥ 2048."
            )

    elif archetype in (KernelArchetype.POINTWISE, KernelArchetype.ACTIVATION):
        if eff < 0.5:
            hypos.append(
                "Pointwise is bandwidth-bound; grow `BLOCK` to 2048–4096 and "
                "fuse with neighbours via `FusionPolicy.fusable_with` if the "
                "producer/consumer allow in-place."
            )

    elif archetype is KernelArchetype.MEMORY:
        hypos.append(
            "Memory ops are latency-bound below a few KB. Consider batching "
            "many small ops into one kernel launch; declare "
            "`DispatchSpec.model=PERSISTENT` if the call site is hot."
        )

    elif archetype is KernelArchetype.TYPE_CONV_INDEX:
        hypos.append(
            "Type conversions fuse cheaply into their consumer — emit an "
            "in-place cast inside the next kernel rather than a standalone "
            "launch when the consumer dtype matches."
        )

    # Always-applicable suggestion when we have headroom.
    if vs_eager_ratio > 1.5 and eff < 0.7:
        hypos.append(
            "Wrap the kernel in `triton.autotune` with 4–6 `triton.Config` "
            "candidates and re-run; the autotune curve itself is strong "
            "signal for where the kernel bottlenecks."
        )

    return tuple(hypos[:3])  # cap at 3 — more is noise.


def diagnose(
    contract: KernelContractV3,
    bench: BenchResult,
) -> KernelDiagnosis:
    """Distil ``bench`` into refinement guidance for ``contract``.

    Correctness-bound takes precedence: if the kernel fails numerically
    there's no point talking about roofline. Fix math first.
    """
    if not bench.passed:
        return KernelDiagnosis(
            kernel_name=bench.name,
            archetype=contract.archetype,
            primary_bottleneck=Bottleneck.CORRECTNESS_BOUND,
            roofline_efficiency=0.0,
            hypotheses=(
                f"Kernel is numerically wrong — max_abs_err={bench.max_abs_err:.2e}, "
                f"max_rel_err={bench.max_rel_err:.2e}. Revisit reduction order "
                "(max-subtract for softmax, f32 accumulator for matmul), "
                "dtype casts, mask handling on boundary tiles.",
            ),
            supporting_metrics={
                "max_abs_err": bench.max_abs_err,
                "max_rel_err": bench.max_rel_err,
                "our_us": bench.our_us,
            },
            compared_to=(
                f"eager ran {bench.eager_us:.1f}μs and produced the reference — "
                f"your output diverges at {bench.max_abs_err:.2e} abs."
            ),
            previous_attempt_summary=f"FAIL correctness at {bench.our_us:.1f}μs",
        )

    hw = (
        contract.orchestration.execution.hardware
        if contract.orchestration.execution is not None
        else HardwareEnvelope(
            target_name="unknown", vector_lanes=1, scratchpad_bytes=0,
            register_bytes=0, native_dtypes=()
        )
    )

    flops, bytes_ = _estimate_flops_and_bytes(
        contract, bench.input_shapes, output_shapes=None,
    )
    bottleneck, eff, metrics = _roofline(flops, bytes_, bench.our_us, hw)

    hypos = _archetype_hypotheses(
        contract.archetype, bottleneck, eff, metrics,
        vs_eager_ratio=bench.us_ratio_vs_eager,
    )

    metrics["our_us"] = bench.our_us
    metrics["eager_us"] = bench.eager_us
    metrics["vs_eager_ratio"] = bench.us_ratio_vs_eager
    if bench.torch_compile_us is not None:
        metrics["torch_compile_us"] = bench.torch_compile_us

    compared_to = f"eager {bench.eager_us:.1f}μs"
    if bench.torch_compile_us is not None:
        compared_to += f", torch.compile {bench.torch_compile_us:.1f}μs"

    return KernelDiagnosis(
        kernel_name=bench.name,
        archetype=contract.archetype,
        primary_bottleneck=bottleneck,
        roofline_efficiency=eff,
        hypotheses=hypos,
        supporting_metrics=metrics,
        compared_to=compared_to,
        previous_attempt_summary=(
            f"{bench.our_us:.1f}μs, {bottleneck.value}, "
            f"{eff*100:.0f}% of peak roof"
        ),
    )


def format_diagnosis(d: KernelDiagnosis) -> str:
    """Human-readable render — also what goes into refinement prompts."""
    lines = [
        f"DIAGNOSIS for {d.kernel_name} ({d.archetype.value})",
        f"  bottleneck : {d.primary_bottleneck.value}",
        f"  efficiency : {d.roofline_efficiency*100:.1f}% of peak roof",
        f"  compared_to: {d.compared_to}",
        "",
        "  metrics:",
    ]
    for k, v in d.supporting_metrics.items():
        lines.append(f"    {k:38s} = {v:.3g}")
    lines.append("")
    if d.hypotheses:
        lines.append("  top hypotheses (try in order):")
        for i, h in enumerate(d.hypotheses, 1):
            lines.append(f"    {i}. {h}")
    else:
        lines.append("  (no hypotheses — likely at roof already)")
    return "\n".join(lines)


__all__ = [
    "Bottleneck",
    "KernelDiagnosis",
    "diagnose",
    "format_diagnosis",
]
