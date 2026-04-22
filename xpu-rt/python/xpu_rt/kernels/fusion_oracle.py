"""Fusion-decision engine — should_fuse(producer, consumer, target).

Cost model + eligibility gate, replacing the heuristic-only canonical-pair
table in ``agent/suggest/suggest_fusion.py`` (which stays as a fallback
for op pairs not covered here).

Cost model components (all in microseconds-equivalent):

  * ``dram_savings``        = bytes_eliminated / target.bandwidth — fewer
                              DRAM round-trips means kernel runs less
                              time bandwidth-bound.
  * ``launch_savings``      = launches_eliminated × per_launch_overhead.
                              For TinyLlama-style small kernels this is
                              5-10% of forward; for big kernels it's noise.
  * ``register_pressure``   = penalty when fused tile no longer fits in
                              ``register_quota_per_thread`` — codegen
                              would spill.
  * ``scratchpad_pressure`` = penalty when fused intermediates exceed
                              the SMEM budget.

The verdict carries an estimated speedup ratio so the agent can decide
whether the fusion is worth the architectural cost (megakernel-style
fusion changes the dispatch model).

Per-target policy: each target architecture has its own thresholds and
cost coefficients, derived from the knowledge store's persistent
lessons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from compgen.kernels.contract_v3 import (
    KernelContractV3,
    MemoryTier,
)
from compgen.memory.knowledge import shared_store


class FusionDecision(Enum):
    FUSE = "fuse"
    DONT_FUSE = "dont_fuse"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class FusionVerdict:
    """Outcome of ``should_fuse``."""

    decision: FusionDecision
    est_speedup_ratio: float  # 1.0 = no change; >1 = win; <1 = regression
    reason: str
    eligibility_failures: tuple[str, ...] = ()
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    knowledge_brief: str = ""


# ---------------------------------------------------------------------------
# Per-target launch overhead + bandwidth coefficients
# ---------------------------------------------------------------------------


# Per-target "what does one kernel launch cost in μs" — informs the
# launch-savings term. Calibrated from observed numbers.
_LAUNCH_OVERHEAD_US: dict[str, float] = {
    "turing": 15.0,
    "ampere": 12.0,
    "hopper": 8.0,
    "hexagon": 50.0,  # NPU dispatch is heavier
    "cpu": 2.0,
    "rocm": 18.0,
}


def _arch_key(target_name: str) -> str:
    n = target_name.lower()
    if "turing" in n or "titan-rtx" in n or "test-gpu-simt" in n:
        return "turing"
    if "ampere" in n or "a100" in n:
        return "ampere"
    if "hopper" in n or "h100" in n:
        return "hopper"
    if "hexagon" in n or "openq" in n:
        return "hexagon"
    if "cpu" in n or "host" in n:
        return "cpu"
    if "rocm" in n or "mi" in n:
        return "rocm"
    return "ampere"


def _dtype_bytes(d: str) -> int:
    return {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i32": 4, "i64": 8}.get(d, 4)


def _tensor_bytes(io_tensor) -> int:
    """Estimate one IO tensor's byte traffic for a single invocation.

    Uses concrete dims when present; for symbolic dims uses a
    conservative ``-1 → 1`` (since fusion savings should be evaluated
    on a per-element basis where unknown dims contribute equally to
    fused vs unfused).
    """
    dt = io_tensor.dtype_class[0] if io_tensor.dtype_class else "f32"
    bytes_per = _dtype_bytes(dt)
    n = 1
    for d in io_tensor.shape.dims:
        n *= d if d is not None and d > 0 else 1
    return n * bytes_per


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------


def _check_eligibility(
    producer: KernelContractV3,
    consumer: KernelContractV3,
) -> tuple[bool, list[str]]:
    """Return ``(eligible, [reasons_failed])``."""
    failures: list[str] = []

    # 1. The producer's output must feed the consumer's input. We can't
    # check SSA edges from contracts alone, but we can check name
    # compatibility — the consumer should accept the producer's output dtype.
    if producer.io.outputs and consumer.io.inputs:
        prod_out_dtype = producer.io.outputs[0].dtype_class
        cons_in_dtype = consumer.io.inputs[0].dtype_class
        if not set(prod_out_dtype).intersection(cons_in_dtype):
            failures.append(
                f"dtype incompatible: producer outputs {prod_out_dtype} but consumer accepts {cons_in_dtype}"
            )

    # 2. Producer must permit being a non-boundary fusion source.
    if producer.orchestration.fusion.is_boundary:
        failures.append("producer is declared as fusion boundary")

    # 3. Consumer's archetype must be in producer's fusable_with set
    # (or the set is empty meaning "fuse with anything reasonable").
    fw = producer.orchestration.fusion.fusable_with
    if fw and consumer.archetype.value not in fw:
        failures.append(f"consumer archetype {consumer.archetype.value!r} not in producer.fusable_with={fw}")

    # 4. SMEM budget: if both contracts declare scratchpad residency,
    # combined working set must fit.
    p_mem = producer.orchestration.memory
    c_mem = consumer.orchestration.memory
    if MemoryTier.SCRATCHPAD in p_mem.output_tiers and MemoryTier.SCRATCHPAD in c_mem.input_tiers:
        # Both want scratchpad — that's the *good* case; just check budget.
        env = producer.orchestration.execution
        if env is not None:
            budget = env.hardware.scratchpad_bytes
            # Crude estimate: producer output + consumer working set
            estimated = sum(_tensor_bytes(t) for t in producer.io.outputs)
            estimated += sum(_tensor_bytes(t) for t in consumer.io.inputs)
            if estimated > budget:
                failures.append(f"combined scratchpad working set {estimated}B exceeds target budget {budget}B")

    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _estimate_costs(
    producer: KernelContractV3,
    consumer: KernelContractV3,
    target_name: str,
) -> dict[str, float]:
    """Return per-component cost estimates in microseconds-equivalent."""
    env = producer.orchestration.execution or consumer.orchestration.execution
    if env is None:
        # No envelope → cannot estimate; return neutral
        return {
            "dram_savings_us": 0.0,
            "launch_savings_us": 0.0,
            "register_pressure_us": 0.0,
            "scratchpad_pressure_us": 0.0,
            "net_us": 0.0,
        }
    hw = env.hardware

    # 1. DRAM round-trip eliminated: producer's output (+ consumer's input
    # if it would re-read) no longer goes through DRAM when fused.
    bytes_eliminated = sum(_tensor_bytes(t) for t in producer.io.outputs)
    peak_bw_bps = max(hw.peak_bandwidth_gbps, 1.0) * 1e9
    dram_savings_us = (bytes_eliminated / peak_bw_bps) * 1e6

    # 2. Launch overhead saved: 1 launch per fused pair (assume both
    # were async dispatches).
    arch = _arch_key(target_name)
    per_launch_us = _LAUNCH_OVERHEAD_US.get(arch, 15.0)
    launch_savings_us = per_launch_us

    # 3. Register pressure penalty: rough — count operands feeding the
    # fused inner loop. Each operand adds 16 bytes of register pressure.
    # When estimated reg use > 50% of quota, start penalising.
    reg_quota = max(hw.register_quota_per_thread, 64)
    estimated_reg_use = (len(producer.io.inputs) + len(consumer.io.inputs)) * 16
    if estimated_reg_use > reg_quota * 0.5:
        # Quadratic penalty: spill cost grows fast.
        spill_factor = min(2.0, estimated_reg_use / reg_quota)
        register_pressure_us = launch_savings_us * spill_factor
    else:
        register_pressure_us = 0.0

    # 4. Scratchpad pressure: linear penalty proportional to fraction over budget.
    scratchpad_pressure_us = 0.0
    smem_budget = max(hw.scratchpad_bytes, 1024)
    smem_use = sum(_tensor_bytes(t) for t in producer.io.outputs)
    smem_use += sum(_tensor_bytes(t) for t in consumer.io.inputs)
    if smem_use > smem_budget * 0.8:
        # Approaching limit; penalty linear in pressure.
        scratchpad_pressure_us = (smem_use / smem_budget - 0.8) * launch_savings_us

    net_us = dram_savings_us + launch_savings_us - register_pressure_us - scratchpad_pressure_us
    return {
        "dram_savings_us": dram_savings_us,
        "launch_savings_us": launch_savings_us,
        "register_pressure_us": register_pressure_us,
        "scratchpad_pressure_us": scratchpad_pressure_us,
        "net_us": net_us,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_fuse(
    producer: KernelContractV3,
    consumer: KernelContractV3,
    *,
    baseline_unfused_us: float | None = None,
) -> FusionVerdict:
    """Recommend whether to fuse ``producer`` → ``consumer``.

    Args:
        producer, consumer: v3 contracts. Producer's output is assumed
            to feed consumer's input (caller verifies SSA edge).
        baseline_unfused_us: Optional measured/estimated time for the
            unfused pair. When provided, ``est_speedup_ratio`` is
            computed against it; otherwise it's a relative win/loss
            scaled to ``net_us``.

    Returns:
        FusionVerdict with decision, est_speedup_ratio, reason,
        eligibility_failures, cost_breakdown, and a knowledge_brief
        scoped to ``stage="fusion" topic="fusion-decision"``.
    """
    # 1. Eligibility
    eligible, failures = _check_eligibility(producer, consumer)
    if not eligible:
        return FusionVerdict(
            decision=FusionDecision.INELIGIBLE,
            est_speedup_ratio=1.0,
            reason="; ".join(failures),
            eligibility_failures=tuple(failures),
        )

    # 2. Resolve target name (use producer's envelope first)
    env = producer.orchestration.execution or consumer.orchestration.execution
    target_name = env.hardware.target_name if env is not None else "unknown"

    # 3. Cost model
    costs = _estimate_costs(producer, consumer, target_name)
    net_us = costs["net_us"]

    # 4. Speedup estimate
    if baseline_unfused_us is not None and baseline_unfused_us > 0:
        ratio = baseline_unfused_us / max(baseline_unfused_us - net_us, 1.0)
    else:
        # Without baseline, infer ratio from net savings vs launch overhead
        # (the smallest meaningful unit). ratio = 1 + savings / per_launch.
        per_launch = max(costs["launch_savings_us"], 1.0)
        ratio = 1.0 + max(net_us, 0.0) / per_launch

    # 5. Decide
    if net_us > 0 and ratio >= 1.05:
        decision = FusionDecision.FUSE
        reason = (
            f"net savings {net_us:.1f}μs (DRAM: -{costs['dram_savings_us']:.1f}μs, "
            f"launch: -{costs['launch_savings_us']:.1f}μs, reg: +{costs['register_pressure_us']:.1f}μs); "
            f"est speedup {ratio:.2f}×"
        )
    else:
        decision = FusionDecision.DONT_FUSE
        reason = (
            f"net cost {-net_us:.1f}μs ≥ savings; pressure terms dominate "
            f"(reg+smem = {costs['register_pressure_us'] + costs['scratchpad_pressure_us']:.1f}μs)"
        )

    # 6. Knowledge brief — narrow to fusion-decision lessons for this target
    brief = shared_store().context_brief(
        target_name,
        stage="fusion",
        topic="fusion-decision",
        max_lessons=5,
    )

    verdict = FusionVerdict(
        decision=decision,
        est_speedup_ratio=ratio,
        reason=reason,
        cost_breakdown=costs,
        knowledge_brief=brief,
    )
    _emit_advisory(
        {
            "target": target_name,
            "producer": producer.op_name,
            "consumer": consumer.op_name,
            "verdict": verdict.decision.value,
            "est_speedup": round(verdict.est_speedup_ratio, 3),
            "cost_breakdown": dict(verdict.cost_breakdown),
            "reason": verdict.reason,
            "baseline_unfused_us": baseline_unfused_us,
            "binding": False,
        }
    )
    return verdict


def _emit_advisory(payload: dict) -> None:
    """Best-effort ``oracle_advisory`` emission for fusion verdicts."""
    try:
        from compgen.trace import OraclePublisher

        OraclePublisher.emit(oracle="fusion", **payload)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "FusionDecision",
    "FusionVerdict",
    "should_fuse",
]
