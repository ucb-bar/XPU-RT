"""Granularity decision: MICRO vs NORMAL vs MEGA.

Today granularity is hand-set per contract. This oracle automates the
decision based on:

  * **MICRO (ukernel)** — register-resident tile primitive, inlined
    into a parent kernel. Decision: estimated working-set fits in
    one warp's registers AND op is not a fusion boundary.
  * **NORMAL** — single dispatch (the default for everything). Decision:
    standalone op, output goes to DRAM, no shared scratchpad with
    neighbours.
  * **MEGA (persistent)** — fused chain with shared scratchpad
    lifetime. Decision: producer + consumers can fit working set in
    scratchpad AND fusion oracle says est_speedup ≥ 1.5× across the
    chain AND chain length ≥ 2.

The oracle takes a *region* (one or more candidate ops + a hardware
envelope + an optional perf target) and returns a recommended
granularity + a rationale string.

Used by:
  * `propose_megakernel_synthesis` — to upgrade clusters from NORMAL
    to MEGA when the cost model says yes.
  * Codegen — to decide whether to emit a function (NORMAL) or inline
    body (MICRO) or persistent kernel (MEGA).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from compgen.kernels.contract_v3 import (
    Granularity,
    HardwareEnvelope,
    KernelArchetype,
    KernelContractV3,
)
from compgen.kernels.fusion_oracle import FusionDecision, should_fuse
from compgen.memory.knowledge import shared_store


@dataclass(frozen=True)
class GranularityVerdict:
    granularity: Granularity
    reason: str
    confidence: float = 0.5
    chain_speedup_estimate: float = 1.0   # MEGA only — combined fusion ratio
    knowledge_brief: str = ""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _dtype_bytes(d: str) -> int:
    return {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i32": 4, "i64": 8}.get(d, 4)


def _working_set_bytes(c: KernelContractV3) -> int:
    """Estimate one-invocation working set in bytes."""
    total = 0
    for io in (*c.io.inputs, *c.io.outputs):
        dt = io.dtype_class[0] if io.dtype_class else "f32"
        bps = _dtype_bytes(dt)
        n = 1
        for d in io.shape.dims:
            n *= d if d is not None and d > 0 else 1
        total += n * bps
    return total


def _fits_in_registers(c: KernelContractV3, envelope: HardwareEnvelope) -> bool:
    """Conservative: working set ≤ register budget per thread × vector_lanes."""
    budget = envelope.register_quota_per_thread * max(envelope.vector_lanes, 1)
    return _working_set_bytes(c) <= budget


def _fits_in_scratchpad(contracts: Sequence[KernelContractV3], envelope: HardwareEnvelope) -> bool:
    """Combined working set ≤ scratchpad budget."""
    total = sum(_working_set_bytes(c) for c in contracts)
    return total <= envelope.scratchpad_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_granularity(
    region: Sequence[KernelContractV3],
    envelope: HardwareEnvelope,
    *,
    perf_target_us: float | None = None,
    is_inlined_callee: bool = False,
) -> GranularityVerdict:
    """Recommend MICRO / NORMAL / MEGA for ``region``.

    Args:
        region: One or more v3 contracts forming a candidate cluster.
            len==1 → single-op decision (MICRO vs NORMAL); len>=2 →
            chain-fusion decision (NORMAL vs MEGA).
        envelope: Target HardwareEnvelope.
        perf_target_us: Optional latency target; influences MEGA
            recommendation (a tight target makes MEGA more attractive).
        is_inlined_callee: When True the caller has already decided
            this region is inlined into a parent kernel — strong
            signal for MICRO.

    Returns:
        ``GranularityVerdict``.
    """
    if not region:
        return GranularityVerdict(
            granularity=Granularity.NORMAL,
            reason="empty region",
            confidence=0.0,
        )

    target_name = envelope.target_name
    brief = shared_store().context_brief(
        target_name, stage="kernel-gen", topic="hardware-envelope",
        max_lessons=3,
    )

    # ---------- single-op decision: MICRO vs NORMAL ----------
    if len(region) == 1:
        c = region[0]

        # MICRO check: explicitly marked as inlined callee, OR fits in
        # registers AND not a fusion boundary AND not a persistent
        # kernel candidate.
        if is_inlined_callee:
            return GranularityVerdict(
                granularity=Granularity.MICRO,
                reason="caller declared this region is inlined → MICRO ukernel",
                confidence=0.9,
                knowledge_brief=brief,
            )

        if (_fits_in_registers(c, envelope)
                and not c.orchestration.fusion.is_boundary
                and c.archetype is not KernelArchetype.COMPUTE_TILED):
            return GranularityVerdict(
                granularity=Granularity.MICRO,
                reason=(
                    f"working set {_working_set_bytes(c)}B fits in registers "
                    f"({envelope.register_quota_per_thread * envelope.vector_lanes}B); "
                    "non-boundary + non-COMPUTE_TILED → MICRO ukernel"
                ),
                confidence=0.7,
                knowledge_brief=brief,
            )

        # Default: NORMAL
        return GranularityVerdict(
            granularity=Granularity.NORMAL,
            reason=(
                f"single-op, working set {_working_set_bytes(c)}B exceeds register "
                f"budget OR archetype={c.archetype.value} requires dispatch"
            ),
            confidence=0.7,
            knowledge_brief=brief,
        )

    # ---------- chain decision: NORMAL vs MEGA ----------

    # Eligibility: combined working set must fit in scratchpad.
    if not _fits_in_scratchpad(region, envelope):
        return GranularityVerdict(
            granularity=Granularity.NORMAL,
            reason=(
                f"chain working set {sum(_working_set_bytes(c) for c in region)}B "
                f"> scratchpad budget {envelope.scratchpad_bytes}B → can't keep "
                "intermediates resident; MEGA would spill"
            ),
            confidence=0.8,
            knowledge_brief=brief,
        )

    # Compute pairwise fusion verdicts; if every pair says FUSE, the
    # chain-level decision is MEGA.
    chain_speedup = 1.0
    pairwise_decisions: list[str] = []
    for i in range(len(region) - 1):
        v = should_fuse(region[i], region[i + 1])
        pairwise_decisions.append(
            f"{region[i].op_name}→{region[i + 1].op_name}: "
            f"{v.decision.value} ({v.est_speedup_ratio:.2f}×)"
        )
        if v.decision is FusionDecision.INELIGIBLE:
            return GranularityVerdict(
                granularity=Granularity.NORMAL,
                reason=(
                    f"pair ineligible: {region[i].op_name}→{region[i + 1].op_name} "
                    f"({v.reason})"
                ),
                confidence=0.8,
                knowledge_brief=brief,
            )
        if v.decision is FusionDecision.DONT_FUSE:
            return GranularityVerdict(
                granularity=Granularity.NORMAL,
                reason=(
                    f"fusion oracle declined pair "
                    f"{region[i].op_name}→{region[i + 1].op_name}: {v.reason}"
                ),
                confidence=0.7,
                knowledge_brief=brief,
            )
        chain_speedup *= v.est_speedup_ratio

    # MEGA threshold: combined chain speedup ≥ 1.5× justifies the
    # architectural cost of becoming a persistent kernel.
    threshold = 1.5 if perf_target_us is None else 1.3
    if chain_speedup >= threshold:
        return GranularityVerdict(
            granularity=Granularity.MEGA,
            reason=(
                f"chain of {len(region)} ops fits in scratchpad + every pair "
                f"FUSE; combined speedup {chain_speedup:.2f}× ≥ {threshold:.2f}× → MEGA"
            ),
            confidence=0.8,
            chain_speedup_estimate=chain_speedup,
            knowledge_brief=brief,
        )

    return GranularityVerdict(
        granularity=Granularity.NORMAL,
        reason=(
            f"chain combined speedup {chain_speedup:.2f}× below MEGA threshold "
            f"{threshold:.2f}×; pairwise: {'; '.join(pairwise_decisions)}"
        ),
        confidence=0.7,
        chain_speedup_estimate=chain_speedup,
        knowledge_brief=brief,
    )


__all__ = ["GranularityVerdict", "recommend_granularity"]


def _emit_advisory(verdict: GranularityVerdict, *, target_name: str, region_size: int) -> None:
    """Best-effort ``oracle_advisory`` emission for granularity verdicts."""
    try:
        from compgen.trace import OraclePublisher

        OraclePublisher.emit(
            oracle="granularity",
            target=target_name,
            region_size=region_size,
            granularity=verdict.granularity.value,
            confidence=verdict.confidence,
            chain_speedup=verdict.chain_speedup_estimate,
            reason=verdict.reason,
            binding=False,
        )
    except Exception:  # noqa: BLE001
        pass


# Wrap the public ``recommend_granularity`` by replacing its return path.
_orig_recommend_granularity = recommend_granularity


def recommend_granularity(region, envelope, *, perf_target_us=None, is_inlined_callee=False):  # type: ignore[no-redef]
    verdict = _orig_recommend_granularity(
        region,
        envelope,
        perf_target_us=perf_target_us,
        is_inlined_callee=is_inlined_callee,
    )
    _emit_advisory(verdict, target_name=envelope.target_name, region_size=len(region))
    return verdict
