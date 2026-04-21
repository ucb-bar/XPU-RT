"""Hardware-aware dispatch decisions — let the LLM (or a deterministic
fallback) choose the best granularity AND target for a region.

Wave 6 layer that sits *above* the deterministic ``granularity_oracle``:
the oracle gives us a strong prior; the LLM gets the prior plus a
compact ``HardwareEnvelope`` summary, the relevant knowledge-store
brief, and a per-target rubric — and decides:

  * which *target* to dispatch the region to (when more than one is
    legal — e.g. CPU + CUDA + Hexagon all available)
  * which *granularity* to use on that target (MICRO ukernel vs NORMAL
    kernel vs MEGA persistent kernel)
  * a short rationale that gets logged to the knowledge store so the
    next dispatch decision can replay the reasoning

Two safety nets:
  * If no LLM client is supplied, we fall back to per-target oracle
    recommendations + a heuristic ranking (lower-latency target wins
    for latency-bound budgets, lower-energy for energy-bound).
  * The LLM's choice is validated against the legal set per target
    (e.g. CPU adapter rejects PERSISTENT — the LLM can't dispatch a
    MEGA contract to CPU even if it tries).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from compgen.kernels.contract_v3 import (
    DispatchModel,
    Granularity,
    HardwareEnvelope,
    KernelContractV3,
)
from compgen.kernels.granularity_oracle import (
    GranularityVerdict,
    recommend_granularity,
)
from compgen.llm.base import (
    CompGenLLMProtocol,
    GenerationRequest,
    LLMConfig,
    Objective,
    PromptContext,
)
from compgen.runtime.glue import RuntimeAdapter, select_adapter


# ---------------------------------------------------------------------------
# Decision records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetDispatchDecision:
    """One target × one granularity decision."""

    target: str
    granularity: Granularity
    adapter_name: str
    rationale: str
    confidence: float
    deterministic_prior: GranularityVerdict
    legal: bool = True                         # adapter rejects this dispatch model?
    illegal_reason: str = ""


@dataclass(frozen=True)
class MultiTargetDispatchDecision:
    """Per-target verdicts + the picked best target."""

    region_summary: str
    per_target: dict[str, TargetDispatchDecision]
    best_target: str
    best_rationale: str
    used_llm: bool

    def best(self) -> TargetDispatchDecision:
        return self.per_target[self.best_target]


# ---------------------------------------------------------------------------
# Region + envelope summarisation
# ---------------------------------------------------------------------------


def _summarise_region(region: Sequence[KernelContractV3]) -> str:
    if not region:
        return "<empty region>"
    if len(region) == 1:
        c = region[0]
        return (
            f"single-op region: {c.op_name} ({c.archetype.value}); "
            f"{len(c.io.inputs)} in / {len(c.io.outputs)} out"
        )
    chain = " → ".join(c.op_name for c in region)
    return f"chain of {len(region)} ops: {chain}"


def _summarise_envelope(env: HardwareEnvelope) -> str:
    """One-line, prompt-friendly envelope summary."""
    parts = [
        f"target={env.target_name}",
        f"vector_lanes={env.vector_lanes}",
        f"scratchpad_bytes={env.scratchpad_bytes}",
        f"register_bytes={env.register_bytes}",
    ]
    if env.peak_bandwidth_gbps:
        parts.append(f"bw={env.peak_bandwidth_gbps:.0f}GB/s")
    if env.native_dtypes:
        parts.append(f"dtypes={','.join(env.native_dtypes)}")
    if getattr(env, "mma_shapes", None):
        shapes = ",".join(f"{k}:{v}" for k, v in env.mma_shapes.items())
        parts.append(f"mma=[{shapes}]")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Legality check
# ---------------------------------------------------------------------------


_GRANULARITY_TO_DISPATCH: dict[Granularity, DispatchModel] = {
    Granularity.MICRO:  DispatchModel.INLINE,
    Granularity.NORMAL: DispatchModel.SYNC,
    Granularity.MEGA:   DispatchModel.PERSISTENT,
}


def _adapter_supports(adapter: RuntimeAdapter, granularity: Granularity) -> tuple[bool, str]:
    """Synthesise a stub contract with the canonical dispatch model for
    this granularity and ask the adapter whether it can host it."""
    from compgen.kernels.contract_v3 import (
        DispatchSpec, IOContract, KernelArchetype, MemorySpec, MemoryTier,
        OrchestrationSpec, ShapeClass, TensorIO,
    )

    model = _GRANULARITY_TO_DISPATCH[granularity]
    if granularity is Granularity.MICRO:
        stub = KernelContractV3(
            op_name="probe", archetype=KernelArchetype.POINTWISE,
            io=IOContract(
                inputs=(TensorIO(name="x", shape=ShapeClass(dims=(None,)),
                                 dtype_class=("f32",)),),
                outputs=(TensorIO(name="y", shape=ShapeClass(dims=(None,)),
                                  dtype_class=("f32",)),),
            ),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.INLINE),
                memory=MemorySpec(
                    input_tiers=(MemoryTier.REGISTER,),
                    output_tiers=(MemoryTier.REGISTER,),
                ),
            ),
        )
    elif granularity is Granularity.MEGA:
        sub = KernelContractV3(
            op_name="sub", archetype=KernelArchetype.POINTWISE,
            io=IOContract(
                inputs=(TensorIO(name="x", shape=ShapeClass(dims=(None,)),
                                 dtype_class=("f32",)),),
                outputs=(TensorIO(name="y", shape=ShapeClass(dims=(None,)),
                                  dtype_class=("f32",)),),
            ),
            orchestration=OrchestrationSpec(
                memory=MemorySpec(
                    input_tiers=(MemoryTier.SCRATCHPAD,),
                    output_tiers=(MemoryTier.SCRATCHPAD,),
                ),
            ),
        )
        stub = KernelContractV3(
            op_name="probe", archetype=KernelArchetype.POINTWISE,
            io=IOContract(
                inputs=(TensorIO(name="x", shape=ShapeClass(dims=(None,)),
                                 dtype_class=("f32",)),),
                outputs=(TensorIO(name="y", shape=ShapeClass(dims=(None,)),
                                  dtype_class=("f32",)),),
            ),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
            ),
            body=(sub,),
        )
    else:
        stub = KernelContractV3(
            op_name="probe", archetype=KernelArchetype.POINTWISE,
            io=IOContract(
                inputs=(TensorIO(name="x", shape=ShapeClass(dims=(None,)),
                                 dtype_class=("f32",)),),
                outputs=(TensorIO(name="y", shape=ShapeClass(dims=(None,)),
                                  dtype_class=("f32",)),),
            ),
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=model)),
        )
    if adapter.supports(stub):
        return True, ""
    return False, (
        f"adapter {adapter.name!r} rejects dispatch model "
        f"{model.value!r} (granularity={granularity.value})"
    )


# ---------------------------------------------------------------------------
# Heuristic ranker (used when LLM not present)
# ---------------------------------------------------------------------------


def _rank_envelopes_by_throughput(
    envelopes: Sequence[HardwareEnvelope],
) -> list[HardwareEnvelope]:
    """Sort by (peak compute bandwidth, vector lanes) descending."""
    def key(e: HardwareEnvelope) -> tuple[float, int]:
        peak = 0.0
        if getattr(e, "peak_compute_per_dtype", None):
            peak = max(e.peak_compute_per_dtype.values()) if e.peak_compute_per_dtype else 0.0
        return (peak, e.vector_lanes)
    return sorted(envelopes, key=key, reverse=True)


# ---------------------------------------------------------------------------
# LLM prompt scaffolding
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATE = """\
You are a compiler dispatch oracle. Given a region of operators and one
or more candidate hardware targets, decide for each target which
granularity (MICRO ukernel / NORMAL kernel / MEGA persistent kernel)
the region should be dispatched as, and pick the BEST target overall
under the stated optimisation budget.

REGION
------
{region}

CANDIDATE TARGETS (envelope summaries)
--------------------------------------
{envelopes}

DETERMINISTIC PRIORS (per target)
---------------------------------
{priors}

PERF BUDGET
-----------
{budget}

OBJECTIVE
---------
{objective}

Reply ONLY with a JSON object of this shape:

{{
  "per_target": {{
    "<target_name>": {{
      "granularity": "micro" | "normal" | "mega",
      "rationale":   "<one short sentence>"
    }}
  }},
  "best_target": "<target_name>",
  "best_rationale": "<one short sentence>"
}}
"""


def _build_prompt(
    region: Sequence[KernelContractV3],
    envelopes: Sequence[HardwareEnvelope],
    priors: dict[str, GranularityVerdict],
    perf_budget_us: float | None,
    objective: Objective,
) -> str:
    return _PROMPT_TEMPLATE.format(
        region=_summarise_region(region),
        envelopes="\n".join(f"  - {_summarise_envelope(e)}" for e in envelopes),
        priors="\n".join(
            f"  - {name}: granularity={v.granularity.value} "
            f"(confidence={v.confidence:.2f}); reason: {v.reason}"
            for name, v in priors.items()
        ),
        budget=f"{perf_budget_us:.1f} us" if perf_budget_us else "unspecified",
        objective=objective.value,
    )


def _parse_llm_decision(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extractor — accepts the response as raw JSON or a
    JSON object embedded in surrounding text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to scanning for the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _granularity_from_str(s: str) -> Granularity | None:
    s = (s or "").strip().lower()
    return {
        "micro":  Granularity.MICRO,
        "normal": Granularity.NORMAL,
        "mega":   Granularity.MEGA,
    }.get(s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide_dispatch(
    region: Sequence[KernelContractV3],
    envelopes: Sequence[HardwareEnvelope],
    *,
    perf_budget_us: float | None = None,
    objective: Objective = Objective.LATENCY,
    llm: CompGenLLMProtocol | None = None,
) -> MultiTargetDispatchDecision:
    """Decide per-target granularity + pick the best target for ``region``.

    Args:
        region: One-or-more KernelContractV3 — the candidate cluster.
        envelopes: Candidate hardware envelopes (must be ≥ 1).
        perf_budget_us: Optional latency budget in microseconds.
        objective: LATENCY / THROUGHPUT / MEMORY / ENERGY.
        llm: Optional LLM client. When ``None``, a deterministic
            heuristic + the granularity_oracle's verdicts decide.

    Returns:
        ``MultiTargetDispatchDecision`` with per-target verdicts and a
        single ``best_target`` selection.

    Raises:
        ValueError: when envelopes is empty or no target offers a legal
            dispatch path.
    """
    if not envelopes:
        raise ValueError("decide_dispatch requires at least one HardwareEnvelope")

    # 1. Per-target deterministic prior from the oracle.
    priors: dict[str, GranularityVerdict] = {}
    for env in envelopes:
        priors[env.target_name] = recommend_granularity(
            region, env, perf_target_us=perf_budget_us,
        )

    # 2. Optional LLM override.
    llm_decisions: dict[str, Granularity] = {}
    llm_rationale: dict[str, str] = {}
    llm_best: str | None = None
    llm_best_rationale = ""
    used_llm = False

    if llm is not None:
        prompt = _build_prompt(region, envelopes, priors, perf_budget_us, objective)
        request = GenerationRequest(
            prompt_template=prompt,
            context=PromptContext(
                model_ir_summary=_summarise_region(region),
                target_profile_summary="\n".join(
                    _summarise_envelope(e) for e in envelopes
                ),
                available_transforms=[],
                kernel_contracts=[],
                objective=objective,
            ),
            config=LLMConfig(model="dispatch-decision", temperature=0.2,
                             max_tokens=512),
            artifact_type="dispatch_decision",
        )
        try:
            response = llm.generate(request)
            parsed = _parse_llm_decision(response.raw_text)
            if parsed:
                used_llm = True
                for tname, payload in (parsed.get("per_target") or {}).items():
                    g = _granularity_from_str(payload.get("granularity", ""))
                    if g is not None:
                        llm_decisions[tname] = g
                        llm_rationale[tname] = str(payload.get("rationale", ""))
                llm_best = parsed.get("best_target")
                llm_best_rationale = str(parsed.get("best_rationale", ""))
        except Exception as exc:                 # noqa: BLE001
            llm_rationale["__error__"] = f"LLM call failed: {type(exc).__name__}: {exc}"

    # 3. Compose per-target decisions, validating legality.
    per_target: dict[str, TargetDispatchDecision] = {}
    for env in envelopes:
        prior = priors[env.target_name]
        chosen = llm_decisions.get(env.target_name, prior.granularity)
        rationale = llm_rationale.get(
            env.target_name,
            prior.reason if not used_llm else f"LLM defaulted to oracle: {prior.reason}",
        )
        adapter = select_adapter(env.target_name)
        legal, illegal_reason = _adapter_supports(adapter, chosen)
        # Fallback chain when the chosen granularity is illegal:
        #   1. oracle's prior (if different from chosen)
        #   2. NORMAL (the universal dispatch model — every adapter
        #      that hosts anything hosts SYNC)
        if not legal:
            chain: list[Granularity] = []
            if prior.granularity is not chosen:
                chain.append(prior.granularity)
            if Granularity.NORMAL not in chain and chosen is not Granularity.NORMAL:
                chain.append(Granularity.NORMAL)
            for fallback in chain:
                fallback_legal, fallback_reason = _adapter_supports(adapter, fallback)
                if fallback_legal:
                    rationale = (
                        f"chose {chosen.value} but {illegal_reason}; "
                        f"falling back to {fallback.value} ({prior.reason})"
                    )
                    chosen = fallback
                    legal = True
                    illegal_reason = ""
                    break
        per_target[env.target_name] = TargetDispatchDecision(
            target=env.target_name,
            granularity=chosen,
            adapter_name=adapter.name,
            rationale=rationale,
            confidence=prior.confidence,
            deterministic_prior=prior,
            legal=legal,
            illegal_reason=illegal_reason,
        )

    # 4. Pick best_target.
    legal_targets = [t for t, d in per_target.items() if d.legal]
    if not legal_targets:
        raise ValueError(
            "no candidate target offers a legal dispatch path: "
            + "; ".join(f"{t}={d.illegal_reason}" for t, d in per_target.items())
        )

    if used_llm and llm_best in legal_targets:
        best = llm_best
        best_rationale = llm_best_rationale or per_target[best].rationale
    else:
        # Heuristic: highest-throughput envelope (or only-legal fall-through).
        ranked = _rank_envelopes_by_throughput(
            [e for e in envelopes if e.target_name in legal_targets]
        )
        best = ranked[0].target_name
        best_rationale = (
            f"heuristic ranker picked {best} as highest-throughput legal target; "
            f"granularity={per_target[best].granularity.value}"
        )

    return MultiTargetDispatchDecision(
        region_summary=_summarise_region(region),
        per_target=per_target,
        best_target=best,
        best_rationale=best_rationale,
        used_llm=used_llm,
    )


__all__ = [
    "MultiTargetDispatchDecision",
    "TargetDispatchDecision",
    "decide_dispatch",
]
