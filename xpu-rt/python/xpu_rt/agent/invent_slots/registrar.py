"""Registers the 5 core invent-slots into the global LLM registry.

Each slot gets the composite default gate (structural + differential),
a baseline-seed callable from :mod:`seeds`, and the Recipe-IR op-name
it produces. Idempotent: second call is a no-op.
"""

from __future__ import annotations

from typing import Any

from compgen.agent.gates import (
    composite_gate,
    differential_gate,
    structural_gate,
)
from compgen.agent.invent_slots import seeds
from compgen.llm.registry import InventSlot, Registry, get_registry


def _default_composite_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    """Default gate: structural first, differential if ctx provides it."""
    gate_list = [structural_gate]
    if ctx.get("ref_fn") is not None and ctx.get("got_fn") is not None:
        gate_list.append(differential_gate)
    return composite_gate(proposal, gates=gate_list, **ctx)


# Registration metadata: name → (phase, output_op, description, seed, autocomp_cost_impact)
_SLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "propose_layout_plan",
        "phase": 3,
        "input_schema": "region + target_resource_model + constraint_graph",
        "output_op": "recipe.propose_layout_plan",
        "autocomp_cost_impact": "very_high",
        "description": "LLM proposes target-aligned physical layouts per region.",
        "gate": "structural + differential (+ SMT opt-in)",
        "baseline_seed": seeds.propose_layout_plan_seed,
    },
    {
        "name": "propose_fusion",
        "phase": 3,
        "input_schema": "region + target_features + cost_budget",
        "output_op": "recipe.propose_fusion",
        "autocomp_cost_impact": "very_high",
        "description": "LLM proposes fusion boundaries. Primary autocomp-cost knob.",
        "gate": "structural + differential (+ cost_model + liveness opt-in)",
        "baseline_seed": seeds.propose_fusion_seed,
    },
    {
        "name": "propose_peephole_pattern",
        "phase": 2,
        "input_schema": "region + input_schema + output_schema",
        "output_op": "recipe.propose_peephole_pattern",
        "autocomp_cost_impact": "very_high",
        "description": "LLM proposes a novel peephole pattern not in ported library.",
        "gate": "structural + differential + SMT-on-demand",
        "baseline_seed": seeds.propose_peephole_pattern_seed,
    },
    {
        "name": "propose_numerics_plan",
        "phase": 2,
        "input_schema": "region + policy_schema",
        "output_op": "recipe.propose_numerics_plan",
        "autocomp_cost_impact": "medium",
        "description": "LLM proposes per-region numerics policy tied to target dtypes.",
        "gate": "structural + differential (tight tolerance)",
        "baseline_seed": seeds.propose_numerics_plan_seed,
    },
    {
        "name": "propose_dequant_fusion",
        "phase": 2,
        "input_schema": "region + extended_patterns",
        "output_op": "recipe.propose_dequant_fusion",
        "autocomp_cost_impact": "very_high",
        "description": "LLM proposes novel dequant fusion patterns (group-wise, 4-bit, etc.).",
        "gate": "structural + differential + SMT-on-demand",
        "baseline_seed": seeds.propose_dequant_fusion_seed,
    },
    # Phase 5 invent-slots (P15)
    {
        "name": "propose_buffer_lifetime_plan",
        "phase": 5,
        "input_schema": "execution_plan + memory_budget",
        "output_op": "recipe.propose_buffer_lifetime_plan",
        "autocomp_cost_impact": "indirect",
        "description": (
            "LLM proposes buffer lifetime + aliasing strategy for memory-"
            "constrained targets. Gate = liveness + memory footprint + "
            "differential."
        ),
        "gate": "structural + liveness (+ differential opt-in)",
        "baseline_seed": seeds.propose_buffer_lifetime_plan_seed,
    },
    {
        "name": "propose_rematerialization_plan",
        "phase": 5,
        "input_schema": "execution_plan + memory_budget + recompute_cost_tolerance",
        "output_op": "recipe.propose_rematerialization_plan",
        "autocomp_cost_impact": "indirect",
        "description": (
            "LLM proposes remat plan bounded by memory_budget and "
            "recompute_cost_tolerance. Gate = liveness + differential."
        ),
        "gate": "structural + liveness + differential",
        "baseline_seed": seeds.propose_rematerialization_plan_seed,
    },
    # Phase 4 ETC megakernel invent-slots
    {
        "name": "propose_megakernel_synthesis",
        "phase": 4,
        "input_schema": (
            "candidate_regions + inter_region_edges + target_features "
            "(persistent_kernels, semaphore_atomics, sm_count) + latency_budget"
        ),
        "output_op": "recipe.propose_megakernel_synthesis",
        "autocomp_cost_impact": "very_high",
        "description": (
            "LLM proposes fusing a region cluster into a single persistent "
            "megakernel coordinated by Event Tensors (counter-based "
            "semaphores).  Replaces kernel-by-kernel launches with one "
            "fused kernel + per-SM task queue.  Models the Event Tensor "
            "Compiler abstraction (Jin et al., MLSys '26)."
        ),
        "gate": "structural + differential (+ cost_model opt-in)",
        "baseline_seed": seeds.propose_megakernel_synthesis_seed,
    },
    {
        "name": "propose_scheduling_policy",
        "phase": 4,
        "input_schema": ("megakernel_ref + per_sm_resource_model + has_data_dependent_edges"),
        "output_op": "recipe.propose_scheduling_policy",
        "autocomp_cost_impact": "high",
        "description": (
            "LLM picks static (Algorithm 1) vs dynamic (Algorithm 2) "
            "scheduling for a megakernel.  Static = precomputed per-SM "
            "queue, minimal runtime overhead.  Dynamic = on-GPU push/pop "
            "scheduler, required for data-dependent task graphs (MoE)."
        ),
        "gate": "structural + differential",
        "baseline_seed": seeds.propose_scheduling_policy_seed,
    },
)


def register_invent_slots(registry: Registry | None = None) -> list[str]:
    """Register every default invent-slot into the registry (idempotent).

    After the canonical slots are registered, the cross-session
    graduation loop runs once — scanning ``~/.compgen/transcripts``
    for previously-accepted patterns that have cleared the
    workload+target thresholds and materialising them as registered
    ``Tool`` entries. Failures there are logged but swallowed; they
    never block canonical-slot registration.

    Returns the list of newly registered slot names (graduated tool
    names are NOT included — those show up in ``registry.list_tools``).
    """
    reg = registry or get_registry()
    newly_registered: list[str] = []
    for spec in _SLOT_SPECS:
        if reg.lookup_invent_slot(spec["name"], phase=spec["phase"]) is not None:
            continue
        slot = InventSlot(
            name=spec["name"],
            phase=spec["phase"],
            input_schema=spec["input_schema"],
            output_op=spec["output_op"],
            gate=spec["gate"],
            autocomp_cost_impact=spec["autocomp_cost_impact"],
            description=spec["description"],
            baseline_seed=spec["baseline_seed"],
            gate_impl=_default_composite_gate,
            stub=False,
        )
        reg.register_invent_slot(slot)
        newly_registered.append(spec["name"])

    # Cross-session invent-slot graduation — promotes patterns the LLM
    # has accepted across multiple workloads/targets into named tools.
    try:
        import os

        if not os.environ.get("COMPGEN_DISABLE_CROSS_SESSION_GRADUATION"):
            from compgen.promotion.cross_session import promote_pending_graduations

            promote_pending_graduations(reg)
    except Exception:  # noqa: BLE001
        pass

    # Self-extension graduation — promotes LLM-authored tools whose
    # sandboxed trials have cleared the N-pass thresholds. Requires an
    # ``authored_index`` to materialise a tool (we never synthesise
    # source from the trial log alone). When no index has been
    # registered yet this call is a no-op.
    try:
        import os

        if not os.environ.get("COMPGEN_DISABLE_AUTHORED_GRADUATION"):
            from compgen.agent.self_extension._index import (
                snapshot_authored_index,
            )
            from compgen.agent.self_extension.graduate import (
                promote_authored_tools,
            )

            promote_authored_tools(reg, authored_index=snapshot_authored_index())
    except Exception:  # noqa: BLE001
        pass

    return newly_registered


__all__ = ["register_invent_slots"]
