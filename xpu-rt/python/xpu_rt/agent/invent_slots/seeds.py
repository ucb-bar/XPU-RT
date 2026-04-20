"""Deterministic baseline seeds for every invent-slot.

Each seed returns a ``ProposePayload``-shaped dict (per
``compgen.ir.recipe.ops_propose.ProposePayload``) with at minimum a
``chosen`` field so the default structural gate accepts it.

These are MVP seeds — more elaborate seeds (CP-SAT solved layouts,
PriorityFusion cost-modelled plans) land in follow-up waves alongside
real pass ports.
"""

from __future__ import annotations

from typing import Any


def _basic_payload(chosen: dict[str, Any], justification: str) -> dict[str, Any]:
    return {
        "candidates": [chosen],
        "chosen": chosen,
        "target_feature_justification": justification,
        "gate_result": {"status": "deferred", "details": {}},
        "select_vs_invent": "invent",
        "baseline_seed_source": "mvp",
    }


def propose_layout_plan_seed(**ctx: Any) -> dict[str, Any]:
    """Default: row-major, tile-free, byte-aligned."""
    chosen = {
        "layout": {
            "rank_order": [0, 1],
            "tile": [],
            "pack": [],
            "alignment_bytes": 64,
        },
        "est_cost": 0.0,
    }
    return _basic_payload(
        chosen,
        justification="preferred_layouts_v2 default (row-major, 64B aligned)",
    )


def propose_fusion_seed(**ctx: Any) -> dict[str, Any]:
    """Default: no fusion (conservative singleton)."""
    chosen = {
        "fusion_spec": {
            "grouped_regions": list(ctx.get("candidate_regions", [])),
            "target_family": "auto:elementwise",
        },
        "est_cost": 0.0,
        "est_peak_bytes": 0,
    }
    return _basic_payload(
        chosen,
        justification="fusion_cost_model.family_affinity default (conservative)",
    )


def propose_peephole_pattern_seed(**ctx: Any) -> dict[str, Any]:
    """Default: no pattern (LLM must override to propose one)."""
    chosen = {
        "pattern_class": "unknown",
        "input_schema": {},
        "output_schema": {},
        "rationale": "mvp seed - awaiting LLM proposal",
    }
    return _basic_payload(chosen, justification="no matching entry in ported pattern library")


def propose_numerics_plan_seed(**ctx: Any) -> dict[str, Any]:
    """Default: bf16 inputs with fp32 accumulator."""
    chosen = {
        "default_dtype": "bf16",
        "accumulator_dtype": "fp32",
        "narrowing_confidence": 0.9,
        "allow_bf16_for_contraction_inputs": True,
        "allow_fp8_for_weights": False,
    }
    return _basic_payload(
        chosen,
        justification="supported_dtypes seed (bf16 widespread, fp32 accumulator safe)",
    )


def propose_dequant_fusion_seed(**ctx: Any) -> dict[str, Any]:
    """Default: no dequant fusion (safe)."""
    chosen = {
        "fusion_pattern": "none",
        "safety": "reassoc_safe_only",
    }
    return _basic_payload(
        chosen,
        justification="reassoc_safe_only (conservative; LLM may propose relaxed variants)",
    )


# -------- Phase 5 invent-slot seeds (P15) --------


def propose_buffer_lifetime_plan_seed(**ctx: Any) -> dict[str, Any]:
    """Default: first-fit greedy lifetime plan — safe, low overhead."""
    chosen = {
        "coloring_policy": "first_fit",
        "alias_io": True,
        "remat_threshold_bytes": 0,
    }
    return _basic_payload(
        chosen,
        justification="first_fit is safe default; solve_memory backs it",
    )


def propose_rematerialization_plan_seed(**ctx: Any) -> dict[str, Any]:
    """Default: no remat — safest on targets that can afford the buffers."""
    chosen = {
        "recompute_set": [],
        "memory_budget_bytes": int(ctx.get("memory_budget_bytes", 0)),
        "recompute_cost_tolerance": 1.2,
    }
    return _basic_payload(
        chosen,
        justification="no remat by default (opt-in when memory pressure observed)",
    )


# -------- Phase 4 ETC megakernel invent-slot seeds --------


def propose_megakernel_synthesis_seed(**ctx: Any) -> dict[str, Any]:
    """Default seed for ETC megakernel synthesis (Phase 4).

    Bundles the candidate region cluster into a single megakernel name
    derived from the regions; declares one device-scope event tensor per
    inter-region edge.  Conservative: when the caller does not supply
    ``candidate_regions`` we leave the cluster empty so the gate rejects
    the seed and the LLM is forced to invent a real plan.
    """
    region_refs: list[str] = list(ctx.get("candidate_regions", []))
    edges: list[dict[str, Any]] = list(ctx.get("inter_region_edges", []))
    mk_name = ctx.get("megakernel_name") or (f"mk_{'_'.join(region_refs)}" if region_refs else "mk_unspecified")
    event_decls = [
        {
            "name": f"E_{i}",
            "shape": list(edge.get("shape", [1])),
            "wait_count": int(edge.get("wait_count", 1)),
            "scope": "device",
            "counter_dtype": "i32",
        }
        for i, edge in enumerate(edges)
    ]
    chosen = {
        "megakernel_name": mk_name,
        "fused_region_refs": region_refs,
        "event_tensor_decls": event_decls,
        "task_partition": {r: list(ctx.get("task_shape", [1])) for r in region_refs},
        "prefetch_annotations": [],
    }
    return _basic_payload(
        chosen,
        justification=(
            "ETC megakernel synthesis seed (Algorithm 1, Jin et al. MLSys '26): "
            "bundles candidate regions into one persistent kernel coordinated "
            "by counter-based Event Tensors.  Requires target capabilities "
            "persistent_kernels and semaphore_atomics."
        ),
    )


def propose_scheduling_policy_seed(**ctx: Any) -> dict[str, Any]:
    """Default seed: ``static`` scheduling unless the megakernel has
    data-dependent edges (then prefer ``dynamic``).

    Phase A of the ETC integration only uses the ``static`` branch; the
    ``dynamic`` branch is kept for Phase B's data-dependent workloads
    (MoE, attention with data-dep masks).
    """
    has_data_dep = bool(ctx.get("has_data_dependent_edges", False))
    policy = "dynamic" if has_data_dep else "static"
    chosen = {
        "policy": policy,
        "sm_count": int(ctx.get("sm_count", 108)),
        "early_push": False,
        "dynamic_features": list(ctx.get("data_dependent_edges", [])) if has_data_dep else [],
    }
    return _basic_payload(
        chosen,
        justification=(
            "static is preferred for predictable graphs (precomputed per-SM "
            "queue, minimal runtime overhead); dynamic is selected when "
            "data-dependent edges (e.g. MoE topk routing) are present."
        ),
    )


__all__ = [
    "propose_buffer_lifetime_plan_seed",
    "propose_dequant_fusion_seed",
    "propose_fusion_seed",
    "propose_layout_plan_seed",
    "propose_megakernel_synthesis_seed",
    "propose_numerics_plan_seed",
    "propose_peephole_pattern_seed",
    "propose_rematerialization_plan_seed",
    "propose_scheduling_policy_seed",
]
