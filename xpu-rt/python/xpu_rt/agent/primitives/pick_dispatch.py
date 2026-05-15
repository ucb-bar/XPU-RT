"""P3.5 — pick_dispatch primitive.

Translates the natural-language deployment constraints into a typed
``SetDispatchMode`` op. The closed-enum dispatch modes match the
existing CompGen runtime:

* ``sync`` — single-stream, blocking. Best for tight latency budgets.
* ``static_plan`` — pre-planned static schedule, no runtime decisions.
* ``async`` — overlapping launches; higher throughput, weaker tail
  guarantees.
* ``megakernel`` — single persistent kernel; only for fused regions
  that fit the target's SM count.

Deterministic fallback: cost-table lookup keyed on
``(workload_class, latency_budget_ms)``.
"""

from __future__ import annotations

from typing import Any, Final

from compgen.llm.call_site import llm_call_site, register_fallback

DISPATCH_MODES: Final[tuple[str, ...]] = (
    "sync",
    "static_plan",
    "async",
    "megakernel",
)

PICK_DISPATCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["dispatch_mode", "rationale", "fallback_used"],
    "properties": {
        "dispatch_mode": {"enum": list(DISPATCH_MODES)},
        "rationale": {"type": "string"},
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def _cost_table_lookup(
    workload_class: str, deployment_constraints: dict[str, Any]
) -> str:
    """Closed-form mapping used by the fallback.

    The table is intentionally short and deterministic — it embodies
    the *default* CompGen would pick without LLM input. The LLM's job
    is only to *override* this when the natural-language context
    clearly demands it.
    """

    latency_ms = float(deployment_constraints.get("latency_budget_ms") or 1_000.0)
    workload = workload_class.lower()
    if workload in {"streaming", "real_time"}:
        return "sync" if latency_ms < 50.0 else "async"
    if workload in {"batched_inference", "throughput"}:
        return "async"
    if workload in {"one_shot", "compile_once"}:
        return "static_plan"
    if workload in {"persistent", "rolling"}:
        return "megakernel"
    # Default — safest mode that any target supports.
    return "sync"


@register_fallback("pick_dispatch_cost_table")
def _pick_dispatch_fallback(
    workload_class: str,
    deployment_constraints: dict[str, Any],
    region_dossier: dict[str, Any],
) -> dict[str, Any]:
    mode = _cost_table_lookup(workload_class, deployment_constraints)
    return {
        "dispatch_mode": mode,
        "rationale": (
            f"cost-table fallback for workload_class={workload_class!r}, "
            f"latency_budget_ms={deployment_constraints.get('latency_budget_ms')}"
        ),
        "fallback_used": True,
    }


@llm_call_site(
    site_id="pick_dispatch",
    leverage="Translate English deployment context (latency budget, "
    "batchedness, memory budget) into a typed dispatch mode.",
    inputs=[
        "workload_class:str",
        "deployment_constraints:dict",
        "region_dossier:dict",
    ],
    output_schema=PICK_DISPATCH_OUTPUT_SCHEMA,
    forbidden=["pick_dispatch_without_cost_table_check"],
    fallback="pick_dispatch_cost_table",
)
def pick_dispatch(
    workload_class: str,
    deployment_constraints: dict[str, Any],
    region_dossier: dict[str, Any],
) -> dict[str, Any]:
    return _pick_dispatch_fallback(workload_class, deployment_constraints, region_dossier)


__all__ = ["DISPATCH_MODES", "PICK_DISPATCH_OUTPUT_SCHEMA", "pick_dispatch"]
