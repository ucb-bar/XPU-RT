"""Phase-4 LLM tools for ETC megakernel synthesis.

Two deterministic tools the LLM can invoke during Phase 4 (kernel
contract):

    * ``propose_megakernel_layout`` -- given a list of candidate regions
      and an optional inter-region edge spec, return a fully-formed
      ``ProposePayload`` for ``recipe.propose_megakernel_synthesis``.
      Backed by the same baseline seed that the deterministic policy
      uses, so the LLM gets a structurally valid starting point and only
      has to reason about which regions belong in which megakernel.

    * ``pick_scheduler_strategy`` -- given a megakernel id and a
      ``has_data_dependent_edges`` flag, return ``static`` or
      ``dynamic``.  Encodes ETC's empirical guidance (Tables 2/3 of the
      paper): static for predictable workloads, dynamic for data-
      dependent ones.

Both tools auto-register on import (consistent with the
``observability`` / ``verification`` modules) so the LLM registry knows
about them as soon as ``compgen.llm.tools`` is imported.
"""

from __future__ import annotations

from typing import Any

from compgen.agent.invent_slots.seeds import (
    propose_megakernel_synthesis_seed,
    propose_scheduling_policy_seed,
)
from compgen.llm.registry import (
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)


# ---------------------------------------------------------------------------
# propose_megakernel_layout
# ---------------------------------------------------------------------------


def _propose_megakernel_layout_impl(
    *,
    candidate_regions: list[str] | tuple[str, ...] = (),
    inter_region_edges: list[dict[str, Any]] | None = None,
    task_shape: list[int] | tuple[int, ...] = (1,),
    megakernel_name: str | None = None,
) -> dict[str, Any]:
    """Build a structurally valid megakernel synthesis payload.

    Returns ``{"status": "ok", "payload": <ProposePayload-shaped dict>}``.
    """
    payload = propose_megakernel_synthesis_seed(
        candidate_regions=list(candidate_regions),
        inter_region_edges=list(inter_region_edges or []),
        task_shape=list(task_shape),
        megakernel_name=megakernel_name,
    )
    return {
        "status": "ok",
        "payload": payload,
        "megakernel_name": payload["chosen"]["megakernel_name"],
    }


propose_megakernel_layout = Tool(
    name="propose_megakernel_layout",
    phase=4,
    kind="tool",
    wraps_pass="recipe.propose_megakernel_synthesis",
    autocomp_cost_impact="very_high",
    args=(
        ToolArg(
            "candidate_regions",
            "list[str]",
            "region symbol refs to fuse into one megakernel",
            required=True,
        ),
        ToolArg(
            "inter_region_edges",
            "list[dict]",
            "[{shape:[..], wait_count:int}, ...] one per producer->consumer edge",
            required=False,
            default=(),
        ),
        ToolArg(
            "task_shape",
            "list[int]",
            "per-region tile-task grid extents",
            required=False,
            default=(1,),
        ),
        ToolArg(
            "megakernel_name",
            "str",
            "optional explicit name for the resulting megakernel",
            required=False,
            default=None,
        ),
    ),
    result=ToolResult(
        "ProposePayload",
        "structurally valid recipe.propose_megakernel_synthesis payload",
    ),
    description=(
        "Construct a megakernel synthesis proposal payload from a region "
        "cluster + edge spec, suitable for emission as a "
        "recipe.propose_megakernel_synthesis op."
    ),
    impl=_propose_megakernel_layout_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# pick_scheduler_strategy
# ---------------------------------------------------------------------------


def _pick_scheduler_strategy_impl(
    *,
    megakernel_ref: str,
    has_data_dependent_edges: bool = False,
    sm_count: int = 108,
    early_push: bool = False,
    data_dependent_edges: list[str] | None = None,
) -> dict[str, Any]:
    """Pick a scheduling policy and return a propose-policy payload."""
    payload = propose_scheduling_policy_seed(
        sm_count=sm_count,
        has_data_dependent_edges=has_data_dependent_edges,
        data_dependent_edges=data_dependent_edges or [],
    )
    payload["chosen"]["early_push"] = bool(early_push)
    return {
        "status": "ok",
        "megakernel_ref": megakernel_ref,
        "policy": payload["chosen"]["policy"],
        "payload": payload,
    }


pick_scheduler_strategy = Tool(
    name="pick_scheduler_strategy",
    phase=4,
    kind="tool",
    wraps_pass="recipe.propose_scheduling_policy",
    autocomp_cost_impact="high",
    args=(
        ToolArg(
            "megakernel_ref",
            "str",
            "symbol ref of the event.graph the policy applies to",
            required=True,
        ),
        ToolArg(
            "has_data_dependent_edges",
            "bool",
            "true when the megakernel contains topk / exp_indptr-style edges",
            required=False,
            default=False,
        ),
        ToolArg(
            "sm_count",
            "int",
            "target SM count (informational; baked into the schedule annotation)",
            required=False,
            default=108,
        ),
        ToolArg(
            "early_push",
            "bool",
            "enable Appendix-E early-push optimization (dynamic only)",
            required=False,
            default=False,
        ),
        ToolArg(
            "data_dependent_edges",
            "list[str]",
            "names of runtime int tensors that drive event triggers",
            required=False,
            default=(),
        ),
    ),
    result=ToolResult(
        "ProposePayload",
        "structurally valid recipe.propose_scheduling_policy payload",
    ),
    description=(
        "Decide static vs dynamic scheduling for a megakernel and emit "
        "the corresponding recipe.propose_scheduling_policy payload."
    ),
    impl=_pick_scheduler_strategy_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> list[str]:
    """Register both megakernel tools.  Idempotent."""
    registry = get_registry()
    registered: list[str] = []
    for tool in (propose_megakernel_layout, pick_scheduler_strategy):
        if registry.lookup_tool(tool.name, phase=tool.phase) is None:
            registry.register_tool(tool)
            registered.append(tool.name)
    return registered


# Auto-register so the registry knows about both tools as soon as
# ``compgen.llm.tools`` is imported.
register()


__all__ = [
    "pick_scheduler_strategy",
    "propose_megakernel_layout",
    "register",
]
