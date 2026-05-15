"""Pass 2: Attach layout hints from analysis plans.

Takes LayoutPlan dict from the LayoutPlanner and attaches
``compgen.layout_hint`` string attributes to ops based on their
region assignment.  Bridges analysis output to IR annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

if TYPE_CHECKING:
    from compgen.analysis.layout.planner import LayoutPlan

log = structlog.get_logger()

LAYOUT_HINT_ATTR = "compgen.layout_hint"
PREPACK_HINT_ATTR = "compgen.prepack_hint"


def attach_layout_hints(
    module: ModuleOp,
    plans: dict[str, LayoutPlan],
) -> ModuleOp:
    """Annotate ops with layout hints from analysis plans.

    For each op, checks if its ``compgen.region_id`` attribute matches a
    plan key.  If so, attaches the preferred output layout and any prepack
    hints.  Falls back to existing ``compgen.encoding`` if no plan exists.

    Args:
        module: The xDSL ModuleOp to annotate.
        plans: Mapping of region id to ``LayoutPlan`` instances.

    Returns:
        The same ModuleOp with layout hint attributes attached.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    if not plans:
        log.debug("layout.attach_hints", msg="no plans provided, skipping")
        return module

    hints_attached = 0
    prepack_hints = 0

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if not op.results:
            continue

        # Try to match op to a region plan via region_id attribute
        region_id: str | None = None
        if "compgen.region_id" in op.attributes:
            region_id_attr = op.attributes["compgen.region_id"]
            if hasattr(region_id_attr, "data"):
                region_id = region_id_attr.data

        # Also match by op name pattern for common cases
        if region_id is None:
            for plan_key in plans:
                if plan_key in op.name:
                    region_id = plan_key
                    break

        if region_id and region_id in plans:
            plan = plans[region_id]
            op.attributes[LAYOUT_HINT_ATTR] = StringAttr(plan.preferred_output_layout)
            hints_attached += 1

            if plan.prepack_candidates:
                op.attributes[PREPACK_HINT_ATTR] = StringAttr(",".join(plan.prepack_candidates))
                prepack_hints += 1

    log.debug(
        "layout.attach_hints",
        hints_attached=hints_attached,
        prepack_hints=prepack_hints,
    )
    return module


__all__ = ["attach_layout_hints"]
