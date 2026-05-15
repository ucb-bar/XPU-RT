"""Bridge between LLM invent-slot proposals and Recipe IR propose-ops.

Parallel to :mod:`compgen.agent.recipe_bridge` (which converts typed
env Actions ↔ candidate recipe ops). This module handles the *invent*
path: when :meth:`LLMDrivenCompiler.step_invent` gets an ``accepted``
gate result from the registered slot, it calls
:func:`proposal_to_recipe_op` to turn the proposal dict into a real
Recipe IR op, then appends that op to the live recipe module. Without
this step, the 12 ``recipe.propose_*`` ops are dead code — they exist
in the dialect but nothing constructs them.

Design mirrors ``recipe_bridge.action_to_recipe_op`` on purpose: one
function, slot-name-driven dispatch, returns None for slots we don't
map yet (the caller just skips appending).

Scope for P5.1 (this pass):
- ``propose_fusion``                 → :class:`ProposeFusionOp`
- ``propose_megakernel_synthesis``   → :class:`ProposeMegakernelSynthesisOp`
- ``propose_layout_plan``            → :class:`ProposeLayoutPlanOp`
- ``propose_dequant_fusion``         → :class:`ProposeDequantFusionOp`

Every other slot name returns None; future passes extend this file.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Operation

from compgen.ir.recipe.attrs import ProvenanceAttr
from compgen.ir.recipe.ops_propose import (
    ProposeDequantFusionOp,
    ProposeFusionOp,
    ProposeLayoutPlanOp,
    ProposeMegakernelSynthesisOp,
    ProposePayload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(iteration: int) -> ProvenanceAttr:
    return ProvenanceAttr("llm_invent", iteration)


def _sym_name(slot_name: str, iteration: int) -> StringAttr:
    return StringAttr(f"{slot_name}_{iteration}")


def _region_refs(regions: Any) -> ArrayAttr:
    """Normalise a list-of-strings / list-of-symbols into an ArrayAttr of SymbolRefAttr.

    Accepts whatever the LLM put in ``proposal['chosen']`` — strings,
    single-element tuples, bare names. Rejects empty inputs with a
    ``ValueError`` since every propose-op that takes regions requires
    at least one.
    """
    if regions is None:
        raise ValueError("chosen.grouped_regions / fused_region_refs is missing")
    if isinstance(regions, str):
        regions = [regions]
    refs: list[SymbolRefAttr] = []
    for r in regions:
        if isinstance(r, SymbolRefAttr):
            refs.append(r)
        elif isinstance(r, str):
            refs.append(SymbolRefAttr(r))
        else:
            # dict entry — pull name/id out
            name = (r.get("name") if isinstance(r, dict) else None) or str(r)
            refs.append(SymbolRefAttr(name))
    if not refs:
        raise ValueError("need at least one region ref")
    return ArrayAttr(refs)


def _payload_attr(
    proposal: dict[str, Any],
    *,
    llm_turn_id: str,
    baseline_seed_source: str = "",
) -> StringAttr:
    """Assemble the JSON payload every propose-op stores in its ``payload`` attr."""
    payload = ProposePayload(
        candidates=list(proposal.get("candidates", [])),
        chosen=dict(proposal.get("chosen", {})),
        target_feature_justification=str(proposal.get("target_feature_justification", "")),
        gate_result=dict(proposal.get("gate_result", {})),
        select_vs_invent=proposal.get("select_vs_invent", "invent"),
        llm_turn_id=llm_turn_id,
        baseline_seed_source=baseline_seed_source,
    )
    return StringAttr(payload.to_json())


# ---------------------------------------------------------------------------
# Per-slot constructors
# ---------------------------------------------------------------------------


def _build_propose_fusion(
    proposal: dict[str, Any],
    iteration: int,
    llm_turn_id: str,
) -> Operation:
    chosen = proposal.get("chosen") or {}
    regions = chosen.get("grouped_regions") or chosen.get("regions") or chosen.get("members")
    return ProposeFusionOp.build(
        properties={
            "sym_name": _sym_name("propose_fusion", iteration),
            "grouped_regions": _region_refs(regions),
            "payload": _payload_attr(proposal, llm_turn_id=llm_turn_id),
            "provenance": _prov(iteration),
        }
    )


def _build_propose_megakernel_synthesis(
    proposal: dict[str, Any],
    iteration: int,
    llm_turn_id: str,
) -> Operation:
    chosen = proposal.get("chosen") or {}
    regions = chosen.get("fused_region_refs") or chosen.get("grouped_regions") or chosen.get("regions")
    props: dict[str, Any] = {
        "sym_name": _sym_name("propose_megakernel_synthesis", iteration),
        "fused_region_refs": _region_refs(regions),
        "payload": _payload_attr(proposal, llm_turn_id=llm_turn_id),
        "provenance": _prov(iteration),
    }
    target_device = chosen.get("target_device_ref") or chosen.get("device_ref")
    if isinstance(target_device, str) and target_device:
        props["target_device_ref"] = SymbolRefAttr(target_device)
    return ProposeMegakernelSynthesisOp.build(properties=props)


def _build_propose_layout_plan(
    proposal: dict[str, Any],
    iteration: int,
    llm_turn_id: str,
) -> Operation:
    chosen = proposal.get("chosen") or {}
    region = chosen.get("region_ref") or chosen.get("region")
    if not isinstance(region, str) or not region:
        raise ValueError("propose_layout_plan requires chosen.region_ref as a string")
    return ProposeLayoutPlanOp.build(
        properties={
            "sym_name": _sym_name("propose_layout_plan", iteration),
            "region_ref": SymbolRefAttr(region),
            "payload": _payload_attr(proposal, llm_turn_id=llm_turn_id),
            "provenance": _prov(iteration),
        }
    )


def _build_propose_dequant_fusion(
    proposal: dict[str, Any],
    iteration: int,
    llm_turn_id: str,
) -> Operation:
    chosen = proposal.get("chosen") or {}
    region = chosen.get("region_ref") or chosen.get("region")
    if not isinstance(region, str) or not region:
        raise ValueError("propose_dequant_fusion requires chosen.region_ref as a string")
    return ProposeDequantFusionOp.build(
        properties={
            "sym_name": _sym_name("propose_dequant_fusion", iteration),
            "region_ref": SymbolRefAttr(region),
            "payload": _payload_attr(proposal, llm_turn_id=llm_turn_id),
            "provenance": _prov(iteration),
        }
    )


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


_BUILDERS = {
    "propose_fusion": _build_propose_fusion,
    "propose_megakernel_synthesis": _build_propose_megakernel_synthesis,
    "propose_layout_plan": _build_propose_layout_plan,
    "propose_dequant_fusion": _build_propose_dequant_fusion,
}


def proposal_to_recipe_op(
    slot_name: str,
    proposal: dict[str, Any],
    *,
    iteration: int = 0,
    llm_turn_id: str = "",
) -> Operation | None:
    """Convert an invent-slot proposal into its Recipe IR propose-op.

    Returns None when ``slot_name`` isn't mapped yet — caller decides
    whether to skip or log. All mapped slots raise ``ValueError`` on
    malformed proposals (missing required fields in ``chosen``); the
    caller should convert that to a gate rejection so the LLM can retry.

    The returned op has ``verify()`` called once before return to fail
    fast on schema violations, matching ``action_to_recipe_op``.
    """
    builder = _BUILDERS.get(slot_name)
    if builder is None:
        return None
    op = builder(proposal, iteration, llm_turn_id)
    # Fail fast — surface schema violations at construction, not at
    # lowering time when the remediation context is long gone.
    op.verify()
    return op


def supported_slot_names() -> tuple[str, ...]:
    """Return every slot name that has a propose-op builder registered."""
    return tuple(sorted(_BUILDERS.keys()))


__all__ = [
    "proposal_to_recipe_op",
    "supported_slot_names",
]
