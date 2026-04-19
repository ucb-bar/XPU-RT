"""Megakernel-specific verification gate.

Composes with :mod:`compgen.agent.gates.composite` to enforce the
ETC-paper invariants on a ``propose_megakernel_synthesis`` /
``propose_scheduling_policy`` proposal:

    1. The proposal payload declares at least one ``fused_region_ref``.
    2. Every event-tensor declaration has a non-empty shape and a
       non-negative ``wait_count``.
    3. The chosen scheduling policy is one of ``static`` / ``dynamic``.
    4. The target's capability spec advertises ``persistent_kernels``
       and ``semaphore_atomics`` (when ``ctx["target_features"]`` is
       supplied).
    5. No ``UkernelCallOp`` appears inside the candidate fusion region
       (preserves CLAUDE.md frozen architecture decision #14: ukernel
       boundary is the stable leaf-call surface and must not be crossed
       by the megakernel body).

Signature matches :class:`compgen.llm.registry.InventSlot.gate_impl`::

    (proposal: dict, **ctx) -> {"status": "accepted"|"rejected"|"deferred",
                                "details": {...}}
"""

from __future__ import annotations

from typing import Any


_REQUIRED_TARGET_FEATURES = ("persistent_kernels", "semaphore_atomics")
_VALID_POLICIES = ("static", "dynamic")


def _gather_event_decls(chosen: dict[str, Any]) -> list[dict[str, Any]]:
    decls = chosen.get("event_tensor_decls") or []
    if not isinstance(decls, list):
        return []
    return [d for d in decls if isinstance(d, dict)]


def _has_ukernel_call(graph: Any) -> bool:
    try:
        from compgen.ir.ukernel.ops import UkernelCallOp
    except Exception:
        return False
    if graph is None or not hasattr(graph, "body"):
        return False
    try:
        for op in graph.body.walk():
            if isinstance(op, UkernelCallOp):
                return True
    except Exception:  # pragma: no cover - graph type may not support walk()
        return False
    return False


def megakernel_persistent_kernel_gate(
    proposal: dict[str, Any],
    **ctx: Any,
) -> dict[str, Any]:
    """Validate an ETC-style megakernel proposal payload.

    Args:
        proposal: dict mirroring ``ProposePayload`` (must have ``chosen``).
        ctx:
            * ``target_features``: optional iterable of capability strings.
            * ``event_graph``:     optional ``event.GraphOp`` to walk for
                                   forbidden ``UkernelCallOp`` instances.

    Returns the gate result dict expected by ``InventSlot.gate_impl``.
    """
    chosen = proposal.get("chosen")
    if not isinstance(chosen, dict):
        return {
            "status": "rejected",
            "details": {"reason": "missing_or_invalid_chosen"},
        }

    # 1. fused_region_refs presence (only required for synthesis proposals).
    if "fused_region_refs" in chosen:
        regions = chosen.get("fused_region_refs") or []
        if not isinstance(regions, list) or not regions:
            return {
                "status": "rejected",
                "details": {"reason": "fused_region_refs is empty"},
            }

    # 2. event-tensor declarations sanity (only when present).
    decls = _gather_event_decls(chosen)
    for d in decls:
        shape = d.get("shape") or []
        if not isinstance(shape, list) or len(shape) == 0:
            return {
                "status": "rejected",
                "details": {"reason": "event_decl shape empty", "decl": d},
            }
        if any((not isinstance(s, int)) or (s < -1) for s in shape):
            return {
                "status": "rejected",
                "details": {"reason": "event_decl shape invalid", "decl": d},
            }
        wait_count = d.get("wait_count", 0)
        if not isinstance(wait_count, int) or wait_count < 0:
            return {
                "status": "rejected",
                "details": {
                    "reason": "event_decl wait_count must be non-negative int",
                    "decl": d,
                },
            }

    # 3. scheduling policy choice (only required for the policy slot).
    if "policy" in chosen:
        if chosen["policy"] not in _VALID_POLICIES:
            return {
                "status": "rejected",
                "details": {
                    "reason": "invalid scheduling policy",
                    "policy": chosen["policy"],
                    "valid": list(_VALID_POLICIES),
                },
            }

    # 4. target capability flags when supplied.
    target_features = ctx.get("target_features")
    if target_features is not None:
        feature_set = {str(f) for f in target_features}
        missing = [
            feat for feat in _REQUIRED_TARGET_FEATURES if feat not in feature_set
        ]
        if missing:
            return {
                "status": "rejected",
                "details": {
                    "reason": "target lacks megakernel capability flags",
                    "missing": missing,
                },
            }

    # 5. ukernel-call exclusion: a megakernel must not cross the ukernel
    # leaf-call boundary (preserves frozen architecture decision #14).
    if _has_ukernel_call(ctx.get("event_graph")):
        return {
            "status": "rejected",
            "details": {
                "reason": (
                    "megakernel body contains UkernelCallOp; ukernels are "
                    "stable leaf-call boundaries and may not be inlined"
                ),
            },
        }

    return {
        "status": "accepted",
        "details": {
            "event_decl_count": len(decls),
            "policy": chosen.get("policy"),
            "fused_region_count": len(chosen.get("fused_region_refs") or []),
        },
    }


__all__ = ["megakernel_persistent_kernel_gate"]
