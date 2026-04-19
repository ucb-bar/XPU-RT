"""Structural gate — xDSL verifier + schema-dict shape check.

Accepts either:
  * a ModuleOp / xDSL Operation via ``ctx["module"]`` (runs ``verify()``).
  * a plain dict proposal (checks ``schema_version`` / ``chosen`` keys).

Mirrors the :func:`compgen.llm.tools.verification._run_structural_check_impl`
but adapts to the invent-slot gate signature.
"""

from __future__ import annotations

from typing import Any


def structural_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    """Return {status, details} per the InventSlot.gate_impl contract."""
    try:
        from xdsl.ir import Operation
    except ImportError:   # pragma: no cover
        Operation = object   # type: ignore

    module = ctx.get("module")
    if module is not None:
        if not isinstance(module, Operation):
            return {
                "status": "rejected",
                "details": {
                    "reason": f"ctx.module is not an xDSL Operation: {type(module).__name__}"
                },
            }
        try:
            module.verify()
        except Exception as e:   # noqa: BLE001
            return {
                "status": "rejected",
                "details": {"reason": "xdsl verify failed", "error": str(e)},
            }
        return {"status": "accepted", "details": {"kind": "xdsl_module"}}

    # No module in ctx — fall back to dict structure checks on the proposal.
    missing: list[str] = []
    if "chosen" not in proposal:
        missing.append("chosen")
    if "select_vs_invent" not in proposal:
        missing.append("select_vs_invent")
    if missing:
        return {
            "status": "rejected",
            "details": {"reason": "missing_required_keys", "missing": missing},
        }
    if proposal["select_vs_invent"] not in ("select", "invent"):
        return {
            "status": "rejected",
            "details": {
                "reason": "select_vs_invent must be 'select' or 'invent'",
                "value": proposal["select_vs_invent"],
            },
        }
    return {"status": "accepted", "details": {"kind": "dict_proposal"}}


__all__ = ["structural_gate"]
