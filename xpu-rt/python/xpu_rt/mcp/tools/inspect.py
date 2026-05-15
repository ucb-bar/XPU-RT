"""MCP inspect tools: view_recipe, diff_recipe, list_phase_tools,
get_dossier, session_summary."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from compgen.llm.registry import get_registry
from compgen.mcp.session import SessionManager


def view_recipe(
    sm: SessionManager,
    *,
    session_id: str,
    max_ops: int = 80,
    focus: str | None = None,
) -> dict[str, Any]:
    """Return the token-efficient Recipe-IR view of the current session."""
    session = sm.get(session_id)
    driver = session.require_driver()
    view = driver.current_view(focus=focus, max_ops=max_ops)
    return {"ok": True, "session_id": session_id, "view": view}


def diff_recipe(
    sm: SessionManager,
    *,
    session_id: str,
    from_ckpt: str = "ckpt_0",
) -> dict[str, Any]:
    """Diff the current view against a prior checkpoint."""
    session = sm.get(session_id)
    driver = session.require_driver()
    diff = driver.diff_since(from_ckpt)
    return {"ok": True, "session_id": session_id, "diff": diff}


def checkpoint(
    sm: SessionManager,
    *,
    session_id: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Freeze the current view as a named checkpoint."""
    session = sm.get(session_id)
    driver = session.require_driver()
    ckpt = driver.checkpoint(label=label)
    return {"ok": True, "session_id": session_id, "ckpt_id": ckpt}


def list_phase_tools(
    sm: SessionManager,
    *,
    phase: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return the catalogue of tools + invent slots the LLM may invoke.

    Shape: ``{tools: [{name, phase, is_stub, args, result}], slots: [...]}``.
    """
    _ = session_id  # unused — registry is process-wide
    reg = get_registry()
    tools_out: list[dict[str, Any]] = []
    for t in reg.list_tools(phase=phase):
        tools_out.append(
            {
                "name": t.name,
                "phase": t.phase,
                "kind": t.kind,
                "is_stub": t.is_stub,
                "wraps_pass": t.wraps_pass,
                "description": t.description,
                "args": [asdict(a) for a in t.args],
                "result": asdict(t.result),
            }
        )
    slots_out: list[dict[str, Any]] = []
    for s in reg.list_invent_slots(phase=phase):
        slots_out.append(
            {
                "name": s.name,
                "phase": s.phase,
                "is_stub": s.is_stub,
                "output_op": s.output_op,
                "gate": s.gate,
                "description": s.description,
                "input_schema": s.input_schema,
            }
        )
    return {
        "ok": True,
        "tools": tools_out,
        "invent_slots": slots_out,
        "counts": reg.counts(),
    }


def get_dossier(
    sm: SessionManager,
    *,
    session_id: str,
    op_id: str | None = None,
) -> dict[str, Any]:
    """Return the deterministic graph-analysis dossier for this session."""
    session = sm.get(session_id)
    compiled = session.require_compiled()
    dossier = compiled.analysis_dossier
    if dossier is None:
        return {"ok": True, "session_id": session_id, "dossier": None}
    payload: dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "critical_path": list(getattr(dossier, "critical_path", ())),
        "repeated_patterns": dict(getattr(dossier, "repeated_patterns", {})),
        "pattern_count": sum(dict(getattr(dossier, "repeated_patterns", {})).values()),
    }
    # Region name translation table: recipe sym ('r_0') → {payload_id, role}.
    # Agents calling propose_* invent slots can pass either name form;
    # both resolve to the same ops. The role tag (e.g. "matmul",
    # "softmax", "rmsnorm") lets agents pick "the q_proj matmul"
    # without paging through view_recipe.
    driver = session.driver
    if driver is not None and driver.env.recipe is not None:
        from xdsl.dialects.builtin import StringAttr

        region_map: dict[str, dict[str, str]] = {}
        for rop in driver.env.recipe.body.block.ops:
            if rop.name != "recipe.region":
                continue
            sym = rop.properties.get("sym_name")
            pid = rop.properties.get("payload_region_id")
            role = rop.properties.get("role")
            if isinstance(sym, StringAttr) and isinstance(pid, StringAttr):
                entry: dict[str, str] = {"payload_id": pid.data}
                if isinstance(role, StringAttr) and role.data:
                    entry["role"] = role.data
                region_map[sym.data] = entry
        payload["region_map"] = region_map
        payload["region_count"] = len(region_map)
        # Reverse-index: role → list of recipe syms with that role.
        # Lets agents fetch "all matmul regions" without scanning.
        regions_by_role: dict[str, list[str]] = {}
        for sym, info in region_map.items():
            r = info.get("role")
            if r:
                regions_by_role.setdefault(r, []).append(sym)
        payload["regions_by_role"] = regions_by_role
    if op_id is not None:
        # Look up the op in the Recipe-IR view's focus so the LLM sees
        # the exact serialised form it referenced by id.
        driver = session.require_driver()
        payload["focused_view"] = driver.current_view(focus=op_id)
    return payload


def session_summary(
    sm: SessionManager,
    *,
    session_id: str,
) -> dict[str, Any]:
    """Return the driver's session summary."""
    session = sm.get(session_id)
    driver = session.require_driver()
    return {"ok": True, "session_id": session_id, "summary": driver.summary()}


INSPECT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "view_recipe",
        "description": "Return a token-efficient view of the current Recipe IR.",
        "phase": "inspect",
        "handler": view_recipe,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "max_ops": {"type": "integer", "default": 80},
                "focus": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "diff_recipe",
        "description": "Diff the current Recipe IR against a prior checkpoint.",
        "phase": "inspect",
        "handler": diff_recipe,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "from_ckpt": {"type": "string", "default": "ckpt_0"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "checkpoint",
        "description": "Freeze the current Recipe IR view as a named checkpoint.",
        "phase": "inspect",
        "handler": checkpoint,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "list_phase_tools",
        "description": "List the tools + invent-slots registered in the LLM registry.",
        "phase": "inspect",
        "handler": list_phase_tools,
        "input_schema": {
            "type": "object",
            "properties": {
                "phase": {"type": "integer"},
                "session_id": {"type": "string"},
            },
        },
    },
    {
        "name": "get_dossier",
        "description": "Return the deterministic graph-analysis dossier for this session.",
        "phase": "inspect",
        "handler": get_dossier,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "op_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "session_summary",
        "description": "Return the driver session summary (step index, hashes, counts).",
        "phase": "inspect",
        "handler": session_summary,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "INSPECT_TOOLS",
    "checkpoint",
    "diff_recipe",
    "get_dossier",
    "list_phase_tools",
    "session_summary",
    "view_recipe",
]
