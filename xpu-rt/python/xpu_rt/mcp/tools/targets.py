"""MCP tools for the target registry — Wave 1.13.

The agentic-compilation extensibility surface. Three tools:

- :func:`compgen_list_targets` — agent introspects what's available.
  Returns the registry's nested tree + per-target audit data.
- :func:`compgen_describe_target` — drill into one target's
  rationale, adapters, metadata.
- :func:`compgen_register_target` — register a custom target at
  session scope, without forking the source. The user's agent
  uses this to plug in cuda-tile, an experimental arch, or a
  full new vendor.

All three are vendor-agnostic. The registry itself doesn't
distinguish between in-tree, entry-point, or MCP-registered
targets — same TargetPackage shape for all three paths.

Per the user's framing: ideally everything is extendible through
MCP so users can hook up with the existing infra case-by-case.
These tools are that hook.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from compgen.mcp.session import SessionManager


def compgen_list_targets(sm: SessionManager | None = None, **_: Any) -> dict[str, Any]:
    """Return the registry's full discovery surface.

    The agent's "what's available?" query. Returns:

    ``{
        "tree": {"gpu": {"nvidia": ["blackwell", ...], ...}, ...},
        "targets": [
            {"target_id": "gpu.nvidia.blackwell", "rationale": "...",
             "registration_path": "in_tree", ...},
            ...
        ],
        "count": 4,
    }``

    Idempotent. Doesn't trigger entry-point discovery — the agent
    can pair with ``compgen_register_target`` to add new ones at
    session scope.

    The ``sm`` arg is part of the MCP-handler dispatch convention —
    the server passes a :class:`SessionManager` to every handler.
    Target ops don't need session state, so the arg is accepted but
    unused.
    """
    del sm  # session-independent; dispatch convention requires the arg
    from compgen.targets.registry import registry

    reg = registry()
    return {
        "tree": reg.tree(),
        "targets": [pkg.to_dict() for pkg in reg.all()],
        "count": len(reg.all()),
    }


def compgen_describe_target(
    sm: SessionManager | None = None,
    *,
    target_id: str,
) -> dict[str, Any]:
    """Audit-query payload for one target.

    Args:
        sm: MCP session manager (unused — accepted for dispatch
            convention; see :func:`compgen_list_targets`).
        target_id: ``"gpu.nvidia.blackwell"`` etc.

    Returns the same shape as
    :meth:`compgen.targets.registry.TargetPackage.to_dict` plus a
    ``status`` field. Unknown target IDs return
    ``{"status": "unknown", "target_id": ...}``; never raises.
    """
    del sm
    from compgen.targets.registry import registry

    d = registry().describe(target_id)
    if not d:
        return {"status": "unknown", "target_id": target_id}
    return {"status": "ok", **d}


def compgen_register_target(
    sm: SessionManager | None = None,
    *,
    target_class: str,
    vendor: str,
    arch: str = "",
    rationale: str = "",
    metadata: dict[str, Any] | None = None,
    probe_module: str | None = None,
    body_emitter_module: str | None = None,
    runtime_module: str | None = None,
    cost_model_module: str | None = None,
) -> dict[str, Any]:
    """Register a custom target at session scope.

    The agent supplies the four adapters as **dotted module paths**
    (e.g. ``"my_pkg.adapters.MyProbe"``). This tool imports each
    module, fetches the named symbol, instantiates with no args,
    and registers the resulting :class:`TargetPackage`.

    The dotted-path approach (rather than passing live callables)
    is the MCP-stdio-safe way to ship an adapter — JSON can carry
    a string but not a Python object. The user's adapters live in
    their own importable package; the agent just hands the registry
    the path.

    Three usage shapes:

    1. **Custom arch under existing vendor** — the agent registers
       just a body_emitter + cost_model for ``sm_130`` while
       inheriting NVIDIA's vendor-common probe + runtime.
    2. **New vendor under existing class** — the agent registers
       all four adapters for ``gpu.tenstorrent.gridx``.
    3. **New class entirely** — register all four under
       ``dataflow.cerebras.cs3``.

    Args:
        target_class: ``"gpu"``, ``"cpu"``, ``"tpu"``, or any new
            class name (registry doesn't constrain).
        vendor: vendor under that class.
        arch: arch under that vendor. Empty string registers a
            vendor-common entry.
        rationale: human-readable description for the audit query.
        metadata: free-form per-target data.
        probe_module: dotted path like ``"my_pkg.adapters.MyProbe"``;
            None to skip (e.g. arch inheriting from vendor-common).
        body_emitter_module: same shape.
        runtime_module: same shape.
        cost_model_module: same shape.

    Returns:
        ``{"status": "ok" | "import_failed", "target_id": str,
         "errors": list[str]}``. Never raises — failures land in
        ``status``.
    """
    del sm
    from compgen.targets.registry import register_target

    errors: list[str] = []

    def _load(path: str | None, label: str) -> Any:
        """Import ``path`` (e.g. ``my_pkg.adapters.MyProbe``) and
        return ``MyProbe()`` instantiated. None passes through."""
        if path is None:
            return None
        try:
            module_path, _, attr_name = path.rpartition(".")
            if not module_path:
                errors.append(f"{label}: bad dotted path {path!r}")
                return None
            mod = importlib.import_module(module_path)
            cls = getattr(mod, attr_name, None)
            if cls is None:
                errors.append(f"{label}: {path!r} not found in module")
                return None
            return cls()  # zero-arg construct
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    probe = _load(probe_module, "probe_module")
    body_emitter = _load(body_emitter_module, "body_emitter_module")
    runtime = _load(runtime_module, "runtime_module")
    cost_model = _load(cost_model_module, "cost_model_module")

    pkg = register_target(
        target_class=target_class,
        vendor=vendor,
        arch=arch,
        probe=probe,
        body_emitter=body_emitter,
        runtime=runtime,
        cost_model=cost_model,
        rationale=rationale,
        registration_path="mcp",
        metadata=metadata or {},
    )

    return {
        "status": "ok" if not errors else "import_failed",
        "target_id": pkg.target_id,
        "errors": errors,
    }


# Tool descriptors for the MCP server's registry. Same shape as
# the entries in `compgen.mcp.tools.compile.COMPILE_TOOLS`.
TARGET_TOOLS = [
    {
        "name": "compgen_list_targets",
        "description": (
            "List every registered target package as a nested tree "
            "and a flat audit table. The agent uses this to "
            "introspect what compile targets are available without "
            "knowing the layout in advance."
        ),
        "phase": "inspect",
        "handler": compgen_list_targets,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compgen_describe_target",
        "description": (
            "Return the full audit-query payload for one registered "
            "target — rationale, adapters, metadata, registration "
            "path. Used by the agent's 'why was X picked?' / 'what "
            "does X do?' queries."
        ),
        "phase": "inspect",
        "handler": compgen_describe_target,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": ("Dotted target identifier — 'gpu.nvidia.blackwell', 'cpu.x86.avx512', etc."),
                },
            },
            "required": ["target_id"],
        },
    },
    {
        "name": "compgen_register_target",
        "description": (
            "Register a custom target at session scope without "
            "editing source. The agent supplies the four adapters "
            "(probe / body_emitter / runtime / cost_model) as "
            "dotted module paths. Three shapes: new arch under "
            "existing vendor, new vendor under existing class, or "
            "new class entirely. After registration, the matcher "
            "and dispatch path treat the custom target identically "
            "to in-tree ones."
        ),
        "phase": "lifecycle",
        "handler": compgen_register_target,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_class": {"type": "string"},
                "vendor": {"type": "string"},
                "arch": {"type": "string", "default": ""},
                "rationale": {"type": "string", "default": ""},
                "metadata": {"type": "object", "default": {}},
                "probe_module": {"type": ["string", "null"], "default": None},
                "body_emitter_module": {"type": ["string", "null"], "default": None},
                "runtime_module": {"type": ["string", "null"], "default": None},
                "cost_model_module": {"type": ["string", "null"], "default": None},
            },
            "required": ["target_class", "vendor"],
        },
    },
]
