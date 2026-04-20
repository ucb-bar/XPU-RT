"""LLM tool & invent-slot registry.

The registry is the typed catalog of everything the LLM may invoke
during the compile pipeline, partitioned by phase per
``user_perspective/analysis/llm_control_boundaries.md``.

**Design stance** (see ``feedback_lean_heavily_on_inductor.md``):
the tools CompGen registers are strictly those inductor does NOT
already do for us: target-feature-driven fusion beyond inductor's
single-device heuristics, heterogeneous placement, target-aligned
layout for non-CUDA accelerators, library matching for NPU/DSP/RVV
paths, quantized-dequant fusion for specific accelerators, and the
LLM invent-slots.

**Three kinds of callable**:

- ``Tool``: LLM selects and calls (``raise_special_ops``,
  ``match_library_call``, ...). Implementation lives in
  ``compgen.ir.payload.passes`` (when ported) or
  ``compgen.llm.tools.*`` (wrappers around existing analyzers).
- ``InventSlot``: LLM proposes a novel plan; a verification gate
  accepts or rejects. Output is a typed Recipe-IR op from
  ``compgen.ir.recipe.ops_propose``.
- ``ObservabilityTool`` / ``VerificationTool``: read-only helpers
  exposed in every phase.

The registry is globally accessible via module-level getters so any
phase of the compile pipeline can enumerate what's available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Typed descriptors
# ---------------------------------------------------------------------------

ToolKind = Literal["tool", "observability", "verification"]
AutocompCostImpact = Literal["very_high", "high", "medium", "low", "indirect", "zero"]


@dataclass(frozen=True)
class ToolArg:
    """One typed argument of a tool or invent-slot input."""

    name: str
    dtype: str
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    default: Any = None


@dataclass(frozen=True)
class ToolResult:
    """Typed return descriptor."""

    dtype: str
    description: str


@dataclass(frozen=True)
class Tool:
    """LLM-callable tool (select mode).

    ``impl`` is the concrete Python callable bound at registration time.
    Until a pass is ported, ``impl`` may be a stub that echoes args;
    callers inspect ``impl`` to tell real from stub via ``tool.is_stub``.
    """

    name: str
    phase: int
    kind: ToolKind
    wraps_pass: str
    autocomp_cost_impact: AutocompCostImpact
    args: tuple[ToolArg, ...]
    result: ToolResult
    description: str
    impl: Callable[..., dict[str, Any]] | None = None
    notes: str = ""
    stub: bool = True

    @property
    def is_stub(self) -> bool:
        return self.stub or self.impl is None

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        """Call the tool with typed arguments. Stubs return a placeholder."""
        if self.impl is None:
            return {
                "status": "no_impl",
                "tool_name": self.name,
                "echoed_args": kwargs,
            }
        return self.impl(**kwargs)


@dataclass(frozen=True)
class InventSlot:
    """LLM invent-slot (invent mode).

    ``gate_impl`` receives (proposal, context) and returns a GateResult
    dict with ``status ∈ {accepted, rejected, deferred}``. ``baseline_seed``
    produces a deterministic default proposal the LLM can refine.
    """

    name: str
    phase: int
    input_schema: str
    output_op: str  # e.g. "recipe.propose_fusion"
    gate: str  # human-readable gate spec
    autocomp_cost_impact: AutocompCostImpact
    description: str
    baseline_seed: Callable[..., dict[str, Any]] | None = None
    gate_impl: Callable[..., dict[str, Any]] | None = None
    notes: str = ""
    stub: bool = True

    @property
    def is_stub(self) -> bool:
        return self.stub or self.gate_impl is None

    def propose_baseline(self, **kwargs: Any) -> dict[str, Any]:
        if self.baseline_seed is None:
            return {"candidates": [], "chosen": None, "seed_source": "no_seed"}
        return self.baseline_seed(**kwargs)

    def verify(self, proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
        if self.gate_impl is None:
            return {"status": "deferred", "details": {"reason": "no_gate_impl"}}
        return self.gate_impl(proposal, **ctx)


# ---------------------------------------------------------------------------
# Registry state
# ---------------------------------------------------------------------------


@dataclass
class PhaseRegistry:
    """Per-phase partition of the registry."""

    phase: int
    tools: dict[str, Tool] = field(default_factory=dict)
    invent_slots: dict[str, InventSlot] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        if tool.phase != self.phase:
            raise ValueError(f"Tool {tool.name!r} declares phase={tool.phase} but registering into phase={self.phase}")
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name!r} already registered in phase {self.phase}")
        self.tools[tool.name] = tool

    def register_invent_slot(self, slot: InventSlot) -> None:
        if slot.phase != self.phase:
            raise ValueError(
                f"InventSlot {slot.name!r} declares phase={slot.phase} but registering into phase={self.phase}"
            )
        if slot.name in self.invent_slots:
            raise ValueError(f"InventSlot {slot.name!r} already registered in phase {self.phase}")
        self.invent_slots[slot.name] = slot


# Canonical phases per proposed_compgen_architecture.md.
# Phase 0 (capture), 1 (DET normalization), 6 (verify), 7 (package) have no
# LLM surface. Phases 2-5 are the LLM-facing phases.
_LLM_PHASES: tuple[int, ...] = (2, 3, 4, 5)


class Registry:
    """Global LLM registry, partitioned by phase."""

    def __init__(self) -> None:
        self._phases: dict[int, PhaseRegistry] = {p: PhaseRegistry(phase=p) for p in _LLM_PHASES}

    def register_tool(self, tool: Tool) -> None:
        if tool.phase not in self._phases:
            raise ValueError(f"Tool {tool.name!r} phase={tool.phase} is not one of {_LLM_PHASES}")
        self._phases[tool.phase].register_tool(tool)

    def register_invent_slot(self, slot: InventSlot) -> None:
        if slot.phase not in self._phases:
            raise ValueError(f"InventSlot {slot.name!r} phase={slot.phase} is not one of {_LLM_PHASES}")
        self._phases[slot.phase].register_invent_slot(slot)

    def list_tools(self, phase: int | None = None) -> list[Tool]:
        if phase is not None:
            return list(self._phases[phase].tools.values())
        return [t for p in _LLM_PHASES for t in self._phases[p].tools.values()]

    def list_invent_slots(self, phase: int | None = None) -> list[InventSlot]:
        if phase is not None:
            return list(self._phases[phase].invent_slots.values())
        return [s for p in _LLM_PHASES for s in self._phases[p].invent_slots.values()]

    def lookup_tool(self, name: str, phase: int | None = None) -> Tool | None:
        if phase is not None:
            return self._phases[phase].tools.get(name)
        for p in _LLM_PHASES:
            if name in self._phases[p].tools:
                return self._phases[p].tools[name]
        return None

    def lookup_invent_slot(self, name: str, phase: int | None = None) -> InventSlot | None:
        if phase is not None:
            return self._phases[phase].invent_slots.get(name)
        for p in _LLM_PHASES:
            if name in self._phases[p].invent_slots:
                return self._phases[p].invent_slots[name]
        return None

    def clear(self) -> None:
        """Reset the registry (used by tests)."""
        self._phases = {p: PhaseRegistry(phase=p) for p in _LLM_PHASES}

    def counts(self) -> dict[int, dict[str, int]]:
        """Summary per phase: {phase: {tools, invent_slots}}."""
        return {
            p: {
                "tools": len(self._phases[p].tools),
                "invent_slots": len(self._phases[p].invent_slots),
            }
            for p in _LLM_PHASES
        }


_GLOBAL_REGISTRY: Registry | None = None
_LOCAL_EXTENSIONS_LOADED: bool = False


def get_registry() -> Registry:
    """Return the process-wide registry (lazy-initialized).

    On first access, user-authored extensions in ``~/.compgen/extensions/``
    (see :mod:`compgen.agent.extensions`) are given a chance to register
    their tools / invent slots. Extension loading never raises — a broken
    file is logged and skipped. Disabled when env var
    ``COMPGEN_DISABLE_LOCAL_EXTENSIONS=1`` is set.
    """
    global _GLOBAL_REGISTRY, _LOCAL_EXTENSIONS_LOADED
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = Registry()
    if not _LOCAL_EXTENSIONS_LOADED:
        _LOCAL_EXTENSIONS_LOADED = True  # set first so failure can't loop
        import os

        if not os.environ.get("COMPGEN_DISABLE_LOCAL_EXTENSIONS"):
            try:
                from compgen.agent.extensions.local_loader import load_local_extensions

                load_local_extensions(_GLOBAL_REGISTRY)
            except Exception:  # noqa: BLE001
                # Loader is designed to never raise, but belt-and-braces
                # here so a truly broken install never breaks registry
                # initialisation.
                pass
    return _GLOBAL_REGISTRY


def reset_registry_for_testing() -> None:
    """Drop the process-wide registry and re-run extension loading.

    Only intended for unit tests that need a clean slate. Clears the
    idempotence flag so the next ``get_registry()`` call rescans
    ``~/.compgen/extensions``.
    """
    global _GLOBAL_REGISTRY, _LOCAL_EXTENSIONS_LOADED
    _GLOBAL_REGISTRY = None
    _LOCAL_EXTENSIONS_LOADED = False


__all__ = [
    "AutocompCostImpact",
    "InventSlot",
    "PhaseRegistry",
    "Registry",
    "Tool",
    "ToolArg",
    "ToolKind",
    "ToolResult",
    "get_registry",
    "reset_registry_for_testing",
]
