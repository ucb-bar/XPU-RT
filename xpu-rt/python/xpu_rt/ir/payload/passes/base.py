"""Base class for Payload-IR passes ported from IREE/XLA.

Every ported pass subclasses :class:`PayloadPass` and provides a
``run(module, **args) -> ModuleOp`` that applies the rewrite. The base
class knows how to turn the pass into a :class:`compgen.llm.registry.Tool`
so the LLM registry automatically picks it up.

Per `analysis/proposed_compgen_architecture.md` and
`user_perspective/reports/stage_b_second_wave_status.md`, the port is:

- Single registration call: pass declares `name`, `phase`,
  `wraps_pass`, `covers_families`, `autocomp_cost_impact`; the helper
  builds the :class:`Tool` with ``impl`` bound to ``run``.
- ``covers_families`` is informational metadata (per
  `feedback_lean_heavily_on_inductor.md` updated guidance, it is NOT a
  registration filter — it informs the cost model via
  :mod:`compgen.llm.registry.target_coverage`).
- Stubs set ``stub=True`` and implement ``run()`` as a no-op until
  ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from xdsl.dialects.builtin import ModuleOp

from compgen.llm.registry import (
    AutocompCostImpact,
    Registry,
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)


@dataclass
class PayloadPass:
    """Base class for a ported xDSL-level payload pass.

    Subclasses override :meth:`run` and populate the class-level
    metadata. :meth:`register` binds ``run`` as a :class:`Tool.impl`
    and registers it into the LLM registry.
    """

    # --- metadata (override in subclass) ---
    name: ClassVar[str] = ""
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = ""
    covers_families: ClassVar[frozenset[str]] = frozenset()
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = ""
    stub: ClassVar[bool] = True

    def tool_args(self) -> tuple[ToolArg, ...]:
        """Override to declare the pass's typed arguments. Default: region ref only."""
        return (
            ToolArg(
                name="region",
                dtype="region_ref",
                description="region to apply the pass to (empty = whole module)",
                required=False,
                default="",
            ),
        )

    def tool_result(self) -> ToolResult:
        return ToolResult(
            dtype="ModuleOp", description="rewritten module (cloned)"
        )

    # --- implementation (override in subclass) ---
    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:  # noqa: D401
        """Apply the pass and return the rewritten module.

        Default behaviour is a no-op (identity) — useful for scaffolded
        stubs. Real ports override.
        """
        return module

    # --- registry integration ---
    def _impl_wrapper(self, **kwargs: Any) -> dict[str, Any]:
        """Thin wrapper so the registry sees a dict-returning callable.

        The LLM registry contract is ``Tool.impl(**kwargs) -> dict``.
        We unwrap ``module`` from kwargs, run the pass, and return a
        typed result dict that mirrors the diff shape used everywhere
        else (``status``, ``module``, ``notes``).
        """
        module = kwargs.pop("module", None)
        if module is None:
            return {"status": "error", "reason": "missing required kwarg 'module'"}
        try:
            new_module = self.run(module, **kwargs)
        except Exception as e:   # noqa: BLE001
            return {
                "status": "error",
                "reason": f"{type(e).__name__}: {e}",
                "pass_name": self.name,
            }
        return {
            "status": "ok",
            "pass_name": self.name,
            "module": new_module,
            "stub": self.stub,
        }

    def as_tool(self) -> Tool:
        """Build the :class:`Tool` descriptor for this pass."""
        return Tool(
            name=self.name,
            phase=self.phase,
            kind="tool",
            wraps_pass=self.wraps_pass,
            autocomp_cost_impact=self.autocomp_cost_impact,
            args=self.tool_args(),
            result=self.tool_result(),
            description=self.description or f"ported pass: {self.wraps_pass}",
            impl=self._impl_wrapper,
            stub=self.stub,
            notes=f"covers_families={sorted(self.covers_families) or 'ALL'}",
        )

    def register(self, registry: Registry | None = None) -> None:
        """Register this pass as a Tool. Idempotent (skips if already present)."""
        reg = registry or get_registry()
        if reg.lookup_tool(self.name, phase=self.phase) is not None:
            return
        reg.register_tool(self.as_tool())


__all__ = ["PayloadPass"]
