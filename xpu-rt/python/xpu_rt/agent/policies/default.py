"""DeterministicDefaultPolicy — non-LLM baseline policy.

Produces a sensible default per-phase step list by inspecting the
registry. Intended uses:

- CI testing of the drive-loop mechanics without an LLM.
- Bootstrapping a compile run when no LLM client is configured.
- A deterministic baseline the real LLM policies can improve over.

Per-phase behaviour:

- Phase 2 (semantic global opt): call every real (``stub=False``) tool
  with its declared defaults, then trigger every Phase 2 invent-slot
  via ``use_baseline_seed=True``.
- Phase 3 (placement/layout): same — tools first, then invent-slots
  with baseline seeds.
- Phase 4 (kernel contract): tools only (no invent-slots by design).
- Phase 5 (runtime contract): real tools first, then invent-slots.

Stub tools are skipped by default (the LLM would still be expected to
consider them); set ``include_stubs=True`` at construction time to
include them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.llm.registry import Registry


@dataclass
class DeterministicDefaultPolicy:
    """Callable policy returning per-phase tool + invent-slot steps.

    Attributes:
        include_stubs: When True, call stub tools too (as identity
            no-ops). Useful for exercising the full registry surface
            in tests. Default False.
        invent_strategy: ``"baseline"`` (seed + gate) or ``"skip"``
            (no invent-slot calls). Default ``"baseline"``.
        default_args: Per-tool override of default arguments. Keyed by
            tool name. Merged on top of ToolArg.default values at call
            time.
    """

    include_stubs: bool = False
    invent_strategy: str = "baseline"
    default_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(
        self,
        phase: int,
        registry: Registry,
        context: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        steps: list[tuple[str, dict[str, Any]]] = []

        # 1. Real tools first (filtered by include_stubs flag)
        for tool in registry.list_tools(phase=phase):
            if tool.is_stub and not self.include_stubs:
                continue
            if tool.kind in ("observability", "verification"):
                # These are read-only/gate helpers — the LLM calls them
                # with real arguments on demand, not as default pipeline
                # steps.
                continue
            steps.append((tool.name, self._default_args_for(tool)))

        # 2. Invent-slots with baseline seeds
        if self.invent_strategy == "baseline":
            for slot in registry.list_invent_slots(phase=phase):
                if slot.is_stub and not self.include_stubs:
                    continue
                steps.append((slot.name, {"use_baseline_seed": True}))

        return steps

    def _default_args_for(self, tool: Any) -> dict[str, Any]:
        """Build default args for a tool: ToolArg defaults + overrides."""
        args: dict[str, Any] = {}
        for arg in tool.args:
            if arg.default is not None:
                args[arg.name] = arg.default
        # Per-tool overrides from the policy instance
        args.update(self.default_args.get(tool.name, {}))
        return args


__all__ = ["DeterministicDefaultPolicy"]
