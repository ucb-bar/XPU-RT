"""Hook code generator — generates C instrumentation from ProfilingSpec.

Given a ``ProfilingSpec``, generates C code that instruments the
HAL dispatch, DMA, and sync operations with trace points and
performance counter reads.

The agentic LLM extends this by:
    1. Calling ``GenerateRuntimeHooksAction`` to produce custom hook code.
    2. The custom hooks are stored in ``ProfilingSpec.custom_hooks``.
    3. This generator merges standard + custom hooks into the final output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.instrumentation import InstrumentationConfig, InstrumentationLevel
from compgen.targetgen.hardware_spec import ProfilingSpec

log = structlog.get_logger()


@dataclass(frozen=True)
class GeneratedHook:
    """A single generated instrumentation hook.

    Attributes:
        hook_point: Where this hook fires (``"pre_dispatch"``,
            ``"post_dispatch"``, ``"pre_dma"``, ``"post_dma"``,
            ``"pre_sync"``, ``"post_sync"``, ``"pre_alloc"``,
            ``"post_alloc"``).
        code: C code snippet to insert at the hook point.
        includes: Additional ``#include`` directives needed.
        guard: Preprocessor guard (e.g., ``"CG_TRACE_ENABLED"``).
    """

    hook_point: str
    code: str
    includes: list[str] = field(default_factory=list)
    guard: str = "CG_TRACE_ENABLED"


@dataclass
class HookCodegenResult:
    """Result of hook code generation.

    Attributes:
        hooks: Generated hooks keyed by hook point.
        header_code: Combined header/include code.
        source_code: Combined source code for a .c file.
        metadata: Generation metadata.
    """

    hooks: dict[str, GeneratedHook] = field(default_factory=dict)
    header_code: str = ""
    source_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HookCodeGenerator:
    """Generates C instrumentation hooks from a profiling spec.

    The generator produces standard hooks based on the instrumentation
    level, then merges any LLM-generated custom hooks from the
    ``ProfilingSpec.custom_hooks`` dict.

    Args:
        profiling_spec: Hardware profiling capabilities.
        instrumentation: Instrumentation configuration.
    """

    def __init__(
        self,
        profiling_spec: ProfilingSpec,
        instrumentation: InstrumentationConfig,
    ) -> None:
        self._spec = profiling_spec
        self._instr = instrumentation

    def generate(self) -> HookCodegenResult:
        """Generate all hooks based on spec and instrumentation level.

        Returns:
            A ``HookCodegenResult`` with all hooks and combined code.
        """
        hooks: dict[str, GeneratedHook] = {}

        if not self._instr.is_enabled:
            return HookCodegenResult(
                hooks=hooks,
                metadata={"level": "none", "hooks_generated": 0},
            )

        # Standard trace hooks
        hooks.update(self._generate_trace_hooks())

        # Performance counter hooks (tile-level and above)
        if self._instr.level >= InstrumentationLevel.TILE_LEVEL:
            hooks.update(self._generate_counter_hooks())

        # LLM-generated custom hooks (override standard ones if same hook point)
        hooks.update(self._merge_custom_hooks())

        # Assemble combined code
        header = self._assemble_header(hooks)
        source = self._assemble_source(hooks)

        return HookCodegenResult(
            hooks=hooks,
            header_code=header,
            source_code=source,
            metadata={
                "level": self._instr.level.name,
                "hooks_generated": len(hooks),
                "custom_hooks": len(self._spec.custom_hooks),
            },
        )

    def _generate_trace_hooks(self) -> dict[str, GeneratedHook]:
        """Generate standard CG_TRACE_* hooks."""
        hooks: dict[str, GeneratedHook] = {}

        hooks["pre_dispatch"] = GeneratedHook(
            hook_point="pre_dispatch",
            code='CG_TRACE_BEGIN("dispatch", kernel_name);',
            includes=["compgen/trace.h"],
        )
        hooks["post_dispatch"] = GeneratedHook(
            hook_point="post_dispatch",
            code="CG_TRACE_END();",
            includes=["compgen/trace.h"],
        )
        hooks["pre_dma"] = GeneratedHook(
            hook_point="pre_dma",
            code='CG_TRACE_BEGIN("dma", transfer_name);',
            includes=["compgen/trace.h"],
        )
        hooks["post_dma"] = GeneratedHook(
            hook_point="post_dma",
            code='CG_TRACE_END();\nCG_TRACE_COUNTER("dma_bytes", transfer_size);',
            includes=["compgen/trace.h"],
        )
        hooks["pre_sync"] = GeneratedHook(
            hook_point="pre_sync",
            code='CG_TRACE_BEGIN("sync", "device_sync");',
            includes=["compgen/trace.h"],
        )
        hooks["post_sync"] = GeneratedHook(
            hook_point="post_sync",
            code="CG_TRACE_END();",
            includes=["compgen/trace.h"],
        )

        return hooks

    def _generate_counter_hooks(self) -> dict[str, GeneratedHook]:
        """Generate performance counter hooks for tile-level profiling."""
        hooks: dict[str, GeneratedHook] = {}

        hooks["pre_tile"] = GeneratedHook(
            hook_point="pre_tile",
            code=(
                "cg_perf_start(perf_ctx);\n"
                'CG_TRACE_BEGIN("tile", tile_name);'
            ),
            includes=["compgen/trace.h", "compgen/perf_counters.h"],
        )
        hooks["post_tile"] = GeneratedHook(
            hook_point="post_tile",
            code=(
                "CG_TRACE_END();\n"
                "cg_perf_stop(perf_ctx);\n"
                "cg_perf_read(perf_ctx, tile_counters, num_counters);\n"
                "CG_TRACE_TILE(region_id, tile_idx, \"cycles\", tile_counters[0]);"
            ),
            includes=["compgen/trace.h", "compgen/perf_counters.h"],
        )

        return hooks

    def _merge_custom_hooks(self) -> dict[str, GeneratedHook]:
        """Convert ProfilingSpec.custom_hooks to GeneratedHook objects."""
        hooks: dict[str, GeneratedHook] = {}

        for hook_point, code in self._spec.custom_hooks.items():
            hooks[hook_point] = GeneratedHook(
                hook_point=hook_point,
                code=code,
                includes=["compgen/trace.h"],
                guard="CG_TRACE_ENABLED",
            )

        return hooks

    def _assemble_header(self, hooks: dict[str, GeneratedHook]) -> str:
        """Assemble combined header code from all hooks."""
        includes: set[str] = set()
        for hook in hooks.values():
            includes.update(hook.includes)

        lines = [
            "/* Auto-generated instrumentation hooks */",
            "/* Do not edit — regenerate via HookCodeGenerator */",
            "",
        ]

        for inc in sorted(includes):
            lines.append(f'#include "{inc}"')

        lines.append("")
        return "\n".join(lines)

    def _assemble_source(self, hooks: dict[str, GeneratedHook]) -> str:
        """Assemble combined source code with all hooks."""
        lines = [
            "/* Auto-generated instrumentation hooks */",
            "",
            "#ifdef CG_TRACE_ENABLED",
            "",
        ]

        for hook_point, hook in sorted(hooks.items()):
            lines.append(f"/* Hook: {hook_point} */")
            lines.append(f"// {hook.code}")
            lines.append("")

        lines.append("#endif /* CG_TRACE_ENABLED */")
        lines.append("")
        return "\n".join(lines)


__all__ = [
    "GeneratedHook",
    "HookCodeGenerator",
    "HookCodegenResult",
]
