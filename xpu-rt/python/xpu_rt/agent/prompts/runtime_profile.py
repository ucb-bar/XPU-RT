"""Prompt for profiling configuration decisions.

The agentic LLM uses this to decide:
    - Which profiling counters to enable for the current bottleneck.
    - What instrumentation level is appropriate.
    - Custom hook code for unknown/new hardware.
    - Analysis strategy (what to look at in the profile data).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProfileHookContext:
    """Context for profiling hook generation prompt.

    Attributes:
        target_name: Hardware target name.
        available_backends: Names of profiler backends available.
        available_counters: All available PMU counter names.
        current_bottlenecks: Detected bottlenecks from prior analysis.
        measured_vs_estimated: Drift between cost model and reality.
        current_level: Current instrumentation level name.
        tile_profiling_available: Whether tile-level is supported.
        runtime_env: Runtime environment (linux/zephyr/bare_metal).
    """

    target_name: str
    available_backends: list[str] = field(default_factory=list)
    available_counters: list[str] = field(default_factory=list)
    current_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    measured_vs_estimated: dict[str, float] = field(default_factory=dict)
    current_level: str = "NONE"
    tile_profiling_available: bool = False
    runtime_env: str = "linux_userspace"


@dataclass(frozen=True)
class ProfileHookConfig:
    """Parsed LLM response for profiling configuration.

    Attributes:
        instrumentation_level: Desired level (``"NONE"``, ``"OP_LEVEL"``,
            ``"TILE_LEVEL"``, ``"FULL"``).
        counters_to_enable: Which counters to activate.
        custom_hooks: Hook point → C code snippets.
        analysis_focus: What to analyze (``"latency"``, ``"memory"``,
            ``"compute"``, ``"dma_overlap"``).
        reasoning: LLM explanation.
    """

    instrumentation_level: str = "OP_LEVEL"
    counters_to_enable: list[str] = field(default_factory=list)
    custom_hooks: dict[str, str] = field(default_factory=dict)
    analysis_focus: str = "latency"
    reasoning: str = ""


PROFILE_HOOK_PROMPT = textwrap.dedent("""\
    You are a performance profiling advisor for a heterogeneous ML compiler.

    ## Target
    - Name: {target_name}
    - Runtime: {runtime_env}
    - Available profiler backends: {backends}
    - Available counters: {counters}
    - Tile-level profiling: {tile_available}
    - Current instrumentation: {current_level}

    ## Current Bottlenecks
    {bottlenecks}

    ## Cost Model Drift
    {drift}

    ## Task
    Configure profiling to diagnose the current bottlenecks.

    Choose:
    1. Instrumentation level: "NONE", "OP_LEVEL", "TILE_LEVEL", or "FULL"
    2. Which counters to enable (from the available list)
    3. Any custom hook code (C snippets for specific hook points)
    4. Analysis focus: "latency", "memory", "compute", or "dma_overlap"

    Respond as JSON:
    {{
        "instrumentation_level": "...",
        "counters_to_enable": ["..."],
        "custom_hooks": {{"hook_point": "C code"}},
        "analysis_focus": "...",
        "reasoning": "..."
    }}
""")


def format_prompt(ctx: ProfileHookContext) -> str:
    """Render profiling configuration prompt."""
    bottleneck_lines = (
        "\n".join(
            f"  - {b.get('region', '?')}: {b.get('kind', '?')} "
            f"(severity {b.get('severity', 0):.2f}) — {b.get('suggestion', '')}"
            for b in ctx.current_bottlenecks
        )
        or "  (none detected)"
    )

    drift_lines = (
        "\n".join(f"  {op}: measured={v:.1f}us vs estimated" for op, v in ctx.measured_vs_estimated.items())
        or "  (no drift data)"
    )

    return PROFILE_HOOK_PROMPT.format(
        target_name=ctx.target_name,
        runtime_env=ctx.runtime_env,
        backends=", ".join(ctx.available_backends) or "(none)",
        counters=", ".join(ctx.available_counters[:20]) or "(none)",
        tile_available="yes" if ctx.tile_profiling_available else "no",
        current_level=ctx.current_level,
        bottlenecks=bottleneck_lines,
        drift=drift_lines,
    )


def parse_response(response_text: str) -> ProfileHookConfig:
    """Parse LLM response into a ProfileHookConfig.

    Args:
        response_text: Raw LLM response (expected JSON).

    Returns:
        Parsed configuration.
    """
    # Extract JSON from response (handle markdown code blocks)
    text = response_text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ProfileHookConfig(reasoning=f"Failed to parse: {text[:200]}")

    return ProfileHookConfig(
        instrumentation_level=data.get("instrumentation_level", "OP_LEVEL"),
        counters_to_enable=data.get("counters_to_enable", []),
        custom_hooks=data.get("custom_hooks", {}),
        analysis_focus=data.get("analysis_focus", "latency"),
        reasoning=data.get("reasoning", ""),
    )


__all__ = [
    "ProfileHookConfig",
    "ProfileHookContext",
    "format_prompt",
    "parse_response",
]
