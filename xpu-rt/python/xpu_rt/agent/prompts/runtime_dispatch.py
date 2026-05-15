"""Prompt for dispatch strategy and transport configuration.

The agentic LLM uses this to decide:
    - Which dispatch strategy to use (pipeline, wavefront, etc.).
    - Which transport to use per inter-node link.
    - DMA tiling and double-buffering configuration.
    - Zephyr thread priorities and stack sizes.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DispatchContext:
    """Context for dispatch strategy prompt.

    Attributes:
        target_name: Hardware target name.
        topology_summary: Compact topology description.
        workload_shape: Workload characteristics.
        device_utilization: Per-device utilization percentages.
        num_cross_device_copies: Number of cross-device data transfers.
        total_transfer_bytes: Total bytes to transfer between devices.
        current_strategy: Currently active dispatch strategy.
        runtime_env: Runtime environment.
    """

    target_name: str
    topology_summary: dict[str, Any] = field(default_factory=dict)
    workload_shape: dict[str, Any] = field(default_factory=dict)
    device_utilization: dict[str, float] = field(default_factory=dict)
    num_cross_device_copies: int = 0
    total_transfer_bytes: int = 0
    current_strategy: str = "bulk_sync"
    runtime_env: str = "linux_userspace"


@dataclass(frozen=True)
class DispatchConfig:
    """Parsed LLM response for dispatch configuration.

    Attributes:
        strategy: Dispatch strategy name.
        transport_overrides: Link key → transport name overrides.
        thread_config: Thread name → priority overrides (Zephyr).
        double_buffer: Whether to use double-buffering for DMA.
        dma_tile_size: DMA transfer tile size in bytes.
        reasoning: LLM explanation.
    """

    strategy: str = "bulk_sync"
    transport_overrides: dict[str, str] = field(default_factory=dict)
    thread_config: dict[str, int] = field(default_factory=dict)
    double_buffer: bool = False
    dma_tile_size: int = 0
    reasoning: str = ""


DISPATCH_PROMPT = textwrap.dedent("""\
    You are a runtime dispatch advisor for a heterogeneous ML system.

    ## Target: {target_name}
    Runtime: {runtime_env}

    ## Topology
    {topology}

    ## Workload
    {workload}

    ## Device Utilization
    {utilization}

    ## Data Movement
    - Cross-device copies: {num_copies}
    - Total transfer: {transfer_bytes} bytes
    - Current strategy: {current_strategy}

    ## Task
    Choose the best dispatch strategy and transport configuration.

    Strategies:
    - "bulk_sync" — simple, all ops per level before advancing
    - "pipeline" — overlap compute and data movement across stages
    - "wavefront" — maximum parallelism, dispatch when dependencies met
    - "streaming" — continuous flow, best for steady-state serving

    Respond as JSON:
    {{
        "strategy": "...",
        "transport_overrides": {{"src->dst": "transport_name"}},
        "thread_config": {{"thread_name": priority}},
        "double_buffer": true/false,
        "dma_tile_size": 0,
        "reasoning": "..."
    }}
""")


def format_prompt(ctx: DispatchContext) -> str:
    """Render dispatch strategy prompt."""
    topo_str = json.dumps(ctx.topology_summary, indent=2) if ctx.topology_summary else "  (single device)"

    workload_str = json.dumps(ctx.workload_shape, indent=2) if ctx.workload_shape else "  (unknown)"

    util_lines = "\n".join(f"  {dev}: {util:.0f}%" for dev, util in ctx.device_utilization.items()) or "  (no data)"

    return DISPATCH_PROMPT.format(
        target_name=ctx.target_name,
        runtime_env=ctx.runtime_env,
        topology=topo_str,
        workload=workload_str,
        utilization=util_lines,
        num_copies=ctx.num_cross_device_copies,
        transfer_bytes=ctx.total_transfer_bytes,
        current_strategy=ctx.current_strategy,
    )


def parse_response(response_text: str) -> DispatchConfig:
    """Parse LLM response into a DispatchConfig.

    Args:
        response_text: Raw LLM response (expected JSON).

    Returns:
        Parsed configuration.
    """
    text = response_text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return DispatchConfig(reasoning=f"Failed to parse: {text[:200]}")

    return DispatchConfig(
        strategy=data.get("strategy", "bulk_sync"),
        transport_overrides=data.get("transport_overrides", {}),
        thread_config=data.get("thread_config", {}),
        double_buffer=data.get("double_buffer", False),
        dma_tile_size=data.get("dma_tile_size", 0),
        reasoning=data.get("reasoning", ""),
    )


__all__ = [
    "DispatchConfig",
    "DispatchContext",
    "format_prompt",
    "parse_response",
]
