"""Prompt for device placement optimization."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementContext:
    """Context for placement prompt."""

    regions: list[dict[str, str | int | float]]
    num_devices: int
    device_names: list[str]
    device_compute: list[float]
    current_assignments: dict[str, int]
    transfer_cost_us: float


PLACEMENT_PROMPT = textwrap.dedent("""\
    You are optimizing device placement for a heterogeneous system.

    ## Devices ({num_devices}):
    {device_list}

    ## Operations to place:
    {regions}

    ## Current placement:
    {current_assignments}

    ## Cross-device transfer cost: {transfer_cost_us:.1f} us per copy

    ## Task
    Suggest a better placement. For each region, assign a device index.
    Minimize total latency considering:
    - Compute-heavy ops should go to the fastest device
    - Data-dependent ops should be co-located to avoid copies
    - Memory must fit on each device

    Respond as JSON: {{"region_id": device_index, ...}}
""")


def format_prompt(ctx: PlacementContext) -> str:
    """Render placement prompt."""
    device_list = "\n".join(
        f"  Device {i}: {name} ({compute:.1f} TFLOPS)"
        for i, (name, compute) in enumerate(zip(ctx.device_names, ctx.device_compute))
    )
    region_lines = "\n".join(
        f"  {r.get('region_id', '?')}: {r.get('op_type', '?')} "
        f"(FLOPs={r.get('flops', 0):,}, bytes={r.get('bytes', 0):,})"
        for r in ctx.regions[:20]
    )
    current = "\n".join(f"  {rid}: device_{dev}" for rid, dev in ctx.current_assignments.items())

    return PLACEMENT_PROMPT.format(
        num_devices=ctx.num_devices,
        device_list=device_list,
        regions=region_lines,
        current_assignments=current or "  (none)",
        transfer_cost_us=ctx.transfer_cost_us,
    )


def parse_response(response_text: str) -> dict[str, int]:
    """Parse placement response into region→device mapping."""
    try:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            raw = json.loads(text[start:end])
            return {str(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}
