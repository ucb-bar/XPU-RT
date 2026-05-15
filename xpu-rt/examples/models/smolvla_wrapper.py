"""Compatibility wrapper around the canonical SmolVLA model adapter.

This module intentionally re-exports the old example-level helpers so existing
tests and scripts continue to work while the real implementation lives under
``compgen.models``.
"""

from __future__ import annotations

from typing import Any

import torch

from compgen.capture import capture_dynamo_partitions
from compgen.models import get_graph_op_summary, load_smolvla_bundle


def load_smolvla(device: str = "cpu") -> tuple[Any, tuple[torch.Tensor, ...], int]:
    """Load SmolVLA and prepare inputs.

    Returns:
        (wrapper, flat_inputs, num_cams)
    """
    return load_smolvla_bundle(device=device)


def capture_fx_graphs(
    wrapper: Any,
    flat_inputs: tuple[torch.Tensor, ...],
) -> list[torch.fx.GraphModule]:
    """Capture FX graphs via TorchDynamo partition capture.

    Returns all captured graph partitions (torch.compile may split
    the model into multiple subgraphs due to graph breaks).
    """
    artifact = capture_dynamo_partitions(wrapper, flat_inputs)
    return list(artifact.graphs)


def get_smolvla_op_summary(graphs: list[torch.fx.GraphModule]) -> dict[str, int]:
    """Get op target counts across all captured graphs."""
    return get_graph_op_summary(graphs)


if __name__ == "__main__":
    print("Loading SmolVLA...")
    wrapper, flat_inputs, num_cams = load_smolvla()
    print(f"  Params: {sum(p.numel() for p in wrapper.parameters()):,}")
    print(f"  Inputs: {len(flat_inputs)} tensors, {num_cams} cameras")

    print("\nCapturing FX graphs via torch.compile...")
    graphs = capture_fx_graphs(wrapper, flat_inputs)
    print(f"  Captured {len(graphs)} graph partitions")

    summary = get_smolvla_op_summary(graphs)
    total = sum(summary.values())
    print(f"  Total ops: {total}")
    print(f"  Unique targets: {len(summary)}")
    print("\n  Top 15 ops:")
    for t, c in list(summary.items())[:15]:
        print(f"    {t}: {c}x")
