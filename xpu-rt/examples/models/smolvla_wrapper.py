"""SmolVLA model wrapper for CompGen analysis.

Loads SmolVLA from Understanding-PI0, captures the FX graph via torch.compile,
and provides the captured graph for CompGen's NetworkAnalyzer.

Usage:
    python examples/models/smolvla_wrapper.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# Add Understanding-PI0 to path
PI0_ROOT = Path("/scratch2/agustin/merlin/third_party/Understanding-PI0")
if str(PI0_ROOT) not in sys.path:
    sys.path.insert(0, str(PI0_ROOT))


def load_smolvla(device: str = "cpu") -> tuple[Any, tuple[torch.Tensor, ...], int]:
    """Load SmolVLA and prepare inputs.

    Returns:
        (wrapper, flat_inputs, num_cams)
    """
    from understanding_pi0.smolvla_mx.loader import build_dummy_processed_inputs, load_smolvla_policy
    from understanding_pi0.smolvla_mx.wrappers import SmolVLAOneStepNoCacheWrapper, flatten_processed_inputs

    policy = load_smolvla_policy(model_id="lerobot/smolvla_base", device=device)
    processed = build_dummy_processed_inputs(policy, batch_size=1, prompt_len=8, device=device)
    flat_inputs = flatten_processed_inputs(processed)
    num_cams = (len(flat_inputs) - 5) // 2
    wrapper = SmolVLAOneStepNoCacheWrapper(policy, num_cams=num_cams)

    return wrapper, flat_inputs, num_cams


def capture_fx_graphs(
    wrapper: Any,
    flat_inputs: tuple[torch.Tensor, ...],
) -> list[torch.fx.GraphModule]:
    """Capture FX graphs via torch.compile custom backend.

    Returns all captured graph partitions (torch.compile may split
    the model into multiple subgraphs due to graph breaks).
    """
    import torch._dynamo as dynamo
    dynamo.reset()

    captured: list[torch.fx.GraphModule] = []

    def capture_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Any:
        captured.append(gm)
        return gm.forward

    compiled = torch.compile(wrapper, backend=capture_backend)
    with torch.no_grad():
        compiled(*flat_inputs)

    return captured


def get_smolvla_op_summary(graphs: list[torch.fx.GraphModule]) -> dict[str, int]:
    """Get op target counts across all captured graphs."""
    targets: dict[str, int] = {}
    for gm in graphs:
        for node in gm.graph.nodes:
            if node.op == "call_function":
                t = str(node.target)
                targets[t] = targets.get(t, 0) + 1
    return dict(sorted(targets.items(), key=lambda x: -x[1]))


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
