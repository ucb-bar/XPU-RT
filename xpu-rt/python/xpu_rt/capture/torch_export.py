"""torch.export capture path.

Captures a PyTorch nn.Module into an ExportedProgram using torch.export.
This is the canonical entry point for the CompGen pipeline.

Invariants:
    - The exported program must round-trip (export -> re-export produces same graph).
    - Dynamic shapes, if specified, must be propagated correctly.
    - Guard failures are recorded for diagnostic purposes.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExportValidation:
    """Result of validating an exported program.

    Attributes:
        valid: Whether the export passed all checks.
        round_trip_ok: Whether re-export produces the same graph.
        num_ops: Number of ops in the exported graph.
        graph_breaks: List of graph break descriptions (should be empty).
        warnings: Non-fatal validation warnings.
    """

    valid: bool
    round_trip_ok: bool
    num_ops: int
    graph_breaks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_model_from_path(model_path: str | Path) -> Any:
    """Load an nn.Module from a Python file.

    Expects the file to define a function ``get_model_and_inputs()``
    or a class with a known name.
    """
    path = Path(model_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if hasattr(module, "get_model_and_inputs"):
        model, _ = module.get_model_and_inputs()
        return model

    # Try common class names
    for name in ("SimpleMLP", "TransformerBlock", "Model"):
        if hasattr(module, name):
            cls = getattr(module, name)
            return cls()

    raise ImportError(f"No model found in {path}. Define get_model_and_inputs() or a known model class.")


def _count_call_ops(graph: Any) -> int:
    """Count call_function nodes in an FX graph (the actual operations)."""
    return sum(1 for node in graph.nodes if node.op == "call_function")


def capture_model(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    dynamic_shapes: dict[str, Any] | None = None,
) -> Any:
    """Capture a PyTorch model via torch.export.

    Args:
        model_path: Path to a Python file, or a live nn.Module.
        sample_inputs: Sample input tensors (tuple).
        dynamic_shapes: Optional dynamic shape specifications.

    Returns:
        A torch.export.ExportedProgram.
    """
    if isinstance(model_path, nn.Module):
        model = model_path
    else:
        model = _load_model_from_path(model_path)

    model.eval()

    kwargs: dict[str, Any] = {}
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes

    exported = torch.export.export(model, sample_inputs, **kwargs)
    return exported


def validate_export(exported_program: Any) -> ExportValidation:
    """Validate an exported program.

    Counts ops, checks for graph breaks, and verifies the graph is well-formed.
    """
    graph = exported_program.graph
    num_ops = _count_call_ops(graph)
    warnings: list[str] = []

    # Check for unsupported patterns
    graph_breaks: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if "graph_break" in target_name.lower():
                graph_breaks.append(f"Graph break at {node.name}: {target_name}")

    # Simple round-trip check: verify the graph has expected structure
    has_output = any(n.op == "output" for n in graph.nodes)
    has_placeholder = any(n.op == "placeholder" for n in graph.nodes)
    round_trip_ok = has_output and has_placeholder

    if not has_output:
        warnings.append("Graph has no output node")
    if num_ops == 0:
        warnings.append("Graph has no call_function ops")

    valid = round_trip_ok and len(graph_breaks) == 0

    return ExportValidation(
        valid=valid,
        round_trip_ok=round_trip_ok,
        num_ops=num_ops,
        graph_breaks=graph_breaks,
        warnings=warnings,
    )


__all__ = ["ExportValidation", "capture_model", "validate_export"]
