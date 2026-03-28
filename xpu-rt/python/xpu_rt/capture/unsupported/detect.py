"""Detection of unsupported operators at the export/import boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.capture.unsupported.introspect import ExampleTensorInfo


@dataclass(frozen=True)
class UnsupportedOperatorIssue:
    """Aggregated unsupported-op detection result for one target."""

    target: str
    stage: str
    reason: str
    count: int
    node_names: tuple[str, ...] = ()
    source_locations: tuple[str, ...] = ()
    example_inputs: tuple[ExampleTensorInfo, ...] = ()
    example_output: ExampleTensorInfo | None = None


def _tensor_info(value: Any) -> ExampleTensorInfo | None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    stride = tuple(value.stride()) if hasattr(value, "stride") else ()
    return ExampleTensorInfo(
        shape=tuple(int(dim) for dim in value.shape),
        dtype=str(value.dtype).replace("torch.", ""),
        stride=stride,
    )


def _source_location(node: Any) -> str:
    for candidate in (
        getattr(node, "stack_trace", None),
        node.meta.get("stack_trace") if hasattr(node, "meta") else None,
    ):
        if candidate:
            first_line = str(candidate).strip().splitlines()[0]
            return first_line.strip()
    return ""


def detect_unsupported_operators(
    exported_program: Any,
    *,
    supported_targets: set[str],
    explicit_targets: set[str] | None = None,
    stage: str = "payload_import",
) -> list[UnsupportedOperatorIssue]:
    """Detect operators without a registered lowering or explicit approval."""

    explicit = explicit_targets or set()
    grouped: dict[str, dict[str, Any]] = {}

    for node in exported_program.graph.nodes:
        if node.op != "call_function":
            continue

        target = str(node.target)
        if target in supported_targets or target in explicit:
            continue

        entry = grouped.setdefault(target, {
            "node_names": [],
            "source_locations": [],
            "example_inputs": (),
            "example_output": None,
        })
        entry["node_names"].append(node.name)
        location = _source_location(node)
        if location:
            entry["source_locations"].append(location)
        if not entry["example_inputs"]:
            tensors = []
            for arg in node.args:
                if hasattr(arg, "meta"):
                    val = arg.meta.get("val")
                    if (info := _tensor_info(val)) is not None:
                        tensors.append(info)
            entry["example_inputs"] = tuple(tensors)
        if entry["example_output"] is None:
            entry["example_output"] = _tensor_info(node.meta.get("val"))

    issues: list[UnsupportedOperatorIssue] = []
    for target, entry in sorted(grouped.items()):
        issues.append(UnsupportedOperatorIssue(
            target=target,
            stage=stage,
            reason="No registered Payload lowering or explicit blackbox approval",
            count=len(entry["node_names"]),
            node_names=tuple(entry["node_names"]),
            source_locations=tuple(entry["source_locations"]),
            example_inputs=entry["example_inputs"],
            example_output=entry["example_output"],
        ))
    return issues


__all__ = ["UnsupportedOperatorIssue", "detect_unsupported_operators"]
