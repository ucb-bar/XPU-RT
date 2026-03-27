"""Unknown op discovery and LLM-generated support.

Scans FX graphs for ops not in the decomposition table or pattern library.
Can use the LLM to generate decompositions for unknown ops and verify
them against the original PyTorch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.agent.patterns import PATTERN_LIBRARY
from compgen.ir.payload.decompositions import DECOMPOSITION_TABLE


@dataclass(frozen=True)
class UnknownOp:
    """An op found in the FX graph that we don't know how to handle."""

    target: str                      # e.g., "aten.special_op.default"
    shape: tuple[int, ...] | None    # output shape
    dtype: str
    count: int                       # how many times it appears
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    example_args: list[str] = field(default_factory=list)  # arg names


class OpDiscovery:
    """Discovers unknown ops in FX graphs."""

    def discover_unknown_ops(self, exported_program: Any) -> list[UnknownOp]:
        """Find all ops not in the decomposition table."""
        unknown_counts: dict[str, UnknownOp] = {}

        for node in exported_program.graph.nodes:
            if node.op != "call_function":
                continue

            target = str(node.target)

            # Known in decomposition table?
            if target in DECOMPOSITION_TABLE:
                continue

            # Known pattern target?
            known_targets: set[str] = set()
            for pat in PATTERN_LIBRARY.values():
                known_targets.update(pat.op_targets)
            if target in known_targets:
                continue

            # It's unknown
            val = node.meta.get("val")
            shape = tuple(val.shape) if hasattr(val, "shape") else None
            dtype = str(val.dtype).replace("torch.", "") if hasattr(val, "dtype") else "unknown"

            input_shapes = []
            arg_names = []
            for a in node.args:
                if hasattr(a, "name"):
                    arg_names.append(a.name)
                    av = a.meta.get("val") if hasattr(a, "meta") else None
                    if av is not None and hasattr(av, "shape"):
                        input_shapes.append(tuple(av.shape))

            if target in unknown_counts:
                existing = unknown_counts[target]
                unknown_counts[target] = UnknownOp(
                    target=target, shape=existing.shape, dtype=existing.dtype,
                    count=existing.count + 1, input_shapes=existing.input_shapes,
                    example_args=existing.example_args,
                )
            else:
                unknown_counts[target] = UnknownOp(
                    target=target, shape=shape, dtype=dtype, count=1,
                    input_shapes=input_shapes, example_args=arg_names,
                )

        return sorted(unknown_counts.values(), key=lambda u: -u.count)


__all__ = ["OpDiscovery", "UnknownOp"]
