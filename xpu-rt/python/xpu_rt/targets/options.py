"""Base compilation options for all CompGen targets.

Inspired by Hexagon-MLIR's ``HexagonOptions`` frozen dataclass pattern.
Every target extends ``TargetOptions`` with hardware-specific fields.
Options control which compilation passes run and with what parameters.

The base class defines the universal options that apply to any target.
Target-specific subclasses add hardware-specific knobs (tile sizes,
memory sizes, vectorization widths, etc.).

Options are propagated to backend stages as a frozen dataclass.
For C++ backends, they can be serialized to a string dict via
``to_string_dict()``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetOptions:
    """Base compilation options for any CompGen target.

    Subclass this for target-specific options. The base class covers
    options that apply universally across all hardware targets.

    Attributes:
        target_name: Target identifier (e.g., ``"npu_v1"``, ``"cuda_sm90"``).
        emit_format: Output format (``"python"``, ``"mlir"``, ``"asm"``, ``"binary"``).
        enable_tiling: Tile operations to fit target geometry.
        enable_fusion: Fuse compatible adjacent operations.
        enable_vectorization: Use target vector instructions.
        enable_profiling: Insert profiling instrumentation.
        tile_sizes: Named tile dimensions (e.g., ``{"M": 32, "N": 32, "K": 32}``).
        scratchpad_bytes: On-chip fast memory size (0 = no scratchpad).
        num_threads: Number of execution threads / cores.
        optimization_level: 0=none, 1=basic, 2=aggressive.
    """

    target_name: str = ""
    emit_format: str = "python"
    enable_tiling: bool = True
    enable_fusion: bool = True
    enable_vectorization: bool = True
    enable_profiling: bool = False
    tile_sizes: dict[str, int] = field(default_factory=dict)
    scratchpad_bytes: int = 0
    num_threads: int = 1
    optimization_level: int = 1

    def to_string_dict(self) -> dict[str, str]:
        """Serialize all options to a string dict for C++ backend consumption.

        Matches Hexagon's pattern of stringifying options for the pybind11 bridge.
        """
        return {str(k): str(v) for k, v in dataclasses.asdict(self).items()}

    def with_overrides(self, **kwargs: Any) -> TargetOptions:
        """Create a new options instance with selected fields overridden."""
        return dataclasses.replace(self, **kwargs)


__all__ = ["TargetOptions"]
