"""Reusable kernel pattern catalog and FX graph pattern detection."""

from xpu_rt.kernels.patterns.catalog import (
    KernelPattern,
    build_pattern_catalog,
    format_pattern_report,
)
from xpu_rt.kernels.patterns.detection import (
    DetectedPattern,
    detect_patterns_in_graphs,
)

__all__ = [
    "DetectedPattern",
    "KernelPattern",
    "build_pattern_catalog",
    "detect_patterns_in_graphs",
    "format_pattern_report",
]
