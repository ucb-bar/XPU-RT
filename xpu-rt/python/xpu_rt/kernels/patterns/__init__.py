"""Reusable kernel pattern catalog and FX graph pattern detection."""

from compgen.kernels.patterns.catalog import (
    KernelPattern,
    build_pattern_catalog,
    format_pattern_report,
)
from compgen.kernels.patterns.detection import (
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
