"""Layout analysis package for XPU-RT.

Provides layout planning, prepack analysis, and transpose profitability
classification for the compilation pipeline.
"""

from __future__ import annotations

from xpu_rt.analysis.layout.planner import LayoutPlan, LayoutPlanner
from xpu_rt.analysis.layout.prepack import PrepackCandidate, PrepackPlanner
from xpu_rt.analysis.layout.transpose import (
    TransposeClassification,
    TransposeProfitabilityAnalyzer,
)

__all__ = [
    "LayoutPlan",
    "LayoutPlanner",
    "PrepackCandidate",
    "PrepackPlanner",
    "TransposeClassification",
    "TransposeProfitabilityAnalyzer",
]
