"""Layout analysis package for CompGen.

Provides layout planning, prepack analysis, and transpose profitability
classification for the compilation pipeline.
"""

from __future__ import annotations

from compgen.analysis.layout.planner import LayoutPlan, LayoutPlanner
from compgen.analysis.layout.prepack import PrepackCandidate, PrepackPlanner
from compgen.analysis.layout.transpose import (
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
