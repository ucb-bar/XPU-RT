"""CompGen analysis infrastructure.

Wave 2 modules:
  * ``dim_semantics``         — per-dim role tagging (parallel/reduce/broadcast)
  * ``transpose_propagation`` — transpose-chain detection + cancellation

Section 20 / :
  * ``checkpoints`` — multi-level analysis index (FX → runtime); pass
    cards' ``invalidates`` lists cross-link to ids registered here.
"""

from __future__ import annotations

from compgen.analysis.checkpoints import (
    ANALYSIS_LEVELS,
    KNOWN_SUMMARIES,
    AnalysisIndex,
    AnalysisLevel,
    AnalysisSummary,
    AnalysisSummaryError,
    KnownSummary,
    summary_id_for_path,
)
from compgen.analysis.dim_semantics import (
    DimRole,
    OpDimAnnotation,
    annotate_dim_roles,
    dim_roles_for_op,
)
from compgen.analysis.transpose_propagation import (
    TransposeChain,
    detect_transpose_chains,
    propose_transpose_cancellations,
)

__all__ = [
    # multi-level analysis
    "ANALYSIS_LEVELS",
    "KNOWN_SUMMARIES",
    "AnalysisIndex",
    "AnalysisLevel",
    "AnalysisSummary",
    "AnalysisSummaryError",
    "KnownSummary",
    "summary_id_for_path",
    # wave-2 analysis (pre-existing)
    "DimRole",
    "OpDimAnnotation",
    "TransposeChain",
    "annotate_dim_roles",
    "detect_transpose_chains",
    "dim_roles_for_op",
    "propose_transpose_cancellations",
]
