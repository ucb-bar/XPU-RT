"""Recipe IR dialect registration.

Registers all Recipe IR operations and attributes with xDSL.
The ``Recipe`` dialect object is used for parser/printer context
registration.
"""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.recipe.attrs import (
    CostAttr,
    DeviceRefAttr,
    EffectClassAttr,
    ProvenanceAttr,
    ShapeSummaryAttr,
)
from compgen.ir.recipe.ops_candidate import (
    BlackboxOp,
    FuseOp,
    InsertCopyBoundaryOp,
    LayoutNormalizeOp,
    LowerToAccelOp,
    MaterializeUkernelOp,
    PlaceOnDeviceOp,
    ReassociateOp,
    RequestExoKernelOp,
    RequestTritonKernelOp,
    SegmentBoundaryOp,
    SelectExoScheduleLibOp,
    TileOp,
    VectorizeOp,
)
from compgen.ir.recipe.ops_choice import (
    AlternativesOp,
    DeferChoiceOp,
    PromoteCandidateOp,
    RankOp,
    RequireEqsatOp,
    RequireSolverOp,
    SearchBudgetOp,
)
from compgen.ir.recipe.ops_fact import (
    BackendAvailableOp,
    BackendEligibleOp,
    CalibrationOp,
    ContiguousLayoutOp,
    ExportIssueOp,
    FusibleWithOp,
    GuardFailureOp,
    GraphBreakOp,
    KernelContractOp,
    LocalMemFitOp,
    QuantizationIntentOp,
    TileDivisibleOp,
    TransferCostOp,
    UnsupportedOperatorOp,
)
from compgen.ir.recipe.ops_provenance import (
    FeedbackOp,
    FromAgentOp,
    FromEqsatOp,
    FromTemplateOp,
    LineageOp,
    PromoteOp,
    RejectOp,
)
from compgen.ir.recipe.ops_scope import (
    AnchorOp,
    BindPayloadOp,
    RecipeGuardOp,
    RecipeRegionOp,
    SegmentOp,
)
from compgen.ir.recipe.ops_verify import (
    RequireCheckFileOp,
    RequireDiffTestOp,
    RequireLayoutInvariantOp,
    RequireMemoryBoundOp,
    RequireProfileBudgetOp,
    RequireTranslationValidationOp,
)

# All operations in the Recipe dialect, grouped by family
_SCOPE_OPS = [RecipeRegionOp, SegmentOp, AnchorOp, RecipeGuardOp, BindPayloadOp]

_FACT_OPS = [
    BackendAvailableOp,
    KernelContractOp,
    TransferCostOp,
    LocalMemFitOp,
    FusibleWithOp,
    CalibrationOp,
    ExportIssueOp,
    GraphBreakOp,
    UnsupportedOperatorOp,
    GuardFailureOp,
    QuantizationIntentOp,
    TileDivisibleOp,
    ContiguousLayoutOp,
    BackendEligibleOp,
]

_CANDIDATE_OPS = [
    TileOp,
    FuseOp,
    VectorizeOp,
    ReassociateOp,
    LayoutNormalizeOp,
    LowerToAccelOp,
    RequestTritonKernelOp,
    RequestExoKernelOp,
    MaterializeUkernelOp,
    PlaceOnDeviceOp,
    InsertCopyBoundaryOp,
    SegmentBoundaryOp,
    SelectExoScheduleLibOp,
    BlackboxOp,
]

_CHOICE_OPS = [
    AlternativesOp,
    RankOp,
    SearchBudgetOp,
    RequireEqsatOp,
    RequireSolverOp,
    DeferChoiceOp,
    PromoteCandidateOp,
]

_VERIFY_OPS = [
    RequireDiffTestOp,
    RequireTranslationValidationOp,
    RequireLayoutInvariantOp,
    RequireMemoryBoundOp,
    RequireCheckFileOp,
    RequireProfileBudgetOp,
]

_PROVENANCE_OPS = [
    FromAgentOp,
    FromEqsatOp,
    FromTemplateOp,
    FeedbackOp,
    RejectOp,
    PromoteOp,
    LineageOp,
]

ALL_OPS = (
    _SCOPE_OPS
    + _FACT_OPS
    + _CANDIDATE_OPS
    + _CHOICE_OPS
    + _VERIFY_OPS
    + _PROVENANCE_OPS
)

ALL_ATTRS = [
    ShapeSummaryAttr,
    EffectClassAttr,
    CostAttr,
    ProvenanceAttr,
    DeviceRefAttr,
]

Recipe = Dialect("recipe", ALL_OPS, ALL_ATTRS)
"""The Recipe IR dialect -- register with ``ctx.register_dialect("recipe", lambda: Recipe)``."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Recipe"]
