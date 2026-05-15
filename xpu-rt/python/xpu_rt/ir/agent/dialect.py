"""Agent IR dialect registration."""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.agent.attrs import (
    ConfidenceAttr,
    CreativityPolicyAttr,
    EvaluatorKindAttr,
    FreshnessAttr,
    SearchBudgetAttr,
)
from compgen.ir.agent.ops_claim import (
    AcceptedByOp,
    ClaimOp,
    DependsOnOp,
    ExpectedProofOp,
    RefutedByOp,
    SupportsOp,
)
from compgen.ir.agent.ops_critique import CompareCandidatesOp, CritiqueOp, ReviseOp
from compgen.ir.agent.ops_evidence import (
    BindAnalysisOp,
    BindArtifactOp,
    BindFactOp,
    BindProfileOp,
    BindVerificationOp,
    EvidenceSetOp,
)
from compgen.ir.agent.ops_frontier import AlternativeOp, CommitOp, DeferOp, FrontierOp, PruneOp
from compgen.ir.agent.ops_intent import (
    AgentAssumptionOp,
    AgentScopeOp,
    AgentSessionOp,
    AgentUncertaintyOp,
)
from compgen.ir.agent.ops_memory import (
    MemoryFailureOp,
    MemoryGeneralizationOp,
    MemoryPatternOp,
    MemoryPromptOp,
    PromoteMemoryOp,
)
from compgen.ir.agent.ops_protocol import AdjudicateOp, DelegateOp, RespondOp, RoleOp
from compgen.ir.agent.ops_synthesis import (
    RequestAnalysisOp,
    RequestBackendPlanOp,
    RequestEqsatSeedOp,
    RequestGuardOp,
    RequestRepairOp,
    RequestRewriteOp,
    RequestRuntimePolicyOp,
    RequestSemanticsOp,
)

_INTENT_OPS = [
    AgentSessionOp,
    AgentScopeOp,
    AgentAssumptionOp,
    AgentUncertaintyOp,
]

_EVIDENCE_OPS = [
    BindFactOp,
    BindVerificationOp,
    BindProfileOp,
    BindAnalysisOp,
    BindArtifactOp,
    EvidenceSetOp,
]

_SYNTHESIS_OPS = [
    RequestRewriteOp,
    RequestGuardOp,
    RequestEqsatSeedOp,
    RequestBackendPlanOp,
    RequestAnalysisOp,
    RequestSemanticsOp,
    RequestRepairOp,
    RequestRuntimePolicyOp,
]

_CLAIM_OPS = [
    ClaimOp,
    SupportsOp,
    DependsOnOp,
    ExpectedProofOp,
    RefutedByOp,
    AcceptedByOp,
]

_FRONTIER_OPS = [
    FrontierOp,
    AlternativeOp,
    DeferOp,
    PruneOp,
    CommitOp,
]

_CRITIQUE_OPS = [
    CritiqueOp,
    CompareCandidatesOp,
    ReviseOp,
]

_MEMORY_OPS = [
    MemoryPatternOp,
    MemoryFailureOp,
    MemoryPromptOp,
    MemoryGeneralizationOp,
    PromoteMemoryOp,
]

_PROTOCOL_OPS = [
    RoleOp,
    DelegateOp,
    RespondOp,
    AdjudicateOp,
]

ALL_OPS = (
    _INTENT_OPS
    + _EVIDENCE_OPS
    + _SYNTHESIS_OPS
    + _CLAIM_OPS
    + _FRONTIER_OPS
    + _CRITIQUE_OPS
    + _MEMORY_OPS
    + _PROTOCOL_OPS
)

ALL_ATTRS = [
    ConfidenceAttr,
    FreshnessAttr,
    SearchBudgetAttr,
    CreativityPolicyAttr,
    EvaluatorKindAttr,
]

Agent = Dialect("agent", ALL_OPS, ALL_ATTRS)

__all__ = ["ALL_ATTRS", "ALL_OPS", "Agent"]
