"""Lower Agent IR into schedulable metadata artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr

from compgen.ir.agent.ops_claim import (
    AcceptedByOp,
    ClaimOp,
    DependsOnOp,
    ExpectedProofOp,
    RefutedByOp,
    SupportsOp,
)
from compgen.ir.agent.ops_critique import CompareCandidatesOp, CritiqueOp, ReviseOp
from compgen.ir.agent.ops_frontier import AlternativeOp, CommitOp, DeferOp, FrontierOp, PruneOp
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

_REQUEST_TYPES = (
    RequestRewriteOp,
    RequestGuardOp,
    RequestEqsatSeedOp,
    RequestBackendPlanOp,
    RequestAnalysisOp,
    RequestSemanticsOp,
    RequestRepairOp,
    RequestRuntimePolicyOp,
)


@dataclass(frozen=True)
class AgentLoweringOutput:
    request_jobs: list[dict[str, Any]] = field(default_factory=list)
    claim_records: list[dict[str, Any]] = field(default_factory=list)
    frontier_states: list[dict[str, Any]] = field(default_factory=list)
    critique_records: list[dict[str, Any]] = field(default_factory=list)
    memory_records: list[dict[str, Any]] = field(default_factory=list)
    protocol_records: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def lower_agent(module: ModuleOp) -> AgentLoweringOutput:
    request_jobs: list[dict[str, Any]] = []
    claim_records: list[dict[str, Any]] = []
    frontier_states: list[dict[str, Any]] = []
    critique_records: list[dict[str, Any]] = []
    memory_records: list[dict[str, Any]] = []
    protocol_records: list[dict[str, Any]] = []

    for op in module.body.block.ops:
        if isinstance(op, _REQUEST_TYPES):
            entry = {
                "request_type": op.name,
                "sym_name": op.sym_name.data,
                "scope_ref": _sym(op.scope_ref),
                "evidence_set_ref": _sym(op.evidence_set_ref),
                "output_kind": op.output_kind.data,
            }
            if hasattr(op, "family"):
                entry["family"] = getattr(op, "family").data
            if hasattr(op, "backend_goal"):
                entry["backend_goal"] = getattr(op, "backend_goal").data
            if hasattr(op, "analysis_kind"):
                entry["analysis_kind"] = getattr(op, "analysis_kind").data
            if hasattr(op, "op_type"):
                entry["op_type"] = getattr(op, "op_type").data
            if hasattr(op, "policy_kind"):
                entry["policy_kind"] = getattr(op, "policy_kind").data
            if hasattr(op, "target_ref"):
                entry["target_ref"] = _sym(getattr(op, "target_ref"))
            if hasattr(op, "rule_categories") and getattr(op, "rule_categories") is not None:
                entry["rule_categories"] = _array_to_python(getattr(op, "rule_categories"))
            request_jobs.append(entry)
            continue

        if isinstance(op, ClaimOp):
            claim_records.append(
                {
                    "kind": op.kind.data,
                    "sym_name": op.sym_name.data,
                    "scope_ref": _sym(op.scope_ref),
                    "text": op.text.data,
                    "confidence_milli": (op.confidence.value_milli.value.data if op.confidence is not None else None),
                }
            )
            continue
        if isinstance(op, SupportsOp):
            claim_records.append(
                {
                    "relation": "supports",
                    "claim_ref": _sym(op.claim_ref),
                    "evidence_refs": _array_to_python(op.evidence_refs),
                }
            )
            continue
        if isinstance(op, DependsOnOp):
            claim_records.append(
                {
                    "relation": "depends_on",
                    "claim_ref": _sym(op.claim_ref),
                    "dependency_refs": _array_to_python(op.dependency_refs),
                }
            )
            continue
        if isinstance(op, ExpectedProofOp):
            claim_records.append(
                {
                    "relation": "expected_proof",
                    "claim_ref": _sym(op.claim_ref),
                    "evaluator": op.evaluator.kind.data,
                }
            )
            continue
        if isinstance(op, RefutedByOp):
            claim_records.append(
                {
                    "relation": "refuted_by",
                    "claim_ref": _sym(op.claim_ref),
                    "evidence_ref": _sym(op.evidence_ref),
                }
            )
            continue
        if isinstance(op, AcceptedByOp):
            claim_records.append(
                {
                    "relation": "accepted_by",
                    "claim_ref": _sym(op.claim_ref),
                    "evidence_ref": _sym(op.evidence_ref),
                }
            )
            continue

        if isinstance(op, FrontierOp):
            frontier_states.append(
                {
                    "kind": "frontier",
                    "sym_name": op.sym_name.data,
                    "scope_ref": _sym(op.scope_ref),
                    "objective": op.objective.data,
                }
            )
            continue
        if isinstance(op, AlternativeOp):
            frontier_states.append(
                {
                    "kind": "alternative",
                    "frontier_ref": _sym(op.frontier_ref),
                    "target_ref": _sym(op.target_ref),
                    "target_kind": op.target_kind.data,
                }
            )
            continue
        if isinstance(op, DeferOp):
            frontier_states.append(
                {
                    "kind": "defer",
                    "frontier_ref": _sym(op.frontier_ref),
                    "reason": op.reason.data if op.reason is not None else "",
                }
            )
            continue
        if isinstance(op, PruneOp):
            frontier_states.append(
                {
                    "kind": "prune",
                    "frontier_ref": _sym(op.frontier_ref),
                    "target_ref": _sym(op.target_ref),
                    "reason": op.reason.data if op.reason is not None else "",
                }
            )
            continue
        if isinstance(op, CommitOp):
            frontier_states.append(
                {
                    "kind": "commit",
                    "frontier_ref": _sym(op.frontier_ref),
                    "selected_ref": _sym(op.selected_ref),
                    "evidence_set_ref": _sym(op.evidence_set_ref),
                }
            )
            continue

        if isinstance(op, CritiqueOp):
            critique_records.append(
                {
                    "kind": "critique",
                    "sym_name": op.sym_name.data,
                    "target_ref": _sym(op.target_ref),
                    "reason": op.reason.data,
                    "severity": op.severity.data,
                }
            )
            continue
        if isinstance(op, CompareCandidatesOp):
            critique_records.append(
                {
                    "kind": "compare",
                    "sym_name": op.sym_name.data,
                    "lhs_ref": _sym(op.lhs_ref),
                    "rhs_ref": _sym(op.rhs_ref),
                    "winner_ref": _sym(op.winner_ref) if op.winner_ref is not None else "",
                }
            )
            continue
        if isinstance(op, ReviseOp):
            critique_records.append(
                {
                    "kind": "revise",
                    "sym_name": op.sym_name.data,
                    "target_ref": _sym(op.target_ref),
                    "critique_ref": _sym(op.critique_ref),
                    "request_ref": _sym(op.request_ref) if op.request_ref is not None else "",
                }
            )
            continue

        if isinstance(op, MemoryPatternOp):
            memory_records.append(
                {
                    "kind": "pattern",
                    "sym_name": op.sym_name.data,
                    "domain": op.domain.data,
                    "pattern": op.pattern.data,
                    "outcome": op.outcome.data,
                }
            )
            continue
        if isinstance(op, MemoryFailureOp):
            memory_records.append(
                {
                    "kind": "failure",
                    "sym_name": op.sym_name.data,
                    "domain": op.domain.data,
                    "failure_mode": op.failure_mode.data,
                    "response": op.response.data,
                }
            )
            continue
        if isinstance(op, MemoryPromptOp):
            memory_records.append(
                {
                    "kind": "prompt",
                    "sym_name": op.sym_name.data,
                    "domain": op.domain.data,
                    "prompt_key": op.prompt_key.data,
                    "outcome": op.outcome.data,
                }
            )
            continue
        if isinstance(op, MemoryGeneralizationOp):
            memory_records.append(
                {
                    "kind": "generalization",
                    "sym_name": op.sym_name.data,
                    "source_ref": _sym(op.source_ref),
                    "generalization": op.generalization.data,
                }
            )
            continue
        if isinstance(op, PromoteMemoryOp):
            memory_records.append(
                {
                    "kind": "promote",
                    "memory_ref": _sym(op.memory_ref),
                    "promotion_key": op.promotion_key.data,
                }
            )
            continue

        if isinstance(op, RoleOp):
            protocol_records.append(
                {
                    "kind": "role",
                    "sym_name": op.sym_name.data,
                    "role_kind": op.kind.data,
                }
            )
            continue
        if isinstance(op, DelegateOp):
            protocol_records.append(
                {
                    "kind": "delegate",
                    "role_ref": _sym(op.role_ref),
                    "assignee_ref": _sym(op.assignee_ref),
                    "request_ref": _sym(op.request_ref),
                }
            )
            continue
        if isinstance(op, RespondOp):
            protocol_records.append(
                {
                    "kind": "respond",
                    "role_ref": _sym(op.role_ref),
                    "request_ref": _sym(op.request_ref),
                    "response_ref": _sym(op.response_ref),
                }
            )
            continue
        if isinstance(op, AdjudicateOp):
            protocol_records.append(
                {
                    "kind": "adjudicate",
                    "role_ref": _sym(op.role_ref),
                    "frontier_ref": _sym(op.frontier_ref),
                    "selected_ref": _sym(op.selected_ref) if op.selected_ref is not None else "",
                }
            )
            continue

    return AgentLoweringOutput(
        request_jobs=request_jobs,
        claim_records=claim_records,
        frontier_states=frontier_states,
        critique_records=critique_records,
        memory_records=memory_records,
        protocol_records=protocol_records,
    )


def _sym(ref: SymbolRefAttr | None) -> str:
    if ref is None:
        return ""
    return ref.root_reference.data


def _array_to_python(attr: ArrayAttr) -> list[Any]:
    values: list[Any] = []
    for item in attr.data:
        if isinstance(item, StringAttr):
            values.append(item.data)
        elif isinstance(item, SymbolRefAttr):
            values.append(item.root_reference.data)
        else:
            values.append(str(item))
    return values


__all__ = ["AgentLoweringOutput", "lower_agent"]
