"""Agent IR family C: synthesis and generation requests."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface

from compgen.ir.agent.attrs import CreativityPolicyAttr, EvaluatorKindAttr, SearchBudgetAttr


@irdl_op_definition
class RequestRewriteOp(IRDLOperation):
    """Request a rewrite or transform family."""

    name = "agent.request_rewrite"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    family = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestGuardOp(IRDLOperation):
    """Request guard synthesis for a transform family."""

    name = "agent.request_guard"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    family = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestEqsatSeedOp(IRDLOperation):
    """Request creative eqsat expansion."""

    name = "agent.request_eqsat_seed"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    rule_categories = opt_prop_def(ArrayAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestBackendPlanOp(IRDLOperation):
    """Request backend selection or backend-specific planning."""

    name = "agent.request_backend_plan"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    backend_goal = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestAnalysisOp(IRDLOperation):
    """Request a derived analysis or search-direction analysis."""

    name = "agent.request_analysis"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    analysis_kind = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestSemanticsOp(IRDLOperation):
    """Request semantics generation for an op family."""

    name = "agent.request_semantics"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    op_type = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestRepairOp(IRDLOperation):
    """Request repair after critique or verification failure."""

    name = "agent.request_repair"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    target_ref = prop_def(SymbolRefAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    diagnosis = opt_prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RequestRuntimePolicyOp(IRDLOperation):
    """Request runtime policy generation."""

    name = "agent.request_runtime_policy"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    policy_kind = prop_def(StringAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)
    output_kind = prop_def(StringAttr)
    search_budget = opt_prop_def(SearchBudgetAttr)
    evaluator = prop_def(EvaluatorKindAttr)
    creativity_policy = opt_prop_def(CreativityPolicyAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


__all__ = [
    "RequestAnalysisOp",
    "RequestBackendPlanOp",
    "RequestEqsatSeedOp",
    "RequestGuardOp",
    "RequestRepairOp",
    "RequestRewriteOp",
    "RequestRuntimePolicyOp",
    "RequestSemanticsOp",
]
