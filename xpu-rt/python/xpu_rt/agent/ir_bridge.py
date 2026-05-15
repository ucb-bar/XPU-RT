"""Bridge between env actions and Agent IR requests."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Operation

from compgen.ir.agent.attrs import CreativityPolicyAttr, EvaluatorKindAttr, SearchBudgetAttr
from compgen.ir.agent.ops_claim import ClaimOp
from compgen.ir.agent.ops_synthesis import (
    RequestAnalysisOp,
    RequestBackendPlanOp,
    RequestEqsatSeedOp,
    RequestRewriteOp,
    RequestRuntimePolicyOp,
    RequestSemanticsOp,
)


def _scope_symbol(region_id: str, known_scopes: set[str] | None = None) -> str:
    scope = f"scope_{region_id}" if region_id else "scope_session"
    if known_scopes is not None and scope not in known_scopes:
        return "scope_session"
    return scope


def _evidence_symbol(region_id: str, known_evidence_sets: set[str] | None = None) -> str:
    evidence = f"evidence_{region_id}" if region_id else "evidence_session"
    if known_evidence_sets is not None and evidence not in known_evidence_sets:
        return "evidence_session"
    return evidence


def _request_budget(max_candidates: int = 1, max_iterations: int = 1, timeout_ms: int = 0) -> SearchBudgetAttr:
    return SearchBudgetAttr(max_candidates, max_iterations, timeout_ms)


def _bounded_creativity() -> CreativityPolicyAttr:
    return CreativityPolicyAttr("bounded", 200)


def action_to_agent_ops(
    action: object,
    iteration: int = 0,
    *,
    known_scopes: set[str] | None = None,
    known_evidence_sets: set[str] | None = None,
) -> list[Operation]:
    """Convert an env action into Agent IR request/claim metadata."""
    from compgen.agent.env import (
        AnalyzeAction,
        AssignDeviceAction,
        ConfigureDispatchAction,
        ConfigureProfilingAction,
        EqSatAction,
        GeneratePassAction,
        GenerateRuntimeHooksAction,
        RequestSemanticsAction,
        RequestTransferAnalysisAction,
        RequestVerificationAction,
        SearchKernelAction,
        TileAction,
    )

    region_id = getattr(action, "region_id", "")
    scope_ref = SymbolRefAttr(_scope_symbol(region_id, known_scopes))
    evidence_set_ref = SymbolRefAttr(_evidence_symbol(region_id, known_evidence_sets))
    ops: list[Operation] = []

    if isinstance(action, TileAction) and region_id:
        ops.append(
            RequestRewriteOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_tile_{region_id}_{iteration}"),
                    "scope_ref": scope_ref,
                    "family": StringAttr("tile"),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("recipe_candidate"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr("translation_validation"),
                    "creativity_policy": _bounded_creativity(),
                }
            )
        )
        ops.append(
            ClaimOp.build(
                properties={
                    "sym_name": StringAttr(f"claim_tile_{region_id}_{iteration}"),
                    "scope_ref": scope_ref,
                    "kind": StringAttr("performance"),
                    "text": StringAttr(f"Tilings for {region_id} are worth exploring."),
                }
            )
        )
        return ops

    if isinstance(action, GeneratePassAction):
        ops.append(
            RequestRewriteOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_pass_{iteration}"),
                    "scope_ref": scope_ref,
                    "family": StringAttr("pass_generation"),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("rewrite_pattern"),
                    "search_budget": _request_budget(1, 1, 30_000),
                    "evaluator": EvaluatorKindAttr("structural_verifier"),
                    "creativity_policy": CreativityPolicyAttr("repairable", 300),
                }
            )
        )
        return ops

    if isinstance(action, EqSatAction):
        properties: dict[str, object] = {
            "sym_name": StringAttr(f"rq_eqsat_{region_id or 'session'}_{iteration}"),
            "scope_ref": scope_ref,
            "evidence_set_ref": evidence_set_ref,
            "output_kind": StringAttr("eqsat_job"),
            "search_budget": SearchBudgetAttr(8, action.max_iterations, 30_000),
            "evaluator": EvaluatorKindAttr("eqsat_cost_model"),
            "creativity_policy": CreativityPolicyAttr("frontier_expansion", 250),
        }
        if action.rule_categories:
            properties["rule_categories"] = ArrayAttr([StringAttr(cat) for cat in action.rule_categories])
        ops.append(RequestEqsatSeedOp.build(properties=properties))
        return ops

    if isinstance(action, AssignDeviceAction) and region_id:
        ops.append(
            RequestBackendPlanOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_backend_{region_id}_{iteration}"),
                    "scope_ref": scope_ref,
                    "backend_goal": StringAttr("device_placement"),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("plan_fragment"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr("solver_or_profile"),
                    "creativity_policy": _bounded_creativity(),
                }
            )
        )
        return ops

    if isinstance(action, SearchKernelAction):
        ops.append(
            RequestBackendPlanOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_kernel_{region_id or 'session'}_{iteration}"),
                    "scope_ref": scope_ref,
                    "backend_goal": StringAttr("kernel_search"),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("kernel_job"),
                    "search_budget": SearchBudgetAttr(action.budget, 1, 0),
                    "evaluator": EvaluatorKindAttr("kernel_validator"),
                    "creativity_policy": CreativityPolicyAttr("backend_specialization", 250),
                }
            )
        )
        return ops

    if isinstance(action, AnalyzeAction):
        ops.append(
            RequestAnalysisOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_analyze_{iteration}"),
                    "scope_ref": SymbolRefAttr("scope_session"),
                    "analysis_kind": StringAttr("model_analysis"),
                    "evidence_set_ref": SymbolRefAttr("evidence_session"),
                    "output_kind": StringAttr("analysis_report"),
                    "search_budget": _request_budget(1, 1, 5_000),
                    "evaluator": EvaluatorKindAttr("analysis_pipeline"),
                    "creativity_policy": CreativityPolicyAttr("hypothesis_generation", 300),
                }
            )
        )
        return ops

    if isinstance(action, RequestVerificationAction):
        ops.append(
            RequestAnalysisOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_verify_{region_id}_{iteration}"),
                    "scope_ref": scope_ref,
                    "analysis_kind": StringAttr(f"verification:{action.level}"),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("verification_obligation"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr(action.level),
                    "creativity_policy": CreativityPolicyAttr("bounded", 0),
                }
            )
        )
        return ops

    if isinstance(action, RequestTransferAnalysisAction) and region_id:
        ops.append(
            RequestAnalysisOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_transfer_{region_id}_{iteration}"),
                    "scope_ref": scope_ref,
                    "analysis_kind": StringAttr(action.analysis_type),
                    "evidence_set_ref": evidence_set_ref,
                    "output_kind": StringAttr("analysis_facts"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr("verified_transfer_analysis"),
                    "creativity_policy": CreativityPolicyAttr("analysis_synthesis", 250),
                }
            )
        )
        return ops

    if isinstance(action, RequestSemanticsAction):
        ops.append(
            RequestSemanticsOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_semantics_{action.op_type}_{iteration}"),
                    "scope_ref": SymbolRefAttr("scope_session"),
                    "op_type": StringAttr(action.op_type),
                    "evidence_set_ref": SymbolRefAttr("evidence_session"),
                    "output_kind": StringAttr("semantics_definition"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr("semantic_verifier"),
                    "creativity_policy": CreativityPolicyAttr("formalized_generation", 250),
                }
            )
        )
        return ops

    if isinstance(action, (ConfigureProfilingAction, ConfigureDispatchAction, GenerateRuntimeHooksAction)):
        policy_kind = {
            "configure_profiling": "profiling",
            "configure_dispatch": "dispatch",
            "generate_runtime_hooks": "hooks",
        }[action.action_type]
        ops.append(
            RequestRuntimePolicyOp.build(
                properties={
                    "sym_name": StringAttr(f"rq_runtime_{policy_kind}_{iteration}"),
                    "scope_ref": SymbolRefAttr("scope_session"),
                    "policy_kind": StringAttr(policy_kind),
                    "evidence_set_ref": SymbolRefAttr("evidence_session"),
                    "output_kind": StringAttr("runtime_artifact"),
                    "search_budget": _request_budget(),
                    "evaluator": EvaluatorKindAttr("runtime_checks"),
                    "creativity_policy": CreativityPolicyAttr("policy_generation", 250),
                }
            )
        )
        return ops

    return ops


__all__ = ["action_to_agent_ops"]
