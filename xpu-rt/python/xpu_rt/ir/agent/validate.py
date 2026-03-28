"""Structural validation of Agent IR programs."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Operation
from xdsl.utils.exceptions import VerifyException

from compgen.ir.agent.ops_claim import ClaimOp, ExpectedProofOp
from compgen.ir.agent.ops_evidence import (
    BindAnalysisOp,
    BindProfileOp,
    BindVerificationOp,
    EvidenceSetOp,
)
from compgen.ir.agent.ops_frontier import CommitOp
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

log = structlog.get_logger()

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

_PROOF_REQUIRED_KINDS = {"legality", "correctness", "performance_and_legality"}


@dataclass(frozen=True)
class AgentValidationError:
    op_index: int
    op_type: str
    message: str


@dataclass(frozen=True)
class AgentValidationResult:
    valid: bool
    errors: list[AgentValidationError] = field(default_factory=list)


def _collect_defined_symbols(module: ModuleOp) -> dict[str, Operation]:
    symbols: dict[str, Operation] = {}
    for op in module.walk():
        if hasattr(op, "sym_name"):
            sym_name = getattr(op, "sym_name")
            if isinstance(sym_name, StringAttr):
                symbols[sym_name.data] = op
    return symbols


def _collect_external_symbols(module: ModuleOp | None) -> set[str]:
    if module is None:
        return set()
    return {
        getattr(op, "sym_name").data
        for op in module.walk()
        if hasattr(op, "sym_name") and isinstance(getattr(op, "sym_name"), StringAttr)
    }


def _collect_symbol_refs_from_attr(attr: Attribute) -> list[str]:
    refs: list[str] = []
    if isinstance(attr, SymbolRefAttr):
        refs.append(attr.root_reference.data)
    elif isinstance(attr, ArrayAttr):
        for item in attr.data:
            refs.extend(_collect_symbol_refs_from_attr(item))
    elif hasattr(attr, "parameters"):
        for param in getattr(attr, "parameters", ()):
            if isinstance(param, Attribute):
                refs.extend(_collect_symbol_refs_from_attr(param))
    return refs


def _collect_symbol_refs(op: Operation) -> list[str]:
    refs: list[str] = []
    if not hasattr(op, "properties"):
        return refs
    for attr in op.properties.values():
        refs.extend(_collect_symbol_refs_from_attr(attr))
    return refs


def validate_agent_module(
    module: ModuleOp,
    *,
    recipe_module: ModuleOp | None = None,
) -> AgentValidationResult:
    """Validate Agent IR with optional external recipe symbols."""
    errors: list[AgentValidationError] = []

    try:
        module.verify()
    except VerifyException as exc:
        errors.append(AgentValidationError(-1, "ModuleOp", f"xDSL verification failed: {exc}"))
    except Exception as exc:  # noqa: BLE001
        errors.append(AgentValidationError(-1, "ModuleOp", f"Unexpected verification error: {exc}"))

    symbols = _collect_defined_symbols(module)
    all_symbols = set(symbols) | _collect_external_symbols(recipe_module)

    for i, op in enumerate(module.walk()):
        if isinstance(op, ModuleOp):
            continue
        for ref in _collect_symbol_refs(op):
            if ref and ref not in all_symbols:
                errors.append(
                    AgentValidationError(i, type(op).__name__, f"Unresolved symbol reference: @{ref}")
                )

    expected_proofs = {
        op.claim_ref.root_reference.data
        for op in module.walk()
        if isinstance(op, ExpectedProofOp)
    }
    evidence_sets = {
        name: op
        for name, op in symbols.items()
        if isinstance(op, EvidenceSetOp)
    }

    for i, op in enumerate(module.walk()):
        if isinstance(op, ClaimOp) and op.kind.data in _PROOF_REQUIRED_KINDS:
            claim_sym = op.sym_name.data
            if claim_sym not in expected_proofs:
                errors.append(
                    AgentValidationError(
                        i,
                        "ClaimOp",
                        f"Claim @{claim_sym} of kind '{op.kind.data}' requires an agent.expected_proof",
                    )
                )

        if isinstance(op, _REQUEST_TYPES):
            evidence_ref = op.evidence_set_ref.root_reference.data
            if evidence_ref not in evidence_sets:
                errors.append(
                    AgentValidationError(
                        i,
                        type(op).__name__,
                        f"Request must reference an internal agent.evidence_set, got @{evidence_ref}",
                    )
                )

        if isinstance(op, CommitOp):
            evidence_ref = op.evidence_set_ref.root_reference.data
            evidence_set = evidence_sets.get(evidence_ref)
            if evidence_set is None:
                errors.append(
                    AgentValidationError(i, "CommitOp", f"Commit references missing evidence set @{evidence_ref}")
                )
                continue
            allowed = False
            for ref_attr in evidence_set.evidence_refs.data:
                if not isinstance(ref_attr, SymbolRefAttr):
                    continue
                bound = symbols.get(ref_attr.root_reference.data)
                if isinstance(bound, (BindVerificationOp, BindProfileOp, BindAnalysisOp)):
                    allowed = True
                    break
            if not allowed:
                errors.append(
                    AgentValidationError(
                        i,
                        "CommitOp",
                        "Commit requires an evidence set containing verification, profile, or analysis evidence",
                    )
                )

    valid = len(errors) == 0
    log.info("validate.agent_module", valid=valid, error_count=len(errors))
    return AgentValidationResult(valid=valid, errors=errors)


__all__ = [
    "AgentValidationError",
    "AgentValidationResult",
    "validate_agent_module",
]
