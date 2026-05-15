"""Agent IR family D: claims and proof expectations."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface

from compgen.ir.agent.attrs import ConfidenceAttr, EvaluatorKindAttr


@irdl_op_definition
class ClaimOp(IRDLOperation):
    """Typed agent claim about a scope."""

    name = "agent.claim"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    kind = prop_def(StringAttr)
    text = prop_def(StringAttr)
    confidence = opt_prop_def(ConfidenceAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class SupportsOp(IRDLOperation):
    """Evidence supporting a claim."""

    name = "agent.supports"

    claim_ref = prop_def(SymbolRefAttr)
    evidence_refs = prop_def(ArrayAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class DependsOnOp(IRDLOperation):
    """Other claims or requests a claim depends on."""

    name = "agent.depends_on"

    claim_ref = prop_def(SymbolRefAttr)
    dependency_refs = prop_def(ArrayAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class ExpectedProofOp(IRDLOperation):
    """Expected evaluator that must discharge a claim."""

    name = "agent.expected_proof"

    claim_ref = prop_def(SymbolRefAttr)
    evaluator = prop_def(EvaluatorKindAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RefutedByOp(IRDLOperation):
    """Evidence that refutes a claim."""

    name = "agent.refuted_by"

    claim_ref = prop_def(SymbolRefAttr)
    evidence_ref = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class AcceptedByOp(IRDLOperation):
    """Evidence that accepts or confirms a claim."""

    name = "agent.accepted_by"

    claim_ref = prop_def(SymbolRefAttr)
    evidence_ref = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


__all__ = [
    "AcceptedByOp",
    "ClaimOp",
    "DependsOnOp",
    "ExpectedProofOp",
    "RefutedByOp",
    "SupportsOp",
]
