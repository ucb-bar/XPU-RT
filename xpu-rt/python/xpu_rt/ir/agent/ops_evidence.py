"""Agent IR family B: evidence binding."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface

from compgen.ir.agent.attrs import FreshnessAttr


@irdl_op_definition
class BindFactOp(IRDLOperation):
    """Binds a surfaced deterministic fact into the agent context."""

    name = "agent.bind_fact"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    fact_name = prop_def(StringAttr)
    fact_payload = opt_prop_def(StringAttr)
    freshness = opt_prop_def(FreshnessAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class BindVerificationOp(IRDLOperation):
    """Binds a verifier result into the admissible evidence set."""

    name = "agent.bind_verification"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    verification_key = prop_def(StringAttr)
    status = opt_prop_def(StringAttr)
    freshness = opt_prop_def(FreshnessAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class BindProfileOp(IRDLOperation):
    """Binds a profile or measurement artifact into context."""

    name = "agent.bind_profile"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    profile_key = prop_def(StringAttr)
    metric_summary = opt_prop_def(StringAttr)
    freshness = opt_prop_def(FreshnessAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class BindAnalysisOp(IRDLOperation):
    """Binds a derived analysis result into context."""

    name = "agent.bind_analysis"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    analysis_key = prop_def(StringAttr)
    analysis_kind = prop_def(StringAttr)
    freshness = opt_prop_def(FreshnessAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class BindArtifactOp(IRDLOperation):
    """Binds an artifact path used as evidence."""

    name = "agent.bind_artifact"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    artifact_path = prop_def(StringAttr)
    artifact_kind = opt_prop_def(StringAttr)
    freshness = opt_prop_def(FreshnessAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class EvidenceSetOp(IRDLOperation):
    """Named set of admissible evidence references."""

    name = "agent.evidence_set"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    evidence_refs = prop_def(ArrayAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


__all__ = [
    "BindAnalysisOp",
    "BindArtifactOp",
    "BindFactOp",
    "BindProfileOp",
    "BindVerificationOp",
    "EvidenceSetOp",
]
