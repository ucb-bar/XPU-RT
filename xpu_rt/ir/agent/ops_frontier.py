"""Agent IR family E: open frontier and commitment."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface


@irdl_op_definition
class FrontierOp(IRDLOperation):
    """Named frontier for a scope."""

    name = "agent.frontier"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    objective = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class AlternativeOp(IRDLOperation):
    """Candidate alternative in a frontier."""

    name = "agent.alternative"

    frontier_ref = prop_def(SymbolRefAttr)
    target_ref = prop_def(SymbolRefAttr)
    target_kind = prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class DeferOp(IRDLOperation):
    """Defers a frontier decision."""

    name = "agent.defer"

    frontier_ref = prop_def(SymbolRefAttr)
    reason = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class PruneOp(IRDLOperation):
    """Prunes an alternative from a frontier."""

    name = "agent.prune"

    frontier_ref = prop_def(SymbolRefAttr)
    target_ref = prop_def(SymbolRefAttr)
    reason = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class CommitOp(IRDLOperation):
    """Commits a selected alternative once judged."""

    name = "agent.commit"

    frontier_ref = prop_def(SymbolRefAttr)
    selected_ref = prop_def(SymbolRefAttr)
    evidence_set_ref = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


__all__ = [
    "AlternativeOp",
    "CommitOp",
    "DeferOp",
    "FrontierOp",
    "PruneOp",
]
