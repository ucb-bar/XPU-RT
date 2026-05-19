"""Agent IR family F: critique and revision."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface


@irdl_op_definition
class CritiqueOp(IRDLOperation):
    """Typed critique against a candidate or claim."""

    name = "agent.critique"

    sym_name = prop_def(StringAttr)
    target_ref = prop_def(SymbolRefAttr)
    reason = prop_def(StringAttr)
    severity = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class CompareCandidatesOp(IRDLOperation):
    """Records a comparison between alternatives."""

    name = "agent.compare_candidates"

    sym_name = prop_def(StringAttr)
    lhs_ref = prop_def(SymbolRefAttr)
    rhs_ref = prop_def(SymbolRefAttr)
    winner_ref = opt_prop_def(SymbolRefAttr)
    reason = opt_prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class ReviseOp(IRDLOperation):
    """Links critique to a repair request."""

    name = "agent.revise"

    sym_name = prop_def(StringAttr)
    target_ref = prop_def(SymbolRefAttr)
    critique_ref = prop_def(SymbolRefAttr)
    request_ref = opt_prop_def(SymbolRefAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


__all__ = [
    "CompareCandidatesOp",
    "CritiqueOp",
    "ReviseOp",
]
