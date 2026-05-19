"""Agent IR family H: roles and delegation."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface


@irdl_op_definition
class RoleOp(IRDLOperation):
    """Named agent role."""

    name = "agent.role"

    sym_name = prop_def(StringAttr)
    kind = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class DelegateOp(IRDLOperation):
    """Delegates a request from one role to another."""

    name = "agent.delegate"

    role_ref = prop_def(SymbolRefAttr)
    assignee_ref = prop_def(SymbolRefAttr)
    request_ref = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RespondOp(IRDLOperation):
    """Records a role responding to a delegated request."""

    name = "agent.respond"

    role_ref = prop_def(SymbolRefAttr)
    request_ref = prop_def(SymbolRefAttr)
    response_ref = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class AdjudicateOp(IRDLOperation):
    """Records a role adjudicating a frontier."""

    name = "agent.adjudicate"

    role_ref = prop_def(SymbolRefAttr)
    frontier_ref = prop_def(SymbolRefAttr)
    selected_ref = opt_prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


__all__ = [
    "AdjudicateOp",
    "DelegateOp",
    "RespondOp",
    "RoleOp",
]
