"""Agent IR family A: intent and scope."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class AgentSessionOp(IRDLOperation):
    """Top-level agent session for a compilation episode."""

    name = "agent.session"

    sym_name = prop_def(StringAttr)
    objective = prop_def(StringAttr)
    target = prop_def(StringAttr)
    search_mode = prop_def(StringAttr)
    constraints = opt_prop_def(ArrayAttr)

    traits = traits_def(SymbolOpInterface(), Pure())

    def verify_(self) -> None:
        valid = {"iterative", "frontier", "evolutionary", "runtime"}
        if self.search_mode.data not in valid:
            raise VerifyException(f"Invalid search_mode '{self.search_mode.data}', expected one of {valid}")


@irdl_op_definition
class AgentScopeOp(IRDLOperation):
    """Maps a deliberation scope to a stable recipe anchor."""

    name = "agent.scope"

    sym_name = prop_def(StringAttr)
    session_ref = prop_def(SymbolRefAttr)
    scope_ref = prop_def(SymbolRefAttr)
    scope_kind = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class AgentAssumptionOp(IRDLOperation):
    """Records an explicit assumption or default chosen by the agent."""

    name = "agent.assumption"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    text = prop_def(StringAttr)
    status = opt_prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class AgentUncertaintyOp(IRDLOperation):
    """Captures uncertainty before commitment."""

    name = "agent.uncertainty"

    sym_name = prop_def(StringAttr)
    scope_ref = prop_def(SymbolRefAttr)
    kind = prop_def(StringAttr)
    alternatives = opt_prop_def(ArrayAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


__all__ = [
    "AgentAssumptionOp",
    "AgentScopeOp",
    "AgentSessionOp",
    "AgentUncertaintyOp",
]
