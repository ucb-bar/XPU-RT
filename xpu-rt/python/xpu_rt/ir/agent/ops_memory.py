"""Agent IR family G: reusable learned memory."""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, traits_def
from xdsl.traits import Pure, SymbolOpInterface


@irdl_op_definition
class MemoryPatternOp(IRDLOperation):
    """Reusable successful pattern."""

    name = "agent.memory_pattern"

    sym_name = prop_def(StringAttr)
    domain = prop_def(StringAttr)
    pattern = prop_def(StringAttr)
    outcome = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class MemoryFailureOp(IRDLOperation):
    """Recorded failure mode with repair guidance."""

    name = "agent.memory_failure"

    sym_name = prop_def(StringAttr)
    domain = prop_def(StringAttr)
    failure_mode = prop_def(StringAttr)
    response = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class MemoryPromptOp(IRDLOperation):
    """Prompt pattern that worked or failed."""

    name = "agent.memory_prompt"

    sym_name = prop_def(StringAttr)
    domain = prop_def(StringAttr)
    prompt_key = prop_def(StringAttr)
    outcome = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class MemoryGeneralizationOp(IRDLOperation):
    """Generalized lesson derived from another memory item."""

    name = "agent.memory_generalization"

    sym_name = prop_def(StringAttr)
    source_ref = prop_def(SymbolRefAttr)
    generalization = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class PromoteMemoryOp(IRDLOperation):
    """Promotes a memory item to reusable shared knowledge."""

    name = "agent.promote_memory"

    memory_ref = prop_def(SymbolRefAttr)
    promotion_key = prop_def(StringAttr)

    traits = traits_def(Pure())


__all__ = [
    "MemoryFailureOp",
    "MemoryGeneralizationOp",
    "MemoryPatternOp",
    "MemoryPromptOp",
    "PromoteMemoryOp",
]
