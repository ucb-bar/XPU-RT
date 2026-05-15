"""Agent IR custom attributes."""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


@irdl_attr_definition
class ConfidenceAttr(ParametrizedAttribute):
    """Confidence in milli-units, keeping serialization deterministic."""

    name = "agent.confidence"
    value_milli: IntegerAttr = param_def(IntegerAttr)

    def __init__(self, value_milli: int | IntegerAttr) -> None:
        if isinstance(value_milli, int):
            value_milli = IntegerAttr(value_milli, IntegerType(64))
        super().__init__(value_milli)


@irdl_attr_definition
class FreshnessAttr(ParametrizedAttribute):
    """Freshness metadata for bound evidence."""

    name = "agent.freshness"
    epoch: IntegerAttr = param_def(IntegerAttr)
    state: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        epoch: int | IntegerAttr,
        state: str | StringAttr = "fresh",
    ) -> None:
        if isinstance(epoch, int):
            epoch = IntegerAttr(epoch, IntegerType(64))
        if isinstance(state, str):
            state = StringAttr(state)
        super().__init__(epoch, state)


@irdl_attr_definition
class SearchBudgetAttr(ParametrizedAttribute):
    """Search budget for a synthesis request."""

    name = "agent.search_budget"
    max_candidates: IntegerAttr = param_def(IntegerAttr)
    max_iterations: IntegerAttr = param_def(IntegerAttr)
    timeout_ms: IntegerAttr = param_def(IntegerAttr)

    def __init__(
        self,
        max_candidates: int | IntegerAttr,
        max_iterations: int | IntegerAttr,
        timeout_ms: int | IntegerAttr,
    ) -> None:
        if isinstance(max_candidates, int):
            max_candidates = IntegerAttr(max_candidates, IntegerType(64))
        if isinstance(max_iterations, int):
            max_iterations = IntegerAttr(max_iterations, IntegerType(64))
        if isinstance(timeout_ms, int):
            timeout_ms = IntegerAttr(timeout_ms, IntegerType(64))
        super().__init__(max_candidates, max_iterations, timeout_ms)


@irdl_attr_definition
class CreativityPolicyAttr(ParametrizedAttribute):
    """How much freedom the model gets for a request."""

    name = "agent.creativity_policy"
    mode: StringAttr = param_def(StringAttr)
    temperature_milli: IntegerAttr = param_def(IntegerAttr)

    def __init__(
        self,
        mode: str | StringAttr,
        temperature_milli: int | IntegerAttr = 200,
    ) -> None:
        if isinstance(mode, str):
            mode = StringAttr(mode)
        if isinstance(temperature_milli, int):
            temperature_milli = IntegerAttr(temperature_milli, IntegerType(64))
        super().__init__(mode, temperature_milli)


@irdl_attr_definition
class EvaluatorKindAttr(ParametrizedAttribute):
    """Evaluator expected to judge a request or claim."""

    name = "agent.evaluator_kind"
    kind: StringAttr = param_def(StringAttr)

    def __init__(self, kind: str | StringAttr) -> None:
        if isinstance(kind, str):
            kind = StringAttr(kind)
        super().__init__(kind)


__all__ = [
    "ConfidenceAttr",
    "CreativityPolicyAttr",
    "EvaluatorKindAttr",
    "FreshnessAttr",
    "SearchBudgetAttr",
]
