"""Event Tensor IR custom attributes.

Defines ParametrizedAttribute types used by the Event Tensor dialect:
    - EventTensorTypeAttr: shape + counter dtype + scope of an event tensor.
    - EventCoordAttr: index expression into an event tensor.
    - SchedulingPolicyAttr: ``static`` or ``dynamic`` scheduling choice.

The Event Tensor abstraction (Jin et al., MLSys '26) represents fine-grained
synchronization between tile-level GPU tasks as a multi-dimensional array of
counter-based semaphores.  Element ``E[i]`` initially holds a wait count, is
decremented by ``notify`` ops, and unblocks dependent ``wait`` ops when it
reaches zero.
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


@irdl_attr_definition
class EventTensorTypeAttr(ParametrizedAttribute):
    """Shape + counter dtype + scope descriptor for an event tensor.

    Shape entries may be concrete positive ints or ``-1`` to mark a symbolic
    dimension whose extent is bound at runtime (e.g. dynamic batch ``B`` or
    sequence length).  Symbolic-dim names are recorded in ``dim_names`` and
    must be the same length as ``shape`` whenever ``-1`` appears.

    Valid scopes:
        ``workgroup`` -- intra-CTA, lowers to shared memory.
        ``device``    -- intra-GPU, lowers to a global int tensor with atomics.
        ``system``    -- multi-GPU, lowers to NVLink/peer-aware atomics.
    """

    name = "event.event_tensor_type"
    shape: ArrayAttr = param_def(ArrayAttr)
    dim_names: ArrayAttr = param_def(ArrayAttr)
    counter_dtype: StringAttr = param_def(StringAttr)
    scope: StringAttr = param_def(StringAttr)

    _VALID_SCOPES: ClassVar[frozenset[str]] = frozenset({"workgroup", "device", "system"})
    _VALID_COUNTER_DTYPES: ClassVar[frozenset[str]] = frozenset({"i32", "u32", "i64", "u64"})

    def __init__(
        self,
        shape: list[int] | ArrayAttr,
        dim_names: list[str] | ArrayAttr | None = None,
        counter_dtype: str | StringAttr = "i32",
        scope: str | StringAttr = "device",
    ) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr([IntegerAttr(d, IntegerType(64)) for d in shape])
        if dim_names is None:
            dim_names = ArrayAttr([StringAttr("") for _ in shape.data])
        elif isinstance(dim_names, list):
            dim_names = ArrayAttr([StringAttr(s) for s in dim_names])
        if isinstance(counter_dtype, str):
            counter_dtype = StringAttr(counter_dtype)
        if isinstance(scope, str):
            scope = StringAttr(scope)
        super().__init__(shape, dim_names, counter_dtype, scope)


@irdl_attr_definition
class EventCoordAttr(ParametrizedAttribute):
    """A coordinate (index expression) into an event tensor.

    ``event_ref`` names the EventTensorOp; ``indices`` is one entry per
    dimension; ``decrement`` is the amount notify subtracts (``1`` by default).
    Each index entry is a string holding an einsum-like expression that
    references task-grid coordinate variables (e.g. ``"i"``, ``"i*32+j"``).
    """

    name = "event.coord"
    event_ref: StringAttr = param_def(StringAttr)
    indices: ArrayAttr = param_def(ArrayAttr)  # ArrayAttr of StringAttr
    decrement: IntegerAttr = param_def(IntegerAttr)

    def __init__(
        self,
        event_ref: str | StringAttr,
        indices: list[str] | ArrayAttr,
        decrement: int | IntegerAttr = 1,
    ) -> None:
        if isinstance(event_ref, str):
            event_ref = StringAttr(event_ref)
        if isinstance(indices, list):
            indices = ArrayAttr([StringAttr(s) for s in indices])
        if isinstance(decrement, int):
            decrement = IntegerAttr(decrement, IntegerType(64))
        super().__init__(event_ref, indices, decrement)


@irdl_attr_definition
class SchedulingPolicyAttr(ParametrizedAttribute):
    """Scheduling policy choice for a megakernel graph.

    Valid policies:
        ``static``  -- precomputed per-SM task queues (Algorithm 1, ETC paper).
        ``dynamic`` -- on-GPU push/pop scheduler (Algorithm 2).
    """

    name = "event.scheduling_policy"
    policy: StringAttr = param_def(StringAttr)

    _VALID: ClassVar[frozenset[str]] = frozenset({"static", "dynamic"})

    def __init__(self, policy: str | StringAttr) -> None:
        if isinstance(policy, str):
            policy = StringAttr(policy)
        super().__init__(policy)


__all__ = [
    "EventCoordAttr",
    "EventTensorTypeAttr",
    "SchedulingPolicyAttr",
]
