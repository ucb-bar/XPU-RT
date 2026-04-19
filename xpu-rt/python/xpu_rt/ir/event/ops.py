"""Event Tensor IR operations.

Implements the IR layer of the Event Tensor abstraction (Jin et al., MLSys '26).
An event tensor is a multi-dimensional array of counter-based semaphores; tile
tasks ``notify`` them on completion and ``wait`` on them before reading
producer state.  This dialect supplies the seven ops needed to express the
paper's three language constructs (device function, event tensor, graph
function) plus its data-dependent dynamism extensions:

    - EventTensorOp     -- declare an event tensor with shape + wait_count.
    - NotifyOp          -- atomic decrement on one event coordinate.
    - WaitOp            -- spin-wait until an event coordinate reaches zero.
    - GraphOp           -- megakernel graph wrapper holding ``call_device``
                           regions with explicit in/out edges.
    - CallDeviceOp      -- launch a device function with task-grid + edges.
    - UpdateOp          -- data-dependent rewrite of event counters.
    - TriggerOp         -- runtime materialization of a variable number of
                           consumer tasks.
    - MaterializeViewOp -- runtime view materialization for symbolic-shape
                           Event Tensors (Fig. 4 of the paper).
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Region
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
)
from xdsl.traits import NoTerminator, Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.event.attrs import (
    EventCoordAttr,
    EventTensorTypeAttr,
    SchedulingPolicyAttr,
)
from compgen.ir.recipe.attrs import ProvenanceAttr


@irdl_op_definition
class EventTensorOp(IRDLOperation):
    """Declare an event tensor.

    Allocates a multi-dimensional counter array with the given shape and
    initial wait count.  Lowered to an ``i32``/``i64`` tensor in global
    memory whose elements are manipulated by ``NotifyOp`` and ``WaitOp``.
    """

    name = "event.event_tensor"

    sym_name = prop_def(StringAttr)
    event_type = prop_def(EventTensorTypeAttr)
    wait_count = prop_def(IntegerAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.wait_count.value.data < 0:
            raise VerifyException(
                f"event.event_tensor wait_count must be non-negative, "
                f"got {self.wait_count.value.data}"
            )
        scope = self.event_type.scope.data
        if scope not in EventTensorTypeAttr._VALID_SCOPES:
            raise VerifyException(
                f"event.event_tensor scope '{scope}' invalid, "
                f"expected one of {sorted(EventTensorTypeAttr._VALID_SCOPES)}"
            )
        ctype = self.event_type.counter_dtype.data
        if ctype not in EventTensorTypeAttr._VALID_COUNTER_DTYPES:
            raise VerifyException(
                f"event.event_tensor counter_dtype '{ctype}' invalid, "
                f"expected one of {sorted(EventTensorTypeAttr._VALID_COUNTER_DTYPES)}"
            )


@irdl_op_definition
class NotifyOp(IRDLOperation):
    """Atomic decrement on a single event-tensor coordinate.

    Lowers to ``atomicSub`` (CUDA) / ``tl.atomic_add(ptr, -k)`` (Triton).
    ``coord.decrement`` defaults to ``1``; values >1 represent grouped
    completion (a producer that satisfies multiple consumers at once).
    """

    name = "event.notify"

    coord = prop_def(EventCoordAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    def verify_(self) -> None:
        if self.coord.decrement.value.data <= 0:
            raise VerifyException(
                f"event.notify decrement must be positive, "
                f"got {self.coord.decrement.value.data}"
            )


@irdl_op_definition
class WaitOp(IRDLOperation):
    """Spin-wait until an event-tensor coordinate reaches zero.

    Lowers to a busy-wait loop:
        ``while atomic_load(E + idx) > 0: pass``
    on the device-scope codegen path.
    """

    name = "event.wait"

    coord = prop_def(EventCoordAttr)
    provenance = opt_prop_def(ProvenanceAttr)


@irdl_op_definition
class CallDeviceOp(IRDLOperation):
    """Launch a device function on a task grid.

    ``device_func`` is a SymbolRefAttr to the ``func.func`` op holding the
    per-tile task body.  ``task_shape`` is the multi-dimensional task grid
    (e.g. ``[n, 32]`` for ``n*32`` tasks).  ``in_edges`` / ``out_edges`` are
    lists of ``EventCoordAttr`` connecting this dispatch to producer/consumer
    Event Tensors.
    """

    name = "event.call_device"

    device_func = prop_def(SymbolRefAttr)
    task_shape = prop_def(ArrayAttr)  # ArrayAttr of IntegerAttr
    in_edges = opt_prop_def(ArrayAttr)  # ArrayAttr of EventCoordAttr
    out_edges = opt_prop_def(ArrayAttr)  # ArrayAttr of EventCoordAttr
    provenance = opt_prop_def(ProvenanceAttr)

    def verify_(self) -> None:
        for dim in self.task_shape.data:
            if isinstance(dim, IntegerAttr) and dim.value.data < 1 and dim.value.data != -1:
                raise VerifyException(
                    f"event.call_device task_shape entries must be >=1 or -1 "
                    f"(symbolic), got {dim.value.data}"
                )


@irdl_op_definition
class GraphOp(IRDLOperation):
    """Megakernel graph wrapper.

    Holds a single region of ``EventTensorOp`` declarations followed by
    ``CallDeviceOp`` dispatches.  The ``policy`` attribute records the
    chosen scheduling strategy (static/dynamic); the static-schedule pass
    rewrites the body in-place once a per-SM execution queue has been
    solved.
    """

    name = "event.graph"

    sym_name = prop_def(StringAttr)
    policy = prop_def(SchedulingPolicyAttr)
    sm_count = opt_prop_def(IntegerAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    body = region_def()

    traits = traits_def(NoTerminator())

    def __init__(
        self,
        sym_name: str | StringAttr,
        policy: str | SchedulingPolicyAttr = "static",
        sm_count: int | IntegerAttr | None = None,
        body: Region | None = None,
        provenance: ProvenanceAttr | None = None,
    ) -> None:
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        if isinstance(policy, str):
            policy = SchedulingPolicyAttr(policy)
        properties: dict[str, object] = {
            "sym_name": sym_name,
            "policy": policy,
        }
        if sm_count is not None:
            from xdsl.dialects.builtin import IntegerType  # local to avoid cycle
            if isinstance(sm_count, int):
                sm_count = IntegerAttr(sm_count, IntegerType(64))
            properties["sm_count"] = sm_count
        if provenance is not None:
            properties["provenance"] = provenance
        super().__init__(
            properties=properties,
            regions=[body if body is not None else Region()],
        )

    def verify_(self) -> None:
        if self.policy.policy.data not in SchedulingPolicyAttr._VALID:
            raise VerifyException(
                f"event.graph policy '{self.policy.policy.data}' invalid, "
                f"expected one of {sorted(SchedulingPolicyAttr._VALID)}"
            )
        if self.sm_count is not None and self.sm_count.value.data <= 0:
            raise VerifyException(
                f"event.graph sm_count must be positive, "
                f"got {self.sm_count.value.data}"
            )


@irdl_op_definition
class UpdateOp(IRDLOperation):
    """Data-dependent rewrite of event-tensor counters.

    Models the MoE pattern: at runtime, ``topk`` decides how many tokens
    each expert receives, so the per-expert event counters are initialised
    from a runtime int tensor.  ``source_tensor`` names the runtime tensor
    (e.g. ``"topk"``); ``index_expr`` is the einsum expression mapping
    source coords into target event coords.
    """

    name = "event.update"

    target = prop_def(EventCoordAttr)
    source_tensor = prop_def(StringAttr)
    index_expr = prop_def(StringAttr)
    provenance = opt_prop_def(ProvenanceAttr)


@irdl_op_definition
class TriggerOp(IRDLOperation):
    """Runtime materialization of a variable number of consumer tasks.

    Models the second half of the MoE pattern: ``exp_indptr`` is a prefix
    sum (CSR-style) telling each expert how many GroupGEMM tiles to
    activate.  ``trigger_range`` names the indptr tensor; the codegen
    reads ``range[exp_indptr[i], exp_indptr[i+1])`` and triggers exactly
    that many ``CallDeviceOp`` instances.
    """

    name = "event.trigger"

    target = prop_def(EventCoordAttr)
    trigger_range = prop_def(StringAttr)  # name of CSR-style int tensor
    provenance = opt_prop_def(ProvenanceAttr)


@irdl_op_definition
class MaterializeViewOp(IRDLOperation):
    """Runtime materialization of a symbolic-shape event-tensor view.

    Implements the Fig. 4 mechanism: a single symbolic-shape Event Tensor
    template is instantiated at runtime once concrete shape values are
    known.  The runtime allocates the int tensor, initialises every entry
    to ``wait_count``, and binds the symbol to a concrete extent.
    """

    name = "event.materialize_view"

    event_ref = prop_def(StringAttr)
    concrete_shape = prop_def(ArrayAttr)  # ArrayAttr of IntegerAttr
    provenance = opt_prop_def(ProvenanceAttr)

    _NEEDS_NONNEG: ClassVar[bool] = True

    def verify_(self) -> None:
        for dim in self.concrete_shape.data:
            if isinstance(dim, IntegerAttr) and dim.value.data < 0:
                raise VerifyException(
                    f"event.materialize_view concrete_shape entries must be "
                    f"non-negative, got {dim.value.data}"
                )


__all__ = [
    "CallDeviceOp",
    "EventTensorOp",
    "GraphOp",
    "MaterializeViewOp",
    "NotifyOp",
    "TriggerOp",
    "UpdateOp",
    "WaitOp",
]
