"""Event Tensor IR -- first-class abstraction for fine-grained GPU sync.

The Event Tensor abstraction (Jin et al., MLSys '26) elevates per-task
counter-based semaphores into a multi-dimensional tensor object. Combined
with persistent megakernel codegen and either static (precomputed per-SM
queue) or dynamic (on-GPU push/pop) scheduling, it eliminates kernel-launch
overhead and the implicit kernel-boundary synchronization that bottlenecks
modern LLM serving.

This dialect is the IR layer of CompGen's ETC integration. It is a sibling
of ``compgen.tile``, ``compgen.accel``, and ``compgen.ukernel``.

Register the dialect on a ``Context`` via::

    ctx.register_dialect("event", lambda: Event)
"""

from __future__ import annotations

from compgen.ir.event.attrs import (
    EventCoordAttr,
    EventTensorTypeAttr,
    SchedulingPolicyAttr,
)
from compgen.ir.event.dialect import ALL_ATTRS, ALL_OPS, Event
from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
    MaterializeViewOp,
    NotifyOp,
    TriggerOp,
    UpdateOp,
    WaitOp,
)

__all__ = [
    "ALL_ATTRS",
    "ALL_OPS",
    "CallDeviceOp",
    "Event",
    "EventCoordAttr",
    "EventTensorOp",
    "EventTensorTypeAttr",
    "GraphOp",
    "MaterializeViewOp",
    "NotifyOp",
    "SchedulingPolicyAttr",
    "TriggerOp",
    "UpdateOp",
    "WaitOp",
]
