"""Event Tensor IR dialect registration.

Registers all Event Tensor IR operations and attributes with xDSL.  The
``Event`` dialect object is intended for parser/printer context
registration::

    ctx.register_dialect("event", lambda: Event)
"""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.event.attrs import (
    EventCoordAttr,
    EventTensorTypeAttr,
    SchedulingPolicyAttr,
)
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

ALL_OPS = [
    EventTensorOp,
    NotifyOp,
    WaitOp,
    CallDeviceOp,
    GraphOp,
    UpdateOp,
    TriggerOp,
    MaterializeViewOp,
]

ALL_ATTRS = [
    EventTensorTypeAttr,
    EventCoordAttr,
    SchedulingPolicyAttr,
]

Event = Dialect("event", ALL_OPS, ALL_ATTRS)
"""The Event Tensor IR dialect."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Event"]
