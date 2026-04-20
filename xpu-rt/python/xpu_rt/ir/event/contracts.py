"""Well-formedness contracts for the Event Tensor dialect.

Implements structural invariants that must hold for any module containing
``event.graph`` regions before scheduling transformations or Triton
emission run.  These checks are surfaced through ``check_event_graph``
and consumed by:

    - ``compgen.ir.payload.passes.megakernel_static_schedule`` (precondition)
    - ``compgen.agent.gates.megakernel`` (structural gate)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import IntegerAttr

from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
    NotifyOp,
    TriggerOp,
    UpdateOp,
    WaitOp,
)


@dataclass
class EventGraphReport:
    """Result of running well-formedness checks over an ``event.graph``."""

    graph_name: str
    event_decls: dict[str, EventTensorOp] = field(default_factory=dict)
    notify_counts: dict[str, int] = field(default_factory=dict)
    wait_counts: dict[str, int] = field(default_factory=dict)
    data_dep_events: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_event_graph(graph: GraphOp) -> EventGraphReport:
    """Validate a single ``event.graph`` op.

    Checks:
        1. Every notify/wait coord references an EventTensorOp declared in
           the graph.
        2. For static policy, every event tensor has at least one producer
           (notify) and at least one consumer (wait or implicit via edge).
        3. Notify totals ``>=`` declared wait count (otherwise consumers
           would deadlock waiting forever).
        4. ``UpdateOp`` / ``TriggerOp`` are forbidden when policy is
           ``static`` (data-dep is dynamic-only territory).
    """
    report = EventGraphReport(graph_name=graph.sym_name.data)
    is_static = graph.policy.policy.data == "static"

    for op in graph.body.ops:
        if isinstance(op, EventTensorOp):
            name = op.sym_name.data
            if name in report.event_decls:
                report.errors.append(f"duplicate event tensor declaration: {name}")
            report.event_decls[name] = op
            report.notify_counts.setdefault(name, 0)
            report.wait_counts.setdefault(name, 0)

    for op in graph.body.walk():
        if isinstance(op, NotifyOp):
            ref = op.coord.event_ref.data
            if ref not in report.event_decls:
                report.errors.append(f"notify on undeclared event {ref!r}")
                continue
            report.notify_counts[ref] += op.coord.decrement.value.data
        elif isinstance(op, WaitOp):
            ref = op.coord.event_ref.data
            if ref not in report.event_decls:
                report.errors.append(f"wait on undeclared event {ref!r}")
                continue
            report.wait_counts[ref] += 1
        elif isinstance(op, (UpdateOp, TriggerOp)):
            ref = op.target.event_ref.data
            if ref not in report.event_decls:
                report.errors.append(f"{op.name} targets undeclared event {ref!r}")
            else:
                report.data_dep_events.add(ref)
            if is_static:
                report.errors.append(f"{op.name} forbidden under static scheduling policy")
        elif isinstance(op, CallDeviceOp):
            # Multiplicity per dispatched task: every entry in task_shape
            # contributes a multiplier so that a CallDeviceOp on a 4-tile
            # grid with 1 out_edge counts as 4 producer notifies.
            task_grid = 1
            for d in op.task_shape.data:
                if isinstance(d, IntegerAttr):
                    extent = d.value.data
                    if extent > 0:
                        task_grid *= extent
            for edge_attr_name in ("in_edges", "out_edges"):
                edges = getattr(op, edge_attr_name)
                if edges is None:
                    continue
                for coord in edges.data:
                    if not hasattr(coord, "event_ref"):
                        continue
                    ref = coord.event_ref.data
                    if ref not in report.event_decls:
                        report.errors.append(f"call_device {edge_attr_name} references undeclared event {ref!r}")
                        continue
                    if edge_attr_name == "out_edges":
                        report.notify_counts[ref] += coord.decrement.value.data * task_grid
                    else:
                        report.wait_counts[ref] += task_grid

    for name, et in report.event_decls.items():
        if name in report.data_dep_events:
            continue  # counters initialised at runtime by event.update
        wait_count = et.wait_count.value.data
        notifies = report.notify_counts.get(name, 0)
        waits = report.wait_counts.get(name, 0)
        if waits == 0 and notifies == 0:
            report.warnings.append(f"event {name!r} declared but unused")
            continue
        if notifies < wait_count:
            report.errors.append(
                f"event {name!r} declared wait_count={wait_count} but only "
                f"{notifies} notify decrements emitted -- consumers will "
                f"deadlock"
            )
        if waits == 0:
            report.warnings.append(f"event {name!r} produced ({notifies}) but never waited on")

    if graph.sm_count is not None and isinstance(graph.sm_count, IntegerAttr):
        if graph.sm_count.value.data <= 0:
            report.errors.append("event.graph sm_count must be positive")

    return report


__all__ = ["EventGraphReport", "check_event_graph"]
