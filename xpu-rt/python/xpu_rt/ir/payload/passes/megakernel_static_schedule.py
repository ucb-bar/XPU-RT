"""Static-scheduling transformation for Event Tensor megakernels.

Implements Algorithm 1 from the Event Tensor Compiler paper (Jin et al.,
MLSys '26): given a module containing an ``event.graph`` op with policy
``static``, the pass

    1. Walks the graph body to extract the per-task event-edge DAG.
    2. Calls :func:`compgen.solve.per_sm_queue.solve_per_sm_queue` to
       decide per-SM task assignment + ordering.
    3. Annotates the graph op with a ``compgen.static_schedule``
       attribute holding the solved schedule (JSON), which the persistent
       Triton emitter consumes.

The pass is *non-destructive*: it does not delete or rewrite any
Event-Tensor IR ops.  Lowering to Triton happens in
``compgen.ir.tile.lower_megakernel`` once the schedule annotation is
present.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr

from compgen.ir.event.contracts import check_event_graph
from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
    NotifyOp,
    WaitOp,
)
from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg, ToolResult
from compgen.solve.per_sm_queue import EventEdge, TileTask, solve_per_sm_queue


_DEFAULT_DURATION_US = 1.0
_SCHEDULE_ATTR = "compgen.static_schedule"


def _coord_indices_to_str(coord_indices_attr: Any) -> str:
    parts: list[str] = []
    for el in coord_indices_attr.data:
        if hasattr(el, "data"):
            parts.append(el.data)
        else:
            parts.append(str(el))
    return ",".join(parts)


def _gather_dispatches(graph: GraphOp) -> list[CallDeviceOp]:
    return [op for op in graph.body.walk() if isinstance(op, CallDeviceOp)]


def _expand_task_grid(call: CallDeviceOp) -> list[str]:
    """Expand a CallDeviceOp's task_shape into per-coord task ids.

    Symbolic dims (``-1``) are treated as extent ``1`` for compile-time
    scheduling -- the runtime materialises additional tasks via
    ``event.materialize_view``.
    """
    name = call.device_func.root_reference.data
    dims: list[int] = []
    for d in call.task_shape.data:
        if isinstance(d, IntegerAttr):
            v = d.value.data
            dims.append(v if v > 0 else 1)
    if not dims:
        return [f"{name}:0"]
    coords = [[]]
    for extent in dims:
        coords = [c + [k] for c in coords for k in range(extent)]
    return [f"{name}:{','.join(str(k) for k in coord)}" for coord in coords]


def _coord_to_tasks(
    call_lookup: dict[str, list[str]],
    func_name: str,
    coord_index_str: str,
) -> list[str]:
    """Map an event coord index expression to concrete task ids.

    For Phase A we keep this conservative: ``coord_index_str`` may be a
    literal-only expression like ``"0"`` or ``"3,1"``.  Anything more
    complex (einsum syntax) widens to *all* tasks of ``func_name`` -- the
    solver then enforces the union of edges.  This is sound (over-
    approximates dependencies) but may serialise more than necessary.
    Phase B's dynamic scheduler removes this restriction entirely.
    """
    candidates = call_lookup.get(func_name, [])
    if not candidates:
        return []
    needle = f"{func_name}:{coord_index_str.strip()}"
    if needle in candidates:
        return [needle]
    # Try numeric-tuple form, e.g. "0" -> "<func>:0"
    try:
        as_int = int(coord_index_str.strip())
        candidate = f"{func_name}:{as_int}"
        if candidate in candidates:
            return [candidate]
    except ValueError:
        pass
    return list(candidates)


def extract_event_edges(graph: GraphOp) -> tuple[list[TileTask], list[EventEdge]]:
    """Walk a graph body, return (tasks, edges) suitable for the solver.

    Tasks are the cartesian expansion of each ``CallDeviceOp``'s
    ``task_shape``.  Edges come from matching ``out_edges`` (notify) on
    one dispatch with ``in_edges`` (wait) on another via the shared
    EventTensor symbol.
    """
    dispatches = _gather_dispatches(graph)
    func_to_tasks: dict[str, list[str]] = {}
    tasks: list[TileTask] = []
    for call in dispatches:
        ids = _expand_task_grid(call)
        func_to_tasks.setdefault(call.device_func.root_reference.data, []).extend(ids)
        for tid in ids:
            tasks.append(
                TileTask(
                    task_id=tid,
                    device_func=call.device_func.root_reference.data,
                    duration_us=_DEFAULT_DURATION_US,
                ),
            )

    # Build event -> producers / consumers maps.
    event_producers: dict[str, list[tuple[str, str]]] = {}
    event_consumers: dict[str, list[tuple[str, str]]] = {}
    for call in dispatches:
        func = call.device_func.root_reference.data
        if call.out_edges is not None:
            for coord in call.out_edges.data:
                if hasattr(coord, "event_ref"):
                    event_producers.setdefault(coord.event_ref.data, []).append(
                        (func, _coord_indices_to_str(coord.indices))
                    )
        if call.in_edges is not None:
            for coord in call.in_edges.data:
                if hasattr(coord, "event_ref"):
                    event_consumers.setdefault(coord.event_ref.data, []).append(
                        (func, _coord_indices_to_str(coord.indices))
                    )

    # Free-floating notify/wait ops also count -- they let the user write
    # synchronisation that doesn't fit a clean dispatch edge.
    for op in graph.body.walk():
        if isinstance(op, NotifyOp):
            ev = op.coord.event_ref.data
            if ev not in event_producers:
                event_producers[ev] = []
        if isinstance(op, WaitOp):
            ev = op.coord.event_ref.data
            if ev not in event_consumers:
                event_consumers[ev] = []

    edges: list[EventEdge] = []
    for event, producers in event_producers.items():
        consumers = event_consumers.get(event, [])
        for pf, pi in producers:
            p_tasks = _coord_to_tasks(func_to_tasks, pf, pi)
            for cf, ci in consumers:
                c_tasks = _coord_to_tasks(func_to_tasks, cf, ci)
                for p in p_tasks:
                    for c in c_tasks:
                        if p != c:
                            edges.append(EventEdge(producer=p, consumer=c, event=event))

    return tasks, edges


class StaticMegakernelSchedule(PayloadPass):
    """Algorithm 1 -- static per-SM scheduling for Event Tensor megakernels.

    Walks every ``event.graph`` with policy ``static`` in the module,
    runs the per-SM CP-SAT solver, and stamps the solved schedule onto
    the graph op as a ``compgen.static_schedule`` JSON attribute.  Idempotent:
    running twice produces identical output.
    """

    name: ClassVar[str] = "megakernel_static_schedule"
    phase: ClassVar[int] = 4
    wraps_pass: ClassVar[str] = "ETC/Algorithm1"
    covers_families: ClassVar[frozenset[str]] = frozenset({"persistent_kernel"})
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "very_high"
    description: ClassVar[str] = (
        "Algorithm 1 from Event Tensor Compiler: solve per-SM execution "
        "queues for an event.graph and bake them into a static schedule "
        "annotation consumed by the persistent Triton emitter."
    )
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg(
                name="sm_count",
                dtype="int",
                description="number of SMs to partition tasks across",
                required=False,
                default="108",
            ),
            ToolArg(
                name="solver_timeout_ms",
                dtype="int",
                description="CP-SAT solver wall-clock budget",
                required=False,
                default="10000",
            ),
        )

    def tool_result(self) -> ToolResult:
        return ToolResult(
            dtype="ModuleOp",
            description="module with compgen.static_schedule annotated on each event.graph",
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        sm_count_default = int(kwargs.get("sm_count", 108) or 108)
        timeout_ms = int(kwargs.get("solver_timeout_ms", 10000) or 10000)

        for graph in [op for op in module.walk() if isinstance(op, GraphOp)]:
            if graph.policy.policy.data != "static":
                continue
            report = check_event_graph(graph)
            if not report.ok:
                graph.attributes[_SCHEDULE_ATTR] = StringAttr(
                    json.dumps({"status": "rejected", "errors": report.errors})
                )
                continue
            sm_count = (
                graph.sm_count.value.data
                if graph.sm_count is not None
                else sm_count_default
            )
            tasks, edges = extract_event_edges(graph)
            sched = solve_per_sm_queue(
                tasks=tasks,
                edges=edges,
                sm_count=sm_count,
                timeout_ms=timeout_ms,
            )
            event_decls = [
                {
                    "name": et.sym_name.data,
                    "shape": [
                        d.value.data
                        for d in et.event_type.shape.data
                        if isinstance(d, IntegerAttr)
                    ],
                    "wait_count": et.wait_count.value.data,
                    "scope": et.event_type.scope.data,
                    "counter_dtype": et.event_type.counter_dtype.data,
                }
                for et in graph.body.ops
                if isinstance(et, EventTensorOp)
            ]
            payload = {
                "status": "ok" if sched.feasible else "infeasible",
                "sm_count": sm_count,
                "makespan_us": sched.makespan_us,
                "solve_time_ms": sched.solve_time_ms,
                "per_sm_order": {
                    str(sm): order for sm, order in sched.per_sm_order.items()
                },
                "assignment": sched.assignment,
                "event_tensor_decls": event_decls,
                "task_count": len(tasks),
                "edge_count": len(edges),
            }
            graph.attributes[_SCHEDULE_ATTR] = StringAttr(json.dumps(payload, sort_keys=True))
        return module


__all__ = [
    "StaticMegakernelSchedule",
    "extract_event_edges",
]
