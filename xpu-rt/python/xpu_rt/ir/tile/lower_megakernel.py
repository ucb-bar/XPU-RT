"""Lower an Event Tensor megakernel graph to a single persistent Triton kernel.

Sibling of :mod:`compgen.ir.tile.lower_triton` and
:mod:`compgen.ir.tile.lower_exo`.  Consumes an ``event.graph`` op that has
already been annotated by
:mod:`compgen.ir.payload.passes.megakernel_static_schedule` (Algorithm 1 of
the Event Tensor Compiler paper) and produces a single ``@triton.jit``
function whose grid equals the target's SM count.

Code-generation strategy (Phase A, static scheduler):

    * Allocate one ``i32``/``i64`` tensor per Event Tensor in the graph,
      zero-initialised at host launch time and seeded with ``wait_count``.
    * Embed the per-SM task queue as a flat ``tl.constexpr`` table.
    * Per-SM body is a ``while task_idx < my_queue_len`` loop that fetches
      ``(task_id, task_type)`` from the table, dispatches into a device-
      function-specific branch, and advances.
    * Each branch invokes a real per-device-function ``@triton.jit`` body
      supplied by the caller via :class:`DeviceFunctionSpec`.  Bodies
      receive the task id, every data pointer, every event pointer, and
      every constexpr arg declared by the caller.
    * ``event.notify`` calls inside a body lower to
      ``tl.atomic_add(E_ptr + linear_idx, -k)``.
    * ``event.wait`` calls inside a body lower to a spin-poll
      ``while tl.atomic_or(E_ptr + linear_idx, 0) > 0: pass``.

When :paramref:`device_functions` is omitted the emitter falls back to
empty-body placeholders (used by structural tests); a real workload must
supply bodies for every device function referenced by the graph.

Phase B will add the dynamic push/pop scheduler in a sibling emitter.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerAttr, StringAttr

from compgen.ir.event.attrs import EventTensorTypeAttr
from compgen.ir.event.ops import EventTensorOp, GraphOp

_SCHEDULE_ATTR = "compgen.static_schedule"


@dataclass(frozen=True)
class DeviceFunctionSpec:
    """A real ``@triton.jit`` body for one device function in the graph.

    Attributes:
        name:        The device-function symbol (matches
                     ``CallDeviceOp.device_func``).
        body_source: Indented Triton source for the function body.  May
                     reference any data pointer in
                     :attr:`MegakernelLoweringSpec.data_pointers`, any
                     event-tensor pointer declared on the graph, any
                     constexpr arg in
                     :attr:`MegakernelLoweringSpec.constexpr_args`, and
                     ``task_id`` (the int task coordinate).  Each line
                     must already be indented relative to the function
                     body (4 spaces) -- the emitter does NOT re-indent.
    """

    name: str
    body_source: str


@dataclass(frozen=True)
class MegakernelLoweringSpec:
    """Caller-supplied wiring around the megakernel emitter.

    Attributes:
        data_pointers:    Names of pointer args (e.g. ``"A_ptr"``).
                          Passed positionally to every device function
                          body and to the megakernel itself.
        constexpr_args:   Names of ``tl.constexpr`` args (e.g. ``"M"``,
                          ``"BLOCK_M"``).
        device_functions: One :class:`DeviceFunctionSpec` per device
                          function referenced by the graph.  When empty
                          the emitter falls back to ``pass``-bodied
                          stubs.
        num_warps:        Triton ``num_warps`` for the launch.
        num_stages:       Triton ``num_stages`` for the launch.
    """

    data_pointers: tuple[str, ...] = ()
    constexpr_args: tuple[str, ...] = ()
    device_functions: tuple[DeviceFunctionSpec, ...] = ()
    num_warps: int = 4
    num_stages: int = 2


@dataclass(frozen=True)
class MegakernelLoweringResult:
    """Output of lowering an event.graph to a persistent Triton kernel.

    Attributes:
        kernel_name:    name of the emitted ``@triton.jit`` function.
        kernel_source:  full Python source (including the @triton.jit decorator).
        launch_config:  ``{"grid": int, "num_warps": int, "num_stages": int}``
                        consumable by the host-side launcher.
        event_layout:   one entry per Event Tensor describing its size +
                        dtype + initial wait count, used by the host to
                        allocate and seed the global int tensors.
        task_queue:     per-SM task list (``sm_idx -> [(task_id, kind), ...]``)
                        baked into the kernel as a constexpr.
        device_function_table: ``{kind_int: device_func_name}`` -- the
                        order callers must use when filling per-task
                        ``task_kind`` entries in ``QUEUE_PTR``.
        diagnostics:    non-fatal warnings produced during lowering.
    """

    kernel_name: str
    kernel_source: str
    launch_config: dict[str, Any] = field(default_factory=dict)
    event_layout: list[dict[str, Any]] = field(default_factory=list)
    task_queue: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    device_function_table: dict[int, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


def _event_layout(graph: GraphOp) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for op in graph.body.ops:
        if not isinstance(op, EventTensorOp):
            continue
        et: EventTensorTypeAttr = op.event_type
        shape = [d.value.data for d in et.shape.data if isinstance(d, IntegerAttr)]
        size = 1
        for d in shape:
            size *= max(d, 1)
        layout.append(
            {
                "name": op.sym_name.data,
                "shape": shape,
                "size": size,
                "wait_count": op.wait_count.value.data,
                "scope": et.scope.data,
                "counter_dtype": et.counter_dtype.data,
            },
        )
    return layout


def _kernel_name(graph: GraphOp) -> str:
    return f"megakernel_{graph.sym_name.data}"


def _emit_task_table(
    per_sm_order: dict[str, list[str]],
    task_kind_map: dict[str, int],
) -> tuple[str, dict[int, list[tuple[str, int]]]]:
    """Emit a Python-source table + return a structured copy.

    Each per-SM list becomes an entry of the form::

        ((task_id_int, kind_int), ...)

    where ``task_id_int`` is the index into the *flat* task list (used by
    each device branch to pick its task coordinate).
    """
    lines: list[str] = ["TASK_TABLE = ["]
    structured: dict[int, list[tuple[str, int]]] = {}
    for sm_str, queue in sorted(per_sm_order.items(), key=lambda kv: int(kv[0])):
        sm_idx = int(sm_str)
        sm_entries: list[tuple[str, int]] = []
        encoded_entries: list[str] = []
        for tid in queue:
            kind = task_kind_map.get(tid, 0)
            sm_entries.append((tid, kind))
            encoded_entries.append(f'("{tid}", {kind})')
        structured[sm_idx] = sm_entries
        lines.append(f"    [{', '.join(encoded_entries) or '# empty'}],   # SM {sm_idx}")
    lines.append("]")
    return "\n".join(lines), structured


def _signature_args(
    data_pointers: Sequence[str],
    event_names: Sequence[str],
    constexpr_args: Sequence[str],
) -> tuple[str, str]:
    """Build the function signature + the call-site argument list.

    Returns ``(decl, call)`` where ``decl`` is suitable for placement
    inside ``def f(<decl>):`` and ``call`` is suitable for ``f(<call>)``.
    """
    decl_parts: list[str] = []
    call_parts: list[str] = []
    for ptr in data_pointers:
        decl_parts.append(ptr)
        call_parts.append(ptr)
    for ev in event_names:
        decl_parts.append(f"{ev}_ptr")
        call_parts.append(f"{ev}_ptr")
    for ce in constexpr_args:
        decl_parts.append(f"{ce}: tl.constexpr")
        call_parts.append(ce)
    return ", ".join(decl_parts), ", ".join(call_parts)


def _device_function_table(
    funcs: Sequence[str],
    spec: MegakernelLoweringSpec,
) -> dict[str, DeviceFunctionSpec]:
    by_name = {df.name: df for df in spec.device_functions}
    out: dict[str, DeviceFunctionSpec] = {}
    for fn in funcs:
        if fn in by_name:
            out[fn] = by_name[fn]
        else:
            out[fn] = DeviceFunctionSpec(
                name=fn,
                body_source="    pass  # stub: no DeviceFunctionSpec supplied",
            )
    return out


def _emit_dispatch_branches(
    funcs: Sequence[str],
    func_to_kind: dict[str, int],
    call_args: str,
) -> str:
    branches: list[str] = []
    for k, fn in enumerate(funcs):
        kind = func_to_kind[fn]
        keyword = "elif" if k else "if"
        branches.append(f"        {keyword} task_kind == {kind}:")
        branches.append(f"            _run_{fn}(task_id, {call_args})")
    if not branches:
        branches = ["        pass  # no tasks"]
    return "\n".join(branches)


def lower_megakernel(
    graph: GraphOp,
    spec: MegakernelLoweringSpec | None = None,
) -> MegakernelLoweringResult:
    """Lower an annotated ``event.graph`` to persistent-Triton source.

    Raises ``ValueError`` if the graph has not been annotated with
    ``compgen.static_schedule``; callers must run
    :class:`StaticMegakernelSchedule` first.
    """
    if _SCHEDULE_ATTR not in graph.attributes:
        raise ValueError(
            f"event.graph {graph.sym_name.data!r} is missing the "
            f"{_SCHEDULE_ATTR!r} attribute; run StaticMegakernelSchedule first"
        )
    payload_attr = graph.attributes[_SCHEDULE_ATTR]
    if not isinstance(payload_attr, StringAttr):
        raise ValueError(f"{_SCHEDULE_ATTR} must be a StringAttr, got {type(payload_attr).__name__}")
    schedule = json.loads(payload_attr.data)
    if schedule.get("status") != "ok":
        return MegakernelLoweringResult(
            kernel_name=_kernel_name(graph),
            kernel_source="",
            diagnostics=[f"static schedule rejected: {schedule.get('errors', [])}"],
        )

    if spec is None:
        spec = MegakernelLoweringSpec()

    events = _event_layout(graph)
    event_names = [e["name"] for e in events]
    sm_count = int(schedule["sm_count"])
    per_sm_order = {str(k): list(v) for k, v in schedule["per_sm_order"].items()}
    assignment = schedule["assignment"]

    funcs = sorted({tid.split(":")[0] for tid in assignment})
    func_to_kind = {fn: i for i, fn in enumerate(funcs)}
    task_kind_map: dict[str, int] = {tid: func_to_kind[tid.split(":")[0]] for tid in assignment}

    task_table_src, structured_queue = _emit_task_table(per_sm_order, task_kind_map)
    dispatch_decl, dispatch_call = _signature_args(spec.data_pointers, event_names, spec.constexpr_args)
    body_table = _device_function_table(funcs, spec)
    dispatch_branches = _emit_dispatch_branches(funcs, func_to_kind, dispatch_call)
    kernel_name = _kernel_name(graph)

    lines: list[str] = []
    lines.append("import triton")
    lines.append("import triton.language as tl")
    lines.append("")
    lines.append("# Per-SM task table baked into the megakernel at compile time.")
    lines.append(task_table_src)
    lines.append("")
    lines.append("# --- atomic notify / wait helpers (event-tensor protocol) ---")
    lines.append("@triton.jit")
    lines.append("def _event_notify(E_ptr, linear_idx, decrement: tl.constexpr):")
    lines.append("    tl.atomic_add(E_ptr + linear_idx, -decrement)")
    lines.append("")
    lines.append("@triton.jit")
    lines.append("def _event_wait(E_ptr, linear_idx):")
    lines.append("    counter = tl.atomic_or(E_ptr + linear_idx, 0)")
    lines.append("    while counter > 0:")
    lines.append("        counter = tl.atomic_or(E_ptr + linear_idx, 0)")
    lines.append("")
    lines.append("# --- per-device-function bodies ---")
    for fn in funcs:
        body = body_table[fn]
        lines.append("@triton.jit")
        lines.append(f"def _run_{fn}(task_id, {dispatch_decl}):")
        body_text = textwrap.dedent(body.body_source).strip("\n")
        if not body_text.strip():
            body_text = "pass"
        # Indent every line of the body by 4 spaces (function-body indent).
        indented = textwrap.indent(body_text, "    ")
        lines.append(indented)
        lines.append("")

    # Megakernel signature: (data_ptrs, event_ptrs, QUEUE, QUEUE_LEN,
    # then all constexprs including user constexprs + SM_COUNT, MAX_QLEN).
    mk_decl_parts: list[str] = []
    for ptr in spec.data_pointers:
        mk_decl_parts.append(ptr)
    for ev in event_names:
        mk_decl_parts.append(f"{ev}_ptr")
    mk_decl_parts.append("QUEUE_PTR")
    mk_decl_parts.append("QUEUE_LEN_PTR")
    for ce in spec.constexpr_args:
        mk_decl_parts.append(f"{ce}: tl.constexpr")
    mk_decl_parts.append("SM_COUNT: tl.constexpr")
    mk_decl_parts.append("MAX_QLEN: tl.constexpr")

    lines.append("# --- persistent megakernel: grid = SM_COUNT ---")
    lines.append("@triton.jit")
    lines.append(f"def {kernel_name}(")
    for part in mk_decl_parts:
        lines.append(f"    {part},")
    lines.append("):")
    lines.append('    """Persistent megakernel emitted by ETC Algorithm 1.')
    lines.append("")
    lines.append("    grid = (SM_COUNT,); each program walks its precomputed queue.")
    lines.append('    """')
    lines.append("    sm_id = tl.program_id(0)")
    lines.append("    qlen = tl.load(QUEUE_LEN_PTR + sm_id)")
    lines.append("    task_idx = 0")
    lines.append("    while task_idx < qlen:")
    lines.append("        task_id = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 0)")
    lines.append("        task_kind = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 1)")
    lines.append(dispatch_branches)
    lines.append("        task_idx += 1")
    lines.append("")

    kernel_source = "\n".join(lines)

    launch_config = {
        "grid": sm_count,
        "num_warps": spec.num_warps,
        "num_stages": spec.num_stages,
    }

    device_function_table = {func_to_kind[fn]: fn for fn in funcs}

    return MegakernelLoweringResult(
        kernel_name=kernel_name,
        kernel_source=kernel_source,
        launch_config=launch_config,
        event_layout=events,
        task_queue=structured_queue,
        device_function_table=device_function_table,
    )


__all__ = [
    "DeviceFunctionSpec",
    "MegakernelLoweringResult",
    "MegakernelLoweringSpec",
    "lower_megakernel",
]
