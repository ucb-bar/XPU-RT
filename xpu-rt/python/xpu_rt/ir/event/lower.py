"""Lowering from ``event`` dialect IR to runtime :class:`MegakernelGraph`.

Closes the IR → execution loop for the paper's megakernel abstraction
(Jin et al., MLSys '26): a ``builtin.module`` containing one or more
``event.graph`` ops is lowered into :class:`compgen.runtime.megakernel.MegakernelGraph`
objects ready to :meth:`~compgen.runtime.megakernel.MegakernelGraph.launch`.

Lowering covers the static / DAG-aware parts of the dialect:

- :class:`~compgen.ir.event.ops.EventTensorOp` →
  :class:`~compgen.runtime.event_tensor.EventTensor` (shape, dtype,
  scope, wait_count_default honoured).
- :class:`~compgen.ir.event.ops.CallDeviceOp` →
  :class:`~compgen.runtime.megakernel.DeviceCall` (task_shape +
  in_edges + out_edges).
- :class:`~compgen.ir.event.ops.GraphOp` →
  :class:`~compgen.runtime.megakernel.MegakernelGraph` (policy,
  sm_count, composed device calls).

**Index expressions** in :class:`~compgen.ir.event.attrs.EventCoordAttr`
are compiled into real Python callables (``compile`` + ``eval`` with a
restricted namespace). The namespace binds task-coord positions to
letters ``i j k l m n o p q r`` — the paper's einsum convention.
Integer literals, arithmetic (`+ - * // %`), and ``topk[i]``-style
subscription are all supported if the caller injects the backing
tensors into ``index_env``.

**Data-dependent ops** (``UpdateOp`` / ``TriggerOp`` /
``MaterializeViewOp``) are recognised — the lowering raises a clear
:class:`NotImplementedError` naming the op so callers know which
symbolic-shape / dynamic-grid extension they hit. These will land
when the symbolic-shape runtime does; the megakernel's static
subset is fully covered today.

Usage::

    from compgen.ir.event.lower import lower_event_module

    graphs = lower_event_module(module, device_funcs={
        "gemm_tile": my_gemm_body_fn,
        "reduce_scatter_row": my_rs_body_fn,
    })
    graphs["gemm_rs"].launch(timeout_s=5.0)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from xdsl.dialects.builtin import IntegerAttr, ModuleOp

from compgen.ir.event.attrs import EventCoordAttr
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
from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Index-expression compilation
# ---------------------------------------------------------------------------


# Paper convention: task-coord position 0 binds to "i", position 1 to
# "j", and so on. Ten positions (i..r) covers every task grid in the
# paper's figures and then some.
_DIM_LETTERS: tuple[str, ...] = ("i", "j", "k", "l", "m", "n", "o", "p", "q", "r")


def _compile_index_fn(
    indices_strs: tuple[str, ...],
    *,
    index_env: dict[str, Any] | None,
    edge_event_name: str,
) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Compile a list of event-coord expressions into a runtime callable.

    Each expression must evaluate to an ``int``. Variables ``i..r`` are
    bound to the task coordinate; integer literals and ordinary
    arithmetic work out of the box. ``index_env`` seeds the namespace
    with extra bindings — useful for the paper's ``topk[i]`` style
    closures over runtime int tensors::

        index_env = {"topk": torch.tensor([2, 0, 3, 1])}

    Args:
        indices_strs: One expression per event-tensor dim.
        index_env: Extra bindings folded into the eval namespace.
            Keys must not collide with the position letters.
        edge_event_name: Used only to surface errors clearly.

    Returns:
        A callable mapping ``task_coord → event_coord`` (both tuples of
        ``int``). Raises ``ValueError`` at call time if an expression
        evaluates to a non-int.
    """
    # Compile once — avoids parse cost per task.
    compiled = []
    for s in indices_strs:
        try:
            compiled.append(compile(s, f"<event_idx:{edge_event_name}:{s}>", "eval"))
        except SyntaxError as exc:
            raise ValueError(
                f"event edge {edge_event_name!r}: index expression {s!r} is not a valid Python expression: {exc}"
            ) from exc

    # Shallow-copy the env so mutations by the caller later don't leak.
    env_base: dict[str, Any] = dict(index_env or {})
    for letter in _DIM_LETTERS:
        if letter in env_base:
            raise ValueError(
                f"index_env must not redefine the position letter {letter!r} — "
                f"it's bound to task coord position {_DIM_LETTERS.index(letter)}"
            )

    def index_fn(task_coord: tuple[int, ...]) -> tuple[int, ...]:
        # Build the namespace per call. Cheap; task_coord is short.
        # Bind as many letters as we have (i, j, ... up to position 9);
        # positions beyond 10 remain unbound, and expressions that
        # reference them surface a NameError we convert to ValueError
        # below. This lets purely-constant expressions work on
        # arbitrary-rank task grids without special casing.
        ns: dict[str, Any] = dict(env_base)
        for pos, val in enumerate(task_coord):
            if pos < len(_DIM_LETTERS):
                ns[_DIM_LETTERS[pos]] = int(val)

        out: list[int] = []
        for c_idx, code in enumerate(compiled):
            try:
                val = eval(code, {"__builtins__": {}}, ns)
            except NameError as exc:
                # Either an overflow letter (rank > 10) or a missing
                # index_env binding. Surface both with the same
                # "supported" wording so callers can catch either.
                raise ValueError(
                    f"event edge {edge_event_name!r}: index expression "
                    f"{indices_strs[c_idx]!r} references unbound name — "
                    f"letters {' '.join(_DIM_LETTERS)} cover positions 0-9 only; "
                    f"task_coord rank {len(task_coord)} exceeds that supported "
                    f"set or the name is missing from index_env ({exc})"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"event edge {edge_event_name!r}: index expression "
                    f"{indices_strs[c_idx]!r} failed at task_coord {task_coord}: {exc}"
                ) from exc
            if not isinstance(val, (int, bool)):
                raise ValueError(
                    f"event edge {edge_event_name!r}: index expression "
                    f"{indices_strs[c_idx]!r} produced {type(val).__name__} {val!r}; "
                    f"expected int"
                )
            out.append(int(val))
        return tuple(out)

    return index_fn


# ---------------------------------------------------------------------------
# Per-op lowering helpers
# ---------------------------------------------------------------------------


def _apply_materialize_view(
    op: MaterializeViewOp,
    tensors: dict[str, EventTensor],
    deferred_templates: dict[str, EventTensorOp],
) -> None:
    """Pre-launch pass: materialise a symbolic-shape event tensor.

    Paper Fig. 4. The IR declared a symbolic ``event.event_tensor``
    (with ``-1`` / ``0`` dims); this op names the concrete shape to
    bind it at. We allocate the concrete ``EventTensor`` and register
    it in ``tensors`` so downstream ``NotifyOp`` / ``WaitOp`` /
    ``CallDeviceOp`` edges resolve normally.

    Mutates ``tensors`` in place. Raises ``ValueError`` if the named
    template doesn't exist, already materialized, or the concrete
    shape is malformed.
    """
    from compgen.runtime.event_tensor import materialize_view

    name = op.event_ref.data
    concrete_shape = tuple(int(d.value.data) for d in op.concrete_shape.data if isinstance(d, IntegerAttr))
    if len(concrete_shape) != len(op.concrete_shape.data):
        raise ValueError(
            f"event.materialize_view {name!r}: concrete_shape must be all IntegerAttr, got {op.concrete_shape.data}"
        )
    for dim in concrete_shape:
        if dim <= 0:
            raise ValueError(
                f"event.materialize_view {name!r}: concrete_shape entries must be positive, got {concrete_shape}"
            )

    if name in tensors:
        raise ValueError(
            f"event.materialize_view {name!r}: tensor already materialized "
            f"(caller pre-seeded or an earlier MaterializeViewOp already ran)"
        )
    if name not in deferred_templates:
        raise ValueError(
            f"event.materialize_view {name!r}: no symbolic template found with "
            f"this symbol. Declare an event.event_tensor with the same sym_name "
            f"and a symbolic shape ({{-1 or 0}}) earlier in the graph."
        )
    template_op = deferred_templates.pop(name)

    # Build a throwaway "template" EventTensor with a shape of the
    # symbolic dims replaced by 1 so materialize_view can enforce
    # rank / wait_count consistency.
    symbolic_shape = tuple(
        (int(a.value.data) if isinstance(a, IntegerAttr) else 1) for a in template_op.event_type.shape.data
    )
    if len(symbolic_shape) != len(concrete_shape):
        raise ValueError(
            f"event.materialize_view {name!r}: template rank {len(symbolic_shape)} "
            f"!= concrete rank {len(concrete_shape)}"
        )
    for axis, (t_dim, c_dim) in enumerate(zip(symbolic_shape, concrete_shape)):
        if t_dim > 0 and t_dim != c_dim:
            raise ValueError(
                f"event.materialize_view {name!r}: template dim {axis} is concrete "
                f"({t_dim}) but concrete_shape[{axis}]={c_dim} disagrees"
            )

    wait_count = int(template_op.wait_count.value.data)
    dtype = template_op.event_type.counter_dtype.data
    scope = template_op.event_type.scope.data
    materialised = EventTensor(
        shape=concrete_shape,
        wait_count_default=wait_count,
        dtype=dtype,
        scope=scope,
        sym_name=name,
    )
    # Reuse the module-level helper for consistency with stand-alone
    # callers that want to materialise programmatically.
    _ = materialize_view  # keep symbol referenced for discoverability
    tensors[name] = materialised


def _apply_update(
    op: UpdateOp,
    tensors: dict[str, EventTensor],
    index_env: dict[str, Any] | None,
) -> None:
    """Pre-launch pass: data-dependent rewrite of event counters.

    Paper Fig. 5b. Reads the source tensor named by ``source_tensor``
    from ``index_env``, evaluates ``index_expr`` to resolve the
    target coord, and calls :meth:`EventTensor.update` to rewrite
    the cell's counter to the value taken from the source at the
    same coord.

    For the paper's MoE example::

        event.update @exp_counter[i] from @topk with expr="i"

    Means ``exp_counter[i] = topk[i]`` — the number of tokens routed
    to expert ``i`` seeds the event counter for the dependent GroupGEMM
    tiles.
    """
    target = op.target
    event_name = target.event_ref.data
    source_name = op.source_tensor.data
    if event_name not in tensors:
        raise ValueError(
            f"event.update: target event tensor {event_name!r} not allocated "
            f"(declare an event.event_tensor with this sym_name first)"
        )
    et = tensors[event_name]
    env = index_env or {}
    if source_name not in env:
        raise ValueError(
            f"event.update: source tensor {source_name!r} not found in index_env. "
            f"Pass it via index_env=dict({source_name}=...) when calling "
            f"lower_graph_op / lower_event_module."
        )
    source = env[source_name]

    indices_strs = tuple(s.data for s in target.indices.data)  # type: ignore[attr-defined]
    index_fn = _compile_index_fn(
        indices_strs,
        index_env=env,
        edge_event_name=event_name,
    )

    # Sweep the source tensor's coordinates and write each into the
    # corresponding event-tensor cell. The paper's MoE pattern has
    # ``source`` one-dim; we support rank N via cartesian product so
    # higher-rank update expressions lower without surprise.
    source_shape = _tensor_shape(source, source_name)
    import itertools

    for coord in itertools.product(*(range(d) for d in source_shape)):
        target_coord = index_fn(coord)
        new_count = _read_scalar(source, coord, source_name)
        et.update(target_coord, int(new_count))


def _apply_trigger(
    op: TriggerOp,
    tensors: dict[str, EventTensor],
    index_env: dict[str, Any] | None,
) -> None:
    """Pre-launch pass: runtime materialization of consumer-tile counts.

    Paper Fig. 5b. Reads the CSR-style ``trigger_range`` tensor from
    ``index_env`` and, for each event-coord, calls
    :meth:`EventTensor.trigger` with ``range[i+1] - range[i]`` (the
    number of consumer tiles that will wait on the cell).

    For the MoE example::

        event.trigger @expert_ready[i] range=@exp_indptr

    Means for each expert ``i``, the number of GroupGEMM tiles about
    to be triggered equals ``exp_indptr[i+1] - exp_indptr[i]``.
    """
    target = op.target
    event_name = target.event_ref.data
    range_name = op.trigger_range.data
    if event_name not in tensors:
        raise ValueError(f"event.trigger: target event tensor {event_name!r} not allocated")
    et = tensors[event_name]
    env = index_env or {}
    if range_name not in env:
        raise ValueError(f"event.trigger: range tensor {range_name!r} not found in index_env")
    indptr = env[range_name]
    indptr_shape = _tensor_shape(indptr, range_name)
    if len(indptr_shape) != 1:
        raise ValueError(
            f"event.trigger: range tensor {range_name!r} must be 1D (CSR-style prefix sum), got shape {indptr_shape}"
        )
    n_entries = indptr_shape[0] - 1
    if n_entries <= 0:
        raise ValueError(
            f"event.trigger: range tensor {range_name!r} must have >=2 elements "
            f"(it's a prefix sum), got len={indptr_shape[0]}"
        )

    indices_strs = tuple(s.data for s in target.indices.data)  # type: ignore[attr-defined]
    index_fn = _compile_index_fn(
        indices_strs,
        index_env=env,
        edge_event_name=event_name,
    )

    for i in range(n_entries):
        target_coord = index_fn((i,))
        lo = int(_read_scalar(indptr, (i,), range_name))
        hi = int(_read_scalar(indptr, (i + 1,), range_name))
        consumer_count = hi - lo
        if consumer_count < 0:
            raise ValueError(
                f"event.trigger: range[{i + 1}]={hi} < range[{i}]={lo} — prefix sum must be monotonic non-decreasing"
            )
        et.trigger(target_coord, consumer_count)


def _tensor_shape(t: Any, name: str) -> tuple[int, ...]:
    """Return ``tuple`` shape of a tensor-like object, with a clear error."""
    shape = getattr(t, "shape", None)
    if shape is None:
        raise ValueError(
            f"index_env[{name!r}] has no .shape attribute (got {type(t).__name__}); "
            f"pass a torch.Tensor / numpy.ndarray / similar"
        )
    return tuple(int(d) for d in shape)


def _read_scalar(t: Any, coord: tuple[int, ...], name: str) -> int:
    """Index into a tensor-like and return a Python int."""
    try:
        v = t
        for c in coord:
            v = v[c]
        return int(v.item() if hasattr(v, "item") else v)
    except Exception as exc:
        raise ValueError(f"failed to read index_env[{name!r}] at {coord}: {exc!r}") from exc


def _lower_event_tensor_op(op: EventTensorOp) -> EventTensor:
    """Allocate an :class:`EventTensor` from an ``event.event_tensor`` op."""
    name = op.sym_name.data
    shape_entries = op.event_type.shape.data
    shape = tuple(int(a.value.data) for a in shape_entries if isinstance(a, IntegerAttr))
    if len(shape) != len(shape_entries):
        raise ValueError(f"EventTensorOp {name!r}: non-integer shape entry in {shape_entries}")
    # Symbolic (-1) dims must be resolved before lowering; that's the
    # job of MaterializeViewOp which we don't support yet. Surface a
    # clear error instead of silently passing garbage through.
    if any(d <= 0 for d in shape):
        from compgen.runtime.errors import SymbolicShapeUnsupportedError

        raise SymbolicShapeUnsupportedError(
            f"EventTensorOp {name!r}: symbolic shape entry ({shape}) requires "
            f"event.materialize_view lowering (paper Fig. 4); not yet implemented"
        )

    wait_count = int(op.wait_count.value.data)
    dtype = op.event_type.counter_dtype.data
    scope = op.event_type.scope.data

    return EventTensor(
        shape=shape,
        wait_count_default=wait_count,
        dtype=dtype,
        scope=scope,
        sym_name=name,
    )


def _coord_attr_to_edge(
    coord_attr: EventCoordAttr,
    *,
    index_env: dict[str, Any] | None,
) -> EventEdge:
    """Convert an ``EventCoordAttr`` into a runtime :class:`EventEdge`."""
    event_ref = coord_attr.event_ref.data
    indices_strs = tuple(s.data for s in coord_attr.indices.data)  # type: ignore[attr-defined]
    decrement = int(coord_attr.decrement.value.data)
    index_fn = _compile_index_fn(
        indices_strs,
        index_env=index_env,
        edge_event_name=event_ref,
    )
    return EventEdge(event_name=event_ref, index_fn=index_fn, decrement=decrement)


def _lower_call_device_op(
    op: CallDeviceOp,
    *,
    device_funcs: dict[str, Callable[[tuple[int, ...]], None]],
    index_env: dict[str, Any] | None,
) -> DeviceCall:
    """Build a :class:`DeviceCall` from an ``event.call_device`` op."""
    # device_func is a SymbolRefAttr. CallDeviceOp uses a flat
    # (unnested) symbol name in practice — the compiler emits
    # @func_name rather than @module::@func_name.
    sym = op.device_func
    # xDSL's SymbolRefAttr stores the root reference via .root_reference
    # (a StringAttr) for the leaf name. Fall back to .string_value for
    # safety across xDSL versions.
    if hasattr(sym, "root_reference"):
        func_name = sym.root_reference.data  # type: ignore[attr-defined]
    else:
        func_name = sym.string_value()  # type: ignore[attr-defined]

    if func_name not in device_funcs:
        raise KeyError(
            f"CallDeviceOp references @{func_name} but no Python body_fn was "
            f"provided in `device_funcs`. Supply {func_name!r} -> Callable."
        )

    body_fn = device_funcs[func_name]

    # task_shape: ArrayAttr of IntegerAttr.
    task_shape_entries = op.task_shape.data
    task_shape: list[int] = []
    for ent in task_shape_entries:
        if not isinstance(ent, IntegerAttr):
            raise ValueError(f"CallDeviceOp @{func_name}: task_shape entry {ent!r} is not an IntegerAttr")
        dim_val = int(ent.value.data)
        if dim_val == -1:
            from compgen.runtime.errors import SymbolicShapeUnsupportedError

            raise SymbolicShapeUnsupportedError(
                f"CallDeviceOp @{func_name}: symbolic task_shape dim (-1) requires "
                f"event.materialize_view lowering (paper Fig. 4); not yet implemented"
            )
        if dim_val <= 0:
            raise ValueError(f"CallDeviceOp @{func_name}: task_shape dim {dim_val} must be >=1 or -1")
        task_shape.append(dim_val)

    in_edges: tuple[EventEdge, ...] = ()
    if op.in_edges is not None:
        in_edges = tuple(
            _coord_attr_to_edge(e, index_env=index_env) for e in op.in_edges.data if isinstance(e, EventCoordAttr)
        )

    out_edges: tuple[EventEdge, ...] = ()
    if op.out_edges is not None:
        out_edges = tuple(
            _coord_attr_to_edge(e, index_env=index_env) for e in op.out_edges.data if isinstance(e, EventCoordAttr)
        )

    return DeviceCall(
        name=func_name,
        body_fn=body_fn,
        task_shape=tuple(task_shape),
        in_edges=in_edges,
        out_edges=out_edges,
    )


# ---------------------------------------------------------------------------
# GraphOp / ModuleOp entry points
# ---------------------------------------------------------------------------


def lower_graph_op(
    graph_op: GraphOp,
    *,
    device_funcs: dict[str, Callable[[tuple[int, ...]], None]],
    event_tensors: dict[str, EventTensor] | None = None,
    index_env: dict[str, Any] | None = None,
) -> tuple[MegakernelGraph, dict[str, EventTensor]]:
    """Lower one ``event.graph`` op to a runtime :class:`MegakernelGraph`.

    Args:
        graph_op: The dialect op to lower.
        device_funcs: ``func_name -> callable`` registry that supplies the
            Python body for every ``CallDeviceOp`` in the graph. Every
            referenced function must be present or lowering raises
            :class:`KeyError`.
        event_tensors: Optional caller-provided event-tensor registry.
            When supplied, these tensors are reused instead of
            allocating fresh ones from the ``EventTensorOp`` attributes
            — useful when multiple graphs share events, or when the
            caller wants to pre-seed counters via ``UpdateOp`` emulation.
        index_env: Extra bindings threaded into the index_fn namespace
            (e.g. ``{"topk": runtime_topk_tensor}``). See
            :func:`_compile_index_fn`.

    Returns:
        Tuple ``(graph, event_tensors)``. The event-tensor dict is also
        available on ``graph.event_tensors`` — returned separately for
        callers that want to seed counters before ``launch()``.

    Raises:
        KeyError: A ``CallDeviceOp`` references a function not in
            ``device_funcs``.
        NotImplementedError: The graph contains ``UpdateOp``,
            ``TriggerOp``, ``MaterializeViewOp``, or a symbolic-shape
            ``EventTensorOp`` / ``CallDeviceOp``.
        ValueError: Malformed attributes (non-integer shape, bad
            index expression, etc.).
    """
    # First pass — allocate event tensors. EventTensorOp with symbolic
    # dims (shape entries <= 0) are deferred to a MaterializeViewOp;
    # see Phase 1.
    tensors = dict(event_tensors) if event_tensors else {}
    deferred_templates: dict[str, EventTensorOp] = {}
    for op in graph_op.body.block.ops:
        if isinstance(op, EventTensorOp):
            name = op.sym_name.data
            if name in tensors:
                # Caller-provided tensor wins (enables UpdateOp-style
                # pre-seeding). Validate the shape still matches to
                # catch silent mismatches.
                et = tensors[name]
                ir_shape = tuple(int(a.value.data) for a in op.event_type.shape.data if isinstance(a, IntegerAttr))
                # If the IR declares any symbolic dim (<= 0), accept the
                # caller-provided concrete shape without forcing equality.
                has_symbolic = any(d <= 0 for d in ir_shape)
                if not has_symbolic and et.shape != ir_shape:
                    raise ValueError(
                        f"caller-provided event_tensors[{name!r}] has shape {et.shape}; IR declares {ir_shape}"
                    )
            else:
                shape_entries = op.event_type.shape.data
                shape = tuple(int(a.value.data) for a in shape_entries if isinstance(a, IntegerAttr))
                if any(d <= 0 for d in shape):
                    # Symbolic — defer. Must be materialised by a
                    # MaterializeViewOp that names this symbol, or
                    # pre-seeded by the caller. Track so we can
                    # distinguish "missing materialize" from "never
                    # declared".
                    deferred_templates[name] = op
                else:
                    tensors[name] = _lower_event_tensor_op(op)
        elif isinstance(op, MaterializeViewOp):
            _apply_materialize_view(op, tensors, deferred_templates)
        elif isinstance(op, UpdateOp):
            _apply_update(op, tensors, index_env)
        elif isinstance(op, TriggerOp):
            _apply_trigger(op, tensors, index_env)
        elif isinstance(op, (NotifyOp, WaitOp)):
            # These live inside device function bodies, not the graph
            # region directly. If they show up here the IR is
            # malformed; better to fail loudly.
            raise ValueError(
                f"event.{op.name.split('.')[-1]} appears directly in graph "
                f"{graph_op.sym_name.data!r}'s body — expected inside a "
                f"func.func referenced by a CallDeviceOp"
            )

    # Any deferred symbolic templates must have been materialised by
    # a MaterializeViewOp — otherwise the graph is ill-formed.
    if deferred_templates:
        names = sorted(deferred_templates.keys())
        raise ValueError(
            f"event.graph {graph_op.sym_name.data!r}: symbolic event tensor(s) "
            f"{names} declared but never materialised. Add an "
            f"event.materialize_view op naming each symbol, or pre-seed the "
            f"tensors via event_tensors=."
        )

    # Second pass — build DeviceCalls.
    device_calls: list[DeviceCall] = []
    for op in graph_op.body.block.ops:
        if isinstance(op, CallDeviceOp):
            device_calls.append(
                _lower_call_device_op(
                    op,
                    device_funcs=device_funcs,
                    index_env=index_env,
                )
            )

    if not device_calls:
        raise ValueError(f"event.graph {graph_op.sym_name.data!r} contains no call_device ops")

    policy_str = graph_op.policy.policy.data
    sm_count: int | None = None
    if graph_op.sm_count is not None:
        sm_count = int(graph_op.sm_count.value.data)

    if policy_str not in ("static", "dynamic"):
        raise ValueError(
            f"event.graph {graph_op.sym_name.data!r}: unexpected policy {policy_str!r} (expected 'static' or 'dynamic')"
        )

    graph = MegakernelGraph(
        name=graph_op.sym_name.data,
        calls=tuple(device_calls),
        event_tensors=tensors,
        policy=policy_str,  # type: ignore[arg-type]
        sm_count=sm_count,
    )

    log.info(
        "event.lower.graph_op",
        graph=graph.name,
        policy=graph.policy,
        num_calls=len(graph.calls),
        num_events=len(tensors),
        sm_count=sm_count,
    )
    return graph, tensors


def lower_event_module(
    module: ModuleOp,
    *,
    device_funcs: dict[str, Callable[[tuple[int, ...]], None]],
    event_tensors: dict[str, EventTensor] | None = None,
    index_env: dict[str, Any] | None = None,
) -> dict[str, MegakernelGraph]:
    """Lower every ``event.graph`` op in ``module``.

    Scans the module's top-level ops, lowers each :class:`GraphOp`, and
    returns a ``{graph_name: MegakernelGraph}`` dict. The caller can
    invoke :meth:`MegakernelGraph.launch` on any of them in any order.

    All :class:`GraphOp`s share the supplied ``device_funcs`` /
    ``event_tensors`` / ``index_env``; graph-specific bindings are not
    supported here (split your module into separate compile units if
    that's needed).

    Args:
        module: The module to lower.
        device_funcs: See :func:`lower_graph_op`.
        event_tensors: See :func:`lower_graph_op`.
        index_env: See :func:`lower_graph_op`.

    Returns:
        ``{graph_name: MegakernelGraph}``. Empty if the module has no
        graph ops (that's legal — the module may contain other
        dialects' ops too).
    """
    graphs: dict[str, MegakernelGraph] = {}
    for op in module.body.block.ops:
        if isinstance(op, GraphOp):
            g, _ = lower_graph_op(
                op,
                device_funcs=device_funcs,
                event_tensors=event_tensors,
                index_env=index_env,
            )
            if g.name in graphs:
                raise ValueError(f"duplicate event.graph name {g.name!r} in module")
            graphs[g.name] = g
    return graphs


__all__ = ["lower_event_module", "lower_graph_op"]
