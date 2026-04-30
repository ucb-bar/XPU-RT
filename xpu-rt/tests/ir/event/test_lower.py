"""Tests for ir/event/lower.py — GraphOp → MegakernelGraph lowering.

Builds IR modules programmatically (no MLIR text roundtrip required),
lowers them via :func:`compgen.ir.event.lower.lower_event_module` and
:func:`compgen.ir.event.lower.lower_graph_op`, and verifies both:

- **Structural correctness** — EventTensor shapes, DeviceCall edges,
  policy/sm_count all come through.
- **Executional correctness** — the resulting MegakernelGraph runs
  the paper's canonical patterns (GEMM+RS, diamond DAG, MoE) end-to-
  end, matching what ``test_megakernel.py`` exercises on hand-built
  graphs.
- **Error paths** — unknown device_func, symbolic shapes, data-
  dependent ops all raise clearly.
"""

from __future__ import annotations

import threading

import pytest
import torch
from compgen.ir.event.attrs import (
    EventCoordAttr,
    EventTensorTypeAttr,
)
from compgen.ir.event.lower import lower_event_module, lower_graph_op
from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
    MaterializeViewOp,
    TriggerOp,
    UpdateOp,
)
from compgen.runtime.event_tensor import EventTensor
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region

# ---------------------------------------------------------------------------
# IR-construction helpers — local, keep tests self-contained
# ---------------------------------------------------------------------------


def _make_event_tensor_op(
    sym_name: str,
    shape: list[int],
    wait_count: int,
    dtype: str = "i32",
    scope: str = "device",
) -> EventTensorOp:
    return EventTensorOp.create(
        properties={
            "sym_name": StringAttr(sym_name),
            "event_type": EventTensorTypeAttr(
                shape=shape,
                dim_names=None,
                counter_dtype=dtype,
                scope=scope,
            ),
            "wait_count": IntegerAttr(wait_count, IntegerType(64)),
        }
    )


def _make_edge(
    event_name: str,
    indices: list[str],
    decrement: int = 1,
) -> EventCoordAttr:
    return EventCoordAttr(
        event_ref=event_name,
        indices=indices,
        decrement=decrement,
    )


def _make_call_device_op(
    func_name: str,
    task_shape: list[int],
    in_edges: list[EventCoordAttr] | None = None,
    out_edges: list[EventCoordAttr] | None = None,
) -> CallDeviceOp:
    props: dict = {
        "device_func": SymbolRefAttr(func_name),
        "task_shape": ArrayAttr([IntegerAttr(d, IntegerType(64)) for d in task_shape]),
    }
    if in_edges:
        props["in_edges"] = ArrayAttr(list(in_edges))
    if out_edges:
        props["out_edges"] = ArrayAttr(list(out_edges))
    return CallDeviceOp.create(properties=props)


def _wrap_in_module(graph_op: GraphOp) -> ModuleOp:
    module = ModuleOp([])
    module.body.block.add_op(graph_op)
    return module


def _make_graph_op(
    name: str,
    ops_in_body: list,
    policy: str = "static",
    sm_count: int | None = None,
) -> GraphOp:
    region = Region(Block(ops_in_body))
    return GraphOp(sym_name=name, policy=policy, sm_count=sm_count, body=region)


# ---------------------------------------------------------------------------
# Per-op lowering
# ---------------------------------------------------------------------------


def test_lower_event_tensor_op_allocates_runtime_tensor() -> None:
    """EventTensorOp → EventTensor preserves shape, wait_count, dtype, scope."""
    et_op = _make_event_tensor_op(
        sym_name="my_events",
        shape=[3, 4],
        wait_count=7,
        dtype="i64",
        scope="workgroup",
    )
    g_op = _make_graph_op(
        "g",
        [
            et_op,
            _make_call_device_op("body_fn", task_shape=[1]),
        ],
    )
    graph, tensors = lower_graph_op(g_op, device_funcs={"body_fn": lambda _c: None})

    assert "my_events" in tensors
    et = tensors["my_events"]
    assert isinstance(et, EventTensor)
    assert et.shape == (3, 4)
    assert et.wait_count_default == 7
    assert et.dtype == "i64"
    assert et.scope == "workgroup"
    assert et.sym_name == "my_events"
    assert graph.event_tensors["my_events"] is et


def test_lower_call_device_op_builds_edges() -> None:
    """CallDeviceOp → DeviceCall with parsed in/out edges."""
    et_op = _make_event_tensor_op("E", shape=[4], wait_count=2)
    cd_op = _make_call_device_op(
        "some_func",
        task_shape=[4, 2],
        in_edges=[_make_edge("E", ["i"])],
        out_edges=[_make_edge("E", ["i"], decrement=2)],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    graph, _ = lower_graph_op(g_op, device_funcs={"some_func": lambda _c: None})

    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call.name == "some_func"
    assert call.task_shape == (4, 2)
    assert len(call.in_edges) == 1
    assert call.in_edges[0].event_name == "E"
    assert call.in_edges[0].decrement == 1
    assert len(call.out_edges) == 1
    assert call.out_edges[0].event_name == "E"
    assert call.out_edges[0].decrement == 2


def test_graph_op_policy_and_sm_count_round_trip() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd_op = _make_call_device_op("f", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, cd_op], policy="dynamic", sm_count=13)

    graph, _ = lower_graph_op(g_op, device_funcs={"f": lambda _c: None})
    assert graph.name == "g"
    assert graph.policy == "dynamic"
    assert graph.sm_count == 13


# ---------------------------------------------------------------------------
# Index-expression compilation
# ---------------------------------------------------------------------------


def test_index_expression_identity_i() -> None:
    """indices=['i'] maps task_coord[0] to event cell."""
    et_op = _make_event_tensor_op("E", shape=[5], wait_count=1)
    cd_op = _make_call_device_op(
        "f",
        task_shape=[5],
        out_edges=[_make_edge("E", ["i"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    graph, _ = lower_graph_op(g_op, device_funcs={"f": lambda _c: None})
    edge = graph.calls[0].out_edges[0]
    assert edge.index_fn((3,)) == (3,)
    assert edge.index_fn((0,)) == (0,)


def test_index_expression_arithmetic_and_projections() -> None:
    """indices=['i*32+j'] and ['i'] both work: arithmetic and
    rank-reduction (paper's ``ij->i``)."""
    et_op = _make_event_tensor_op("E", shape=[1024], wait_count=1)
    cd_op = _make_call_device_op(
        "f",
        task_shape=[32, 32],
        out_edges=[_make_edge("E", ["i*32+j"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    graph, _ = lower_graph_op(g_op, device_funcs={"f": lambda _c: None})
    edge = graph.calls[0].out_edges[0]
    # i=3, j=5 → 3*32+5 = 101
    assert edge.index_fn((3, 5)) == (101,)


def test_index_expression_data_dependent_via_index_env() -> None:
    """index_env injects runtime tensors — paper's
    ``i->topk[i]`` / ``i->topk[i,:]`` pattern."""
    et_op = _make_event_tensor_op("E", shape=[4], wait_count=1)
    cd_op = _make_call_device_op(
        "f",
        task_shape=[8],
        out_edges=[_make_edge("E", ["topk[i]"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    topk = [2, 0, 3, 1, 2, 3, 0, 1]  # token i's expert is topk[i]
    graph, _ = lower_graph_op(
        g_op,
        device_funcs={"f": lambda _c: None},
        index_env={"topk": topk},
    )
    edge = graph.calls[0].out_edges[0]
    assert edge.index_fn((0,)) == (2,)
    assert edge.index_fn((3,)) == (1,)


def test_index_expression_bad_syntax_raises() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd_op = _make_call_device_op(
        "f",
        task_shape=[1],
        out_edges=[_make_edge("E", ["i + + j"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    with pytest.raises(ValueError, match="index expression"):
        lower_graph_op(g_op, device_funcs={"f": lambda _c: None})


def test_index_env_cannot_override_letters() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd_op = _make_call_device_op(
        "f",
        task_shape=[1],
        out_edges=[_make_edge("E", ["i"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    with pytest.raises(ValueError, match="position letter"):
        lower_graph_op(
            g_op,
            device_funcs={"f": lambda _c: None},
            index_env={"i": 999},
        )


def test_constant_index_expression_works_on_any_rank_task_grid() -> None:
    """A purely-constant index expression doesn't need any letter
    binding, so it works on any task-grid rank — even beyond the
    10-letter i..r window. This matters for tasks that fan into a
    single sink event regardless of their own coord."""
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    # Task shape larger than the letter set; constant expr so it's fine.
    cd_op = _make_call_device_op(
        "f",
        task_shape=[1] * 11,
        out_edges=[_make_edge("E", ["0"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    graph, _ = lower_graph_op(g_op, device_funcs={"f": lambda _c: None})
    edge = graph.calls[0].out_edges[0]
    # Evaluates regardless of task-coord rank.
    assert edge.index_fn(tuple([7] * 11)) == (0,)


def test_index_expression_referencing_unbound_letter_raises() -> None:
    """Expressions that reference a letter outside the i..r window
    (or a name missing from index_env) produce a clear error at
    evaluation time."""
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    # "s" is one past the 10-letter window. Task shape is too small to
    # bind it so the expression is unbound.
    cd_op = _make_call_device_op(
        "f",
        task_shape=[1],
        out_edges=[_make_edge("E", ["s"])],
    )
    g_op = _make_graph_op("g", [et_op, cd_op])
    # Graph construction probes the index_fn on the origin coord, so
    # the error surfaces here rather than at launch.
    with pytest.raises(ValueError, match="unbound"):
        lower_graph_op(g_op, device_funcs={"f": lambda _c: None})


# ---------------------------------------------------------------------------
# End-to-end execution — paper patterns through the IR → runtime path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_gemm_plus_reduce_scatter_lowered_from_ir(policy) -> None:
    """Paper's GEMM+RS: build an event.graph in IR, lower, execute.

    Event: row_events shape=(M,) wait_count=N.
    Call 1: gemm_tile, task_shape=(M, N), out_edges=[("row_events", "i")]
    Call 2: rs_row, task_shape=(M,), in_edges=[("row_events", "i")]
    """
    M, N = 4, 5
    produced = torch.zeros((M, N), dtype=torch.int32)
    consumed = torch.zeros(M, dtype=torch.int32)

    def gemm_body(coord):
        m, n = coord
        produced[m, n] = 1

    def rs_body(coord):
        (m,) = coord
        assert int(produced[m].sum().item()) == N
        consumed[m] = 1

    et_op = _make_event_tensor_op("row_events", shape=[M], wait_count=N)
    gemm_call = _make_call_device_op(
        "gemm_tile",
        task_shape=[M, N],
        out_edges=[_make_edge("row_events", ["i"])],
    )
    rs_call = _make_call_device_op(
        "rs_row",
        task_shape=[M],
        in_edges=[_make_edge("row_events", ["i"])],
    )
    g_op = _make_graph_op(
        "gemm_rs",
        [et_op, gemm_call, rs_call],
        policy=policy,
        sm_count=4,
    )
    module = _wrap_in_module(g_op)

    graphs = lower_event_module(
        module,
        device_funcs={"gemm_tile": gemm_body, "rs_row": rs_body},
    )
    assert "gemm_rs" in graphs
    graphs["gemm_rs"].launch(timeout_s=5.0)

    assert torch.equal(produced, torch.ones((M, N), dtype=torch.int32))
    assert torch.equal(consumed, torch.ones(M, dtype=torch.int32))


@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_diamond_dag_lowered_from_ir(policy) -> None:
    """Paper's diamond DAG: producer → workers → sink via two events,
    all composed in one event.graph."""
    produced = [False]
    workers_done = [0, 0]
    workers_lock = threading.Lock()
    sink_done = [False]

    def producer_body(_c):
        produced[0] = True

    def worker_body(coord):
        (i,) = coord
        assert produced[0]
        with workers_lock:
            workers_done[i] += 1

    def sink_body(_c):
        assert all(workers_done[i] > 0 for i in range(2))
        sink_done[0] = True

    eb_op = _make_event_tensor_op("E_b", shape=[2], wait_count=1)
    er_op = _make_event_tensor_op("E_r", shape=[1], wait_count=2)

    producer_cd = _make_call_device_op(
        "producer",
        task_shape=[1],
        out_edges=[
            _make_edge("E_b", ["0"]),
            _make_edge("E_b", ["1"]),
        ],
    )
    workers_cd = _make_call_device_op(
        "workers",
        task_shape=[2],
        in_edges=[_make_edge("E_b", ["i"])],
        out_edges=[_make_edge("E_r", ["0"])],
    )
    sink_cd = _make_call_device_op(
        "sink",
        task_shape=[1],
        in_edges=[_make_edge("E_r", ["0"])],
    )

    g_op = _make_graph_op(
        "diamond",
        [eb_op, er_op, producer_cd, workers_cd, sink_cd],
        policy=policy,
        sm_count=3,
    )
    module = _wrap_in_module(g_op)

    graphs = lower_event_module(
        module,
        device_funcs={
            "producer": producer_body,
            "workers": worker_body,
            "sink": sink_body,
        },
    )
    graphs["diamond"].launch(timeout_s=5.0)

    assert produced[0]
    assert workers_done == [1, 1]
    assert sink_done[0]


def test_moe_lowered_from_ir_with_topk_index_env() -> None:
    """Paper's MoE pattern: lowered via index_env for the topk lookup."""
    NUM_TOKENS = 8
    NUM_EXPERTS = 3
    torch.manual_seed(1)
    topk = torch.randint(low=0, high=NUM_EXPERTS, size=(NUM_TOKENS,)).tolist()

    histogram = [0] * NUM_EXPERTS
    for t in topk:
        histogram[t] += 1

    # Per-expert event tensors — caller pre-allocates with per-expert
    # wait counts (stands in for UpdateOp pre-seed).
    expert_events = {
        f"E{i}": EventTensor((1,), wait_count_default=max(1, histogram[i]), sym_name=f"E{i}")
        for i in range(NUM_EXPERTS)
    }
    for i, h in enumerate(histogram):
        if h == 0:
            expert_events[f"E{i}"].notify(0)

    tokens_seen = [set() for _ in range(NUM_EXPERTS)]
    lk = threading.Lock()

    def make_token_body(t: int):
        return lambda _c: (lk.__enter__(), tokens_seen[topk[t]].add(t), lk.__exit__(None, None, None))[1]

    def make_expert_body(i: int):
        def body(_c):
            expected = {t for t in range(NUM_TOKENS) if topk[t] == i}
            with lk:
                observed = set(tokens_seen[i])
            assert observed == expected

        return body

    # Build one EventTensorOp per expert (caller will reuse the
    # pre-allocated tensors). One CallDeviceOp per token (its single
    # out_edge writes to a constant event name). One CallDeviceOp per
    # expert waiting on its own event.
    ops_in_body: list = []
    for i in range(NUM_EXPERTS):
        ops_in_body.append(_make_event_tensor_op(f"E{i}", shape=[1], wait_count=max(1, histogram[i])))

    device_funcs = {}
    for t in range(NUM_TOKENS):
        name = f"token_{t}"
        device_funcs[name] = make_token_body(t)
        expert_name = f"E{topk[t]}"
        ops_in_body.append(
            _make_call_device_op(
                name,
                task_shape=[1],
                out_edges=[_make_edge(expert_name, ["0"])],
            )
        )
    for i in range(NUM_EXPERTS):
        name = f"expert_{i}"
        device_funcs[name] = make_expert_body(i)
        ops_in_body.append(
            _make_call_device_op(
                name,
                task_shape=[1],
                in_edges=[_make_edge(f"E{i}", ["0"])],
            )
        )

    g_op = _make_graph_op("moe", ops_in_body, policy="dynamic", sm_count=4)
    module = _wrap_in_module(g_op)

    graphs = lower_event_module(
        module,
        device_funcs=device_funcs,
        event_tensors=expert_events,  # caller-seeded
    )
    graphs["moe"].launch(timeout_s=5.0)

    for i in range(NUM_EXPERTS):
        expected = {t for t in range(NUM_TOKENS) if topk[t] == i}
        assert tokens_seen[i] == expected


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_device_func_raises_key_error() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd_op = _make_call_device_op("not_provided", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, cd_op])
    with pytest.raises(KeyError, match="not_provided"):
        lower_graph_op(g_op, device_funcs={})


def test_symbolic_shape_event_tensor_without_materialize_view_fails() -> None:
    """A symbolic-shape EventTensorOp must be paired with a
    MaterializeViewOp that names the same symbol — else the graph is
    ill-formed because no EventTensor gets allocated and downstream
    ``notify``/``wait`` targets don't resolve.

    The honest error is "no tensor for this event name" — not a
    silent pass. Phase 1 closes the `SymbolicShapeUnsupportedError`
    path that used to raise here."""
    et_op = _make_event_tensor_op("E", shape=[-1], wait_count=1)
    # Deliberately omit the MaterializeViewOp.
    cd_op = _make_call_device_op("f", task_shape=[1], in_edges=[("E", ["0"])])
    g_op = _make_graph_op("g", [et_op, cd_op])
    with pytest.raises(ValueError):
        lower_graph_op(g_op, device_funcs={"f": lambda _c: None})


def test_symbolic_task_shape_raises_not_implemented() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd_op = _make_call_device_op("f", task_shape=[-1])
    g_op = _make_graph_op("g", [et_op, cd_op])
    with pytest.raises(NotImplementedError, match="symbolic"):
        lower_graph_op(g_op, device_funcs={"f": lambda _c: None})


def test_update_op_rewrites_event_counters_from_index_env() -> None:
    """Paper Fig. 5b first half: ``event.update E[i] from topk expr=i``
    rewrites each cell of ``E`` to the value of ``topk[i]`` before the
    graph runs. Validates the Phase-1 Python-reference lowering."""
    import torch

    et_op = _make_event_tensor_op("E", shape=[4], wait_count=0)
    update_op = UpdateOp.create(
        properties={
            "target": _make_edge("E", ["i"]),
            "source_tensor": StringAttr("topk"),
            "index_expr": StringAttr("i"),
        }
    )
    cd_op = _make_call_device_op("f", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, update_op, cd_op])
    topk = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    graph, tensors = lower_graph_op(
        g_op,
        device_funcs={"f": lambda _c: None},
        index_env={"topk": topk},
    )
    et = tensors["E"]
    # Each cell's counter should now be exp_topk[i].
    assert et.load((0,)) == 2
    assert et.load((1,)) == 0
    assert et.load((2,)) == 3
    assert et.load((3,)) == 1


def test_trigger_op_seeds_counters_from_prefix_sum() -> None:
    """Paper Fig. 5b second half: ``event.trigger E[i] range=exp_indptr``
    sets ``E[i]`` to ``exp_indptr[i+1] - exp_indptr[i]`` (the count of
    consumer tiles each expert will wait for)."""
    import torch

    et_op = _make_event_tensor_op("E", shape=[3], wait_count=0)
    trigger_op = TriggerOp.create(
        properties={
            "target": _make_edge("E", ["i"]),
            "trigger_range": StringAttr("exp_indptr"),
        }
    )
    cd_op = _make_call_device_op("f", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, trigger_op, cd_op])
    # CSR-style prefix sum: 3 experts, counts [2, 5, 1].
    indptr = torch.tensor([0, 2, 7, 8], dtype=torch.int64)
    _graph, tensors = lower_graph_op(
        g_op,
        device_funcs={"f": lambda _c: None},
        index_env={"exp_indptr": indptr},
    )
    et = tensors["E"]
    assert et.load((0,)) == 2
    assert et.load((1,)) == 5
    assert et.load((2,)) == 1


def test_materialize_view_op_concretises_symbolic_event_tensor() -> None:
    """Paper Fig. 4: a symbolic-shape EventTensorOp is deferred until a
    matching MaterializeViewOp supplies the concrete shape."""
    et_op = _make_event_tensor_op("E", shape=[-1], wait_count=1)
    mv_op = MaterializeViewOp.create(
        properties={
            "event_ref": StringAttr("E"),
            "concrete_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
        }
    )
    cd_op = _make_call_device_op("f", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, mv_op, cd_op])
    _graph, tensors = lower_graph_op(g_op, device_funcs={"f": lambda _c: None})
    et = tensors["E"]
    assert et.shape == (4,)
    # Default wait_count honoured.
    assert et.load((0,)) == 1
    assert et.load((3,)) == 1


def test_graph_with_no_call_device_op_raises() -> None:
    et_op = _make_event_tensor_op("E", shape=[1], wait_count=1)
    g_op = _make_graph_op("g", [et_op])
    with pytest.raises(ValueError, match="no call_device"):
        lower_graph_op(g_op, device_funcs={})


def test_caller_provided_tensor_shape_mismatch_raises() -> None:
    et_op = _make_event_tensor_op("E", shape=[4], wait_count=1)
    cd_op = _make_call_device_op("f", task_shape=[1])
    g_op = _make_graph_op("g", [et_op, cd_op])
    # Supply a tensor of wrong shape.
    prebuilt = {"E": EventTensor((3,), wait_count_default=1)}
    with pytest.raises(ValueError, match="shape"):
        lower_graph_op(
            g_op,
            device_funcs={"f": lambda _c: None},
            event_tensors=prebuilt,
        )


# ---------------------------------------------------------------------------
# Module-level lowering
# ---------------------------------------------------------------------------


def test_lower_event_module_handles_multiple_graphs() -> None:
    """Two event.graph ops in one module — each lowers to its own graph."""
    et_a = _make_event_tensor_op("E_a", shape=[1], wait_count=1)
    cd_a = _make_call_device_op("f_a", task_shape=[1])
    g_a = _make_graph_op("graph_a", [et_a, cd_a], policy="static")

    et_b = _make_event_tensor_op("E_b", shape=[1], wait_count=1)
    cd_b = _make_call_device_op("f_b", task_shape=[1])
    g_b = _make_graph_op("graph_b", [et_b, cd_b], policy="dynamic")

    module = ModuleOp([])
    module.body.block.add_op(g_a)
    module.body.block.add_op(g_b)

    a_ran = [False]
    b_ran = [False]
    graphs = lower_event_module(
        module,
        device_funcs={
            "f_a": lambda _c: a_ran.__setitem__(0, True),
            "f_b": lambda _c: b_ran.__setitem__(0, True),
        },
    )
    assert set(graphs) == {"graph_a", "graph_b"}
    graphs["graph_a"].launch(timeout_s=2.0)
    graphs["graph_b"].launch(timeout_s=2.0)
    assert a_ran[0] and b_ran[0]


def test_lower_event_module_rejects_duplicate_graph_names() -> None:
    et1 = _make_event_tensor_op("E", shape=[1], wait_count=1)
    cd1 = _make_call_device_op("f", task_shape=[1])
    g1 = _make_graph_op("dup", [et1, cd1])

    et2 = _make_event_tensor_op("E2", shape=[1], wait_count=1)
    cd2 = _make_call_device_op("f", task_shape=[1])
    g2 = _make_graph_op("dup", [et2, cd2])

    module = ModuleOp([])
    module.body.block.add_op(g1)
    module.body.block.add_op(g2)

    with pytest.raises(ValueError, match="duplicate"):
        lower_event_module(module, device_funcs={"f": lambda _c: None})


def test_lower_event_module_empty_returns_empty() -> None:
    """A module with no event.graph ops just returns an empty dict
    (legal — the module may have non-event ops too)."""
    module = ModuleOp([])
    graphs = lower_event_module(module, device_funcs={})
    assert graphs == {}
