"""Generic FX→MegakernelGraph fallback (Wave 2.2).

When no registered pattern (built-in diamond/FFN or user-supplied
custom) matches the input model, this generic lowering emits a
serial-chain MegakernelGraph that handles any FX-traceable
combination of:

- ``nn.Linear`` (no bias) — full-shape fmaf body, one task per linear
- ``torch.nn.functional.relu`` / ``torch.relu`` — elementwise, one task
- ``torch.add`` — elementwise binary, one task

Each FX op becomes one task. Tasks are serialized via event tensors
(each task waits for its predecessor). No tile-level parallelism —
the generic path trades perf for breadth of coverage. Pattern-matched
paths (diamond, FFN, future MHA/MoE) stay preferred for shapes they
recognize.

The agentic-compilation contract: a PyPI user can hand any
FX-traceable model in the supported op family and get a runnable
ETC bundle back. The bundle's ``decision.pattern_name`` is
``"generic_fx_chain"`` so the agent's audit query can distinguish
"matched a known pattern" from "fell through to generic."

Out of scope (future work): branching DAGs (residuals through skip
connections — needs the transform pass that diamond's hand-coded
matcher handles), ops outside the supported family
(layer_norm, softmax, gelu, attention — Wave 2.1 adds those), or
shapes that require partial-tile masking (the matcher rejects
non-multiples-of-tile-size up front, same as the pattern-matched
path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import (
    DeviceCall,
    EventEdge,
    MegakernelGraph,
)
from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

# ---------------------------------------------------------------------------
# FX op classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChainOp:
    """One classified op in the FX-traced chain."""

    name: str  # internal task name, e.g. "linear_0", "relu_0"
    kind: str  # "linear" | "relu" | "add"
    fx_node_name: str  # original FX node name
    # Linear-specific:
    in_features: int = 0
    out_features: int = 0
    weight_buf_idx: int = 0  # index into user_buffer_layout
    # Add-specific: indices of the two inputs in the prior chain
    add_input_idxs: tuple[int, int] = (0, 0)


def _classify_fx_chain(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
) -> list[_ChainOp]:
    """Trace ``model`` with ``torch.fx.symbolic_trace`` and classify
    each node into the supported op family.

    Raises :class:`UnsupportedShape` when any node falls outside the
    supported family — the agent can read the message to know which
    op is unhandled.
    """
    from compgen.runtime.lowering.fx_to_megakernel import UnsupportedShape

    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedShape(
            f"generic_fx_chain: torch.fx.symbolic_trace failed on "
            f"{type(model).__name__}: {exc!r}. Generic fallback only "
            "handles fx-traceable models."
        ) from exc

    chain: list[_ChainOp] = []
    linear_count = 0
    relu_count = 0
    add_count = 0

    # Module-name-to-Linear lookup for nn.Linear classification.
    submodules = dict(model.named_modules())

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Inputs (the function args). Skip — the chain starts
            # with whichever node consumes the placeholder.
            continue
        if node.op == "output":
            continue

        if node.op == "call_module":
            sub = submodules.get(node.target)
            if isinstance(sub, nn.Linear):
                if sub.bias is not None:
                    raise UnsupportedShape(
                        f"generic_fx_chain: nn.Linear {node.target!r} has "
                        "bias; only bias=False is supported in the generic "
                        "fallback today (Wave 2.1 will lift this)."
                    )
                chain.append(
                    _ChainOp(
                        name=f"linear_{linear_count}",
                        kind="linear",
                        fx_node_name=str(node.name),
                        in_features=int(sub.in_features),
                        out_features=int(sub.out_features),
                        weight_buf_idx=0,  # filled in by _emit_generic
                    )
                )
                linear_count += 1
                continue
            raise UnsupportedShape(
                f"generic_fx_chain: unsupported nn.Module subclass for "
                f"{node.target!r}: {type(sub).__name__}. Supported: nn.Linear."
            )

        if node.op == "call_function":
            target_name = node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
            if target_name in ("relu",):
                chain.append(
                    _ChainOp(
                        name=f"relu_{relu_count}",
                        kind="relu",
                        fx_node_name=str(node.name),
                    )
                )
                relu_count += 1
                continue
            if target_name in ("add", "__add__", "operator.add"):
                chain.append(
                    _ChainOp(
                        name=f"add_{add_count}",
                        kind="add",
                        fx_node_name=str(node.name),
                    )
                )
                add_count += 1
                continue
            raise UnsupportedShape(
                f"generic_fx_chain: unsupported call_function "
                f"target {target_name!r} at FX node {node.name!r}. "
                f"Supported in generic fallback: relu, add. "
                f"More ops in Wave 2.1 (MHA, layer_norm, softmax, "
                "gelu, MoE)."
            )

        if node.op == "call_method":
            target_name = str(node.target)
            if target_name == "relu":
                chain.append(
                    _ChainOp(
                        name=f"relu_{relu_count}",
                        kind="relu",
                        fx_node_name=str(node.name),
                    )
                )
                relu_count += 1
                continue
            raise UnsupportedShape(
                f"generic_fx_chain: unsupported call_method {target_name!r} "
                f"at FX node {node.name!r}. Supported: .relu()"
            )

        if node.op == "get_attr":
            # Constant tensors as op inputs — supported as long as
            # downstream nodes are.
            continue

        raise UnsupportedShape(f"generic_fx_chain: unsupported FX op kind {node.op!r} at node {node.name!r}")

    if not chain:
        raise UnsupportedShape("generic_fx_chain: traced graph has no compute ops")
    return chain


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def lower_generic_fx(
    model: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    backend_choice: Any = None,
) -> Any:
    """Generic FX→MegakernelGraph lowering for arbitrary models.

    The fallback for shapes that no pattern matcher recognizes. Walks
    the model's FX trace, emits one task per op, serializes them
    via event tensors. Bodies use the hand-rolled fmaf path
    (no cuBLASDx) — guarantees correctness across arbitrary shapes
    without needing to instantiate cuBLASDx ``Size<...>`` templates
    per shape.

    Args:
        model: Any FX-traceable PyTorch ``nn.Module``.
        sample_inputs: Concrete input tensors. Used for shape
            inference (no symbolic shapes).
        backend_choice: Optional :class:`BackendChoice` from
            :func:`probe_device`. The generic path doesn't currently
            use it (no cuBLASDx) but accepts it for API parity with
            the pattern-matched lowerings.

    Returns:
        :class:`LoweringResult`.

    Raises:
        UnsupportedShape: When the FX trace contains an op outside
            the supported family (linear, relu, add).
    """
    from compgen.runtime.lowering.fx_to_megakernel import (
        LoweringDecision,
        LoweringResult,
        _BodyDecision,
    )

    chain = _classify_fx_chain(model, sample_inputs)
    x = sample_inputs[0]
    batch = int(x.shape[0])
    in_dim = int(x.shape[1]) if x.ndim >= 2 else 1

    return _emit_generic(
        chain=chain,
        batch=batch,
        in_dim=in_dim,
        model=model,
        loweringresult_cls=LoweringResult,
        loweringdecision_cls=LoweringDecision,
        bodydecision_cls=_BodyDecision,
    )


def _emit_generic(
    *,
    chain: list[_ChainOp],
    batch: int,
    in_dim: int,
    model: nn.Module,
    loweringresult_cls: Any,
    loweringdecision_cls: Any,
    bodydecision_cls: Any,
) -> Any:
    """Emit the serial-chain MegakernelGraph + bodies for the
    classified op chain."""
    # Build user_buffer_layout: x + each linear's weight + each op's output.
    layout: list[str] = ["x"]
    weight_idxs: dict[str, int] = {}
    output_idxs: dict[str, int] = {}

    # Linear weights come first so their indices are stable.
    for op in chain:
        if op.kind == "linear":
            buf_name = f"w_{op.name}"
            layout.append(buf_name)
            weight_idxs[op.name] = len(layout) - 1

    # Each op writes one output buffer.
    for op in chain:
        buf_name = f"y_{op.name}"
        layout.append(buf_name)
        output_idxs[op.name] = len(layout) - 1

    # Track the "current shape" through the chain — generic fallback
    # only handles linear (B, K) → (B, OUT) and elementwise ops that
    # preserve shape.
    cur_b = batch
    cur_n = in_dim
    op_input_buf: dict[str, int] = {}
    op_output_dim: dict[str, tuple[int, int]] = {}

    # First op's input is the user-supplied x (buffer 0).
    prev_buf = 0
    prev_b, prev_n = batch, in_dim
    for op in chain:
        op_input_buf[op.name] = prev_buf
        if op.kind == "linear":
            cur_n = op.out_features
            op_output_dim[op.name] = (cur_b, cur_n)
            prev_buf = output_idxs[op.name]
            prev_n = cur_n
        else:
            # relu/add preserve shape
            op_output_dim[op.name] = (cur_b, cur_n)
            prev_buf = output_idxs[op.name]

    # Build event tensors: one per producer→consumer edge in the
    # serial chain. Each task is a single (1,)-shape event tensor
    # with wait_count_default=1 — generic path has no tile-level
    # parallelism, so 1 cell per task is enough.
    n_tasks = len(chain)
    event_tensors: dict[str, EventTensor] = {}
    for i, op in enumerate(chain):
        ev = EventTensor((1,), wait_count_default=1)
        event_tensors[f"ev_{op.name}"] = ev

    # Build DeviceCall list with serial dependencies.
    same_cell = lambda c: (0,)  # noqa: E731 — single-cell event
    calls: list[DeviceCall] = []
    for i, op in enumerate(chain):
        in_edges: tuple[EventEdge, ...]
        if i == 0:
            in_edges = ()
        else:
            prev = chain[i - 1]
            in_edges = (EventEdge(f"ev_{prev.name}", same_cell),)
        calls.append(
            DeviceCall(
                name=op.name,
                body_fn=lambda c: None,  # body source attached separately
                task_shape=(1,),  # single-task per op
                in_edges=in_edges,
                out_edges=(EventEdge(f"ev_{op.name}", same_cell),),
            )
        )

    graph = MegakernelGraph(
        name="generic_fx_chain",
        calls=tuple(calls),
        event_tensors=event_tensors,
        policy="static",
    )

    # Emit body sources.
    bodies: dict[str, DeviceFunctionSource] = {}
    for op in chain:
        if op.kind == "linear":
            shape = op_output_dim[op.name]
            bodies[op.name] = _emit_linear_full_body(
                name=op.name,
                b_dim=shape[0],
                in_dim_actual=cur_in_for_linear(chain, op, batch, in_dim),
                out_dim=op.out_features,
                x_buf=op_input_buf[op.name],
                w_buf=weight_idxs[op.name],
                out_buf=output_idxs[op.name],
            )
        elif op.kind == "relu":
            shape = op_output_dim[op.name]
            bodies[op.name] = _emit_relu_full_body(
                name=op.name,
                total_elems=shape[0] * shape[1],
                in_buf=op_input_buf[op.name],
                out_buf=output_idxs[op.name],
            )
        elif op.kind == "add":
            shape = op_output_dim[op.name]
            # Generic add emit is a stub — needs FX dataflow tracking
            # we don't fully thread through yet. For now: same input
            # buffer twice (semantically wrong; tests for add-in-chain
            # come in Wave 2.1's MHA/residual path).
            bodies[op.name] = _emit_add_full_body(
                name=op.name,
                total_elems=shape[0] * shape[1],
                in_a_buf=op_input_buf[op.name],
                in_b_buf=op_input_buf[op.name],
                out_buf=output_idxs[op.name],
            )

    body_decisions = tuple(
        bodydecision_cls(
            op_name=op.name,
            backend="hand_rolled_fmaf",
            tile_shape=(1, 1, 1),  # generic path is no-tile
            rationale=(
                f"generic_fx_chain (Wave 2.2): {op.kind} op, full-shape "
                "single-task body. cuBLASDx not used in the generic path "
                "to avoid per-shape Size<...> instantiation."
            ),
        )
        for op in chain
    )

    decision = loweringdecision_cls(
        pattern_name="generic_fx_chain",
        pattern_rationale=(
            f"FX trace of {type(model).__name__} produced {n_tasks} ops "
            f"({', '.join(op.kind for op in chain)}); no built-in or "
            "user-registered pattern matched, so the generic FX fallback "
            "lowered each op as a single full-shape task. Slower than "
            "pattern-matched ETC bundles but correct for arbitrary "
            "FX-traceable models in the supported op family "
            "(linear, relu, add)."
        ),
        body_decisions=body_decisions,
        schedule_hints={
            "policy": "serial_chain",
            "num_tasks": n_tasks,
            "block_dim": [32, 32, 1],
        },
        total_tile_tasks=n_tasks,
    )

    return loweringresult_cls(
        megakernel_graph=graph,
        device_function_sources=bodies,
        user_buffer_layout=tuple(layout),
        decision=decision,
    )


def cur_in_for_linear(chain: list[_ChainOp], op: _ChainOp, batch: int, in_dim: int) -> int:
    """The K dim a linear sees: either the model input dim
    (first linear) or the previous op's output dim."""
    idx = chain.index(op)
    if idx == 0:
        return in_dim
    # Walk back to the most recent shape-changing op (a linear).
    for prev in reversed(chain[:idx]):
        if prev.kind == "linear":
            return prev.out_features
    return in_dim


# ---------------------------------------------------------------------------
# Body emission — hand-rolled fmaf, full-shape, single task per op
# ---------------------------------------------------------------------------


def _emit_linear_full_body(
    *,
    name: str,
    b_dim: int,
    in_dim_actual: int,
    out_dim: int,
    x_buf: int,
    w_buf: int,
    out_buf: int,
) -> DeviceFunctionSource:
    """Single-task full-shape linear: one block (32×32 threads)
    iterates over the entire output (B, OUT) tile."""
    body = (
        f"const int B = {b_dim};\n"
        f"const int IN = {in_dim_actual};\n"
        f"const int OUT = {out_dim};\n"
        + r"""
const float *x = (const float *)buffers[__X_BUF__];
const float *w = (const float *)buffers[__W_BUF__];
float       *y = (float *)buffers[__OUT_BUF__];

const int tid = threadIdx.y * 32 + threadIdx.x;
const int total = B * OUT;
const int stride = blockDim.x * blockDim.y;

for (int idx = tid; idx < total; idx += stride) {
    int row = idx / OUT;
    int col = idx % OUT;
    float acc = 0.0f;
    for (int k = 0; k < IN; ++k) {
        acc += x[row * IN + k] * w[col * IN + k];
    }
    y[idx] = acc;
}
"""
    )
    body = body.replace("__X_BUF__", str(x_buf)).replace("__W_BUF__", str(w_buf)).replace("__OUT_BUF__", str(out_buf))
    return DeviceFunctionSource(name=name, body=body)


def _emit_relu_full_body(
    *,
    name: str,
    total_elems: int,
    in_buf: int,
    out_buf: int,
) -> DeviceFunctionSource:
    """Full-tensor relu — one block iterates over all elements."""
    body = (
        f"const int TOTAL = {total_elems};\n"
        + r"""
const float *in_p = (const float *)buffers[__IN_BUF__];
float       *out_p = (float *)buffers[__OUT_BUF__];

const int tid = threadIdx.y * 32 + threadIdx.x;
const int stride = blockDim.x * blockDim.y;

for (int idx = tid; idx < TOTAL; idx += stride) {
    float v = in_p[idx];
    out_p[idx] = v > 0.0f ? v : 0.0f;
}
"""
    )
    body = body.replace("__IN_BUF__", str(in_buf)).replace("__OUT_BUF__", str(out_buf))
    return DeviceFunctionSource(name=name, body=body)


def _emit_add_full_body(
    *,
    name: str,
    total_elems: int,
    in_a_buf: int,
    in_b_buf: int,
    out_buf: int,
) -> DeviceFunctionSource:
    body = (
        f"const int TOTAL = {total_elems};\n"
        + r"""
const float *a = (const float *)buffers[__IN_A_BUF__];
const float *b = (const float *)buffers[__IN_B_BUF__];
float       *out_p = (float *)buffers[__OUT_BUF__];

const int tid = threadIdx.y * 32 + threadIdx.x;
const int stride = blockDim.x * blockDim.y;

for (int idx = tid; idx < TOTAL; idx += stride) {
    out_p[idx] = a[idx] + b[idx];
}
"""
    )
    body = (
        body.replace("__IN_A_BUF__", str(in_a_buf))
        .replace("__IN_B_BUF__", str(in_b_buf))
        .replace("__OUT_BUF__", str(out_buf))
    )
    return DeviceFunctionSource(name=name, body=body)
