"""Load QNN converter JSON artifacts into ``MemoryPlanInput`` buffers.

The QNN converter emits a sidecar ``*_net.json`` next to every
``.cpp``/``.bin`` model. It is the most accessible representation of
the post-conversion dataflow graph: ``graph.tensors`` enumerates every
tensor (input, output, intermediate, static parameter) with its dims
and dtype enum, and ``graph.nodes`` enumerates every op with its
``input_names`` / ``output_names`` referencing the tensor dict. From
those two we can derive:

* ``size_bytes`` per tensor — product of dims times bit-width of the
  QNN ``data_type`` enum.
* ``lifetime_start`` / ``lifetime_end`` per activation — first op
  whose ``output_names`` produces it, last op whose ``input_names``
  consumes it.
* Alias candidates — for each elementwise op whose input has no other
  consumer, the input tensor's lifetime ends at the op and the output
  tensor's lifetime starts at the op, so they are disjoint and can
  share storage.

This is the **dronet_net.json / yolov8n_net.json path**. The .cpp/.bin
and .dlc artifacts are not parsed; the JSON is a complete superset of
what we need for memory planning.

Example:
    >>> plan = extract_buffer_specs(Path("/tmp/qnn_build/dronet_net.json"))
    >>> len(plan.buffers)
    27
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from xpu_rt.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
)

logger = structlog.get_logger(__name__)


# QNN data_type enum -> bytes-per-element. Sourced from
# QnnTypes.h::Qnn_DataType_t. The most common ones we expect in
# converter output:
#   0x232 = 562  QNN_DATATYPE_FLOAT_32           (4 bytes)
#   0x216 = 534  QNN_DATATYPE_FLOAT_16           (2 bytes)
#   0x408 = 1032 QNN_DATATYPE_UFIXED_POINT_8     (1 byte)
#   0x416 = 1046 QNN_DATATYPE_UFIXED_POINT_16    (2 bytes)
#   0x308 = 776  QNN_DATATYPE_SFIXED_POINT_8     (1 byte)
#   0x316 = 790  QNN_DATATYPE_SFIXED_POINT_16    (2 bytes)
#   0x008 = 8    QNN_DATATYPE_INT_8              (1 byte)
#   0x016 = 22   QNN_DATATYPE_INT_16             (2 bytes)
#   0x032 = 50   QNN_DATATYPE_INT_32             (4 bytes)
#   0x108 = 264  QNN_DATATYPE_UINT_8             (1 byte)
#   0x132 = 306  QNN_DATATYPE_UINT_32            (4 bytes)
#   0x508 = 1288 QNN_DATATYPE_BOOL_8             (1 byte)
_QNN_DTYPE_BYTES: dict[int, int] = {
    562: 4,
    534: 2,
    1032: 1,
    1046: 2,
    776: 1,
    790: 2,
    8: 1,
    22: 2,
    50: 4,
    264: 1,
    306: 4,
    1288: 1,
}

# Op types whose output can plausibly alias their only input. The
# planner re-checks lifetime disjointness before activating an alias,
# so the candidate list is a hint, not a guarantee: extra entries are
# safe (just ignored), missing entries are missed opportunities.
#
# We allow:
#   * Elementwise unary (ReLU / Sigmoid / Tanh / generic activations)
#     — same shape, in-place safe.
#   * Pure data-movement (Reshape, Transpose, StridedSlice) — the
#     output occupies the same or smaller storage than the input.
#   * Pool / Softmax / single-output normalizations — semantics permit
#     in-place when the consumer count is exactly one.
_ALIAS_SAFE_OPS: frozenset[str] = frozenset(
    {
        "ElementWiseNeuron",
        "Eltwise_Unary",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Reshape",
        "Transpose",
        "StridedSlice",
        "Pool",
        "Softmax",
        "Batchnorm",
    }
)


@dataclass(frozen=True)
class ExtractionStats:
    """Summary of what was pulled from a QNN net.json file."""

    source_path: str
    parser: str
    num_ops: int
    num_activations: int
    num_static_params: int
    skipped_unknown_dtype: int
    alias_candidates_proposed: int
    total_activation_bytes: int
    max_activation_bytes: int


def _dtype_bytes(code: int) -> int | None:
    """Return bytes-per-element for a QNN data_type enum, or ``None`` if unknown."""

    return _QNN_DTYPE_BYTES.get(int(code))


def _numel(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def _build_producer_consumer(
    nodes: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[int]], list[str]]:
    """Index nodes by integer order; build producer + consumer maps.

    Args:
        nodes: ``graph.nodes`` dict from a QNN net.json.

    Returns:
        ``(producer, consumers, ordered_op_names)`` where ``producer``
        maps tensor name to the op index that emits it (if any) and
        ``consumers`` maps tensor name to the list of op indices that
        read it.
    """

    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    ordered: list[str] = []
    for op_idx, (name, ndata) in enumerate(nodes.items()):
        ordered.append(name)
        for out_name in ndata.get("output_names", []) or []:
            producer[out_name] = op_idx
        for in_name in ndata.get("input_names", []) or []:
            consumers.setdefault(in_name, []).append(op_idx)
    return producer, consumers, ordered


def _propose_aliases(
    nodes: dict[str, dict[str, Any]],
    consumers: dict[str, list[int]],
    activation_ids: set[str],
) -> list[AliasCandidate]:
    """Propose alias pairs for elementwise ops with single-use inputs.

    An elementwise op consumes ``in_tensor`` and writes ``out_tensor``
    at the same op index. If ``in_tensor`` has no other consumer, its
    lifetime ends at this op and ``out_tensor``'s lifetime starts at
    this op — they are disjoint and can share storage.
    """

    proposals: list[AliasCandidate] = []
    for op_idx, (_, ndata) in enumerate(nodes.items()):
        if ndata.get("type") not in _ALIAS_SAFE_OPS:
            continue
        ins = [n for n in (ndata.get("input_names") or []) if n in activation_ids]
        outs = [n for n in (ndata.get("output_names") or []) if n in activation_ids]
        if len(ins) != 1 or len(outs) != 1:
            continue
        in_t = ins[0]
        out_t = outs[0]
        # The input must be consumed ONLY by this op (single-use).
        if consumers.get(in_t, []) != [op_idx]:
            continue
        proposals.append(AliasCandidate(buffer_a=in_t, buffer_b=out_t))
    return proposals


def _default_tier_capacities(total_activation_bytes: int) -> tuple[TierCapacity, ...]:
    """Pick scratch/dram tier sizes that fit the model.

    We don't have a real target profile here, so we synthesise two
    tiers sized to the workload: a 1 MB scratch (representative of
    Hexagon HMX TCM at ~512KB-2MB depending on SKU) and a DRAM tier
    sized to comfortably hold the entire activation working set.
    """

    KB = 1024
    MB = 1024 * KB
    scratch = 1 * MB  # representative TCM
    dram = max(64 * MB, total_activation_bytes * 4)
    return (
        TierCapacity(tier_id="scratch", capacity_bytes=scratch, weight=1.0),
        TierCapacity(tier_id="dram", capacity_bytes=dram, weight=4.0),
    )


def extract_buffer_specs(
    model_path: Path | str,
    *,
    include_static_params: bool = False,
    propose_aliases: bool = True,
    objective_lambda: float = 1e-9,
    time_budget_ms: int = 60_000,
    tier_capacities: tuple[TierCapacity, ...] | None = None,
) -> tuple[MemoryPlanInput, ExtractionStats]:
    """Extract a ``MemoryPlanInput`` from a QNN ``*_net.json`` file.

    Parser path: ``qnn_net_json_v1``. We do not parse ``.cpp``,
    ``.bin``, or ``.dlc`` — the converter's sidecar JSON carries all
    the dataflow information needed for memory planning.

    Args:
        model_path: Path to a ``*_net.json`` file emitted by
            ``qnn-onnx-converter`` (or ``qnn-pytorch-converter``).
        include_static_params: When ``True``, include weights and bias
            constants (``type == 4``) as buffers with full-graph
            lifetimes. Most planners treat statics as resident in DRAM
            and exclude them from the activation working set; the
            default is ``False``.
        propose_aliases: When ``True``, walk elementwise ops and
            propose alias candidates whose lifetimes are guaranteed
            disjoint. The MILP planner is then free to activate them.
        objective_lambda: Weight on per-tier peak in the MILP
            objective. ``0.0`` means "only minimize spill cost"; a
            small positive value (default ``1e-9``) tips the planner
            toward compact packings without overwhelming the spill
            term.
        time_budget_ms: Per-solve budget passed to the planner.
        tier_capacities: Override the synthesised scratch/dram tiers.

    Returns:
        ``(plan_input, stats)`` — the input ready to hand to either
        ``plan_memory`` or ``plan_memory_greedy``, and a stats record
        for the experiment report.
    """

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"QNN net.json not found: {path}")
    if path.suffix != ".json":
        raise ValueError(
            f"extract_buffer_specs expects a QNN *_net.json; got {path.suffix}"
        )

    with path.open() as f:
        doc = json.load(f)

    graph = doc.get("graph")
    if not isinstance(graph, dict):
        raise ValueError(f"{path}: missing 'graph' object — not a QNN net.json?")
    tensors_dict: dict[str, dict[str, Any]] = graph.get("tensors") or {}
    nodes_dict: dict[str, dict[str, Any]] = graph.get("nodes") or {}
    if not tensors_dict or not nodes_dict:
        raise ValueError(f"{path}: graph has no tensors or nodes")

    producer, consumers, _ordered_ops = _build_producer_consumer(nodes_dict)
    n_ops = len(nodes_dict)
    last_op_idx = n_ops - 1

    bufs: list[BufferSpec] = []
    activation_ids: set[str] = set()
    skipped_unknown_dtype = 0
    n_static = 0
    total_act = 0
    max_act = 0

    for tname, tval in tensors_dict.items():
        ttype = tval.get("type")
        dtype_code = tval.get("data_type")
        dims = tval.get("dims") or []
        bpe = _dtype_bytes(dtype_code) if isinstance(dtype_code, int) else None
        if bpe is None:
            skipped_unknown_dtype += 1
            continue
        size_bytes = _numel(dims) * bpe
        if size_bytes <= 0:
            continue

        # Type 0 = network input, 1 = network output, 3 = intermediate
        # activation, 4 = static parameter.
        if ttype == 4:
            n_static += 1
            if not include_static_params:
                continue
            # Statics live the whole graph; force them into DRAM.
            bufs.append(
                BufferSpec(
                    buffer_id=f"const::{tname}",
                    size_bytes=size_bytes,
                    lifetime_start=0,
                    lifetime_end=last_op_idx,
                    allowed_tiers=("dram",),
                    alignment=64,
                    spill_cost=0.5,
                )
            )
            continue

        # Lifetime: producer .. last consumer.
        # We encode op steps in half-units so a tensor that is
        # *produced at op k* and *last-read at op k* (in-place
        # elementwise) has end < the produce-step of its successor.
        # Concretely: read-phase of op k = 2k, write-phase of op k =
        # 2k+1. A tensor's lifetime is then
        #   [2 * producer + 1, 2 * last_consumer]
        # — it is "alive" starting just after its producer writes,
        # and ends just as its last consumer reads.  Two tensors
        # whose only overlap is one being produced as the other is
        # last-consumed (the classic in-place case) become disjoint.
        # Inputs (type 0) have no producer — pin start at 0.
        # Outputs (type 1) have no consumer — pin end at last write.
        p_idx = producer.get(tname)
        c_idxs = consumers.get(tname, [])
        if p_idx is None:
            start = 0  # graph input / defensive
        else:
            start = 2 * int(p_idx) + 1
        if c_idxs:
            end = 2 * int(max(c_idxs))
        elif ttype == 1:
            end = 2 * last_op_idx + 1
        else:
            end = start  # producer-only tensor (rare)
        if end < start:
            end = start

        activation_ids.add(tname)
        total_act += size_bytes
        max_act = max(max_act, size_bytes)
        bufs.append(
            BufferSpec(
                buffer_id=tname,
                size_bytes=size_bytes,
                lifetime_start=start,
                lifetime_end=end,
                allowed_tiers=("scratch", "dram"),
                alignment=64,
                spill_cost=2.0 if ttype == 3 else 4.0,
            )
        )

    if not bufs:
        raise ValueError(f"{path}: no activation buffers extracted")

    aliases: list[AliasCandidate] = []
    if propose_aliases:
        aliases = _propose_aliases(nodes_dict, consumers, activation_ids)

    tiers = tier_capacities or _default_tier_capacities(total_act)

    plan = MemoryPlanInput(
        buffers=tuple(bufs),
        tier_capacities=tiers,
        alias_candidates=tuple(aliases),
        objective_lambda=objective_lambda,
        time_budget_ms=time_budget_ms,
    )

    stats = ExtractionStats(
        source_path=str(path),
        parser="qnn_net_json_v1",
        num_ops=n_ops,
        num_activations=len(activation_ids),
        num_static_params=n_static,
        skipped_unknown_dtype=skipped_unknown_dtype,
        alias_candidates_proposed=len(aliases),
        total_activation_bytes=total_act,
        max_activation_bytes=max_act,
    )

    logger.info(
        "qnn_net_json_extracted",
        path=str(path),
        ops=n_ops,
        activations=len(activation_ids),
        statics=n_static,
        unknown_dtype=skipped_unknown_dtype,
        aliases=len(aliases),
        total_act_bytes=total_act,
    )

    return plan, stats


__all__ = [
    "ExtractionStats",
    "extract_buffer_specs",
]
