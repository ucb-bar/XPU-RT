"""QNN real-cost workload helpers for Experiment 7 / 7b.

Loaders and DAG builders that turn the profiled per-op cost matrix
(`xpu-rt/data/profiled/qnn_cost_matrix.json`) into schedulable
``SyntheticDag``-shaped tuples for the greedy / CP-SAT / MOSEK
solvers.

Topology choice
---------------
The cost matrix carries no explicit dependency edges. We therefore
build a **chain DAG** in QNN execution order (the order the ops appear
in the matrix dict). This under-estimates parallelism — real YOLOv8 has
residual fan-outs — and that bias is documented at the call site.

A ``k_lookahead`` relaxation is offered: op ``i`` depends on
``i-1, i-2, ..., i-k`` (so ``k=1`` is the strict chain). Larger ``k``
still forbids reorderings beyond ``k`` positions but admits limited
parallel placement across backends.

Transfer cost
-------------
Cross-backend transfer is hard to derive from
``qrb5165_costs.json`` per-op (the dequant/quant fits are
elements-per-microsecond and we lack per-op tensor sizes here). We
use a fixed ``100 µs`` cross-backend penalty (zero on diagonal) — this
matches the order of magnitude observed in the ``dequant_quant`` fits
for the smallest activations in the YOLO stem and is conservative
enough to surface real bottlenecks at the K=273 granularity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")
"""Stable backend ordering: device index 0=CPU, 1=GPU, 2=DSP."""

DEFAULT_TRANSFER_US: float = 100.0
"""Fixed cross-backend penalty (µs). On-diagonal entries are 0."""


@dataclass(frozen=True)
class QnnDag:
    """Schedulable QNN DAG (shape-compatible with the synthetic DAGs).

    Attributes:
        partition_ids: Topological order of partition IDs.
        durations_us_by_device: ``pid -> [duration on CPU, GPU, DSP]``.
            ``None`` marks the op as unsupported on that backend
            (treated as infeasible by the solvers).
        dependencies: ``pid -> [predecessor ids]``.
        num_devices: Always ``len(BACKENDS)`` (=3 here).
        transfer_us: ``num_devices x num_devices`` µs transfer matrix.
        name: Workload label (e.g., ``yolov8n_k1``).
        backends: Stable backend ordering (== :data:`BACKENDS`).
    """

    partition_ids: list[str]
    durations_us_by_device: dict[str, list[float | None]]
    dependencies: dict[str, list[str]]
    num_devices: int
    transfer_us: list[list[float]]
    name: str
    backends: tuple[str, ...] = field(default=BACKENDS)


def load_cost_matrix(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load the profiled per-op cost matrix.

    Args:
        path: Path to ``qnn_cost_matrix.json``.

    Returns:
        Dict mapping workload name → op name → backend → µs cost.
        The top-level ``_meta`` key is stripped.

    Raises:
        FileNotFoundError: If the cost matrix file is missing.
        ValueError: If the schema version is not ``qnn_cost_matrix_v1``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"cost matrix not found: {p}")
    with p.open() as fh:
        raw = json.load(fh)
    meta = raw.get("_meta", {})
    if meta.get("schema_version") != "qnn_cost_matrix_v1":
        raise ValueError(
            f"unsupported schema_version in {p}: {meta.get('schema_version')!r}"
        )
    return {k: v for k, v in raw.items() if k != "_meta"}


def _default_transfer_matrix(num_devices: int, penalty_us: float) -> list[list[float]]:
    m = [[0.0] * num_devices for _ in range(num_devices)]
    for i in range(num_devices):
        for j in range(num_devices):
            if i != j:
                m[i][j] = float(penalty_us)
    return m


def make_chain_dag(
    workload_name: str,
    cost_matrix: dict[str, dict[str, dict[str, float]]] | None = None,
    *,
    cost_matrix_path: str | Path | None = None,
    k_lookahead: int = 1,
    transfer_us_value: float = DEFAULT_TRANSFER_US,
) -> QnnDag:
    """Build a chain (or k-lookahead relaxed) DAG for a real QNN workload.

    Args:
        workload_name: ``yolov8n`` or ``dronet`` — must exist in the
            cost matrix.
        cost_matrix: Pre-loaded cost matrix (top-level keys are
            workload names). If ``None``, ``cost_matrix_path`` is read.
        cost_matrix_path: Optional path to load if ``cost_matrix`` is
            ``None``. Defaults to the in-repo profiled file.
        k_lookahead: Dependency relaxation. ``k=1`` produces a pure
            chain (op ``i`` depends on ``i-1``). ``k=4`` lets ops
            reorder within a 4-position window. Must be ``>= 1``.
        transfer_us_value: Cross-backend transfer penalty in µs. Used
            for every off-diagonal entry of the transfer matrix.

    Returns:
        A :class:`QnnDag` with ``num_devices=3`` (CPU, GPU, DSP) and
        per-op durations in µs. Unsupported (op, backend) cells appear
        as ``None`` in ``durations_us_by_device`` and are interpreted
        as infeasible by all solvers.

    Raises:
        KeyError: If ``workload_name`` is absent.
        ValueError: If ``k_lookahead < 1``.
    """
    if k_lookahead < 1:
        raise ValueError(f"k_lookahead must be >= 1, got {k_lookahead}")

    if cost_matrix is None:
        if cost_matrix_path is None:
            cost_matrix_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "profiled"
                / "qnn_cost_matrix.json"
            )
        cost_matrix = load_cost_matrix(cost_matrix_path)

    if workload_name not in cost_matrix:
        raise KeyError(
            f"workload {workload_name!r} not in cost matrix; have {sorted(cost_matrix)}"
        )

    ops = cost_matrix[workload_name]
    pids = list(ops.keys())
    durations: dict[str, list[float | None]] = {}
    for pid in pids:
        row = ops[pid]
        durations[pid] = [
            (float(row[b]) if row.get(b) is not None else None) for b in BACKENDS
        ]

    deps: dict[str, list[str]] = {}
    for i, pid in enumerate(pids):
        start = max(0, i - k_lookahead)
        deps[pid] = list(pids[start:i])  # may be empty for i == 0

    transfer = _default_transfer_matrix(len(BACKENDS), transfer_us_value)
    name = f"{workload_name}_k{k_lookahead}"
    logger.debug(
        "qnn_chain_dag_built",
        workload=workload_name,
        n_ops=len(pids),
        k_lookahead=k_lookahead,
        transfer_us=transfer_us_value,
    )
    return QnnDag(
        partition_ids=pids,
        durations_us_by_device=durations,
        dependencies=deps,
        num_devices=len(BACKENDS),
        transfer_us=transfer,
        name=name,
    )


def chunk_dag(dag: QnnDag, n_chunks: int) -> QnnDag:
    """Collapse a DAG into ``n_chunks`` topologically-consecutive groups.

    Each chunk's cost on a backend is the sum of constituent op costs
    on that backend. If **any** constituent op is unsupported on a
    backend, the whole chunk is marked unsupported on that backend
    (``None``) — promoting a single missing cell would otherwise hide
    real placement constraints.

    Chunk-level dependencies inherit from op-level dependencies:
    chunk ``c`` depends on chunk ``c'`` iff any op in ``c`` had a
    predecessor that landed in ``c'`` (and ``c' != c``). For a chain
    input this produces a chunk-level chain. For a ``k_lookahead``
    relaxed input it produces the minimal-fan-in chunk dependency
    graph.

    Args:
        dag: The op-granularity DAG to chunk.
        n_chunks: Target number of chunks. Must be in ``[1, n_ops]``.
            The op-to-chunk mapping is roughly even (``ceil`` for the
            first ``n_ops % n_chunks`` chunks).

    Returns:
        A new :class:`QnnDag` with ``n_chunks`` partitions. The
        ``name`` field is suffixed with ``_chunk{n_chunks}``.

    Raises:
        ValueError: If ``n_chunks`` is out of range or the input DAG
            is empty.
    """
    pids = list(dag.partition_ids)
    n = len(pids)
    if n == 0:
        raise ValueError("cannot chunk empty DAG")
    if not 1 <= n_chunks <= n:
        raise ValueError(f"n_chunks must be in [1, {n}], got {n_chunks}")

    if n_chunks == n:
        # Identity chunking; just relabel to a normalised name.
        return QnnDag(
            partition_ids=list(dag.partition_ids),
            durations_us_by_device=dict(dag.durations_us_by_device),
            dependencies={k: list(v) for k, v in dag.dependencies.items()},
            num_devices=dag.num_devices,
            transfer_us=[row[:] for row in dag.transfer_us],
            name=f"{dag.name}_chunk{n_chunks}",
            backends=dag.backends,
        )

    base = n // n_chunks
    rem = n % n_chunks
    op_to_chunk: dict[str, int] = {}
    chunk_members: list[list[str]] = []
    cursor = 0
    for c in range(n_chunks):
        size = base + (1 if c < rem else 0)
        members = pids[cursor : cursor + size]
        cursor += size
        chunk_members.append(members)
        for pid in members:
            op_to_chunk[pid] = c

    chunk_ids = [f"chunk_{c:03d}" for c in range(n_chunks)]
    chunk_durations: dict[str, list[float | None]] = {}
    for c, members in enumerate(chunk_members):
        per_dev: list[float | None] = []
        for d in range(dag.num_devices):
            total = 0.0
            unsupported = False
            for pid in members:
                v = dag.durations_us_by_device[pid][d]
                if v is None:
                    unsupported = True
                    break
                total += float(v)
            per_dev.append(None if unsupported else total)
        chunk_durations[chunk_ids[c]] = per_dev

    chunk_deps: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
    seen_edges: set[tuple[int, int]] = set()
    for succ_pid, pred_pids in dag.dependencies.items():
        c_succ = op_to_chunk[succ_pid]
        for pred_pid in pred_pids:
            c_pred = op_to_chunk[pred_pid]
            if c_pred == c_succ:
                continue
            edge = (c_pred, c_succ)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            chunk_deps[chunk_ids[c_succ]].append(chunk_ids[c_pred])

    logger.debug(
        "qnn_chunked_dag_built",
        source=dag.name,
        n_ops=n,
        n_chunks=n_chunks,
        chunk_edges=len(seen_edges),
    )
    return QnnDag(
        partition_ids=chunk_ids,
        durations_us_by_device=chunk_durations,
        dependencies=chunk_deps,
        num_devices=dag.num_devices,
        transfer_us=[row[:] for row in dag.transfer_us],
        name=f"{dag.name}_chunk{n_chunks}",
        backends=dag.backends,
    )


def chunk_dag_from_chunks(dag: QnnDag, chunks: list) -> QnnDag:
    """Materialize a chunked :class:`QnnDag` from an explicit chunk list.

    Sibling of :func:`chunk_dag` for the specialty-driven granularity
    path: instead of slicing into ``n_chunks`` even segments, the caller
    provides a list of :class:`xpu_rt.scheduling.granularity.Chunk`
    objects (or any object with ``chunk_id``, ``op_ids``, and
    ``durations_us_by_backend`` attributes) describing the desired
    partitioning.

    Args:
        dag: The op-granularity DAG the chunks were built from.
        chunks: Topologically ordered chunk objects whose ``op_ids``
            partition ``dag.partition_ids`` exactly (no overlap, no
            missing op).

    Returns:
        A new :class:`QnnDag` with one partition per chunk. Per-backend
        durations come from each chunk's ``durations_us_by_backend``
        (mapped onto :data:`BACKENDS` ordering); ``math.inf`` entries
        are mapped to ``None`` (the QnnDag convention for "infeasible
        on this backend").

    Raises:
        ValueError: If ``chunks`` doesn't partition ``dag.partition_ids``
            or is empty.
    """
    import math as _math

    if not chunks:
        raise ValueError("chunks list must not be empty")

    seen: dict[str, int] = {}
    for c_idx, c in enumerate(chunks):
        for pid in c.op_ids:
            if pid in seen:
                raise ValueError(
                    f"op {pid!r} appears in chunks {seen[pid]} and {c_idx}"
                )
            seen[pid] = c_idx
    expected = set(dag.partition_ids)
    if set(seen) != expected:
        missing = expected - set(seen)
        extra = set(seen) - expected
        raise ValueError(
            f"chunk op_ids do not partition dag.partition_ids "
            f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
        )

    chunk_ids = [c.chunk_id for c in chunks]
    chunk_durations: dict[str, list[float | None]] = {}
    for c in chunks:
        per_dev: list[float | None] = []
        for b in dag.backends:
            v = c.durations_us_by_backend.get(b)
            if v is None or (isinstance(v, float) and _math.isinf(v)):
                per_dev.append(None)
            else:
                per_dev.append(float(v))
        chunk_durations[c.chunk_id] = per_dev

    # Build chunk-level dependency graph from op-level edges that cross
    # chunk boundaries (matches chunk_dag's behaviour).
    op_to_chunk: dict[str, int] = seen
    chunk_deps: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
    seen_edges: set[tuple[int, int]] = set()
    for succ_pid, pred_pids in dag.dependencies.items():
        c_succ = op_to_chunk[succ_pid]
        for pred_pid in pred_pids:
            c_pred = op_to_chunk[pred_pid]
            if c_pred == c_succ:
                continue
            edge = (c_pred, c_succ)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            chunk_deps[chunk_ids[c_succ]].append(chunk_ids[c_pred])

    logger.debug(
        "qnn_chunked_dag_built_from_chunks",
        source=dag.name,
        n_ops=len(dag.partition_ids),
        n_chunks=len(chunks),
        chunk_edges=len(seen_edges),
    )
    return QnnDag(
        partition_ids=chunk_ids,
        durations_us_by_device=chunk_durations,
        dependencies=chunk_deps,
        num_devices=dag.num_devices,
        transfer_us=[row[:] for row in dag.transfer_us],
        name=f"{dag.name}_specialty{len(chunks)}",
        backends=dag.backends,
    )


__all__ = [
    "BACKENDS",
    "DEFAULT_TRANSFER_US",
    "QnnDag",
    "chunk_dag",
    "chunk_dag_from_chunks",
    "load_cost_matrix",
    "make_chain_dag",
]
