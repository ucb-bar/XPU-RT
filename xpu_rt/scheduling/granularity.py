"""Specialty-driven granularity heuristic for the compile-time scheduler.

Walks an op-level DAG in topological order and groups consecutive ops that
share a preferred backend (the *specialty matrix*, derived from
:mod:`xpu_rt.audit.cost_table_audit`). The output is a sequence of
:class:`Chunk` objects with summed per-backend durations and a preferred
backend label, suitable for handoff to a coarser-grained scheduler.

Why specialty-driven chunking instead of fixed K
-------------------------------------------------
Experiment V1 (cost-table audit, see ``build/experiments/exp10_full_audit``)
showed that op-family backend specialty is workload-dependent: only the
convolution family transfers robustly across workloads. Rather than picking
a fixed chunk count, we let the empirical specialty *of this workload*
decide chunk boundaries, then apply solver-policy caps on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from xpu_rt.audit.cost_table_audit import canonical_family, matrix_op_family
from xpu_rt.scheduler.qnn_real_workload import QnnDag

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A topologically consecutive group of ops sharing a preferred backend.

    Attributes:
        chunk_id: Stable label (``chunk_000``, ``chunk_001``, ...).
        op_ids: The constituent op IDs in topological order.
        preferred_backend: Backend the chunk argmin-prefers across its ops.
            ``"UNKNOWN"`` if no family in the chunk had a specialty entry.
        durations_us_by_backend: Sum of constituent per-op costs per backend.
            A backend missing from any op in the chunk is mapped to
            ``math.inf`` (infeasible at chunk granularity — promoting a
            single missing cell would hide a real placement constraint).
    """

    chunk_id: str
    op_ids: tuple[str, ...]
    preferred_backend: str
    durations_us_by_backend: dict[str, float]


@dataclass(frozen=True)
class GranularityPlan:
    """Specialty-driven chunking decision for one workload.

    Attributes:
        workload_id: The workload these chunks belong to.
        chunks: Topologically ordered chunk sequence.
        specialty_summary: ``op_family → preferred backend`` (post-V1 fold).
        n_partitions: ``len(chunks)`` (named to mirror scheduler vocabulary).
    """

    workload_id: str
    chunks: tuple[Chunk, ...]
    specialty_summary: dict[str, str]
    n_partitions: int


_DEFAULT_BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")


def _family_for_op(op_id: str) -> str:
    """Canonical (ONNX → QNN folded) op family name for an op ID."""

    return canonical_family(matrix_op_family(op_id))


def compute_specialty_matrix(
    cost_matrix: dict,
    workload_id: str,
    *,
    backends: tuple[str, ...] = _DEFAULT_BACKENDS,
) -> dict[str, str]:
    """Argmin preferred backend per op family for one workload.

    Mirrors :func:`xpu_rt.audit.cost_table_audit.family_backend_specialty`
    but takes the raw ``workload_id → op_id → backend → cost`` dict
    (the shape returned by
    :func:`xpu_rt.scheduler.qnn_real_workload.load_cost_matrix`).

    Args:
        cost_matrix: Cost matrix, either the raw mapping ``{workload_id:
            {op_id: {backend: us}}}`` or the audit-side
            ``{workload_id: list[MatrixOp]}`` mapping. Both shapes are
            accepted to keep the helper drop-in across call sites.
        workload_id: Workload key to look up.
        backends: Backends to consider when picking the argmin.

    Returns:
        ``{op_family: preferred_backend}``. Families with no row that
        has all backends measured are absent (callers should default to
        ``"UNKNOWN"``).

    Raises:
        KeyError: If ``workload_id`` is not present in ``cost_matrix``.
    """

    if workload_id not in cost_matrix:
        raise KeyError(
            f"workload {workload_id!r} not in cost matrix; "
            f"have {sorted(cost_matrix)}"
        )

    workload = cost_matrix[workload_id]

    # Support both the raw dict and the MatrixOp-list shape.
    rows: list[tuple[str, dict[str, float]]]
    if isinstance(workload, dict):
        rows = [(op_id, costs) for op_id, costs in workload.items()]
    else:
        rows = [(row.op_id, row.costs) for row in workload]

    by_family_counts: dict[str, dict[str, int]] = {}
    by_family_total: dict[str, int] = {}
    for op_id, costs in rows:
        if not all(b in costs and costs[b] is not None for b in backends):
            continue
        family = _family_for_op(op_id)
        argmin_backend = min(backends, key=lambda b: costs[b])
        by_family_counts.setdefault(family, {})
        by_family_counts[family][argmin_backend] = (
            by_family_counts[family].get(argmin_backend, 0) + 1
        )
        by_family_total[family] = by_family_total.get(family, 0) + 1

    specialty: dict[str, str] = {}
    for family, counts in by_family_counts.items():
        # Tie-break deterministically: highest count then backend name.
        winner = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        specialty[family] = winner

    log.debug(
        "specialty_matrix_computed",
        workload=workload_id,
        n_families=len(specialty),
        n_ops_considered=sum(by_family_total.values()),
    )
    return specialty


def _chunk_durations(
    op_ids: tuple[str, ...],
    dag: QnnDag,
) -> dict[str, float]:
    """Sum per-op durations across each backend; ``inf`` if any op is missing.

    Using ``math.inf`` (rather than dropping the key) keeps the dict shape
    stable across chunks — the scheduler can treat ``inf`` uniformly as
    "infeasible on this backend".
    """

    backends = dag.backends
    out: dict[str, float] = {b: 0.0 for b in backends}
    infeasible: set[str] = set()
    for pid in op_ids:
        row = dag.durations_us_by_device.get(pid)
        if row is None:
            # Should not happen for in-DAG ids, but guard anyway.
            for b in backends:
                infeasible.add(b)
            continue
        for i, b in enumerate(backends):
            v = row[i] if i < len(row) else None
            if v is None:
                infeasible.add(b)
            else:
                out[b] += float(v)
    for b in infeasible:
        out[b] = math.inf
    return out


def _argmin_backend(durations: dict[str, float]) -> str:
    """Pick the cheapest backend; ``"UNKNOWN"`` if all are infeasible."""

    finite = {b: v for b, v in durations.items() if math.isfinite(v)}
    if not finite:
        return "UNKNOWN"
    return min(finite.items(), key=lambda kv: kv[1])[0]


def propose_chunks(
    dag: QnnDag,
    cost_matrix: dict,
    workload_id: str,
    specialty: dict[str, str],
    *,
    max_chunk_ops: int = 16,
    max_partitions: int = 200,
) -> GranularityPlan:
    """Walk the DAG and group consecutive same-specialty ops into chunks.

    Args:
        dag: Op-granularity DAG (its ``partition_ids`` give topo order).
        cost_matrix: Raw cost matrix; kept for signature symmetry with
            other planner helpers (durations are read off ``dag``).
        workload_id: Workload label propagated to the returned plan.
        specialty: ``op_family → preferred_backend`` (from
            :func:`compute_specialty_matrix`). Families absent from the
            dict are treated as ``"UNKNOWN"`` and form their own chunks.
        max_chunk_ops: Hard cap on chunk size; a chunk is closed when it
            reaches this many ops even if the next op shares the
            preference.
        max_partitions: Hard cap on total chunk count. Natural chunking
            that exceeds this is collapsed by repeatedly merging the
            smallest adjacent same-backend pair (then, if forced, the
            smallest adjacent pair regardless of backend).

    Returns:
        A :class:`GranularityPlan` with chunks in topological order.

    Raises:
        ValueError: If ``max_chunk_ops`` or ``max_partitions`` is < 1,
            or if ``dag`` has no ops.
    """

    if max_chunk_ops < 1:
        raise ValueError(f"max_chunk_ops must be >= 1, got {max_chunk_ops}")
    if max_partitions < 1:
        raise ValueError(f"max_partitions must be >= 1, got {max_partitions}")
    if not dag.partition_ids:
        raise ValueError("cannot propose chunks for empty DAG")

    # Phase 1: natural specialty walk.
    natural_groups: list[tuple[str, list[str]]] = []
    current_ops: list[str] = []
    current_backend: str | None = None
    for pid in dag.partition_ids:
        family = _family_for_op(pid)
        pref = specialty.get(family, "UNKNOWN")
        if (
            current_backend is None
            or (pref == current_backend and len(current_ops) < max_chunk_ops)
        ):
            current_ops.append(pid)
            current_backend = pref
        else:
            natural_groups.append((current_backend or "UNKNOWN", current_ops))
            current_ops = [pid]
            current_backend = pref
    if current_ops:
        natural_groups.append((current_backend or "UNKNOWN", current_ops))

    # Phase 2: cap the partition count by merging.
    natural_groups = _enforce_partition_cap(
        natural_groups,
        dag=dag,
        max_chunk_ops=max_chunk_ops,
        max_partitions=max_partitions,
    )

    # Phase 3: materialize Chunks.
    chunks: list[Chunk] = []
    for idx, (backend, ops) in enumerate(natural_groups):
        op_tuple = tuple(ops)
        durations = _chunk_durations(op_tuple, dag)
        # If the natural specialty backend is now infeasible (a forced
        # cross-backend merge happened), fall back to argmin of the
        # summed durations so the label stays informative.
        if backend == "UNKNOWN" or not math.isfinite(
            durations.get(backend, math.inf)
        ):
            backend = _argmin_backend(durations)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{idx:03d}",
                op_ids=op_tuple,
                preferred_backend=backend,
                durations_us_by_backend=durations,
            )
        )

    log.debug(
        "chunks_proposed",
        workload=workload_id,
        n_chunks=len(chunks),
        n_ops=len(dag.partition_ids),
        max_chunk_ops=max_chunk_ops,
        max_partitions=max_partitions,
    )
    return GranularityPlan(
        workload_id=workload_id,
        chunks=tuple(chunks),
        specialty_summary=dict(specialty),
        n_partitions=len(chunks),
    )


def _enforce_partition_cap(
    groups: list[tuple[str, list[str]]],
    *,
    dag: QnnDag,
    max_chunk_ops: int,
    max_partitions: int,
) -> list[tuple[str, list[str]]]:
    """Merge adjacent chunks until ``len(groups) <= max_partitions``.

    Strategy:
        1. Prefer to merge the smallest adjacent *same-backend* pair
           (lowest cumulative op count). This keeps specialty signal.
        2. If no same-backend mergers exist, merge the smallest adjacent
           pair regardless. The merged group inherits ``"UNKNOWN"`` —
           the chunk-materialization phase will assign a backend by
           re-running argmin on the summed durations.

    ``max_chunk_ops`` is intentionally *not* re-enforced here: violating
    it locally is the lesser evil compared to exceeding the partition
    cap that the downstream solver requires.
    """

    while len(groups) > max_partitions:
        # Find the smallest same-backend adjacent pair.
        best_idx = -1
        best_size = math.inf
        for i in range(len(groups) - 1):
            if groups[i][0] != groups[i + 1][0]:
                continue
            sz = len(groups[i][1]) + len(groups[i + 1][1])
            if sz < best_size:
                best_size = sz
                best_idx = i
        if best_idx < 0:
            # No same-backend pair; merge the smallest adjacent regardless.
            best_idx = 0
            best_size = len(groups[0][1]) + len(groups[1][1])
            for i in range(1, len(groups) - 1):
                sz = len(groups[i][1]) + len(groups[i + 1][1])
                if sz < best_size:
                    best_size = sz
                    best_idx = i
            new_backend = "UNKNOWN"
        else:
            new_backend = groups[best_idx][0]
        merged_ops = groups[best_idx][1] + groups[best_idx + 1][1]
        groups = (
            groups[:best_idx]
            + [(new_backend, merged_ops)]
            + groups[best_idx + 2 :]
        )
    return groups


def should_fuse(
    a: Chunk,
    b: Chunk,
    *,
    transfer_us: float,
    fusion_gain_threshold: float = 0.3,
) -> bool:
    """Decide whether two adjacent chunks should be fused.

    Args:
        a: Predecessor chunk.
        b: Successor chunk.
        transfer_us: Cross-backend transfer cost incurred between ``a``
            and ``b`` if they stay on different backends. Should be 0
            (or near 0) if both chunks already share a backend.
        fusion_gain_threshold: Fraction of ``b``'s serial cost that
            transfer must exceed to trigger fusion. Default 0.3 — fuse
            if the cross-backend hop eats ≥30% of ``b``'s minimum
            serial duration.

    Returns:
        ``True`` iff fusion is recommended (same backend, or transfer
        dominates ``b``'s minimum-backend serial cost).
    """

    if a.preferred_backend == b.preferred_backend and a.preferred_backend != "UNKNOWN":
        return True
    finite_b = [v for v in b.durations_us_by_backend.values() if math.isfinite(v)]
    if not finite_b:
        # ``b`` is infeasible everywhere; fusing won't make it worse.
        return True
    min_b = min(finite_b)
    if min_b <= 0:
        return transfer_us > 0
    return transfer_us >= fusion_gain_threshold * min_b


def apply_fusion(
    plan: GranularityPlan,
    *,
    transfer_matrix: list[list[float]] | None = None,
    backends: tuple[str, ...] = _DEFAULT_BACKENDS,
    fusion_gain_threshold: float = 0.3,
) -> GranularityPlan:
    """One-pass left-to-right fusion of adjacent chunks under transfer cost.

    Args:
        plan: Plan to refine. Returned unchanged if ``transfer_matrix``
            is ``None`` (the caller hasn't decided transfer costs yet).
        transfer_matrix: ``len(backends) × len(backends)`` cost matrix
            in µs; off-diagonal entries are the cross-backend hop cost.
            If ``None``, this function is a no-op.
        backends: Backend ordering matching ``transfer_matrix``.
        fusion_gain_threshold: Forwarded to :func:`should_fuse`.

    Returns:
        A new :class:`GranularityPlan` with merged chunks. Chunks
        re-numbered ``chunk_000``, ``chunk_001``, ... ; the merged
        backend is the argmin of the summed durations (which may differ
        from either input chunk's preferred backend if fusion crosses
        a specialty boundary).
    """

    if transfer_matrix is None or not plan.chunks:
        return plan

    b_idx = {b: i for i, b in enumerate(backends)}

    def transfer_between(a: Chunk, b: Chunk) -> float:
        ai = b_idx.get(a.preferred_backend)
        bi = b_idx.get(b.preferred_backend)
        if ai is None or bi is None:
            return 0.0
        return float(transfer_matrix[ai][bi])

    chunks: list[Chunk] = list(plan.chunks)
    i = 0
    while i + 1 < len(chunks):
        a = chunks[i]
        b = chunks[i + 1]
        transfer_us = transfer_between(a, b)
        if should_fuse(
            a, b, transfer_us=transfer_us, fusion_gain_threshold=fusion_gain_threshold
        ):
            merged_ops = a.op_ids + b.op_ids
            merged_durations: dict[str, float] = {}
            for backend in set(a.durations_us_by_backend) | set(b.durations_us_by_backend):
                va = a.durations_us_by_backend.get(backend, math.inf)
                vb = b.durations_us_by_backend.get(backend, math.inf)
                merged_durations[backend] = va + vb
            merged_backend = _argmin_backend(merged_durations)
            chunks[i] = Chunk(
                chunk_id=a.chunk_id,  # renumbered below
                op_ids=merged_ops,
                preferred_backend=merged_backend,
                durations_us_by_backend=merged_durations,
            )
            del chunks[i + 1]
            # Don't advance i — try to fuse the newly merged chunk with the next.
            continue
        i += 1

    # Renumber chunk IDs to keep them contiguous.
    chunks = [
        Chunk(
            chunk_id=f"chunk_{idx:03d}",
            op_ids=c.op_ids,
            preferred_backend=c.preferred_backend,
            durations_us_by_backend=c.durations_us_by_backend,
        )
        for idx, c in enumerate(chunks)
    ]

    log.debug(
        "fusion_applied",
        workload=plan.workload_id,
        n_before=len(plan.chunks),
        n_after=len(chunks),
    )
    return GranularityPlan(
        workload_id=plan.workload_id,
        chunks=tuple(chunks),
        specialty_summary=dict(plan.specialty_summary),
        n_partitions=len(chunks),
    )


__all__ = [
    "Chunk",
    "GranularityPlan",
    "apply_fusion",
    "compute_specialty_matrix",
    "propose_chunks",
    "should_fuse",
]
