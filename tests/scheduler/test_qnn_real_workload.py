"""Tests for ``xpu_rt.scheduler.qnn_real_workload``.

Three guarantees are exercised:

1. ``make_chain_dag`` produces the right partition count and a valid
   chain / k-lookahead dependency structure.
2. ``chunk_dag`` preserves total per-backend cost (sum of chunks equals
   sum of constituent ops) when every op is supported on that backend.
3. ``chunk_dag`` produces a self-consistent DAG: all dependency targets
   exist as partition IDs, and a chain input yields a chunk-level
   chain.
"""

from __future__ import annotations

from xpu_rt.scheduler.qnn_real_workload import (
    BACKENDS,
    chunk_dag,
    load_cost_matrix,
    make_chain_dag,
)


def _load() -> dict[str, dict[str, dict[str, float]]]:
    return load_cost_matrix(
        "xpu-rt/data/profiled/qnn_cost_matrix.json"
    )


def test_chain_dag_structure() -> None:
    """yolov8n chain has 273 partitions; deps form a proper k-chain."""
    matrix = _load()
    dag_k1 = make_chain_dag("yolov8n", matrix, k_lookahead=1)
    assert len(dag_k1.partition_ids) == 273
    assert dag_k1.num_devices == 3
    assert dag_k1.backends == BACKENDS
    # Pure chain: op 0 has 0 deps; op i (i>=1) has exactly 1 (=op i-1).
    pids = dag_k1.partition_ids
    assert dag_k1.dependencies[pids[0]] == []
    for i in range(1, len(pids)):
        deps = dag_k1.dependencies[pids[i]]
        assert deps == [pids[i - 1]], f"op {i} expected [{pids[i-1]}], got {deps}"

    # k=4 relaxation: op i has min(i, 4) deps (pids[i-4 .. i-1]).
    dag_k4 = make_chain_dag("yolov8n", matrix, k_lookahead=4)
    for i in [0, 1, 3, 4, 5, 100, 272]:
        expected = pids[max(0, i - 4) : i]
        assert dag_k4.dependencies[pids[i]] == list(expected), (
            f"k=4 op {i}: expected {expected}, got {dag_k4.dependencies[pids[i]]}"
        )

    # Transfer matrix is symmetric off-diagonal, zero on diagonal.
    for i in range(dag_k1.num_devices):
        assert dag_k1.transfer_us[i][i] == 0.0
        for j in range(dag_k1.num_devices):
            if i != j:
                assert dag_k1.transfer_us[i][j] > 0.0


def test_chunking_preserves_total_cost() -> None:
    """Sum of chunk durations == sum of op durations (per backend, when
    every op is supported on that backend)."""
    matrix = _load()
    dag = make_chain_dag("yolov8n", matrix, k_lookahead=1)

    # DSP supports every yolov8n op (matrix metadata: 273/273 on DSP).
    dsp_idx = BACKENDS.index("DSP")
    op_total_dsp = sum(
        float(dag.durations_us_by_device[pid][dsp_idx]) for pid in dag.partition_ids
    )

    for n_chunks in [1, 2, 4, 8, 16, 32, 273]:
        chunked = chunk_dag(dag, n_chunks)
        assert len(chunked.partition_ids) == n_chunks
        chunk_total_dsp = sum(
            float(chunked.durations_us_by_device[cid][dsp_idx])
            for cid in chunked.partition_ids
        )
        # Floating-point sums of ~270k µs accumulate ~1e-9 relative error.
        assert abs(chunk_total_dsp - op_total_dsp) < 1e-3, (
            f"n_chunks={n_chunks}: total DSP cost drifted "
            f"{chunk_total_dsp} vs op total {op_total_dsp}"
        )


def test_chunked_dependencies_consistent() -> None:
    """Chunk-level dependencies reference only existing chunk IDs and
    form a chain when the input is a strict chain."""
    matrix = _load()
    dag = make_chain_dag("yolov8n", matrix, k_lookahead=1)

    for n_chunks in [1, 4, 16, 32]:
        chunked = chunk_dag(dag, n_chunks)
        chunk_id_set = set(chunked.partition_ids)
        for cid, preds in chunked.dependencies.items():
            assert cid in chunk_id_set, f"missing chunk id {cid}"
            for p in preds:
                assert p in chunk_id_set, (
                    f"chunk {cid} references unknown predecessor {p}"
                )

        # Strict chain → chunk-level chain: chunk c depends on c-1 only.
        for idx, cid in enumerate(chunked.partition_ids):
            preds = chunked.dependencies[cid]
            if idx == 0:
                assert preds == [], f"first chunk should have no deps, got {preds}"
            else:
                assert preds == [chunked.partition_ids[idx - 1]], (
                    f"chunk {idx} ({cid}): expected single pred "
                    f"{chunked.partition_ids[idx-1]}, got {preds}"
                )

        # Identity chunking (n_chunks == n_ops) preserves original
        # partition IDs.
    n = len(dag.partition_ids)
    identity = chunk_dag(dag, n)
    assert identity.partition_ids == dag.partition_ids
