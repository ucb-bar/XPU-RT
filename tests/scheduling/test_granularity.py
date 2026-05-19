"""Tests for the specialty-driven granularity heuristic."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from xpu_rt.scheduler.qnn_real_workload import (
    chunk_dag_from_chunks,
    load_cost_matrix,
    make_chain_dag,
)
from xpu_rt.scheduling.granularity import (
    Chunk,
    apply_fusion,
    compute_specialty_matrix,
    propose_chunks,
    should_fuse,
)

COST_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "profiled"
    / "qnn_cost_matrix.json"
)


@pytest.fixture(scope="module")
def cost_matrix() -> dict:
    return load_cost_matrix(COST_MATRIX_PATH)


@pytest.fixture(scope="module")
def yolov8n_dag(cost_matrix: dict):
    return make_chain_dag("yolov8n", cost_matrix=cost_matrix)


@pytest.fixture(scope="module")
def yolov8n_specialty(cost_matrix: dict) -> dict[str, str]:
    return compute_specialty_matrix(cost_matrix, "yolov8n")


def test_chunks_cover_all_ops(yolov8n_dag, cost_matrix, yolov8n_specialty) -> None:
    plan = propose_chunks(
        yolov8n_dag,
        cost_matrix,
        "yolov8n",
        yolov8n_specialty,
        max_chunk_ops=16,
        max_partitions=200,
    )
    seen: list[str] = []
    for chunk in plan.chunks:
        seen.extend(chunk.op_ids)
    assert len(seen) == len(yolov8n_dag.partition_ids)
    assert seen == list(yolov8n_dag.partition_ids), "chunk order must be topological"
    assert len(set(seen)) == len(seen), "no op should appear in two chunks"


def test_chunk_count_under_cap(yolov8n_dag, cost_matrix, yolov8n_specialty) -> None:
    plan = propose_chunks(
        yolov8n_dag,
        cost_matrix,
        "yolov8n",
        yolov8n_specialty,
        max_chunk_ops=16,
        max_partitions=64,
    )
    assert plan.n_partitions <= 64
    assert plan.n_partitions == len(plan.chunks)


def test_majority_chunks_prefer_dsp(yolov8n_dag, cost_matrix, yolov8n_specialty) -> None:
    plan = propose_chunks(
        yolov8n_dag,
        cost_matrix,
        "yolov8n",
        yolov8n_specialty,
        max_chunk_ops=16,
        max_partitions=200,
    )
    n_total = sum(len(c.op_ids) for c in plan.chunks)
    n_dsp = sum(
        len(c.op_ids) for c in plan.chunks if c.preferred_backend == "DSP"
    )
    # V1's specialty distribution: DSP wins on the bulk of yolov8n ops.
    assert n_dsp / n_total > 0.70, (
        f"expected >70% of ops in DSP-preferred chunks, got {n_dsp}/{n_total}"
    )


def test_fusion_respects_threshold() -> None:
    # Synthetic 3-chunk fixture: alternating GPU / DSP / GPU. b's
    # min-backend serial cost = 100 us. With transfer = 50 us (>30% of
    # 100), adjacent chunks fuse. With transfer = 10 us (<30%), they don't.
    chunks = [
        Chunk(
            chunk_id="chunk_000",
            op_ids=("op_a",),
            preferred_backend="GPU",
            durations_us_by_backend={"CPU": 200.0, "GPU": 50.0, "DSP": 80.0},
        ),
        Chunk(
            chunk_id="chunk_001",
            op_ids=("op_b",),
            preferred_backend="DSP",
            durations_us_by_backend={"CPU": 300.0, "GPU": 150.0, "DSP": 100.0},
        ),
        Chunk(
            chunk_id="chunk_002",
            op_ids=("op_c",),
            preferred_backend="GPU",
            durations_us_by_backend={"CPU": 200.0, "GPU": 60.0, "DSP": 90.0},
        ),
    ]

    above = should_fuse(chunks[0], chunks[1], transfer_us=50.0, fusion_gain_threshold=0.3)
    below = should_fuse(chunks[0], chunks[1], transfer_us=10.0, fusion_gain_threshold=0.3)
    assert above is True
    assert below is False

    # Transfer matrix mirrors the GPU<->DSP hop at 50us — fusion should
    # collapse all three chunks into one.
    backends = ("CPU", "GPU", "DSP")
    transfer_matrix = [
        [0.0, 50.0, 50.0],
        [50.0, 0.0, 50.0],
        [50.0, 50.0, 0.0],
    ]
    from xpu_rt.scheduling.granularity import GranularityPlan

    plan = GranularityPlan(
        workload_id="synthetic",
        chunks=tuple(chunks),
        specialty_summary={},
        n_partitions=3,
    )
    fused = apply_fusion(
        plan,
        transfer_matrix=transfer_matrix,
        backends=backends,
        fusion_gain_threshold=0.3,
    )
    assert fused.n_partitions < 3, "above-threshold transfer must trigger fusion"

    # Below threshold (transfer = 5us) → no fusion of cross-backend chunks.
    transfer_matrix_low = [
        [0.0, 5.0, 5.0],
        [5.0, 0.0, 5.0],
        [5.0, 5.0, 0.0],
    ]
    not_fused = apply_fusion(
        plan,
        transfer_matrix=transfer_matrix_low,
        backends=backends,
        fusion_gain_threshold=0.3,
    )
    assert not_fused.n_partitions == 3


def test_chunk_dag_from_chunks_round_trips(
    yolov8n_dag, cost_matrix, yolov8n_specialty
) -> None:
    plan = propose_chunks(
        yolov8n_dag,
        cost_matrix,
        "yolov8n",
        yolov8n_specialty,
        max_chunk_ops=16,
        max_partitions=64,
    )
    chunked = chunk_dag_from_chunks(yolov8n_dag, list(plan.chunks))
    assert len(chunked.partition_ids) == plan.n_partitions
    # Every partition must have at least one feasible (positive) backend.
    for pid, durations in chunked.durations_us_by_device.items():
        # None marks infeasible; remaining must be > 0.
        feasible = [d for d in durations if d is not None]
        assert feasible, f"chunk {pid} has no feasible backend"
        assert all(
            d > 0 or math.isinf(d) for d in feasible
        ), f"chunk {pid} has non-positive duration: {durations}"
    # Chain-DAG topology: chunk i depends on chunk i-1.
    chunk_ids = chunked.partition_ids
    for i, cid in enumerate(chunk_ids):
        if i == 0:
            assert chunked.dependencies[cid] == []
        else:
            assert chunk_ids[i - 1] in chunked.dependencies[cid]
