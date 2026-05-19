"""Unit tests for the ``multi_model`` synthetic DAG generator.

The generator lives under ``scripts/experiments`` (sibling to other
experiment harnesses, not under the importable package) so the tests
add that path to ``sys.path`` before importing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tools.experiments._synthetic_dag import multi_model  # noqa: E402


def _is_dag(pids: list[str], deps: dict[str, list[str]]) -> bool:
    color = {p: 0 for p in pids}

    def visit(p: str) -> bool:
        if color[p] == 1:
            return False
        if color[p] == 2:
            return True
        color[p] = 1
        for d in deps.get(p, []):
            if d not in color:
                return False
            if not visit(d):
                return False
        color[p] = 2
        return True

    return all(visit(p) for p in pids)


def test_per_op_preserves_op_count() -> None:
    specs = [("chain", 50), ("transformer", 4), ("fan_out", 20)]
    dag = multi_model(specs, granularity="per_op", num_devices=4, seed=0)
    expected = 50 + (1 + 4 * 8) + (1 + 20 + 1)
    assert len(dag.partition_ids) == expected
    assert _is_dag(dag.partition_ids, dag.dependencies)


def test_per_model_collapses_to_one_partition_each() -> None:
    specs = [("chain", 30), ("transformer", 3), ("fan_out", 12)]
    dag = multi_model(specs, granularity="per_model", num_devices=4, seed=1)
    assert len(dag.partition_ids) == len(specs)
    # Each per-model partition is independent (no cross-model edges).
    for pid in dag.partition_ids:
        assert dag.dependencies[pid] == []


def test_per_layer_strictly_fewer_than_per_op() -> None:
    specs = [("chain", 40), ("transformer", 6), ("fan_out", 16)]
    per_op = multi_model(specs, granularity="per_op", num_devices=4, seed=2)
    per_layer = multi_model(specs, granularity="per_layer", num_devices=4, seed=2)
    assert len(per_layer.partition_ids) < len(per_op.partition_ids)
    assert _is_dag(per_layer.partition_ids, per_layer.dependencies)


@pytest.mark.parametrize("granularity", ["per_op", "per_layer", "per_block", "per_model"])
def test_all_granularities_are_dags(granularity: str) -> None:
    specs = [("chain", 25), ("transformer", 4), ("fan_out", 10)]
    dag = multi_model(specs, granularity=granularity, num_devices=4, seed=3)
    assert _is_dag(dag.partition_ids, dag.dependencies)
    # Every dependency target must be a declared partition.
    pid_set = set(dag.partition_ids)
    for succ, preds in dag.dependencies.items():
        assert succ in pid_set
        for p in preds:
            assert p in pid_set
            assert p != succ
    # Every partition has a duration vector of the right shape.
    for pid in dag.partition_ids:
        assert len(dag.durations_us_by_device[pid]) == dag.num_devices
        assert all(d >= 0.0 for d in dag.durations_us_by_device[pid])


def test_per_model_duration_equals_sum_per_op() -> None:
    specs = [("chain", 12)]
    per_op = multi_model(specs, granularity="per_op", num_devices=4, seed=7)
    per_model = multi_model(specs, granularity="per_model", num_devices=4, seed=7)
    assert len(per_model.partition_ids) == 1
    coarse = per_model.durations_us_by_device[per_model.partition_ids[0]]
    fine_sum = [0.0] * per_op.num_devices
    for pid in per_op.partition_ids:
        for d, v in enumerate(per_op.durations_us_by_device[pid]):
            fine_sum[d] += v
    for d in range(per_op.num_devices):
        assert coarse[d] == pytest.approx(fine_sum[d], rel=1e-9)
