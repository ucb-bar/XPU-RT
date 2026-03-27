"""Tests for Ray actors — TargetRegistry, HardwareBroker, ArtifactIndex, PlanSearch.

All tests are automatically skipped when Ray is not installed.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")


@pytest.fixture
def registry(ray_cluster):
    from infra.ray.actors.target_registry import TargetRegistryActor

    return TargetRegistryActor.remote()


@pytest.fixture
def broker(ray_cluster):
    from infra.ray.actors.hardware_broker import HardwareBrokerActor

    return HardwareBrokerActor.remote()


@pytest.fixture
def artifact_index(ray_cluster):
    from infra.ray.actors.artifact_index import ArtifactIndexActor

    return ArtifactIndexActor.remote()


@pytest.fixture
def plan_search(ray_cluster):
    from infra.ray.actors.plan_search import PlanSearchActor

    return PlanSearchActor.remote()


class TestTargetRegistryActor:
    def test_empty_list(self, registry) -> None:
        targets = ray.get(registry.list_targets.remote())
        assert targets == []

    def test_get_nonexistent(self, registry) -> None:
        result = ray.get(registry.get_target.remote("nonexistent"))
        assert result is None

    def test_export_snapshot_empty(self, registry) -> None:
        snap = ray.get(registry.export_snapshot.remote())
        assert snap == {}

    def test_maturity_unknown(self, registry) -> None:
        level = ray.get(registry.get_maturity.remote("nonexistent"))
        assert level == "unknown"


class TestHardwareBrokerActor:
    def test_empty_resources(self, broker) -> None:
        resources = ray.get(broker.list_resources.remote())
        assert resources == []

    def test_register_and_list(self, broker) -> None:
        rid = ray.get(broker.register_resource.remote({
            "resource_id": "test-board-01",
            "resource_type": "board",
            "target_name": "test-target",
        }))
        assert rid == "test-board-01"
        resources = ray.get(broker.list_resources.remote())
        assert len(resources) == 1
        assert resources[0]["available"] is True

    def test_reserve_and_release(self, broker) -> None:
        ray.get(broker.register_resource.remote({
            "resource_id": "test-fpga-01",
            "resource_type": "fpga",
            "target_name": "xilinx",
        }))

        lease = ray.get(broker.reserve.remote("fpga", "tester", 60.0))
        assert lease is not None
        assert lease["status"] == "active"

        # Resource should now be unavailable
        lease2 = ray.get(broker.reserve.remote("fpga", "tester2", 60.0))
        assert lease2 is None

        # Release
        released = ray.get(broker.release.remote(lease["lease_id"]))
        assert released is True

        # Now available again
        lease3 = ray.get(broker.reserve.remote("fpga", "tester3", 60.0))
        assert lease3 is not None

    def test_reserve_wrong_type(self, broker) -> None:
        ray.get(broker.register_resource.remote({
            "resource_id": "test-board-02",
            "resource_type": "board",
        }))
        result = ray.get(broker.reserve.remote("fpga", "tester", 60.0))
        assert result is None


class TestArtifactIndexActor:
    def test_empty(self, artifact_index) -> None:
        count = ray.get(artifact_index.count.remote())
        assert count == 0

    def test_register_and_get(self, artifact_index) -> None:
        aid = ray.get(artifact_index.register_artifact.remote(
            artifact_type="bundle",
            target_name="cuda-a100",
            storage_path="/artifacts/bundle_001",
            model_hash="abc123",
            objective="latency",
        ))
        assert isinstance(aid, str)

        entry = ray.get(artifact_index.get_artifact.remote(aid))
        assert entry is not None
        assert entry["target_name"] == "cuda-a100"
        assert entry["artifact_type"] == "bundle"

    def test_find_by_target(self, artifact_index) -> None:
        ray.get(artifact_index.register_artifact.remote(
            artifact_type="bundle",
            target_name="target-a",
            storage_path="/a",
        ))
        ray.get(artifact_index.register_artifact.remote(
            artifact_type="bundle",
            target_name="target-b",
            storage_path="/b",
        ))

        results = ray.get(artifact_index.find_artifacts.remote(
            target_name="target-a",
        ))
        assert len(results) >= 1
        assert all(r["target_name"] == "target-a" for r in results)

    def test_delete(self, artifact_index) -> None:
        aid = ray.get(artifact_index.register_artifact.remote(
            artifact_type="kernel",
            target_name="test",
            storage_path="/k",
        ))
        deleted = ray.get(artifact_index.delete_artifact.remote(aid))
        assert deleted is True
        assert ray.get(artifact_index.get_artifact.remote(aid)) is None


class TestPlanSearchActor:
    def test_start_evolutionary(self, plan_search) -> None:
        exp_id = ray.get(plan_search.start_evolutionary_search.remote(
            target_name="test",
            target_profile_path="specs/test.yaml",
        ))
        assert isinstance(exp_id, str)

        status = ray.get(plan_search.get_experiment_status.remote(exp_id))
        assert status["experiment_type"] == "evolutionary"
        assert status["status"] == "completed"

    def test_start_tile_search(self, plan_search) -> None:
        exp_id = ray.get(plan_search.start_tile_search.remote(
            target_name="test",
            op_type="matmul",
        ))
        status = ray.get(plan_search.get_experiment_status.remote(exp_id))
        assert status["experiment_type"] == "tile"

    def test_list_experiments(self, plan_search) -> None:
        ray.get(plan_search.start_eqsat_ablation.remote(
            target_name="test",
        ))
        exps = ray.get(plan_search.list_experiments.remote())
        assert len(exps) >= 1

    def test_nonexistent_experiment(self, plan_search) -> None:
        result = ray.get(plan_search.get_experiment_status.remote("bogus"))
        assert "error" in result
