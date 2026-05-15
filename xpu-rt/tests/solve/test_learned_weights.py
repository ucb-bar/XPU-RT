"""Tests for learned cost model weights (Unit 12)."""

from __future__ import annotations

import pytest
from compgen.solve.learned_weights import retrieve_best_weights, store_cost_weights
from compgen.solve.objectives import CompositeCost


@pytest.fixture
def memory(tmp_path):
    from compgen.memory.store import CompilerMemory

    return CompilerMemory(
        db_path=tmp_path / "test.db",
        blob_root=tmp_path / "blobs",
    )


class TestLearnedWeights:
    def test_store_returns_id(self, memory):
        kid = store_cost_weights(
            memory,
            target_key="gpu_a100",
            weights={"fusion_weight": 1.5, "transfer_weight": 0.8, "backend_match_weight": 2.0},
            measured_gain=5.0,
        )
        assert kid
        assert len(kid) > 0

    def test_store_and_retrieve(self, memory):
        store_cost_weights(
            memory,
            target_key="gpu_a100",
            weights={"fusion_weight": 1.5, "transfer_weight": 0.8},
            measured_gain=5.0,
        )
        result = retrieve_best_weights(memory, "gpu_a100")
        assert result is not None
        assert result["fusion_weight"] == 1.5
        assert result["transfer_weight"] == 0.8

    def test_retrieve_no_data(self, memory):
        result = retrieve_best_weights(memory, "nonexistent_target")
        assert result is None


class TestCompositeCostFromLearned:
    def test_basic_weights(self):
        cost = CompositeCost.from_learned({"latency_weight": 2.0})
        assert len(cost.terms) >= 1
        assert cost.terms[0].weight == 2.0

    def test_with_memory_weight(self):
        cost = CompositeCost.from_learned(
            {
                "fusion_weight": 1.0,
                "memory_weight": 0.5,
            }
        )
        assert len(cost.terms) >= 2

    def test_from_learned_with_energy(self):
        cost = CompositeCost.from_learned(
            {
                "fusion_weight": 1.0,
                "energy_weight": 0.1,
            }
        )
        assert any(hasattr(t, "energy_per_us") for t in cost.terms)
