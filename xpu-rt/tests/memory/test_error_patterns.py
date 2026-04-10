"""Tests for error pattern learning (Unit 14)."""
from __future__ import annotations
import pytest
from pathlib import Path
from compgen.memory.error_patterns import ErrorPattern, record_error_pattern, retrieve_error_patterns, error_patterns_to_prompt


@pytest.fixture
def memory(tmp_path):
    from compgen.memory.store import CompilerMemory
    return CompilerMemory(
        db_path=tmp_path / "test.db",
        blob_root=tmp_path / "blobs",
    )


class TestRecordErrorPattern:
    def test_record_returns_id(self, memory):
        kid = record_error_pattern(
            memory,
            action_type="tile",
            region_context="matmul_0",
            failure_reason="dimension not divisible by tile size",
            target_key="gpu_a100",
        )
        assert kid
        assert len(kid) > 0

    def test_record_and_retrieve(self, memory):
        record_error_pattern(memory, "tile", "matmul_0", "dim too small", "gpu_a100")
        record_error_pattern(memory, "fuse", "gelu_0", "not adjacent", "gpu_a100")

        patterns = retrieve_error_patterns(memory, action_type="tile", top_k=5)
        assert len(patterns) >= 1
        assert any(p.action_type == "tile" for p in patterns)
        assert any("dim too small" in p.failure_reason for p in patterns)

    def test_retrieve_empty(self, memory):
        patterns = retrieve_error_patterns(memory, action_type="nonexistent")
        assert patterns == []


class TestErrorPatternsToPrompt:
    def test_converts_to_dicts(self):
        patterns = [
            ErrorPattern("tile", "matmul_0", "dim too small", "gpu", 3),
            ErrorPattern("fuse", "gelu_0", "not adjacent", "gpu", 1),
        ]
        result = error_patterns_to_prompt(patterns)
        assert len(result) == 2
        assert result[0]["action_type"] == "tile"
        assert result[0]["failure_reason"] == "dim too small"
        assert result[0]["occurrences"] == 3
