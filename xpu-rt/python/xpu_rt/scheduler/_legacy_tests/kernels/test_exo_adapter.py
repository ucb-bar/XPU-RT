"""Tests for Exo kernel backend adapter.

Covers ExoAdapter instantiation, quick_check without Exo installed,
ExoKernelResult dataclass, and search_kernel for supported ops.
"""

from __future__ import annotations

from compgen.kernels.exo_adapter import ExoAdapter, ExoKernelResult


def test_exo_adapter_instantiation_default() -> None:
    """ExoAdapter can be instantiated with default target."""
    adapter = ExoAdapter()
    assert adapter._target_name == "generic"


def test_exo_adapter_instantiation_custom_target() -> None:
    """ExoAdapter can be instantiated with a custom target name."""
    adapter = ExoAdapter(target_name="x86_avx2")
    assert adapter._target_name == "x86_avx2"


def test_exo_kernel_result_dataclass() -> None:
    """ExoKernelResult can be constructed and accessed."""
    result = ExoKernelResult(
        cluster_id="exo_matmul_generic",
        proc_code="@proc\ndef matmul(): pass",
        scheduled_code="@proc\ndef matmul(): pass",
        c_code="void matmul() {}",
        latency_us=42.0,
        correct=True,
        schedule_ops_applied=3,
    )
    assert result.cluster_id == "exo_matmul_generic"
    assert result.latency_us == 42.0
    assert result.correct is True
    assert result.schedule_ops_applied == 3


def test_exo_adapter_search_kernel_matmul() -> None:
    """search_kernel returns a result for supported matmul op."""
    adapter = ExoAdapter(target_name="test")
    result = adapter.search_kernel(
        op_name="matmul",
        input_shapes=[(64, 128), (128, 32)],
        output_shapes=[(64, 32)],
        dtype="f32",
    )
    assert result is not None
    assert result.cluster_id == "exo_matmul_test"
    assert "matmul" in result.proc_code
    assert result.correct is True
    assert result.schedule_ops_applied == 0


def test_exo_adapter_search_kernel_unsupported() -> None:
    """search_kernel returns None for unsupported ops."""
    adapter = ExoAdapter()
    result = adapter.search_kernel(
        op_name="unknown_fancy_op",
        input_shapes=[(64,)],
        output_shapes=[(64,)],
    )
    assert result is None
