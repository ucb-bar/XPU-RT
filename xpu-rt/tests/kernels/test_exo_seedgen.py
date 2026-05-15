"""Tests for Exo seed proc generation.

Covers seed generation for matmul, conv2d, reduction, elementwise,
and unsupported operations.
"""

from __future__ import annotations

from compgen.kernels.exo_seedgen import ExoSeedProc, generate_seed_proc


def test_generate_matmul_default() -> None:
    """generate_seed_proc produces a matmul proc with default shapes."""
    result = generate_seed_proc("matmul", [], [], "f32")
    assert result is not None
    assert "matmul" in result.name
    assert "@proc" in result.proc_source
    assert "C[i, j] += A[i, k] * B[k, j]" in result.proc_source


def test_generate_matmul_with_shapes() -> None:
    """generate_seed_proc uses provided shapes for matmul."""
    result = generate_seed_proc(
        "matmul",
        [(64, 128), (128, 32)],
        [(64, 32)],
        "f32",
    )
    assert result is not None
    assert result.name == "matmul_64x128x32"
    assert "f32" in result.proc_source
    assert len(result.input_types) == 2
    assert len(result.output_types) == 1


def test_generate_linalg_matmul_alias() -> None:
    """linalg.matmul maps to the same generator as matmul."""
    result = generate_seed_proc("linalg.matmul", [(8, 8), (8, 8)], [(8, 8)], "f32")
    assert result is not None
    assert "matmul" in result.name


def test_generate_conv2d() -> None:
    """generate_seed_proc produces a conv2d proc."""
    result = generate_seed_proc("conv2d", [], [], "f32")
    assert result is not None
    assert result.name == "conv2d_seed"
    assert "@proc" in result.proc_source
    assert "inp" in result.proc_source
    assert "weights" in result.proc_source


def test_generate_reduction() -> None:
    """generate_seed_proc produces a reduction proc."""
    result = generate_seed_proc("reduction", [(512,)], [(1,)], "f32")
    assert result is not None
    assert result.name == "reduce_sum"
    assert "result[0] += x[i]" in result.proc_source
    assert "f32[512]" in result.input_types[0]


def test_generate_elementwise() -> None:
    """generate_seed_proc produces an elementwise proc."""
    result = generate_seed_proc("elementwise", [(256,)], [(256,)], "f16")
    assert result is not None
    assert result.name == "elementwise_add"
    assert "c[i] = a[i] + b[i]" in result.proc_source
    assert "f16" in result.proc_source


def test_generate_unsupported_op() -> None:
    """generate_seed_proc returns None for unsupported ops."""
    result = generate_seed_proc("fancy_attention", [(32, 64)], [(32, 64)], "f32")
    assert result is None


def test_seed_proc_dataclass() -> None:
    """ExoSeedProc frozen dataclass can be constructed."""
    proc = ExoSeedProc(
        name="test_proc",
        proc_source="@proc\ndef test_proc(): pass",
        c_skeleton="void test_proc() {}",
        input_types=["f32[8]"],
        output_types=["f32[8]"],
    )
    assert proc.name == "test_proc"
    assert proc.input_types == ["f32[8]"]
    assert proc.output_types == ["f32[8]"]
