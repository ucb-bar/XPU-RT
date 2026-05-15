"""Tests for the Exo-backed Saturn OPU kernel provider."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest
from compgen.kernels.provider import KernelContract, SearchBudget
from compgen.kernels.providers.exo_riscv_opu import (
    ExoRiscvOpuProvider,
    emit_kernels,
    kernel_contract_yaml,
)


def _mmt4d_contract(*, features: tuple[str, ...] = ("v", "xopu"), **overrides) -> KernelContract:
    """Build a matmul contract for Saturn OPU.

    ``features`` advertises the target's available instruction lanes — the
    provider reads this to pick the OPU or RVV variant. Tests that want
    the RVV fallback pass ``features=("v",)``.
    """
    data = dict(
        region_id="dispatch_0",
        op_family="matmul",
        dtypes=("int8", "int8", "int32"),
        input_shapes=((16, 128), (128, 16)),
        output_shapes=((16, 16),),
        target_name="saturn-opu-v128d64",
        hardware_key="saturn-opu-v128d64",
        constraints={"inner_tile": [16, 16, 128], "features": list(features)},
    )
    data.update(overrides)
    return KernelContract(**data)


def _im2col_contract(**overrides) -> KernelContract:
    data = dict(
        region_id="dispatch_1",
        op_family="im2col",
        dtypes=("int8",),
        target_name="saturn-opu-v128d64",
        hardware_key="saturn-opu-v128d64",
    )
    data.update(overrides)
    return KernelContract(**data)


def test_accepts_int8_mmt4d_on_saturn() -> None:
    provider = ExoRiscvOpuProvider()
    assert provider.accepts_contract(_mmt4d_contract())


def test_rejects_non_saturn_target() -> None:
    provider = ExoRiscvOpuProvider()
    c = _mmt4d_contract(target_name="cuda-a100", hardware_key="cuda-a100")
    assert not provider.accepts_contract(c)


def test_rejects_float_matmul() -> None:
    provider = ExoRiscvOpuProvider()
    c = _mmt4d_contract(dtypes=("float32", "float32", "float32"))
    assert not provider.accepts_contract(c)


def test_rejects_non_16x16x128_tile() -> None:
    provider = ExoRiscvOpuProvider()
    c = _mmt4d_contract(constraints={"inner_tile": [8, 8, 64]})
    assert not provider.accepts_contract(c)


def test_search_returns_mmt4d_kernel() -> None:
    provider = ExoRiscvOpuProvider()
    result = provider.search(_mmt4d_contract(), SearchBudget())
    assert result.found
    assert result.correct
    assert result.language == "c"
    assert "mmt4d_s8s8s32_16x16x128_xopu" in result.metadata["kernels"]
    # VOPACC encoding is referenced in the emitted source.
    assert ".insn r 0x57, 0x2, 0x51" in result.kernel_code
    # 16x16x128 tile shape is in the source.
    for literal in ("16", "128"):
        assert literal in result.kernel_code


def test_search_emits_im2col_kernel() -> None:
    provider = ExoRiscvOpuProvider()
    result = provider.search(_im2col_contract(), SearchBudget())
    assert result.found
    assert "im2col_s8_rvv" in result.metadata["kernels"]
    # vle8.v / vsse8.v tagged in metadata for contract book-keeping.
    tags = result.metadata["instruction_tags"]["im2col_s8_rvv"]
    assert "vle8.v" in tags
    assert "vsse8.v" in tags


def test_contract_feedback_suggests_encoding_swap() -> None:
    provider = ExoRiscvOpuProvider()
    result = provider.search(_mmt4d_contract(), SearchBudget())
    feedback_fields = {fb.field for fb in result.contract_feedback}
    assert "layout" in feedback_fields
    fb = next(fb for fb in result.contract_feedback if fb.field == "layout")
    assert fb.suggested_value == "mmt4d_encoding_swap_lhs"
    assert fb.measured_gain > 1.0


def test_knowledge_export_is_nonempty() -> None:
    provider = ExoRiscvOpuProvider()
    exports = provider.export_knowledge()
    assert exports
    assert exports[0].scope == "target"
    assert "Saturn OPU" in exports[0].content


def test_kernel_contract_yaml_shape() -> None:
    kernels = emit_kernels(_mmt4d_contract())
    yaml_text = kernel_contract_yaml(kernels)
    assert "name: mmt4d_s8s8s32_16x16x128_xopu" in yaml_text
    assert "tile_shape: [16, 16, 128]" in yaml_text
    assert "schedule_notes" in yaml_text


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc available")
def test_host_reference_mmt4d_matches_numpy(tmp_path: Path) -> None:
    """Compile the host-reference mmt4d and diff against a numpy ground truth.

    This is the verification ladder's level-2 check. It runs on the
    host (the ``#else`` branch of the emitted C) rather than on RISC-V,
    so we don't need a cross-compiler to catch algorithmic regressions.
    """
    pytest.importorskip("numpy")
    import numpy as np

    kernels = emit_kernels(_mmt4d_contract())
    mmt4d = next(k for k in kernels if k.name.startswith("mmt4d"))
    src = tmp_path / "mmt4d.c"
    src.write_text(mmt4d.c_source)
    so_path = tmp_path / "libmmt4d.so"
    subprocess.run(
        ["cc", "-O2", "-std=c17", "-Wall", "-Werror", "-fPIC", "-shared", str(src), "-o", str(so_path)],
        check=True,
    )
    lib = ctypes.CDLL(str(so_path))
    lib.compgen_mmt4d_s8s8s32_16x16x128_xopu.restype = None
    lib.compgen_mmt4d_s8s8s32_16x16x128_xopu.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
    ]

    rng = np.random.default_rng(0xBEEF)
    lhs_km = rng.integers(-4, 4, size=(128, 16), dtype=np.int8)
    rhs_kn = rng.integers(-4, 4, size=(128, 16), dtype=np.int8)
    out = np.zeros((16 * 16,), dtype=np.int32)

    lhs_c = lhs_km.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    rhs_c = rhs_kn.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    out_c = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    lib.compgen_mmt4d_s8s8s32_16x16x128_xopu(out_c, lhs_c, rhs_c)

    # Reference: the kernel uses encoding-swap LHS = [K, M] contiguous,
    # RHS = [K, N] contiguous → output[m, n] = sum_k LHS[k, m] * RHS[k, n].
    expected = (lhs_km.astype(np.int32).T @ rhs_kn.astype(np.int32)).ravel()
    np.testing.assert_array_equal(out.reshape(-1), expected)


def test_rvv_fallback_kernel_has_same_tile_shape() -> None:
    """Pure-RVV fallback matches the VOPACC path's tile for A/B comparability."""
    opu = emit_kernels(_mmt4d_contract(features=("v", "xopu")))
    rvv = emit_kernels(_mmt4d_contract(features=("v",)))
    assert opu[0].name.endswith("_xopu")
    assert rvv[0].name.endswith("_rvv")
    assert opu[0].tile_shape == rvv[0].tile_shape == (16, 16, 128)
    # RVV fallback does NOT reference the +xopu VOPACC encoding.
    assert ".insn r 0x57" not in rvv[0].c_source
    # Instruction tags reflect RVV-only lane.
    assert "vmul.vv" in rvv[0].instruction_tags


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc available")
def test_rvv_fallback_numerically_matches_opu(tmp_path: Path) -> None:
    """Numeric A/B: RVV fallback == VOPACC host-reference on the same inputs."""
    pytest.importorskip("numpy")
    import numpy as np

    sources: dict[str, Path] = {}
    libs: dict[str, ctypes.CDLL] = {}
    for variant, features in (("opu", ("v", "xopu")), ("rvv", ("v",))):
        k = next(iter(emit_kernels(_mmt4d_contract(features=features))))
        use_opu = "xopu" in features
        src = tmp_path / f"{variant}.c"
        src.write_text(k.c_source)
        so = tmp_path / f"{variant}.so"
        subprocess.run(
            ["cc", "-O2", "-std=c17", "-Wall", "-Werror", "-fPIC", "-shared", str(src), "-o", str(so)],
            check=True,
        )
        lib = ctypes.CDLL(str(so))
        sym = "compgen_mmt4d_s8s8s32_16x16x128_xopu" if use_opu else "compgen_mmt4d_s8s8s32_16x16x128_rvv"
        fn = getattr(lib, sym)
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
        ]
        sources[variant] = src
        libs[variant] = lib

    rng = np.random.default_rng(0xB00B)
    lhs = rng.integers(-4, 4, size=(128, 16), dtype=np.int8)
    rhs = rng.integers(-4, 4, size=(128, 16), dtype=np.int8)
    out_opu = np.zeros((16 * 16,), dtype=np.int32)
    out_rvv = np.zeros((16 * 16,), dtype=np.int32)

    libs["opu"].compgen_mmt4d_s8s8s32_16x16x128_xopu(
        out_opu.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lhs.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        rhs.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    libs["rvv"].compgen_mmt4d_s8s8s32_16x16x128_rvv(
        out_rvv.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lhs.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        rhs.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    np.testing.assert_array_equal(out_opu, out_rvv)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc available")
def test_host_reference_im2col_matches_torch(tmp_path: Path) -> None:
    """Differential test for im2col on the host fallback."""
    pytest.importorskip("numpy")
    import numpy as np

    kernels = emit_kernels(_im2col_contract())
    im2col = next(k for k in kernels if k.name.startswith("im2col"))
    src = tmp_path / "im2col.c"
    src.write_text(im2col.c_source)
    so_path = tmp_path / "libim2col.so"
    subprocess.run(
        ["cc", "-O2", "-std=c17", "-Wall", "-Werror", "-fPIC", "-shared", str(src), "-o", str(so_path)],
        check=True,
    )
    lib = ctypes.CDLL(str(so_path))
    lib.compgen_im2col_s8_rvv.restype = None
    lib.compgen_im2col_s8_rvv.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    H, W, C, KH, KW = 4, 4, 3, 3, 3
    rng = np.random.default_rng(0xFEED)
    inp = rng.integers(-8, 8, size=(H, W, C), dtype=np.int8)
    out = np.zeros((H * W * KH * KW * C,), dtype=np.int8)
    lib.compgen_im2col_s8_rvv(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        inp.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        H,
        W,
        C,
        KH,
        KW,
    )

    # Reference: zero-padded im2col matching the kernel's loop order.
    expected = []
    for h in range(H):
        for w in range(W):
            for kh in range(KH):
                for kw in range(KW):
                    for c in range(C):
                        ih, iw = h + kh, w + kw
                        expected.append(inp[ih, iw, c] if (ih < H and iw < W) else 0)
    np.testing.assert_array_equal(out, np.array(expected, dtype=np.int8))


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc available")
def test_emitted_source_compiles_cleanly(tmp_path: Path) -> None:
    """Host compile catches template drift for both ukernels at once."""
    kernels = emit_kernels(_mmt4d_contract()) + emit_kernels(_im2col_contract())
    for k in kernels:
        src = tmp_path / f"{k.name}.c"
        src.write_text(k.c_source)
        obj = tmp_path / f"{k.name}.o"
        subprocess.run(
            ["cc", "-std=c17", "-Wall", "-Werror", "-c", str(src), "-o", str(obj)],
            check=True,
        )
        assert obj.exists()
