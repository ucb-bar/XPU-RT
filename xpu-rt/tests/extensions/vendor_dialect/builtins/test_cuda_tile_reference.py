"""Unit tests for the in-tree cuda_tile reference adapter.

These tests do NOT require ``cuda-tile-translate`` on PATH. The
toolchain-driven path is covered by the integration test
:func:`test_emit_with_toolchain_when_present`, which is auto-skipped
when the binary is missing.
"""

from __future__ import annotations

import shutil

import pytest
from compgen.extensions.vendor_dialect import (
    LoweringResult,
    VendorDialectAdapter,
    available_adapters,
    list_builtin_adapters,
    make_builtin_adapter,
    register_builtin_adapter,
    reset_registry,
)
from compgen.extensions.vendor_dialect.builtins.cuda_tile import (
    CudaTileReferenceAdapter,
    make_adapter,
)
from compgen.extensions.vendor_dialect.builtins.cuda_tile.lowering import (
    FfnShapes,
    emit_ffn_single_tile_mlir,
    lower_to_cuda_tile,
)
from compgen.targets.backend import CompiledArtifact

# --------------------------------------------------------------------------- #
# Discovery / registry surface
# --------------------------------------------------------------------------- #


class TestBuiltinDiscovery:
    def test_cuda_tile_listed(self) -> None:
        assert "cuda_tile" in list_builtin_adapters()

    def test_make_builtin_adapter_returns_instance(self) -> None:
        adapter = make_builtin_adapter("cuda_tile")
        assert isinstance(adapter, VendorDialectAdapter)
        assert isinstance(adapter, CudaTileReferenceAdapter)

    def test_register_builtin_inserts_into_registry(self) -> None:
        reset_registry()
        try:
            register_builtin_adapter("cuda_tile")
            assert "cuda_tile" in available_adapters()
        finally:
            reset_registry()

    def test_unknown_builtin_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown builtin adapter"):
            make_builtin_adapter("not_a_real_adapter")


# --------------------------------------------------------------------------- #
# Descriptor + capabilities surface
# --------------------------------------------------------------------------- #


class TestDescriptor:
    def test_descriptor_has_canonical_fields(self) -> None:
        adapter = make_adapter()
        d = adapter.descriptor
        assert d.name == "cuda_tile"
        assert d.target == "nvidia-blackwell"
        assert d.output_format == "cuda-tile-bitcode"
        assert "cuda-tile-translate" in d.compile_entry.cli_tools
        assert d.lowering.mode == "kernel_authoring"

    def test_capabilities_declare_op_types(self) -> None:
        caps = make_adapter().capabilities()
        assert "linear" in caps["supported_op_types"]
        assert "relu" in caps["supported_op_types"]
        assert "fp32" in caps["supported_dtypes"]
        assert caps["source"] == "in-tree-builtin"
        assert caps["validated_against"] == "bridge#144"


# --------------------------------------------------------------------------- #
# Lowering: deterministic MLIR emission
# --------------------------------------------------------------------------- #


class TestLowering:
    def test_default_shapes_emit_module(self) -> None:
        text = emit_ffn_single_tile_mlir()
        assert text.startswith("cuda_tile.module @ffn_kernels")
        assert "cuda_tile.entry @ffn_matmul_relu_matmul" in text
        assert "cuda_tile.return" in text

    def test_template_uses_canonical_ops(self) -> None:
        text = emit_ffn_single_tile_mlir()
        for op in [
            "cuda_tile.make_tensor_view",
            "cuda_tile.make_partition_view",
            "cuda_tile.load_view_tko",
            "cuda_tile.mmaf",
            "cuda_tile.maxf",
            "cuda_tile.store_view_tko",
        ]:
            assert op in text, f"missing op {op}"

    def test_two_mmafs_for_ffn(self) -> None:
        """FFN = matmul + relu + matmul → two ``mmaf`` ops."""
        text = emit_ffn_single_tile_mlir()
        assert text.count("cuda_tile.mmaf") == 2

    def test_one_relu_via_maxf(self) -> None:
        text = emit_ffn_single_tile_mlir()
        assert text.count("cuda_tile.maxf") == 1

    def test_shape_overrides_propagate(self) -> None:
        text = emit_ffn_single_tile_mlir(FfnShapes(M=4, K=8, N=16, M_out=8))
        assert "tile<4x8xf32>" in text
        assert "tile<8x16xf32>" in text
        assert "tile<16x8xf32>" in text  # w_down K=N → M_out
        assert "tile<4x16xf32>" in text  # acc accumulator

    def test_default_shapes_match_bridge_144(self) -> None:
        """Bridge #144 default: M=8, K=16, N=32, M_out=16."""
        s = FfnShapes()
        assert (s.M, s.K, s.N, s.M_out) == (8, 16, 32, 16)

    def test_lower_to_cuda_tile_returns_lowering_result(self, tmp_path: object) -> None:
        adapter = make_adapter()
        result = lower_to_cuda_tile(
            payload_mlir="// payload\n",
            descriptor=adapter.descriptor,
            kernel_provider=None,
            output_dir=tmp_path,  # type: ignore[arg-type]
        )
        assert isinstance(result, LoweringResult)
        assert result.vendor_mlir.startswith("cuda_tile.module")
        assert result.metadata["lowering_mode"] == "ffn_matmul_relu_template"
        assert result.metadata["shapes"] == {"M": 8, "K": 16, "N": 32, "M_out": 16}
        assert "make_tensor_view" in result.metadata["ops_used"]
        assert result.metadata["payload_mlir_consumed"] is False  # reference is template-only

    def test_lowering_writes_mlir_to_disk(self, tmp_path: object) -> None:
        adapter = make_adapter()
        result = adapter.lower_payload("// payload\n", output_dir=tmp_path)  # type: ignore[arg-type]
        from pathlib import Path

        mlir_path = Path(result.metadata["vendor_mlir_path"])
        assert mlir_path.exists()
        assert mlir_path.read_text() == result.vendor_mlir

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            FfnShapes(M=0, K=16, N=32, M_out=16).validate()


# --------------------------------------------------------------------------- #
# Bundle: graceful degradation when toolchain absent
# --------------------------------------------------------------------------- #


class TestBundleDegrade:
    """Toolchain-absent path. Always-on test (no skip)."""

    def test_degrades_to_mlir_text_when_translate_missing(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force PATH lookup to miss regardless of the host install.
        from compgen.extensions.vendor_dialect.builtins.cuda_tile import bundle as bundle_mod

        monkeypatch.setattr(bundle_mod, "_toolchain_path", lambda: None)

        adapter = make_adapter()
        artifact = adapter.compile("// payload\n", output_dir=tmp_path)
        assert isinstance(artifact, CompiledArtifact)
        assert artifact.format == "mlir-cuda-tile"
        assert artifact.target_name == "nvidia-blackwell"
        assert artifact.metadata["toolchain_present"] is False
        assert artifact.metadata["toolchain_required"] == "cuda-tile-translate"
        # The "code" field is the MLIR text inline.
        assert artifact.code.startswith("cuda_tile.module")

    def test_degraded_artifact_path_points_to_mlir(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path

        from compgen.extensions.vendor_dialect.builtins.cuda_tile import bundle as bundle_mod

        monkeypatch.setattr(bundle_mod, "_toolchain_path", lambda: None)

        adapter = make_adapter()
        artifact = adapter.compile("// payload\n", output_dir=tmp_path)
        path = Path(artifact.metadata["artifact_path"])
        assert path.exists()
        assert path.suffix == ".mlir"
        assert path.read_text().startswith("cuda_tile.module")


# --------------------------------------------------------------------------- #
# Bundle: real toolchain (auto-skip when absent)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    shutil.which("cuda-tile-translate") is None,
    reason="cuda-tile-translate not on PATH",
)
class TestBundleWithToolchain:
    def test_real_translate_emits_bytecode(self, tmp_path: object) -> None:
        adapter = make_adapter()
        artifact = adapter.compile("// payload\n", output_dir=tmp_path)
        assert artifact.format == "cuda-tile-bitcode"
        assert artifact.metadata["toolchain_present"] is True
        from pathlib import Path

        bytecode_path = Path(artifact.metadata["artifact_path"])
        assert bytecode_path.exists()
        assert bytecode_path.suffix == ".tileirbc"
        # Magic check — the bundle stage already enforced this; this is
        # belt-and-braces for the test report.
        assert bytecode_path.read_bytes().startswith(b"\x7fTileIR\x00")
