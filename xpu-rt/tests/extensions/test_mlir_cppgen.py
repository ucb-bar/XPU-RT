"""Tests for the MLIR C++ compiler generator.

Validates:
1. Introspection extracts correct ops/attrs/properties/verifiers
2. Generated TableGen files have correct structure
3. Generated C++ files have correct structure
4. Generated CMake files are valid
5. Full end-to-end generation produces expected file tree
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Introspection tests
# ---------------------------------------------------------------------------


class TestIntrospection:
    """Test xDSL dialect introspection."""

    def test_layout_dialect_ops(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        assert info.name == "layout"
        assert info.prefix == "Layout"
        assert info.cpp_namespace == "compgen::layout"
        assert len(info.ops) == 4
        assert len(info.attrs) == 2

        op_names = [op.class_name for op in info.ops]
        assert "SetLayoutOp" in op_names
        assert "UnsetLayoutOp" in op_names
        assert "PackOp" in op_names
        assert "UnpackOp" in op_names

    def test_layout_set_layout_properties(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        set_layout = next(op for op in info.ops if op.class_name == "SetLayoutOp")

        assert len(set_layout.properties) == 3
        prop_names = [p.name for p in set_layout.properties]
        assert "encoding" in prop_names
        assert "source_ref" in prop_names
        assert "provenance" in prop_names

        # encoding is required, provenance is optional
        encoding = next(p for p in set_layout.properties if p.name == "encoding")
        assert not encoding.is_optional
        assert encoding.tablegen_type == "Layout_LayoutEncodingAttr"

        provenance = next(p for p in set_layout.properties if p.name == "provenance")
        assert provenance.is_optional
        assert provenance.tablegen_type == "RecipeBase_ProvenanceAttr"

    def test_layout_pack_verifier(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        pack = next(op for op in info.ops if op.class_name == "PackOp")
        assert pack.verifier is not None
        assert pack.verifier.kind == "range_check"
        assert pack.verifier.property_name == "is_prepack"

    def test_layout_set_layout_no_verifier(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        set_layout = next(op for op in info.ops if op.class_name == "SetLayoutOp")
        assert set_layout.verifier is None

    def test_layout_attrs(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        attr_names = [a.class_name for a in info.attrs]
        assert "LayoutEncodingAttr" in attr_names
        assert "PackSpecAttr" in attr_names

        encoding = next(a for a in info.attrs if a.class_name == "LayoutEncodingAttr")
        assert encoding.mlir_mnemonic == "encoding"
        assert len(encoding.fields) == 5
        field_names = [f.name for f in encoding.fields]
        assert "op_type" in field_names
        assert "operand_index" in field_names
        assert "tile_dims" in field_names

    def test_tile_dialect(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_tile_dialect

        info = introspect_tile_dialect()
        assert info.name == "tile"
        assert len(info.ops) == 7
        assert len(info.attrs) == 3

    def test_tile_elementwise_enum_verifier(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_tile_dialect

        info = introspect_tile_dialect()
        ew = next(op for op in info.ops if op.class_name == "TileElementwiseOp")
        assert ew.verifier is not None
        assert ew.verifier.kind == "enum_check"
        assert ew.verifier.property_name == "op_kind"
        assert "relu" in ew.verifier.valid_values
        assert "gelu" in ew.verifier.valid_values
        assert len(ew.verifier.valid_values) == 16

    def test_tile_mma_dimension_verifier(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_tile_dialect

        info = introspect_tile_dialect()
        mma = next(op for op in info.ops if op.class_name == "TileMMAOp")
        assert mma.verifier is not None
        assert mma.verifier.kind == "dimension_check"

    def test_accel_dialect(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_accel_dialect

        info = introspect_accel_dialect()
        assert info.name == "compgen.accel"
        assert len(info.ops) == 6
        assert len(info.attrs) == 0

    def test_accel_matrix_engine_verifier(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_accel_dialect

        info = introspect_accel_dialect()
        me = next(op for op in info.ops if op.class_name == "AccelMatrixEngineIROp")
        assert me.verifier is not None
        assert me.verifier.kind == "enum_check"
        assert "matmul" in me.verifier.valid_values

    def test_recipe_base(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_recipe_base

        info = introspect_recipe_base()
        assert info.name == "recipe_base"
        assert len(info.attrs) == 2
        assert len(info.ops) == 0
        attr_names = [a.class_name for a in info.attrs]
        assert "ProvenanceAttr" in attr_names
        assert "DeviceRefAttr" in attr_names

    def test_traits_extracted(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect

        info = introspect_layout_dialect()
        for op in info.ops:
            assert "Pure" in op.traits


# ---------------------------------------------------------------------------
# TableGen generation tests
# ---------------------------------------------------------------------------


class TestTableGenEmission:
    """Test TableGen file generation."""

    def test_layout_dialect_td(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect
        from compgen.extensions.mlir_cppgen.tablegen_emitter import emit_dialect_td

        info = introspect_layout_dialect()
        content = emit_dialect_td(info)

        assert 'def Layout_Dialect : Dialect' in content
        assert 'let name = "layout"' in content
        assert 'let cppNamespace = "::compgen::layout"' in content
        assert 'let useDefaultAttributePrinterParser = 1' in content

    def test_layout_attrs_td(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect
        from compgen.extensions.mlir_cppgen.tablegen_emitter import emit_attrs_td

        info = introspect_layout_dialect()
        content = emit_attrs_td(info)

        assert "def Layout_LayoutEncodingAttr" in content
        assert "def Layout_PackSpecAttr" in content
        assert "StrAttr:$op_type" in content
        assert "I64Attr:$operand_index" in content
        assert "ArrayAttr:$tile_dims" in content

    def test_layout_ops_td(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_layout_dialect
        from compgen.extensions.mlir_cppgen.tablegen_emitter import emit_ops_td

        info = introspect_layout_dialect()
        content = emit_ops_td(info)

        assert 'def Layout_SetLayoutOp : Layout_Op<"set_layout"' in content
        assert "Layout_LayoutEncodingAttr:$encoding" in content
        assert "FlatSymbolRefAttr:$source_ref" in content
        assert "OptionalAttr<RecipeBase_ProvenanceAttr>:$provenance" in content

        # Only PackOp should have hasVerifier
        assert content.count("let hasVerifier = 1") == 1

    def test_tile_ops_td_verifiers(self) -> None:
        from compgen.extensions.mlir_cppgen.introspect import introspect_tile_dialect
        from compgen.extensions.mlir_cppgen.tablegen_emitter import emit_ops_td

        info = introspect_tile_dialect()
        content = emit_ops_td(info)

        # 4 ops with verifiers: MMA, Elementwise, Reduce, Barrier
        assert content.count("let hasVerifier = 1") == 4


# ---------------------------------------------------------------------------
# End-to-end generation tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test full compiler generation."""

    def test_generate_all_dialects(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout", "tile", "accel"],
            output_dir=tmp_path / "compiler",
        )
        assert output.exists()

        # Check directory structure
        assert (output / "CMakeLists.txt").exists()
        assert (output / "include" / "Layout" / "LayoutDialect.td").exists()
        assert (output / "include" / "Layout" / "LayoutAttrs.td").exists()
        assert (output / "include" / "Layout" / "LayoutOps.td").exists()
        assert (output / "include" / "Tile" / "TileDialect.td").exists()
        assert (output / "include" / "Tile" / "TileOps.td").exists()
        assert (output / "include" / "RecipeBase" / "RecipeBaseDialect.td").exists()
        assert (output / "include" / "RecipeBase" / "RecipeBaseAttrs.td").exists()
        assert (output / "lib" / "Layout" / "LayoutDialect.cpp").exists()
        assert (output / "lib" / "Layout" / "LayoutOps.cpp").exists()
        assert (output / "lib" / "Tile" / "TileOps.cpp").exists()
        assert (output / "compgen-opt" / "compgen-opt.cpp").exists()

    def test_generate_with_docker(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout"],
            output_dir=tmp_path / "compiler",
            include_docker=True,
        )
        assert (output / "Dockerfile").exists()
        content = (output / "Dockerfile").read_text()
        assert "compgen-opt" in content

    def test_generate_layout_only(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout"],
            output_dir=tmp_path / "compiler",
        )

        # Should still have RecipeBase (shared dep)
        assert (output / "include" / "RecipeBase").exists()
        # Should NOT have Tile or Accel
        assert not (output / "include" / "Tile").exists()
        assert not (output / "include" / "CompgenAccel").exists()

    def test_driver_includes_all_dialects(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout", "tile", "accel"],
            output_dir=tmp_path / "compiler",
        )
        driver = (output / "compgen-opt" / "compgen-opt.cpp").read_text()

        assert "LayoutDialect" in driver
        assert "TileDialect" in driver
        assert "CompgenAccelDialect" in driver
        assert "RecipeBaseDialect" in driver
        assert "MlirOptMain" in driver

    def test_cmake_structure(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout"],
            output_dir=tmp_path / "compiler",
        )

        top_cmake = (output / "CMakeLists.txt").read_text()
        assert "find_package(MLIR REQUIRED CONFIG)" in top_cmake
        assert "add_subdirectory(include)" in top_cmake
        assert "add_subdirectory(lib)" in top_cmake
        assert "add_subdirectory(compgen-opt)" in top_cmake

        include_cmake = (output / "include" / "CMakeLists.txt").read_text()
        assert "add_subdirectory(Layout)" in include_cmake

        lib_cmake = (output / "lib" / "Layout" / "CMakeLists.txt").read_text()
        assert "add_mlir_dialect_library" in lib_cmake

    def test_unknown_dialect_raises(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        with pytest.raises(ValueError, match="Unknown dialect"):
            generate_compiler(
                dialects=["nonexistent"],
                output_dir=tmp_path / "compiler",
            )

    def test_generate_with_passes(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout", "tile", "accel"],
            output_dir=tmp_path / "compiler",
            include_passes=["layout"],
        )

        # Check pass files exist
        assert (output / "include" / "Layout" / "LayoutPasses.td").exists()
        assert (output / "include" / "Layout" / "LayoutPasses.h").exists()
        assert (output / "lib" / "Layout" / "Passes" / "CMakeLists.txt").exists()
        assert (output / "lib" / "Layout" / "Passes" / "PropagateLayouts.cpp").exists()
        assert (output / "lib" / "Layout" / "Passes" / "CanonicalizeTransposes.cpp").exists()
        assert (output / "lib" / "Layout" / "Passes" / "MaterializeBoundaries.cpp").exists()
        assert (output / "lib" / "Layout" / "Passes" / "CleanupArtifacts.cpp").exists()

        # All 10 passes
        passes_dir = output / "lib" / "Layout" / "Passes"
        cpp_files = list(passes_dir.glob("*.cpp"))
        assert len(cpp_files) == 10

    def test_generate_with_tests(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout", "tile"],
            output_dir=tmp_path / "compiler",
        )

        assert (output / "test" / "Layout" / "roundtrip.mlir").exists()
        assert (output / "test" / "Tile" / "roundtrip.mlir").exists()
        assert (output / "test" / "lit.cfg.py").exists()
        assert (output / "test" / "Passes" / "propagate_layouts.mlir").exists()

    def test_driver_registers_passes(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout"],
            output_dir=tmp_path / "compiler",
            include_passes=["layout"],
        )
        driver = (output / "compgen-opt" / "compgen-opt.cpp").read_text()

        assert '#include "Layout/LayoutPasses.h"' in driver
        assert "registerPasses()" in driver

    def test_lib_cmake_includes_passes(self, tmp_path: Path) -> None:
        from compgen.extensions.mlir_cppgen import generate_compiler

        output = generate_compiler(
            dialects=["layout"],
            output_dir=tmp_path / "compiler",
            include_passes=["layout"],
        )
        lib_cmake = (output / "lib" / "Layout" / "CMakeLists.txt").read_text()
        assert "add_subdirectory(Passes)" in lib_cmake


# ---------------------------------------------------------------------------
# Pass emitter tests
# ---------------------------------------------------------------------------


class TestPassEmitter:
    """Test pass C++ code generation."""

    def test_layout_passes_count(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import get_layout_passes

        passes = get_layout_passes()
        assert len(passes) == 10

    def test_pass_names(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import get_layout_passes

        passes = get_layout_passes()
        names = [p.name for p in passes]
        assert "propagate_layouts" in names
        assert "canonicalize_transposes" in names
        assert "materialize_layout_boundaries" in names
        assert "cleanup_layout_artifacts" in names

    def test_passes_td_content(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import emit_passes_td, get_layout_passes

        content = emit_passes_td("Layout", get_layout_passes())
        assert "layout-propagate-layouts" in content
        assert "layout-canonicalize-transposes" in content
        assert "layout-cleanup-artifacts" in content
        assert "::mlir::ModuleOp" in content

    def test_pass_cpp_propagate(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import emit_pass_cpp, get_layout_passes

        passes = get_layout_passes()
        prop = next(p for p in passes if p.name == "propagate_layouts")
        content = emit_pass_cpp(prop, "Layout", "compgen::layout")

        assert "PropagateLayoutsPass" in content
        assert "valueEncoding" in content
        assert "isTransparent" in content
        assert 'compgen.propagated_encoding' in content
        assert "arith." in content
        assert "math." in content

    def test_pass_cpp_hoist(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import emit_pass_cpp, get_layout_passes

        passes = get_layout_passes()
        hoist = next(p for p in passes if p.name == "hoist_layout_ops")
        content = emit_pass_cpp(hoist, "Layout", "compgen::layout")

        assert "HoistLayoutOpsPass" in content
        assert "0.8" in content  # 80% threshold
        assert "compgen.hoisted_encoding" in content

    def test_pass_cpp_enum_check_in_set_virtual(self) -> None:
        from compgen.extensions.mlir_cppgen.pass_emitter import emit_pass_cpp, get_layout_passes

        passes = get_layout_passes()
        sv = next(p for p in passes if p.name == "set_virtual_encodings")
        content = emit_pass_cpp(sv, "Layout", "compgen::layout")

        assert "linalg.matmul" in content
        assert "linalg.generic" in content
        assert "func.call" in content


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestRunner:
    """Test the compgen-opt runner wrapper."""

    def test_layout_pipeline_passes(self) -> None:
        from compgen.extensions.mlir_cppgen.runner import LAYOUT_PIPELINE_PASSES

        assert len(LAYOUT_PIPELINE_PASSES) == 10
        assert "--layout-propagate-layouts" in LAYOUT_PIPELINE_PASSES
        assert "--layout-cleanup-artifacts" in LAYOUT_PIPELINE_PASSES

    def test_find_compgen_opt_not_found(self) -> None:
        from compgen.extensions.mlir_cppgen.runner import find_compgen_opt

        result = find_compgen_opt(search_paths=["/nonexistent/path"])
        # Should return None when binary doesn't exist
        assert result is None or result.is_file()
