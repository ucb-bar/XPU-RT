"""Phase D gap closure — Batch D regression tests.

Covers gaps #10, #11, #12, #13:

- #10: coverage extends to pointwise + reduce archetypes (was matmul-only).
- #11: NumericalWithinEps runtime emit via assert_postconditions.
- #12: DtypeIn precondition emit (was skipped as redundant; now emits
  with dtype-alias normalisation).
- #13: multi-file user kernels (kernel_source accepts str or list).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Gap #10 — coverage covers pointwise + reduce
# --------------------------------------------------------------------------- #


class TestGap10CoverageBreadth:
    def test_pointwise_dossier_produces_signature(self) -> None:
        from compgen.graph_compilation.coverage_first import _coverage_signature

        dossier = {
            "kind": "elementwise_relu",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 32]],
                "output_shapes": [[16, 32]],
            },
        }
        sig = _coverage_signature(dossier=dossier, target_name="host_cpu")
        assert sig
        assert "pointwise" in sig

    def test_reduce_dossier_produces_signature(self) -> None:
        from compgen.graph_compilation.coverage_first import _coverage_signature

        dossier = {
            "kind": "reduce_sum",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 32]],
                "output_shapes": [[16]],
            },
        }
        sig = _coverage_signature(dossier=dossier, target_name="host_cpu")
        assert sig
        assert "reduce" in sig

    def test_unknown_kind_returns_empty_signature(self) -> None:
        from compgen.graph_compilation.coverage_first import _coverage_signature

        dossier = {
            "kind": "invent_new_archetype",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16]],
                "output_shapes": [[16]],
            },
        }
        sig = _coverage_signature(dossier=dossier, target_name="host_cpu")
        assert sig == ""


# --------------------------------------------------------------------------- #
# Gap #11 — NumericalWithinEps runtime
# --------------------------------------------------------------------------- #


class TestGap11NumericalWithinEpsRuntime:
    def test_render_no_postconditions_returns_noop(self) -> None:
        from compgen.runtime.glue_emit.plan_assertions import (
            _RegionAssertions,
            render_assert_postconditions_body,
        )

        body = render_assert_postconditions_body([])
        # Empty body returns early; emitter inserts a "no postconditions" comment.
        assert "no-op" in body or "return" in body

    def test_render_emits_numerical_within_eps_check(self) -> None:
        from compgen.runtime.glue_emit.plan_assertions import (
            _RegionAssertions,
            render_assert_postconditions_body,
        )

        ra = _RegionAssertions(
            region_id="matmul_0",
            contract_hash="abc",
            inputs=[], outputs=[], accumulator_dtype="f32",
            aliasing=[], in_place_safe=False, event_decls=[],
            postconditions=[{
                "kind": "numerical_within_eps",
                "out": "out",
                "ref": "reference",
                "eps": 1e-5,
            }],
        )
        body = render_assert_postconditions_body([ra])
        assert "PLAN_VIOLATION_POSTCONDITION_NUMERICAL_WITHIN_EPS" in body
        assert "numerical_within_eps('out', ref='reference', eps=1e-05)" in body


# --------------------------------------------------------------------------- #
# Gap #12 — DtypeIn precondition emit
# --------------------------------------------------------------------------- #


class TestGap12DtypeInEmit:
    def test_dtype_in_emit_with_alias_normalisation(self) -> None:
        from compgen.runtime.glue_emit.plan_assertions import (
            _RegionAssertions,
            render_assert_plan_body,
        )

        ra = _RegionAssertions(
            region_id="matmul_0",
            contract_hash="abc",
            inputs=[{
                "name": "lhs", "dims": [16, 16],
                "dtype_class": ["fp32"], "layout": "row_major",
                "expected_bytes": 1024,
            }],
            outputs=[], accumulator_dtype="f32",
            aliasing=[], in_place_safe=False, event_decls=[],
            preconditions=[{
                "kind": "dtype_in",
                "arg": "lhs",
                "dtype_set": ["fp32", "fp16"],
            }],
        )
        body = render_assert_plan_body([ra])
        # The DtypeIn allowlist is normalised: fp32 → f32, fp16 → f16.
        assert "PLAN_VIOLATION_PRECONDITION_DTYPE_IN" in body
        # The emitted allowed_dtypes list has the canonical forms.
        assert "'f16'" in body and "'f32'" in body


# --------------------------------------------------------------------------- #
# Gap #13 — multi-file user kernels
# --------------------------------------------------------------------------- #


class TestGap13MultiFileUserKernels:
    def test_kernel_source_accepts_list(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
            index_one_manifest,
        )

        root = tmp_path / "k"
        root.mkdir()
        # Write a main + helper file.
        (root / "main.c").write_text(
            "void user_kernel(void) {}\n", encoding="utf-8",
        )
        (root / "helper.h").write_text("/* helper */\n", encoding="utf-8")
        manifest_body = {
            "schema_version": "user_kernel_manifest_v1",
            "op_name": "test_op",
            "archetype": "compute_tiled",
            "target_name": "host_cpu",
            "language": "c",
            "kernel_source": ["main.c", "helper.h"],
            "entry_symbol": "user_kernel",
            "inputs": [{"name": "x", "dtype": "f32", "layout": "row_major"}],
            "outputs": [{"name": "y", "dtype": "f32", "layout": "row_major"}],
            "numerics": {"accumulator_dtype": "f32"},
        }
        try:
            import yaml
            (root / "kernel_manifest.yaml").write_text(
                yaml.safe_dump(manifest_body, sort_keys=True), encoding="utf-8",
            )
        except ImportError:
            (root / "kernel_manifest.yaml").write_text(
                json.dumps(manifest_body), encoding="utf-8",
            )

        entry = index_one_manifest(
            manifest_path=root / "kernel_manifest.yaml",
            index_root=tmp_path / "index",
        )
        # Locked-files map carries SHAs for manifest + both source files.
        assert "kernel_manifest.yaml" in entry.locked_files
        assert "main.c" in entry.locked_files
        assert "helper.h" in entry.locked_files
        # Manifest internal form is a tuple.
        assert entry.manifest.kernel_source == ("main.c", "helper.h")
        assert entry.manifest.primary_kernel_source == "main.c"

    def test_kernel_source_string_still_works_back_compat(self, tmp_path: Path) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
            index_one_manifest,
        )

        root = tmp_path / "k"
        root.mkdir()
        (root / "main.c").write_text("void f(void) {}\n", encoding="utf-8")
        manifest_body = {
            "schema_version": "user_kernel_manifest_v1",
            "op_name": "test_op",
            "archetype": "compute_tiled",
            "target_name": "host_cpu",
            "language": "c",
            "kernel_source": "main.c",  # string form
            "entry_symbol": "f",
            "inputs": [{"name": "x", "dtype": "f32", "layout": "row_major"}],
            "outputs": [{"name": "y", "dtype": "f32", "layout": "row_major"}],
            "numerics": {"accumulator_dtype": "f32"},
        }
        try:
            import yaml
            (root / "kernel_manifest.yaml").write_text(
                yaml.safe_dump(manifest_body, sort_keys=True), encoding="utf-8",
            )
        except ImportError:
            (root / "kernel_manifest.yaml").write_text(
                json.dumps(manifest_body), encoding="utf-8",
            )
        entry = index_one_manifest(
            manifest_path=root / "kernel_manifest.yaml",
            index_root=tmp_path / "index",
        )
        # String form converts to single-element tuple.
        assert entry.manifest.kernel_source == ("main.c",)
        # Serialisation back to string for single-file.
        assert entry.manifest.to_dict()["kernel_source"] == "main.c"

    def test_kernel_source_unsupported_type_raises(self) -> None:
        from compgen.kernels.user_kernel_index import (
            UserKernelManifest,
            UserKernelManifestError,
        )

        body = {
            "schema_version": "user_kernel_manifest_v1",
            "op_name": "test_op",
            "archetype": "compute_tiled",
            "target_name": "host_cpu",
            "language": "c",
            "kernel_source": 42,  # invalid
            "entry_symbol": "f",
            "inputs": [{"name": "x", "dtype": "f32", "layout": "row_major"}],
            "outputs": [{"name": "y", "dtype": "f32", "layout": "row_major"}],
            "numerics": {"accumulator_dtype": "f32"},
        }
        with pytest.raises(UserKernelManifestError, match="must be str or list"):
            UserKernelManifest.from_dict(body)
