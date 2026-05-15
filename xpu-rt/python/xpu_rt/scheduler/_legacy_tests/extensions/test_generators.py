"""Tests for declarative xDSL and LLVM extension generation."""

from __future__ import annotations

from compgen.extensions import generate_llvm_patch_bundle, generate_xdsl_dialect


def test_generate_xdsl_dialect_from_dict() -> None:
    files = generate_xdsl_dialect(
        {
            "name": "my_npu",
            "python_module": "my_npu",
            "ops": [
                {
                    "name": "mma",
                    "operands": [{"name": "lhs", "type_expr": "AnyAttr()"}],
                    "results": [{"name": "out", "type_expr": "AnyAttr()"}],
                    "traits": ["Pure"],
                    "summary": "matrix multiply-accumulate",
                }
            ],
        }
    )

    assert "dialect.py" in files
    assert 'name = "my_npu.mma"' in files["dialect.py"]
    assert "matrix multiply-accumulate" in files["README.md"]


def test_generate_llvm_patch_bundle_from_dict() -> None:
    files = generate_llvm_patch_bundle(
        {
            "dialect_name": "my_npu",
            "intrinsics": [
                {
                    "name": "llvm.my_npu.mma",
                    "ret_type": "llvm_any_ty",
                    "arg_types": ["llvm_any_ty", "llvm_any_ty"],
                    "summary": "matrix multiply-accumulate",
                }
            ],
        }
    )

    assert "IntrinsicsMyNpu.td" in files
    assert "llvm.my_npu.mma" in files["my_npu_intrinsics.ll"]
    assert "int_llvm_my_npu_mma" in files["IntrinsicsMyNpu.td"]
