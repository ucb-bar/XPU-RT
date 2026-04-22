"""Scaffolded packages are syntactically valid and import cleanly."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest
from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.scaffold import scaffold_package


def _descriptor(kernel_authoring: bool = False) -> VendorDialectDescriptor:
    return VendorDialectDescriptor(
        name="toy",
        package_name="compgen_toy",
        repo_path="/tmp/toy",
        target="toy-target",
        input_ir=("linalg",),
        output_format="toy_bin",
        compile_entry=CompileEntry(cli_tools=("toy-opt",)),
        lowering=LoweringStrategy(
            mode="kernel_authoring" if kernel_authoring else "direct_linalg",
            op_families=("matmul",),
        ),
        bundle=BundlePlan(steps=("toy-opt",), output_format="toy_bin"),
        verification=VerificationPlan(matmul_diff_test=False),
        kernel_authoring_required=kernel_authoring,
        license="Apache-2.0",
    )


def test_scaffold_emits_expected_files(tmp_path: Path) -> None:
    result = scaffold_package(_descriptor(), tmp_path)
    assert result.package_dir == tmp_path / "compgen_toy"
    must_exist = [
        "pyproject.toml",
        "README.md",
        "compgen_toy/__init__.py",
        "compgen_toy/adapter.py",
        "compgen_toy/lowering.py",
        "compgen_toy/bundle.py",
        "compgen_toy/descriptor.yaml",
        "compgen_toy/tests/test_smoke.py",
        "examples/workload.py",
    ]
    for rel in must_exist:
        assert (result.package_dir / rel).is_file(), rel


def test_scaffold_rendered_python_is_valid(tmp_path: Path) -> None:
    result = scaffold_package(_descriptor(kernel_authoring=True), tmp_path)
    for py in result.package_dir.rglob("*.py"):
        src = py.read_text()
        ast.parse(src, filename=str(py))


def test_scaffold_refuses_to_overwrite(tmp_path: Path) -> None:
    scaffold_package(_descriptor(), tmp_path)
    with pytest.raises(FileExistsError):
        scaffold_package(_descriptor(), tmp_path)


def test_scaffold_overwrite(tmp_path: Path) -> None:
    scaffold_package(_descriptor(), tmp_path)
    scaffold_package(_descriptor(), tmp_path, overwrite=True)


def test_scaffolded_package_imports(tmp_path: Path, monkeypatch) -> None:
    """The emitted package imports and its ``load_adapter`` returns an adapter."""
    result = scaffold_package(_descriptor(), tmp_path)
    # Put the package dir on sys.path and import fresh.
    monkeypatch.syspath_prepend(str(result.package_dir))
    sys.modules.pop("compgen_toy", None)
    pkg = importlib.import_module("compgen_toy")
    adapter = pkg.load_adapter()
    from compgen.extensions.vendor_dialect.adapter import VendorDialectAdapter

    assert isinstance(adapter, VendorDialectAdapter)
    assert adapter.name == "toy"
    assert adapter.target == "toy-target"


def test_scaffolded_package_with_kernel_authoring_defers_provider(tmp_path: Path, monkeypatch) -> None:
    """When kernel_authoring_required=True, ``kernels.py`` wires a provider.

    Phase-A does not yet ship :class:`ClaudeKernelProvider`, so importing
    the inner ``kernels`` module may fail; the top-level ``load_adapter``
    must still return a working adapter when the Phase-B provider lands.
    For Phase-A we just assert that the emitted file is syntactically
    valid Python.
    """
    result = scaffold_package(_descriptor(kernel_authoring=True), tmp_path)
    src = (result.package_dir / "compgen_toy" / "kernels.py").read_text()
    ast.parse(src)
    assert "build_toy_kernel_provider" in src
