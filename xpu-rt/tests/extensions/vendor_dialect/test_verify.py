"""End-to-end verify harness: scaffold → verify_package."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.scaffold import scaffold_package
from compgen.extensions.vendor_dialect.verify import verify_package


def _descriptor() -> VendorDialectDescriptor:
    return VendorDialectDescriptor(
        name="toyv",
        package_name="compgen_toyv",
        repo_path="/tmp/toyv",
        target="toyv-target",
        input_ir=("linalg",),
        output_format="toyv_bin",
        compile_entry=CompileEntry(cli_tools=("toyv-opt",)),
        lowering=LoweringStrategy(mode="direct_linalg"),
        bundle=BundlePlan(steps=("toyv-opt",), output_format="toyv_bin"),
        verification=VerificationPlan(structural=True, matmul_diff_test=True, workload_diff_test=False),
        kernel_authoring_required=False,
        license="Apache-2.0",
    )


def test_verify_structural_and_matmul_gates_pass(tmp_path: Path, monkeypatch) -> None:
    result = scaffold_package(_descriptor(), tmp_path)
    monkeypatch.syspath_prepend(str(result.package_dir))
    sys.modules.pop("compgen_toyv", None)

    report = verify_package(result.package_dir)
    assert report.adapter_name == "toyv"
    assert {g.name for g in report.gates} == {"structural", "matmul_diff"}
    assert all(g.passed for g in report.gates)
    assert report.passed
