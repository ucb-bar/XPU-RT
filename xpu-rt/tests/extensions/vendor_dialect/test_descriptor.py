"""Descriptor round-trips cleanly through YAML."""

from __future__ import annotations

from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    OpEntry,
    VendorDialectDescriptor,
    VerificationPlan,
)


def _make() -> VendorDialectDescriptor:
    return VendorDialectDescriptor(
        name="fake",
        package_name="compgen_fake",
        repo_path="/tmp/fake",
        target="nvidia-h100",
        input_ir=("linalg", "tosa"),
        output_format="fake_bin",
        compile_entry=CompileEntry(
            cli_tools=("fake-opt", "fake-translate"),
            python_module="fake._bindings",
            python_symbols=("register_dialect",),
        ),
        td_files=("include/fake/FakeDialect.td",),
        op_registry=(
            OpEntry(name="fake.matmul", summary="tiled matmul"),
            OpEntry(name="fake.softmax", summary="row softmax"),
        ),
        lowering=LoweringStrategy(
            mode="kernel_authoring",
            op_families=("matmul", "softmax"),
            template_ops=("matmul",),
        ),
        bundle=BundlePlan(
            steps=("fake-opt --canonicalize", "fake-translate -o out"),
            output_format="fake_bin",
            runtime_entry="fake::launch",
        ),
        verification=VerificationPlan(
            structural=True,
            matmul_diff_test=True,
            workload_diff_test=True,
            workloads=("tinyllama",),
            tolerance_atol=1e-2,
        ),
        kernel_authoring_required=True,
        dependencies=("numpy>=1.26",),
        license="Apache-2.0",
        extras={"source": "test"},
    )


def test_roundtrip_yaml() -> None:
    original = _make()
    reloaded = VendorDialectDescriptor.from_yaml(original.to_yaml())
    assert reloaded == original


def test_write_and_load(tmp_path) -> None:
    d = _make()
    path = d.write(tmp_path / "descriptor.yaml")
    assert path.is_file()
    loaded = VendorDialectDescriptor.load(path)
    assert loaded == d
