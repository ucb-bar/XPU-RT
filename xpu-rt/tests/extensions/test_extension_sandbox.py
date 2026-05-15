"""extension sandbox path enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.extensions.errors import ExtensionSandboxViolation
from compgen.extensions.sandbox import (
    is_under_sandbox,
    validate_sandboxed_path,
)


def test_path_inside_sandbox_is_returned_resolved(tmp_path: Path):
    root = tmp_path / "ext_a"
    root.mkdir()
    resolved = validate_sandboxed_path(root / "kernel.c", allowed_write_root=root)
    assert resolved == (root / "kernel.c").resolve()


def test_path_traversal_via_dotdot_rejected(tmp_path: Path):
    root = tmp_path / "ext_a"
    root.mkdir()
    with pytest.raises(ExtensionSandboxViolation) as exc:
        validate_sandboxed_path(root / ".." / "evil.c", allowed_write_root=root)
    assert exc.value.reason == "escapes_allowed_root"


def test_absolute_path_outside_root_rejected(tmp_path: Path):
    root = tmp_path / "ext_a"
    root.mkdir()
    sibling = tmp_path / "elsewhere" / "file.c"
    with pytest.raises(ExtensionSandboxViolation) as exc:
        validate_sandboxed_path(sibling, allowed_write_root=root)
    assert exc.value.reason == "escapes_allowed_root"


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "payload.mlir",
        "execution_plan.yaml",
        "memory_plan.yaml",
        "compgen_extension.yaml",
        "kernel_contract.yaml",
        "run_manifest.json",
        "manifest.json",
    ],
)
def test_forbidden_filenames_rejected_even_inside_sandbox(
    tmp_path: Path, forbidden_name: str
):
    root = tmp_path / "ext_a"
    root.mkdir()
    with pytest.raises(ExtensionSandboxViolation) as exc:
        validate_sandboxed_path(root / forbidden_name, allowed_write_root=root)
    assert exc.value.reason.startswith("forbidden_filename:")


def test_subdirectory_within_sandbox_is_allowed(tmp_path: Path):
    root = tmp_path / "ext_a"
    (root / "nested" / "deeper").mkdir(parents=True)
    p = validate_sandboxed_path(
        root / "nested" / "deeper" / "kernel.c", allowed_write_root=root
    )
    assert p.name == "kernel.c"


def test_is_under_sandbox_non_raising_variant(tmp_path: Path):
    root = tmp_path / "ext_a"
    root.mkdir()
    assert is_under_sandbox(root / "ok.c", root) is True
    assert is_under_sandbox(tmp_path / "elsewhere.c", root) is False
    assert is_under_sandbox(root / "payload.mlir", root) is False


def test_violation_payload_carries_path_and_root(tmp_path: Path):
    root = tmp_path / "ext_a"
    root.mkdir()
    with pytest.raises(ExtensionSandboxViolation) as exc:
        validate_sandboxed_path("/etc/passwd", allowed_write_root=root)
    assert exc.value.path.endswith("/etc/passwd") or exc.value.path == "/etc/passwd"
    assert exc.value.allowed_root == str(root.resolve())
    assert "extension_sandbox_violation" in str(exc.value)
