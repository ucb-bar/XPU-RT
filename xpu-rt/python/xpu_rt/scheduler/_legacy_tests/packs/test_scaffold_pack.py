"""Tests for ``compgen scaffold-pack`` and :func:`scaffold_pack`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from compgen.cli import main
from compgen.packs import load_pack
from compgen.packs.scaffolding import SUPPORTED_KINDS, scaffold_pack


@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
def test_scaffold_pack_writes_complete_skeleton(tmp_path: Path, kind: str) -> None:
    result = scaffold_pack(kind=kind, name="my_pack", out_dir=tmp_path)

    assert result.pack_root == tmp_path / "my_pack"
    assert (result.pack_root / "pyproject.toml").exists()
    assert (result.pack_root / "README.md").exists()
    assert (result.package_root / "__init__.py").exists()
    assert (result.package_root / "manifest.yaml").exists()
    assert result.scheme_path.exists()
    assert result.scheme_path.parent == result.package_root


def test_scaffolded_manifest_is_valid_yaml(tmp_path: Path) -> None:
    result = scaffold_pack(kind="quantization", name="my_fp4", out_dir=tmp_path)
    data = yaml.safe_load(result.manifest_path.read_text())
    assert data["name"] == "my_fp4"
    assert data["kinds"] == ["KernelPack"]
    assert data["source_root"] == "src/my_fp4"


def test_scaffolded_pyproject_declares_entry_point(tmp_path: Path) -> None:
    result = scaffold_pack(kind="provider", name="my_provider", out_dir=tmp_path)
    content = result.pyproject_path.read_text()
    assert '[project.entry-points."compgen.packs"]' in content
    assert 'my_provider = "my_provider"' in content


def test_scaffolded_package_exposes_pack_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = scaffold_pack(kind="dialect", name="my_dial", out_dir=tmp_path)
    # Make the scaffolded src directory importable
    monkeypatch.syspath_prepend(str(result.pack_root / "src"))
    module = __import__("my_dial")
    try:
        assert isinstance(module.PACK_ROOT, Path)
        assert (module.PACK_ROOT / "manifest.yaml").exists()
    finally:
        sys.modules.pop("my_dial", None)


def test_scaffold_then_load_pack_by_entry_point_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "my_fp8" / "src"))

    # Entry-point value form: resolves module → __file__.parent → pack root
    loaded = load_pack("my_fp8")
    try:
        assert loaded.manifest.name == "my_fp8"
        assert "KernelPack" in loaded.manifest.kinds
    finally:
        sys.modules.pop("my_fp8", None)


def test_scaffold_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="1bad", out_dir=tmp_path)
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="my-pack", out_dir=tmp_path)
    with pytest.raises(ValueError):
        scaffold_pack(kind="quantization", name="class", out_dir=tmp_path)


def test_scaffold_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_pack(kind="bogus", name="mypack", out_dir=tmp_path)


def test_scaffold_existing_directory_without_overwrite_errors(tmp_path: Path) -> None:
    scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    with pytest.raises(FileExistsError):
        scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)


def test_scaffold_overwrite_replaces_existing(tmp_path: Path) -> None:
    first = scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    # Drop a sentinel file that should be erased by overwrite
    sentinel = first.pack_root / "LEGACY.txt"
    sentinel.write_text("old")
    second = scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path, overwrite=True)
    assert not sentinel.exists()
    assert second.pack_root == first.pack_root


def test_cli_scaffold_pack_happy_path(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["scaffold-pack", "--kind", "provider", "--name", "my_prov", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "my_prov" / "pyproject.toml").exists()
    assert (tmp_path / "my_prov" / "src" / "my_prov" / "manifest.yaml").exists()


def test_cli_scaffold_pack_bad_kind_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["scaffold-pack", "--kind", "bogus", "--name", "mypack", "--out", str(tmp_path)],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# target_pack — full multi-surface scaffold
# ---------------------------------------------------------------------------


def test_target_pack_emits_full_radiance_shaped_layout(tmp_path: Path) -> None:
    """``target_pack`` produces Backend + Provider + MCP + HardwareSpec + tests."""
    result = scaffold_pack(kind="target_pack", name="acme_npu", out_dir=tmp_path)

    pkg = result.package_root
    assert (pkg / "__init__.py").exists()
    assert (pkg / "manifest.yaml").exists()
    assert (pkg / "backend.py").exists()
    assert (pkg / "kernels.py").exists()
    assert (pkg / "mcp.py").exists()
    assert (pkg / "specs" / "acme_npu.yaml").exists()
    assert (result.pack_root / "tests" / "test_pack_smoke.py").exists()


def test_target_pack_pyproject_declares_all_four_entry_points(tmp_path: Path) -> None:
    result = scaffold_pack(kind="target_pack", name="acme_npu", out_dir=tmp_path)
    content = result.pyproject_path.read_text()
    assert '[project.entry-points."compgen.packs"]' in content
    assert '[project.entry-points."compgen.targets.backends"]' in content
    assert '[project.entry-points."compgen.kernels.providers"]' in content
    assert '[project.entry-points."compgen.mcp.tools"]' in content
    # Class-name derivation: snake_case → CamelCase.
    assert "AcmeNpuBackend" in content
    assert "AcmeNpuProvider" in content
    assert "ACME_NPU_TOOLS" in content


def test_target_pack_provider_declines_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Provider must satisfy the ``KernelProvider`` Protocol but
    decline every contract until the user fills it in."""
    scaffold_pack(kind="target_pack", name="acme_npu", out_dir=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "acme_npu" / "src"))

    try:
        from compgen.kernels.provider import KernelContract, SearchBudget

        provider_module = __import__("acme_npu.kernels", fromlist=["AcmeNpuProvider"])
        provider = provider_module.AcmeNpuProvider()
        contract = KernelContract(
            region_id="r0",
            op_family="add",
            input_shapes=((4,),),
            output_shapes=((4,),),
            dtypes=("f32",),
            target_name="acme_npu",
            hardware_key="",
            objective="latency",
        )
        assert provider.name == "acme_npu"
        assert provider.accepts_contract(contract) is False
        result = provider.search(contract, SearchBudget())
        assert result.found is False
    finally:
        sys.modules.pop("acme_npu", None)
        sys.modules.pop("acme_npu.kernels", None)
        sys.modules.pop("acme_npu.backend", None)
        sys.modules.pop("acme_npu.mcp", None)


def test_target_pack_hardware_spec_loads(tmp_path: Path) -> None:
    """The scaffolded HardwareSpec stub must load through CompGen's loader."""
    from compgen.targetgen.load import load_hardware_spec

    result = scaffold_pack(kind="target_pack", name="acme_npu", out_dir=tmp_path)
    spec_path = result.package_root / "specs" / "acme_npu.yaml"
    spec = load_hardware_spec(str(spec_path))
    assert spec.name == "acme_npu"


def test_target_pack_mcp_tools_list_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty MCP tools list passes the ``compgen.mcp.tools`` validator."""
    scaffold_pack(kind="target_pack", name="acme_npu", out_dir=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path / "acme_npu" / "src"))

    try:
        from compgen.plugins import _VALIDATORS

        mcp_module = __import__("acme_npu.mcp", fromlist=["ACME_NPU_TOOLS"])
        validator = _VALIDATORS["compgen.mcp.tools"]
        ok, _msg = validator(mcp_module.ACME_NPU_TOOLS)
        assert ok, f"validator rejected the scaffolded MCP tools list: {_msg}"
    finally:
        for k in list(sys.modules):
            if k == "acme_npu" or k.startswith("acme_npu."):
                sys.modules.pop(k, None)
