"""MCP vendor-dialect tools: scan → scaffold → verify against fake vendor."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from compgen.mcp.tools.vendor_dialect import (
    VENDOR_DIALECT_TOOLS,
    scan_vendor_repo,
    scaffold_vendor_package,
    verify_vendor_package,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "fake_vendor"


class _StubSessionManager:
    """Placeholder — the vendor tools are session-less, so just a sentinel."""


def test_tool_registry_lists_all_four_tools() -> None:
    names = [t["name"] for t in VENDOR_DIALECT_TOOLS]
    assert set(names) == {
        "scan_vendor_repo",
        "propose_vendor_spec",
        "scaffold_vendor_package",
        "verify_vendor_package",
    }
    for tool in VENDOR_DIALECT_TOOLS:
        assert callable(tool["handler"])
        assert "input_schema" in tool


def test_scan_returns_descriptor_yaml() -> None:
    res = scan_vendor_repo(
        _StubSessionManager(),
        repo_path=str(FIXTURE),
        target="toy-target",
        workloads=["tinyllama"],
    )
    assert res["ok"]
    assert "descriptor_yaml" in res
    parsed = yaml.safe_load(res["descriptor_yaml"])
    assert parsed["target"] == "toy-target"
    assert parsed["repo_path"].endswith("fake_vendor")
    assert "scan" in res
    assert res["scan"]["num_td_ops"] >= 2


def test_scaffold_rejects_conflicting_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold_vendor_package(
            _StubSessionManager(),
            descriptor_yaml="dummy",
            descriptor_path=str(tmp_path / "d.yaml"),
            out_dir=str(tmp_path),
        )


def test_full_flow_scan_scaffold_verify(tmp_path: Path, monkeypatch) -> None:
    scan_res = scan_vendor_repo(
        _StubSessionManager(),
        repo_path=str(FIXTURE),
        target="toy-target",
    )
    descriptor_yaml = scan_res["descriptor_yaml"]

    out_dir = tmp_path / "user_perspective_MLIR"
    scaffold_res = scaffold_vendor_package(
        _StubSessionManager(),
        descriptor_yaml=descriptor_yaml,
        out_dir=str(out_dir),
    )
    assert scaffold_res["ok"]
    pkg_dir = Path(scaffold_res["package_dir"])
    assert pkg_dir.is_dir()
    assert (pkg_dir / "pyproject.toml").is_file()

    # Make the scaffolded package importable for verify_vendor_package.
    monkeypatch.syspath_prepend(str(pkg_dir))

    verify_res = verify_vendor_package(
        _StubSessionManager(),
        package_dir=str(pkg_dir),
    )
    assert verify_res["ok"]
    gate_names = [g["name"] for g in verify_res["report"]["gates"]]
    assert "structural" in gate_names
