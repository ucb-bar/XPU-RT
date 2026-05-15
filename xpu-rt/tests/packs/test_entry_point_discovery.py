"""Tests for ``discover_packs`` and entry-point / env-var sources."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from compgen.packs import (
    discover_packs,
    load_discovered_packs,
    load_pack,
    resolve_entry_point_target,
)
from compgen.packs.registry import ENTRY_POINT_GROUP, ENV_VAR


def _write_manifest(root: Path, name: str, *, entry_module: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = textwrap.dedent(
        f"""
        name: {name}
        version: "0.1.0"
        kinds: ["KernelPack"]
        owned_surfaces: []
        sealed_surfaces: []
        generation_apertures: []
        integration_mode: readonly
        benchmark_suite: pack_integrations
        benchmark_targets: []
        reference_runner: ""
        source_root: ""
        workspace_keys: []
        third_party_names: []
        expected_files: []
        available_profilers: []
        llvm_fork_key: ""
        entry_module: "{entry_module}"
        metadata: {{}}
        """
    ).strip()
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(manifest)
    return root


class _FakeEntryPoint:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value
        self.group = ENTRY_POINT_GROUP


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, entries: list[_FakeEntryPoint]) -> None:
    def fake_entry_points(*, group: str | None = None, **_: object):
        if group == ENTRY_POINT_GROUP:
            return entries
        return []

    monkeypatch.setattr("compgen.packs.registry.importlib.metadata.entry_points", fake_entry_points)


# --- resolve_entry_point_target ------------------------------------------------


def test_resolve_entry_point_without_attr_uses_module_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "fakepack_a"
    _write_manifest(pkg, "fakepack_a")
    (pkg / "__init__.py").write_text("")

    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = resolve_entry_point_target("fakepack_a")
    assert resolved == pkg


def test_resolve_entry_point_with_path_attr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack_root = tmp_path / "fakepack_b_pack"
    _write_manifest(pack_root, "fakepack_b")

    helper = tmp_path / "fakepack_b_helper"
    helper.mkdir()
    (helper / "__init__.py").write_text(f"from pathlib import Path\nPACK_ROOT = Path({str(pack_root)!r})\n")

    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = resolve_entry_point_target("fakepack_b_helper:PACK_ROOT")
    assert resolved == pack_root


def test_resolve_entry_point_with_callable_attr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack_root = tmp_path / "fakepack_c_pack"
    _write_manifest(pack_root, "fakepack_c")

    helper = tmp_path / "fakepack_c_helper"
    helper.mkdir()
    (helper / "__init__.py").write_text(f"def get_pack_root():\n    return {str(pack_root)!r}\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = resolve_entry_point_target("fakepack_c_helper:get_pack_root")
    assert resolved == pack_root


def test_resolve_entry_point_bad_attr_type_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = tmp_path / "fakepack_d_helper"
    helper.mkdir()
    (helper / "__init__.py").write_text("PACK_ROOT = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(TypeError):
        resolve_entry_point_target("fakepack_d_helper:PACK_ROOT")


# --- discover_packs ------------------------------------------------------------


def test_discover_packs_repo_only(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_manifest(repo_root / "userpacks" / "alpha", "alpha")
    _write_manifest(repo_root / "userpacks" / "beta", "beta")

    found = discover_packs(repo_root=repo_root, include_entry_points=False, include_env=False)
    names = [p.name for p in found]
    assert names == ["alpha", "beta"]


def test_discover_packs_env_var_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = tmp_path / "my_pack"
    _write_manifest(pack, "my_pack")

    monkeypatch.setenv(ENV_VAR, str(pack))
    found = discover_packs(repo_root=tmp_path / "no_repo", include_entry_points=False)
    assert [p.resolve() for p in found] == [pack.resolve()]


def test_discover_packs_env_var_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "packs_dir"
    _write_manifest(parent / "one", "one")
    _write_manifest(parent / "two", "two")

    monkeypatch.setenv(ENV_VAR, str(parent))
    found = discover_packs(repo_root=tmp_path / "no_repo", include_entry_points=False)
    assert sorted(p.name for p in found) == ["one", "two"]


def test_discover_packs_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "ep_fake_pack"
    _write_manifest(pkg, "ep_fake_pack")
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    _patch_entry_points(monkeypatch, [_FakeEntryPoint("ep_fake_pack", "ep_fake_pack")])
    monkeypatch.delenv(ENV_VAR, raising=False)

    found = discover_packs(repo_root=tmp_path / "no_repo", include_env=False)
    assert [p.name for p in found] == ["ep_fake_pack"]


def test_discover_packs_deduplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = tmp_path / "dup_pack"
    _write_manifest(pack, "dup_pack")

    # Same pack surfaced by repo scan AND env var AND entry point
    repo_root = tmp_path / "repo"
    (repo_root / "userpacks").mkdir(parents=True)
    (repo_root / "userpacks" / "dup_pack").symlink_to(pack)

    monkeypatch.setenv(ENV_VAR, str(pack))
    pkg_dir = tmp_path / "dup_pack_module"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(f"from pathlib import Path\nPACK_ROOT = Path({str(pack)!r})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("dup_pack", "dup_pack_module:PACK_ROOT")])

    found = discover_packs(repo_root=repo_root)
    # Three sources reference the same pack — must dedupe to one
    resolved = {p.resolve() for p in found}
    assert resolved == {pack.resolve()}


def test_load_discovered_packs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pack = tmp_path / "loaded_pack"
    _write_manifest(pack, "loaded_pack")
    monkeypatch.setenv(ENV_VAR, str(pack))

    loaded = load_discovered_packs(repo_root=tmp_path / "no_repo", include_entry_points=False)
    assert len(loaded) == 1
    assert loaded[0].manifest.name == "loaded_pack"


def test_load_pack_accepts_entry_point_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "ep_load_pack"
    _write_manifest(pkg, "ep_load_pack")
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    loaded = load_pack("ep_load_pack")
    assert loaded.manifest.name == "ep_load_pack"


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    before = set(sys.modules)
    yield
    # Drop any test-fixture modules that were imported dynamically
    for mod_name in set(sys.modules) - before:
        if mod_name.startswith(("fakepack_", "ep_fake_pack", "dup_pack_module", "ep_load_pack")):
            sys.modules.pop(mod_name, None)
