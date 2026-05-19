"""Tests for merlin target_spec parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from xpu_rt.targets.backends.merlin.target_spec import (
    MerlinTargetSpec,
    list_targets,
    load_target_spec,
)


def _write_spec(merlin_root: Path, name: str, body: str) -> None:
    spec_dir = merlin_root / "target_specs" / "examples" / name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "capability.yaml").write_text(body)


def test_list_targets_empty(tmp_path: Path) -> None:
    assert list_targets(tmp_path) == []


def test_list_targets_skips_dirs_without_capability_yaml(tmp_path: Path) -> None:
    base = tmp_path / "target_specs" / "examples"
    base.mkdir(parents=True)
    (base / "bare").mkdir()
    (base / "with_yaml").mkdir()
    (base / "with_yaml" / "capability.yaml").write_text(
        "schema_version: 1\ntarget: {name: with_yaml}\n"
    )
    assert list_targets(tmp_path) == ["with_yaml"]


def test_load_target_spec_full(tmp_path: Path) -> None:
    _write_spec(
        tmp_path,
        "saturn_opu_v128",
        """
schema_version: 1
target:
  name: saturn_opu_v128
  display_name: Saturn OPU V128
  vendor: UCB-BAR
  maturity: experimental
platform:
  host_isa: riscv64
  environments: [firesim]
execution_model:
  kind: matrix_coprocessor
isa:
  base: riscv64
  features: [+m, +v, +xopu]
runtime:
  executable_format: llvm_cpu_vmfb
verification:
  simulator:
    available: true
    kind: firesim
""",
    )
    spec = load_target_spec("saturn_opu_v128", tmp_path)
    assert isinstance(spec, MerlinTargetSpec)
    assert spec.name == "saturn_opu_v128"
    assert spec.host_isa == "riscv64"
    assert "+xopu" in spec.isa_features
    assert spec.execution_kind == "matrix_coprocessor"
    assert spec.has_simulator is True
    assert spec.simulator_kind == "firesim"


def test_load_target_spec_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_target_spec("nope", tmp_path)
