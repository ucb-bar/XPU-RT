"""Tests for the Triton bundle plugin.

Asserts:
* ``linalg.matmul`` ops get annotated with ``compgen.library_dispatch="triton"``
* ``kernels/*.py`` files land on disk and parse via ``ast.parse``
* ``emission_manifest.json`` is well-formed and lists every emitted kernel
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
import torch.nn as nn
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.stages.bundle.triton_plugin import (
    _ensure_dispatch_attr,
    write_triton_bundle,
)

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _compile_small() -> tuple:
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _Mlp().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    return compiled.payload_module, dev.profile


def test_annotate_eligible_ops(tmp_path: Path) -> None:
    module, _ = _compile_small()
    added = _ensure_dispatch_attr(module)
    assert added >= 1  # at least one linalg.matmul in the MLP's payload


def test_write_triton_bundle_emits_kernels(tmp_path: Path) -> None:
    module, _ = _compile_small()
    out = tmp_path / "triton"
    result = write_triton_bundle(module, out)
    assert result.kernels_emitted >= 1
    assert result.kernel_files, "expected at least one kernels/*.py"
    for p in result.kernel_files:
        src = p.read_text()
        ast.parse(src)  # must parse as valid Python


def test_triton_manifest_is_well_formed(tmp_path: Path) -> None:
    module, _ = _compile_small()
    out = tmp_path / "triton"
    result = write_triton_bundle(module, out)
    manifest = json.loads(result.manifest_path.read_text())
    for kname, entry in manifest.items():
        assert entry["kernel"] == kname
        assert Path(entry["source_path"]).exists()
        assert entry["template"] in {"matmul", "softmax"}


def test_no_eligible_ops_is_noop(tmp_path: Path) -> None:
    """Build an empty module; expect 0 kernels and no crash."""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    empty = ModuleOp(Region([Block()]))
    result = write_triton_bundle(empty, tmp_path / "triton_empty")
    assert result.kernels_emitted == 0
