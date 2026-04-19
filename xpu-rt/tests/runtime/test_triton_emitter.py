"""Tests for the Wave 11 Triton emitter skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.runtime.triton_emitter import (
    TritonEmitterReport,
    emit_triton_kernels,
    triton_available,
)


def _ft(shape):
    return TensorType(Float32Type(), list(shape))


def _matmul_module_with_triton_tag() -> tuple[ModuleOp, MatmulOp]:
    a = EmptyOp([], _ft([4, 8]))
    b = EmptyOp([], _ft([8, 16]))
    out = EmptyOp([], _ft([4, 16]))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_ft([4, 16])],
    )
    mm.attributes["compgen.library_dispatch"] = StringAttr("triton")
    block = Block()
    for op in (a, b, out, mm):
        block.add_op(op)
    block.add_op(ReturnOp(mm.res[0]))
    func = FuncOp(
        "forward",
        FunctionType.from_lists([], [_ft([4, 16])]),
        Region([block]),
    )
    return ModuleOp([func]), mm


def test_emit_matmul_writes_source_file(tmp_path: Path):
    m, _ = _matmul_module_with_triton_tag()
    report = emit_triton_kernels(m, out_dir=tmp_path)
    assert report.kernels_emitted == 1
    assert (tmp_path / "kernels").is_dir()
    sources = list((tmp_path / "kernels").glob("*.py"))
    assert len(sources) == 1
    text = sources[0].read_text()
    assert "@triton.jit" in text
    assert "tl.dot" in text


def test_manifest_contains_every_emitted_kernel(tmp_path: Path):
    m, _ = _matmul_module_with_triton_tag()
    report = emit_triton_kernels(m, out_dir=tmp_path)
    manifest_path = tmp_path / "emission_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == report.kernels_emitted


def test_ops_without_triton_tag_are_skipped(tmp_path: Path):
    a = EmptyOp([], _ft([4, 8]))
    b = EmptyOp([], _ft([8, 16]))
    out = EmptyOp([], _ft([4, 16]))
    mm = MatmulOp(
        inputs=[a.results[0], b.results[0]],
        outputs=[out.results[0]],
        res=[_ft([4, 16])],
    )
    # no library_dispatch tag
    block = Block()
    for op in (a, b, out, mm):
        block.add_op(op)
    block.add_op(ReturnOp(mm.res[0]))
    func = FuncOp(
        "forward",
        FunctionType.from_lists([], [_ft([4, 16])]),
        Region([block]),
    )
    m = ModuleOp([func])
    report = emit_triton_kernels(m, out_dir=tmp_path)
    assert report.kernels_emitted == 0


def test_triton_available_returns_bool():
    assert isinstance(triton_available(), bool)


def test_idempotent_emit_overwrites_sources(tmp_path: Path):
    m, _ = _matmul_module_with_triton_tag()
    first = emit_triton_kernels(m, out_dir=tmp_path)
    second = emit_triton_kernels(m, out_dir=tmp_path)
    # Each pass emits the same kernel (re-writes the file).
    assert first.kernels_emitted == second.kernels_emitted


def test_emitter_report_initial_values():
    r = TritonEmitterReport()
    assert r.kernels_emitted == 0
    assert r.manifest == {}
