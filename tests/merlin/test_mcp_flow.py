"""Shape tests for the merlin MCP handlers.

We don't go end-to-end through MerlinBridge (covered by
test_compile_smoke); these tests confirm the MCP wrappers return the
documented dict shapes and never raise out to the MCP runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xpu_rt.mcp.tools.merlin_flow import (
    MERLIN_FLOW_TOOLS,
    xpu_rt_merlin_describe_target,
    xpu_rt_merlin_list_targets,
    xpu_rt_merlin_onnx_to_mlir,
    xpu_rt_merlin_profile,
)


class _FakeSession:
    """Minimal SessionManager stand-in — none of our handlers touch it."""


def test_merlin_flow_tools_registry_shape() -> None:
    names = {t["name"] for t in MERLIN_FLOW_TOOLS}
    assert names == {
        "xpu_rt_merlin_list_targets",
        "xpu_rt_merlin_describe_target",
        "xpu_rt_merlin_onnx_to_mlir",
        "xpu_rt_merlin_compile",
        "xpu_rt_merlin_compile_dispatch_matrix",
        "xpu_rt_merlin_chipyard_build",
        "xpu_rt_merlin_profile",
    }
    for tool in MERLIN_FLOW_TOOLS:
        assert callable(tool["handler"])
        assert tool["phase"] in {"inspect", "transform", "job"}
        assert "input_schema" in tool


def test_list_targets_missing_root(tmp_path: Path) -> None:
    out = xpu_rt_merlin_list_targets(
        _FakeSession(), merlin_root=str(tmp_path / "nope"),
    )
    assert out["ok"] is True
    assert out["targets"] == []
    assert out["merlin_root_available"] is False


def test_describe_target_missing(tmp_path: Path) -> None:
    out = xpu_rt_merlin_describe_target(
        _FakeSession(), name="nope", merlin_root=str(tmp_path),
    )
    assert out["ok"] is False
    assert out["exception_type"] == "FileNotFoundError"


def test_describe_target_round_trip(tmp_path: Path) -> None:
    spec_dir = tmp_path / "target_specs" / "examples" / "fake"
    spec_dir.mkdir(parents=True)
    (spec_dir / "capability.yaml").write_text(
        "schema_version: 1\n"
        "target: {name: fake, display_name: Fake, vendor: tt, maturity: experimental}\n"
        "platform: {host_isa: riscv64, environments: [firesim]}\n"
        "execution_model: {kind: matrix_coprocessor}\n"
        "isa: {features: [+v]}\n"
        "runtime: {executable_format: llvm_cpu_vmfb}\n"
        "verification: {simulator: {available: true, kind: firesim}}\n"
    )
    out = xpu_rt_merlin_describe_target(
        _FakeSession(), name="fake", merlin_root=str(tmp_path),
    )
    assert out["ok"] is True
    assert out["name"] == "fake"
    assert out["isa_features"] == ["+v"]
    assert out["has_simulator"] is True


def test_onnx_to_mlir_stub_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No merlin importer and no torch-mlir → stub path lands a placeholder MLIR."""
    monkeypatch.setenv("MERLIN_ROOT", str(tmp_path / "no-merlin"))
    monkeypatch.setattr(
        "xpu_rt.targets.backends.merlin.onnx_bridge._torch_mlir_binary",
        lambda: None,
    )
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"\x00ONNX-fake")
    out_mlir = tmp_path / "out.mlir"
    out = xpu_rt_merlin_onnx_to_mlir(
        _FakeSession(),
        onnx_path=str(onnx_path),
        out_mlir=str(out_mlir),
        allow_stub=True,
    )
    assert out["ok"] is True
    assert out["importer"] == "stub"
    assert out_mlir.is_file()
    assert "xpu_rt.onnx.stub" in out_mlir.read_text()


def test_profile_with_noop_runner(tmp_path: Path) -> None:
    """Noop runner fills measurements from the supplied table."""
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({
        "dispatches": [
            {"target": "saturn_opu_v128", "dispatch": "d0",
             "vmfb": str(tmp_path / "d0.vmfb")},
            {"target": "saturn_opu_v128", "dispatch": "d1",
             "vmfb": str(tmp_path / "d1.vmfb")},
        ],
    }))
    out_path = tmp_path / "profiled.json"
    out = xpu_rt_merlin_profile(
        _FakeSession(),
        matrix_path=str(matrix),
        out_path=str(out_path),
        runner="noop",
        noop_table={"d0": 100.0, "d1": 250.0},
    )
    assert out["ok"] is True
    assert out["via"] == "local-runner"
    assert out["mean_us_by_dispatch"] == {"d0": 100.0, "d1": 250.0}
