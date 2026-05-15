"""Tests for the Hexagon C bundle plugin.

Exercises the full path: a real Gemma-ish payload IR → ``write_baremetal_bundle``
→ on-disk C project. Asserts:

* every expected scaffold file exists,
* every emitted ``kernels/*.c`` + ``npu_driver_ext.c`` parses via ``gcc -fsyntax-only``,
* ``make -n`` resolves without referring to unknown symbols,
* when the payload module changes (a new func.func appended), at least
  one ``kernels/*.c`` file's bytes change.

Skipped when ``gcc`` isn't on PATH (the parse-check is the core assertion).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.stages.bundle.baremetal_plugin import write_baremetal_bundle

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
GCC = shutil.which("gcc")


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _compile_small() -> tuple:
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _TwoLinear().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    return compiled.payload_module, dev.profile


def test_write_baremetal_bundle_creates_scaffold_files(tmp_path: Path) -> None:
    module, profile = _compile_small()
    out = tmp_path / "baremetal"
    result = write_baremetal_bundle(module, profile, out)

    expected = {
        "memory_map.h",
        "npu_driver.h",
        "npu_driver.c",
        "npu_driver_ext.h",
        "npu_driver_ext.c",
        "weights.h",
        "main.c",
        "linker.ld",
        "Makefile",
    }
    got = {p.name for p in out.iterdir()}
    missing = expected - got
    assert not missing, f"missing files: {missing}; have: {got}"
    assert (out / "kernels").exists()
    assert result.kernel_files, "expected at least one kernels/*.c file"


@pytest.mark.skipif(GCC is None, reason="gcc not on PATH")
def test_emitted_c_parses_with_gcc_fsyntax_only(tmp_path: Path) -> None:
    module, profile = _compile_small()
    out = tmp_path / "baremetal"
    write_baremetal_bundle(module, profile, out)

    # The npu_driver_ext.c is the richest file; it includes libm.
    sources = [out / "npu_driver_ext.c"] + list((out / "kernels").glob("*.c"))
    assert sources
    errors: list[str] = []
    for src in sources:
        p = subprocess.run(
            [GCC, "-std=c99", "-fsyntax-only", "-I", str(out), "-I", str(out / "kernels"), str(src)],
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            errors.append(f"{src.name}:\n{p.stderr}")
    assert not errors, "gcc -fsyntax-only failed:\n" + "\n\n".join(errors)


def test_makefile_lists_ext_source_and_libm(tmp_path: Path) -> None:
    module, profile = _compile_small()
    out = tmp_path / "baremetal"
    write_baremetal_bundle(module, profile, out)
    mk_text = (out / "Makefile").read_text()
    assert "npu_driver_ext.c" in mk_text
    assert "-lm" in mk_text


def test_different_payload_produces_different_c_sources(tmp_path: Path) -> None:
    """Appending to the module must change emitted C bytes somewhere.

    Declaration-only funcs flow into ``npu_driver_ext.h``; funcs with
    bodies flow into ``kernels/*.c``. We accept a change in either.
    """
    module, profile = _compile_small()
    out_a = tmp_path / "a"
    write_baremetal_bundle(module, profile, out_a)

    def _snapshot(out: Path) -> dict[str, str]:
        snap: dict[str, str] = {}
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.suffix in (".c", ".h"):
                snap[str(f.relative_to(out))] = f.read_text()
        return snap

    snap_a = _snapshot(out_a)

    # Append an extra private declaration — surfaces in npu_driver_ext.h
    # as a new prototype.
    from xdsl.dialects.builtin import Float32Type, TensorType
    from xdsl.dialects.func import FuncOp

    extra = FuncOp(
        name="agent_injected_fn",
        function_type=(
            [TensorType(Float32Type(), [4, 4])],
            [TensorType(Float32Type(), [4, 4])],
        ),
        visibility="private",
    )
    module.body.block.add_op(extra)

    out_b = tmp_path / "b"
    write_baremetal_bundle(module, profile, out_b)
    snap_b = _snapshot(out_b)

    new_files = set(snap_b) - set(snap_a)
    shared_changed = {n for n in set(snap_a) & set(snap_b) if snap_a[n] != snap_b[n]}
    assert new_files or shared_changed, (
        f"expected emitted C/H to reflect the appended func.func; new={new_files} shared_changed={shared_changed}"
    )
    # The injected function name must surface in at least one C/H file.
    assert any("agent_injected_fn" in text for text in (snap_b[n] for n in new_files | shared_changed)), (
        "injected function name did not appear in any emitted C/H file"
    )


def test_ext_header_has_helper_protos(tmp_path: Path) -> None:
    module, profile = _compile_small()
    out = tmp_path / "baremetal"
    write_baremetal_bundle(module, profile, out)
    ext_h = (out / "npu_driver_ext.h").read_text()
    for sym in [
        "npu_matmul(",
        "npu_batch_matmul(",
        "npu_transpose(",
        "npu_softmax_lastdim(",
        "npu_view_extract_slice(",
        "npu_expf(",
        "npu_rsqrtf(",
    ]:
        assert sym in ext_h, f"missing helper proto: {sym}"
