"""Phase C end-to-end test: build a synthetic bundle, cross-compile it
through ``xpu_rt.runtime.cross_compile.riscv64_bare``, run the
resulting ELF on Chipyard's spike, parse the output, and assert
numerical correctness against the analytical reference.

Synthetic bundle structure:

  golden_inputs.pt                              # torch.save([tensor4])
  generated_kernels/xnnpack/region0_fc.c        # via XnnpackProvider
  generated_kernels/xnnpack/region0_fc.json     # kernel metadata
  (no execution_plan.yaml — Phase C v1 runs all kernels sequentially)

Skips when:
  * Chipyard spike binary not present.
  * Merlin IREE clang toolchain not present.
  * Pre-built ``build/riscv-spike/libxpu_rt_static.a`` not present.

The third skip means the test assumes the libxpu_rt cross-compile has
been run separately (it's a slow build; we don't fold it into pytest).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import torch

from xpu_rt.graph_compilation.region_dossier import (
    CrossCompileConfig,
    load_target_profile,
)
from xpu_rt.kernels.providers.xnnpack import XnnpackProvider
from xpu_rt.providers.kernel_provider import KernelCodegenRequest
from xpu_rt.runtime.cross_compile.riscv64_bare import (
    CrossCompileError,
    cross_compile_riscv64_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_YAML = REPO_ROOT / "xpu-rt" / "configs" / "targets" / "riscv64_spike_rvv.yaml"
SPIKE = Path("/scratch2/agustin/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/spike")
CLANG = Path(
    "/scratch2/agustin/merlin/build_tools/riscv-tools-iree/"
    "toolchain/clang/linux/RISCV/bin/clang"
)
LIBXPU_RT = REPO_ROOT / "build" / "riscv-spike" / "libxpu_rt_static.a"


def _skip_if_env_missing() -> None:
    if not SPIKE.is_file():
        pytest.skip(f"spike binary missing: {SPIKE}")
    if not CLANG.is_file():
        pytest.skip(f"merlin clang missing: {CLANG}")
    if not LIBXPU_RT.is_file():
        pytest.skip(
            f"pre-built libxpu_rt_static.a not at {LIBXPU_RT} — "
            "run the libxpu_rt riscv-spike cross-compile first"
        )


class _ContractStub:
    """Minimal duck-typed contract the XnnpackProvider's can_bid /
    propose paths recognise. Mirrors the fields xnnpack.py reads:
    op_kind, dtype, layout."""

    def __init__(self, op_kind: str, dtype: str = "f32", layout: str = "NC"):
        self.op_kind = op_kind
        self.dtype = dtype
        self.layout = layout


class _TargetStub:
    def __init__(self, family: str = "host_cpu"):
        self.family = family


def _build_fc_kernel(
    bundle_dir: Path,
    *,
    in_c: int = 4,
    out_c: int = 3,
    weights: list[float],
    bias: list[float],
) -> None:
    """Run XnnpackProvider.propose() with a contract that embeds real
    FC shape + weights, place the artifacts under generated_kernels/xnnpack/.
    """
    xnn_dir = bundle_dir / "generated_kernels" / "xnnpack"
    xnn_dir.mkdir(parents=True, exist_ok=True)

    provider = XnnpackProvider()

    # extras carries the real shape + weight bytes the cross-compile
    # bundle needs. shape_dims = [in_c, out_c] per the FC contract.
    static_weights_f32 = list(weights) + list(bias)
    req = KernelCodegenRequest(
        task_id="t-fc-region0",
        contract=_ContractStub(op_kind="matmul", dtype="f32", layout="NC"),
        target=_TargetStub(family="host_cpu"),
        artifact_dir=str(xnn_dir),
        extras={
            "shape_dims": [in_c, out_c],
            "int_params": [0],
            "float_params": [-1e30, 1e30],
            "static_weights_f32": static_weights_f32,
        },
    )
    result = provider.propose(req)
    assert result.status == "generated", f"propose failed: {result}"
    assert (xnn_dir / "kernel_metadata.json").is_file()


def _parse_spike_output(stdout: str) -> tuple[list[float], str | None]:
    """Parse the driver's `output:` and `checksum:` lines.

    The driver emits hex-bits like ``output: 4040a3d8 3fa8f5c3 bfbc28f6``
    (no ``0x`` prefix, lowercase). Each token is a 32-bit fp32 bit
    pattern; decode and pack back into a Python float.
    """
    output_match = re.search(r"output:\s+([0-9a-fA-Fx ]+)", stdout)
    cs_match = re.search(r"checksum:\s+(0x[0-9a-fA-F]+)", stdout)
    if not output_match:
        return [], None
    floats = []
    for tok in output_match.group(1).split():
        if not tok:
            continue
        bits = int(tok, 16)
        if bits > 0xffffffff:
            continue
        f = struct.unpack("<f", struct.pack("<I", bits))[0]
        floats.append(f)
    cs = cs_match.group(1) if cs_match else None
    return floats, cs


@pytest.mark.slow
def test_phase_c_fc_f32_end_to_end(tmp_path: Path) -> None:
    _skip_if_env_missing()

    # Weights matching the smoke test in test_xnnpack_bridge_riscv.c
    # so we get the same numerical reference: [3.01, 1.32, -1.47].
    in_c = 4
    out_c = 3
    weights = [
         0.1,  0.2,  0.3,  0.4,
        -0.1,  0.5,  0.0,  0.1,
         1.0, -1.0,  0.5, -0.5,
    ]
    bias = [0.01, 0.02, 0.03]

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Stage golden_inputs.pt.
    inputs = [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)]
    torch.save(inputs, bundle_dir / "golden_inputs.pt")

    # Build the FC kernel artifacts.
    _build_fc_kernel(
        bundle_dir,
        in_c=in_c,
        out_c=out_c,
        weights=weights,
        bias=bias,
    )

    # Read the target profile.
    profile = load_target_profile(TARGET_YAML)
    assert profile.cross_compile is not None, (
        "riscv64_spike_rvv.yaml should carry a cross_compile block"
    )

    # Run the cross-compile orchestrator.
    result = cross_compile_riscv64_bundle(
        bundle_dir,
        target_id="riscv64_spike_rvv",
        cross=profile.cross_compile,
        repo_root=REPO_ROOT,
        model_id="phase_c_fc_smoke",
    )
    assert result.status == "ok", (
        f"cross-compile failed: status={result.status} "
        f"reason={result.reason} log={result.cmake_log_path}"
    )
    assert result.elf_path is not None
    assert result.elf_path.is_file()

    # Run the ELF under spike.
    proc = subprocess.run(
        [str(SPIKE), "--isa=rv64gcv", str(result.elf_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ},
    )
    stdout = proc.stdout + proc.stderr  # spike often prints HTIF to stderr too
    assert "PASS" in stdout, (
        f"spike run did not print PASS\n"
        f"---stdout---\n{proc.stdout}\n"
        f"---stderr---\n{proc.stderr}"
    )

    floats, _cs = _parse_spike_output(stdout)
    assert len(floats) == out_c, f"expected {out_c} outputs, got {floats}"

    # Analytical reference:
    #   y[0] = 0.1+0.4+0.9+1.6+0.01 = 3.01
    #   y[1] = -0.1+1.0+0.0+0.4+0.02 = 1.32
    #   y[2] = 1.0-2.0+1.5-2.0+0.03 = -1.47
    expected = [3.01, 1.32, -1.47]
    for got, want in zip(floats, expected, strict=True):
        assert abs(got - want) < 1e-5, f"got {got}, want {want}"
