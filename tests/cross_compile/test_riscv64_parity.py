"""Phase E — riscv64 spike parity test against a real torch model.

Extends Phase C's synthetic test by:

  * Building a real ``torch.nn.Linear`` module (one fully-connected
    layer, fp32, batch=1) — the simplest network the XNNPACK FC bridge
    case covers end-to-end.
  * Computing the eager output (the "golden" reference).
  * Extracting the linear's ``weight`` + ``bias`` tensors and packing
    them into the XnnpackProvider's ``request.extras`` exactly as the
    pipeline's full integration would.
  * Driving the Phase C cross-compile orchestrator → emitting
    ``program.elf``.
  * Running on Chipyard's spike, parsing the hex-bit output, and
    asserting numerical parity vs the eager output using
    ``torch.allclose(atol=1e-4, rtol=1e-5)``.

This is the v1 acceptance bar from `hazy-singing-grove.md`. A larger
model (graph_break_mlp's full Linear→ReLU→Linear→Add path) requires
the bridge to also carry UNARY(relu) and BINARY(add) regions in the
generated_kernels, plus the multi-region wiring the driver template
already supports. That extension is straightforward but additive.

Skips when the chipyard/merlin/libxpu_rt environment isn't present
(same gates as the Phase C smoke).
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
from pathlib import Path

import pytest
import torch

from xpu_rt.graph_compilation.region_dossier import load_target_profile
from xpu_rt.kernels.providers.xnnpack import XnnpackProvider
from xpu_rt.providers.kernel_provider import KernelCodegenRequest
from xpu_rt.runtime.cross_compile.riscv64_bare import (
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
    def __init__(self, op_kind: str, dtype: str = "f32", layout: str = "NC"):
        self.op_kind = op_kind
        self.dtype = dtype
        self.layout = layout


class _TargetStub:
    def __init__(self, family: str = "host_cpu"):
        self.family = family


def _pack_linear_weights(linear: torch.nn.Linear) -> list[float]:
    """Pack a ``nn.Linear``'s weight + bias into the bridge's expected
    fp32 row-major layout: ``weight[out, in]`` then ``bias[out]``.

    The XNNPACK FC bridge case expects exactly this packing — see
    ``runtime/native/libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.c``
    in the ``XPU_RT_XNN_OP_FULLY_CONNECTED_F32`` arm.
    """
    weight = linear.weight.detach().contiguous().to(torch.float32)
    flat: list[float] = []
    out_c, in_c = weight.shape
    for o in range(out_c):
        for i in range(in_c):
            flat.append(float(weight[o, i].item()))
    if linear.bias is not None:
        for o in range(out_c):
            flat.append(float(linear.bias[o].item()))
    return flat


def _parse_spike_output(stdout: str) -> tuple[list[float], str | None]:
    """Parse driver's `output:` and `checksum:` lines (hex32 tokens)."""
    output_match = re.search(r"output:\s+([0-9a-fA-Fx ]+)", stdout)
    cs_match = re.search(r"checksum:\s+(0x[0-9a-fA-F]+)", stdout)
    if not output_match:
        return [], None
    floats: list[float] = []
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
def test_phase_e_linear_eager_parity(tmp_path: Path) -> None:
    """Single nn.Linear: spike output matches eager within tolerance_eps."""

    _skip_if_env_missing()

    # Build a real torch model: y = Linear(8, 5)(x). Deterministic
    # initialisation so the test is bit-stable across reruns.
    torch.manual_seed(20260515)
    in_c, out_c = 8, 5
    linear = torch.nn.Linear(in_c, out_c, bias=True, dtype=torch.float32)

    # Sample input — also deterministic.
    x = torch.randn(in_c, dtype=torch.float32)

    # Eager reference.
    with torch.no_grad():
        eager = linear(x.unsqueeze(0)).squeeze(0).to(torch.float32)
    eager_list = eager.detach().contiguous().tolist()

    # Bundle the model state into a synthetic xpu_rt bundle.
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    torch.save([x], bundle_dir / "golden_inputs.pt")
    torch.save(eager, bundle_dir / "golden_outputs.pt")

    # Emit the XNNPACK FC kernel via the provider with real weights.
    xnn_dir = bundle_dir / "generated_kernels" / "xnnpack"
    xnn_dir.mkdir(parents=True, exist_ok=True)

    provider = XnnpackProvider()
    req = KernelCodegenRequest(
        task_id="t-fc-parity-region0",
        contract=_ContractStub(op_kind="matmul", dtype="f32", layout="NC"),
        target=_TargetStub(family="host_cpu"),
        artifact_dir=str(xnn_dir),
        extras={
            "shape_dims": [in_c, out_c],
            "int_params": [0],
            "float_params": [-1.0e30, 1.0e30],
            "static_weights_f32": _pack_linear_weights(linear),
        },
    )
    result = provider.propose(req)
    assert result.status == "generated"

    # Cross-compile.
    profile = load_target_profile(TARGET_YAML)
    assert profile.cross_compile is not None
    cc = cross_compile_riscv64_bundle(
        bundle_dir,
        target_id="riscv64_spike_rvv",
        cross=profile.cross_compile,
        repo_root=REPO_ROOT,
        model_id="phase_e_linear_parity",
    )
    assert cc.status == "ok", (
        f"cross-compile failed: {cc.status}/{cc.reason} log={cc.cmake_log_path}"
    )
    assert cc.elf_path is not None and cc.elf_path.is_file()

    # Run on spike.
    proc = subprocess.run(
        [str(SPIKE), "--isa=rv64gcv", str(cc.elf_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ},
    )
    stdout = proc.stdout + proc.stderr
    assert "PASS" in stdout, (
        f"spike run did not print PASS\n--stdout--\n{proc.stdout}"
    )

    spike_floats, _cs = _parse_spike_output(stdout)
    assert len(spike_floats) == out_c, (
        f"expected {out_c} outputs, got {spike_floats}"
    )
    spike = torch.tensor(spike_floats, dtype=torch.float32)

    # Parity check vs eager torch.nn.Linear. tolerance from the plan:
    # max_abs < 1e-4, max_rel < 1e-5 (the spike build links the same
    # XNNPACK we use on host; numerical drift should be ~zero ULPs for
    # this size). torch.allclose's default rtol=1e-05, atol=1e-08 is
    # tighter — we use the looser plan-spec values here.
    max_abs = float((spike - eager).abs().max().item())
    max_rel = float(
        ((spike - eager).abs() / (eager.abs() + 1e-7)).max().item()
    )
    assert max_abs < 1.0e-4, (
        f"max_abs parity exceeded: {max_abs} > 1e-4 "
        f"(eager={eager_list}, spike={spike_floats})"
    )
    assert max_rel < 1.0e-5, (
        f"max_rel parity exceeded: {max_rel} > 1e-5 "
        f"(eager={eager_list}, spike={spike_floats})"
    )
