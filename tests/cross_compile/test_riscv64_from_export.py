"""Phase G — production-path parity test.

Demonstrates that the weight-extraction helper can drive the full
Phase C + E flow from a real ``torch.export.ExportedProgram`` —
without hand-packing weights in the test body. This is the closest
the test suite gets today to the production pipeline's path:

  torch.nn.Linear
      ↓ torch.export.export
  exported_program.pt2 (saved to bundle/00_graph_capture/)
      ↓ load_state_dict_from_bundle
  state_dict
      ↓ populate_provider_extras  (with a RegionWeightSpec)
  extras dict
      ↓ XnnpackProvider.propose
  generated_kernels/xnnpack/<region>.c with real shape + kWeights
      ↓ cross_compile_riscv64_bundle
  program.elf
      ↓ spike --isa=rv64gcv
  output ≈ eager(input)

Same skip gates as the Phase C/E tests (chipyard / merlin / pre-built
libxpu_rt_static.a).
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
from xpu_rt.graph_compilation.weight_extraction import (
    RegionWeightSpec,
    load_state_dict_from_bundle,
    populate_provider_extras,
)
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


def _parse_spike_output(stdout: str) -> list[float]:
    output_match = re.search(r"output:\s+([0-9a-fA-Fx ]+)", stdout)
    if not output_match:
        return []
    floats: list[float] = []
    for tok in output_match.group(1).split():
        if not tok:
            continue
        bits = int(tok, 16)
        if bits > 0xffffffff:
            continue
        floats.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return floats


@pytest.mark.slow
def test_from_export_to_spike_parity(tmp_path: Path) -> None:
    """nn.Linear → torch.export → bundle → spike, all via the helper.

    Asserts the same ``max_abs<1e-4, max_rel<1e-5`` parity bar as
    Phase E, but drives weight extraction through
    :func:`load_state_dict_from_bundle` + :func:`populate_provider_extras`
    instead of hand-packing in the test body.
    """

    _skip_if_env_missing()

    torch.manual_seed(20260515)
    in_c, out_c = 6, 4
    linear = torch.nn.Linear(in_c, out_c, bias=True, dtype=torch.float32)
    x = torch.randn(in_c, dtype=torch.float32)

    # Eager reference.
    with torch.no_grad():
        eager = linear(x.unsqueeze(0)).squeeze(0).to(torch.float32)

    # Bundle layout: capture the ExportedProgram canonically.
    bundle_dir = tmp_path / "bundle"
    capture_dir = bundle_dir / "00_graph_capture"
    capture_dir.mkdir(parents=True)

    sample = (x.unsqueeze(0),)
    with torch.no_grad():
        ep = torch.export.export(linear, sample)
    torch.export.save(ep, str(capture_dir / "exported_program.pt2"))

    # Stage goldens at the bundle root (same convention as Phase E).
    torch.save([x], bundle_dir / "golden_inputs.pt")
    torch.save(eager, bundle_dir / "golden_outputs.pt")

    # ---- the helper path ----------------------------------------------
    state_dict = load_state_dict_from_bundle(bundle_dir)
    assert state_dict, "weight extraction couldn't load exported_program state_dict"

    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=in_c,
        out_c=out_c,
    )
    extras: dict[str, object] = {}
    populate_provider_extras(extras, spec, state_dict)
    assert "static_weights_f32" in extras, (
        "populate_provider_extras did not find weights — "
        f"state_dict keys: {list(state_dict.keys())[:8]}"
    )
    assert extras["shape_dims"] == [in_c, out_c]
    # ---------------------------------------------------------------

    # Drive the XnnpackProvider with the helper-populated extras.
    xnn_dir = bundle_dir / "generated_kernels" / "xnnpack"
    xnn_dir.mkdir(parents=True, exist_ok=True)
    provider = XnnpackProvider()
    req = KernelCodegenRequest(
        task_id="t-export-region0",
        contract=_ContractStub(op_kind="matmul"),
        target=_TargetStub(),
        artifact_dir=str(xnn_dir),
        extras=extras,
    )
    result = provider.propose(req)
    assert result.status == "generated"

    # Cross-compile + spike, same as Phase E.
    profile = load_target_profile(TARGET_YAML)
    assert profile.cross_compile is not None
    cc = cross_compile_riscv64_bundle(
        bundle_dir,
        target_id="riscv64_spike_rvv",
        cross=profile.cross_compile,
        repo_root=REPO_ROOT,
        model_id="phase_g_from_export",
    )
    assert cc.status == "ok", (
        f"cross-compile failed: status={cc.status} reason={cc.reason}"
    )
    assert cc.elf_path is not None and cc.elf_path.is_file()

    proc = subprocess.run(
        [str(SPIKE), "--isa=rv64gcv", str(cc.elf_path)],
        capture_output=True, text=True, timeout=300, env={**os.environ},
    )
    stdout = proc.stdout + proc.stderr
    assert "PASS" in stdout, (
        f"spike run did not print PASS\n--stdout--\n{proc.stdout}"
    )

    spike_floats = _parse_spike_output(stdout)
    assert len(spike_floats) == out_c, (
        f"expected {out_c} outputs, got {spike_floats}"
    )
    spike = torch.tensor(spike_floats, dtype=torch.float32)

    max_abs = float((spike - eager).abs().max().item())
    max_rel = float(
        ((spike - eager).abs() / (eager.abs() + 1e-7)).max().item()
    )
    assert max_abs < 1.0e-4, (
        f"max_abs parity exceeded: {max_abs} > 1e-4 "
        f"(eager={eager.tolist()}, spike={spike_floats})"
    )
    assert max_rel < 1.0e-5, (
        f"max_rel parity exceeded: {max_rel} > 1e-5"
    )
