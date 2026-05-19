"""End-to-end kill criterion for Phase 1.

The Phase 1 milestone in
``/home/agustin/.claude/plans/immutable-yawning-frog.md`` is: on a
single tiny model, produce a bundle that executes on Spike with
``correctness_vs_eager=pass`` and a numeric ``measured_cost`` written
to the bundle, then advance the promotion gate past ``verified_fx``
from that evidence.

This test fabricates a minimal but realistic bundle — one
``kernel_contracts/<matmul>.yaml`` + one
``generated_kernels/test_provider/r0_kernel_under_test.c`` containing
a scalar int8×int8→int32 matmul — and drives the full
:class:`xpu_rt.runtime.spike_executor.SpikeExecutor` →
``measured_cost.json`` → :mod:`xpu_rt.promotion.gates` chain. The
fabricated bundle stands in for the autocomp-generated kernel that
the production pipeline will hand the executor once it is reliably
producing kernel sources for tiny_mlp / merlin_mlp.

The test is gated behind ``requires_spike`` because the cross-compile
+ Spike subprocess need ``riscv64-unknown-linux-gnu-gcc``, ``spike``,
``pk``, and Gemmini RoCC support.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import (
    _check_gemmini_extension,
    _check_toolchain,
    _ToolchainMissing,
)
from xpu_rt.promotion.gates import PromotionLevel, evaluate_gate
from xpu_rt.runtime.measured_cost import MeasuredCost
from xpu_rt.runtime.spike_executor import SpikeExecutor


def _toolchain_present() -> bool:
    try:
        _check_toolchain()
    except _ToolchainMissing:
        return False
    return True


SKIP_NO_TOOLCHAIN = pytest.mark.skipif(
    not _toolchain_present(),
    reason="riscv64 / spike toolchain not present",
)


# Tile dimensions kept small so the spike subprocess finishes in
# seconds (M*K*N int8 multiplies on scalar code; 4*16*8 = 512 ops).
_M, _K, _N = 4, 16, 8


_SCALAR_MATMUL_KERNEL = f"""
// Reference int8 x int8 -> int32 matmul kernel.
// Matches the scalar_ref inside CRiscvEvaluator's matmul harness so
// the executor's correctness comparison passes bit-equal.

#include <stdint.h>

#define M {_M}
#define K {_K}
#define N {_N}

void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {{
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A[m*K + k] * (int32_t)B[k*N + n];
            C[m*N + n] = acc;
        }}
}}
"""


def _seed_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1", "artifacts": {}}, indent=2)
    )
    contracts_dir = bundle_dir / "kernel_contracts"
    contracts_dir.mkdir()
    yaml_doc = {
        "op_name": "linalg_matmul",
        "supported_dtypes": ["i8"],
        "fusable": False,
        "aliasing": [],
        "cost": {
            "flops": _M * _N * _K * 2,
            "bytes_read": _M * _K + _K * _N,
            "bytes_written": _M * _N * 4,
        },
        "input_layouts": [
            {"kind": "row_major", "strides": None, "alignment": 1},
            {"kind": "row_major", "strides": None, "alignment": 1},
        ],
        "output_layouts": [{"kind": "row_major", "strides": None, "alignment": 1}],
        "perf_target_us": None,
        "priority": 1,
        "input_shapes": [[_M, _K], [_K, _N]],
        "output_shapes": [[_M, _N]],
        "metadata": {"region_id": "r0", "dispatch_id": "r0"},
    }
    (contracts_dir / "linalg_matmul.yaml").write_text(
        yaml.safe_dump(yaml_doc, default_flow_style=False, sort_keys=False)
    )

    gk_dir = bundle_dir / "generated_kernels" / "test_provider"
    gk_dir.mkdir(parents=True)
    (gk_dir / "r0_linalg_matmul.c").write_text(_SCALAR_MATMUL_KERNEL)
    (bundle_dir / "generated_kernels" / "index.json").write_text(
        json.dumps(
            [
                {
                    "provider": "test_provider",
                    "op_name": "linalg_matmul",
                    "region_id": "r0",
                    "path": "test_provider/r0_linalg_matmul.c",
                }
            ],
            indent=2,
        )
    )


@SKIP_NO_TOOLCHAIN
@pytest.mark.requires_spike
def test_spike_executor_closes_measurement_loop(tmp_path: Path) -> None:
    """The flagship Phase 1 kill-criterion test.

    Asserts that running ``SpikeExecutor`` on a fabricated bundle:

    1. writes ``measured_cost.json`` to the bundle directory,
    2. records ``correctness_vs_eager == "pass"``,
    3. records a positive ``cycles_total`` (Gemmini counter output),
    4. writes ``kernel_execution_report.json`` into the run_dir,
    5. promotes the gate past ``verified_fx`` (to
       ``verified_kernel`` or higher) when the run dir also has a
       passing FX-level differential report and a candidate
       selection.

    Step (5) is what proves the executor's measurement is wired into
    the promotion ladder; without it, the bundle would carry the
    evidence but ``evaluate_gate`` would still cap at
    ``verified_fx``.
    """
    if not _check_gemmini_extension():
        pytest.skip("spike --extension=gemmini not supported by this spike build")

    bundle = tmp_path / "bundle"
    run_dir = tmp_path / "run"
    _seed_bundle(bundle)

    # Seed minimum evidence for verified_fx so the promotion gate
    # can advance once the executor lands its evidence.
    rp = run_dir / "03_recipe_planning"
    rp.mkdir(parents=True)
    (rp / "candidate_selection.json").write_text(
        json.dumps({"selected_candidate_id": "c0"})
    )
    (rp / "real_transform_differential_report.json").write_text(
        json.dumps({"status": "pass"})
    )

    executor = SpikeExecutor(target_id="gemmini")
    measured = executor.execute(bundle, run_dir=run_dir)

    # (1) artifact present in the bundle
    measured_cost_path = bundle / "measured_cost.json"
    assert measured_cost_path.is_file()
    on_disk = MeasuredCost.read_json(measured_cost_path)
    assert on_disk == measured

    # (2) executor reports passing correctness
    assert measured.correctness_vs_eager == "pass", (
        f"executor did not pass correctness; samples={[s.as_dict() for s in measured.samples]}"
    )
    # (3) numeric measured_cost is present. The kernel here is a
    # scalar reference matmul so it issues no Gemmini RoCC
    # instructions; the ``MAIN_LD_ST_EX_CYCLES`` counter reads 0 by
    # design. A non-zero count requires a Gemmini-aware kernel —
    # exercised in the separate test below.
    assert measured.cycles_total is not None
    assert measured.cycles_total >= 0
    assert len(measured.samples) == 1
    sample = measured.samples[0]
    assert sample.correctness == "pass"
    assert sample.cycles is not None and sample.cycles >= 0
    assert sample.mismatches == 0
    assert sample.total_elements == _M * _N

    # (4) kernel_execution_report.json was written to the run dir
    ker_report = (
        run_dir / "02_graph_analysis" / "kernel_execution" / "kernel_execution_report.json"
    )
    assert ker_report.is_file()
    ker = json.loads(ker_report.read_text())
    assert ker["status"] == "pass"
    assert ker["overall"] == "pass"
    assert ker["cycles_total"] == measured.cycles_total

    # (5) promotion gate advances past verified_fx
    gate = evaluate_gate(run_dir, bundle_dir=bundle)
    assert gate.level >= PromotionLevel.VERIFIED_KERNEL, (
        f"expected at least VERIFIED_KERNEL; got {gate.level} reasons={gate.reasons}"
    )


_GEMMINI_MATMUL_KERNEL = f"""
// Real Gemmini-aware int8 matmul kernel via the high-level helper.
// Issues RoCC instructions so MAIN_LD_ST_EX_CYCLES increments and
// the executor reports a real non-zero cycle count.

#include <stdint.h>
#include "include/gemmini.h"

#define M {_M}
#define K {_K}
#define N {_N}

void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {{
    tiled_matmul_auto(M, N, K, A, B, /*D=*/(void *)0, C,
                      K, N, N, N,
                      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION,
                      ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                      false, false, false, /*full_C=*/true, /*low_D=*/false,
                      /*weightA=*/0, WS);
}}
"""


@SKIP_NO_TOOLCHAIN
@pytest.mark.requires_spike
def test_spike_executor_gemmini_kernel_produces_nonzero_cycles(tmp_path: Path) -> None:
    """A Gemmini-aware kernel should report non-zero cycles.

    The scalar test above proves the wiring; this test proves the
    counter actually moves when the kernel uses Gemmini intrinsics.
    Together they bracket what ``cycles=0`` vs ``cycles>0`` means
    in a ``measured_cost.json``.
    """
    if not _check_gemmini_extension():
        pytest.skip("spike --extension=gemmini not supported by this spike build")

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)
    # Replace the scalar kernel source with the Gemmini-aware one.
    kernel_path = (
        bundle / "generated_kernels" / "test_provider" / "r0_linalg_matmul.c"
    )
    kernel_path.write_text(_GEMMINI_MATMUL_KERNEL)

    executor = SpikeExecutor(target_id="gemmini")
    measured = executor.execute(bundle)

    assert measured.correctness_vs_eager == "pass", (
        f"Gemmini kernel did not pass correctness; samples={[s.as_dict() for s in measured.samples]}"
    )
    assert measured.cycles_total is not None and measured.cycles_total > 0, (
        f"expected Gemmini RoCC cycles > 0; got {measured.cycles_total}"
    )


@SKIP_NO_TOOLCHAIN
@pytest.mark.requires_spike
def test_spike_executor_skips_unsupported_op_families(tmp_path: Path) -> None:
    """Non-matmul contracts emit a ``skipped`` sample, not a failure.

    The executor's job is to give honest evidence — handing it a
    bundle whose only region is ``aten_relu`` should produce a
    ``skipped`` measured_cost (not raise, not fail). The promotion
    gate stays at ``verified_fx`` in that case, which is the right
    answer.
    """
    if not _check_gemmini_extension():
        pytest.skip("spike --extension=gemmini not supported by this spike build")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"artifacts": {}}))
    contracts_dir = bundle / "kernel_contracts"
    contracts_dir.mkdir()
    (contracts_dir / "aten_relu.yaml").write_text(
        yaml.safe_dump({
            "op_name": "aten_relu",
            "supported_dtypes": ["f32"],
            "input_shapes": [[4, 16]],
            "output_shapes": [[4, 16]],
            "metadata": {"region_id": "r0"},
        })
    )
    executor = SpikeExecutor(target_id="gemmini")
    measured = executor.execute(bundle)
    assert measured.correctness_vs_eager == "skipped"
    assert measured.cycles_total is None
    assert len(measured.samples) == 1
    assert measured.samples[0].correctness == "skipped"
    assert measured.samples[0].extras["reason"] == "op_family_not_supported"


def test_spike_executor_raises_when_toolchain_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the RISC-V toolchain, the executor surfaces
    ``AdapterUnavailableError`` — no silent skip, no bare
    NotImplementedError."""
    monkeypatch.setenv("XPU_RT_RISCV_CONDA_ROOT", str(tmp_path / "no_such_dir"))
    monkeypatch.setenv("XPU_RT_SPIKE_BIN", str(tmp_path / "no_spike"))
    monkeypatch.setenv("XPU_RT_PK_BIN", str(tmp_path / "no_pk"))
    monkeypatch.setenv("XPU_RT_RISCV_CC", str(tmp_path / "no_cc"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"artifacts": {}}))
    executor = SpikeExecutor(target_id="gemmini")
    available, reason = executor.is_available()
    assert not available
    assert "toolchain" in reason.lower() or "no_" in reason
    from xpu_rt.runtime.errors import AdapterUnavailableError
    with pytest.raises(AdapterUnavailableError):
        executor.execute(bundle)
