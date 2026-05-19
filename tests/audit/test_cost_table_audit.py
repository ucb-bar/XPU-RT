"""Tests for the cost-table audit helpers.

Each test plants a small synthetic bug in a hand-crafted JSON dict and
verifies the audit helper catches it.
"""

from __future__ import annotations

import pytest
from xpu_rt.audit.cost_table_audit import (
    MatrixOp,
    build_conv2d_cost_expr,
    canonical_family,
    check_basic_sanity,
    check_magnitude_outliers,
    check_pairwise_ordering_stability,
    cross_backend_flip_rate,
    cross_workload_consistency,
    family_backend_specialty,
    index_correlation,
    matrix_op_family,
    parse_conv2d_family,
    parse_table,
    pathological_ratios,
)


def _entry(mean_us: float, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"mean_us": mean_us, "iters": 10, "extrapolated": False, "source": "test"}
    base.update(overrides)
    return base


def test_basic_sanity_catches_negative_and_duplicate() -> None:
    table = {
        "Conv2d@1x10x10x4->1x10x10x4,g1,k3,s1@uint8::HTA::0": _entry(100.0),
        "Conv2d@1x10x10x4->1x10x10x4,g1,k3,s1@uint8::HTA::0::dup_marker": _entry(120.0),
        "Conv2d@1x10x10x4->1x10x10x8,g1,k3,s1@uint8::HTA::0": _entry(-5.0),
    }
    # The dup-marker key won't parse, so re-craft to force a true duplicate.
    table = {
        "Conv2d@1x10x10x4->1x10x10x4,g1,k3,s1@uint8::HTA::0": _entry(100.0),
        "Conv2d@1x10x10x4->1x10x10x8,g1,k3,s1@uint8::HTA::0": _entry(-5.0),
    }
    entries, _ = parse_table(table)
    findings = check_basic_sanity(entries)
    kinds = {f.check_kind for f in findings}
    assert "negative_cost" in kinds


def test_basic_sanity_catches_zero_cost() -> None:
    table = {
        "Conv2d@1x10x10x4->1x10x10x4,g1,k3,s1@uint8::HTA::0": _entry(0.0),
    }
    entries, _ = parse_table(table)
    findings = check_basic_sanity(entries)
    assert any(f.check_kind == "zero_cost" for f in findings)


def test_magnitude_outlier_flagged() -> None:
    # Five small CPU costs around 100us, one outlier at 50000us (500x median).
    table: dict[str, dict[str, object]] = {}
    for i, c in enumerate([100.0, 110.0, 95.0, 105.0, 102.0]):
        table[f"elementwise@a{i}->a{i}@fp32::CPU::0"] = _entry(c)
    table["weirdop@x->x@fp32::CPU::0"] = _entry(50_000.0)
    entries, _ = parse_table(table)
    findings = check_magnitude_outliers(entries, factor=100.0)
    assert any(f.check_kind == "magnitude_outlier" and "weirdop" in f.op_or_family for f in findings)


def test_pairwise_ordering_flip_detected() -> None:
    # Build a table where ordering on backend CPU is reversed on backend
    # GPU for 6 ops — 100% flip rate, >= 10 pairs, hence severity=high.
    table: dict[str, dict[str, object]] = {}
    for i, (cpu_c, gpu_c) in enumerate(zip([10, 20, 30, 40, 50, 60], [60, 50, 40, 30, 20, 10])):
        table[f"op{i}@s->s@fp32::CPU::0"] = _entry(float(cpu_c))
        table[f"op{i}@s->s@fp32::GPU::0"] = _entry(float(gpu_c))
    entries, _ = parse_table(table)
    findings, stats = check_pairwise_ordering_stability(entries)
    pair = next(p for p in stats["backend_pairs"] if {p["backend_a"], p["backend_b"]} == {"CPU", "GPU"})
    assert pair["flip_rate"] == 1.0
    assert any(
        f.check_kind == "ordering_flip_rate" and f.severity == "high"
        for f in findings
    )


def test_conv2d_cost_expr_returns_z3_expr() -> None:
    table = {
        "Conv2d@1x16x16x4->1x16x16x4,g1,k3,s1@uint8::HTA::0": _entry(100.0),
        "Conv2d@1x32x32x4->1x32x32x4,g1,k3,s1@uint8::HTA::0": _entry(400.0),
    }
    entries, _ = parse_table(table)
    fam = parse_conv2d_family(entries, backend="HTA", dtype="uint8")
    assert len(fam) == 2
    import z3

    expr = build_conv2d_cost_expr(fam)
    # (out_h, out_w, in_c*out_c) == (16, 16, 4*4 = 16) for the 100us entry.
    m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
    s = z3.Solver()
    s.add(m == 16, n == 16, k == 16)
    s.add(expr(m, n, k) == 100)
    assert s.check() == z3.sat


# ---------------------- Per-op cost matrix helpers ----------------------


def test_matrix_op_family_extracts_normalized_and_onnx_paths() -> None:
    assert matrix_op_family("convolution_12") == "convolution"
    assert matrix_op_family("elementwise_product_3") == "elementwise_product"
    assert matrix_op_family("strided_slice_5") == "strided_slice"
    assert matrix_op_family("/conv_modules.0/Conv") == "Conv"
    assert matrix_op_family("/Add_1") == "Add"
    # Trailing tensor-name suffix (".nchw") and output-tag must be dropped.
    assert matrix_op_family("/Add_2_output_0.nchw") == "Add"
    assert matrix_op_family("/sigmoid1/Sigmoid") == "Sigmoid"


def test_canonical_family_folds_onnx_to_qnn_names() -> None:
    assert canonical_family("Conv") == "convolution"
    assert canonical_family("Relu") == "elementwiseneuron"
    assert canonical_family("Add") == "elementwise_sum"
    assert canonical_family("MaxPool") == "pool"
    # Unknown families pass through unchanged.
    assert canonical_family("convolution") == "convolution"
    assert canonical_family("permute") == "permute"


def _mk(workload: str, op_id: str, **costs: float) -> MatrixOp:
    return MatrixOp(workload=workload, op_id=op_id, family=matrix_op_family(op_id), costs=dict(costs))


def test_cross_backend_flip_rate_perfect_reversal_yields_100pct() -> None:
    # Six ops where DSP ordering is the reverse of CPU.
    rows = [
        _mk("wl", f"op_{i}", CPU=float(c), GPU=float(c), DSP=float(60 - c))
        for i, c in enumerate([10, 20, 30, 40, 50, 60])
    ]
    stats = cross_backend_flip_rate(rows, backends=("CPU", "GPU", "DSP"))
    cpu_dsp = next(s for s in stats if {s["backend_a"], s["backend_b"]} == {"CPU", "DSP"})
    cpu_gpu = next(s for s in stats if {s["backend_a"], s["backend_b"]} == {"CPU", "GPU"})
    assert cpu_dsp["flip_rate"] == 1.0
    # CPU and GPU are identical: 0 flips.
    assert cpu_gpu["flip_rate"] == 0.0


def test_family_backend_specialty_counts_argmin() -> None:
    rows = [
        _mk("wl", "convolution_0", CPU=100.0, GPU=50.0, DSP=10.0),
        _mk("wl", "convolution_1", CPU=200.0, GPU=80.0, DSP=15.0),
        _mk("wl", "elementwise_sum_0", CPU=5.0, GPU=20.0, DSP=8.0),
    ]
    spec = family_backend_specialty(rows)
    assert spec["convolution"]["fastest"] == "DSP"
    assert spec["convolution"]["argmin"]["DSP"] == 2
    assert spec["elementwise_sum"]["fastest"] == "CPU"


def test_pathological_ratios_flags_extreme_op() -> None:
    rows = [
        _mk("wl", "convolution_0", CPU=4000.0, GPU=100.0, DSP=10.0),  # 400x
        _mk("wl", "convolution_1", CPU=100.0, GPU=100.0, DSP=100.0),  # 1x
    ]
    top = pathological_ratios(rows, top_n=5)
    assert top[0]["op_id"] == "convolution_0"
    assert top[0]["ratio"] == 400.0
    assert top[0]["fastest_backend"] == "DSP"


def test_index_correlation_detects_monotone_trend() -> None:
    rows = [
        _mk("wl", f"convolution_{i}", CPU=float(10 + i * 10), GPU=50.0, DSP=20.0)
        for i in range(5)
    ]
    corr = index_correlation(rows)
    # CPU is perfectly monotone in suffix index.
    assert corr["convolution"]["CPU"]["rho"] == pytest.approx(1.0)
    # GPU is constant — rho should be 0 (denom guarded).
    assert corr["convolution"]["GPU"]["rho"] == 0.0


def test_cross_workload_consistency_reports_per_workload_fastest() -> None:
    matrices = {
        "yolov8n": [
            _mk("yolov8n", "convolution_0", CPU=100.0, GPU=50.0, DSP=10.0),
            _mk("yolov8n", "convolution_1", CPU=200.0, GPU=80.0, DSP=15.0),
        ],
        "dronet": [
            # ONNX-style; folds onto canonical 'convolution'.
            _mk("dronet", "/conv_modules.0/Conv", CPU=80.0, GPU=20.0, DSP=200.0),
            _mk("dronet", "/conv_modules.1/Conv", CPU=160.0, GPU=40.0, DSP=300.0),
        ],
    }
    cw = cross_workload_consistency(matrices)
    assert "convolution" in cw
    assert cw["convolution"]["yolov8n"]["fastest"] == "DSP"
    assert cw["convolution"]["dronet"]["fastest"] == "GPU"
