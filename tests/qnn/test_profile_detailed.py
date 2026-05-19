"""Unit tests for the qnn-profile-viewer CSV parser."""

from __future__ import annotations

from xpu_rt.targets.backends.qnn.profile_detailed import (
    OpTiming, _extract_csv, parse_profile_csv,
)


SAMPLE_QNN_PROFILE_CSV = (
    "Op Name,Op Type,Compute Unit,Backend (us)\n"
    "Conv2d_0,Conv2D,HVX,1850.0\n"
    "Relu_0,Relu,HMX,12.4\n"
    "Conv2d_1,Conv2D,HVX,2100.5\n"
    "Pool2d_0,MaxPool,HVX,84.2\n"
)


def test_parse_canonical_csv_yields_op_timings():
    rows = parse_profile_csv(SAMPLE_QNN_PROFILE_CSV, backend="DSP")
    assert len(rows) == 4
    assert isinstance(rows[0], OpTiming)
    assert rows[0].op_name == "Conv2d_0"
    assert rows[0].op_kind == "Conv2D"
    assert rows[0].backend == "DSP"
    assert rows[0].compute_unit == "HVX"
    assert rows[0].mean_us == 1850.0
    assert rows[2].mean_us == 2100.5


def test_parse_tolerates_alternate_column_names():
    csv_text = (
        "Node Name,Node Type,Execution Unit,Total Time (us)\n"
        "Conv,Conv,HVX,500.0\n"
    )
    rows = parse_profile_csv(csv_text, backend="GPU")
    assert len(rows) == 1
    assert rows[0].op_name == "Conv"
    assert rows[0].mean_us == 500.0
    assert rows[0].backend == "GPU"


def test_parse_skips_malformed_rows():
    csv_text = (
        "Op Name,Op Type,Compute Unit,Backend (us)\n"
        "GoodOp,Conv2D,HVX,100.0\n"
        "BadOp,Conv2D,HVX,not_a_number\n"
        "AnotherGood,Conv2D,HVX,200.0\n"
    )
    rows = parse_profile_csv(csv_text, backend="DSP")
    assert {r.op_name for r in rows} == {"GoodOp", "AnotherGood"}


def test_extract_csv_pulls_block_between_markers():
    stdout = (
        "junk line 1\n"
        "===CSV===\n"
        "Op Name,Backend (us)\n"
        "X,1.0\n"
        "===END===\n"
        "tail noise\n"
    )
    csv_text = _extract_csv(stdout)
    assert "Op Name" in csv_text
    assert "tail noise" not in csv_text


def test_extract_csv_returns_empty_when_markers_missing():
    assert _extract_csv("totally unrelated output") == ""


def test_empty_csv_returns_empty_list():
    assert parse_profile_csv("", backend="CPU") == []


def test_missing_required_columns_returns_empty():
    csv_text = "Op Name,Compute Unit\nConv,HVX\n"  # no latency column
    assert parse_profile_csv(csv_text, backend="DSP") == []
