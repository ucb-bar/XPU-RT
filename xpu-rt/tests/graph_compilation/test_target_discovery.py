"""Acceptance tests for ``discover-target``.

Probes the running host (Linux). Tests assert the *shape* of the
discovery output, not specific hardware values, so they remain portable
across CI runners with different CPUs and accelerator availability.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from compgen.graph_compilation.region_dossier import load_target_profile
from compgen.graph_compilation.target_discovery import (
    CPUInfo,
    build_target_yaml,
    cpu_supported_dtypes,
    discover_cpu,
    estimate_cpu_peak_bandwidth,
    estimate_cpu_peak_gflops,
)


def test_discover_cpu_returns_real_data() -> None:
    cpu = discover_cpu()
    assert cpu.logical_cores >= 1
    assert cpu.physical_cores >= 1
    assert cpu.physical_cores <= cpu.logical_cores
    assert cpu.flags, "expected at least one CPU flag from /proc/cpuinfo"
    # Page size is the most universal getconf field; expect 4 KiB on x86_64.
    if cpu.page_size_bytes:
        assert cpu.page_size_bytes >= 4096


def test_estimate_cpu_peak_gflops_avx512() -> None:
    cpu = CPUInfo(
        physical_cores=4, logical_cores=8, base_freq_mhz=3000, max_freq_mhz=4000,
        flags=["avx512f", "avx2", "fma", "avx"],
    )
    peak = estimate_cpu_peak_gflops(cpu)
    # 32 ops/cycle × 4 GHz × 4 cores = 512 GFLOPS
    assert peak == pytest.approx(512.0, rel=0.01)


def test_estimate_cpu_peak_gflops_avx2() -> None:
    cpu = CPUInfo(
        physical_cores=8, logical_cores=16, base_freq_mhz=3500, max_freq_mhz=4500,
        flags=["avx2", "fma", "avx"],
    )
    peak = estimate_cpu_peak_gflops(cpu)
    # 16 ops/cycle × 4.5 GHz × 8 cores = 576 GFLOPS
    assert peak == pytest.approx(576.0, rel=0.01)


def test_estimate_cpu_peak_gflops_scalar_only() -> None:
    cpu = CPUInfo(
        physical_cores=2, logical_cores=2, base_freq_mhz=2000, max_freq_mhz=2000,
        flags=[],
    )
    peak = estimate_cpu_peak_gflops(cpu)
    # 2 ops/cycle × 2 GHz × 2 cores = 8 GFLOPS
    assert peak == pytest.approx(8.0, rel=0.01)


def test_estimate_cpu_peak_bandwidth_class_buckets() -> None:
    workstation = CPUInfo(physical_cores=24, logical_cores=48, sockets=1)
    desktop = CPUInfo(physical_cores=4, logical_cores=8, sockets=1)
    server = CPUInfo(physical_cores=32, logical_cores=64, sockets=2)
    assert estimate_cpu_peak_bandwidth(workstation) == 80.0
    assert estimate_cpu_peak_bandwidth(desktop) == 30.0
    assert estimate_cpu_peak_bandwidth(server) == 200.0


def test_cpu_supported_dtypes_basic() -> None:
    base = CPUInfo(flags=[])
    half = CPUInfo(flags=["f16c"])
    bf = CPUInfo(flags=["avx512_bf16", "f16c"])
    assert cpu_supported_dtypes(base) == ["fp32"]
    assert "fp16" in cpu_supported_dtypes(half)
    assert "bf16" in cpu_supported_dtypes(bf)


def test_build_target_yaml_writes_loadable_profile(tmp_path: Path) -> None:
    out = tmp_path / "auto.yaml"
    obj = build_target_yaml(out_path=out, target_id="test_host_cpu")
    # Required structure
    assert obj["schema_version"] == "graphcomp_target_config_v1"
    assert obj["target_id"] == "test_host_cpu"
    assert obj["auto_discovered"] is True
    assert "discovery_provenance" in obj
    assert "cpu" in obj["discovery_provenance"]
    assert "memory_tiers" in obj
    assert "numerical_budgets" in obj
    assert obj["peak_compute_gflops"] > 0
    assert obj["peak_bandwidth_gb_s"] > 0
    # Round-trips through load_target_profile.
    profile = load_target_profile(out)
    assert profile.target_id == "test_host_cpu"
    assert profile.peak_compute_gflops == obj["peak_compute_gflops"]
    assert profile.scratchpad_bytes >= 4096
    # Numerical budgets monotone (invariant).
    nb = profile.numerical_budgets
    assert nb["fp32"] <= nb["fast_math"] <= nb["fp16_accum"] <= nb["fp8_e4m3"]


def test_discover_target_yaml_round_trip_drives_run_suite(tmp_path: Path) -> None:
    """End-to-end: an auto-discovered profile drives a real graph-analysis
    run without crashing and yields a healthy dossier."""
    from compgen.graph_compilation.run import run_graph_compilation

    out_yaml = tmp_path / "host.yaml"
    build_target_yaml(out_path=out_yaml, target_id="auto_host_test")
    repo = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "run"
    run_graph_compilation(
        model_config_path=repo / "configs" / "models" / "tiny_mlp.yaml",
        target_config_path=out_yaml,
        out_dir=run_dir,
        stop_after="graph-analysis",
        run_id="auto_target_smoke",
    )
    # Region dossiers were produced and dossier_validation passes.
    import json
    val = json.loads((run_dir / "02_graph_analysis" / "dossier_validation.json").read_text())
    assert val["overall"] == "pass"
    audit = json.loads(
        (run_dir / "02_graph_analysis" / "numerical_sensitivity_audit.json").read_text()
    )
    assert audit["status"] == "pass"


def test_provenance_records_uname(tmp_path: Path) -> None:
    obj = build_target_yaml(out_path=tmp_path / "p.yaml")
    uname = obj["discovery_provenance"]["host_uname"]
    # Linux runners always have these fields populated.
    if uname:
        assert {"sysname", "machine"} <= set(uname.keys())


def test_cpu_info_dataclass_serializes() -> None:
    cpu = discover_cpu()
    d = asdict(cpu)
    # The provenance YAML embeds this dict; ensure it's plain-data-serializable.
    text = yaml.safe_dump(d, sort_keys=True)
    assert "model_name" in text
