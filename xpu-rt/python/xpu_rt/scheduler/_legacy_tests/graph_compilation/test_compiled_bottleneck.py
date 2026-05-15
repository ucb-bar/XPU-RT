"""Acceptance tests Compiled Bottleneck Analysis.

Verifies:

Deterministic post-hoc derivation: same / + target
  YAML inputs → byte-identical output across reruns.
- Achieved compute/bandwidth fractions are correctly derived from
  measured time × analytical flops/bytes.
Bottleneck classification cross-references the analytical
  prediction; agreement / disagreement counted explicitly.
- ``compiled_evidence`` overlay is layered onto
  ``hardware_resource_report.json`` per region (additive only —
  the existing fields stay untouched).
- Top-level ``kernel_calibration_status`` flips ``not_kernel_calibrated``
  → ``kernel_calibrated`` (or ``partial_kernel_calibration``) based on
  region coverage.
analytical_cost / cost_preview_v2 / region_map / candidate_actions
  byte-identical across reruns (only writes to its own
  directory + the hardware_resource_report overlay).
Best-effort: missing /measurements → typed ``no_measurements``
  with kernel_calibration_status=not_kernel_calibrated; never raises.
- No compiler-core imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _run(model: str, out_dir: Path, *, run_kernels: bool) -> None:
    env = os.environ.copy()
    if run_kernels:
        env["COMPGEN_RUN_KERNELS"] = "1"
    else:
        env.pop("COMPGEN_RUN_KERNELS", None)
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)
    subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide with /kernels ON — gives us per-region
    compiled measurements to derive utilization from."""
    out = tmp_path_factory.mktemp("m22_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide with kernels OFF — emits no_measurements."""
    out = tmp_path_factory.mktemp("m22_no_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


# --------------------------------------------------------------------------- #
# Always-on emission
# --------------------------------------------------------------------------- #


def test_compiled_bottleneck_dir_exists_when_kernels_on(
    kernels_run: Path,
) -> None:
    base = kernels_run / "02_graph_analysis" / "compiled_bottleneck"
    assert base.is_dir()
    assert (base / "compiled_bottleneck_report.json").exists()
    assert (base / "compiled_bottleneck_summary.md").exists()


def test_compiled_bottleneck_emits_no_measurements_when_kernels_off(
    no_kernels_run: Path,
) -> None:
    p = (
        no_kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if not p.exists():
        pytest.skip("M-22 not wired or capture failed")
    r = _read(p)
    assert r["overall"] == "no_measurements"
    assert r["kernel_calibration_status"] == "not_kernel_calibrated"
    assert r["regions"] == []


def test_artifact_schema_version(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    assert r["schema_version"] == "compiled_bottleneck_report_v1"
    assert r["model_kind"] == "post_hoc_utilization_v1"
    assert r["deterministic"] is True


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_byte_identical_reruns(kernels_run: Path) -> None:
    """Calling run_compiled_bottleneck twice on the same run dir must
    produce byte-identical JSON output (modulo generated_at_utc and
    the profiler_evidence overlay, which is layered post-hoc by
    a separate pass)."""
    from compgen.graph_compilation.compiled_bottleneck import (
        run_compiled_bottleneck,
    )

    p = (
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    d1 = _read(p)
    run_compiled_bottleneck(kernels_run)
    d2 = _read(p)
    d1.pop("generated_at_utc", None)
    d2.pop("generated_at_utc", None)
    # layers profiler_evidence and replaces cache_evidence after
    # finishes. Re-running alone overwrites the file without
    # the overlay; strip both fields before comparing 's
    # own determinism.
    for d in (d1, d2):
        for r in d.get("regions", []) or []:
            r.pop("profiler_evidence", None)
            r["cache_evidence"] = "not_collected"
    assert d1 == d2, "M-22 output drifted on rerun"


def test_pure_function_derive_utilization_is_deterministic() -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        derive_utilization,
    )

    args = dict(
        flops=16384, bytes_moved=6144,
        measured_us=21.33,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    a = derive_utilization(**args)
    b = derive_utilization(**args)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Math: utilization = analytical_quantity / (peak * measured_s)
# --------------------------------------------------------------------------- #


def test_achieved_compute_gflops_math() -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        derive_utilization,
    )

    # 16384 flops in 21.33 us → 16384 / (21.33e-6) = 768e6 flops/s = 0.768 GFLOPS
    out = derive_utilization(
        flops=16384, bytes_moved=6144,
        measured_us=21.33,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    expected = (16384 / 1e9) / (21.33e-6)
    assert out["achieved_compute_gflops"] == pytest.approx(expected, rel=1e-9)
    # compute_utilization = achieved / peak = 0.768 / 100 ≈ 0.00768
    assert out["compute_utilization"] == pytest.approx(expected / 100.0, rel=1e-9)


def test_achieved_bandwidth_gb_s_math() -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        derive_utilization,
    )

    out = derive_utilization(
        flops=16384, bytes_moved=6144,
        measured_us=21.33,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    expected = (6144 / 1e9) / (21.33e-6)
    assert out["achieved_bandwidth_gb_s"] == pytest.approx(expected, rel=1e-9)
    assert out["bandwidth_utilization"] == pytest.approx(
        expected / 30.0, rel=1e-9
    )


def test_doubling_measured_time_halves_utilization() -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        derive_utilization,
    )

    base = derive_utilization(
        flops=16384, bytes_moved=6144,
        measured_us=10.0,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    doubled = derive_utilization(
        flops=16384, bytes_moved=6144,
        measured_us=20.0,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    assert base["compute_utilization"] == pytest.approx(
        2 * doubled["compute_utilization"], rel=1e-9
    )
    assert base["bandwidth_utilization"] == pytest.approx(
        2 * doubled["bandwidth_utilization"], rel=1e-9
    )


def test_measured_bottleneck_classification() -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        derive_utilization,
    )

    # High AI: 1e9 flops / 1e3 bytes = compute-bound on any peak.
    high_ai = derive_utilization(
        flops=10**9, bytes_moved=10**3,
        measured_us=1.0,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    assert high_ai["measured_bottleneck"] == "compute"

    # Low AI: 10 flops / 1e9 bytes = memory-bound on any peak.
    low_ai = derive_utilization(
        flops=10, bytes_moved=10**9,
        measured_us=1.0,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
    )
    assert low_ai["measured_bottleneck"] == "memory"


# --------------------------------------------------------------------------- #
# Per-region coverage
# --------------------------------------------------------------------------- #


def test_every_compiled_region_is_in_m22_report(
    kernels_run: Path,
) -> None:
    """Every region with an compiled measurement should
    appear 's per-region list with model_status=ok."""
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    m20_path = base / "region_compiled_differential_report.json"
    if not m20_path.exists():
        pytest.skip("M-20 report not on disk")
    m20 = _read(m20_path)
    compiled_region_ids = {
        r["region_id"] for r in m20.get("regions", []) or []
        if (r.get("gpu") or {}).get("compile_status") == "compiled"
        or (r.get("cpu") or {}).get("compile_status") == "compiled"
    }
    if not compiled_region_ids:
        pytest.skip("no compiled regions in M-20 report")

    m22 = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    m22_ok_ids = {
        r["region_id"] for r in m22.get("regions", [])
        if r.get("model_status") == "ok"
    }
    missing = compiled_region_ids - m22_ok_ids
    assert not missing, (
        f"M-22 missing regions present in M-20: {sorted(missing)}"
    )


def test_per_region_evidence_has_expected_fields(kernels_run: Path) -> None:
    m22 = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    ok_regions = [
        r for r in m22.get("regions", [])
        if r.get("model_status") == "ok"
    ]
    assert ok_regions, "no ok regions in M-22 report"
    for r in ok_regions:
        for field in (
            "region_id", "candidate_id",
            "matmul_shape", "tile",
            "analytical_flops", "analytical_bytes_moved",
            "analytical_bottleneck", "measured_bottleneck",
            "canonical_track", "bottleneck_classification_agreement",
            "cache_evidence", "source",
        ):
            assert field in r, f"region missing field {field}"
        assert r["cache_evidence"] == "not_collected"
        assert r["source"] in ("m19", "m20")
        # At least one of gpu/cpu blocks present.
        assert r.get("gpu") is not None or r.get("cpu") is not None


# --------------------------------------------------------------------------- #
# Layered overlay onto hardware_resource_report
# --------------------------------------------------------------------------- #


def test_hardware_resource_report_has_compiled_evidence_overlay(
    kernels_run: Path,
) -> None:
    hrr_path = (
        kernels_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    hrr = _read(hrr_path)
    assert "kernel_calibration_status" in hrr, (
        "hardware_resource_report missing top-level kernel_calibration_status"
    )
    overlaid = [
        r for r in hrr.get("regions", []) or []
        if "compiled_evidence" in r
    ]
    assert overlaid, (
        "no compiled_evidence overlay on any region in hardware_resource_report"
    )
    for r in overlaid:
        ev = r["compiled_evidence"]
        assert "measured_bottleneck" in ev
        assert "analytical_bottleneck" in ev
        assert "bottleneck_classification_agreement" in ev


def test_hardware_resource_calibration_status_field_unchanged(
    kernels_run: Path,
) -> None:
    """the deterministic-baseline ``calibration_status`` field
    must stay ``not_profiler_calibrated``. only adds a NEW
    ``kernel_calibration_status`` field; it does not mutate the
    existing field."""
    hrr = _read(
        kernels_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    assert hrr.get("calibration_status") == "not_profiler_calibrated"


def test_kernel_calibration_status_enum(kernels_run: Path) -> None:
    hrr = _read(
        kernels_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    assert hrr["kernel_calibration_status"] in (
        "kernel_calibrated",
        "partial_kernel_calibration",
        "not_kernel_calibrated",
    )


# --------------------------------------------------------------------------- #
# Cross-reference with
# --------------------------------------------------------------------------- #


def test_m21_analytical_bottleneck_matches_m22_evidence(
    kernels_run: Path,
) -> None:
    """For every ok-region, the analytical_bottleneck recorded
    must match the standalone report for the same candidate."""
    m21 = _read(
        kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    m21_by_cid = {
        c["candidate_id"]: c.get("bottleneck_resource")
        for c in m21.get("candidates", [])
        if c.get("model_status") == "ok"
    }
    m22 = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    for r in m22.get("regions", []):
        if r.get("model_status") != "ok":
            continue
        cid = r["candidate_id"]
        if cid in m21_by_cid:
            assert r["analytical_bottleneck"] == m21_by_cid[cid], (
                f"{cid}: M-22 analytical_bottleneck "
                f"({r['analytical_bottleneck']}) != "
                f"M-21 ({m21_by_cid[cid]})"
            )


def test_agreement_count_matches_region_evidence(kernels_run: Path) -> None:
    m22 = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    expected_agree = sum(
        1 for r in m22.get("regions", [])
        if r.get("model_status") == "ok"
        and r.get("bottleneck_classification_agreement") is True
        and r.get("measured_bottleneck") in ("compute", "memory")
    )
    expected_disagree = sum(
        1 for r in m22.get("regions", [])
        if r.get("model_status") == "ok"
        and r.get("bottleneck_classification_agreement") is False
        and r.get("measured_bottleneck") in ("compute", "memory")
    )
    s = m22["summary"]["agreement_with_analytical"]
    assert s["agreement_count"] == expected_agree
    assert s["disagreement_count"] == expected_disagree


# --------------------------------------------------------------------------- #
# Byte-identity of unrelated artifacts (only writes its own dir +
# hardware_resource_report overlay)
# --------------------------------------------------------------------------- #


def test_m21_analytical_cost_unchanged_when_m22_reruns(
    kernels_run: Path,
) -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        run_compiled_bottleneck,
    )
    p = (
        kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    before = _sha(p)
    run_compiled_bottleneck(kernels_run)
    after = _sha(p)
    assert before == after, "M-21 analytical_cost mutated by M-22"


def test_m22_does_not_mutate_region_map_or_candidate_actions(
    kernels_run: Path,
) -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        run_compiled_bottleneck,
    )
    rm_path = kernels_run / "02_graph_analysis" / "region_map.json"
    ca_path = kernels_run / "02_graph_analysis" / "candidate_actions.json"
    cp_path = kernels_run / "02_graph_analysis" / "cost_preview_v2.json"
    before = {
        "region_map": _sha(rm_path),
        "candidate_actions": _sha(ca_path),
        "cost_preview_v2": _sha(cp_path),
    }
    run_compiled_bottleneck(kernels_run)
    after = {
        "region_map": _sha(rm_path),
        "candidate_actions": _sha(ca_path),
        "cost_preview_v2": _sha(cp_path),
    }
    assert before == after, (
        f"M-22 mutated immutable artifact(s): "
        f"{[k for k in before if before[k] != after[k]]}"
    )


def test_m20_report_unchanged_by_m22(kernels_run: Path) -> None:
    from compgen.graph_compilation.compiled_bottleneck import (
        run_compiled_bottleneck,
    )
    m20_path = (
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json"
    )
    if not m20_path.exists():
        pytest.skip("M-20 report not on disk")
    before = _sha(m20_path)
    run_compiled_bottleneck(kernels_run)
    after = _sha(m20_path)
    assert before == after, "M-20 report mutated by M-22"


def test_hardware_resource_overlay_byte_stable_across_m22_reruns(
    kernels_run: Path,
) -> None:
    """After the FIRST run, re-running produces a byte-
    identical hardware_resource_report (the overlay is deterministic)."""
    from compgen.graph_compilation.compiled_bottleneck import (
        run_compiled_bottleneck,
    )
    hrr_path = (
        kernels_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    before = _sha(hrr_path)
    run_compiled_bottleneck(kernels_run)
    after = _sha(hrr_path)
    assert before == after, (
        "hardware_resource_report overlay drifted on M-22 rerun"
    )


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "compiled_bottleneck.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "from compgen.capture",
        "from compgen.pipeline",
        "from compgen.runtime.bundle_emit",
    )
    for f in forbidden:
        assert f not in src, f"compiled_bottleneck imports forbidden module: {f}"
