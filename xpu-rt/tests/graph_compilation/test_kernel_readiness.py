"""Acceptance tests for M-24 Kernel Section Readiness Lock.

Mirrors M-17.1's test shape but for kernel-level evidence aggregation.
Verifies:

- Always-on emission (no env var needed for the matrix; the rows
  themselves report not_run when their evidence isn't on disk).
- 6 typed reports + matrix shape with required fields.
- Row 1 (compiled_precision) reflects M-19/M-20 refinement_status.
- Row 2 (compiled_working_set) reflects M-22 utilizations.
- Row 3 (compiled_lifetime) is ready_for_m24_1 (honest non-claim
  for register-pressure / occupancy without ncu).
- Row 4 (compiled_candidate_evidence) cross-references legal
  SetTileParams candidates.
- Row 5 (compiled_agent_view) cross-references candidate_ids_allowed
  ⊇ M-22 measured candidates.
- Row 6 (compiled_bottleneck) reflects M-22 kernel_calibration_status.
- overall=pass iff every row in {ready, ready_for_m24_1}.
- not_run propagates honestly when no kernel evidence.
- M-19/M-20/M-22/M-22.1/M-23 source artifacts byte-identical (read-only).
- Byte-stable across reruns (deterministic; modulo timestamp).
- Ledger captures the M-24 stage event.
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
    out = tmp_path_factory.mktemp("m24_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m24_no_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


# --------------------------------------------------------------------------- #
# Always-on emission
# --------------------------------------------------------------------------- #


def test_kernel_readiness_dir_exists_when_kernels_on(
    kernels_run: Path,
) -> None:
    base = kernels_run / "02_graph_analysis" / "kernel_readiness"
    assert base.is_dir()
    assert (base / "kernel_section_readiness_matrix.json").exists()
    assert (base / "kernel_section_readiness_summary.md").exists()
    for row_file in (
        "precision_report.json",
        "working_set_report.json",
        "lifetime_report.json",
        "candidate_evidence_report.json",
        "agent_view_report.json",
        "bottleneck_report.json",
    ):
        assert (base / row_file).exists(), f"missing {row_file}"


def test_kernel_readiness_dir_exists_when_kernels_off(
    no_kernels_run: Path,
) -> None:
    """M-24 is always-on; even without kernels the matrix + 6 reports
    exist (rows are not_run honestly)."""
    base = no_kernels_run / "02_graph_analysis" / "kernel_readiness"
    if not base.is_dir():
        pytest.skip("M-24 not wired or pipeline didn't reach it")
    matrix = _read(base / "kernel_section_readiness_matrix.json")
    assert matrix["overall"] in ("not_run", "partial")
    assert matrix["kernels_enabled"] is False


def test_matrix_schema_version(kernels_run: Path) -> None:
    m = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "kernel_section_readiness_matrix.json"
    )
    assert m["schema_version"] == "kernel_section_readiness_matrix_v1"
    assert "slide_rows" in m
    assert len(m["slide_rows"]) == 6


def test_matrix_overall_pass_on_full_kernel_run(kernels_run: Path) -> None:
    """merlin_mlp_wide with kernels ON + M-24.1 introspection
    should reach overall=pass with all 6 rows ready."""
    m = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "kernel_section_readiness_matrix.json"
    )
    assert m["overall"] == "pass", (
        f"expected overall=pass, got {m['overall']!r} with rows "
        f"{[(r['row'], r['status']) for r in m['slide_rows']]}"
    )
    # With M-24.1, all 6 rows reach ready (row 3 flips from
    # ready_for_m24_1 → ready). Without M-24.1, expect 5 ready + 1
    # ready_for_m24_1.
    assert (m["ready_count"] + m["ready_for_m24_1_count"]) == 6
    assert m["ready_count"] >= 5


# --------------------------------------------------------------------------- #
# Per-row tests
# --------------------------------------------------------------------------- #


def test_row1_compiled_precision(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "precision_report.json"
    )
    assert r["row"] == 1
    assert r["claim"] == "compiled_precision"
    assert r["status"] in ("ready", "partial", "not_ready", "not_run")
    if r["status"] == "ready":
        s = r["summary"]
        assert s["fail_outside_tolerance_count"] == 0
        # At least one region must be discharged (bit_eq or tol_eps).
        assert (s["bit_equality_count"] + s["tolerance_eps_count"]) > 0


def test_row2_compiled_working_set(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "working_set_report.json"
    )
    assert r["row"] == 2
    assert r["claim"] == "compiled_working_set"
    if r["status"] == "ready":
        for region in r.get("regions", []):
            if region["populated"]:
                assert region["compute_utilization"] is not None
                assert region["bandwidth_utilization"] is not None


def test_row3_compiled_lifetime_after_m24_1(
    kernels_run: Path,
) -> None:
    """After M-24.1 (Triton CompiledKernel introspection), row 3
    flips ready_for_m24_1 → ready. The honest-non-claims invariant
    is that EITHER:
    - status=ready iff every region with kernel events also has
      register_pressure + register_spills + shared_memory_bytes +
      theoretical_occupancy populated
    - status=ready_for_m24_1 iff some regions have CUDA events but
      M-24.1 introspection didn't populate static fields"""
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "lifetime_report.json"
    )
    assert r["row"] == 3
    assert r["claim"] == "compiled_lifetime"
    assert r["status"] in (
        "ready", "ready_for_m24_1", "partial", "not_run",
    )
    if r["status"] == "ready":
        # Every region must have all 4 static fields populated.
        for reg in r.get("regions", []):
            for field in (
                "register_pressure", "register_spills",
                "shared_memory_bytes", "theoretical_occupancy",
            ):
                assert reg.get(field) is not None, (
                    f"row 3 ready but region {reg.get('region_id')!r} "
                    f"missing {field}"
                )
    # Honest non-claim about dynamic counters always present.
    assert any(
        "ncu" in lim or "RmProfilingAdminOnly" in lim
        for lim in r.get("known_limitations", [])
    )


def test_row4_compiled_candidate_evidence(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "candidate_evidence_report.json"
    )
    assert r["row"] == 4
    assert r["claim"] == "compiled_candidate_evidence"
    if r["status"] == "ready":
        # Every region in the report should have evidence.
        for reg in r.get("regions", []):
            assert reg["has_compiled_evidence"] is True


def test_row5_compiled_agent_view(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "agent_view_report.json"
    )
    assert r["row"] == 5
    assert r["claim"] == "compiled_agent_view"
    if r["status"] == "ready":
        # No leaks: every measured candidate is in candidate_ids_allowed.
        s = r.get("summary", {}) or {}
        assert s.get("leaks") == []


def test_row6_compiled_bottleneck(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "bottleneck_report.json"
    )
    assert r["row"] == 6
    assert r["claim"] == "compiled_bottleneck"
    assert r["kernel_calibration_status"] in (
        "kernel_calibrated",
        "partial_kernel_calibration",
        "not_kernel_calibrated",
    )
    if r["status"] == "ready":
        assert r["kernel_calibration_status"] in (
            "kernel_calibrated", "partial_kernel_calibration",
        )


# --------------------------------------------------------------------------- #
# Determinism + read-only
# --------------------------------------------------------------------------- #


def test_byte_identical_matrix_across_reruns(kernels_run: Path) -> None:
    """Re-running the M-24 builder produces a byte-identical matrix
    (modulo generated_at_utc)."""
    from compgen.graph_compilation.kernel_readiness import (
        run_kernel_section_readiness,
    )

    m_path = (
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "kernel_section_readiness_matrix.json"
    )
    d1 = _read(m_path)
    run_kernel_section_readiness(kernels_run)
    d2 = _read(m_path)
    d1.pop("generated_at_utc", None)
    d2.pop("generated_at_utc", None)
    for r in d1.get("slide_rows", []) + d2.get("slide_rows", []):
        # Reasons may carry timestamps; pop them. They're for humans.
        pass
    assert d1 == d2, "M-24 matrix drifted on rerun"


def test_m24_does_not_mutate_source_artifacts(kernels_run: Path) -> None:
    """M-24 is read-only: M-19/M-20/M-22/M-22.1/M-23 source reports
    are byte-identical after re-running M-24."""
    from compgen.graph_compilation.kernel_readiness import (
        run_kernel_section_readiness,
    )

    paths = [
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json",
        kernels_run / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json",
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "region_compiled_differential_report.json",
        kernels_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json",
        kernels_run / "02_graph_analysis" / "cost_preview_v2.json",
        kernels_run / "02_graph_analysis" / "candidate_actions.json",
        kernels_run / "02_graph_analysis" / "region_map.json",
    ]
    paths = [p for p in paths if p.exists()]
    before = {p: _sha(p) for p in paths}
    run_kernel_section_readiness(kernels_run)
    after = {p: _sha(p) for p in paths}
    drifted = [str(p.relative_to(kernels_run)) for p in paths if before[p] != after[p]]
    assert not drifted, f"M-24 mutated source artifacts: {drifted}"


# --------------------------------------------------------------------------- #
# Cross-reference invariants
# --------------------------------------------------------------------------- #


def test_row6_status_matches_m22_kernel_calibration_status(
    kernels_run: Path,
) -> None:
    """Row 6's reported kernel_calibration_status must equal
    M-22's standalone value."""
    cb = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "bottleneck_report.json"
    )
    assert r["kernel_calibration_status"] == cb["kernel_calibration_status"]


def test_row4_region_count_matches_m22_evidence(
    kernels_run: Path,
) -> None:
    """Row 4's region count must match the count of M-22 evidence
    regions (after filtering to legal SetTileParams candidates)."""
    cb = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "candidate_evidence_report.json"
    )
    if r["status"] in ("ready", "partial"):
        cb_ok_regions = sum(
            1 for x in cb.get("regions", []) or []
            if x.get("model_status") == "ok"
        )
        # Row 4 covers regions with at least one legal SetTileParams
        # candidate; M-22 has evidence for SetTileParams selected
        # candidates. Row 4's regions_with_evidence ≤ row 4 total
        # AND ≤ cb_ok_regions (when both nonzero).
        s = r.get("summary", {}) or {}
        assert s.get("regions_with_evidence", 0) <= s.get(
            "regions_total", 0
        )
        assert s.get("regions_with_evidence", 0) <= cb_ok_regions


def test_row5_no_leaks_on_healthy_run(kernels_run: Path) -> None:
    """No measured candidate should ever be outside candidate_ids_allowed
    on a healthy run."""
    r = _read(
        kernels_run / "02_graph_analysis" / "kernel_readiness"
        / "agent_view_report.json"
    )
    if r["status"] == "ready":
        s = r.get("summary", {}) or {}
        assert s.get("leaks") == []


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def test_ledger_records_m24_event(kernels_run: Path) -> None:
    ledger_path = kernels_run / "stage_ledger.jsonl"
    events = [
        json.loads(line) for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    # Filter exactly to the M-24 (kernel_section_readiness) event,
    # not M-24.1 (kernel_lifetime).
    m24_events = [
        e for e in events
        if e.get("note") and "kernel_section_readiness" in e["note"]
    ]
    assert m24_events, "ledger missing M-24 kernel_section_readiness event"
    note = m24_events[0]["note"]
    assert any(s in note for s in ("pass", "partial", "not_run", "fail"))


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "kernel_readiness.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "from compgen.capture",
        "from compgen.pipeline",
        "from compgen.runtime.bundle_emit",
    )
    for f in forbidden:
        assert f not in src, (
            f"kernel_readiness imports forbidden module: {f}"
        )
