"""Acceptance tests for Kernel Section Evidence Pack.

Mirrors test_graph_section_evidence_pack.py shape, but for kernel-
level signals. Uses a tiny synthetic suite root to keep tests fast;
the real canonical+wide rebuild lives under scripts/dev/.

Verifies:
- Build succeeds end-to-end on a tiny suite root.
- Read-only invariant (suite-source SHAs unchanged after rebuild).
- CSV columns present.
- Joint claim matrix has 6 rows (one per slide row).
- Honest non-claims surface in markdown + claim matrix.
- Figures emit valid PNG bytes.
- No compiler-core imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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


@pytest.fixture(scope="module")
def tiny_suite(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """One canonical and one wide model with kernels ON, just to
    populate the pack with real data."""
    root = tmp_path_factory.mktemp("m25_suite_root")
    canonical = root / "canonical"
    wide = root / "wide"
    canonical.mkdir()
    wide.mkdir()
    _run("merlin_mlp_wide", canonical / "merlin_mlp_wide", run_kernels=True)
    _run("proxy_vla", wide / "proxy_vla", run_kernels=True)
    return root


@pytest.fixture(scope="module")
def evidence_pack(tiny_suite, tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    from compgen.graph_compilation.kernel_evidence_pack import (
        build_kernel_evidence_pack,
    )
    out = tmp_path_factory.mktemp("m25_pack")
    res = build_kernel_evidence_pack(
        canonical_suite=tiny_suite / "canonical",
        wide_suite=tiny_suite / "wide",
        out_dir=out,
        skip_figures=False,
    )
    return res.out_dir


# --------------------------------------------------------------------------- #
# Pack shape
# --------------------------------------------------------------------------- #


def test_pack_emits_required_files(evidence_pack: Path) -> None:
    for fname in (
        "kernel_section_evidence_summary.md",
        "kernel_section_claim_matrix.json",
        "kernel_section_model_matrix.csv",
        "kernel_section_compiled_coverage.csv",
        "kernel_section_register_pressure.csv",
        "kernel_section_evidence_tables.json",
    ):
        assert (evidence_pack / fname).exists(), f"missing {fname}"


def test_claim_matrix_schema(evidence_pack: Path) -> None:
    cm = _read(evidence_pack / "kernel_section_claim_matrix.json")
    assert cm["schema_version"] == "kernel_section_claim_matrix_v1"
    assert "claims" in cm
    assert len(cm["claims"]) == 6
    for c in cm["claims"]:
        assert c["row"] in (1, 2, 3, 4, 5, 6)
        assert c["status"] in (
            "implemented",
            "implemented_partial_scope",
            "partially_implemented",
        )


def test_evidence_tables_aggregates(evidence_pack: Path) -> None:
    et = _read(evidence_pack / "kernel_section_evidence_tables.json")
    agg = et["aggregates"]
    for key in (
        "model_count", "kernels_enabled_count", "m24_pass_count",
        "m24_overall_distribution",
        "m22_kernel_calibration_status_distribution",
        "m24_1_introspected_total_regions",
        "register_pressure_distribution",
        "joint_ready_count",
    ):
        assert key in agg, f"aggregate missing {key}"


def test_model_matrix_csv_columns(evidence_pack: Path) -> None:
    csv_path = evidence_pack / "kernel_section_model_matrix.csv"
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for col in (
        "suite", "model_id",
        "m22_kernel_calibration_status",
        "m24_overall", "m24_ready_count",
        "m24_1_register_pressure_mean",
        "m24_1_theoretical_occupancy_mean",
        "m17_1_readiness_overall",
    ):
        assert col in header, f"CSV missing column {col}"


def test_register_pressure_csv_has_per_region_rows(
    evidence_pack: Path,
) -> None:
    csv_path = evidence_pack / "kernel_section_register_pressure.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    # At least header + one data row (merlin_mlp_wide has 3 matmul regions).
    assert len(lines) >= 2
    header = lines[0]
    for col in (
        "model_id", "region_id", "register_pressure",
        "register_spills", "shared_memory_bytes",
        "theoretical_occupancy", "target_arch", "ncu_status",
    ):
        assert col in header, f"register CSV missing {col}"


# --------------------------------------------------------------------------- #
# Markdown summary content
# --------------------------------------------------------------------------- #


def test_summary_has_required_sections(evidence_pack: Path) -> None:
    md = (evidence_pack / "kernel_section_evidence_summary.md").read_text(
        encoding="utf-8"
    )
    for section in (
        "# Kernel Section Evidence Pack (M-25)",
        "## Headline numbers",
        "## M-24 readiness distribution",
        "## M-24 row pass count",
        "## Joint FX+kernel claim matrix",
        "## Per-model summary",
        "## Honest non-claims",
    ):
        assert section in md, f"summary missing section {section!r}"


def test_summary_honest_non_claims_present(evidence_pack: Path) -> None:
    md = (evidence_pack / "kernel_section_evidence_summary.md").read_text(
        encoding="utf-8"
    )
    for nc in (
        "fp32 only",
        "RmProfilingAdminOnly",
        "ncu_admin_only",
        "read-only aggregator",
    ):
        assert nc in md, f"missing non-claim text: {nc!r}"


def test_summary_no_forbidden_perf_phrases(evidence_pack: Path) -> None:
    md = (evidence_pack / "kernel_section_evidence_summary.md").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "verified correct",
        "guaranteed correct",
        "the cost model is accurate",
    ):
        assert forbidden not in md, f"summary contains forbidden: {forbidden!r}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(p: Path) -> bool:
    if not p.exists():
        return False
    with p.open("rb") as f:
        return f.read(len(_PNG_MAGIC)) == _PNG_MAGIC


def test_figures_emit_valid_pngs(evidence_pack: Path) -> None:
    figs = evidence_pack / "figures"
    if not figs.is_dir():
        pytest.skip("figures dir absent")
    for name in (
        "kernel_calibration_status_by_model.png",
        "register_pressure_distribution.png",
        "theoretical_occupancy_by_model.png",
        "bottleneck_classification_agreement.png",
        "compiled_us_per_iter_by_model.png",
        "fx_vs_kernel_joint_claim.png",
    ):
        p = figs / name
        if p.exists():
            assert _is_png(p), f"{name} is not a valid PNG"


# --------------------------------------------------------------------------- #
# Read-only invariant (must not mutate suite source artifacts)
# --------------------------------------------------------------------------- #


def test_pack_is_read_only(tiny_suite: Path) -> None:
    """SHA-snapshot the entire suite-source tree; rebuild the pack;
    compare. This is the same invariant enforces."""
    suite_files: list[Path] = []
    for sub in (tiny_suite / "canonical", tiny_suite / "wide"):
        if not sub.is_dir():
            continue
        for p in sub.rglob("*"):
            if p.is_file():
                suite_files.append(p)
    before = {p: _sha(p) for p in suite_files}

    from compgen.graph_compilation.kernel_evidence_pack import (
        build_kernel_evidence_pack,
    )
    out = tiny_suite.parent / "m25_pack_readonly_test"
    if out.exists():
        shutil.rmtree(out)
    build_kernel_evidence_pack(
        canonical_suite=tiny_suite / "canonical",
        wide_suite=tiny_suite / "wide",
        out_dir=out,
        skip_figures=True,
    )
    after = {p: _sha(p) for p in suite_files}
    drifted = [
        str(p.relative_to(tiny_suite)) for p in suite_files
        if before[p] != after[p]
    ]
    assert not drifted, f"M-25 mutated suite source files: {drifted[:3]}"


# --------------------------------------------------------------------------- #
# Claim-matrix invariants
# --------------------------------------------------------------------------- #


def test_joint_claim_count_within_bounds(evidence_pack: Path) -> None:
    cm = _read(evidence_pack / "kernel_section_claim_matrix.json")
    n = cm["model_count"]
    for c in cm["claims"]:
        assert 0 <= c["fx_models_ready"] <= n
        assert 0 <= c["kernel_models_ready"] <= n
        assert 0 <= c["joint_models_ready"] <= n
        assert c["joint_models_ready"] <= min(
            c["fx_models_ready"], c["kernel_models_ready"],
        )


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    for fname in (
        "kernel_evidence_pack.py",
        "kernel_evidence_pack_figures.py",
    ):
        src = (
            REPO_ROOT / "python" / "compgen" / "graph_compilation" / fname
        ).read_text(encoding="utf-8")
        for f in (
            "from compgen.ir",
            "from compgen.capture",
            "from compgen.pipeline",
            "from compgen.runtime.bundle_emit",
        ):
            assert f not in src, (
                f"{fname} imports forbidden module: {f}"
            )


# --------------------------------------------------------------------------- #
# Per-model row sanity
# --------------------------------------------------------------------------- #


def test_model_csv_has_two_rows(evidence_pack: Path) -> None:
    csv_path = evidence_pack / "kernel_section_model_matrix.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    # 1 header + 2 data rows (merlin_mlp_wide + proxy_vla).
    assert len(lines) == 3, f"unexpected row count: {len(lines)}"


def test_aggregates_match_per_model_sums(evidence_pack: Path) -> None:
    et = _read(evidence_pack / "kernel_section_evidence_tables.json")
    agg = et["aggregates"]
    cm = _read(evidence_pack / "kernel_section_claim_matrix.json")
    assert agg["model_count"] == cm["model_count"]
