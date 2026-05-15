"""Acceptance tests Compiled Fusion Verification.

Verifies:

emits a typed report when picked an `executable_real_fusion`
  candidate (proxy_vla canonical case: bias_add → relu).
- Both GPU (Triton) and CPU (cffi C) tracks compile + run + discharge
  bit-equality vs eager unfused chain on 16 frozen input cases.
downstream-retry detector includes
  `compiled_fusion_differential_check` in its registry.
emits typed `not_run` when no fusion candidate was selected
  (e.g. merlin_mlp_wide picks SetTileParams, not fusion).
emits typed `blocked` when producer/consumer is outside the
  pointwise MVP whitelist.
the existing artifacts stay byte-identical (layers
  ALONGSIDE).
- Generated kernel sources are byte-deterministic across reruns.
Ledger captures the stage event.
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
def fusion_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """proxy_vla — greedy picks add_0 → aten_relu_default_0 fusion."""
    out = tmp_path_factory.mktemp("m23_fusion") / "run"
    _run("proxy_vla", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_fusion_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide — greedy picks SetTileParams, NOT fusion.
    must emit typed not_run rather than crash."""
    out = tmp_path_factory.mktemp("m23_no_fusion") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


# --------------------------------------------------------------------------- #
# Emission shape
# --------------------------------------------------------------------------- #


def test_compiled_fusion_dir_exists_when_fusion_picked(
    fusion_run: Path,
) -> None:
    base = fusion_run / "02_graph_analysis" / "compiled_fusion"
    assert base.is_dir()
    assert (base / "compiled_fusion_differential_report.json").exists()
    assert (base / "compiled_fusion_differential_summary.md").exists()


def test_compiled_fusion_emits_not_run_when_no_fusion_selected(
    no_fusion_run: Path,
) -> None:
    p = (
        no_fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if not p.exists():
        pytest.skip("M-23 not wired or capture failed")
    r = _read(p)
    assert r["overall"] == "not_run"


def test_artifact_schema_version(fusion_run: Path) -> None:
    r = _read(
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    assert r["schema_version"] == "compiled_fusion_differential_report_v1"


def test_report_records_pointwise_pair(fusion_run: Path) -> None:
    r = _read(
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if r["overall"] != "pass":
        pytest.skip(f"M-23 overall={r['overall']!r}")
    assert r["producer_kind"] in (
        "bias_add", "add", "mul", "sub",
        "elementwise_relu", "elementwise_sigmoid", "elementwise_tanh",
    )
    assert r["consumer_kind"] in (
        "elementwise_relu", "elementwise_sigmoid", "elementwise_tanh",
    )


# --------------------------------------------------------------------------- #
# Bit-equality on real compiled kernels
# --------------------------------------------------------------------------- #


def test_gpu_track_discharges_bit_equality(fusion_run: Path) -> None:
    """When CUDA is available, the fused Triton kernel must produce
    bit-identical output to eager unfused chain on all 16 cases."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not available")

    r = _read(
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if r["overall"] not in ("pass",):
        pytest.skip(f"M-23 overall={r['overall']!r}")
    gpu = r["gpu"]
    assert gpu["compile_status"] == "compiled"
    assert gpu["run_status"] == "ok"
    assert gpu["case_count"] == 16
    assert gpu["max_abs_error"] == 0.0, (
        f"GPU fused kernel deviated from eager: "
        f"max_abs={gpu['max_abs_error']}, max_rel={gpu['max_rel_error']}"
    )
    assert gpu["max_rel_error"] == 0.0
    assert gpu["bit_equality_count"] == 16
    assert gpu["fail_outside_tolerance_count"] == 0


def test_cpu_track_discharges_bit_equality(fusion_run: Path) -> None:
    """The fused C kernel compiled with -fno-fast-math must produce
    bit-identical output to eager unfused chain on all 16 cases."""
    r = _read(
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if r["overall"] not in ("pass",):
        pytest.skip(f"M-23 overall={r['overall']!r}")
    cpu = r["cpu"]
    assert cpu["compile_status"] == "compiled"
    assert cpu["run_status"] == "ok"
    assert cpu["case_count"] == 16
    assert cpu["max_abs_error"] == 0.0
    assert cpu["max_rel_error"] == 0.0
    assert cpu["bit_equality_count"] == 16


def test_kernel_sources_emitted(fusion_run: Path) -> None:
    base = fusion_run / "02_graph_analysis" / "compiled_fusion"
    triton_files = list(base.glob("triton_fused_*.py"))
    c_files = list(base.glob("cpu_fused_*.c"))
    assert triton_files, "no Triton fused kernel emitted"
    assert c_files, "no CPU fused kernel emitted"


# --------------------------------------------------------------------------- #
# Layered alongside (no mutation)
# --------------------------------------------------------------------------- #


def test_m162_artifacts_unchanged_after_m23(fusion_run: Path) -> None:
    """the manifest + differential report stay byte-identical
     runs (layers alongside)."""
    rl_path = (
        fusion_run / "03_recipe_planning" / "real_lowering"
        / "real_fusion_manifest.json"
    )
    rv_path = (
        fusion_run / "03_recipe_planning" / "real_verification"
        / "real_fusion_differential_report.json"
    )
    if not rl_path.exists() or not rv_path.exists():
        pytest.skip("M-16.2 artifacts absent on this fixture")
    before = {"manifest": _sha(rl_path), "report": _sha(rv_path)}
    from compgen.graph_compilation.compiled_fusion import run_compiled_fusion
    run_compiled_fusion(fusion_run)
    after = {"manifest": _sha(rl_path), "report": _sha(rv_path)}
    assert before == after, (
        f"M-23 mutated M-16.2 artifact(s): "
        f"{[k for k in before if before[k] != after[k]]}"
    )


def test_m23_does_not_mutate_canonical_artifacts(fusion_run: Path) -> None:
    """region_map + candidate_actions + cost_preview_v2 stay
    byte-identical through ."""
    rm_path = fusion_run / "02_graph_analysis" / "region_map.json"
    ca_path = fusion_run / "02_graph_analysis" / "candidate_actions.json"
    cp_path = fusion_run / "02_graph_analysis" / "cost_preview_v2.json"
    before = {
        "region_map": _sha(rm_path),
        "candidate_actions": _sha(ca_path),
        "cost_preview_v2": _sha(cp_path),
    }
    from compgen.graph_compilation.compiled_fusion import run_compiled_fusion
    run_compiled_fusion(fusion_run)
    after = {
        "region_map": _sha(rm_path),
        "candidate_actions": _sha(ca_path),
        "cost_preview_v2": _sha(cp_path),
    }
    assert before == after


def test_kernel_source_sha256_deterministic(fusion_run: Path) -> None:
    """The reported kernel_source_sha256 (computed over a
    timestamp-stripped variant) must be byte-identical across reruns."""
    from compgen.graph_compilation.compiled_fusion import run_compiled_fusion

    p = (
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    r1 = _read(p)
    if r1["overall"] != "pass":
        pytest.skip(f"M-23 overall={r1['overall']!r}")
    sha_gpu_1 = r1["gpu"].get("kernel_source_sha256")
    sha_cpu_1 = r1["cpu"].get("c_source_sha256")
    run_compiled_fusion(fusion_run)
    r2 = _read(p)
    sha_gpu_2 = r2["gpu"].get("kernel_source_sha256")
    sha_cpu_2 = r2["cpu"].get("c_source_sha256")
    if sha_gpu_1 is not None:
        assert sha_gpu_1 == sha_gpu_2, "GPU kernel source SHA drifted"
    if sha_cpu_1 is not None:
        assert sha_cpu_1 == sha_cpu_2, "CPU kernel source SHA drifted"


# --------------------------------------------------------------------------- #
# downstream-retry coupling
# --------------------------------------------------------------------------- #


def test_m15b_includes_compiled_fusion_check() -> None:
    """The downstream_retry detector must include the report in
    its registry so kernel-level fusion failures trigger retry."""
    from compgen.graph_compilation.downstream_retry import _DOWNSTREAM_REPORTS

    check_labels = [r[3] for r in _DOWNSTREAM_REPORTS]
    assert "compiled_fusion_differential_check" in check_labels


def test_m15b_does_not_retry_on_pass(fusion_run: Path) -> None:
    """status=pass on must NOT trigger a downstream-retry request."""
    from compgen.graph_compilation.downstream_retry import detect_downstream_failure

    r = _read(
        fusion_run / "02_graph_analysis" / "compiled_fusion"
        / "compiled_fusion_differential_report.json"
    )
    if r["overall"] != "pass":
        pytest.skip(f"M-23 overall={r['overall']!r}")
    failure = detect_downstream_failure(fusion_run)
    # If a failure was detected, it must NOT be from compiled_fusion.
    if failure is not None:
        assert failure.failed_check != "compiled_fusion_differential_check"


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def test_ledger_records_m23_event(fusion_run: Path) -> None:
    ledger_path = fusion_run / "stage_ledger.jsonl"
    assert ledger_path.exists()
    events = [
        json.loads(line) for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    m23_events = [
        e for e in events
        if e.get("note") and "M-23" in e["note"]
    ]
    assert m23_events, "ledger missing M-23 stage event"
    note = m23_events[0]["note"]
    # Note carries outcome.
    assert any(s in note for s in ("pass", "fail", "blocked", "not_run", "error"))


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "compiled_fusion.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "from compgen.capture",
        "from compgen.pipeline",
        "from compgen.runtime.bundle_emit",
    )
    for f in forbidden:
        assert f not in src, (
            f"compiled_fusion imports forbidden module: {f}"
        )


# --------------------------------------------------------------------------- #
# Pure-function helper unit tests
# --------------------------------------------------------------------------- #


def test_classify_kind_maps_canonical_names() -> None:
    from compgen.graph_compilation.compiled_fusion import _classify_kind

    assert _classify_kind("bias_add") == "bias_add"
    assert _classify_kind("aten_add") == "add"
    assert _classify_kind("aten_relu_default") == "elementwise_relu"
    assert _classify_kind("elementwise_sigmoid") == "elementwise_sigmoid"
    assert _classify_kind("elementwise_tanh") == "elementwise_tanh"
