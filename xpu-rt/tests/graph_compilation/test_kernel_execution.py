"""Acceptance tests Kernel Execution Foundation.

Verifies:

- Default OFF: no env var ⇒ no kernel_execution/ directory.
- With ``COMPGEN_RUN_KERNELS=1`` on merlin_mlp_wide (committed
  candidate is SetTileParams, executable_structured_ir):
  - Two artifacts emit: ``compiled_kernel_run_gpu.json`` and
    ``compiled_kernel_run_cpu.json``.
  - Each artifact has ``schema_version=compiled_kernel_run_v1``,
    matmul_shape, tile, region_id, candidate_id.
  - When CUDA + Triton are available: GPU track's ``compile_status ==
    "compiled"`` and ``numerical.refinement_status`` is
    ``discharged_compiled_bit_equality`` OR ``discharged_tolerance_eps``.
  - When gcc + cffi are available: CPU track's ``compile_status ==
    "compiled"`` and ``numerical.refinement_status`` is in the same
    discharged set.
- For non-SetTileParams runs (e.g. proxy_vla → fuse_producer_consumer
  selected by greedy): kernel_execution emits artifacts marked
  ``not_applicable``.
- Kernel source SHA256 is byte-deterministic across reruns (the
  deterministic body, not the body with the UTC timestamp in the
  docstring).
Existing FX-level reports are byte-identical
  before vs after kernel_execution runs.
- No compiler-core imports.
- Best-effort: missing manifest → ``not_applicable`` cleanly.
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


def _run(model: str, out_dir: Path, *, run_kernels: bool) -> int:
    env = os.environ.copy()
    if run_kernels:
        env["COMPGEN_RUN_KERNELS"] = "1"
    else:
        env.pop("COMPGEN_RUN_KERNELS", None)
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)
    res = subprocess.run(
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
    return res.returncode


@pytest.fixture(scope="module")
def kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide → SetTileParams executable; fires."""
    out = tmp_path_factory.mktemp("m19_run") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """Default OFF — kernel_execution should not exist."""
    out = tmp_path_factory.mktemp("m19_off") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


@pytest.fixture(scope="module")
def fusion_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """proxy_vla → FuseProducerConsumer; emits not_applicable."""
    out = tmp_path_factory.mktemp("m19_fusion") / "run"
    _run("proxy_vla", out, run_kernels=True)
    return out


# --------------------------------------------------------------------------- #
# Default OFF
# --------------------------------------------------------------------------- #


def test_default_off_no_kernel_execution_dir(no_kernels_run: Path) -> None:
    """Without COMPGEN_RUN_KERNELS, the directory must not exist."""
    base = no_kernels_run / "02_graph_analysis" / "kernel_execution"
    assert not base.exists()


# --------------------------------------------------------------------------- #
# Artifacts emit
# --------------------------------------------------------------------------- #


def test_kernel_execution_dir_exists_when_env_set(kernels_run: Path) -> None:
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    assert base.is_dir()
    assert (base / "compiled_kernel_run_gpu.json").exists()
    assert (base / "compiled_kernel_run_cpu.json").exists()
    assert (base / "kernel_execution_summary.md").exists()


def test_artifact_schema_version_v1(kernels_run: Path) -> None:
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    for name in ("compiled_kernel_run_gpu.json", "compiled_kernel_run_cpu.json"):
        d = _read(base / name)
        assert d["schema_version"] == "compiled_kernel_run_v1"
        assert "matmul_shape" in d
        assert "tile" in d
        assert "region_id" in d
        assert "candidate_id" in d
        assert "iterations" in d
        assert "warmup" in d


def test_gpu_track_compiled_and_run_when_cuda_available(
    kernels_run: Path,
) -> None:
    import torch

    g = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "compiled_kernel_run_gpu.json"
    )
    if not torch.cuda.is_available():
        assert g["compile_status"] == "device_unavailable"
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        assert g["compile_status"] == "triton_unavailable"
        return
    assert g["compile_status"] == "compiled"
    assert g["run_status"] == "ok"
    assert g["measured_us_per_iter"] > 0
    assert g["numerical"]["refinement_status"] in (
        "discharged_compiled_bit_equality",
        "discharged_tolerance_eps",
    )


def test_cpu_track_compiled_and_run_when_compiler_available(
    kernels_run: Path,
) -> None:
    import shutil

    c = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "compiled_kernel_run_cpu.json"
    )
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        assert c["compile_status"] == "compiler_unavailable"
        return
    try:
        import cffi  # noqa: F401
    except ImportError:
        assert c["compile_status"] == "cffi_unavailable"
        return
    assert c["compile_status"] == "compiled"
    assert c["run_status"] == "ok"
    assert c["measured_us_per_iter"] > 0
    assert c["numerical"]["refinement_status"] in (
        "discharged_compiled_bit_equality",
        "discharged_tolerance_eps",
    )


def test_kernel_source_files_emitted(kernels_run: Path) -> None:
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    triton_files = list(base.glob("triton_kernel_*.py"))
    cpu_files = list(base.glob("cpu_kernel_*.c"))
    assert len(triton_files) >= 1
    assert len(cpu_files) >= 1
    # Triton source contains the @triton.jit decorator.
    assert "@triton.jit" in triton_files[0].read_text(encoding="utf-8")
    # CPU C source declares the matmul function.
    assert "void compgen_m19_matmul" in cpu_files[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_kernel_source_sha256_recorded(kernels_run: Path) -> None:
    """Both artifacts record kernel_source_sha256 / c_source_sha256."""
    g = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "compiled_kernel_run_gpu.json"
    )
    if g["compile_status"] in ("compiled",):
        assert g["kernel_source_sha256"].startswith("sha256:")

    c = _read(
        kernels_run / "02_graph_analysis" / "kernel_execution"
        / "compiled_kernel_run_cpu.json"
    )
    if c["compile_status"] in ("compiled",):
        assert c["c_source_sha256"].startswith("sha256:")


def test_emit_kernel_source_deterministic_is_byte_identical() -> None:
    """The deterministic-body emitter (used for SHA pinning) must return
    byte-identical output for the same inputs across calls."""
    from compgen.graph_compilation.kernel_execution_gpu import (
        _emit_kernel_source_deterministic,
    )

    args = dict(
        candidate_id="cand_x",
        region_id="matmul_0",
        M=16, N=32, K=16, tM=16, tN=16, tK=16,
    )
    a = _emit_kernel_source_deterministic(**args)
    b = _emit_kernel_source_deterministic(**args)
    assert a == b


# --------------------------------------------------------------------------- #
# Not-applicable path (non-SetTileParams)
# --------------------------------------------------------------------------- #


def test_fusion_run_emits_not_applicable(fusion_run: Path) -> None:
    """proxy_vla greedy picks fusion; must mark itself
    not_applicable rather than try to run a tile kernel."""
    base = fusion_run / "02_graph_analysis" / "kernel_execution"
    assert base.is_dir()
    summary = (base / "kernel_execution_summary.md").read_text(encoding="utf-8")
    assert "not_applicable" in summary
    # No GPU/CPU artifact files when not applicable.
    assert not (base / "compiled_kernel_run_gpu.json").exists()
    assert not (base / "compiled_kernel_run_cpu.json").exists()


# --------------------------------------------------------------------------- #
# FX-level artifacts unchanged (regression invariant)
# --------------------------------------------------------------------------- #


def test_fx_level_reports_unchanged_when_m19_reruns(
    no_kernels_run: Path,
) -> None:
    """must not mutate any FX-level report when invoked. Snapshot
    the protected files on a no-kernels run, then call run_kernel_execution
    in-place and re-snapshot — the protected reports must be byte-identical.
    (Comparing two separate full pipeline runs would fail on natural UTC
    timestamp drift; this test isolates the mutation surface.)
    """
    protected = [
        "03_recipe_planning/real_verification/real_differential_report.json",
        "02_graph_analysis/cost_preview_v2.json",
        "02_graph_analysis/region_map.json",
        "02_graph_analysis/candidate_actions.json",
    ]

    def _sha(p: Path) -> str:
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    before = {
        rel: _sha(no_kernels_run / rel)
        for rel in protected
        if (no_kernels_run / rel).exists()
    }
    assert len(before) >= 1, "no protected files to check"

    from compgen.graph_compilation.kernel_execution import run_kernel_execution
    run_kernel_execution(no_kernels_run)

    after = {
        rel: _sha(no_kernels_run / rel)
        for rel in protected
        if (no_kernels_run / rel).exists()
    }
    assert before == after, (
        "M-19 mutated FX-level reports: "
        f"{[k for k in before if before[k] != after.get(k)]}"
    )


# --------------------------------------------------------------------------- #
# Best-effort handling
# --------------------------------------------------------------------------- #


def test_handles_missing_manifest(tmp_path: Path) -> None:
    """If real_transform_manifest.json is absent, emits a typed
    not_applicable summary instead of raising."""
    fake = tmp_path / "fake_run"
    (fake / "02_graph_analysis").mkdir(parents=True)
    (fake / "03_recipe_planning" / "real_lowering").mkdir(parents=True)
    # No manifest file written.

    from compgen.graph_compilation.kernel_execution import run_kernel_execution

    res = run_kernel_execution(fake)
    assert res.overall == "not_run"
    assert res.gpu_status == "not_applicable"
    assert res.cpu_status == "not_applicable"
    assert res.summary_md_path.exists()


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports() -> None:
    forbidden = (
        "from compgen.ir.payload",
        "import compgen.ir.payload",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for src_name in (
        "kernel_execution.py",
        "kernel_execution_gpu.py",
        "kernel_execution_cpu.py",
    ):
        src = (
            REPO_ROOT / "python" / "compgen" / "graph_compilation" / src_name
        ).read_text(encoding="utf-8")
        for pat in forbidden:
            assert pat not in src, f"{src_name} imports forbidden: {pat}"


# --------------------------------------------------------------------------- #
# Common metadata equivalence between GPU and CPU artifacts
# --------------------------------------------------------------------------- #


def test_gpu_and_cpu_artifacts_share_metadata(kernels_run: Path) -> None:
    """Both artifacts must report the same matmul_shape, tile,
    region_id, candidate_id (they are tracks of the same compilation)."""
    base = kernels_run / "02_graph_analysis" / "kernel_execution"
    g = _read(base / "compiled_kernel_run_gpu.json")
    c = _read(base / "compiled_kernel_run_cpu.json")
    for k in ("matmul_shape", "tile", "region_id",
              "candidate_id", "recipe_op_id", "model_id"):
        assert g[k] == c[k], f"GPU and CPU disagree on {k}"
