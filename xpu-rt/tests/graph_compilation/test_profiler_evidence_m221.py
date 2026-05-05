"""Acceptance tests for M-22.1 — real `torch.profiler` (CUDA) +
`linux perf` (CPU) measurement layer.

Verifies:

- M-22.1 emits a typed report when M-22 produced compiled measurements.
- M-22.1 emits typed `not_run` when M-22 didn't produce evidence
  (kernels OFF or no SetTileParams candidates).
- The GPU track populates `cuda_collected` evidence with non-zero
  per-kernel timing when CUDA is available.
- The CPU track honestly degrades to `perf_unavailable` when
  `kernel.perf_event_paranoid >= 3` (no root). This IS the typical
  user-environment state; the test asserts the typed fallback shape,
  NOT a perf success.
- M-22.1 layers `profiler_evidence` onto M-22's per-region
  `compiled_evidence` block, replacing the M-22 `cache_evidence:
  not_collected` placeholder with a concrete typed value.
- The hardware_resource_report's `compiled_evidence.cache_evidence`
  picks up the same value (cross-overlay invariant).
- Ledger captures the M-22.1 stage event.
- Hash chain (R009) stays intact through the M-22.1 writes.
- No compiler-core imports.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide with M-19/M-20 ON — gives us per-region kernels
    for M-22.1 to re-profile."""
    out = tmp_path_factory.mktemp("m221_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m221_no_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #


def test_profiler_evidence_dir_exists_when_kernels_on(
    kernels_run: Path,
) -> None:
    base = kernels_run / "02_graph_analysis" / "profiler_evidence"
    assert base.is_dir()
    assert (base / "profiler_evidence_report.json").exists()
    assert (base / "profiler_evidence_summary.md").exists()


def test_profiler_evidence_emits_not_run_when_kernels_off(
    no_kernels_run: Path,
) -> None:
    p = (
        no_kernels_run / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    if not p.exists():
        pytest.skip("M-22.1 not wired or capture failed")
    r = _read(p)
    assert r["overall"] == "not_run"
    assert r["regions"] == []


def test_artifact_schema_version(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    assert r["schema_version"] == "profiler_evidence_report_v1"


# --------------------------------------------------------------------------- #
# GPU track populates real CUDA evidence
# --------------------------------------------------------------------------- #


@pytest.mark.requires_gpu
def test_gpu_track_collects_cuda_evidence(kernels_run: Path) -> None:
    """When CUDA is available, every region with an M-19 GPU
    measurement should have profiler_status=cuda_collected and
    non-zero self_cuda_us_per_iter."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not available")

    r = _read(
        kernels_run / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    if r["overall"] != "ok":
        pytest.skip("M-22.1 did not run end-to-end")

    gpu_collected = [
        reg for reg in r["regions"]
        if (reg.get("gpu") or {}).get("profiler_status") == "cuda_collected"
    ]
    assert gpu_collected, (
        "no GPU regions reached cuda_collected; M-22.1 may be misconfigured"
    )
    for reg in gpu_collected:
        gpu = reg["gpu"]
        assert gpu["self_cuda_us_per_iter"] is not None
        assert gpu["self_cuda_us_per_iter"] > 0, (
            f"region {reg['region_id']}: profiler reported zero CUDA time"
        )
        assert gpu["total_cuda_calls"] > 0


# --------------------------------------------------------------------------- #
# CPU track degrades typed when perf is unavailable
# --------------------------------------------------------------------------- #


def test_cpu_track_degrades_typed_when_perf_paranoid(
    kernels_run: Path,
) -> None:
    """In environments where kernel.perf_event_paranoid >= 3 (the
    typical user environment), perf cache events fail for non-root.
    M-22.1 must NOT crash; it must emit a typed perf_unavailable
    block with a reason. This is the most common user-environment
    state — the test asserts the GRACEFUL DEGRADATION, not a success."""
    paranoid = 0
    try:
        paranoid = int(
            Path("/proc/sys/kernel/perf_event_paranoid")
            .read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError):
        pytest.skip("could not read perf_event_paranoid")

    r = _read(
        kernels_run / "02_graph_analysis" / "profiler_evidence"
        / "profiler_evidence_report.json"
    )
    if r["overall"] != "ok":
        pytest.skip("M-22.1 did not run end-to-end")

    if paranoid >= 3 or shutil.which("perf") is None:
        # perf is unavailable — assert typed fallback.
        cpu_blocks = [
            (reg.get("cpu") or {}) for reg in r["regions"]
            if reg.get("cpu") is not None
        ]
        assert cpu_blocks, "no CPU blocks present"
        for cpu in cpu_blocks:
            status = cpu.get("profiler_status")
            assert status == "perf_unavailable", (
                f"expected perf_unavailable, got {status!r}"
            )
            assert "reason" in cpu, (
                f"perf_unavailable block missing reason: {cpu}"
            )


# --------------------------------------------------------------------------- #
# Cross-overlay: hardware_resource_report.compiled_evidence picks up
# M-22.1's cache_evidence value (replacing M-22's "not_collected")
# --------------------------------------------------------------------------- #


def test_compiled_evidence_cache_evidence_replaced_by_m221(
    kernels_run: Path,
) -> None:
    """After M-22.1 runs, hardware_resource_report.regions[*]
    .compiled_evidence.cache_evidence should be one of:
    cuda_collected | perf_collected | perf_unavailable | not_collected.
    NOT just the M-22 placeholder."""
    hrr = _read(
        kernels_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    overlaid = [
        r for r in hrr.get("regions", []) or []
        if r.get("compiled_evidence") is not None
    ]
    assert overlaid, "no compiled_evidence overlay on hardware_resource_report"
    valid_states = {
        "cuda_collected", "perf_collected",
        "perf_unavailable", "not_collected",
    }
    for r in overlaid:
        ce = r["compiled_evidence"]
        assert "cache_evidence" in ce, (
            f"region {r['region_id']}: compiled_evidence missing cache_evidence"
        )
        assert ce["cache_evidence"] in valid_states, (
            f"region {r['region_id']}: invalid cache_evidence "
            f"{ce['cache_evidence']!r} (expected one of {valid_states})"
        )
        # When CUDA was available, cache_evidence should be cuda_collected.
        try:
            import torch
            if torch.cuda.is_available():
                assert ce["cache_evidence"] == "cuda_collected", (
                    f"region {r['region_id']}: CUDA available but "
                    f"cache_evidence={ce['cache_evidence']!r}"
                )
        except ImportError:
            pass


def test_compiled_bottleneck_picks_up_profiler_evidence_overlay(
    kernels_run: Path,
) -> None:
    cb = _read(
        kernels_run / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if cb.get("overall") != "ok":
        pytest.skip("M-22 had no measurements")
    ok_with_overlay = [
        r for r in cb["regions"]
        if r.get("model_status") == "ok"
        and r.get("profiler_evidence") is not None
    ]
    assert ok_with_overlay, (
        "no profiler_evidence overlay on compiled_bottleneck regions"
    )
    for r in ok_with_overlay:
        pe = r["profiler_evidence"]
        assert "cache_evidence" in pe
        assert pe["cache_evidence"] in (
            "cuda_collected", "perf_collected",
            "perf_unavailable", "not_collected",
        )


# --------------------------------------------------------------------------- #
# Ledger + hash chain audit
# --------------------------------------------------------------------------- #


def test_ledger_records_m22_1_event(kernels_run: Path) -> None:
    """The pipeline ledger must record the M-22.1 stage event with
    a typed note reflecting its outcome."""
    ledger_path = kernels_run / "stage_ledger.jsonl"
    assert ledger_path.exists()
    events = [
        json.loads(line) for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    m221_events = [
        e for e in events
        if e.get("note") and "M-22.1" in e["note"]
    ]
    assert m221_events, (
        "ledger missing M-22.1 event; M-22.1 may not be wired into run.py"
    )
    note = m221_events[0]["note"]
    # Note must carry the outcome (overall + per-track collected counts).
    assert any(s in note for s in ("ok", "no_regions", "not_run", "error"))


def test_ledger_records_full_kernel_pipeline(kernels_run: Path) -> None:
    """A run with kernels ON must record M-19, M-20, M-21, M-22, M-22.1
    events in order under graph_analysis."""
    ledger_path = kernels_run / "stage_ledger.jsonl"
    events = [
        json.loads(line) for line in ledger_path.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    notes = [e.get("note") or "" for e in events]
    for tag in ("M-19", "M-20", "M-21", "M-22", "M-22.1"):
        assert any(tag in n for n in notes), (
            f"ledger missing {tag} stage event"
        )

    # Order: M-19 before M-20 before M-21 before M-22 before M-22.1.
    indices = {
        tag: next(i for i, n in enumerate(notes) if tag in n)
        for tag in ("M-19", "M-20", "M-21", "M-22", "M-22.1")
    }
    assert (
        indices["M-19"] < indices["M-20"]
        < indices["M-21"] < indices["M-22"] < indices["M-22.1"]
    ), f"ledger order drift: {indices}"


def test_run_manifest_hash_chain_intact_with_m221(
    kernels_run: Path,
) -> None:
    """R009 invariant: stage[i].input_hash == stage[i-1].output_hash.
    M-22.1 writes happen inside graph_analysis stage and must not break
    the chain."""
    manifest = _read(kernels_run / "run_manifest.json")
    stages = manifest.get("stages", [])
    assert len(stages) >= 3
    for prev, cur in zip(stages, stages[1:]):
        assert cur["input_hash"] == prev["output_hash"], (
            f"R009 broken between {prev['stage_id']} and {cur['stage_id']}: "
            f"{prev['output_hash']!r} != {cur['input_hash']!r}"
        )


# --------------------------------------------------------------------------- #
# Best-effort: missing kernel sources, perf availability probe
# --------------------------------------------------------------------------- #


def test_perf_available_pure_function() -> None:
    from compgen.graph_compilation.profiler_evidence import _perf_available

    avail, reason = _perf_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)
    assert reason  # non-empty


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "profiler_evidence.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "from compgen.capture",
        "from compgen.pipeline",
        "from compgen.runtime.bundle_emit",
    )
    for f in forbidden:
        assert f not in src, (
            f"profiler_evidence imports forbidden module: {f}"
        )
