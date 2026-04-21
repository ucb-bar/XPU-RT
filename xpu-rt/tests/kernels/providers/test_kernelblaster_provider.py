"""Tests for the KernelBlaster adapter + provider.

The adapter shells out to KB's ``run_single_kernelblaster.sh`` or to
``docker run``; both are unavailable in CI. These tests fake the
subprocess runner via the adapter's ``_run`` hook and build a fake KB
``out/`` tree in a tmp dir, so we exercise the real input staging /
command assembly / output parsing paths without touching the network or
a real GPU.

Coverage:
  * ``accepts_contract`` filters by CUDA-ness + required payloads
  * the staged workdir matches KB's expected layout
    (``data/<dataset>/<level>/<problem>/{init.cu,driver.cpp}``)
  * local and docker invocations produce the expected argv + env
  * successful output parsing surfaces kernel source, latency, speedup,
    knowledge exports, and contract feedback
  * missing artifacts + non-zero return codes map to ``found=False``
  * ``KernelBlasterUnavailable`` translates to ``ProviderResult(found=False)``
    with a ``reason`` in metadata
  * budget maps onto ``KB_MAX_ITERATIONS`` / ``KB_MAX_CANDIDATES`` env
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from compgen.kernels.kernelblaster_adapter import (
    DEFAULT_DATASET,
    KernelBlasterAdapter,
    KernelBlasterConfig,
    KernelBlasterUnavailable,
)
from compgen.kernels.provider import KernelContract, SearchBudget
from compgen.kernels.providers.kernelblaster import KernelBlasterProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeCompleted:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _CapturedInvocation:
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    staged: dict[str, str] = field(default_factory=dict)


def _make_runner(
    *,
    returncode: int = 0,
    stderr: str = "",
    db_payload: dict[str, Any] | None = None,
    kernel_body: str = "__global__ void optimized(...) {}",
    dataset: str = DEFAULT_DATASET,
    precision: str = "fp16",
    experiment: str = "compgen_run",
):
    """Build a fake subprocess runner that drops KB's ``out/`` tree.

    The captured invocation is returned out-of-band. We snapshot the
    staged input files (``data/<dataset>/<level>/<problem>/``) into
    ``captured.staged`` *during* the call so tests can assert on them
    after the adapter's ``TemporaryDirectory`` context has been torn down.
    """
    captured = _CapturedInvocation()

    def _run(
        argv: list[str],
        *,
        env: dict[str, str],
        cwd: str,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
        check: bool = False,
    ) -> _FakeCompleted:
        captured.argv = list(argv)
        captured.env = dict(env)
        captured.cwd = Path(cwd)

        data_root = Path(cwd) / "data"
        if data_root.exists():
            for path in data_root.rglob("*"):
                if path.is_file():
                    captured.staged[str(path.relative_to(Path(cwd)))] = path.read_text()

        out_dir = Path(cwd) / "out" / dataset / precision / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_rl_cuda_perf.cu").write_text(kernel_body)
        if db_payload is not None:
            (out_dir / "optimization_database.json").write_text(json.dumps(db_payload))
        return _FakeCompleted(returncode=returncode, stderr=stderr)

    return _run, captured


def _cuda_contract(**constraints_overrides: Any) -> KernelContract:
    kb = {
        "init_cu": "__global__ void init(float* x) { *x = 1.0f; }",
        "driver_cpp": "int main() { return 0; }",
        **constraints_overrides,
    }
    return KernelContract(
        region_id="matmul_0",
        op_family="matmul",
        input_shapes=((128, 128),),
        output_shapes=((128, 128),),
        dtypes=("fp16",),
        target_name="cuda",
        hardware_key="H100",
        constraints={"kernelblaster": kb},
    )


def _local_config(tmp_path: Path) -> KernelBlasterConfig:
    """Config pointing at a fake local KB checkout inside ``tmp_path``.

    We also drop a couple of sibling directories (``src/``, ``utils/``)
    so we can verify the overlay-symlink path picks them up.
    """
    kb_root = tmp_path / "kb_repo"
    (kb_root / "scripts").mkdir(parents=True)
    (kb_root / "scripts" / "run_single_kernelblaster.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (kb_root / "src").mkdir()
    (kb_root / "src" / "kernelblaster").mkdir()
    (kb_root / "src" / "kernelblaster" / "__init__.py").write_text("")
    (kb_root / "utils").mkdir()
    (kb_root / "utils" / "helper.py").write_text("# helper\n")
    return KernelBlasterConfig(
        mode="local",
        repo_root=kb_root,
        openai_api_key="test-key",
    )


def _docker_config() -> KernelBlasterConfig:
    return KernelBlasterConfig(
        mode="docker",
        image="kernelblaster:test",
        openai_api_key="test-key",
    )


# ---------------------------------------------------------------------------
# accepts_contract
# ---------------------------------------------------------------------------


def test_accepts_contract_requires_cuda_target():
    provider = KernelBlasterProvider()
    assert provider.accepts_contract(_cuda_contract())
    cpu_contract = KernelContract(
        target_name="cpu",
        constraints={"kernelblaster": {"init_cu": "x", "driver_cpp": "y"}},
    )
    assert not provider.accepts_contract(cpu_contract)


def test_accepts_contract_requires_kb_payload():
    provider = KernelBlasterProvider()
    bare = KernelContract(target_name="cuda", hardware_key="H100")
    partial = KernelContract(
        target_name="cuda", constraints={"kernelblaster": {"init_cu": "x"}}
    )
    assert not provider.accepts_contract(bare)
    assert not provider.accepts_contract(partial)


# ---------------------------------------------------------------------------
# adapter.is_available
# ---------------------------------------------------------------------------


def test_is_available_reports_missing_repo(tmp_path: Path):
    # Forcing mode="local" with a nonexistent repo_root must report a
    # specific reason rather than silently falling back to docker.
    cfg = KernelBlasterConfig(
        mode="local",
        repo_root=tmp_path / "does-not-exist",
        openai_api_key="k",
    )
    ok, reason = KernelBlasterAdapter(config=cfg).is_available()
    assert ok is False
    # Message can be about missing root or missing script — both are
    # acceptable; either one pinpoints the problem for the caller.
    assert "does not exist" in reason or "missing" in reason


def test_is_available_reports_missing_api_key(tmp_path: Path):
    cfg = _local_config(tmp_path)
    cfg.openai_api_key = ""
    ok, reason = KernelBlasterAdapter(config=cfg).is_available()
    assert ok is False
    assert "OPENAI_API_KEY" in reason


def test_is_available_reports_unconfigured(tmp_path: Path, monkeypatch):
    for env_var in (
        "COMPGEN_KERNELBLASTER_ROOT",
        "COMPGEN_KERNELBLASTER_IMAGE",
        "COMPGEN_KERNELBLASTER_MODE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.chdir(tmp_path)
    ok, reason = KernelBlasterAdapter(config=KernelBlasterConfig.from_env()).is_available()
    assert ok is False
    assert "no KernelBlaster" in reason or "not set" in reason


# ---------------------------------------------------------------------------
# search — local mode, happy path
# ---------------------------------------------------------------------------


def test_search_stages_inputs_and_parses_output(tmp_path: Path):
    cfg = _local_config(tmp_path)
    runner, captured = _make_runner(
        db_payload={
            "final_latency_us": 42.0,
            "baseline_latency_us": 100.0,
            "iterations": 7,
            "candidates_evaluated": 20,
            "plan": "register-tile matmul",
            "lessons": [
                {
                    "kind": "optimization_tactic",
                    "scope": "operator_family",
                    "scope_key": "matmul",
                    "content": "Use 128x128 tiles with async copy.",
                    "confidence": 0.8,
                }
            ],
            "contract_feedback": [
                {
                    "field": "layout",
                    "current_value": "row_major",
                    "suggested_value": "column_major",
                    "reason": "KB measured 1.9x via transpose",
                    "measured_gain": 0.9,
                }
            ],
            "final_correct": True,
        },
    )
    adapter = KernelBlasterAdapter(config=cfg, _run=runner)
    contract = _cuda_contract()

    result = adapter.search_kernel(contract, SearchBudget(max_iterations=5, max_candidates=40))

    # Output parsing
    assert result.found is True
    assert result.language == "cuda"
    assert result.latency_us == pytest.approx(42.0)
    assert result.speedup == pytest.approx(100.0 / 42.0)
    assert result.iterations_used == 7
    assert result.total_candidates == 20
    assert result.correct is True
    assert result.kernel_code.startswith("__global__ void optimized")
    assert result.plan == "register-tile matmul"
    assert len(result.knowledge_exports) == 1
    assert result.knowledge_exports[0].content.startswith("Use 128x128")
    assert len(result.contract_feedback) == 1
    assert result.contract_feedback[0].suggested_value == "column_major"

    # Input staging: init.cu + driver.cpp landed under data/<dataset>/level1/<NNN_name>/
    staged_rel = f"data/{DEFAULT_DATASET}/level1/001_compgen_custom"
    assert captured.staged[f"{staged_rel}/init.cu"].startswith("__global__")
    assert captured.staged[f"{staged_rel}/driver.cpp"] == "int main() { return 0; }"

    # argv invokes bash on the shell script inside the overlay workdir
    # so KB's ROOT_DIR resolves to our staging tree, not the user's
    # KB checkout.
    assert captured.argv[0] == "bash"
    assert captured.argv[1].endswith("scripts/run_single_kernelblaster.sh")
    assert str(captured.cwd) in captured.argv[1]
    problem_idx = captured.argv.index("--problem-numbers")
    assert captured.argv[problem_idx + 1] == "1"
    subset_idx = captured.argv.index("--subset")
    assert captured.argv[subset_idx + 1] == "level1"
    # Env carries KB's required vars + the budget-derived limits
    assert captured.env["OPENAI_API_KEY"] == "test-key"
    assert captured.env["KB_MAX_ITERATIONS"] == "5"
    assert captured.env["KB_MAX_CANDIDATES"] == "40"
    assert captured.env["DATASET"] == DEFAULT_DATASET
    assert captured.env["PRECISION"] == "fp16"


# ---------------------------------------------------------------------------
# search — overlay
# ---------------------------------------------------------------------------


def test_local_mode_overlays_kb_repo(tmp_path: Path):
    """Every top-level item in the KB repo (minus data/out/.git) should be
    reachable from the workdir via symlink, so KB's relative imports +
    ROOT_DIR computation work without writing into the user's checkout."""
    cfg = _local_config(tmp_path)

    captured_snapshot: dict[str, Any] = {"src_ok": False, "utils_ok": False, "script_ok": False}

    def _inspect_runner(argv, *, env, cwd, **_):
        cwd_path = Path(cwd)
        # Workdir should now have the overlayed top-level items.
        captured_snapshot["src_ok"] = (cwd_path / "src" / "kernelblaster" / "__init__.py").exists()
        captured_snapshot["utils_ok"] = (cwd_path / "utils" / "helper.py").exists()
        captured_snapshot["script_ok"] = (cwd_path / "scripts" / "run_single_kernelblaster.sh").exists()
        # Still drop the expected KB output so the parser has something to read.
        out_dir = cwd_path / "out" / DEFAULT_DATASET / "fp16" / cfg.experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_rl_cuda_perf.cu").write_text("__global__ void x(){}")
        (out_dir / "optimization_database.json").write_text(json.dumps({"final_latency_us": 1.0}))
        return _FakeCompleted(returncode=0)

    adapter = KernelBlasterAdapter(config=cfg, _run=_inspect_runner)
    result = adapter.search_kernel(_cuda_contract(), SearchBudget())

    assert result.found is True
    assert captured_snapshot == {"src_ok": True, "utils_ok": True, "script_ok": True}


# ---------------------------------------------------------------------------
# search — docker mode
# ---------------------------------------------------------------------------


def test_docker_mode_builds_docker_argv(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    cfg = _docker_config()
    runner, captured = _make_runner(db_payload={"final_latency_us": 10.0})
    adapter = KernelBlasterAdapter(config=cfg, _run=runner)

    result = adapter.search_kernel(_cuda_contract(), SearchBudget())

    assert result.found is True
    assert captured.argv[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in captured.argv and "all" in captured.argv
    assert "kernelblaster:test" in captured.argv
    assert "scripts/run_single_kernelblaster.sh" in captured.argv
    # docker mode also forwards env via -e switches
    joined = " ".join(captured.argv)
    assert "OPENAI_API_KEY=test-key" in joined


def test_docker_mode_requires_docker_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    cfg = _docker_config()
    adapter = KernelBlasterAdapter(config=cfg)
    ok, reason = adapter.is_available()
    assert ok is False
    assert "docker" in reason


# ---------------------------------------------------------------------------
# search — failure modes
# ---------------------------------------------------------------------------


def test_missing_kernel_artifact_is_not_found(tmp_path: Path):
    cfg = _local_config(tmp_path)

    def _silent_runner(argv, *, env, cwd, **_):
        # Don't write any artifacts.
        return _FakeCompleted(returncode=0)

    adapter = KernelBlasterAdapter(config=cfg, _run=_silent_runner)
    result = adapter.search_kernel(_cuda_contract(), SearchBudget())
    assert result.found is False
    assert "kernel artifact missing" in result.metadata.get("reason", "")


def test_nonzero_return_is_not_found(tmp_path: Path):
    cfg = _local_config(tmp_path)

    def _failing_runner(argv, *, env, cwd, **_):
        return _FakeCompleted(returncode=2, stderr="boom: out of memory")

    adapter = KernelBlasterAdapter(config=cfg, _run=_failing_runner)
    result = adapter.search_kernel(_cuda_contract(), SearchBudget())
    assert result.found is False
    assert result.metadata["returncode"] == 2
    assert "boom" in result.metadata["stderr_tail"]


def test_contract_missing_payload_raises_value_error(tmp_path: Path):
    cfg = _local_config(tmp_path)
    adapter = KernelBlasterAdapter(config=cfg, _run=_make_runner()[0])
    bad = KernelContract(
        target_name="cuda", constraints={"kernelblaster": {"init_cu": "x"}}
    )
    with pytest.raises(ValueError, match="init_cu"):
        adapter.search_kernel(bad, SearchBudget())


def test_provider_translates_unavailable_to_not_found():
    # No config -> is_available returns False -> provider catches KernelBlasterUnavailable
    provider = KernelBlasterProvider(config=KernelBlasterConfig())
    result = provider.search(_cuda_contract(), SearchBudget())
    assert result.found is False
    assert result.metadata["provider"] == "kernelblaster"
    assert "reason" in result.metadata


def test_provider_search_accumulates_knowledge(tmp_path: Path):
    cfg = _local_config(tmp_path)
    runner, _ = _make_runner(
        db_payload={
            "final_latency_us": 10.0,
            "baseline_latency_us": 20.0,
            "lessons": ["prefer async copy"],
        },
    )
    adapter = KernelBlasterAdapter(config=cfg, _run=runner)
    provider = KernelBlasterProvider(adapter=adapter)

    result = provider.search(_cuda_contract(), SearchBudget())
    assert result.found is True
    exports = provider.export_knowledge()
    assert len(exports) == 1
    # Second drain is empty.
    assert provider.export_knowledge() == []


# ---------------------------------------------------------------------------
# Timeouts + missing executables
# ---------------------------------------------------------------------------


def test_subprocess_timeout_becomes_unavailable(tmp_path: Path):
    cfg = _local_config(tmp_path)

    def _timeout_runner(argv, *, env, cwd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout") or 1)

    adapter = KernelBlasterAdapter(config=cfg, _run=_timeout_runner, max_runtime_seconds=1)
    with pytest.raises(KernelBlasterUnavailable, match="timed out"):
        adapter.search_kernel(_cuda_contract(), SearchBudget())


def test_missing_executable_becomes_unavailable(tmp_path: Path):
    cfg = _local_config(tmp_path)

    def _missing_runner(argv, *, env, cwd, **_):
        raise FileNotFoundError(argv[0])

    adapter = KernelBlasterAdapter(config=cfg, _run=_missing_runner)
    with pytest.raises(KernelBlasterUnavailable, match="cannot invoke"):
        adapter.search_kernel(_cuda_contract(), SearchBudget())
