"""M-53 --resume-from kernel-codegen-response tests.

Coverage:
- Resume preserves out_dir (cert + .so + attempts trail survive across the
  pipeline-restart boundary).
- Resume errors honestly when the prerequisite artefacts are missing
  (no requests dir, no contracts dir).
- Resume + commit + glue-emit closes M-46 → M-47 with a real cert.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "compgen.graph_compilation", *args],
        cwd=cwd, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def merlin_kernel_codegen_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """A merlin_mlp_wide run driven through --stop-after kernel-codegen-request.

    Module-scoped: every M-53 test reuses the same baseline so we don't
    repeat the (slow) early-stage work for each case.
    """
    out = tmp_path_factory.mktemp("m53_baseline") / "run"
    res = _run_cli([
        "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "kernel-codegen-request",
        "--selection-mode", "greedy",
    ])
    assert res.returncode == 0, res.stderr
    return out


def _commit_real_cffi_response(run_dir: Path) -> dict:
    """Write a real cffi-C matmul artifact + commit the response.
    Returns dict with task_id, contract_hash, M, K, N."""
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from cffi import FFI
    from compgen.graph_compilation.kernel_codegen_response import (
        commit_response,
    )

    req = json.loads(
        sorted((run_dir / "04_kernel_codegen" / "requests").glob("*.json"))[0]
        .read_text()
    )
    contract = json.loads(
        sorted((run_dir / "04_kernel_codegen" / "contracts").glob("*.json"))[0]
        .read_text()
    )
    inp = [tuple(i.get("shape", {}).get("dims") or []) for i in contract["io"]["inputs"]]
    M, K = inp[0]
    K2, N = inp[1]

    artifact_dir = run_dir / "04_kernel_codegen" / "artifacts" / req["task_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sym = f"matmul_{M}x{K}_x_{K}x{N}"
    c_source = (
        "#include <string.h>\n"
        f"void {sym}(const float *A, const float *B, float *C) {{\n"
        f"    memset(C, 0, sizeof(float) * {M} * {N});\n"
        f"    for (int i = 0; i < {M}; ++i)\n"
        f"        for (int k = 0; k < {K}; ++k)\n"
        f"            for (int j = 0; j < {N}; ++j)\n"
        f"                C[i*{N} + j] += A[i*{K} + k] * B[k*{N} + j];\n"
        "}\n"
    )
    (artifact_dir / "kernel.c").write_text(c_source)

    ffi = FFI()
    ffi.cdef(f"void {sym}(const float *A, const float *B, float *C);")
    ffi.set_source(
        "_m53_test_kernel", c_source,
        extra_compile_args=["-O2", "-fno-fast-math"],
    )
    ffi.compile(tmpdir=str(artifact_dir))

    (artifact_dir / "kernel_metadata.json").write_text(json.dumps({
        "symbol": sym, "args": ["A", "B", "C"],
        "inputs": [
            {"dims": [M, K], "dtype": "f32", "layout": "row_major"},
            {"dims": [K, N], "dtype": "f32", "layout": "row_major"},
        ],
        "outputs": [{"dims": [M, N], "dtype": "f32", "layout": "row_major"}],
        "accumulator_dtype": "f32", "deterministic": True,
    }))
    (artifact_dir / "launch_config.json").write_text(json.dumps({
        "grid": [1, 1, 1], "block": [1, 1, 1], "smem_bytes": 0,
    }))
    (artifact_dir / "provider_claims.json").write_text(json.dumps({
        "backend": "c_reference", "supports_dispatch": ["sync"],
        "estimated_registers": 0, "estimated_smem_bytes": 0,
        "expected_numerics": "bit_equality",
    }))

    response = {
        "schema_version": "kernel_codegen_response_v1",
        "task_id": req["task_id"],
        "provider": {
            "kind": "m53_test_provider", "model": "real-cffi-c-by-hand",
            "started_at": "2026-05-07T00:00:00Z",
            "finished_at": "2026-05-07T00:00:01Z",
        },
        "contract_hash": req["contract_hash"],
        "artifacts": {
            "kernel_source": str((artifact_dir / "kernel.c").relative_to(run_dir)),
            "kernel_metadata": str((artifact_dir / "kernel_metadata.json").relative_to(run_dir)),
            "launch_config": str((artifact_dir / "launch_config.json").relative_to(run_dir)),
            "provider_claims": str((artifact_dir / "provider_claims.json").relative_to(run_dir)),
        },
        "claims": {
            "backend": "c_reference", "supports_dispatch": ["sync"],
            "estimated_registers": 0, "estimated_smem_bytes": 0,
            "expected_numerics": "bit_equality",
        },
        "contract_feedback": [], "notes": "M-53 test",
    }
    result = commit_response(
        run_dir=run_dir, task_id=req["task_id"], response=response,
    )
    assert result.accepted, f"commit_response rejected: {result.failure_summary}"
    return {
        "task_id": req["task_id"],
        "contract_hash": req["contract_hash"],
        "M": M, "K": K, "N": N, "sym": sym,
    }


# --------------------------------------------------------------------------- #
# Resume-mode preservation
# --------------------------------------------------------------------------- #


class TestResumePreservesArtifacts:
    def test_resume_does_not_wipe_committed_response(
        self, merlin_kernel_codegen_run: Path, tmp_path: Path,
    ) -> None:
        """The cardinal M-53 invariant: the committed response, attempts
        trail, certificate, and provider's compiled .so MUST survive a
        --resume-from run. Without this, the agentic provider chain
        cannot run from CLI."""
        run = tmp_path / "run"
        shutil.copytree(merlin_kernel_codegen_run, run)
        meta = _commit_real_cffi_response(run)

        artifact_dir = run / "04_kernel_codegen" / "artifacts" / meta["task_id"]
        cert_dir = run / "04_kernel_codegen" / "certificates"
        attempts_dir = run / "04_kernel_codegen" / "attempts" / meta["task_id"]

        # Snapshot pre-resume.
        pre_certs = sorted(p.name for p in cert_dir.glob("*.json"))
        pre_so = sorted(p.name for p in artifact_dir.glob("_m53_test_kernel*.so"))
        pre_attempts = sorted(p.name for p in attempts_dir.iterdir())
        assert pre_certs, "certificate not emitted (M-44/M-45 regression)"
        assert pre_so, "cffi-compiled .so missing pre-resume"
        assert pre_attempts, "attempts trail empty pre-resume"

        # Resume the pipeline via CLI.
        res = _run_cli([
            "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(run),
            "--stop-after", "glue-emit",
            "--resume-from", "kernel-codegen-response",
            "--selection-mode", "greedy",
        ])
        assert res.returncode == 0, res.stderr

        # Snapshot post-resume.
        post_certs = sorted(p.name for p in cert_dir.glob("*.json"))
        post_so = sorted(p.name for p in artifact_dir.glob("_m53_test_kernel*.so"))
        post_attempts = sorted(p.name for p in attempts_dir.iterdir())
        assert post_certs == pre_certs, "certificate was wiped by --resume-from"
        assert post_so == pre_so, ".so was wiped by --resume-from"
        assert post_attempts == pre_attempts, "attempts trail wiped"

    def test_resume_drives_m46_to_glue_emit(
        self, merlin_kernel_codegen_run: Path, tmp_path: Path,
    ) -> None:
        """After resume, the M-46 binding must surface the cert and
        the M-47 emitted executor must list the bound region."""
        run = tmp_path / "run"
        shutil.copytree(merlin_kernel_codegen_run, run)
        _commit_real_cffi_response(run)

        res = _run_cli([
            "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(run),
            "--stop-after", "glue-emit",
            "--resume-from", "kernel-codegen-response",
            "--selection-mode", "greedy",
        ])
        assert res.returncode == 0, res.stderr

        bindings_path = run / "05_execution_plan" / "region_kernel_bindings.json"
        bindings = json.loads(bindings_path.read_text())
        assert bindings["bound_count"] >= 1, (
            f"M-46 binding did not pick up the cert; bindings: {bindings}"
        )

        manifest = json.loads(
            (run / "06_glue_emit" / "plan_executor_manifest.json").read_text()
        )
        assert "matmul_0" in manifest["bound_regions"]


# --------------------------------------------------------------------------- #
# Honest error paths
# --------------------------------------------------------------------------- #


class TestResumeHonestErrors:
    def test_resume_fails_when_requests_dir_missing(self, tmp_path: Path) -> None:
        """Resume against a non-existent run dir errors honestly,
        not silently."""
        run = tmp_path / "noexist"
        # Create the dir but leave it empty — no committed response.
        run.mkdir()
        res = _run_cli([
            "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(run),
            "--stop-after", "glue-emit",
            "--resume-from", "kernel-codegen-response",
            "--selection-mode", "greedy",
        ])
        assert res.returncode != 0
        assert "no committed requests" in (res.stderr + res.stdout)

    def test_resume_fails_when_contracts_dir_missing(self, tmp_path: Path) -> None:
        """Requests present but contracts missing — error honestly."""
        run = tmp_path / "halfbaked"
        (run / "04_kernel_codegen" / "requests").mkdir(parents=True)
        (run / "04_kernel_codegen" / "requests" / "kcodegen_x.request.json").write_text(
            json.dumps({"task_id": "kcodegen_x"})
        )
        res = _run_cli([
            "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(run),
            "--stop-after", "glue-emit",
            "--resume-from", "kernel-codegen-response",
            "--selection-mode", "greedy",
        ])
        assert res.returncode != 0
        assert "no materialized contracts" in (res.stderr + res.stdout) \
            or "did not reach M-40" in (res.stderr + res.stdout)


# --------------------------------------------------------------------------- #
# End-to-end real execution through CLI alone
# --------------------------------------------------------------------------- #


class TestResumeEndToEnd:
    def test_real_chain_via_cli_only(
        self, merlin_kernel_codegen_run: Path, tmp_path: Path,
    ) -> None:
        """The acceptance criterion: drive M-40 → M-47 with a real
        cffi-compiled C matmul, run the emitted executor, and verify
        the output matches torch.matmul within Higham bound. The full
        chain works through the CLI alone (no operator drops into
        Python, except to commit the response — which is the M-43 API
        the operator owns).
        """
        sys.path.insert(0, str(REPO_ROOT / "python"))
        import importlib.util
        import numpy as np
        import torch
        from cffi import FFI
        from compgen.runtime.glue import CpuRuntimeAdapter

        run = tmp_path / "run"
        shutil.copytree(merlin_kernel_codegen_run, run)
        meta = _commit_real_cffi_response(run)

        res = _run_cli([
            "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(run),
            "--stop-after", "glue-emit",
            "--resume-from", "kernel-codegen-response",
            "--selection-mode", "greedy",
        ])
        assert res.returncode == 0, res.stderr

        # Load the .so back.
        artifact_dir = run / "04_kernel_codegen" / "artifacts" / meta["task_id"]
        so_files = sorted(artifact_dir.glob("_m53_test_kernel*.so"))
        spec_k = importlib.util.spec_from_file_location("_m53_test_kernel", so_files[0])
        kmod = importlib.util.module_from_spec(spec_k)
        spec_k.loader.exec_module(kmod)
        lib = kmod.lib
        sym = meta["sym"]; M, K, N = meta["M"], meta["K"], meta["N"]

        ffi = FFI()

        def _kernel_real(*args, **kwargs):
            tensors = [a if hasattr(a, "cpu") else torch.as_tensor(a) for a in args]
            A = tensors[0].contiguous().to(torch.float32).numpy()
            B = tensors[1].contiguous().to(torch.float32).numpy()
            C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
            getattr(lib, sym)(
                ffi.cast("const float *", A.ctypes.data),
                ffi.cast("const float *", B.ctypes.data),
                ffi.cast("float *", C.ctypes.data),
            )
            return torch.from_numpy(C)

        # Import the emitted executor and run.
        exec_path = run / "06_glue_emit" / "generated_plan_executor.py"
        spec = importlib.util.spec_from_file_location("emitted_exec_m53", exec_path)
        emod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emod)

        bound = list(emod.KERNEL_BINDINGS.keys())
        assert bound, "M-46 binding empty after resume — wiring still broken"
        kernels = {r: _kernel_real for r in bound}
        torch.manual_seed(0)
        io = {"A": torch.randn(M, K), "B": torch.randn(K, N)}
        out = emod.compgen_run(io, kernels, runtime=CpuRuntimeAdapter())
        eager = io["A"] @ io["B"]
        max_abs = float((out - eager).abs().max())

        # Higham bound for f32 matmul: 4 * K * eps * max|A| * max|B|.
        EPS = 1.19e-7
        higham = (
            4 * K * EPS
            * float(io["A"].abs().max()) * float(io["B"].abs().max())
        )
        assert max_abs <= higham, (
            f"emitted-executor output exceeds Higham bound: "
            f"max_abs={max_abs:.4e}, bound={higham:.4e}"
        )
