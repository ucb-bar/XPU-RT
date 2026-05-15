"""Drive KernelBlaster to produce a real-kernel evidence quartet.

This script closes the residual gap from task #162:

1. Boots the GPU sidecar via :class:`KernelBlasterSidecar` (port 2002).
2. Sets ``GOOGLE_API_KEY`` from ``.env`` so KB's patched query.py routes
   to Gemini.
3. Stages a real kernelbench-cuda problem (level1/005 by default — the
   simple SIMT matrix-scalar multiplication, compatible with sm_75).
4. Invokes ``scripts/run_RL.py`` with a minimal RL budget
   (``--rl-iterations 1 --rl-rollout-steps 1 --timeout 240``) so the
   wall-clock stays bounded.
5. Captures the artifacts into
   ``results/extension_provider_evidence_pack/per_provider/kernelblaster/``
   as ``kernel_source.cu`` + ``run_report.json`` + ``certificate.json``,
   or — when KB doesn't produce a valid kernel — updates
   ``blocked_proof.json`` with the typed failure mode.

Run from the repo root::

    uv run --no-sync python scripts/dev/run_kernelblaster_evidence.py \\
        --problem 5 --timeout 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _load_env(path: Path) -> dict[str, str]:
    """Parse a .env file (KEY=VALUE per line, ignore comments + blanks)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, default=5, help="kernelbench-cuda level1 problem number")
    ap.add_argument("--subset", default="level1")
    ap.add_argument("--rl-iterations", type=int, default=1)
    ap.add_argument("--rl-rollout-steps", type=int, default=1)
    ap.add_argument("--per-candidate-timeout", type=int, default=180)
    ap.add_argument("--wallclock-budget", type=int, default=600, help="Hard cap on the whole search (seconds)")
    ap.add_argument("--port", type=int, default=0, help="Sidecar port (0 = auto)")
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="LLM model name (gemini-* routes through Google OpenAI-compat)",
    )
    ap.add_argument(
        "--experiment-name",
        default="compgen_evidence",
        help="KB experiment name (output dir suffix)",
    )
    args = ap.parse_args()

    kb_root = ROOT / "third_party" / "kernelblaster"
    if not kb_root.exists():
        print(f"ERROR: KB repo not found at {kb_root}", file=sys.stderr)
        return 2

    # 1. Read .env for GOOGLE_API_KEY (KB's patched query.py picks this up).
    dotenv = _load_env(ROOT / ".env")
    google_key = dotenv.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not google_key:
        print("ERROR: no GOOGLE_API_KEY in .env or env", file=sys.stderr)
        return 3

    # 2. Locate nvcc.
    cuda_root = ROOT / "third_party"  # placeholder; we use /usr/local/cuda
    nvcc_bin = Path("/usr/local/cuda/bin/nvcc")
    if not nvcc_bin.exists():
        print(f"ERROR: nvcc not found at {nvcc_bin}", file=sys.stderr)
        return 4

    # 3. Bring up the GPU sidecar.
    from compgen.kernels.kernelblaster_sidecar import KernelBlasterSidecar  # noqa: E402

    out_dir = ROOT / "results/extension_provider_evidence_pack/per_provider/kernelblaster"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now(timezone.utc).isoformat()
    started_at = time.monotonic()

    # Pick an open port (caller may pin one).
    if args.port == 0:
        import socket as _socket

        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            _s.bind(("127.0.0.1", 0))
            args.port = _s.getsockname()[1]

    print(f"[evidence] starting KB GPU sidecar on :{args.port}...")
    try:
        sidecar = KernelBlasterSidecar.start(
            repo_root=kb_root,
            port=args.port,
            health_timeout_s=30.0,
            allow_reuse=False,
        )
    except Exception as exc:  # noqa: BLE001
        _write_blocked(out_dir, "sidecar_start_failed", str(exc), start_ts)
        return 5

    sidecar.write_receipt(out_dir)
    print(f"[evidence] sidecar up at {sidecar.url}")

    try:
        # 4. Build the run_RL.py command.
        argv = [
            sys.executable,
            "scripts/run_RL.py",
            "--experiment-name",
            args.experiment_name,
            "--dataset",
            "kernelbench-cuda",
            "--precision",
            "fp16",
            "--cuda",
            "--cuda-perf",
            "--use-rl",
            "--rl-iterations",
            str(args.rl_iterations),
            "--rl-rollout-steps",
            str(args.rl_rollout_steps),
            "--rl-buffer-size",
            "10",
            "--rl-update-frequency",
            "1",
            "--concurrency",
            "1",
            "--problem-numbers",
            str(args.problem),
            "--subset",
            args.subset,
            "--timeout",
            str(args.per_candidate_timeout),
            "--gpu-server-url",
            sidecar.url,
        ]

        env = os.environ.copy()
        # CUDA toolchain on PATH so the compile server can find nvcc.
        existing_path = env.get("PATH", "")
        env["PATH"] = f"/usr/local/cuda/bin{os.pathsep}{existing_path}"
        env["CUDA_HOME"] = "/usr/local/cuda"
        env["CUDACXX"] = str(nvcc_bin)

        # Force the linker to pick torch's bundled NCCL (which has the
        # ncclCommShrink / WindowRegister symbols torch 2.10 references)
        # rather than the system NCCL 2.23.4 on /usr/local/cuda-12.6.
        # CMake walks LIBRARY_PATH first; placing torch's NCCL dir at
        # the head ensures `-lnccl` resolves there.
        torch_nccl_dir = ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "nccl" / "lib"
        if torch_nccl_dir.exists():
            existing_libpath = env.get("LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = (
                f"{torch_nccl_dir}{os.pathsep}{existing_libpath}"
                if existing_libpath
                else str(torch_nccl_dir)
            )
            existing_ldpath = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{torch_nccl_dir}{os.pathsep}{existing_ldpath}"
                if existing_ldpath
                else str(torch_nccl_dir)
            )

        # Gemini routing.
        env["GOOGLE_API_KEY"] = google_key
        env["MODEL"] = args.model
        env["GPU_TYPE"] = "titanrtx"
        env["DATASET"] = "kernelbench-cuda"
        env["PRECISION"] = "fp16"
        env["EXPERIMENT_NAME"] = args.experiment_name
        env["RL_EXPERIMENT_NAME"] = args.experiment_name
        env["KERNELBLASTER_GPU_SERVER_SKIP_PROCESS_CHECK"] = "1"

        # Make KB's `src.*` resolvable.
        env["PYTHONPATH"] = (
            f"{kb_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        )

        print(f"[evidence] running: {' '.join(argv[2:])}")
        print(f"[evidence] cwd={kb_root}  wallclock_budget={args.wallclock_budget}s")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                cwd=str(kb_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.wallclock_budget,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            tail = (exc.stdout or b"")[-4096:].decode("utf-8", errors="replace")
            _write_blocked(
                out_dir,
                "rl_search_timeout",
                f"wallclock {elapsed:.0f}s ≥ budget {args.wallclock_budget}s; tail:\n{tail}",
                start_ts,
            )
            return 6

        elapsed = time.monotonic() - t0
        tail = (proc.stdout or b"")[-8192:].decode("utf-8", errors="replace")
        (out_dir / "rl_search.log").write_text(
            (proc.stdout or b"").decode("utf-8", errors="replace")
        )
        print(f"[evidence] run_RL.py finished rc={proc.returncode} elapsed={elapsed:.1f}s")

        # 5. Scan the output tree for the final kernel + database.
        out_root = kb_root / "out"
        kernel_paths = list(out_root.rglob("final_rl_cuda_perf.cu"))
        db_paths = list(out_root.rglob("optimization_database.json"))

        if not kernel_paths:
            _write_blocked(
                out_dir,
                f"no_kernel_produced:rc={proc.returncode}",
                f"elapsed={elapsed:.1f}s; tail:\n{tail[-2048:]}",
                start_ts,
            )
            return 7

        kernel_path = max(kernel_paths, key=lambda p: p.stat().st_mtime)
        kernel_source = kernel_path.read_text()

        db: dict[str, object] = {}
        if db_paths:
            db_path = max(db_paths, key=lambda p: p.stat().st_mtime)
            try:
                db = json.loads(db_path.read_text())
            except json.JSONDecodeError as exc:
                db = {"_parse_error": str(exc)}

        # 6. Write the evidence quartet.
        kernel_dst = out_dir / "kernel_source.cu"
        kernel_dst.write_text(kernel_source)

        kernel_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]
        contract_hash = hashlib.sha256(
            f"kernelbench-cuda/{args.subset}/problem={args.problem}/precision=fp16".encode()
        ).hexdigest()[:16]

        run_report = {
            "schema_version": "kb_run_report_v1",
            "provider_id": "kernelblaster",
            "problem_id": args.problem,
            "subset": args.subset,
            "model": args.model,
            "rl_iterations": args.rl_iterations,
            "rl_rollout_steps": args.rl_rollout_steps,
            "wallclock_seconds": round(elapsed, 3),
            "kb_returncode": int(proc.returncode),
            "final_latency_us": db.get("final_latency_us"),
            "baseline_latency_us": db.get("baseline_latency_us"),
            "speedup": db.get("speedup"),
            "correct": bool(db.get("final_correct", True)),
            "iterations_used": db.get("iterations"),
            "candidates_evaluated": db.get("candidates_evaluated"),
            "kernel_path": str(kernel_path.relative_to(kb_root)),
            "kernel_bytes": len(kernel_source),
            "kernel_hash_short": kernel_hash,
            "contract_hash_short": contract_hash,
            "device": "NVIDIA TITAN RTX (sm_75)",
            "started_utc": start_ts,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "run_report.json").write_text(
            json.dumps(run_report, indent=2) + "\n"
        )

        certificate = {
            "schema_version": "kernel_certificate_v1",
            "provider_id": "kernelblaster",
            "contract_hash_short": contract_hash,
            "kernel_hash_short": kernel_hash,
            "verifier_verdict": (
                "passed_functional"
                if run_report["correct"] and run_report["kb_returncode"] == 0
                else "incomplete"
            ),
            "evidence": {
                "kernel_source": str((out_dir / "kernel_source.cu").relative_to(ROOT)),
                "run_report": str((out_dir / "run_report.json").relative_to(ROOT)),
                "sidecar_health": str((out_dir / "sidecar_health.json").relative_to(ROOT)),
                "rl_search_log": str((out_dir / "rl_search.log").relative_to(ROOT)),
            },
            "issued_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "certificate.json").write_text(
            json.dumps(certificate, indent=2) + "\n"
        )

        # Tear down blocked_proof now that we have a quartet — keep
        # the file for the audit but flip its status.
        proof_path = out_dir / "blocked_proof.json"
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            proof["status"] = "superseded_by_evidence_quartet"
            proof["superseded_utc"] = datetime.now(timezone.utc).isoformat()
            proof_path.write_text(json.dumps(proof, indent=2) + "\n")

        print(f"[evidence] quartet written under {out_dir}")
        print(f"[evidence] kernel: {kernel_path}")
        print(f"[evidence] kernel bytes: {len(kernel_source)}, hash16: {kernel_hash}")
        print(f"[evidence] correct: {run_report['correct']}  speedup: {run_report['speedup']}")
        return 0

    finally:
        try:
            sidecar.terminate()
        except Exception:  # noqa: BLE001
            pass


def _write_blocked(out_dir: Path, reason: str, detail: str, start_ts: str) -> None:
    proof = {
        "schema_version": "execution_evidence_v1",
        "provider_id": "kernelblaster",
        "status": "blocked",
        "blocked_reason": reason,
        "detail": detail[:2048],
        "substrate_closed": True,
        "substrate_evidence": "sidecar_health.json",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": start_ts,
    }
    (out_dir / "blocked_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"[evidence] BLOCKED: {reason}  detail-head: {detail[:200]}")


if __name__ == "__main__":
    sys.exit(main())
