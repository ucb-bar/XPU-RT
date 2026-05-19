"""Target-dispatched Spike-run FastAPI server.

Drop-in for vanilla KernelBlaster's
``third_party/kernelblaster/src/kernelblaster/servers/gpu.py``,
unified across targets. ``--target`` at startup picks which
:class:`SpikeTargetSpec` to use; each request executes the ELF
under ``spike <spec.spike_flag> pk <elf>``.

Run with::

    uv run python -m xpu_rt.spike_harness.run_server \\
        --target gemmini_mx --port 8202
    uv run python -m xpu_rt.spike_harness.run_server \\
        --target saturn_opu_v128 --port 8302
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from xpu_rt.spike_harness.targets import SpikeTargetSpec, resolve_target


logger = logging.getLogger("xpu_rt.spike_harness.run_server")


def _conda_root() -> Path:
    return Path(
        os.environ.get(
            "XPU_RT_RISCV_CONDA_ROOT",
            "/scratch2/agustin/chipyard/.conda-env/riscv-tools",
        )
    )


def _spike_bin() -> Path:
    env = os.environ.get("XPU_RT_SPIKE_BIN")
    return Path(env) if env else _conda_root() / "bin" / "spike"


def _pk_bin() -> Path:
    env = os.environ.get("XPU_RT_PK_BIN")
    return Path(env) if env else _conda_root() / "riscv64-unknown-elf" / "bin" / "pk"


class GpuCommandResult(BaseModel):
    stdout: str | list[str] = ""
    stderr: str | list[str] = ""
    success: bool = False
    message: str | None = None


def _save_binary_to_temp(binary_data: bytes, filename: str, *, target_id: str) -> Path:
    safe_name = Path(filename).name or f"{target_id}_executable"
    path = Path(tempfile.mkdtemp(prefix=f"spike_harness_{target_id}_run_")) / safe_name
    path.write_bytes(binary_data)
    path.chmod(0o755)
    return path


async def _run_one(
    binary_path: Path,
    args: str,
    env_vars: dict | None,
    timeout: float,
    *,
    spec: SpikeTargetSpec,
) -> tuple[str, str, int]:
    cmd = [str(_spike_bin()), *spec.spike_flag, str(_pk_bin()), str(binary_path)]
    if args:
        cmd.extend(args.split())
    env = os.environ.copy()
    if env_vars:
        env.update({str(k): str(v) for k, v in env_vars.items()})
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return "", f"spike timed out after {timeout}s: {exc}", 124
    return proc.stdout, proc.stderr, proc.returncode


def make_app(spec: SpikeTargetSpec) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sp = _spike_bin()
        pk = _pk_bin()
        if not sp.is_file() or not pk.is_file():
            logger.error("toolchain missing: spike=%s pk=%s", sp, pk)
        else:
            logger.info(
                "spike-harness run server up. target=%s spike=%s pk=%s flag=%s",
                spec.target_id, sp, pk, spec.spike_flag,
            )
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "gpu-server",
            "target": spec.target_id,
        }

    @app.post("/gpu/binary", response_model=GpuCommandResult)
    async def execute_gpu_binary(
        binary: UploadFile = File(..., description="Binary executable to run on Spike"),
        args: Optional[str] = Form("", description="Command line arguments for the binary"),
        env_vars: Optional[str] = Form(None, description="Environment variables JSON"),
        prefix_command: Optional[str] = Form(
            None,
            description="Ignored — KB expects this for ncu; cycles read inside the harness.",
        ),
        n_runs: Optional[int] = Form(1),
        timeout: Optional[float] = Form(600),
    ):
        if prefix_command:
            logger.info(
                "prefix_command=%r ignored on Spike (counter is in the harness stdout).",
                prefix_command,
            )
        binary_bytes = await binary.read()
        if not binary_bytes:
            raise HTTPException(status_code=400, detail="empty binary uploaded")
        binary_path = _save_binary_to_temp(
            binary_bytes, binary.filename or f"{spec.target_id}_executable",
            target_id=spec.target_id,
        )

        parsed_env: dict | None = None
        if env_vars:
            try:
                parsed_env = json.loads(env_vars)
                if not isinstance(parsed_env, dict):
                    raise ValueError("env_vars must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"invalid env_vars json: {exc}")

        n = max(int(n_runs or 1), 1)
        stdouts: list[str] = []
        stderrs: list[str] = []
        overall_ok = True
        for i in range(n):
            out, err, rc = await _run_one(
                binary_path, args or "", parsed_env, float(timeout or 600),
                spec=spec,
            )
            stdouts.append(out)
            stderrs.append(err)
            if rc != 0:
                overall_ok = False
                logger.warning(
                    "[run %d/%d] spike rc=%d, stderr tail: %s",
                    i + 1, n, rc, (err or "")[-200:],
                )
        if n == 1:
            return GpuCommandResult(stdout=stdouts[0], stderr=stderrs[0], success=overall_ok)
        return GpuCommandResult(stdout=stdouts, stderr=stderrs, success=overall_ok)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target", default="gemmini_mx",
        help="Target id resolved via xpu_rt.spike_harness.targets.resolve_target.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8202)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    spec = resolve_target(args.target)
    app = make_app(spec)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GpuCommandResult", "main", "make_app"]
