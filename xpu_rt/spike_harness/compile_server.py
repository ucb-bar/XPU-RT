"""Target-dispatched cross-compile FastAPI server.

Drop-in for vanilla KernelBlaster's
``third_party/kernelblaster/src/kernelblaster/servers/compile.py``,
unified across targets. The ``--target`` flag at startup picks which
:class:`SpikeTargetSpec` to use; each request reads
``march_flags`` / ``include_args`` / ``extra_compile_flags`` from
that spec.

Run with::

    uv run python -m xpu_rt.spike_harness.compile_server \\
        --target gemmini_mx --port 8201
    uv run python -m xpu_rt.spike_harness.compile_server \\
        --target saturn_opu_v128 --port 8301
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from xpu_rt.spike_harness.targets import SpikeTargetSpec, resolve_target


logger = logging.getLogger("xpu_rt.spike_harness.compile_server")


def _conda_root() -> Path:
    return Path(
        os.environ.get(
            "XPU_RT_RISCV_CONDA_ROOT",
            "/scratch2/agustin/chipyard/.conda-env/riscv-tools",
        )
    )


def _cc_bin() -> Path:
    env = os.environ.get("XPU_RT_RISCV_CC")
    return (
        Path(env)
        if env
        else _conda_root() / "bin" / "riscv64-unknown-linux-gnu-gcc"
    )


def _artifacts_dir(target_id: str) -> Path:
    p = Path(
        os.environ.get(
            f"XPU_RT_{target_id.upper()}_ARTIFACTS",
            os.environ.get(
                "XPU_RT_SPIKE_HARNESS_ARTIFACTS",
                tempfile.gettempdir() + f"/spike_harness_{target_id}_artifacts",
            ),
        )
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


class CompilationResult(BaseModel):
    job_name: str
    main_file: str
    cuda_file: str
    success: bool = False
    message: str | None = None
    output_path: str | None = None
    persistent_artifacts_dir: str | None = None


def _build_compile_cmd(
    *,
    cc: Path,
    spec: SpikeTargetSpec,
    kernel_path: Path,
    main_path: Path,
    out_path: Path,
) -> list[str]:
    return [
        str(cc),
        # Force C even for files KB names .cpp/.cu.
        "-x", "c", str(kernel_path),
        "-x", "c", str(main_path),
        "-std=gnu99", "-O2", "-static",
        "-fno-common", "-fno-builtin-printf",
        *spec.march_flags,
        *spec.include_args,
        *spec.extra_compile_flags,
        "-o", str(out_path),
        "-lm", "-lgcc",
    ]


def make_app(spec: SpikeTargetSpec) -> FastAPI:
    """Build a FastAPI app pinned to one target spec.

    Keeping ``spec`` as a closure rather than a global lets the test
    suite spin up multiple apps in one process when needed.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cc = _cc_bin()
        if not cc.is_file():
            logger.error("compiler not found at %s", cc)
        else:
            logger.info(
                "spike-harness compile server up. target=%s cc=%s artifacts_dir=%s",
                spec.target_id, cc, _artifacts_dir(spec.target_id),
            )
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "compile-server",
            "target": spec.target_id,
        }

    @app.get("/compile", response_model=CompilationResult)
    async def process_compilation_request(
        job_name: str,
        main_file: str,
        cuda_file: str,
        sm_version: str = "",
        persistent_artifacts: int = 0,
    ):
        main_p = Path(main_file)
        kernel_p = Path(cuda_file)

        if not main_p.is_file():
            return CompilationResult(
                job_name=job_name, main_file=main_file, cuda_file=cuda_file,
                success=False, message=f"main_file not found: {main_p}",
            )
        if not kernel_p.is_file():
            return CompilationResult(
                job_name=job_name, main_file=main_file, cuda_file=cuda_file,
                success=False, message=f"cuda_file (kernel) not found: {kernel_p}",
            )

        out_path = _artifacts_dir(spec.target_id) / f"{job_name}_{uuid.uuid4().hex[:8]}.elf"
        cmd = _build_compile_cmd(
            cc=_cc_bin(),
            spec=spec,
            kernel_path=kernel_p,
            main_path=main_p,
            out_path=out_path,
        )

        logger.info("[%s] compile: %s", job_name, " ".join(cmd))
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            return CompilationResult(
                job_name=job_name, main_file=main_file, cuda_file=cuda_file,
                success=False, message=f"compile timeout: {exc}",
            )

        if proc.returncode != 0:
            msg = (proc.stdout + proc.stderr)[-4000:]
            logger.warning(
                "[%s] compile failed: %s", job_name,
                msg.splitlines()[-1] if msg else "(no output)",
            )
            return CompilationResult(
                job_name=job_name, main_file=main_file, cuda_file=cuda_file,
                success=False, message=f"compile failed:\n{msg}",
            )

        result = CompilationResult(
            job_name=job_name, main_file=main_file, cuda_file=cuda_file,
            success=True, message="Compilation successful",
            output_path=str(out_path),
        )

        if persistent_artifacts:
            pdir = _artifacts_dir(spec.target_id) / "persistent" / out_path.name
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / kernel_p.name).write_text(kernel_p.read_text())
            (pdir / main_p.name).write_text(main_p.read_text())
            result.persistent_artifacts_dir = str(pdir)

        logger.info("[%s] compile success -> %s", job_name, out_path)
        return result

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target", default="gemmini_mx",
        help="Target id resolved via xpu_rt.spike_harness.targets.resolve_target.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8201)
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


__all__ = ["CompilationResult", "main", "make_app"]
