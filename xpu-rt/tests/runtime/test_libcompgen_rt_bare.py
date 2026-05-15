"""Bare-metal libcompgen_rt smoke test on Spike.

Cross-compiles the static library for RISC-V freestanding using the
Zephyr SDK (``CG_RT_PLATFORM=bare``), links a tiny HTIF-exit program
that drives the public primitives (device open, buffer alloc, copy
command buffer, queue_submit, semaphore query, event tensor notify +
wait), and runs it through Spike. The ELF uses non-zero HTIF exit
codes for individual assertion failures, so Spike's exit status
points directly at the failing line in ``tests/bare/smoke.c``.

Skipped when either the Zephyr SDK or Spike is missing on the host.
Honours ``$ZEPHYR_SDK`` and ``$SPIKE`` env overrides via the wrapper
shell script ``scripts/libcompgen_rt_bare_spike.sh``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "libcompgen_rt_bare_spike.sh"


def _toolchain_available() -> bool:
    sdk = Path(
        os.environ.get(
            "ZEPHYR_SDK",
            "/scratch2/dima/testing/zephyr-chipyard-sw-torch-dryrun-2/tools-manual/zephyr-sdk-1.0.0-beta1",
        )
    )
    spike = Path(
        os.environ.get(
            "SPIKE",
            "/scratch2/dima/testing/zephyr-chipyard-sw-torch-dryrun-2/tools/miniforge3/envs/zephyr/bin/spike",
        )
    )
    gcc = sdk / "gnu" / "riscv64-zephyr-elf" / "bin" / "riscv64-zephyr-elf-gcc"
    return gcc.is_file() and spike.is_file() and shutil.which("/usr/bin/cmake") is not None


@pytest.mark.skipif(
    not _toolchain_available(),
    reason="Zephyr SDK or Spike not available on this host",
)
def test_bare_metal_smoke_on_spike(tmp_path: Path) -> None:
    """Build + run the bare-metal ELF via the wrapper script; exit
    status 0 means every assertion in smoke.c passed."""
    assert SCRIPT.is_file(), f"script missing: {SCRIPT}"
    env = os.environ.copy()
    env.setdefault("BUILD_DIR", str(tmp_path / "build-riscv"))

    proc = subprocess.run(
        [str(SCRIPT)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"bare-metal smoke test failed (exit={proc.returncode})\n{combined}"
