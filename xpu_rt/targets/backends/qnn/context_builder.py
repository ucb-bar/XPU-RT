"""On-board QNN context-binary construction + measurement.

The QRB5165 board ships ``qnn-context-binary-generator`` (at
``/root/qairt/bin/target/``) which compiles a (sub)graph of a
``.dlc`` model into a backend-specific context binary that
``qnn-net-run --retrieve_context`` can replay without re-finalising
the graph each time. This is how the original XPU-RT work measured
sub-network blocks ("medium" granularity) on the board.

This module wraps that workflow:

* :func:`build_context` SSHes the board, invokes
  ``qnn-context-binary-generator``, returns the on-board path of
  the produced ``.bin``.
* :func:`measure_context` then runs ``qnn-net-run
  --retrieve_context=<ctx.bin>`` and returns the per-inference
  latency, reusing the timing protocol from
  :mod:`xpu_rt.targets.backends.qnn.on_board_runner`.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
import subprocess
from pathlib import Path
from typing import Literal

from xpu_rt.targets.backends.qnn.board import BoardConfig
from xpu_rt.targets.backends.qnn.on_board_runner import (
    BACKEND_LIB,
    MeasurementResult,
    _ssh_cmd,
)


@dataclasses.dataclass(frozen=True)
class BlockSpec:
    """One sub-network block to extract from a model as a context."""

    name: str                    # logical name (used for filenames / events)
    dlc_path: str                # remote path to the source DLC
    backend: str                 # "CPU" | "GPU" | "DSP" | "HTA" | "HTP"
    out_dir: str = "/root/contexts"
    # Optional: subgraph-name filter (when the DLC carries multiple
    # named subgraphs). When None the whole DLC compiles to a context.
    subgraph_name: str | None = None


@dataclasses.dataclass(frozen=True)
class ContextBuildResult:
    block_name: str
    backend: str
    remote_bin_path: str
    ok: bool
    error: str = ""
    stderr_tail: str = ""


def _remote_build_script(spec: BlockSpec, *, qnn_sdk_root: str = "/root/qairt") -> str:
    """Bash one-liner that invokes qnn-context-binary-generator."""
    lib = BACKEND_LIB[spec.backend]
    binary_file = f"{spec.name}_{spec.backend.lower()}"
    out_path = f"{spec.out_dir}/{binary_file}.bin"
    args = [
        f"{qnn_sdk_root}/bin/target/qnn-context-binary-generator",
        f"--model={shlex.quote(spec.dlc_path)}",
        f"--backend={qnn_sdk_root}/lib/target/{lib}",
        f"--output_dir={shlex.quote(spec.out_dir)}",
        f"--binary_file={shlex.quote(binary_file)}",
    ]
    return (
        f'export LD_LIBRARY_PATH={qnn_sdk_root}/lib/target:${{LD_LIBRARY_PATH:-}} && '
        f'export ADSP_LIBRARY_PATH="{qnn_sdk_root}/lib/hexagon-v66;/dsp/cdsp;/dsp" && '
        f'mkdir -p {shlex.quote(spec.out_dir)} && '
        + " ".join(args) +
        f' 1>&2 && '
        f'echo "BUILT_BIN={out_path}"'
    )


_BUILT_RE = re.compile(r"BUILT_BIN=(\S+)")


def build_context(
    cfg: BoardConfig,
    spec: BlockSpec,
    *,
    timeout_s: int = 600,
) -> ContextBuildResult:
    """Run qnn-context-binary-generator on the board for ``spec``."""
    if spec.backend not in BACKEND_LIB:
        return ContextBuildResult(
            block_name=spec.name, backend=spec.backend,
            remote_bin_path="", ok=False,
            error=f"unknown backend {spec.backend!r}",
        )
    remote = _remote_build_script(spec)
    cmd = _ssh_cmd(cfg, remote, timeout=timeout_s)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ContextBuildResult(
            block_name=spec.name, backend=spec.backend,
            remote_bin_path="", ok=False,
            error=f"ssh timeout after {timeout_s}s",
        )
    if proc.returncode != 0:
        return ContextBuildResult(
            block_name=spec.name, backend=spec.backend,
            remote_bin_path="", ok=False,
            error=f"rc={proc.returncode}",
            stderr_tail=proc.stderr[-400:],
        )
    m = _BUILT_RE.search(proc.stdout)
    if not m:
        return ContextBuildResult(
            block_name=spec.name, backend=spec.backend,
            remote_bin_path="", ok=False,
            error="no BUILT_BIN line in output",
            stderr_tail=(proc.stdout + proc.stderr)[-400:],
        )
    return ContextBuildResult(
        block_name=spec.name, backend=spec.backend,
        remote_bin_path=m.group(1), ok=True,
    )


def _retrieve_context_script(
    ctx_remote: str,
    backend_lib: str,
    input_list: str,
    iters: int,
    *,
    qnn_sdk_root: str = "/root/qairt",
    out_dir: str = "/tmp/qnn_ctx_out",
) -> str:
    return (
        f'export LD_LIBRARY_PATH={qnn_sdk_root}/lib/target:${{LD_LIBRARY_PATH:-}} && '
        f'export ADSP_LIBRARY_PATH="{qnn_sdk_root}/lib/hexagon-v66;/dsp/cdsp;/dsp" && '
        f'MULTI=$(mktemp) && '
        f'LINE=$(head -n1 {shlex.quote(input_list)}) && '
        f'i=0 && while [ $i -lt {iters} ]; do echo "$LINE" >> "$MULTI"; i=$((i+1)); done && '
        f'rm -rf {shlex.quote(out_dir)} && mkdir -p {shlex.quote(out_dir)} && '
        f'START=$(date +%s%N) && '
        f'{qnn_sdk_root}/bin/target/qnn-net-run '
        f'--retrieve_context {shlex.quote(ctx_remote)} '
        f'--backend {qnn_sdk_root}/lib/target/{backend_lib} '
        f'--input_list $MULTI '
        f'--output_dir {shlex.quote(out_dir)} 1>/dev/null 2>&1 && '
        f'END=$(date +%s%N) && '
        f'echo "TOTAL_NS=$((END-START))" && '
        f'echo "ITERS={iters}" && '
        f'rm -f "$MULTI"'
    )


_TOTAL_NS_RE = re.compile(r"TOTAL_NS=(\d+)")
_ITERS_RE = re.compile(r"ITERS=(\d+)")


def measure_context(
    cfg: BoardConfig,
    *,
    block_name: str,
    ctx_remote: str,
    backend: str,
    input_list: str,
    iters: int = 10,
    timeout_s: int = 300,
) -> MeasurementResult:
    """Run qnn-net-run --retrieve_context for a built context binary."""
    lib = BACKEND_LIB.get(backend)
    if lib is None:
        return MeasurementResult(
            workload_id=block_name, backend=backend,
            dlc_path=ctx_remote, iters=iters,
            mean_us=None, total_ms=None, ok=False,
            error=f"unknown backend {backend!r}",
        )
    remote = _retrieve_context_script(ctx_remote, lib, input_list, iters)
    cmd = _ssh_cmd(cfg, remote, timeout=timeout_s)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return MeasurementResult(
            workload_id=block_name, backend=backend,
            dlc_path=ctx_remote, iters=iters,
            mean_us=None, total_ms=None, ok=False,
            error=f"ssh timeout after {timeout_s}s",
        )
    if proc.returncode != 0:
        return MeasurementResult(
            workload_id=block_name, backend=backend,
            dlc_path=ctx_remote, iters=iters,
            mean_us=None, total_ms=None, ok=False,
            error=f"rc={proc.returncode}",
            raw_stderr_tail=proc.stderr[-400:],
        )
    m_total = _TOTAL_NS_RE.search(proc.stdout)
    m_iters = _ITERS_RE.search(proc.stdout)
    if not m_total or not m_iters:
        return MeasurementResult(
            workload_id=block_name, backend=backend,
            dlc_path=ctx_remote, iters=iters,
            mean_us=None, total_ms=None, ok=False,
            error="no TOTAL_NS/ITERS in output",
            raw_stderr_tail=proc.stdout[-400:],
        )
    total_ns = int(m_total.group(1))
    n = int(m_iters.group(1))
    total_ms = total_ns / 1_000_000
    mean_us = (total_ns / max(1, n)) / 1_000
    return MeasurementResult(
        workload_id=block_name, backend=backend,
        dlc_path=ctx_remote, iters=n,
        mean_us=mean_us, total_ms=total_ms, ok=True,
    )
