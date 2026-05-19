"""QNN-native on-board measurement (qnn-net-run + DLCs already on board).

The canonical Qualcomm benchmarking path: SSH the QRB5165 board, run
``qnn-net-run`` against a pre-converted ``.dlc`` with one of the QNN
backend libraries (CPU / GPU / DSP / HTA / HTP), time the run, and
parse the per-inference latency.

We deliberately do NOT go through merlin's iree-compile +
``compile_dispatch_matrix.py`` + ``profile_dispatch_matrix.py`` path
— the user prefers the QNN-native flow for board work (see memory
``feedback_qnn_native_flow``).

Two API levels:

* :func:`measure_workload` — one (model, backend, dlc-variant)
  measurement, returns a typed result.
* :func:`measure_matrix` — sweep a list of workloads × backends,
  returns a nested dict ready for the placement scheduler.

The runner reuses the conventions from
``models/qnn/benchmark_qnn.sh`` so users get the same numbers they'd
get from running that script by hand:

* ``LD_LIBRARY_PATH`` includes ``/root/qairt/lib/target``.
* ``ADSP_LIBRARY_PATH`` is set for Hexagon backends.
* The input list is repeated ``iters`` times to amortise startup.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path

from xpu_rt.targets.backends.qnn.board import BoardConfig

# Mapping the scheduler's machine names to (QNN backend lib, DLC suffix).
# DSP and HTA are different physical lanes on QRB5165; we expose both
# under HTA-style "quantized" DLC ingest. The board ships
# libQnnHta.so + libQnnHtp.so + libQnnDsp.so — we let the caller pick.
BACKEND_LIB = {
    "CPU": "libQnnCpu.so",
    "GPU": "libQnnGpu.so",
    "DSP": "libQnnDsp.so",
    "HTA": "libQnnHta.so",
    "HTP": "libQnnHtp.so",
}
# Default DLC variant per backend (float for CPU/GPU, quantised int8 for
# DSP/HTA/HTP). Callers may override.
BACKEND_DLC_VARIANT = {
    "CPU": "",
    "GPU": "_fp16",
    "DSP": "_quantized",
    "HTA": "_quantized",
    "HTP": "_quantized",
}


@dataclasses.dataclass(frozen=True)
class WorkloadSpec:
    """One workload to benchmark on the board."""

    workload_id: str
    model_dir: str          # remote path, e.g. /root/models/yolov8n
    model_name: str         # base name, e.g. yolov8n
    input_list: str | None = None  # remote path; falls back to <dir>/input_list.txt
    sla_us: float | None = None


@dataclasses.dataclass(frozen=True)
class MeasurementResult:
    """One (workload, backend) measurement."""

    workload_id: str
    backend: str
    dlc_path: str
    iters: int
    mean_us: float | None
    total_ms: float | None
    raw_stderr_tail: str = ""
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _ssh_cmd(cfg: BoardConfig, remote: str, *, timeout: int = 600) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes",
           "-o", f"ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=accept-new"]
    if cfg.ssh_identity is not None:
        cmd.extend(["-i", str(cfg.ssh_identity)])
    cmd.append(cfg.ssh_host)
    cmd.append(remote)
    return cmd


def _resolve_dlc_path(spec: WorkloadSpec, backend: str,
                      *, variant_override: str | None = None) -> str:
    """Return the remote DLC path for ``(workload, backend)``."""
    variant = variant_override
    if variant is None:
        variant = BACKEND_DLC_VARIANT.get(backend, "")
    return f"{spec.model_dir}/{spec.model_name}{variant}.dlc"


def _remote_benchmark_script(
    *,
    dlc_path: str,
    backend_lib: str,
    input_list: str,
    iters: int,
    qnn_sdk_root: str = "/root/qairt",
    out_dir: str | None = None,
) -> str:
    """Build the bash one-liner that runs qnn-net-run on the board.

    The script wraps qnn-net-run with timing capture + a small
    ``MULTI_INPUT_TMP`` file (input list repeated ``iters`` times,
    matching ``benchmark_qnn.sh``'s convention).
    """
    out_dir = out_dir or "/tmp/qnn_net_run_out"
    # Build the input-list expansion in pure shell so we don't have to
    # quote a Python one-liner through SSH (one extra layer of shell
    # quoting eats inner quotes).
    return (
        f'export LD_LIBRARY_PATH={qnn_sdk_root}/lib/target:${{LD_LIBRARY_PATH:-}} && '
        f'export ADSP_LIBRARY_PATH="{qnn_sdk_root}/lib/hexagon-v66;/dsp/cdsp;/dsp" && '
        f'MULTI=$(mktemp) && '
        f'LINE=$(head -n1 {shlex.quote(input_list)}) && '
        f'i=0 && while [ $i -lt {iters} ]; do echo "$LINE" >> "$MULTI"; i=$((i+1)); done && '
        f'rm -rf {shlex.quote(out_dir)} && mkdir -p {shlex.quote(out_dir)} && '
        f'START=$(date +%s%N) && '
        f'{qnn_sdk_root}/bin/target/qnn-net-run '
        f'--dlc_path {shlex.quote(dlc_path)} '
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


def measure_workload(
    cfg: BoardConfig,
    spec: WorkloadSpec,
    backend: str,
    *,
    iters: int = 10,
    dlc_variant_override: str | None = None,
    timeout_s: int = 300,
) -> MeasurementResult:
    """Run qnn-net-run once for ``(spec, backend)`` and parse the latency."""
    lib = BACKEND_LIB.get(backend)
    if lib is None:
        return MeasurementResult(
            workload_id=spec.workload_id, backend=backend,
            dlc_path="", iters=iters, mean_us=None, total_ms=None,
            ok=False, error=f"unknown backend {backend!r}",
        )
    dlc = _resolve_dlc_path(spec, backend, variant_override=dlc_variant_override)
    input_list = spec.input_list or f"{spec.model_dir}/input_list.txt"
    remote = _remote_benchmark_script(
        dlc_path=dlc, backend_lib=lib, input_list=input_list, iters=iters,
    )
    cmd = _ssh_cmd(cfg, remote, timeout=timeout_s)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return MeasurementResult(
            workload_id=spec.workload_id, backend=backend,
            dlc_path=dlc, iters=iters, mean_us=None, total_ms=None,
            ok=False, error=f"ssh timeout after {timeout_s}s",
        )
    if proc.returncode != 0:
        return MeasurementResult(
            workload_id=spec.workload_id, backend=backend,
            dlc_path=dlc, iters=iters, mean_us=None, total_ms=None,
            ok=False, error=f"rc={proc.returncode}",
            raw_stderr_tail=proc.stderr[-500:],
        )
    m_total = _TOTAL_NS_RE.search(proc.stdout)
    m_iters = _ITERS_RE.search(proc.stdout)
    if not m_total or not m_iters:
        return MeasurementResult(
            workload_id=spec.workload_id, backend=backend,
            dlc_path=dlc, iters=iters, mean_us=None, total_ms=None,
            ok=False, error="no TOTAL_NS/ITERS in output",
            raw_stderr_tail=proc.stdout[-300:],
        )
    total_ns = int(m_total.group(1))
    n = int(m_iters.group(1))
    total_ms = total_ns / 1_000_000
    mean_us = (total_ns / max(1, n)) / 1_000  # ns → µs per iteration
    return MeasurementResult(
        workload_id=spec.workload_id, backend=backend,
        dlc_path=dlc, iters=n,
        mean_us=mean_us, total_ms=total_ms,
        ok=True,
    )


def measure_matrix(
    cfg: BoardConfig,
    workloads: Iterable[WorkloadSpec],
    backends: Iterable[str],
    *,
    iters: int = 10,
    timeout_s: int = 300,
) -> dict[str, dict[str, MeasurementResult]]:
    """Run :func:`measure_workload` for every (workload, backend) pair."""
    out: dict[str, dict[str, MeasurementResult]] = {}
    workloads = list(workloads)
    backends = list(backends)
    for spec in workloads:
        out[spec.workload_id] = {}
        for b in backends:
            out[spec.workload_id][b] = measure_workload(
                cfg, spec, b, iters=iters, timeout_s=timeout_s,
            )
    return out


# --------------------------------------------------------------------------- #
# Convenience helpers used by the MCP tool / CLI to format results.
# --------------------------------------------------------------------------- #


def measure_concurrent(
    cfg: BoardConfig,
    assignment: dict[str, WorkloadSpec],
    backend_of: dict[str, str],
    *,
    iters: int = 10,
    timeout_s: int = 300,
) -> dict[str, MeasurementResult]:
    """Launch one qnn-net-run per (workload, backend) **concurrently**.

    Each workload runs on its assigned backend in parallel SSH sessions
    so on-backend contention (shared DDR, thermal, scheduler queue) is
    captured in the timing. The wall-clock makespan of the
    concurrent run is ``max(r.total_ms for r in results.values())``.
    """
    import concurrent.futures

    out: dict[str, MeasurementResult] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(assignment)),
    ) as pool:
        futures = {
            wid: pool.submit(
                measure_workload, cfg, spec, backend_of[wid],
                iters=iters, timeout_s=timeout_s,
            )
            for wid, spec in assignment.items()
        }
        for wid, fut in futures.items():
            out[wid] = fut.result()
    return out


def latency_table_dict(
    matrix: dict[str, dict[str, MeasurementResult]],
) -> list[dict]:
    """Flatten the measurement matrix into Δ-table-style rows."""
    rows: list[dict] = []
    for wid, by_backend in matrix.items():
        for b, m in by_backend.items():
            rows.append({
                "workload": wid,
                "backend": b,
                "iters": m.iters,
                "mean_us": m.mean_us,
                "total_ms": m.total_ms,
                "ok": m.ok,
                "error": m.error,
                "dlc_path": m.dlc_path,
            })
    return rows
