"""Real per-op profiling via qnn-net-run + qnn-profile-viewer.

QNN supports per-op profiling out of the box. The full pipeline is:

    qnn-net-run --dlc_path X --backend lib --profiling_level=detailed \
                --profiling_option=stats --input_list ... \
                --output_dir OUT
    qnn-profile-viewer --input_log OUT/qnn-profiling-data.log \
                       --output_file timings.csv

The CSV gives one row per op with the columns we care about
(``Op Name``, ``Compute Unit``, ``Backend (us)``, ...). This module
wraps that workflow into a single SSH round-trip and returns a typed
``OpTiming`` list. **These are real on-board per-op measurements**;
no estimation involved.

Used by the closed-loop MCP tool
``xpu_rt_qnn_profile_detailed_on_board`` to populate per-op cells
on every backend a workload runs on.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import shlex
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

from xpu_rt.targets.backends.qnn.board import BoardConfig
from xpu_rt.targets.backends.qnn.on_board_runner import BACKEND_LIB, _ssh_cmd


@dataclasses.dataclass(frozen=True)
class OpTiming:
    """One row of qnn-profile-viewer output, normalised."""

    op_name: str
    op_kind: str           # e.g. Conv2d, Relu, BatchNorm
    backend: str           # CPU / GPU / DSP / HTA / HTP
    compute_unit: str      # what the profiler reported (HMX, HVX, ...)
    mean_us: float
    min_us: float | None = None
    max_us: float | None = None
    iters: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Remote shell script
# --------------------------------------------------------------------------- #


def _remote_profile_script(
    dlc_path: str,
    backend: str,
    input_list: str,
    iters: int,
    *,
    qnn_sdk_root: str = "/root/qairt",
    work_dir: str = "/tmp/qnn_profile_detailed",
) -> str:
    lib = BACKEND_LIB[backend]
    return (
        f'export LD_LIBRARY_PATH={qnn_sdk_root}/lib/target:${{LD_LIBRARY_PATH:-}} && '
        f'export ADSP_LIBRARY_PATH="{qnn_sdk_root}/lib/hexagon-v66;/dsp/cdsp;/dsp" && '
        f'rm -rf {shlex.quote(work_dir)} && mkdir -p {shlex.quote(work_dir)} && '
        f'cd {shlex.quote(work_dir)} && '
        # Build input list repeated iters times.
        f'MULTI=$(mktemp) && '
        f'LINE=$(head -n1 {shlex.quote(input_list)}) && '
        f'i=0 && while [ $i -lt {iters} ]; do echo "$LINE" >> "$MULTI"; i=$((i+1)); done && '
        f'OUT={shlex.quote(work_dir)}/run && rm -rf "$OUT" && mkdir -p "$OUT" && '
        f'{qnn_sdk_root}/bin/target/qnn-net-run '
        f'--dlc_path {shlex.quote(dlc_path)} '
        f'--backend {qnn_sdk_root}/lib/target/{lib} '
        f'--input_list $MULTI --output_dir "$OUT" '
        f'--profiling_level=detailed 1>/dev/null 2>&1 && '
        # qnn-profile-viewer dumps to CSV.
        f'{qnn_sdk_root}/bin/target/qnn-profile-viewer '
        f'--input_log "$OUT"/qnn-profiling-data_0.log '
        f'--output_file "$OUT"/timings.csv 1>/dev/null 2>&1 && '
        f'echo "===CSV==="; cat "$OUT"/timings.csv; echo "===END==="; '
        f'rm -f "$MULTI"'
    )


_CSV_BEGIN = "===CSV==="
_CSV_END = "===END==="


def _extract_csv(stdout: str) -> str:
    start = stdout.find(_CSV_BEGIN)
    end = stdout.find(_CSV_END)
    if start < 0 or end < 0 or end <= start:
        return ""
    return stdout[start + len(_CSV_BEGIN):end].strip()


# Heuristic header detection — QNN's profile-viewer CSV column names
# vary a bit across versions. We match the canonical fields by
# substring, case-insensitively.
def _column_index(header: list[str], *needles: str) -> int | None:
    h = [c.strip().lower() for c in header]
    for n in needles:
        n_l = n.lower()
        for i, c in enumerate(h):
            if n_l in c:
                return i
    return None


def parse_profile_csv(csv_text: str, *, backend: str) -> list[OpTiming]:
    """Parse a qnn-profile-viewer CSV into ``OpTiming`` rows.

    Tolerant to column-order drift across QAIRT versions.
    """
    if not csv_text.strip():
        return []
    reader = csv.reader(StringIO(csv_text))
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        return []
    header = rows[0]
    name_i = _column_index(header, "op name", "node name", "name")
    kind_i = _column_index(header, "op type", "node type", "kind")
    unit_i = _column_index(header, "compute unit", "execution unit", "unit")
    # Try several spellings for the backend latency column.
    mean_i = _column_index(header,
                            "backend (us)", "execution time (us)",
                            "exec time (us)", "time (us)", "duration (us)",
                            "total time (us)")
    if name_i is None or mean_i is None:
        return []
    out: list[OpTiming] = []
    for row in rows[1:]:
        if len(row) <= max(filter(None, [name_i, mean_i])):
            continue
        try:
            mean_us = float(row[mean_i])
        except ValueError:
            continue
        out.append(OpTiming(
            op_name=row[name_i].strip(),
            op_kind=(row[kind_i].strip() if kind_i is not None
                                            and kind_i < len(row) else ""),
            backend=backend,
            compute_unit=(row[unit_i].strip() if unit_i is not None
                                                and unit_i < len(row) else ""),
            mean_us=mean_us,
        ))
    return out


def profile_whole_net(
    cfg: BoardConfig,
    dlc_path: str,
    backend: str,
    *,
    input_list: str,
    iters: int = 10,
    timeout_s: int = 600,
) -> tuple[list[OpTiming], str]:
    """Run a whole-network DLC with profiling=detailed; parse per-op times.

    Returns ``(timings, stderr_tail)``. An empty timings list with
    non-empty stderr means the profiler failed; the caller should
    record that as a measurement gap rather than substituting an
    estimate.
    """
    if backend not in BACKEND_LIB:
        return [], f"unknown backend {backend!r}"
    remote = _remote_profile_script(
        dlc_path=dlc_path, backend=backend,
        input_list=input_list, iters=iters,
    )
    cmd = _ssh_cmd(cfg, remote, timeout=timeout_s)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return [], f"ssh timeout after {timeout_s}s"
    if proc.returncode != 0:
        return [], proc.stderr[-500:]
    csv_text = _extract_csv(proc.stdout)
    return parse_profile_csv(csv_text, backend=backend), proc.stderr[-200:]
