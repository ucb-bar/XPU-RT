"""Parse qnn-profile-viewer per-op text dumps into a cross-backend cost matrix.

Produces ``data/profiled/qnn_cost_matrix.json`` with shape::

    {
      "workload_id": {
        "op_id": {"CPU": float_us, "GPU": float_us, "DSP": float_us},
        ...
      },
      "_meta": {"source_files": [...], "schema_version": "qnn_cost_matrix_v1"},
    }

Missing backend entries are simply absent (not zero) — callers must
treat absence as "unsupported on that backend".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path("/tmp/perop")
OUT_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"

SECTION_START = re.compile(r"^Execute Stats \(Average\):\s*$")
SECTION_END = re.compile(r"^Execute Stats \(Min\):\s*$")
GRAPH_LINE = re.compile(r"^Graph \d+ \(([^)]+)\):\s*$")
# CPU/GPU format: `        _convolution_62: 545 us`
OP_LINE_CG = re.compile(r"^ {8}(\S[^:]*?):\s+(\d+)\s+us\s*$")
# DSP format: `        convolution_0:OpId_1 (us): 1662 us`
OP_LINE_DSP = re.compile(r"^ {8}(\S[^:]*?):OpId_\d+\s+\(us\):\s+(\d+)\s+us\s*$")
SKIP_PREFIXES = ("Backend (",)


def _canonical_op(name: str) -> str:
    # CPU/GPU prefix op names with `_`; DSP does not. Strip leading `_` so
    # they cross-join on the same key. Empty names rejected.
    if name.startswith("_"):
        name = name[1:]
    return name


def parse_execute_avg(text: str) -> tuple[str, dict[str, float]]:
    """Return (graph_name, {op_id: latency_us}) from one profile file."""
    lines = text.splitlines()
    in_section = False
    graph_name = ""
    ops: dict[str, float] = {}
    for line in lines:
        if not in_section:
            if SECTION_START.match(line):
                in_section = True
            continue
        if SECTION_END.match(line):
            break
        m_graph = GRAPH_LINE.match(line)
        if m_graph:
            graph_name = m_graph.group(1)
            continue
        m_op = OP_LINE_DSP.match(line) or OP_LINE_CG.match(line)
        if not m_op:
            continue
        name, val = m_op.group(1), float(m_op.group(2))
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        if name in ("NetRun",):
            continue
        ops[_canonical_op(name)] = val
    return graph_name, ops


def build_matrix() -> dict[str, object]:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    sources: list[str] = []
    for path in sorted(RAW_DIR.glob("perop_*_*.txt")):
        # filename: perop_<workload>_<backend>.txt
        stem = path.stem  # e.g. perop_yolov8n_cpu
        parts = stem.split("_")
        backend = parts[-1].upper()
        workload = "_".join(parts[1:-1])
        text = path.read_text()
        _, ops = parse_execute_avg(text)
        # Graph names diverge across backends (`yolov8n` vs `yolov8n_fp16`)
        # because of per-backend quantization variants. We collapse to the
        # workload-from-filename key so cross-backend cost rows align.
        matrix.setdefault(workload, {})
        for op_id, lat_us in ops.items():
            matrix[workload].setdefault(op_id, {})[backend] = lat_us
        sources.append(str(path))
    matrix_obj: dict[str, object] = dict(matrix)
    matrix_obj["_meta"] = {
        "schema_version": "qnn_cost_matrix_v1",
        "source_files": sources,
        "raw_dir": str(RAW_DIR),
    }
    return matrix_obj


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    OUT_PATH.write_text(json.dumps(matrix, indent=2))
    # Summary
    for wk, ops in matrix.items():
        if wk == "_meta":
            continue
        assert isinstance(ops, dict)
        n_ops = len(ops)
        coverage: dict[str, int] = {"CPU": 0, "GPU": 0, "DSP": 0}
        for op_costs in ops.values():
            for bk in coverage:
                if bk in op_costs:
                    coverage[bk] += 1
        print(f"  {wk}: {n_ops} ops; coverage CPU={coverage['CPU']} GPU={coverage['GPU']} DSP={coverage['DSP']}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
