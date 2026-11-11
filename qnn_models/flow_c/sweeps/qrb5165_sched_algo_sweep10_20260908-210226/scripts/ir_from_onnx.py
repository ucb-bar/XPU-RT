#!/usr/bin/env python3
"""Fallback IR door: build a modelblaster-shaped graph.json from an ONNX file.

`modelblaster.pipeline.extract_graph` is the faithful ingest and is what Phase 1
uses. It is also expensive on the larger yolov8_nano rungs -- it runs a
quantization calibration pass per operator, and cost grows with input area, so
the 320 px rung can exceed the build's 3600 s ceiling.

This is the same door the SHIPPED yolov8n binding already comes through:
flow_c/README.md's "Known gaps" says "yolov8n's IR comes through the graph_json
door, not extract_graph". `flowc/ir.py::normalize` maps ONNX `op_type` to the
modelblaster IR op vocabulary through `_OP_TYPE_MAP` precisely so a
graph_json-sourced IR answers registry capability queries the same way a
PyTorch-sourced one does, and the same map is reused here.

What is lost: per-op quantization metadata and shapes. What this sweep needs
from the IR is the op COUNT (the tile's `n_ir_ops`) and the op KINDS (the
registry capability check), and both survive. Any network built this way is
flagged in its manifest and in the compose record, so no result rests on the
two doors being identical.

    python3 ir_from_onnx.py --onnx gen/onnx/x.onnx --out gen/ir/x/int8/graph.json
"""
from __future__ import annotations

import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
sys.path.insert(0, FLOWC)
from flowc.ir import _OP_TYPE_MAP                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quant", default="int8")
    a = ap.parse_args()
    import onnx
    m = onnx.load(a.onnx, load_external_data=False)
    ops, byout, unknown = [], {}, {}
    for i, n in enumerate(m.graph.node):
        for o in n.output:
            byout[o] = i
    for i, n in enumerate(m.graph.node):
        kind = _OP_TYPE_MAP.get(n.op_type)
        if kind is None:
            unknown[n.op_type] = unknown.get(n.op_type, 0) + 1
            kind = "unknown"
        deps = sorted({byout[x] for x in n.input if x in byout})
        ops.append({"name": n.name or f"{n.op_type}_{i}", "op": kind,
                    "op_type": n.op_type, "dispatch_id": i,
                    "depends_on": deps, "hardware_target": "any"})
    doc = {"name": a.name or os.path.splitext(os.path.basename(a.onnx))[0],
           "quant": a.quant, "version": 1,
           "_provenance": ("derived from the ONNX node list by "
                           "scripts/ir_from_onnx.py, NOT by "
                           "modelblaster.pipeline.extract_graph -- the same "
                           "graph_json door the shipped yolov8n binding uses"),
           "ops": ops}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1)
    from collections import Counter
    print(f"{len(ops)} ops -> {a.out}")
    print("  kinds: " + json.dumps(dict(Counter(o["op"] for o in ops))))
    if unknown:
        print(f"  UNMAPPED op_types (kind='unknown', which no backend "
              f"declares, so they block every lane in the capability check): "
              f"{unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
