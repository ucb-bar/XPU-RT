#!/usr/bin/env python3
"""Emit one ONNX model + calibration raws per size point.

Runs INSIDE the qnn-convert container (python3.10 + onnx), so the sweep needs
no ML packages on the host.  Reads a plan produced by sweep.py stage_gen.

  python3.10 genmodels.py /workspace/_plan.json /workspace
"""
import json
import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opzoo import BUILDERS  # noqa: E402


def main(plan_path, work):
    plan = json.load(open(plan_path))
    rng = np.random.default_rng(7)
    made = 0
    for op, params, axis, val, t in plan:
        d = os.path.join(work, t)
        if os.path.exists(os.path.join(d, "m.onnx")):
            continue
        os.makedirs(d, exist_ok=True)
        model, shapes, macs = BUILDERS[op](*params)
        onnx.save(model, os.path.join(d, "m.onnx"))
        # qairt-quantizer wants float32 raws for EVERY input, whatever the
        # declared dtype -- int64 indices included.  Four calibration vectors.
        lines = []
        for k in range(4):
            parts = []
            for nm, shape in shapes.items():
                np.ascontiguousarray(
                    rng.standard_normal(shape), dtype=np.float32
                ).tofile(os.path.join(d, f"{nm}_{k}.raw"))
                parts.append(f"{nm}:=/workspace/{t}/{nm}_{k}.raw")
            lines.append(" ".join(parts))
        open(os.path.join(d, "list.txt"), "w").write("\n".join(lines) + "\n")
        json.dump({"op": op, "params": list(params), "axis": axis, "value": val,
                   "macs": macs, "shapes": {k: list(v) for k, v in shapes.items()}},
                  open(os.path.join(d, "meta.json"), "w"))
        made += 1
    print(f"GENERATED {made} {len(plan)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
