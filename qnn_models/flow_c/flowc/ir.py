"""Stage 1 — model ingestion.

Two sources, one IR schema:

  pytorch:<model_id>   modelblaster's `extract_graph` on models/<id>.py,
                       exactly as the RISC-V flow ingests it, plus a
                       torch.onnx.export of the *same* nn.Module so the
                       DLC the board runs and the IR the scheduler reasons
                       about come from one definition and one checkpoint.

  graph_json:<path>    an existing graph.json already in modelblaster's IR
                       shape (qnn_models/export_graph_json.py emits this
                       from a compiled DLC).  Used for networks that only
                       exist downstream of the QNN converter — yolov8n's
                       partitioned IR, for one.

Both paths return `{name, quant, ops:[{name, op, dispatch_id, depends_on}]}`
with a dispatch_id space the rest of Flow C indexes into.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap

from . import mb

# op_type (ONNX/QNN route) -> modelblaster IR op vocabulary, so a
# graph_json-sourced IR answers registry capability queries the same way
# a PyTorch-sourced one does.
_OP_TYPE_MAP = {
    "Conv2d": "conv2d_s8", "Conv": "conv2d_s8", "DepthwiseConv2d": "conv2d_s8",
    "Linear": "linear_s8", "Gemm": "linear_s8", "FullyConnected": "linear_s8",
    "MaxPool2d": "maxpool2d", "MaxPool": "maxpool2d",
    "AvgPool2d": "avgpool2d", "GlobalAveragePool": "avgpool2d",
    "Concat": "concat", "Add": "add_s8", "Mul": "mul_s8", "Sub": "sub_s8",
    "Relu": "relu", "Relu6": "relu6", "Sigmoid": "sigmoid_s8", "Tanh": "tanh",
    "Elu": "elu_s8", "Pad": "pad_s8", "Resize": "resize", "Slice": "slice",
    "Softmax": "softmax", "Split": "split", "Transpose": "transpose",
    "Reshape": "reshape", "BatchNorm2d": "batchnorm2d", "Neuron": "relu",
    # QNN-route op_types (what export_graph_json.py reads off a DLC).
    # Eltwise_Binary covers add/mul/sub; ElementWiseNeuron covers the
    # activation family. Both are conservatively mapped to the most
    # restrictive member so a capability check can't wave through an op
    # the backend would reject at compose time.
    "Eltwise_Binary": "mul_s8", "ElementWiseNeuron": "sigmoid_s8",
    "Pool": "maxpool2d", "StridedSlice": "slice",
}


def normalize(ir: dict) -> dict:
    """Coerce an IR from either route into the shape Flow C indexes."""
    ops = []
    for op in ir.get("ops", []):
        kind = op.get("op") or _OP_TYPE_MAP.get(op.get("op_type", ""), "unknown")
        ops.append({
            "name": op.get("name", ""),
            "op": kind,
            "dispatch_id": op.get("dispatch_id"),
            "depends_on": list(op.get("depends_on", [])),
            "hardware_target": op.get("hardware_target", "any"),
        })
    ops = [o for o in ops if o["dispatch_id"] is not None]
    ops.sort(key=lambda o: o["dispatch_id"])
    return {"name": ir.get("name", "unnamed"),
            "quant": ir.get("quant", "int8"),
            "ops": ops}


def from_graph_json(path: str) -> dict:
    with open(path) as f:
        return normalize(json.load(f))


def from_pytorch(model_id: str, out_dir: str, quant: str = "int8",
                 core_registry: str | None = None,
                 num_calibration: int = 1) -> dict:
    """Run modelblaster's extract_graph out-of-process; return the IR."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [mb.mb_python(), "-m", "modelblaster.pipeline.extract_graph",
           "--model", model_id, "--out-dir", out_dir, "--quant", quant,
           "--num-calibration", str(num_calibration)]
    if core_registry:
        cmd += ["--core-registry", core_registry]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(mb.modelblaster_root(), "src") + \
        os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=mb.modelblaster_root(), env=env)
    return from_graph_json(os.path.join(out_dir, "graph.json"))


_ONNX_EXPORT = textwrap.dedent('''
    import importlib, inspect, sys, torch
    model_id, out_path, input_name, opset = sys.argv[1:5]
    mod = importlib.import_module(f"modelblaster.models.{model_id}")
    model = mod.get_model()
    model.eval()
    sample = mod.get_sample_input()
    # Multi-input models (fused_full: front_grey + tof_cross + lowdim) hand
    # back a tuple. Name the inputs after forward()'s parameters so the ONNX
    # graph, the IR's input list and the converter flags all agree.
    if isinstance(sample, (tuple, list)):
        names = [p for p in inspect.signature(model.forward).parameters
                 if p not in ("self",)][:len(sample)]
        args = tuple(sample)
        shapes = [tuple(t.shape) for t in sample]
    else:
        names = [input_name]
        args = (sample,)
        shapes = [tuple(sample.shape)]
    torch.onnx.export(
        model, args, out_path,
        input_names=names, output_names=["output"],
        opset_version=int(opset), do_constant_folding=True,
        dynamo=False)
    print(f"wrote {out_path} from modelblaster.models.{model_id} "
          f"(inputs {dict(zip(names, shapes))})")
''')


def onnx_from_pytorch(model_id: str, out_path: str, input_name: str = "input",
                      opset: int = 17) -> str:
    """Export the same nn.Module the IR came from to ONNX for the QNN
    converter.  opset 17 is the QAIRT 2.45 ceiling (see qnn_models/README)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(mb.modelblaster_root(), "src") + \
        os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([mb.onnx_python(), "-c", _ONNX_EXPORT,
                    model_id, os.path.abspath(out_path), input_name, str(opset)],
                   check=True, cwd=mb.modelblaster_root(), env=env)
    return out_path


def load(spec: str, work_dir: str, quant: str = "int8",
         core_registry: str | None = None) -> dict:
    """`pytorch:<id>` | `graph_json:<path>` | a bare path to a graph.json."""
    if spec.startswith("pytorch:"):
        model_id = spec.split(":", 1)[1]
        out_dir = os.path.join(work_dir, model_id, quant)
        cached = os.path.join(out_dir, "graph.json")
        if os.path.exists(cached) and os.environ.get("FLOWC_REEXTRACT") != "1":
            return from_graph_json(cached)
        return from_pytorch(model_id, out_dir, quant, core_registry)
    if spec.startswith("graph_json:"):
        return from_graph_json(spec.split(":", 1)[1])
    if spec.startswith("onnx:"):
        raise ValueError(
            f"{spec!r}: an onnx-sourced network needs its IR too — point "
            f"`ir.graph_json` at a graph.json emitted by "
            f"qnn_models/export_graph_json.py")
    return from_graph_json(spec)
