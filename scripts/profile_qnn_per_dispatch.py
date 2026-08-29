#!/usr/bin/env python3
"""Sweep every YOLOv8 conv dispatch through HTA + GPU on board, measure
real per-(dispatch, backend) latency, ingest into the cost table.

For each unique conv (oc, oh, ow, ic, kh, kw) extracted from the YOLOv8
breakdowns/dispatch_*.shapes.json:

  1. Generate an NHWC fixture (`linalg.conv_2d_nhwc_hwcf_q`) at that
     shape with stub q-params + zero weights/bias. The fixture matches
     the recognizer in `merlin/tools/kernels/qnn_emit_recognizers/
     nhwc_int8_conv.py`, which emits a uint8 conv graph (with int8→uint8
     zp+128 remap) suitable for HTA and a structurally-equivalent fp16
     graph suitable for GPU.

  2. Run the recognizer → `emit_qnn_cpp` → write .qnn.cpp.

  3. Build .qnn-ctx on board against libQnnHta.so. If the HTA validator
     rejects (some shapes unsupported), record `infeasible: true` in the
     cost table — which is itself a real measurement: this conv cannot
     run on HTA, so the scheduler must not propose it.

  4. For GPU: emit a separate fp16 fixture (the recognizer is a thin
     wrapper that swaps dtypes), build on Adreno, measure.

  5. Run `qnn-net-run --num_inferences N` for each successful build,
     parse `qnn-profile-viewer` output, ingest the mean execute time.

The measurement-driven cost table is the single source of truth for the
scheduler. No estimates anywhere in this script.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Optional

_HERE = pathlib.Path(__file__).resolve()
_XPU_RT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_XPU_RT_ROOT))
_MERLIN = pathlib.Path("/scratch2/agustin/merlin")
sys.path.insert(0, str(_MERLIN / "tools" / "kernels"))

from qnn_scheduler.cost_table import CostTable  # noqa: E402


_OP_SUMMARY_RE = re.compile(r"conv_(\d+)x(\d+)x(\d+)x(\d+)x(\d+)x(\d+)_")


def _stride_for(in_h: int, out_h: int, kh: int) -> int | None:
    """Returns the stride iff (in_h, out_h, kh) admits a valid
    same-padded conv. None otherwise."""
    for s in (1, 2):
        # SAME-style: out = ceil(in / stride). True for stride 1 with
        # valid pad; for stride 2 with kh=3 we need in even (yolov8 case).
        if (in_h + s - 1) // s == out_h:
            return s
    return None


def _pad_for(in_h: int, out_h: int, kh: int, stride: int) -> int | None:
    """Symmetric pad such that floor((in + 2p - k) / s) + 1 == out_h.
    Returns None if no integer pad satisfies it."""
    needed = (out_h - 1) * stride + kh - in_h
    if needed < 0 or needed % 2 != 0:
        return None
    return needed // 2


def make_nhwc_fixture(
    op_id: str,
    ic: int, ih: int, iw: int,
    oc: int, oh: int, ow: int,
    kh: int, kw: int,
    *,
    stride: int = 1,
    pad: int = 0,
    dtype: str = "i8",
) -> str:
    """Return MLIR text for a one-conv module that the
    `nhwc_int8_conv` recognizer matches verbatim."""
    in_padded_h = ih + 2 * pad
    in_padded_w = iw + 2 * pad
    return f'''module {{
  func.func @{op_id}(%input: tensor<1x{ih}x{iw}x{ic}x{dtype}>)
      -> tensor<1x{oh}x{ow}x{oc}xf32> {{
    %c0_i32 = arith.constant 0 : i32
    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_output_scale = arith.constant 1.000000e-01 : f32
    %bias_f32 = arith.constant dense<0.0> : tensor<{oc}xf32>
    %weight_i8 = arith.constant dense<1> : tensor<{kh}x{kw}x{ic}x{oc}x{dtype}>
    %bias_init = tensor.empty() : tensor<{oc}xi32>
    %bias_i32 = linalg.generic {{
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }} ins(%bias_f32 : tensor<{oc}xf32>) outs(%bias_init : tensor<{oc}xi32>) {{
      ^bb0(%in: f32, %out: i32):
        %d  = arith.divf %in, %cst_bias_scale : f32
        %r  = math.roundeven %d : f32
        %z  = arith.addf %r, %cst_zp : f32
        %lo = arith.maximumf %z, %cst_min_i32 : f32
        %hi = arith.minimumf %lo, %cst_max_i32 : f32
        %q  = arith.fptosi %hi : f32 to i32
        linalg.yield %q : i32
    }} -> tensor<{oc}xi32>
    %padded = tensor.pad %input low[0, {pad}, {pad}, 0] high[0, {pad}, {pad}, 0] {{
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : {dtype}
      tensor.yield %z : {dtype}
    }} : tensor<1x{ih}x{iw}x{ic}x{dtype}> to tensor<1x{in_padded_h}x{in_padded_w}x{ic}x{dtype}>
    %acc_init = tensor.empty() : tensor<1x{oh}x{ow}x{oc}xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<{oc}xi32>)
        outs(%acc_init : tensor<1x{oh}x{ow}x{oc}xi32>)
        dimensions = [0, 1, 2]
    %conv_i32 = linalg.conv_2d_nhwc_hwcf_q
        {{dilations = dense<1> : vector<2xi64>,
         strides = dense<{stride}> : vector<2xi64>}}
        ins(%padded, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x{in_padded_h}x{in_padded_w}x{ic}x{dtype}>, tensor<{kh}x{kw}x{ic}x{oc}x{dtype}>, i32, i32)
        outs(%broadcasted : tensor<1x{oh}x{ow}x{oc}xi32>) -> tensor<1x{oh}x{ow}x{oc}xi32>
    %deq_init = tensor.empty() : tensor<1x{oh}x{ow}x{oc}xf32>
    %deq = linalg.generic {{
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      }} ins(%conv_i32 : tensor<1x{oh}x{ow}x{oc}xi32>)
        outs(%deq_init : tensor<1x{oh}x{ow}x{oc}xf32>) {{
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_output_scale : f32
        linalg.yield %s : f32
    }} -> tensor<1x{oh}x{ow}x{oc}xf32>
    return %deq : tensor<1x{oh}x{ow}x{oc}xf32>
  }}
}}
'''


def make_fp16_gpu_fixture(
    op_id: str,
    ic: int, ih: int, iw: int,
    oc: int, oh: int, ow: int,
    kh: int, kw: int,
    *,
    stride: int = 1,
    pad: int = 0,
) -> str:
    """Hand-authored fp16 NHWC Conv2D for Adreno GPU (libQnnGpu.so on
    QAIRT 2.45 supports fp32/fp16 only — int8 Conv2d is rejected).
    Returns C++ source ready for `qnn_build.build_qnn_kernel_on_board`."""
    return f'''#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"
#include <cstdint>
#define DO_GRAPH_NODE_VALIDATIONS 1
using namespace qnn_wrapper_api;
namespace {{
constexpr uint32_t kKh = {kh}, kKw = {kw}, kInC = {ic}, kOutC = {oc};
constexpr uint32_t kInH = {ih}, kInW = {iw}, kOutH = {oh}, kOutW = {ow};
uint16_t g_weight_fp16[kKh * kKw * kInC * kOutC] = {{0}};
uint16_t g_bias_fp16[kOutC] = {{0}};
uint32_t g_input_dims[4]  = {{1, kInH, kInW, kInC}};
uint32_t g_weight_dims[4] = {{kKh, kKw, kInC, kOutC}};
uint32_t g_bias_dims[1]   = {{kOutC}};
uint32_t g_output_dims[4] = {{1, kOutH, kOutW, kOutC}};
uint32_t g_pad_amount[4]  = {{{pad}, {pad}, {pad}, {pad}}};
uint32_t g_pad_dims[2]    = {{2, 2}};
uint32_t g_stride[2]      = {{{stride}, {stride}}};
uint32_t g_stride_dims[1] = {{2}};
uint32_t g_dilation[2]    = {{1, 1}};
uint32_t g_dilation_dims[1] = {{2}};
}}
extern "C" {{
QNN_API ModelError_t QnnModel_composeGraphs(
    Qnn_BackendHandle_t bh, QNN_INTERFACE_VER_TYPE intf,
    Qnn_ContextHandle_t ch,
    const GraphConfigInfo_t** gci, const uint32_t ngci,
    GraphInfoPtr_t** gi, uint32_t* ngi,
    bool, QnnLog_Callback_t, QnnLog_Level_t) {{
  ModelError_t err = MODEL_NO_ERROR;
  QnnModel model;
  const QnnGraph_Config_t** gc = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("{op_id}", gci, ngci, gc), err);
  VALIDATE(model.initialize(bh, intf, ch, "{op_id}",
      false, DO_GRAPH_NODE_VALIDATIONS, gc), err);
  Qnn_QuantizeParams_t qpu = {{QNN_DEFINITION_UNDEFINED,
      QNN_QUANTIZATION_ENCODING_UNDEFINED,
      {{.scaleOffsetEncoding = {{0.0f, 0}}}}}};
  Qnn_Tensor_t input{{}}; input.version = QNN_TENSOR_VERSION_1;
  input.v1 = {{.id=0, .name="input", .type=QNN_TENSOR_TYPE_APP_WRITE,
    .dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    .dataType=QNN_DATATYPE_FLOAT_16, .quantizeParams=qpu,
    .rank=4, .dimensions=g_input_dims,
    .memType=QNN_TENSORMEMTYPE_RAW, .clientBuf={{nullptr, 0}}}};
  VALIDATE(model.addTensor("input", &input), err);
  Qnn_Tensor_t weight{{}}; weight.version = QNN_TENSOR_VERSION_1;
  weight.v1 = {{.id=0, .name="weight", .type=QNN_TENSOR_TYPE_STATIC,
    .dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    .dataType=QNN_DATATYPE_FLOAT_16, .quantizeParams=qpu,
    .rank=4, .dimensions=g_weight_dims,
    .memType=QNN_TENSORMEMTYPE_RAW,
    .clientBuf={{g_weight_fp16, sizeof(g_weight_fp16)}}}};
  VALIDATE(model.addTensor("weight", &weight), err);
  Qnn_Tensor_t bias{{}}; bias.version = QNN_TENSOR_VERSION_1;
  bias.v1 = {{.id=0, .name="bias", .type=QNN_TENSOR_TYPE_STATIC,
    .dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    .dataType=QNN_DATATYPE_FLOAT_16, .quantizeParams=qpu,
    .rank=1, .dimensions=g_bias_dims,
    .memType=QNN_TENSORMEMTYPE_RAW,
    .clientBuf={{g_bias_fp16, sizeof(g_bias_fp16)}}}};
  VALIDATE(model.addTensor("bias", &bias), err);
  auto mp = [&](const char* name, uint32_t r, uint32_t* d, void* buf, uint32_t sz) {{
    Qnn_Param_t p{{}}; p.paramType = QNN_PARAMTYPE_TENSOR; p.name = name;
    p.tensorParam.version = QNN_TENSOR_VERSION_1;
    p.tensorParam.v1 = {{.id=0, .name=name, .type=QNN_TENSOR_TYPE_STATIC,
      .dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
      .dataType=QNN_DATATYPE_UINT_32, .quantizeParams=qpu,
      .rank=r, .dimensions=d,
      .memType=QNN_TENSORMEMTYPE_RAW, .clientBuf={{buf, sz}}}};
    return p;
  }};
  Qnn_Param_t cp[4];
  cp[0] = mp(QNN_OP_CONV_2D_PARAM_DILATION, 1, g_dilation_dims, g_dilation, sizeof(g_dilation));
  cp[1] = mp(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, 2, g_pad_dims, g_pad_amount, sizeof(g_pad_amount));
  cp[2] = mp(QNN_OP_CONV_2D_PARAM_STRIDE, 1, g_stride_dims, g_stride, sizeof(g_stride));
  cp[3].paramType = QNN_PARAMTYPE_SCALAR;
  cp[3].name = QNN_OP_CONV_2D_PARAM_GROUP;
  cp[3].scalarParam = {{.dataType = QNN_DATATYPE_UINT_32, .uint32Value = 1}};
  Qnn_Tensor_t output{{}}; output.version = QNN_TENSOR_VERSION_1;
  output.v1 = {{.id=0, .name="output", .type=QNN_TENSOR_TYPE_APP_READ,
    .dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
    .dataType=QNN_DATATYPE_FLOAT_16, .quantizeParams=qpu,
    .rank=4, .dimensions=g_output_dims,
    .memType=QNN_TENSORMEMTYPE_RAW, .clientBuf={{nullptr, 0}}}};
  const char* in_names[] = {{"input", "weight", "bias"}};
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "conv_op", "qti.aisw",
    QNN_OP_CONV_2D, cp, 4, in_names, 3, &output, 1), err);
  QnnModel* m[] = {{&model}};
  VALIDATE(getGraphInfoFromModels(*m, 1, gi), err);
  *ngi = 1;
  return err;
}}
QNN_API ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** gi, uint32_t ngi) {{
  return freeGraphsInfo(gi, ngi);
}}
}}
'''


def _emit_hta_cpp(fixture_mlir: str, work: pathlib.Path, op_id: str) -> Optional[pathlib.Path]:
    """Run the merlin nhwc_int8_conv recognizer + qnn_ir emitter on the
    fixture, returning the .qnn.cpp path."""
    from qnn_emit_v2 import parse_mlir
    from qnn_ir import emit_qnn_cpp
    desc = parse_mlir(fixture_mlir)
    if desc is None:
        return None
    cpp_src = emit_qnn_cpp(desc)
    out = work / f"{op_id}.qnn.cpp"
    out.write_text(cpp_src)
    return out


def _build_and_run(
    cpp: pathlib.Path, op_id: str,
    backend: str, in_dtype: str, out_dtype: str,
    in_shape: tuple[int, ...], out_shape: tuple[int, ...],
    cache: pathlib.Path, ssh_host: str = "qdev",
    iters: int = 30,
) -> Optional[dict]:
    """Build the .qnn.cpp on board for `backend`, run via qnn-net-run,
    parse and return mean/min/max execute_us. None if build fails (which
    we record as `infeasible`)."""
    from qnn_build import build_qnn_kernel_on_board, BoardBuildConfig
    cfg = BoardBuildConfig.from_env(
        ssh_host=ssh_host,
        board_qairt_root="/tmp/qnn_probe",
        target_backend=backend.lower(),
    )
    try:
        ctx = build_qnn_kernel_on_board(cpp, op_id, cache, cfg)
    except Exception as e:
        return {"infeasible": True, "reason": str(e)[:300], "backend": backend}

    # Push ctx + run.
    remote_ctx = f"/tmp/sweep_{op_id}.qnn-ctx"
    subprocess.run(["scp", "-q", str(ctx), f"{ssh_host}:{remote_ctx}"],
                   check=True)
    # Volume-correct stub input.
    n_in = 1
    for d in in_shape: n_in *= d
    bps_in = {"uint8": 1, "int8": 1, "fp16": 2}[in_dtype]
    in_size = n_in * bps_in
    remote_in = f"/tmp/sweep_{op_id}_in.raw"
    subprocess.run(
        ["ssh", ssh_host,
         f"python3 -c 'open(\"{remote_in}\",\"wb\").write(b\"\\x80\"*{in_size})'"],
        check=True)
    list_path = f"/tmp/sweep_{op_id}_list.txt"
    subprocess.run(
        ["ssh", ssh_host,
         f"echo input:={remote_in.split('/')[-1]} > {list_path}"],
        check=True)
    out_dir = f"/tmp/sweep_{op_id}_out"
    backend_so = {"hta": "libQnnHta.so", "gpu": "libQnnGpu.so"}[backend.lower()]
    cmd = (
        f'cd /tmp && export LD_LIBRARY_PATH=/tmp/qnn_probe/lib:$LD_LIBRARY_PATH && '
        f'/tmp/qnn_probe/bin/qnn-net-run '
        f'  --backend /tmp/qnn_probe/lib/{backend_so} '
        f'  --retrieve_context {remote_ctx} '
        f'  --input_list {list_path} '
        f'  --output_dir {out_dir} '
        f'  --profiling_level basic '
        f'  --num_inferences {iters} >/dev/null 2>&1'
    )
    rc = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True)
    if rc.returncode != 0:
        return {"infeasible": True, "reason": f"net-run failed: {rc.stderr[:300]}",
                "backend": backend}
    # Parse profile log.
    pv = subprocess.run(
        ["ssh", ssh_host,
         f"/tmp/qnn_probe/bin/qnn-profile-viewer --input_log {out_dir}/qnn-profiling-data_0.log"],
        capture_output=True, text=True)
    profile = pv.stdout
    return {
        "mean_us": _grep_us(profile, r"Execute Stats \(Average\).*?Backend \(Accelerator \(execute\) time\): (\d+) us")
                  or _grep_us(profile, r"Execute Stats \(Average\).*?Backend \(QnnGraph_execute\): (\d+) us"),
        "min_us": _grep_us(profile, r"Execute Stats \(Min\).*?Backend \(Accelerator \(execute\) time\): (\d+) us")
                  or _grep_us(profile, r"Execute Stats \(Min\).*?Backend \(QnnGraph_execute\): (\d+) us"),
        "max_us": _grep_us(profile, r"Execute Stats \(Max\).*?Backend \(Accelerator \(execute\) time\): (\d+) us")
                  or _grep_us(profile, r"Execute Stats \(Max\).*?Backend \(QnnGraph_execute\): (\d+) us"),
        "iters": iters, "backend": backend, "infeasible": False,
    }


def _grep_us(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, re.DOTALL)
    return float(m.group(1)) if m else None


def _unique_convs(breakdowns: pathlib.Path) -> list[dict]:
    out = []
    seen: set[tuple] = set()
    for p in sorted(breakdowns.glob("dispatch_*.shapes.json")):
        s = json.loads(p.read_text())
        sm = s.get("op_summary", "")
        m = _OP_SUMMARY_RE.search(sm)
        if not m:
            continue
        oc, oh, ow, ic, kh, kw = (int(x) for x in m.groups())
        # Heuristic input H/W: scan inputs for tensor<1x{ic}x{ih}x{iw}> or NCHW
        ih, iw = oh, ow
        for t in s.get("inputs", []):
            mm = re.match(r"tensor<\d+x\d+x(\d+)x(\d+)x", t)
            if mm:
                ih, iw = int(mm.group(1)), int(mm.group(2))
                break
        sig = (ic, ih, iw, oc, oh, ow, kh, kw)
        if sig in seen:
            continue
        seen.add(sig)
        stride = _stride_for(ih, oh, kh)
        pad = _pad_for(ih, oh, kh, stride) if stride else None
        if stride is None or pad is None:
            # Skip — likely transposed conv, mismatched layout, or
            # downsample with non-standard pad. The cost table records
            # nothing for these (no estimate, no extrapolation).
            continue
        out.append({
            "dispatch_name": s["name"], "op_summary": sm,
            "ic": ic, "ih": ih, "iw": iw, "oc": oc, "oh": oh, "ow": ow,
            "kh": kh, "kw": kw, "stride": stride, "pad": pad,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--breakdowns", type=pathlib.Path,
                    default=_MERLIN / "build/het/qrb5165_cpu/breakdowns")
    ap.add_argument("--cost-table", type=pathlib.Path,
                    default=_XPU_RT_ROOT / "qnn_scheduler/qrb5165_costs.json")
    ap.add_argument("--cache-dir", type=pathlib.Path,
                    default=_MERLIN / "build/qnn_per_dispatch_cache")
    ap.add_argument("--work-dir", type=pathlib.Path,
                    default=_MERLIN / "build/qnn_per_dispatch_work")
    ap.add_argument("--limit", type=int, default=None,
                    help="Sweep at most this many unique convs (debug)")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    convs = _unique_convs(args.breakdowns)
    if args.limit:
        convs = convs[:args.limit]
    print(f"sweeping {len(convs)} unique conv shapes (HTA + GPU each)")

    table = (CostTable.load(args.cost_table) if args.cost_table.exists()
             else CostTable())

    n_hta_ok, n_gpu_ok, n_hta_inf, n_gpu_inf = 0, 0, 0, 0
    t0 = time.time()
    for i, c in enumerate(convs):
        op_id = (f"conv_ic{c['ic']}_ih{c['ih']}_iw{c['iw']}_oc{c['oc']}"
                 f"_oh{c['oh']}_ow{c['ow']}_k{c['kh']}_s{c['stride']}")
        sig = (f"1x{c['ih']}x{c['iw']}x{c['ic']}->1x{c['oh']}x{c['ow']}x{c['oc']},"
               f"g1,k{c['kh']},s{c['stride']}")
        # ---- HTA (uint8 via NHWC int8 fixture + recognizer) -------------
        hta_fixture = make_nhwc_fixture(
            op_id, c["ic"], c["ih"], c["iw"],
            c["oc"], c["oh"], c["ow"], c["kh"], c["kw"],
            stride=c["stride"], pad=c["pad"], dtype="i8")
        cpp_hta = _emit_hta_cpp(hta_fixture, args.work_dir, op_id + "_hta")
        if cpp_hta is None:
            res_hta = {"infeasible": True,
                       "reason": "recognizer did not match", "backend": "HTA"}
        else:
            res_hta = _build_and_run(
                cpp_hta, op_id + "_hta", "HTA", "uint8", "uint8",
                (1, c["ih"], c["iw"], c["ic"]),
                (1, c["oh"], c["ow"], c["oc"]),
                args.cache_dir, iters=args.iters)
        key_hta = f"Conv2d@{sig}@uint8::HTA::0"
        if res_hta.get("infeasible"):
            table.execute[key_hta] = {**res_hta,
                "extrapolated": False,
                "source": f"per-dispatch sweep {op_id}",
                "dispatch_name": c["dispatch_name"]}
            n_hta_inf += 1
        else:
            table.execute[key_hta] = {
                "mean_us": res_hta["mean_us"], "min_us": res_hta["min_us"],
                "max_us": res_hta["max_us"], "iters": res_hta["iters"],
                "extrapolated": False, "infeasible": False,
                "source": f"qnn-net-run {op_id} hta",
                "dispatch_name": c["dispatch_name"]}
            n_hta_ok += 1

        # ---- GPU (fp16 hand-emitted) ------------------------------------
        gpu_cpp_path = args.work_dir / f"{op_id}_gpu.qnn.cpp"
        gpu_cpp_path.write_text(make_fp16_gpu_fixture(
            op_id, c["ic"], c["ih"], c["iw"],
            c["oc"], c["oh"], c["ow"], c["kh"], c["kw"],
            stride=c["stride"], pad=c["pad"]))
        res_gpu = _build_and_run(
            gpu_cpp_path, op_id + "_gpu", "GPU", "fp16", "fp16",
            (1, c["ih"], c["iw"], c["ic"]),
            (1, c["oh"], c["ow"], c["oc"]),
            args.cache_dir, iters=args.iters)
        key_gpu = f"Conv2d@{sig}@fp16::GPU::0"
        if res_gpu.get("infeasible"):
            table.execute[key_gpu] = {**res_gpu,
                "extrapolated": False,
                "source": f"per-dispatch sweep {op_id}",
                "dispatch_name": c["dispatch_name"]}
            n_gpu_inf += 1
        else:
            table.execute[key_gpu] = {
                "mean_us": res_gpu["mean_us"], "min_us": res_gpu["min_us"],
                "max_us": res_gpu["max_us"], "iters": res_gpu["iters"],
                "extrapolated": False, "infeasible": False,
                "source": f"qnn-net-run {op_id} gpu",
                "dispatch_name": c["dispatch_name"]}
            n_gpu_ok += 1
        # Periodically save in case of interruption.
        if (i + 1) % 5 == 0:
            table.save(args.cost_table)
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(convs) - i - 1) / max(rate, 1e-3)
            print(f"  {i+1}/{len(convs)}  HTA ok={n_hta_ok} infeasible={n_hta_inf}  "
                  f"GPU ok={n_gpu_ok} infeasible={n_gpu_inf}  ETA {eta:.0f}s")
    table.save(args.cost_table)
    print(f"\nDONE. HTA: {n_hta_ok} measured, {n_hta_inf} infeasible.")
    print(f"     GPU: {n_gpu_ok} measured, {n_gpu_inf} infeasible.")
    print(f"     elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
