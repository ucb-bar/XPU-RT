//==============================================================================
//  FlowC GPU op package -- Conv2d / DepthWiseConv2d at 8-bit fixed point.
//
//  The stock qti.aisw GPU package implements Conv2d for FLOAT_32/FLOAT_16 only,
//  so every int8 CNN (dronet, yolov8n, FusedSensorNet) fails GPU compose at its
//  very first node.  This is that node.
//
//  Layout: QNN hands 4-D activations as NHWC and the filter as
//  [Kh][Kw][Cin/group][Cout], which is the layout an Adreno kernel wants
//  anyway -- the Cout axis is contiguous, so a work item that owns four output
//  channels loads its four filter taps with one 32-bit vload4.
//
//  Kernel design, Adreno 650:
//   * Work decomposition is (Cout/4, Wout, Hout*N).  One work item produces a
//     float4-shaped group of four output channels for one output pixel.  That
//     is the decomposition that makes both operand streams cheap: the filter
//     read is a contiguous dword (vload4 over Cout), and the activation read is
//     scalar but *shared* by the four channels the item owns, so the inner loop
//     does one input load per four MACs instead of one per MAC.
//   * Work-group is 8x8x1 = 64 items = one Adreno wave, and neighbouring items
//     in x share the filter row while neighbours in y share most of the input
//     patch (stride<=2 overlap), so the 32 KB L1 does the reuse a manual
//     __local tile would otherwise have to do.  Local memory is deliberately
//     not used: at these tile sizes the barrier cost outweighs the saved loads,
//     and the driver reports isLocalMemorySupported but with only 32 KB shared
//     across the wave.
//   * Accumulation is int4.  Per-tap products are bounded by 255*255 and the
//     tap count here (Kh*Kw*Cin <= 4608 for these networks) keeps the sum
//     inside int32 with two orders of magnitude to spare, so the result is
//     exactly the integer sum the reference computes -- no float rounding
//     inside the reduction.
//   * Padding is skipped rather than materialised.  For a quantized tensor the
//     pad value is the zero point, i.e. (q + offset) == 0, so an out-of-range
//     tap contributes exactly nothing; skipping it is bit-exact, not an
//     approximation.
//   * Shapes, offsets and scales are baked into the source as literals and the
//     kernel name carries a hash of that source (see uniqueKernelName) because
//     the backend's program cache is keyed on the kernel name.
//==============================================================================
#include "OpConv2d.hpp"

#include <cstdlib>
#include <sstream>

#include "QnnOpDef.h"

namespace flowc {

const std::string Conv2dOp::s_opType = QNN_OP_CONV_2D;
const std::string Conv2dOp::s_opTypeDw = QNN_OP_DEPTH_WISE_CONV_2D;

namespace {
struct ConvShape {
  uint32_t N = 1, H = 0, W = 0, Cin = 0;
  uint32_t Kh = 0, Kw = 0, Cout = 0;
  uint32_t Hout = 0, Wout = 0;
  int32_t strideH = 1, strideW = 1;
  int32_t padT = 0, padL = 0;
  int32_t dilH = 1, dilW = 1;
  int32_t group = 1;
  bool depthwise = false;
};

bool shapeOf(const Qnn_OpConfig_t& cfg, ConvShape* s) {
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (tRank(in) != 4u || tRank(out) != 4u || tRank(w) != 4u) return false;
  const uint32_t* di = tDims(in);
  const uint32_t* dw = tDims(w);
  const uint32_t* dobuf = tDims(out);
  s->N = di[0]; s->H = di[1]; s->W = di[2]; s->Cin = di[3];
  s->Kh = dw[0]; s->Kw = dw[1];
  s->Hout = dobuf[1]; s->Wout = dobuf[2]; s->Cout = dobuf[3];

  std::vector<int32_t> v;
  if (readIntParam(cfg, QNN_OP_CONV_2D_PARAM_STRIDE, &v) && v.size() >= 2u) {
    s->strideH = v[0];
    s->strideW = v[1];
  }
  if (readIntParam(cfg, QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, &v) && v.size() >= 4u) {
    s->padT = v[0];
    s->padL = v[2];
  }
  if (readIntParam(cfg, QNN_OP_CONV_2D_PARAM_DILATION, &v) && v.size() >= 2u) {
    s->dilH = v[0];
    s->dilW = v[1];
  }
  Qnn_Scalar_t sc;
  if (findScalarParam(cfg, QNN_OP_CONV_2D_PARAM_GROUP, &sc)) {
    s->group = (int32_t)scalarAsInt(sc, 1);
  }
  s->depthwise = (std::string(cfgTypeName(cfg)) == QNN_OP_DEPTH_WISE_CONV_2D) ||
                 (s->group > 1 && (uint32_t)s->group == s->Cin && s->Cin == s->Cout);
  return true;
}
}  // namespace

std::shared_ptr<Operation> Conv2dOp::create(const QnnGpuOpPackage_Node_t* node,
                                            Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<Conv2dOp>(new (std::nothrow) Conv2dOp(node, status));
}

Qnn_ErrorHandle_t Conv2dOp::validate(const Qnn_OpConfig_t& cfg) {
  const uint32_t nIn = cfgNumInputs(cfg);
  if (nIn < 2u || nIn > 3u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(in))) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;  // float: stock
  if (!isQuant8(tDataType(w)) || !isQuant8(tDataType(out))) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  // Per-axis (per-channel) weight encodings are not implemented; refuse them
  // rather than quietly using the wrong scale.
  if (!quantOf(in).valid || !quantOf(w).valid || !quantOf(out).valid) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  if (nIn == 3u) {
    const Qnn_Tensor_t& b = cfgInput(cfg, 2u);
    if (!isQuant8(tDataType(b)) || !quantOf(b).valid) {
      return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    }
  }
  ConvShape s;
  if (!shapeOf(cfg, &s)) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  if (s.group != 1 && !s.depthwise) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  const uint32_t* dw = tDims(w);
  if (!s.depthwise && dw[2] != s.Cin) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  return QNN_SUCCESS;
}

Conv2dOp::Conv2dOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = QNN_SUCCESS;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  const Qnn_ErrorHandle_t v = validate(cfg);
  if (v != QNN_SUCCESS) {
    *status = v;
    return;
  }
  ConvShape sh;
  shapeOf(cfg, &sh);

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const bool hasBias = (cfgNumInputs(cfg) == 3u);
  const QuantInfo qi = quantOf(in), qw = quantOf(w), qo = quantOf(out);
  QuantInfo qb;
  if (hasBias) qb = quantOf(cfgInput(cfg, 2u));

  for (uint32_t i = 0u; i < cfgNumInputs(cfg); ++i) {
    const QnnGpu_MemoryObject_t* mo = storageOf(node, tId(cfgInput(cfg, i)));
    logStorage(cfgName(cfg), i == 0 ? "in" : (i == 1 ? "w" : "b"), mo);
    if (!bufferCompatible(mo)) {
      *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
      return;
    }
  }
  logStorage(cfgName(cfg), "out", storageOf(node, tId(out)));
  claimOutputBuffer(node, cfg, 0u);

  const bool inSigned = isS8(tDataType(in));
  const bool wSigned = isS8(tDataType(w));
  const bool outSigned = isS8(tDataType(out));
  const char* inT = inSigned ? "char" : "uchar";
  const char* wT = wSigned ? "char" : "uchar";
  const char* outT = outSigned ? "char" : "uchar";
  const int outLo = outSigned ? -128 : 0;
  const int outHi = outSigned ? 127 : 255;

  const uint32_t cout4 = (sh.Cout + 3u) / 4u;   // channel groups of four
  const bool coutIsMul4 = (sh.Cout % 4u) == 0u;
  const float scaleProd = qi.scale * qw.scale;
  const float invOutScale = 1.0f / qo.scale;

  std::ostringstream s;
  s.setf(std::ios::scientific);
  s.precision(9);
  s << "__kernel void flowc_conv_q8(__global const " << inT << "* restrict src,\n"
    << "                           __global const " << wT << "* restrict wt,\n";
  if (hasBias) s << "                           __global const uchar* restrict bs,\n";
  s << "                           __global " << outT << "* restrict dst) {\n"
    << "  const int cg   = get_global_id(0);   // output-channel group of 4\n"
    << "  const int ow   = get_global_id(1);\n"
    << "  const int ohn  = get_global_id(2);   // oh + Hout*n\n"
    << "  if (cg >= " << cout4 << " || ow >= " << sh.Wout << " || ohn >= "
    << (sh.Hout * sh.N) << ") return;\n"
    << "  const int oh = ohn % " << sh.Hout << ";\n"
    << "  const int n  = ohn / " << sh.Hout << ";\n"
    << "  const int oc = cg * 4;\n"
    << "  int4 acc = (int4)(0);\n"
    << "  const int ih0 = oh * " << sh.strideH << " - " << sh.padT << ";\n"
    << "  const int iw0 = ow * " << sh.strideW << " - " << sh.padL << ";\n";

  if (sh.depthwise) {
    // One input channel feeds one output channel; the four channels a work
    // item owns read four contiguous input bytes, so the activation load is a
    // vload4 as well.
    s << "  for (int kh = 0; kh < " << sh.Kh << "; ++kh) {\n"
      << "    const int ih = ih0 + kh * " << sh.dilH << ";\n"
      << "    if (ih < 0 || ih >= " << sh.H << ") continue;\n"
      << "    for (int kw = 0; kw < " << sh.Kw << "; ++kw) {\n"
      << "      const int iw = iw0 + kw * " << sh.dilW << ";\n"
      << "      if (iw < 0 || iw >= " << sh.W << ") continue;\n"
      << "      const int ibase = ((n * " << sh.H << " + ih) * " << sh.W << " + iw) * "
      << sh.Cin << " + oc;\n"
      << "      const int wbase = ((kh * " << sh.Kw << " + kw) * 1) * " << sh.Cout << " + oc;\n"
      << "      int4 a = convert_int4(vload4(0, src + ibase)) + (" << qi.offset << ");\n"
      << "      int4 c = convert_int4(vload4(0, wt + wbase)) + (" << qw.offset << ");\n"
      << "      acc += a * c;\n"
      << "    }\n  }\n";
  } else {
    s << "  for (int kh = 0; kh < " << sh.Kh << "; ++kh) {\n"
      << "    const int ih = ih0 + kh * " << sh.dilH << ";\n"
      << "    if (ih < 0 || ih >= " << sh.H << ") continue;\n"
      << "    for (int kw = 0; kw < " << sh.Kw << "; ++kw) {\n"
      << "      const int iw = iw0 + kw * " << sh.dilW << ";\n"
      << "      if (iw < 0 || iw >= " << sh.W << ") continue;\n"
      << "      __global const " << inT << "* restrict prow = src + ((n * " << sh.H
      << " + ih) * " << sh.W << " + iw) * " << sh.Cin << ";\n"
      << "      __global const " << wT << "* restrict wrow = wt + ((kh * " << sh.Kw
      << " + kw) * " << sh.Cin << ") * " << sh.Cout << " + oc;\n"
      << "      for (int ci = 0; ci < " << sh.Cin << "; ++ci) {\n"
      << "        const int a = (int)prow[ci] + (" << qi.offset << ");\n"
      << "        int4 c = convert_int4(vload4(0, wrow + ci * " << sh.Cout << ")) + ("
      << qw.offset << ");\n"
      << "        acc += a * c;\n"
      << "      }\n"
      << "    }\n  }\n";
  }

  s << "  float4 y = convert_float4(acc) * " << scaleProd << "f;\n";
  if (hasBias) {
    s << "  int4 bq = convert_int4(vload4(0, bs + oc)) + (" << qb.offset << ");\n"
      << "  y += convert_float4(bq) * " << qb.scale << "f;\n";
  }
  s << "  int4 q = convert_int4(round(y * " << invOutScale << "f)) - (" << qo.offset << ");\n"
    << "  q = clamp(q, " << outLo << ", " << outHi << ");\n"
    << "  const int obase = ((n * " << sh.Hout << " + oh) * " << sh.Wout << " + ow) * "
    << sh.Cout << " + oc;\n";
  if (coutIsMul4) {
    s << "  vstore4(convert_" << (outSigned ? "char4" : "uchar4") << "(q), 0, dst + obase);\n";
  } else {
    // Ragged channel count: write the lanes that exist.
    s << "  const int lanes = min(4, " << sh.Cout << " - oc);\n"
      << "  int qa[4] = {q.x, q.y, q.z, q.w};\n"
      << "  for (int l = 0; l < lanes; ++l) dst[obase + l] = (" << outT << ")qa[l];\n";
  }
  s << "}\n";

  m_source = s.str();
  m_name = uniqueKernelName("flowc_conv_q8", m_source);
  m_source.replace(m_source.find("flowc_conv_q8"), std::string("flowc_conv_q8").size(), m_name);

  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  // 8x8x1 = one 64-wide wave; pad the global size up to a multiple of it.
  size_t lx = 8u, ly = 8u;
  while (lx > 1u && cout4 < lx) lx >>= 1u;
  while (ly > 1u && sh.Wout < ly) ly >>= 1u;
  k.globalWorkDim = 3u;
  k.globalWorkSizes[0] = ((cout4 + lx - 1u) / lx) * lx;
  k.globalWorkSizes[1] = ((sh.Wout + ly - 1u) / ly) * ly;
  k.globalWorkSizes[2] = sh.Hout * sh.N;
  k.localWorkDim = 3u;
  k.localWorkSizes[0] = lx;
  k.localWorkSizes[1] = ly;
  k.localWorkSizes[2] = 1u;

  m_args.clear();
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1u));
  if (hasBias) m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 2u));
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finishArgs(m_args, m_argPtrs, k);
  k.name = m_name.c_str();
  k.sourceType = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  k.kernelSource = m_source.c_str();
  k.sourceLength = m_source.size();
  k.buildOptions = "-cl-std=CL2.0";
  m_kernels.push_back(k);

  log(QNN_LOG_LEVEL_INFO,
      "FlowC Conv2d(q8) %s: %ux%ux%u -> %ux%ux%u K=%ux%u stride=%dx%d pad=%d,%d dw=%d",
      cfgName(cfg), sh.H, sh.W, sh.Cin, sh.Hout, sh.Wout, sh.Cout, sh.Kh, sh.Kw, sh.strideH,
      sh.strideW, sh.padT, sh.padL, (int)sh.depthwise);
  *status = QNN_SUCCESS;
}

}  // namespace flowc
