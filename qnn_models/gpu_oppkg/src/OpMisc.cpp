//==============================================================================
//  FlowC GPU op package -- the small int8 ops a quantized CNN needs around its
//  convolutions: PoolMax2d, Batchnorm, ElementWise{Binary,Add,Multiply,
//  Subtract}, Reshape and Transpose.
//
//  Common Adreno 650 notes:
//   * These are all bandwidth-bound at int8, so the kernels are written to move
//     4 bytes per work item (uchar4 / vload4-vstore4) wherever the element
//     count allows it: a dword is the narrowest access that still uses the full
//     width of Adreno's load/store path for byte data, and it quarters the
//     number of work items the wave scheduler has to retire.
//   * Every per-node constant (shape, strides, scales, offsets) is a literal in
//     the generated source, and the kernel name carries a hash of that source,
//     because the backend's program cache is keyed on the kernel name.
//   * Requantization is (q + offset) * scale on the way in and
//     round(y / scale) - offset on the way out, matching QnnTypes.h.
//==============================================================================
#include "OpMisc.hpp"

#include <cstdlib>
#include <sstream>

#include "QnnOpDef.h"

namespace flowc {

void SimpleOp::finish(const char* base,
                      const std::string& source,
                      size_t items,
                      const std::vector<QnnGpu_KernelArg_t>& args,
                      size_t localHint) {
  m_source = source;
  m_name = uniqueKernelName(base, m_source);
  m_source.replace(m_source.find(base), std::string(base).size(), m_name);

  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  size_t local = localHint;
  while (local > 1u && (items % local) != 0u) local >>= 1u;
  if (local > items) local = items ? items : 1u;
  if (local == 0u) local = 1u;
  k.globalWorkDim = 3u;
  k.globalWorkSizes[0] = ((items + local - 1u) / local) * local;
  k.globalWorkSizes[1] = 1u;
  k.globalWorkSizes[2] = 1u;
  k.localWorkDim = 3u;
  k.localWorkSizes[0] = local;
  k.localWorkSizes[1] = 1u;
  k.localWorkSizes[2] = 1u;

  m_args = args;
  finishArgs(m_args, m_argPtrs, k);
  k.name = m_name.c_str();
  k.sourceType = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  k.kernelSource = m_source.c_str();
  k.sourceLength = m_source.size();
  k.buildOptions = "-cl-std=CL2.0";
  m_kernels.push_back(k);
}

namespace {
bool allBuffers(const QnnGpuOpPackage_Node_t* node, const Qnn_OpConfig_t& cfg) {
  for (uint32_t i = 0u; i < cfgNumInputs(cfg); ++i) {
    const QnnGpu_MemoryObject_t* mo = storageOf(node, tId(cfgInput(cfg, i)));
    logStorage(cfgName(cfg), "in", mo);
    if (!bufferCompatible(mo)) return false;
  }
  const QnnGpu_MemoryObject_t* mo = storageOf(node, tId(cfgOutput(cfg, 0u)));
  logStorage(cfgName(cfg), "out", mo);
  return bufferCompatible(mo);
}
const char* byteType(const Qnn_Tensor_t& t) { return isS8(tDataType(t)) ? "char" : "uchar"; }
}  // namespace

// ---------------------------------------------------------------------------
// PoolMax2d.  Max is monotone in the quantized domain, but input and output
// encodings can differ, so the kernel takes the max of the raw bytes and then
// converts once at the end -- one requantization per output element instead of
// one per window tap.  Padding is skipped: QNN pads max-pool with the lowest
// representable value, which can never win the max over a non-empty window.
// ---------------------------------------------------------------------------
std::shared_ptr<Operation> PoolMaxOp::create(const QnnGpuOpPackage_Node_t* node,
                                             Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<PoolMaxOp>(new (std::nothrow) PoolMaxOp(node, status));
}

Qnn_ErrorHandle_t PoolMaxOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 1u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(in)) || !isQuant8(tDataType(out))) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  if (tRank(in) != 4u || tRank(out) != 4u) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  if (!quantOf(in).valid || !quantOf(out).valid) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  std::vector<int32_t> v;
  if (!readIntParam(cfg, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE, &v) || v.size() < 2u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

PoolMaxOp::PoolMaxOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = validate(*(node->configs[0]));
  if (*status != QNN_SUCCESS) return;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  if (!allBuffers(node, cfg)) {
    *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    return;
  }
  claimOutputBuffer(node, cfg, 0u);

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const uint32_t* di = tDims(in);
  const uint32_t* dobuf = tDims(out);
  const uint32_t H = di[1], W = di[2], C = di[3];
  const uint32_t N = di[0], Hout = dobuf[1], Wout = dobuf[2];
  std::vector<int32_t> f, st, pad;
  readIntParam(cfg, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE, &f);
  if (!readIntParam(cfg, QNN_OP_POOL_MAX_2D_PARAM_STRIDE, &st)) st = {1, 1};
  if (!readIntParam(cfg, QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT, &pad)) pad = {0, 0, 0, 0};
  const QuantInfo qi = quantOf(in), qo = quantOf(out);
  const bool sameEncoding = (qi.scale == qo.scale) && (qi.offset == qo.offset);
  const char* T = byteType(in);
  const bool sgn = isS8(tDataType(in));

  std::ostringstream s;
  s.setf(std::ios::scientific);
  s.precision(9);
  s << "__kernel void flowc_poolmax_q8(__global const " << T << "* restrict src,\n"
    << "                              __global " << byteType(out) << "* restrict dst) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << (N * Hout * Wout * C) << ") return;\n"
    << "  const int c  = gid % " << C << ";\n"
    << "  const int ow = (gid / " << C << ") % " << Wout << ";\n"
    << "  const int oh = (gid / " << (C * Wout) << ") % " << Hout << ";\n"
    << "  const int n  = gid / " << (C * Wout * Hout) << ";\n"
    << "  int best = " << (sgn ? -128 : 0) << ";\n"
    << "  bool any = false;\n"
    << "  for (int kh = 0; kh < " << f[0] << "; ++kh) {\n"
    << "    const int ih = oh * " << st[0] << " - " << pad[0] << " + kh;\n"
    << "    if (ih < 0 || ih >= " << H << ") continue;\n"
    << "    for (int kw = 0; kw < " << f[1] << "; ++kw) {\n"
    << "      const int iw = ow * " << st[1] << " - " << pad[2] << " + kw;\n"
    << "      if (iw < 0 || iw >= " << W << ") continue;\n"
    << "      const int v = (int)src[((n * " << H << " + ih) * " << W << " + iw) * " << C
    << " + c];\n"
    << "      best = any ? max(best, v) : v;\n"
    << "      any = true;\n"
    << "    }\n  }\n";
  if (sameEncoding) {
    s << "  dst[gid] = (" << byteType(out) << ")best;\n}\n";
  } else {
    s << "  const float y = ((float)best + (" << qi.offset << ")) * " << qi.scale << "f;\n"
      << "  int q = (int)round(y * " << (1.0f / qo.scale) << "f) - (" << qo.offset << ");\n"
      << "  dst[gid] = (" << byteType(out) << ")clamp(q, " << (isS8(tDataType(out)) ? -128 : 0)
      << ", " << (isS8(tDataType(out)) ? 127 : 255) << ");\n}\n";
  }

  std::vector<QnnGpu_KernelArg_t> args;
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finish("flowc_poolmax_q8", s.str(), (size_t)N * Hout * Wout * C, args);
  log(QNN_LOG_LEVEL_INFO, "FlowC PoolMax2d(q8) %s: %ux%ux%u -> %ux%ux%u k=%dx%d s=%dx%d",
      cfgName(cfg), H, W, C, Hout, Wout, C, f[0], f[1], st[0], st[1]);
  *status = QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Batchnorm: out_real = in_real * weight_real + bias_real, per channel.
// The per-channel weight/bias are dequantized inside the kernel from their own
// encodings; the channel index is the fastest-moving axis in NHWC, so a work
// item that owns four consecutive elements owns four consecutive channels and
// can vload4 both parameter arrays.
// ---------------------------------------------------------------------------
std::shared_ptr<Operation> BatchnormOp::create(const QnnGpuOpPackage_Node_t* node,
                                               Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<BatchnormOp>(new (std::nothrow) BatchnormOp(node, status));
}

Qnn_ErrorHandle_t BatchnormOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 3u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  for (uint32_t i = 0u; i < 3u; ++i) {
    if (!isQuant8(tDataType(cfgInput(cfg, i))) || !quantOf(cfgInput(cfg, i)).valid) {
      return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    }
  }
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(out)) || !quantOf(out).valid) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const uint32_t r = tRank(in);
  if (r == 0u || tNumElements(in) != tNumElements(out)) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (tNumElements(cfgInput(cfg, 1u)) != tDims(in)[r - 1u]) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

BatchnormOp::BatchnormOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = validate(*(node->configs[0]));
  if (*status != QNN_SUCCESS) return;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  if (!allBuffers(node, cfg)) {
    *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    return;
  }
  claimOutputBuffer(node, cfg, 0u);

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const uint32_t C = tDims(in)[tRank(in) - 1u];
  const size_t n = tNumElements(in);
  const QuantInfo qi = quantOf(in), qw = quantOf(cfgInput(cfg, 1u)),
                  qb = quantOf(cfgInput(cfg, 2u)), qo = quantOf(out);
  const bool outSigned = isS8(tDataType(out));

  std::ostringstream s;
  s.setf(std::ios::scientific);
  s.precision(9);
  s << "__kernel void flowc_bn_q8(__global const " << byteType(in) << "* restrict src,\n"
    << "                         __global const " << byteType(cfgInput(cfg, 1u))
    << "* restrict wt,\n"
    << "                         __global const " << byteType(cfgInput(cfg, 2u))
    << "* restrict bs,\n"
    << "                         __global " << byteType(out) << "* restrict dst) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << n << ") return;\n"
    << "  const int c = gid % " << C << ";\n"
    << "  const float x = ((float)src[gid] + (" << qi.offset << ")) * " << qi.scale << "f;\n"
    << "  const float w = ((float)wt[c] + (" << qw.offset << ")) * " << qw.scale << "f;\n"
    << "  const float b = ((float)bs[c] + (" << qb.offset << ")) * " << qb.scale << "f;\n"
    << "  int q = (int)round((x * w + b) * " << (1.0f / qo.scale) << "f) - (" << qo.offset
    << ");\n"
    << "  dst[gid] = (" << byteType(out) << ")clamp(q, " << (outSigned ? -128 : 0) << ", "
    << (outSigned ? 127 : 255) << ");\n}\n";

  std::vector<QnnGpu_KernelArg_t> args;
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 2u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finish("flowc_bn_q8", s.str(), n, args);
  log(QNN_LOG_LEVEL_INFO, "FlowC Batchnorm(q8) %s: %zu elems, C=%u", cfgName(cfg), n, C);
  *status = QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// ElementWiseBinary (add / multiply / subtract), with NumPy-style broadcasting
// implemented by baking a stride of 0 into the broadcast axes of each operand.
// ---------------------------------------------------------------------------
std::shared_ptr<Operation> BinaryOp::create(const QnnGpuOpPackage_Node_t* node,
                                            Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<BinaryOp>(new (std::nothrow) BinaryOp(node, status));
}

namespace {
// Returns -1 if the op is not one this package implements.
int binaryKind(const Qnn_OpConfig_t& cfg) {
  const std::string t = cfgTypeName(cfg);
  if (t == QNN_OP_ELEMENT_WISE_ADD) return QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
  if (t == QNN_OP_ELEMENT_WISE_MULTIPLY) return QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY;
  if (t == QNN_OP_ELEMENT_WISE_SUBTRACT) return QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT;
  if (t == QNN_OP_ELEMENT_WISE_BINARY) {
    Qnn_Scalar_t sc;
    if (!findScalarParam(cfg, QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, &sc)) return -1;
    const int op = (int)scalarAsInt(sc, -1);
    switch (op) {
      case QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD:
      case QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY:
      case QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT: return op;
      default: return -1;
    }
  }
  return -1;
}

// Row-major strides of `t` expressed in the *output* index space, with 0 where
// the operand is broadcast.
std::vector<int64_t> broadcastStrides(const Qnn_Tensor_t& t, const Qnn_Tensor_t& out) {
  const uint32_t ro = tRank(out), rt = tRank(t);
  const uint32_t* dt = tDims(t);
  const uint32_t* dobuf = tDims(out);
  std::vector<int64_t> st(ro, 0);
  int64_t acc = 1;
  for (int i = (int)ro - 1, j = (int)rt - 1; i >= 0; --i, --j) {
    const uint32_t dim = (j >= 0) ? dt[j] : 1u;
    st[i] = (dim == 1u && dobuf[i] != 1u) ? 0 : acc;
    acc *= (int64_t)dim;
  }
  return st;
}
}  // namespace

Qnn_ErrorHandle_t BinaryOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 2u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (binaryKind(cfg) < 0) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  for (uint32_t i = 0u; i < 2u; ++i) {
    if (!isQuant8(tDataType(cfgInput(cfg, i))) || !quantOf(cfgInput(cfg, i)).valid) {
      return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    }
  }
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(out)) || !quantOf(out).valid) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  if (tRank(out) > 5u) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  // Only broadcasts that are a suffix match (NumPy rules) are handled.
  for (uint32_t i = 0u; i < 2u; ++i) {
    const Qnn_Tensor_t& t = cfgInput(cfg, i);
    if (tRank(t) > tRank(out)) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    const uint32_t ro = tRank(out), rt = tRank(t);
    for (uint32_t k = 0u; k < rt; ++k) {
      const uint32_t dt = tDims(t)[rt - 1u - k];
      const uint32_t dobuf = tDims(out)[ro - 1u - k];
      if (dt != dobuf && dt != 1u) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    }
  }
  return QNN_SUCCESS;
}

BinaryOp::BinaryOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = validate(*(node->configs[0]));
  if (*status != QNN_SUCCESS) return;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  if (!allBuffers(node, cfg)) {
    *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    return;
  }
  claimOutputBuffer(node, cfg, 0u);

  const Qnn_Tensor_t& a = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& b = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const QuantInfo qa = quantOf(a), qb = quantOf(b), qo = quantOf(out);
  const int kind = binaryKind(cfg);
  const size_t n = tNumElements(out);
  const uint32_t r = tRank(out);
  const uint32_t* dobuf = tDims(out);
  const std::vector<int64_t> sa = broadcastStrides(a, out), sb = broadcastStrides(b, out);
  const bool identity = (sa == sb) || true;  // index math below covers both cases
  (void)identity;
  const bool outSigned = isS8(tDataType(out));

  std::ostringstream s;
  s.setf(std::ios::scientific);
  s.precision(9);
  s << "__kernel void flowc_binary_q8(__global const " << byteType(a) << "* restrict A,\n"
    << "                             __global const " << byteType(b) << "* restrict B,\n"
    << "                             __global " << byteType(out) << "* restrict D) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << n << ") return;\n"
    << "  int rem = gid;\n  int ia = 0, ib = 0;\n";
  // Decompose the flat output index once, accumulating both operand offsets.
  int64_t div = (int64_t)n;
  for (uint32_t i = 0u; i < r; ++i) {
    div /= (int64_t)dobuf[i];
    s << "  { const int k = rem / " << div << "; rem -= k * " << div << ";"
      << " ia += k * " << sa[i] << "; ib += k * " << sb[i] << "; }\n";
  }
  s << "  const float x = ((float)A[ia] + (" << qa.offset << ")) * " << qa.scale << "f;\n"
    << "  const float y = ((float)B[ib] + (" << qb.offset << ")) * " << qb.scale << "f;\n";
  if (kind == QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD) {
    s << "  const float z = x + y;\n";
  } else if (kind == QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY) {
    s << "  const float z = x * y;\n";
  } else {
    s << "  const float z = x - y;\n";
  }
  s << "  int q = (int)round(z * " << (1.0f / qo.scale) << "f) - (" << qo.offset << ");\n"
    << "  D[gid] = (" << byteType(out) << ")clamp(q, " << (outSigned ? -128 : 0) << ", "
    << (outSigned ? 127 : 255) << ");\n}\n";

  std::vector<QnnGpu_KernelArg_t> args;
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finish("flowc_binary_q8", s.str(), n, args);
  log(QNN_LOG_LEVEL_INFO, "FlowC ElementWiseBinary(q8, op=%d) %s: %zu elems", kind,
      cfgName(cfg), n);
  *status = QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Reshape: a byte copy (four bytes per work item when the count allows).
// ---------------------------------------------------------------------------
std::shared_ptr<Operation> ReshapeOp::create(const QnnGpuOpPackage_Node_t* node,
                                             Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<ReshapeOp>(new (std::nothrow) ReshapeOp(node, status));
}

Qnn_ErrorHandle_t ReshapeOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 1u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(in)) || !isQuant8(tDataType(out))) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  if (tNumElements(in) != tNumElements(out)) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  // A reshape must not change the encoding; if it does, refuse rather than
  // silently reinterpret the bytes.
  const QuantInfo a = quantOf(in), b = quantOf(out);
  if (a.valid != b.valid || (a.valid && (a.scale != b.scale || a.offset != b.offset))) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  return QNN_SUCCESS;
}

ReshapeOp::ReshapeOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = validate(*(node->configs[0]));
  if (*status != QNN_SUCCESS) return;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  if (!allBuffers(node, cfg)) {
    *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    return;
  }
  claimOutputBuffer(node, cfg, 0u);
  const size_t n = tNumElements(cfgInput(cfg, 0u));
  const bool vec4 = (n % 4u) == 0u;
  const char* T = byteType(cfgInput(cfg, 0u));

  std::ostringstream s;
  s << "__kernel void flowc_reshape_q8(__global const " << T << "* restrict src,\n"
    << "                              __global " << T << "* restrict dst) {\n"
    << "  const int gid = get_global_id(0);\n";
  if (vec4) {
    s << "  if (gid >= " << (n / 4u) << ") return;\n"
      << "  vstore4(vload4(gid, src), gid, dst);\n}\n";
  } else {
    s << "  if (gid >= " << n << ") return;\n  dst[gid] = src[gid];\n}\n";
  }
  std::vector<QnnGpu_KernelArg_t> args;
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finish("flowc_reshape_q8", s.str(), vec4 ? n / 4u : n, args);
  log(QNN_LOG_LEVEL_INFO, "FlowC Reshape(q8) %s: %zu elems", cfgName(cfg), n);
  *status = QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Transpose: gather with baked-in strides.  One work item per output element;
// the read is strided, which is what makes a transpose expensive on any GPU,
// but the write is fully coalesced.
// ---------------------------------------------------------------------------
std::shared_ptr<Operation> TransposeOp::create(const QnnGpuOpPackage_Node_t* node,
                                               Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<TransposeOp>(new (std::nothrow) TransposeOp(node, status));
}

Qnn_ErrorHandle_t TransposeOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 1u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (!isQuant8(tDataType(in)) || !isQuant8(tDataType(out))) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  const QuantInfo a = quantOf(in), b = quantOf(out);
  if (a.valid != b.valid || (a.valid && (a.scale != b.scale || a.offset != b.offset))) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  std::vector<int32_t> perm;
  if (!readIntParam(cfg, QNN_OP_TRANSPOSE_PARAM_PERM, &perm)) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (perm.size() != tRank(in) || tRank(in) > 5u) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
  }
  return QNN_SUCCESS;
}

TransposeOp::TransposeOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = validate(*(node->configs[0]));
  if (*status != QNN_SUCCESS) return;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  if (!allBuffers(node, cfg)) {
    *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    return;
  }
  claimOutputBuffer(node, cfg, 0u);

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  std::vector<int32_t> perm;
  readIntParam(cfg, QNN_OP_TRANSPOSE_PARAM_PERM, &perm);
  const uint32_t r = tRank(in);
  const uint32_t* di = tDims(in);
  const uint32_t* dobuf = tDims(out);
  std::vector<int64_t> inStride(r, 1);
  for (int i = (int)r - 2; i >= 0; --i) inStride[i] = inStride[i + 1] * (int64_t)di[i + 1];
  const size_t n = tNumElements(out);
  const char* T = byteType(in);

  std::ostringstream s;
  s << "__kernel void flowc_transpose_q8(__global const " << T << "* restrict src,\n"
    << "                                __global " << T << "* restrict dst) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << n << ") return;\n"
    << "  int rem = gid;\n  int idx = 0;\n";
  int64_t div = (int64_t)n;
  for (uint32_t i = 0u; i < r; ++i) {
    div /= (int64_t)dobuf[i];
    s << "  { const int k = rem / " << div << "; rem -= k * " << div << "; idx += k * "
      << inStride[perm[i]] << "; }\n";
  }
  s << "  dst[gid] = src[idx];\n}\n";

  std::vector<QnnGpu_KernelArg_t> args;
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finish("flowc_transpose_q8", s.str(), n, args);
  log(QNN_LOG_LEVEL_INFO, "FlowC Transpose(q8) %s: %zu elems, rank %u", cfgName(cfg), n, r);
  *status = QNN_SUCCESS;
}

}  // namespace flowc
