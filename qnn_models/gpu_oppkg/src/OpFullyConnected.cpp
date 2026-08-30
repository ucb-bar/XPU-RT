//==============================================================================
//  FlowC GPU op package -- FullyConnected.
//
//  Covers what the stock qti.aisw GPU package does not: QNN_DATATYPE_UFIXED_POINT_8
//  (and SFIXED_POINT_8) activations/weights/bias.  A float path (fp32/fp16) is
//  provided too so the same package can serve a half-precision graph.
//
//  Kernel design, Adreno 650 (2 x SP, 64-wide wave in fp32, 128-wide in fp16,
//  128-bit load/store path, 32 KB L1 per SP, no int8 dot-product instruction):
//
//   * The shapes this package exists for are GEMV, not GEMM: mlp_control runs
//     batch=1 with K,N in {16,256,128,64}.  A tiled GEMM would leave most of
//     the machine idle at these sizes, so the kernel is one work-item per
//     output element, K-loop inside.  Global work size = batch*N, local = 64
//     (one wave; the preferred work-group multiple reported by Adreno's
//     compiler for these kernels).  For N < 64 the local size shrinks to N so
//     we do not launch a partially-masked wave.
//   * The K loop is unrolled by 4 with vload4 + mad24.  vload4 of uchar is a
//     32-bit dword load, which is the natural granularity of Adreno's load
//     unit for byte data; mad24 is a full-rate integer op on Adreno (the full
//     32x32 multiply is quarter-rate), and the operands here are bounded by
//     +-255 so 24 bits is ample.
//   * Accumulation is integer, not float: (q + offset) products are exact in
//     int32 for any K we will see (255*255*K < 2^31 for K < 33000), whereas a
//     float accumulator starts losing integers past 2^24 and would make
//     bit-exact agreement with the CPU reference impossible.
//   * The per-node constants (K, N, offsets, scales) are baked into the kernel
//     source as literals rather than passed as arguments.  The GPU backend
//     compiles each node's program once at graph-finalize, so this costs
//     nothing at run time and lets the compiler fold the K loop bounds and
//     unroll cleanly.
//   * Dequantize/requantize follows QnnTypes.h: real = (q + offset) * scale,
//     q = round(real / scale) - offset.  The bias in a QAIRT-quantized graph
//     carries its *own* scale/offset (it is not pre-scaled by scale_in *
//     scale_w), so it is dequantized to float and added in the float domain --
//     which is exactly what the CPU reference does.
//==============================================================================
#include "OpFullyConnected.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace flowc {

const std::string FullyConnectedOp::s_opType = "FullyConnected";

std::shared_ptr<Operation> FullyConnectedOp::create(const QnnGpuOpPackage_Node_t* node,
                                                    Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<FullyConnectedOp>(new (std::nothrow) FullyConnectedOp(node, status));
}

Qnn_ErrorHandle_t FullyConnectedOp::validate(const Qnn_OpConfig_t& cfg) {
  const uint32_t nIn = cfgNumInputs(cfg);
  if (nIn < 2u || nIn > 3u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const Qnn_DataType_t dt = tDataType(in);

  if (isQuant8(dt)) {
    // All three of input/weight/output must be 8-bit fixed point with a
    // per-tensor scale-offset encoding; per-axis weights are not handled yet.
    if (!isQuant8(tDataType(w)) || !isQuant8(tDataType(out))) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (!quantOf(in).valid || !quantOf(w).valid || !quantOf(out).valid) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    if (nIn == 3u) {
      const Qnn_Tensor_t& b = cfgInput(cfg, 2u);
      if (!isQuant8(tDataType(b)) && tDataType(b) != QNN_DATATYPE_INT_32 &&
          tDataType(b) != QNN_DATATYPE_SFIXED_POINT_32) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
      }
      if (isQuant8(tDataType(b)) && !quantOf(b).valid) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
      }
    }
  } else if (isFloat(dt)) {
    if (tDataType(w) != dt || tDataType(out) != dt) {
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  } else {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  if (tRank(w) != 2u) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  return QNN_SUCCESS;
}

FullyConnectedOp::FullyConnectedOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = QNN_SUCCESS;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);

  Qnn_ErrorHandle_t v = validate(cfg);
  if (v != QNN_SUCCESS) {
    *status = v;
    return;
  }

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const bool hasBias = (cfgNumInputs(cfg) == 3u);

  const uint32_t N = tDims(w)[0];
  const uint32_t K = tDims(w)[1];
  const size_t inElems = tNumElements(in);
  if (K == 0u || inElems % K != 0u) {
    log(QNN_LOG_LEVEL_ERROR, "FlowC FC %s: input elems %zu not divisible by K %u",
        cfgName(cfg), inElems, K);
    *status = QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return;
  }
  const uint32_t batch = (uint32_t)(inElems / K);
  if (tNumElements(out) != (size_t)batch * N) {
    log(QNN_LOG_LEVEL_ERROR, "FlowC FC %s: output elems mismatch", cfgName(cfg));
    *status = QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return;
  }

  // The kernels below index plain __global buffers.  Refuse (loudly) rather
  // than miscompute if the backend handed us image-backed tensors.
  const uint32_t ids[4] = {tId(in), tId(w), hasBias ? tId(cfgInput(cfg, 2u)) : 0u, tId(out)};
  for (int i = 0; i < 4; ++i) {
    if (ids[i] == 0u) continue;
    const QnnGpu_MemoryObject_t* mo = storageOf(node, ids[i]);
    logStorage(cfgName(cfg), i == 3 ? "out" : (i == 0 ? "in" : (i == 1 ? "w" : "b")), mo);
    if (!bufferCompatible(mo)) {
      log(QNN_LOG_LEVEL_ERROR, "FlowC FC %s: tensor id %u is %s, buffer kernels only",
          cfgName(cfg), ids[i], memObjTypeName(mo->type));
      *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
      return;
    }
  }

  // Tell the backend we want our output in a linear buffer (see
  // Operation::claimOutputBuffer -- an unclaimed int8 NATIVE tensor fails
  // allocation in the GPU backend).
  claimOutputBuffer(node, cfg, 0u);

  const bool quantized = isQuant8(tDataType(in));
  if (quantized) {
    buildQuantKernel(cfg, batch, N, K, hasBias, status);
  } else {
    buildFloatKernel(cfg, batch, N, K, hasBias, status);
  }
}

// ---------------------------------------------------------------------------
// int8 path
// ---------------------------------------------------------------------------
void FullyConnectedOp::buildQuantKernel(const Qnn_OpConfig_t& cfg,
                                        uint32_t batch,
                                        uint32_t N,
                                        uint32_t K,
                                        bool hasBias,
                                        Qnn_ErrorHandle_t* status) {
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& w = cfgInput(cfg, 1u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);

  const QuantInfo qi = quantOf(in), qw = quantOf(w), qo = quantOf(out);
  QuantInfo qb;
  bool biasIs8 = false;
  if (hasBias) {
    const Qnn_Tensor_t& b = cfgInput(cfg, 2u);
    qb = quantOf(b);
    biasIs8 = isQuant8(tDataType(b));
  }

  const bool inSigned = isS8(tDataType(in));
  const bool wSigned = isS8(tDataType(w));
  const bool outSigned = isS8(tDataType(out));
  const char* inT = inSigned ? "char" : "uchar";
  const char* wT = wSigned ? "char" : "uchar";
  const char* outT = outSigned ? "char" : "uchar";
  const int outLo = outSigned ? -128 : 0;
  const int outHi = outSigned ? 127 : 255;

  const float scaleProd = qi.scale * qw.scale;
  const float invOutScale = 1.0f / qo.scale;

  const uint32_t kVec = K / 4u;    // number of vload4 iterations
  const uint32_t kTail = K % 4u;   // scalar remainder

  std::ostringstream s;
  s.setf(std::ios::scientific);
  s.precision(9);
  s << "__kernel void flowc_fc_q8(__global const " << inT << "* restrict in,\n"
    << "                         __global const " << wT << "* restrict wt,\n";
  if (hasBias) {
    s << "                         __global const " << (biasIs8 ? "uchar" : "int")
      << "* restrict bs,\n";
  }
  s << "                         __global " << outT << "* restrict out) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << (batch * N) << ") return;\n"
    << "  const int n = gid % " << N << ";\n"
    << "  const int b = gid / " << N << ";\n"
    << "  __global const " << inT << "* restrict irow = in + b * " << K << ";\n"
    << "  __global const " << wT << "* restrict wrow = wt + n * " << K << ";\n"
    << "  int4 acc4 = (int4)(0);\n";
  // FLOWC_FC_MODE selects the K-loop shape.  Kept as a switch because it is
  // the knob that had to be bisected on real hardware: Adreno's compiler
  // miscompiles some int4/mad24 forms at larger K (see README).
  const char* modeEnv = std::getenv("FLOWC_FC_MODE");
  const std::string mode = modeEnv ? modeEnv : "vec_mul";
  if (mode == "probe_w" || mode == "probe_in" || mode == "probe_b") {
    // Diagnostic kernels: copy raw bytes of the weight (or activation) buffer
    // straight to the output so the host can see exactly what the device holds.
    const char* offEnv = std::getenv("FLOWC_PROBE_OFF");
    const int probeOff = offEnv ? atoi(offEnv) : 0;
    std::string expr;
    if (mode == "probe_w") {
      expr = "wt[gid + " + std::to_string(probeOff) + "]";
    } else if (mode == "probe_b") {
      expr = hasBias ? "bs[gid % " + std::to_string(N) + "]" : "(uchar)0";
    } else {
      expr = "in[(gid + " + std::to_string(probeOff) + ") % " + std::to_string(K) + "]";
    }
    std::ostringstream p;
    p << "__kernel void flowc_fc_q8(__global const " << inT << "* restrict in,\n"
      << "                         __global const " << wT << "* restrict wt,\n";
    if (hasBias) p << "                         __global const uchar* restrict bs,\n";
    p << "                         __global " << outT << "* restrict out) {\n"
      << "  const int gid = get_global_id(0);\n"
      << "  if (gid >= " << (batch * N) << ") return;\n"
      << "  out[gid] = " << expr << ";\n}\n";
    m_source = p.str();
    m_name = uniqueKernelName("flowc_fc_q8", m_source);
    m_source.replace(m_source.find("flowc_fc_q8"), std::string("flowc_fc_q8").size(), m_name);
    QnnGpu_Kernel_t pk = QNN_GPU_KERNEL_INIT;
    setWorkSizes(pk, (size_t)batch * N);
    m_args.clear();
    m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
    m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 1u));
    if (hasBias) m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 2u));
    m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
    finishArgs(m_args, m_argPtrs, pk);
    pk.name = m_name.c_str();
    pk.sourceType = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
    pk.kernelSource = m_source.c_str();
    pk.sourceLength = m_source.size();
    pk.buildOptions = "-cl-std=CL2.0";
    m_kernels.push_back(pk);
    *status = QNN_SUCCESS;
    return;
  }
  if (mode == "scalar" || kVec == 0u) {
    s << "  int acc = 0;\n"
      << "  for (int k = 0; k < " << K << "; ++k) {\n"
      << "    acc += ((int)irow[k] + (" << qi.offset << ")) * ((int)wrow[k] + (" << qw.offset
      << "));\n"
      << "  }\n";
  } else if (mode == "mad24") {
    s << "  #pragma unroll 4\n"
      << "  for (int k = 0; k < " << kVec << "; ++k) {\n"
      << "    int4 a = convert_int4(vload4(k, irow)) + (" << qi.offset << ");\n"
      << "    int4 c = convert_int4(vload4(k, wrow)) + (" << qw.offset << ");\n"
      << "    acc4 = mad24(a, c, acc4);\n"
      << "  }\n"
      << "  int acc = acc4.x + acc4.y + acc4.z + acc4.w;\n";
  } else {
    s << "  #pragma unroll 4\n"
      << "  for (int k = 0; k < " << kVec << "; ++k) {\n"
      << "    int4 a = convert_int4(vload4(k, irow)) + (" << qi.offset << ");\n"
      << "    int4 c = convert_int4(vload4(k, wrow)) + (" << qw.offset << ");\n"
      << "    acc4 += a * c;\n"
      << "  }\n"
      << "  int acc = acc4.x + acc4.y + acc4.z + acc4.w;\n";
  }
  if (mode != "scalar" && kVec > 0u) {
    for (uint32_t t = 0u; t < kTail; ++t) {
      const uint32_t idx = kVec * 4u + t;
      s << "  acc += ((int)irow[" << idx << "] + (" << qi.offset << ")) * ((int)wrow[" << idx
        << "] + (" << qw.offset << "));\n";
    }
  }
  s << "  float y = (float)acc * " << scaleProd << "f;\n";
  if (hasBias) {
    if (biasIs8) {
      s << "  y += ((float)((int)bs[n] + (" << qb.offset << "))) * " << qb.scale << "f;\n";
    } else {
      s << "  y += ((float)bs[n]) * " << qb.scale << "f;\n";
    }
  }
  s << "  int q = (int)round(y * " << invOutScale << "f) - (" << qo.offset << ");\n"
    << "  out[gid] = (" << outT << ")clamp(q, " << outLo << ", " << outHi << ");\n"
    << "}\n";

  m_source = s.str();
  m_name = uniqueKernelName("flowc_fc_q8", m_source);
  // rename the entry point inside the source to match
  m_source.replace(m_source.find("flowc_fc_q8"), std::string("flowc_fc_q8").size(), m_name);

  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  setWorkSizes(k, (size_t)batch * N);

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

  log(QNN_LOG_LEVEL_INFO, "FlowC FC(q8) %s: batch=%u N=%u K=%u bias=%d", cfgName(cfg), batch, N,
      K, (int)hasBias);
  *status = QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// float path (fp32 / fp16).  Accumulation is in float even for the half case:
// Adreno 650 runs fp16 arithmetic at 2x rate but a half accumulator overflows
// its 11-bit mantissa within a few dozen terms.
// ---------------------------------------------------------------------------
void FullyConnectedOp::buildFloatKernel(const Qnn_OpConfig_t& cfg,
                                        uint32_t batch,
                                        uint32_t N,
                                        uint32_t K,
                                        bool hasBias,
                                        Qnn_ErrorHandle_t* status) {
  const bool half = (tDataType(cfgInput(cfg, 0u)) == QNN_DATATYPE_FLOAT_16);
  const char* T = half ? "half" : "float";
  const uint32_t kVec = K / 4u;
  const uint32_t kTail = K % 4u;

  std::ostringstream s;
  if (half) s << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  s << "__kernel void flowc_fc_f(__global const " << T << "* restrict in,\n"
    << "                        __global const " << T << "* restrict wt,\n";
  if (hasBias) s << "                        __global const " << T << "* restrict bs,\n";
  s << "                        __global " << T << "* restrict out) {\n"
    << "  const int gid = get_global_id(0);\n"
    << "  if (gid >= " << (batch * N) << ") return;\n"
    << "  const int n = gid % " << N << ";\n"
    << "  const int b = gid / " << N << ";\n"
    << "  __global const " << T << "* restrict irow = in + b * " << K << ";\n"
    << "  __global const " << T << "* restrict wrow = wt + n * " << K << ";\n"
    << "  float4 acc4 = (float4)(0.0f);\n";
  if (kVec > 0u) {
    s << "  #pragma unroll 4\n"
      << "  for (int k = 0; k < " << kVec << "; ++k) {\n"
      << "    float4 a = convert_float4(vload4(k, irow));\n"
      << "    float4 c = convert_float4(vload4(k, wrow));\n"
      << "    acc4 = fma(a, c, acc4);\n"
      << "  }\n";
  }
  s << "  float acc = acc4.x + acc4.y + acc4.z + acc4.w;\n";
  for (uint32_t t = 0u; t < kTail; ++t) {
    const uint32_t idx = kVec * 4u + t;
    s << "  acc += (float)irow[" << idx << "] * (float)wrow[" << idx << "];\n";
  }
  if (hasBias) s << "  acc += (float)bs[n];\n";
  s << "  out[gid] = (" << T << ")acc;\n}\n";

  m_source = s.str();
  m_name = uniqueKernelName("flowc_fc_f", m_source);
  m_source.replace(m_source.find("flowc_fc_f"), std::string("flowc_fc_f").size(), m_name);

  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  setWorkSizes(k, (size_t)batch * N);
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

  log(QNN_LOG_LEVEL_INFO, "FlowC FC(float%s) %s: batch=%u N=%u K=%u", half ? "16" : "32",
      cfgName(cfg), batch, N, K);
  *status = QNN_SUCCESS;
}

void FullyConnectedOp::setWorkSizes(QnnGpu_Kernel_t& k, size_t total) {
  // One wave per work-group where the tensor is big enough for it.  Adreno 650
  // reports a 64-element preferred work-group multiple for these kernels; a
  // group larger than the whole output would launch masked-off lanes.
  size_t local = 64u;
  while (local > 1u && (total % local) != 0u) local >>= 1u;
  if (local > total) local = total;
  const size_t global = ((total + local - 1u) / local) * local;

  k.globalWorkDim = 3u;
  k.globalWorkSizes[0] = global;
  k.globalWorkSizes[1] = 1u;
  k.globalWorkSizes[2] = 1u;
  k.localWorkDim = 3u;
  k.localWorkSizes[0] = local;
  k.localWorkSizes[1] = 1u;
  k.localWorkSizes[2] = 1u;
}

}  // namespace flowc
