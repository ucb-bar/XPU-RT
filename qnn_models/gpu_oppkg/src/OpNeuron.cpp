//==============================================================================
//  FlowC GPU op package -- pointwise activations (ElementWiseNeuron & friends).
//
//  Kernel design, Adreno 650:
//
//   * For 8-bit fixed point the activation is a *function of a byte*, so there
//     are only 256 possible outputs.  Instead of evaluating exp/tanh per work
//     item, the host evaluates the activation 256 times in double precision at
//     graph-build time and bakes the result into the kernel source as a
//     __constant lookup table.  On Adreno __constant lives in the (cached)
//     uniform/constant path, so the whole table stays resident; the kernel then
//     costs one byte load, one table read and one byte store per element, with
//     no transcendental ALU work at all.  This is why one kernel covers ELU,
//     RELU, SIGMOID, TANH, HARD_SWISH, RELU_MIN_MAX ... identically: only the
//     table changes.  It also makes the int8 result *exactly* the correctly
//     rounded value of the reference formula, since the table is built in
//     double precision on the CPU.
//   * Work items process uchar4 (a 32-bit dword, Adreno's natural byte load
//     granularity) when the element count is a multiple of 4, which it is for
//     every activation tensor in these four networks; otherwise a scalar
//     variant is emitted.  Local size 64 = one wave.
//   * The float path keeps the arithmetic in the tensor's own precision but
//     evaluates in float for fp16 inputs (half exp() on Adreno is fast but the
//     accuracy is not worth the risk for an activation feeding a conv).
//==============================================================================
#include "OpNeuron.hpp"

#include <cmath>
#include <cstdlib>
#include <sstream>

#include "QnnOpDef.h"

namespace flowc {
namespace {

// Neuron operation codes, from QnnOpDef.h.
enum NeuronKind {
  N_ELU = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU,
  N_GELU = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU,
  N_HARD_SIGMOID = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID,
  N_HARD_SWISH = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH,
  N_RELU = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU,
  N_RELU_MIN_MAX = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
  N_SIGMOID = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID,
  N_SOFTPLUS = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS,
  N_TANH = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH,
};

struct NeuronSpec {
  int kind = N_RELU;
  double alpha = 1.0;
  double beta = 1.0;
  double minValue = 0.0;
  double maxValue = 0.0;
  double threshold = 20.0;
  bool ok = false;
};

NeuronSpec specOf(const Qnn_OpConfig_t& cfg) {
  NeuronSpec s;
  const std::string type = cfgTypeName(cfg);
  Qnn_Scalar_t sc;

  if (type == QNN_OP_ELEMENT_WISE_NEURON) {
    if (!findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, &sc)) return s;
    s.kind = (int)scalarAsInt(sc, -1);
    s.alpha = (s.kind == N_ELU || s.kind == N_HARD_SIGMOID) ? 1.0 : 1.0;
  } else if (type == QNN_OP_RELU) {
    s.kind = N_RELU;
  } else if (type == QNN_OP_RELU1) {
    s.kind = N_RELU_MIN_MAX;
    s.minValue = -1.0;
    s.maxValue = 1.0;
  } else if (type == QNN_OP_RELU6) {
    s.kind = N_RELU_MIN_MAX;
    s.minValue = 0.0;
    s.maxValue = 6.0;
  } else if (type == QNN_OP_RELU_MIN_MAX) {
    s.kind = N_RELU_MIN_MAX;
  } else if (type == QNN_OP_SIGMOID) {
    s.kind = N_SIGMOID;
  } else if (type == QNN_OP_TANH) {
    s.kind = N_TANH;
  } else if (type == QNN_OP_ELU) {
    s.kind = N_ELU;
  } else if (type == QNN_OP_HARD_SWISH) {
    s.kind = N_HARD_SWISH;
  } else {
    return s;
  }

  if (findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA, &sc)) {
    s.alpha = scalarAsFloat(sc, 1.0f);
  }
  if (findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA, &sc)) {
    s.beta = scalarAsFloat(sc, 1.0f);
  }
  if (findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE, &sc)) {
    s.minValue = scalarAsFloat(sc, 0.0f);
  }
  if (findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE, &sc)) {
    s.maxValue = scalarAsFloat(sc, 0.0f);
  }
  if (findScalarParam(cfg, QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD, &sc)) {
    s.threshold = scalarAsFloat(sc, 20.0f);
  }

  // Only the activations whose definition is unambiguous (and which these four
  // networks actually use) are claimed.  GELU and SOFTPLUS have several
  // in-the-wild definitions; refusing them is better than a silent mismatch.
  switch (s.kind) {
    case N_ELU:
    case N_RELU:
    case N_RELU_MIN_MAX:
    case N_SIGMOID:
    case N_TANH:
    case N_HARD_SWISH:
    case N_HARD_SIGMOID:
      s.ok = true;
      break;
    default:
      s.ok = false;
  }
  return s;
}

double evalNeuron(double x, const NeuronSpec& s) {
  switch (s.kind) {
    case N_ELU: return x >= 0.0 ? x : s.alpha * std::expm1(x);
    case N_RELU: return x > 0.0 ? x : 0.0;
    case N_RELU_MIN_MAX: return std::min(std::max(x, s.minValue), s.maxValue);
    case N_SIGMOID: return 1.0 / (1.0 + std::exp(-x));
    case N_TANH: return std::tanh(x);
    case N_HARD_SWISH: return x * std::min(std::max(x / 6.0 + 0.5, 0.0), 1.0);
    case N_HARD_SIGMOID: return std::min(std::max(s.alpha * x + s.beta, 0.0), 1.0);
    default: return x;
  }
}

// Float-domain OpenCL expression for the same activation.
std::string floatExpr(const NeuronSpec& s, const std::string& v) {
  std::ostringstream e;
  switch (s.kind) {
    case N_ELU:
      e << "(" << v << " >= 0.0f ? " << v << " : " << litf(s.alpha) << " * expm1(" << v << "))";
      break;
    case N_RELU: e << "fmax(" << v << ", 0.0f)"; break;
    case N_RELU_MIN_MAX:
      e << "clamp(" << v << ", " << litf(s.minValue) << ", " << litf(s.maxValue) << ")";
      break;
    case N_SIGMOID: e << "(1.0f / (1.0f + exp(-" << v << ")))"; break;
    case N_TANH: e << "tanh(" << v << ")"; break;
    case N_HARD_SWISH:
      e << "(" << v << " * clamp(" << v << " * 0.16666667f + 0.5f, 0.0f, 1.0f))";
      break;
    case N_HARD_SIGMOID:
      e << "clamp(" << litf(s.alpha) << " * " << v << " + " << litf(s.beta)
        << ", 0.0f, 1.0f)";
      break;
    default: e << v;
  }
  return e.str();
}

}  // namespace

bool NeuronOp::handles(const std::string& t) {
  return t == QNN_OP_ELEMENT_WISE_NEURON || t == QNN_OP_RELU || t == QNN_OP_RELU1 ||
         t == QNN_OP_RELU6 || t == QNN_OP_RELU_MIN_MAX || t == QNN_OP_SIGMOID ||
         t == QNN_OP_TANH || t == QNN_OP_ELU || t == QNN_OP_HARD_SWISH;
}

std::shared_ptr<Operation> NeuronOp::create(const QnnGpuOpPackage_Node_t* node,
                                            Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<NeuronOp>(new (std::nothrow) NeuronOp(node, status));
}

Qnn_ErrorHandle_t NeuronOp::validate(const Qnn_OpConfig_t& cfg) {
  if (cfgNumInputs(cfg) != 1u || cfgNumOutputs(cfg) != 1u) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  const NeuronSpec s = specOf(cfg);
  if (!s.ok) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  if (tNumElements(in) != tNumElements(out)) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;

  const Qnn_DataType_t dt = tDataType(in);
  if (isQuant8(dt)) {
    if (!isQuant8(tDataType(out))) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (!quantOf(in).valid || !quantOf(out).valid) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return QNN_SUCCESS;
  }
  if (isFloat(dt) && tDataType(out) == dt) return QNN_SUCCESS;
  return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
}

NeuronOp::NeuronOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  *status = QNN_SUCCESS;
  const Qnn_OpConfig_t& cfg = *(node->configs[0]);
  Qnn_ErrorHandle_t v = validate(cfg);
  if (v != QNN_SUCCESS) {
    *status = v;
    return;
  }

  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const uint32_t ids[2] = {tId(in), tId(out)};
  for (int i = 0; i < 2; ++i) {
    const QnnGpu_MemoryObject_t* mo = storageOf(node, ids[i]);
    logStorage(cfgName(cfg), i == 0 ? "in" : "out", mo);
    if (!bufferCompatible(mo)) {
      log(QNN_LOG_LEVEL_ERROR, "FlowC Neuron %s: tensor id %u is %s, buffer kernels only",
          cfgName(cfg), ids[i], memObjTypeName(mo->type));
      *status = QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
      return;
    }
  }

  claimOutputBuffer(node, cfg, 0u);

  const size_t n = tNumElements(in);
  if (isQuant8(tDataType(in))) {
    buildLutKernel(cfg, n, status);
  } else {
    buildFloatKernel(cfg, n, status);
  }
}

void NeuronOp::buildLutKernel(const Qnn_OpConfig_t& cfg,
                              size_t numElements,
                              Qnn_ErrorHandle_t* status) {
  const Qnn_Tensor_t& in = cfgInput(cfg, 0u);
  const Qnn_Tensor_t& out = cfgOutput(cfg, 0u);
  const NeuronSpec spec = specOf(cfg);
  const QuantInfo qi = quantOf(in), qo = quantOf(out);
  const bool inSigned = isS8(tDataType(in));
  const bool outSigned = isS8(tDataType(out));
  const int outLo = outSigned ? -128 : 0;
  const int outHi = outSigned ? 127 : 255;

  // Build the 256-entry table.  Index is the *raw byte*, so for signed input
  // the byte b maps to the quantized value (int8)b.
  int table[256];
  for (int b = 0; b < 256; ++b) {
    const int q = inSigned ? (int8_t)b : b;
    const double x = ((double)q + (double)qi.offset) * (double)qi.scale;
    const double y = evalNeuron(x, spec);
    double qy = std::round(y / (double)qo.scale) - (double)qo.offset;
    if (qy < outLo) qy = outLo;
    if (qy > outHi) qy = outHi;
    table[b] = (int)qy;
  }

  const bool vec4 = (numElements % 4u) == 0u;
  const char* inT = inSigned ? "char" : "uchar";
  const char* outT = outSigned ? "char" : "uchar";

  // Debug hook: FLOWC_DEBUG_CONST makes the activation write a constant byte.
  // Used to prove whether a downstream op is actually reading this op's output
  // buffer (data-flow) or stale memory.
  const char* dbgConst = std::getenv("FLOWC_DEBUG_CONST");
  if (dbgConst) {
    for (int b = 0; b < 256; ++b) table[b] = atoi(dbgConst);
  }

  std::ostringstream s;
  s << "__constant " << outT << " LUT[256] = {";
  for (int b = 0; b < 256; ++b) {
    s << table[b] << (b == 255 ? "" : ",");
    if ((b & 31) == 31) s << "\n";
  }
  s << "};\n";
  if (vec4) {
    s << "__kernel void flowc_neuron_q8(__global const " << inT << "* restrict in,\n"
      << "                             __global " << outT << "* restrict out) {\n"
      << "  const int gid = get_global_id(0);\n"
      << "  if (gid >= " << (numElements / 4u) << ") return;\n"
      << "  " << (inSigned ? "char4" : "uchar4") << " v = vload4(gid, in);\n"
      << "  " << (outSigned ? "char4" : "uchar4") << " r;\n"
      << "  r.x = LUT[(uchar)v.x];\n  r.y = LUT[(uchar)v.y];\n"
      << "  r.z = LUT[(uchar)v.z];\n  r.w = LUT[(uchar)v.w];\n"
      << "  vstore4(r, gid, out);\n}\n";
  } else {
    s << "__kernel void flowc_neuron_q8(__global const " << inT << "* restrict in,\n"
      << "                             __global " << outT << "* restrict out) {\n"
      << "  const int gid = get_global_id(0);\n"
      << "  if (gid >= " << numElements << ") return;\n"
      << "  out[gid] = LUT[(uchar)in[gid]];\n}\n";
  }

  m_source = s.str();
  m_name = uniqueKernelName("flowc_neuron_q8", m_source);
  m_source.replace(m_source.find("flowc_neuron_q8"), std::string("flowc_neuron_q8").size(), m_name);

  const size_t items = vec4 ? numElements / 4u : numElements;
  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  size_t local = 64u;
  while (local > 1u && (items % local) != 0u) local >>= 1u;
  if (local > items) local = items;
  k.globalWorkDim = 3u;
  k.globalWorkSizes[0] = ((items + local - 1u) / local) * local;
  k.globalWorkSizes[1] = 1u;
  k.globalWorkSizes[2] = 1u;
  k.localWorkDim = 3u;
  k.localWorkSizes[0] = local;
  k.localWorkSizes[1] = 1u;
  k.localWorkSizes[2] = 1u;

  m_args.clear();
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finishArgs(m_args, m_argPtrs, k);
  k.name = m_name.c_str();
  k.sourceType = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  k.kernelSource = m_source.c_str();
  k.sourceLength = m_source.size();
  k.buildOptions = "-cl-std=CL2.0";
  m_kernels.push_back(k);

  log(QNN_LOG_LEVEL_INFO, "FlowC Neuron(q8 LUT) %s: kind=%d elems=%zu vec4=%d", cfgName(cfg),
      spec.kind, numElements, (int)vec4);
  *status = QNN_SUCCESS;
}

void NeuronOp::buildFloatKernel(const Qnn_OpConfig_t& cfg,
                                size_t numElements,
                                Qnn_ErrorHandle_t* status) {
  const NeuronSpec spec = specOf(cfg);
  const bool half = (tDataType(cfgInput(cfg, 0u)) == QNN_DATATYPE_FLOAT_16);
  const char* T = half ? "half" : "float";
  const bool vec4 = (numElements % 4u) == 0u;

  std::ostringstream s;
  if (half) s << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  s << "__kernel void flowc_neuron_f(__global const " << T << "* restrict in,\n"
    << "                            __global " << T << "* restrict out) {\n"
    << "  const int gid = get_global_id(0);\n";
  if (vec4) {
    s << "  if (gid >= " << (numElements / 4u) << ") return;\n"
      << "  float4 v = convert_float4(vload4(gid, in));\n"
      << "  float4 r;\n"
      << "  r.x = " << floatExpr(spec, "v.x") << ";\n"
      << "  r.y = " << floatExpr(spec, "v.y") << ";\n"
      << "  r.z = " << floatExpr(spec, "v.z") << ";\n"
      << "  r.w = " << floatExpr(spec, "v.w") << ";\n"
      << "  vstore4(convert_" << (half ? "half4" : "float4") << "(r), gid, out);\n}\n";
  } else {
    s << "  if (gid >= " << numElements << ") return;\n"
      << "  float v = (float)in[gid];\n"
      << "  out[gid] = (" << T << ")(" << floatExpr(spec, "v") << ");\n}\n";
  }

  m_source = s.str();
  m_name = uniqueKernelName("flowc_neuron_f", m_source);
  m_source.replace(m_source.find("flowc_neuron_f"), std::string("flowc_neuron_f").size(), m_name);
  const size_t items = vec4 ? numElements / 4u : numElements;

  QnnGpu_Kernel_t k = QNN_GPU_KERNEL_INIT;
  size_t local = 64u;
  while (local > 1u && (items % local) != 0u) local >>= 1u;
  if (local > items) local = items;
  k.globalWorkDim = 3u;
  k.globalWorkSizes[0] = ((items + local - 1u) / local) * local;
  k.globalWorkSizes[1] = 1u;
  k.globalWorkSizes[2] = 1u;
  k.localWorkDim = 3u;
  k.localWorkSizes[0] = local;
  k.localWorkSizes[1] = 1u;
  k.localWorkSizes[2] = 1u;

  m_args.clear();
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_INPUT_READ, 0u));
  m_args.push_back(tensorArg(QNN_GPU_KERNEL_ARG_TYPE_OP_OUTPUT_WRITE, 0u));
  finishArgs(m_args, m_argPtrs, k);
  k.name = m_name.c_str();
  k.sourceType = QNN_GPU_KERNEL_SOURCE_TYPE_TEXT;
  k.kernelSource = m_source.c_str();
  k.sourceLength = m_source.size();
  k.buildOptions = "-cl-std=CL2.0";
  m_kernels.push_back(k);

  log(QNN_LOG_LEVEL_INFO, "FlowC Neuron(float) %s: kind=%d elems=%zu", cfgName(cfg), spec.kind,
      numElements);
  *status = QNN_SUCCESS;
}

}  // namespace flowc
