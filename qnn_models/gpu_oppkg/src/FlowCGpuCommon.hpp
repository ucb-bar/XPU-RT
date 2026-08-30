//==============================================================================
//  FlowC GPU op package -- shared helpers.
//
//  Written for the QNN GPU (Adreno) backend, QAIRT 2.45, target QRB5165
//  (Adreno 650).  Nothing here is copied from the SDK examples: the accessors
//  below are written directly against the public structs in QnnTypes.h so the
//  package has no dependency on SDK example sources.
//==============================================================================
#pragma once

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "GPU/QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

namespace flowc {

// ---------------------------------------------------------------------------
// Logging: the backend hands us a QnnLog callback at package init.
// ---------------------------------------------------------------------------
extern QnnLog_Callback_t g_logCallback;
extern QnnLog_Level_t g_logLevel;
void log(QnnLog_Level_t level, const char* fmt, ...);

// ---------------------------------------------------------------------------
// Qnn_OpConfig_t / Qnn_Tensor_t accessors (version-tolerant).
// ---------------------------------------------------------------------------
inline const char* cfgTypeName(const Qnn_OpConfig_t& c) { return c.v1.typeName; }
inline const char* cfgName(const Qnn_OpConfig_t& c) { return c.v1.name; }
inline const char* cfgPackageName(const Qnn_OpConfig_t& c) { return c.v1.packageName; }
inline uint32_t cfgNumInputs(const Qnn_OpConfig_t& c) { return c.v1.numOfInputs; }
inline uint32_t cfgNumOutputs(const Qnn_OpConfig_t& c) { return c.v1.numOfOutputs; }
inline const Qnn_Tensor_t& cfgInput(const Qnn_OpConfig_t& c, uint32_t i) {
  return c.v1.inputTensors[i];
}
inline const Qnn_Tensor_t& cfgOutput(const Qnn_OpConfig_t& c, uint32_t i) {
  return c.v1.outputTensors[i];
}
inline uint32_t cfgNumParams(const Qnn_OpConfig_t& c) { return c.v1.numOfParams; }
inline const Qnn_Param_t& cfgParam(const Qnn_OpConfig_t& c, uint32_t i) { return c.v1.params[i]; }

inline uint32_t tRank(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.rank : t.v2.rank;
}
inline const uint32_t* tDims(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.dimensions : t.v2.dimensions;
}
inline Qnn_DataType_t tDataType(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.dataType : t.v2.dataType;
}
inline Qnn_TensorType_t tType(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.type : t.v2.type;
}
inline uint32_t tId(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.id : t.v2.id;
}
inline const char* tName(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.name : t.v2.name;
}
inline const Qnn_QuantizeParams_t& tQuant(const Qnn_Tensor_t& t) {
  return (t.version == QNN_TENSOR_VERSION_1) ? t.v1.quantizeParams : t.v2.quantizeParams;
}
inline size_t tNumElements(const Qnn_Tensor_t& t) {
  size_t n = 1u;
  const uint32_t r = tRank(t);
  const uint32_t* d = tDims(t);
  for (uint32_t i = 0u; i < r; ++i) n *= d[i];
  return n;
}

// ---------------------------------------------------------------------------
// Quantization.  QNN's convention (QnnTypes.h, Qnn_ScaleOffset_t) is
//     real = (quantized + offset) * scale
// with `offset` normally negative for unsigned fixed point.
// ---------------------------------------------------------------------------
struct QuantInfo {
  bool valid = false;
  float scale = 1.0f;
  int32_t offset = 0;
};

inline QuantInfo quantOf(const Qnn_Tensor_t& t) {
  QuantInfo q;
  const Qnn_QuantizeParams_t& p = tQuant(t);
  if (p.encodingDefinition == QNN_DEFINITION_DEFINED &&
      p.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    q.valid = true;
    q.scale = p.scaleOffsetEncoding.scale;
    q.offset = p.scaleOffsetEncoding.offset;
  }
  return q;
}

inline bool isU8(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_UFIXED_POINT_8; }
inline bool isS8(Qnn_DataType_t dt) { return dt == QNN_DATATYPE_SFIXED_POINT_8; }
inline bool isQuant8(Qnn_DataType_t dt) { return isU8(dt) || isS8(dt); }
inline bool isFloat(Qnn_DataType_t dt) {
  return dt == QNN_DATATYPE_FLOAT_32 || dt == QNN_DATATYPE_FLOAT_16;
}

// Tensor param lookup (stride / pad_amount / filter_size / dilation ...).
// Param tensors are static and arrive with their data in clientBuf.
const Qnn_Tensor_t* findTensorParam(const Qnn_OpConfig_t& cfg, const char* name);
// Reads a rank-1 or rank-2 int32/uint32 param tensor into a flat vector.
bool readIntParam(const Qnn_OpConfig_t& cfg, const char* name, std::vector<int32_t>* out);

// Scalar param lookup by name.  Returns false when absent.
bool findScalarParam(const Qnn_OpConfig_t& cfg, const char* name, Qnn_Scalar_t* out);
float scalarAsFloat(const Qnn_Scalar_t& s, float fallback);
int64_t scalarAsInt(const Qnn_Scalar_t& s, int64_t fallback);

// Storage-type lookup: the backend tells us, per tensor id, which OpenCL memory
// object it has assigned.  Our kernels are written against plain __global
// buffers, so an op whose tensors landed in image memory must be refused rather
// than silently miscomputed.
const QnnGpu_MemoryObject_t* storageOf(const QnnGpuOpPackage_Node_t* node, uint32_t tensorId);
const char* memObjTypeName(QnnGpu_MemoryObjectType_t t);
// True when the tensor is either already a linear buffer or still unclaimed
// (in which case we claim it as a buffer).
bool bufferCompatible(const QnnGpu_MemoryObject_t* obj);
void logStorage(const char* opName, const char* what, const QnnGpu_MemoryObject_t* mo);

// ---------------------------------------------------------------------------
// Base class for a compiled operation.  Owns every string / vector referenced
// by the QnnGpu_Kernel_t structs it hands back, which must outlive finalize.
// ---------------------------------------------------------------------------
class Operation {
 public:
  virtual ~Operation() {}

  QnnGpu_Operation_t* info() {
    m_op = QNN_GPU_OPERATION_INIT;
    m_kernelPtrs.clear();
    for (size_t i = 0u; i < m_kernels.size(); ++i) m_kernelPtrs.push_back(&m_kernels[i]);
    m_kernelPtrs.push_back(nullptr);
    m_op.kernels = m_kernels.empty() ? nullptr : m_kernelPtrs.data();
    if (!m_claims.empty()) {
      m_claimPtrs.clear();
      for (size_t i = 0u; i < m_claims.size(); ++i) m_claimPtrs.push_back(&m_claims[i]);
      m_claimPtrs.push_back(nullptr);
      m_op.outputClaims = m_claimPtrs.data();
    }
    return &m_op;
  }

 protected:
  Operation() {}

  // Claim a NATIVE output as a plain linear device buffer.
  //
  // Why this is needed: the GPU backend's default storage choice for a NATIVE
  // tensor is an image object, and Adreno has no image channel format for
  // 8-bit fixed point -- graph finalize fails with
  // GPU_ERROR_INVALID_TYPE("Tensor memory error") before a kernel ever runs.
  // The op package is allowed to state what it wants ("principle of least
  // work"), so every int8 output we produce is claimed as a 1-D buffer of
  // numElements bytes, which is what our kernels index.
  void claimOutputBuffer(const QnnGpuOpPackage_Node_t* node,
                         const Qnn_OpConfig_t& cfg,
                         uint32_t outputIndex);

  QnnGpu_Operation_t m_op;
  std::vector<QnnGpu_Kernel_t> m_kernels;
  std::vector<QnnGpu_Kernel_t*> m_kernelPtrs;
  std::vector<QnnGpu_OutputClaim_t> m_claims;
  std::vector<QnnGpu_OutputClaim_t*> m_claimPtrs;
  // Backing store for the claimed memory objects: the backend keeps the
  // pointers we hand it, so nothing here may be reallocated or go out of scope.
  std::vector<std::unique_ptr<QnnGpu_MemoryObject_t>> m_memObjs;
  std::vector<std::unique_ptr<uint32_t>> m_memDims;
  std::vector<std::unique_ptr<uint32_t>> m_memOffs;
  std::vector<std::unique_ptr<std::string>> m_memNames;
};

// The GPU backend keys its compiled-program cache on the *kernel name*: two
// nodes whose kernels share a name run the same compiled program, whichever
// was compiled first.  Because this package specialises the kernel source per
// node (shapes, offsets, scales and activation tables are baked in as
// literals), every node must publish a distinct kernel name -- otherwise later
// nodes silently execute the first node's arithmetic.  This was measured, not
// assumed: with a shared name, layers 2/4/6 of mlp_control reproduced layer 0's
// kernel bit-for-bit on their own buffers.
inline std::string uniqueKernelName(const std::string& base, const std::string& source) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (char c : source) {
    h ^= (uint64_t)(unsigned char)c;
    h *= 1099511628211ull;
  }
  char buf[24];
  std::snprintf(buf, sizeof(buf), "_%016llx", (unsigned long long)h);
  return base + buf;
}

// A valid OpenCL float literal.  Printing a double with the default stream
// formatting yields "1" for 1.0, and "1f" is not a legal literal -- that
// mistake cost an OpenCL build error (-11) in the float activation kernel.
inline std::string litf(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.9ef", v);
  return std::string(buf);
}

// Helper to build the null-terminated arg pointer array a kernel needs.
inline void finishArgs(std::vector<QnnGpu_KernelArg_t>& args,
                       std::vector<QnnGpu_KernelArg_t*>& ptrs,
                       QnnGpu_Kernel_t& kernel) {
  ptrs.clear();
  for (size_t i = 0u; i < args.size(); ++i) ptrs.push_back(&args[i]);
  ptrs.push_back(nullptr);
  kernel.args = ptrs.data();
}

inline QnnGpu_KernelArg_t tensorArg(QnnGpu_KernelArgType_t type, uint32_t tensorIndex) {
  QnnGpu_KernelArg_t a = QNN_GPU_KERNEL_ARG_INIT;
  a.type = type;
  a.tensor.opConfigIndex = 0u;
  a.tensor.tensorIndex = tensorIndex;
  a.tensor.element = 0u;
  return a;
}

inline QnnGpu_KernelArg_t floatArg(float v) {
  QnnGpu_KernelArg_t a = QNN_GPU_KERNEL_ARG_INIT;
  a.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  a.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_FLOAT;
  a.data.qnnFloat = v;
  return a;
}

inline QnnGpu_KernelArg_t intArg(int32_t v) {
  QnnGpu_KernelArg_t a = QNN_GPU_KERNEL_ARG_INIT;
  a.type = QNN_GPU_KERNEL_ARG_TYPE_DATA;
  a.data.type = QNN_GPU_KERNEL_ARG_CL_TYPE_INT;
  a.data.qnnInt = v;
  return a;
}

}  // namespace flowc
