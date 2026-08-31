//==============================================================================
//  FlowC GPU op package -- shared helpers (implementation).
//==============================================================================
#include "FlowCGpuCommon.hpp"

#include <cstdlib>

namespace flowc {

QnnLog_Callback_t g_logCallback = nullptr;
QnnLog_Level_t g_logLevel = QNN_LOG_LEVEL_ERROR;

void log(QnnLog_Level_t level, const char* fmt, ...) {
  if (!g_logCallback || level > g_logLevel) return;
  va_list ap;
  va_start(ap, fmt);
  (*g_logCallback)(fmt, level, 0, ap);
  va_end(ap);
}

bool findScalarParam(const Qnn_OpConfig_t& cfg, const char* name, Qnn_Scalar_t* out) {
  for (uint32_t i = 0u; i < cfgNumParams(cfg); ++i) {
    const Qnn_Param_t& p = cfgParam(cfg, i);
    if (p.paramType == QNN_PARAMTYPE_SCALAR && p.name && std::string(p.name) == name) {
      if (out) *out = p.scalarParam;
      return true;
    }
  }
  return false;
}

const Qnn_Tensor_t* findTensorParam(const Qnn_OpConfig_t& cfg, const char* name) {
  for (uint32_t i = 0u; i < cfgNumParams(cfg); ++i) {
    const Qnn_Param_t& p = cfgParam(cfg, i);
    if (p.paramType == QNN_PARAMTYPE_TENSOR && p.name && std::string(p.name) == name) {
      return &p.tensorParam;
    }
  }
  return nullptr;
}

bool readIntParam(const Qnn_OpConfig_t& cfg, const char* name, std::vector<int32_t>* out) {
  const Qnn_Tensor_t* t = findTensorParam(cfg, name);
  if (!t) return false;
  const void* data = (t->version == QNN_TENSOR_VERSION_1) ? t->v1.clientBuf.data
                                                          : t->v2.clientBuf.data;
  if (!data) return false;
  const size_t n = tNumElements(*t);
  out->resize(n);
  const Qnn_DataType_t dt = tDataType(*t);
  for (size_t i = 0u; i < n; ++i) {
    if (dt == QNN_DATATYPE_UINT_32) {
      (*out)[i] = (int32_t)((const uint32_t*)data)[i];
    } else if (dt == QNN_DATATYPE_INT_32) {
      (*out)[i] = ((const int32_t*)data)[i];
    } else {
      return false;
    }
  }
  return true;
}

float scalarAsFloat(const Qnn_Scalar_t& s, float fallback) {
  switch (s.dataType) {
    case QNN_DATATYPE_FLOAT_32: return s.floatValue;
    case QNN_DATATYPE_FLOAT_64: return (float)s.doubleValue;
    case QNN_DATATYPE_INT_32: return (float)s.int32Value;
    case QNN_DATATYPE_UINT_32: return (float)s.uint32Value;
    case QNN_DATATYPE_INT_64: return (float)s.int64Value;
    case QNN_DATATYPE_UINT_64: return (float)s.uint64Value;
    default: return fallback;
  }
}

int64_t scalarAsInt(const Qnn_Scalar_t& s, int64_t fallback) {
  switch (s.dataType) {
    case QNN_DATATYPE_INT_8: return s.int8Value;
    case QNN_DATATYPE_UINT_8: return s.uint8Value;
    case QNN_DATATYPE_INT_16: return s.int16Value;
    case QNN_DATATYPE_UINT_16: return s.uint16Value;
    case QNN_DATATYPE_INT_32: return s.int32Value;
    case QNN_DATATYPE_UINT_32: return s.uint32Value;
    case QNN_DATATYPE_INT_64: return s.int64Value;
    case QNN_DATATYPE_UINT_64: return (int64_t)s.uint64Value;
    case QNN_DATATYPE_FLOAT_32: return (int64_t)s.floatValue;
    default: return fallback;
  }
}

const QnnGpu_MemoryObject_t* storageOf(const QnnGpuOpPackage_Node_t* node, uint32_t tensorId) {
  if (!node || !node->storageTypes) return nullptr;
  for (const QnnGpu_TensorStorageType_t** p = node->storageTypes; *p != nullptr; ++p) {
    if ((*p)->id == tensorId) return (*p)->memoryObject;
  }
  return nullptr;
}

const char* memObjTypeName(QnnGpu_MemoryObjectType_t t) {
  switch (t) {
    case QNN_GPU_MEM_OBJ_TYPE_HOST: return "HOST";
    case QNN_GPU_MEM_OBJ_TYPE_BUFFER: return "BUFFER";
    case QNN_GPU_MEM_OBJ_TYPE_IMAGE2D: return "IMAGE2D";
    case QNN_GPU_MEM_OBJ_TYPE_IMAGE2D_ARRAY: return "IMAGE2D_ARRAY";
    case QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D: return "AGG_IMAGE2D";
    case QNN_GPU_MEM_OBJ_TYPE_AGGREGATED_IMAGE2D_ARRAY: return "AGG_IMAGE2D_ARRAY";
    case QNN_GPU_MEM_OBJ_TYPE_UNCLAIMED: return "UNCLAIMED";
    case QNN_GPU_MEM_OBJ_TYPE_IMAGE1D_BUFFER: return "IMAGE1D_BUFFER";
    default: return "?";
  }
}

void logStorage(const char* opName, const char* what, const QnnGpu_MemoryObject_t* mo) {
  if (!mo) {
    log(QNN_LOG_LEVEL_INFO, "FlowC storage %s %s: <none>", opName, what);
    return;
  }
  log(QNN_LOG_LEVEL_INFO,
      "FlowC storage %s %s: type=%s dt=0x%x nDims=%u dim0=%u off0=%d layout=%d",
      opName, what, memObjTypeName(mo->type), (unsigned)mo->dataType,
      (unsigned)mo->numDimensions, mo->numDimensions ? mo->dimensions[0] : 0u,
      mo->offsets ? (int)mo->offsets[0] : -1, (int)mo->layout);
}

bool bufferCompatible(const QnnGpu_MemoryObject_t* obj) {
  if (!obj) return true;  // backend has not decided yet
  return obj->type == QNN_GPU_MEM_OBJ_TYPE_BUFFER || obj->type == QNN_GPU_MEM_OBJ_TYPE_UNCLAIMED ||
         obj->type == QNN_GPU_MEM_OBJ_TYPE_HOST;
}

void Operation::claimOutputBuffer(const QnnGpuOpPackage_Node_t* node,
                                  const Qnn_OpConfig_t& cfg,
                                  uint32_t outputIndex) {
  const Qnn_Tensor_t& out = cfgOutput(cfg, outputIndex);
  if (tType(out) != QNN_TENSOR_TYPE_NATIVE) return;  // only NATIVE may be claimed
  const QnnGpu_MemoryObject_t* existing = storageOf(node, tId(out));
  if (existing && existing->type != QNN_GPU_MEM_OBJ_TYPE_UNCLAIMED) return;

  m_memDims.emplace_back(new uint32_t((uint32_t)tNumElements(out)));
  m_memOffs.emplace_back(new uint32_t(0u));
  m_memNames.emplace_back(new std::string(tName(out) ? tName(out) : ""));
  m_memObjs.emplace_back(new QnnGpu_MemoryObject_t(QNN_GPU_MEMORY_OBJECT_INIT));

  QnnGpu_MemoryObject_t* mo = m_memObjs.back().get();
  // FLOWC_CLAIM_DT lets the storage element type be probed independently of the
  // tensor's logical type (uFxp8 vs plain uint8) -- the backend's allocator is
  // the thing that rejects int8 tensors, so which spelling it accepts matters.
  const char* dtEnv = std::getenv("FLOWC_CLAIM_DT");
  Qnn_DataType_t claimDt = tDataType(out);
  if (dtEnv && std::string(dtEnv) == "uint8") claimDt = QNN_DATATYPE_UINT_8;
  if (dtEnv && std::string(dtEnv) == "int8") claimDt = QNN_DATATYPE_INT_8;

  mo->type = QNN_GPU_MEM_OBJ_TYPE_BUFFER;
  mo->dataType = claimDt;
  mo->dimensions = m_memDims.back().get();
  mo->offsets = m_memOffs.back().get();
  mo->numDimensions = 1u;
  mo->layout = QNN_GPU_MEM_LAYOUT_UNDEFINED;
  mo->quantizeParams.encodingType = QNN_GPU_QUANTIZATION_ENCODING_QNN_IMPL;
  mo->name = m_memNames.back()->c_str();

  QnnGpu_OutputClaim_t claim = QNN_GPU_OUTPUT_CLAIM_INIT;
  claim.opConfigIndex = 0u;
  claim.outputIndex = outputIndex;
  claim.memoryObject = mo;
  m_claims.push_back(claim);

  log(QNN_LOG_LEVEL_VERBOSE, "FlowC: claimed output %u of %s as BUFFER[%u]", outputIndex,
      cfgName(cfg), *m_memDims.back());
}

}  // namespace flowc
