//==============================================================================
//  FlowC GPU op package -- QnnOpPackage interface provider.
//
//  Registers int8 (and float) implementations of the ops the stock qti.aisw
//  GPU package refuses at QNN_DATATYPE_UFIXED_POINT_8.  Load with:
//
//    --op_packages <path>/libQnnGpuOpPackageFlowC.so:FlowCGpuOpPackage_interfaceProvider
//
//  The package name defaults to "flowc.gpu" and can be overridden at load time
//  with QNN_FLOWC_GPU_PKG_NAME -- useful for probing how the backend resolves a
//  node whose Qnn_OpConfig_t::packageName says "qti.aisw".
//==============================================================================
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "FlowCGpuCommon.hpp"
#include "OpConv2d.hpp"
#include "OpFullyConnected.hpp"
#include "OpMisc.hpp"
#include "OpNeuron.hpp"
#include "QnnOpDef.h"
#include "QnnSdkBuildId.h"

#ifdef _WIN32
#define FLOWC_UNUSED
#else
#define FLOWC_UNUSED __attribute__((unused))
#endif

namespace flowc {

typedef std::shared_ptr<Operation> (*CreateFn)(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
typedef Qnn_ErrorHandle_t (*ValidateFn)(const Qnn_OpConfig_t&);

class Package {
 public:
  explicit Package(const std::string& name) : m_name(name) {
    reg(FullyConnectedOp::s_opType, &FullyConnectedOp::create, &FullyConnectedOp::validate);
    reg(Conv2dOp::s_opType, &Conv2dOp::create, &Conv2dOp::validate);
    reg(Conv2dOp::s_opTypeDw, &Conv2dOp::create, &Conv2dOp::validate);
    // Every pointwise activation shares one LUT-based int8 implementation.
    const char* neuronTypes[] = {QNN_OP_ELEMENT_WISE_NEURON,
                                 QNN_OP_RELU,
                                 QNN_OP_RELU1,
                                 QNN_OP_RELU6,
                                 QNN_OP_RELU_MIN_MAX,
                                 QNN_OP_SIGMOID,
                                 QNN_OP_TANH,
                                 QNN_OP_ELU,
                                 QNN_OP_HARD_SWISH};
    for (const char* t : neuronTypes) reg(t, &NeuronOp::create, &NeuronOp::validate);

    reg(QNN_OP_POOL_MAX_2D, &PoolMaxOp::create, &PoolMaxOp::validate);
    reg(QNN_OP_BATCHNORM, &BatchnormOp::create, &BatchnormOp::validate);
    const char* binaryTypes[] = {QNN_OP_ELEMENT_WISE_BINARY, QNN_OP_ELEMENT_WISE_ADD,
                                 QNN_OP_ELEMENT_WISE_MULTIPLY, QNN_OP_ELEMENT_WISE_SUBTRACT};
    for (const char* t : binaryTypes) reg(t, &BinaryOp::create, &BinaryOp::validate);
    reg(QNN_OP_RESHAPE, &ReshapeOp::create, &ReshapeOp::validate);
    reg(QNN_OP_TRANSPOSE, &TransposeOp::create, &TransposeOp::validate);

    m_apiVersion = QNN_GPU_API_VERSION_INIT;
    m_info.packageName = m_name.c_str();
    m_info.operationNames = m_opNames.data();
    m_info.operationInfo = nullptr;
    m_info.numOperations = (uint32_t)m_opNames.size();
    m_info.optimizations = nullptr;
    m_info.numOptimizations = 0u;
    m_info.sdkBuildId = QNN_SDK_BUILD_ID;
    m_info.sdkApiVersion = &m_apiVersion;
    m_info.packageInfo = nullptr;
    m_info.opsetVersion = nullptr;
  }

  Qnn_ErrorHandle_t getInfo(const QnnOpPackage_Info_t** info) {
    *info = &m_info;
    return QNN_SUCCESS;
  }

  Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t& cfg) {
    const char* type = cfgTypeName(cfg);
    if (!type) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    auto it = m_validate.find(type);
    if (it == m_validate.end()) {
      log(QNN_LOG_LEVEL_VERBOSE, "FlowC: no implementation for op type %s", type);
      return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    }
    Qnn_ErrorHandle_t rc = (it->second)(cfg);
    log(QNN_LOG_LEVEL_INFO, "FlowC: validate %s (%s, pkg %s) -> %llu", cfgName(cfg), type,
        cfgPackageName(cfg) ? cfgPackageName(cfg) : "?", (unsigned long long)rc);
    return rc;
  }

  Qnn_ErrorHandle_t create(const QnnGpuOpPackage_Node_t* node, QnnGpu_Operation_t** operation) {
    const Qnn_OpConfig_t& cfg = *(node->configs[0]);
    auto it = m_create.find(cfgTypeName(cfg));
    if (it == m_create.end()) return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
    Qnn_ErrorHandle_t status = QNN_SUCCESS;
    std::shared_ptr<Operation> op = (it->second)(node, &status);
    if (!op || status != QNN_SUCCESS) {
      log(QNN_LOG_LEVEL_ERROR, "FlowC: failed to create %s", cfgName(cfg));
      return status == QNN_SUCCESS ? (Qnn_ErrorHandle_t)QNN_OP_PACKAGE_ERROR_GENERAL : status;
    }
    QnnGpu_Operation_t* info = op->info();
    m_live[info] = op;
    *operation = info;
    return QNN_SUCCESS;
  }

  Qnn_ErrorHandle_t free(QnnGpu_Operation_t* op) {
    auto it = m_live.find(op);
    if (it == m_live.end()) return QNN_OP_PACKAGE_ERROR_GENERAL;
    m_live.erase(it);
    return QNN_SUCCESS;
  }

 private:
  void reg(const std::string& type, CreateFn c, ValidateFn v) {
    auto inserted = m_create.emplace(type, c);
    if (inserted.second) m_opNames.push_back(inserted.first->first.c_str());
    m_validate[type] = v;
  }

  std::string m_name;
  std::map<std::string, CreateFn> m_create;
  std::map<std::string, ValidateFn> m_validate;
  std::vector<const char*> m_opNames;
  std::map<QnnGpu_Operation_t*, std::shared_ptr<Operation>> m_live;
  QnnOpPackage_Info_t m_info;
  Qnn_ApiVersion_t m_apiVersion;
};

static std::unique_ptr<Package> g_package;
static std::mutex g_mutex;

}  // namespace flowc

using namespace flowc;

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcInit(
    QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  if (!globalInfrastructure) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

  const char* override = std::getenv("QNN_FLOWC_GPU_PKG_NAME");
  const std::string name = override ? override : "flowc.gpu";

  const QnnGpu_DeviceProperties_t* props = globalInfrastructure->deviceProperties;
  if (props) {
    log(QNN_LOG_LEVEL_INFO,
        "FlowC GPU op package init: device='%s' tier='%s' maxWG=%zu localMem=%d vec64=%d",
        props->deviceVersion, props->tierName, props->maxWorkGroupSize,
        (int)props->isLocalMemorySupported, (int)props->vector64Support);
  }
  g_package.reset(new (std::nothrow) Package(name));
  if (!g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  log(QNN_LOG_LEVEL_INFO, "FlowC GPU op package registered as '%s'", name.c_str());
  return QNN_SUCCESS;
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcGetInfo(const QnnOpPackage_Info_t** info) {
  if (!g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  return g_package->getInfo(info);
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcValidate(Qnn_OpConfig_t opConfig) {
  if (!g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  return g_package->validate(opConfig);
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcCreateOpImpl(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* operation) {
  if (!graphInfrastructure || !node || !operation) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  if (!g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  return g_package->create(node, operation);
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcFreeOpImpl(QnnOpPackage_OpImpl_t operation) {
  if (!g_package) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  return g_package->free(operation);
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcTerminate() {
  g_package.reset();
  return QNN_SUCCESS;
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcLogInit(QnnLog_Callback_t callback,
                                                   QnnLog_Level_t maxLogLevel) {
  flowc::g_logCallback = callback;
  flowc::g_logLevel = maxLogLevel;
  return QNN_SUCCESS;
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcLogSetLevel(QnnLog_Level_t maxLogLevel) {
  flowc::g_logLevel = maxLogLevel;
  return QNN_SUCCESS;
}

FLOWC_UNUSED static Qnn_ErrorHandle_t flowcLogTerminate(void) {
  flowc::g_logCallback = nullptr;
  return QNN_SUCCESS;
}

extern "C" QNN_API Qnn_ErrorHandle_t
FlowCGpuOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init = flowcInit;
  interface->v1_4.terminate = flowcTerminate;
  interface->v1_4.getInfo = flowcGetInfo;
  interface->v1_4.validateOpConfig = flowcValidate;
  interface->v1_4.createOpImpl = flowcCreateOpImpl;
  interface->v1_4.freeOpImpl = flowcFreeOpImpl;
  interface->v1_4.logInitialize = flowcLogInit;
  interface->v1_4.logSetLevel = flowcLogSetLevel;
  interface->v1_4.logTerminate = flowcLogTerminate;
  return QNN_SUCCESS;
}

// Same entry point under the conventional name, so the package can also be
// loaded with the default ":QnnOpPackage_interfaceProvider" spelling.
extern "C" QNN_API Qnn_ErrorHandle_t
QnnOpPackage_interfaceProvider(QnnOpPackage_Interface_t* interface) {
  return FlowCGpuOpPackage_interfaceProvider(interface);
}

// The GPU backend may ask for precompiled kernel binaries; we ship source only.
extern "C" QNN_API Qnn_ErrorHandle_t QnnGpuOpPackage_getKernelBinary(const char* name,
                                                                    const uint8_t** binary,
                                                                    const uint32_t* numBytes) {
  (void)name;
  (void)binary;
  (void)numBytes;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}
