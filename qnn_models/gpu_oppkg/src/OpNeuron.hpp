#pragma once
#include <memory>
#include <string>
#include <vector>

#include "FlowCGpuCommon.hpp"

namespace flowc {

// Covers ElementWiseNeuron (operation param) plus the standalone pointwise op
// types Relu / Relu6 / ReluMinMax / Sigmoid / Tanh / Elu / HardSwish / Gelu.
class NeuronOp : public Operation {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t* node,
                                           Qnn_ErrorHandle_t* status);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t& cfg);
  static bool handles(const std::string& typeName);

 private:
  NeuronOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);
  void buildLutKernel(const Qnn_OpConfig_t& cfg, size_t numElements, Qnn_ErrorHandle_t* status);
  void buildFloatKernel(const Qnn_OpConfig_t& cfg, size_t numElements, Qnn_ErrorHandle_t* status);

  std::string m_name;
  std::string m_source;
  std::vector<QnnGpu_KernelArg_t> m_args;
  std::vector<QnnGpu_KernelArg_t*> m_argPtrs;
};

}  // namespace flowc
