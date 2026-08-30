#pragma once
#include <memory>
#include <string>
#include <vector>

#include "FlowCGpuCommon.hpp"

namespace flowc {

class FullyConnectedOp : public Operation {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t* node,
                                           Qnn_ErrorHandle_t* status);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t& cfg);
  static const std::string s_opType;

 private:
  FullyConnectedOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);
  void buildQuantKernel(const Qnn_OpConfig_t& cfg, uint32_t batch, uint32_t N, uint32_t K,
                        bool hasBias, Qnn_ErrorHandle_t* status);
  void buildFloatKernel(const Qnn_OpConfig_t& cfg, uint32_t batch, uint32_t N, uint32_t K,
                        bool hasBias, Qnn_ErrorHandle_t* status);
  static void setWorkSizes(QnnGpu_Kernel_t& k, size_t total);

  std::string m_name;
  std::string m_source;
  std::vector<QnnGpu_KernelArg_t> m_args;
  std::vector<QnnGpu_KernelArg_t*> m_argPtrs;
};

}  // namespace flowc
