#pragma once
#include <memory>
#include <string>
#include <vector>

#include "FlowCGpuCommon.hpp"

namespace flowc {

class Conv2dOp : public Operation {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t* node,
                                           Qnn_ErrorHandle_t* status);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t& cfg);
  static const std::string s_opType;
  static const std::string s_opTypeDw;

 private:
  Conv2dOp(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status);

  std::string m_name;
  std::string m_source;
  std::vector<QnnGpu_KernelArg_t> m_args;
  std::vector<QnnGpu_KernelArg_t*> m_argPtrs;
};

}  // namespace flowc
