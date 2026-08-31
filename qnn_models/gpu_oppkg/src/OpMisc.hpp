#pragma once
#include <memory>
#include <string>
#include <vector>

#include "FlowCGpuCommon.hpp"

namespace flowc {

// Shared scaffolding for the small int8 ops: they all emit one kernel with a
// baked-in constant set and a hashed name.
class SimpleOp : public Operation {
 protected:
  void finish(const char* base, const std::string& source, size_t items,
              const std::vector<QnnGpu_KernelArg_t>& args, size_t localHint = 64u);
  std::string m_name;
  std::string m_source;
  std::vector<QnnGpu_KernelArg_t> m_args;
  std::vector<QnnGpu_KernelArg_t*> m_argPtrs;
};

class PoolMaxOp : public SimpleOp {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t&);
 private:
  PoolMaxOp(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
};

class BatchnormOp : public SimpleOp {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t&);
 private:
  BatchnormOp(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
};

class BinaryOp : public SimpleOp {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t&);
 private:
  BinaryOp(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
};

// Reshape / Squeeze / ExpandDims: a byte copy at int8.
class ReshapeOp : public SimpleOp {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t&);
 private:
  ReshapeOp(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
};

class TransposeOp : public SimpleOp {
 public:
  static std::shared_ptr<Operation> create(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
  static Qnn_ErrorHandle_t validate(const Qnn_OpConfig_t&);
 private:
  TransposeOp(const QnnGpuOpPackage_Node_t*, Qnn_ErrorHandle_t*);
};

}  // namespace flowc
