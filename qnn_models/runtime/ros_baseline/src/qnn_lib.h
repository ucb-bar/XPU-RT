// Tiny QNN wrapper for the ROS baseline: load a pre-built context
// binary, hold the graph + buffers, expose `run()`. One instance per
// network sub-DLC. Mirrors the bringup logic in generate_runtime.py's
// emitted runtime, stripped to what the ROS nodes need.

#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

namespace qnn_baseline {

class QnnContextHolder {
public:
    QnnContextHolder(const std::string& bin_path,
                      const std::string& lib_path);
    ~QnnContextHolder();

    // graphExecute with zero-init buffers (the data is irrelevant — we
    // measure latency, not correctness, same as profile_seg).
    void run();

    // Warmup: N graphExecutes; first call on a cold context can be
    // 5-30x slower than steady-state per the cold-start finding from
    // qnn_models/runtime/HETEROGENEOUS_SCHEDULING_QRB5165.md.
    void warmup(int iters);

    const std::string& backend_label() const { return backend_label_; }
    const std::string& graph_name() const { return graph_name_; }

private:
    // Backend lib state (dlopen + Qnn handles).
    void*                              lib_ = nullptr;
    std::string                        lib_path_;
    std::string                        backend_label_;  // "Hta" / "Dsp" / "Cpu" derived from lib basename
    QNN_INTERFACE_VER_TYPE             iface_{};
    Qnn_LogHandle_t                    log_ = nullptr;
    Qnn_BackendHandle_t                backend_ = nullptr;
    Qnn_ContextHandle_t                ctx_ = nullptr;
    Qnn_GraphHandle_t                  graph_ = nullptr;

    // Tensor descriptors + storage owned for the lifetime of the
    // context. Buffers are zero-init at construction; we don't write
    // realistic input data because the latency we measure is the
    // graphExecute call, not the whole pipeline.
    std::string                        graph_name_;
    std::vector<Qnn_Tensor_t>          inputs_;
    std::vector<Qnn_Tensor_t>          outputs_;
    std::vector<std::vector<uint32_t>> dim_storage_;
    std::vector<std::string>           name_storage_;
    std::vector<std::vector<uint8_t>>  in_bufs_;
    std::vector<std::vector<uint8_t>>  out_bufs_;
};

inline double now_us_double() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace qnn_baseline
