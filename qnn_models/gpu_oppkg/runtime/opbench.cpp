// opbench -- compose a converter-generated QNN model library on a chosen
// backend, optionally registering custom op packages, then time
// QnnGraph_execute and dump the output tensors.
//
// This is the sibling of qnn_models/runtime/profile_segments.cpp: same
// wallclock-around-graphExecute measurement (so the numbers are directly
// comparable to flow_c/measurements/qrb5165_v66.json), but it composes from a
// model .so instead of a context binary, because a context binary cannot carry
// ops that live in an op package the runtime has not registered yet.
//
// Usage:
//   opbench <model.so> <backend.so> <iters> [options]
//     --op-package <lib>:<iface>   register an op package (repeatable)
//     --input <tensorName>=<file>  fill an input tensor from a raw file
//     --dump-dir <dir>             write every output tensor as <name>.raw
//     --csv <path>                 per-iteration timings
//
// Build on the QRB5165:
//   g++ -std=c++11 -O2 -I/root/qairt/include/QNN opbench.cpp -o opbench -ldl
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnTypes.h"

#define CHECK(expr)                                                                            \
  do {                                                                                         \
    Qnn_ErrorHandle_t _e = (expr);                                                             \
    if (_e != QNN_SUCCESS) {                                                                   \
      std::fprintf(stderr, "QNN err 0x%llx at %s:%d (%s)\n", (unsigned long long)_e, __FILE__, \
                   __LINE__, #expr);                                                           \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

// Mirrors qnn_wrapper_api::GraphInfo_t from the SDK's converter/jni headers.
// Declared here so the harness builds without the SDK example sources.
struct GraphInfo_t {
  Qnn_GraphHandle_t graph;
  char* graphName;
  Qnn_Tensor_t* inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t* outputTensors;
  uint32_t numOutputTensors;
};
typedef GraphInfo_t* GraphInfoPtr_t;
struct GraphConfigInfo_t {
  char* graphName;
  const QnnGraph_Config_t** graphConfigs;
};

typedef int (*ComposeFn_t)(Qnn_BackendHandle_t,
                           QNN_INTERFACE_VER_TYPE,
                           Qnn_ContextHandle_t,
                           const GraphConfigInfo_t**,
                           const uint32_t,
                           GraphInfoPtr_t**,
                           uint32_t*,
                           bool,
                           QnnLog_Callback_t,
                           QnnLog_Level_t);

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}
static void log_cb(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
}

static uint32_t trank(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
static const uint32_t* tdims(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
static Qnn_DataType_t tdtype(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
static const char* tname(const Qnn_Tensor_t& t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
}
static size_t bpe(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8: return 1;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: return 2;
    default: return 4;
  }
}
static size_t tbytes(const Qnn_Tensor_t& t) {
  size_t n = 1;
  const uint32_t r = trank(t);
  const uint32_t* d = tdims(t);
  for (uint32_t i = 0; i < r; ++i) n *= d[i];
  return n * bpe(tdtype(t));
}
static void set_buf(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
  if (t.version == QNN_TENSOR_VERSION_2) {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf.data = data;
    t.v2.clientBuf.dataSize = bytes;
  } else {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = data;
    t.v1.clientBuf.dataSize = bytes;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <model.so> <backend.so> <iters> [--op-package lib:iface] "
                 "[--input name=file] [--dump-dir dir] [--csv path]\n",
                 argv[0]);
    return 1;
  }
  const std::string modelPath = argv[1];
  const std::string backendPath = argv[2];
  const int iters = std::atoi(argv[3]);
  std::vector<std::string> opPackages;
  std::vector<std::pair<std::string, std::string>> inputFiles;
  std::string dumpDir, csvPath;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--op-package" && i + 1 < argc) {
      opPackages.push_back(argv[++i]);
    } else if (a == "--input" && i + 1 < argc) {
      const std::string kv = argv[++i];
      const size_t eq = kv.find('=');
      inputFiles.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    } else if (a == "--dump-dir" && i + 1 < argc) {
      dumpDir = argv[++i];
    } else if (a == "--csv" && i + 1 < argc) {
      csvPath = argv[++i];
    }
  }

  void* blib = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!blib) {
    std::fprintf(stderr, "dlopen %s: %s\n", backendPath.c_str(), dlerror());
    return 1;
  }
  auto getProviders = (Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))dlsym(
      blib, "QnnInterface_getProviders");
  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  CHECK(getProviders(&providers, &numProviders));
  QNN_INTERFACE_VER_TYPE iface = providers[0]->QNN_INTERFACE_VER_NAME;

  Qnn_LogHandle_t logH = nullptr;
  CHECK(iface.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &logH));
  Qnn_BackendHandle_t backH = nullptr;
  CHECK(iface.backendCreate(logH, nullptr, &backH));

  for (const std::string& spec : opPackages) {
    const size_t colon = spec.find(':');
    const std::string path = spec.substr(0, colon);
    const std::string provider = spec.substr(colon + 1);
    CHECK(iface.backendRegisterOpPackage(backH, path.c_str(), provider.c_str(), nullptr));
    std::fprintf(stderr, "registered op package %s (%s)\n", path.c_str(), provider.c_str());
  }

  Qnn_ContextHandle_t ctx = nullptr;
  CHECK(iface.contextCreate(backH, nullptr, nullptr, &ctx));

  void* mlib = dlopen(modelPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!mlib) {
    std::fprintf(stderr, "dlopen %s: %s\n", modelPath.c_str(), dlerror());
    return 1;
  }
  auto compose = (ComposeFn_t)dlsym(mlib, "QnnModel_composeGraphs");
  if (!compose) {
    std::fprintf(stderr, "no QnnModel_composeGraphs in %s\n", modelPath.c_str());
    return 1;
  }

  GraphInfoPtr_t* graphs = nullptr;
  uint32_t numGraphs = 0;
  const double t_compose = now_us();
  const int cerr = compose(backH, iface, ctx, nullptr, 0, &graphs, &numGraphs, false, log_cb,
                           QNN_LOG_LEVEL_ERROR);
  if (cerr != 0 || numGraphs == 0) {
    std::printf("{\"model\":\"%s\",\"backend\":\"%s\",\"status\":\"compose_fail\",\"err\":%d}\n",
                modelPath.c_str(), backendPath.c_str(), cerr);
    return 2;
  }
  GraphInfo_t& g = *graphs[0];
  const double compose_us = now_us() - t_compose;

  const double t_final = now_us();
  const Qnn_ErrorHandle_t ferr = iface.graphFinalize(g.graph, nullptr, nullptr);
  const double finalize_us = now_us() - t_final;
  if (ferr != QNN_SUCCESS) {
    std::printf("{\"model\":\"%s\",\"backend\":\"%s\",\"status\":\"finalize_fail\","
                "\"err\":\"0x%llx\"}\n",
                modelPath.c_str(), backendPath.c_str(), (unsigned long long)ferr);
    return 3;
  }

  std::vector<std::vector<uint8_t>> inBufs(g.numInputTensors), outBufs(g.numOutputTensors);
  for (uint32_t i = 0; i < g.numInputTensors; ++i) {
    const size_t sz = tbytes(g.inputTensors[i]);
    inBufs[i].assign(sz, 0);
    const char* nm = tname(g.inputTensors[i]);
    for (const auto& kv : inputFiles) {
      if (nm && kv.first == nm) {
        std::ifstream f(kv.second, std::ios::binary);
        if (!f) {
          std::fprintf(stderr, "cannot open %s\n", kv.second.c_str());
          return 1;
        }
        f.read((char*)inBufs[i].data(), sz);
      }
    }
    set_buf(g.inputTensors[i], inBufs[i].data(), (uint32_t)sz);
  }
  for (uint32_t i = 0; i < g.numOutputTensors; ++i) {
    const size_t sz = tbytes(g.outputTensors[i]);
    outBufs[i].assign(sz, 0);
    set_buf(g.outputTensors[i], outBufs[i].data(), (uint32_t)sz);
  }

  const int warmup = std::min(5, iters / 4 + 1);
  for (int i = 0; i < warmup; ++i) {
    CHECK(iface.graphExecute(g.graph, g.inputTensors, g.numInputTensors, g.outputTensors,
                             g.numOutputTensors, nullptr, nullptr));
  }
  std::vector<double> per_call;
  per_call.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const double t0 = now_us();
    CHECK(iface.graphExecute(g.graph, g.inputTensors, g.numInputTensors, g.outputTensors,
                             g.numOutputTensors, nullptr, nullptr));
    per_call.push_back(now_us() - t0);
  }

  if (!dumpDir.empty()) {
    for (uint32_t i = 0; i < g.numOutputTensors; ++i) {
      const char* nm = tname(g.outputTensors[i]);
      std::string safe = nm ? nm : ("out" + std::to_string(i));
      for (char& c : safe)
        if (c == '/' || c == ':') c = '_';
      std::ofstream f(dumpDir + "/" + safe + ".raw", std::ios::binary);
      f.write((const char*)outBufs[i].data(), outBufs[i].size());
    }
  }

  std::vector<double> s = per_call;
  std::sort(s.begin(), s.end());
  const double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
  double sumsq = 0;
  for (double x : s) sumsq += (x - mean) * (x - mean);
  std::printf("{\"model\":\"%s\",\"backend\":\"%s\",\"status\":\"ok\",\"graph\":\"%s\","
              "\"iters\":%d,\"compose_us\":%.1f,\"finalize_us\":%.1f,\"mean_us\":%.2f,"
              "\"median_us\":%.2f,\"min_us\":%.2f,\"max_us\":%.2f,\"std_us\":%.2f,"
              "\"p99_us\":%.2f}\n",
              modelPath.c_str(), backendPath.c_str(), g.graphName ? g.graphName : "", iters,
              compose_us, finalize_us, mean, s[s.size() / 2], s.front(), s.back(),
              std::sqrt(sumsq / s.size()), s[std::min(s.size() - 1, s.size() * 99 / 100)]);
  if (!csvPath.empty()) {
    std::ofstream f(csvPath);
    f << "iter,call_us\n";
    for (size_t i = 0; i < per_call.size(); ++i) f << i << "," << per_call[i] << "\n";
  }
  iface.contextFree(ctx, nullptr);
  iface.backendFree(backH);
  iface.logFree(logH);
  return 0;
}
