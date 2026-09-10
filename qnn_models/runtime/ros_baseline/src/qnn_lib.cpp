#include "qnn_lib.h"

#include <cstdio>

namespace qnn_baseline {

namespace {

#define QCHECK(expr) do { Qnn_ErrorHandle_t _e=(expr); \
    if (_e != QNN_SUCCESS) throw std::runtime_error( \
        std::string("QNN err 0x") + std::to_string((unsigned long long)_e) \
        + " at " + #expr); } while(0)

void log_cb(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
    std::vfprintf(stderr, fmt, ap); std::fprintf(stderr, "\n");
}

std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("can't open " + p);
    auto n = f.tellg(); f.seekg(0); std::vector<uint8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n); return v;
}

uint32_t trank(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
const uint32_t* tdims(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
Qnn_DataType_t tdtype(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
const char* tname(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
}
size_t bpe(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8: return 1;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16: return 2;
        case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32: return 4;
        default: return 4;
    }
}
size_t tbytes(const Qnn_Tensor_t& t) {
    size_t n = 1; uint32_t r = trank(t); auto* d = tdims(t);
    for (uint32_t i = 0; i < r; ++i) n *= d[i];
    return n * bpe(tdtype(t));
}
void rebind(Qnn_Tensor_t& t, std::vector<uint32_t>& d, std::string& nm) {
    uint32_t r = trank(t); auto* dp = tdims(t); const char* np = tname(t);
    d.assign(dp, dp + r); nm = np ? np : "";
    if (t.version == QNN_TENSOR_VERSION_2) { t.v2.dimensions = d.data(); t.v2.name = nm.c_str(); }
    else                                   { t.v1.dimensions = d.data(); t.v1.name = nm.c_str(); }
}
void set_buf(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
    Qnn_ClientBuffer_t cb{data, bytes};
    if (t.version == QNN_TENSOR_VERSION_2) {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf = cb;
    } else {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf = cb;
    }
}

std::string label_from_lib(const std::string& lib_path) {
    auto p = lib_path.rfind('/');
    std::string base = (p == std::string::npos) ? lib_path : lib_path.substr(p + 1);
    // libQnnHta.so → "Hta"; libQnnDsp.so → "Dsp"; libQnnCpu.so → "Cpu"
    if (base.rfind("libQnn", 0) == 0 && base.size() > 9) {
        size_t end = base.find('.');
        std::string s = base.substr(6, end - 6);
        if (!s.empty()) {
            s[0] = static_cast<char>(std::toupper(s[0]));
            for (size_t i = 1; i < s.size(); ++i) s[i] = static_cast<char>(std::tolower(s[i]));
        }
        return s;
    }
    return base;
}

}  // namespace

QnnContextHolder::QnnContextHolder(const std::string& bin_path,
                                     const std::string& lib_path)
    : lib_path_(lib_path), backend_label_(label_from_lib(lib_path)) {
    // 1. Introspect the binary via libQnnSystem.so to learn graph name + IO.
    void* slib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!slib) throw std::runtime_error(std::string("dlopen libQnnSystem.so: ") + dlerror());
    auto sgp = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*)>(
                  dlsym(slib, "QnnSystemInterface_getProviders"));
    const QnnSystemInterface_t** sprov = nullptr; uint32_t sn = 0;
    QCHECK(sgp(&sprov, &sn));
    auto& siface = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    auto bin = slurp(bin_path);
    QnnSystemContext_Handle_t sh = nullptr;
    QCHECK(siface.systemContextCreate(&sh));
    const QnnSystemContext_BinaryInfo_t* bi = nullptr;
    Qnn_ContextBinarySize_t bisz = 0;
    QCHECK(siface.systemContextGetBinaryInfo(sh, bin.data(), bin.size(), &bi, &bisz));

    const QnnSystemContext_GraphInfo_t* gs = nullptr;
    if      (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) gs = bi->contextBinaryInfoV3.graphs;
    else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) gs = bi->contextBinaryInfoV2.graphs;
    else                                                              gs = bi->contextBinaryInfoV1.graphs;
    const auto& g = gs[0];
    const char* gn = nullptr; uint32_t nIn = 0, nOut = 0;
    const Qnn_Tensor_t* inT = nullptr; const Qnn_Tensor_t* outT = nullptr;
    if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        gn = g.graphInfoV3.graphName;
        nIn = g.graphInfoV3.numGraphInputs; inT = g.graphInfoV3.graphInputs;
        nOut = g.graphInfoV3.numGraphOutputs; outT = g.graphInfoV3.graphOutputs;
    } else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        gn = g.graphInfoV2.graphName;
        nIn = g.graphInfoV2.numGraphInputs; inT = g.graphInfoV2.graphInputs;
        nOut = g.graphInfoV2.numGraphOutputs; outT = g.graphInfoV2.graphOutputs;
    } else {
        gn = g.graphInfoV1.graphName;
        nIn = g.graphInfoV1.numGraphInputs; inT = g.graphInfoV1.graphInputs;
        nOut = g.graphInfoV1.numGraphOutputs; outT = g.graphInfoV1.graphOutputs;
    }
    inputs_.assign(inT, inT + nIn);
    outputs_.assign(outT, outT + nOut);
    dim_storage_.resize(nIn + nOut);
    name_storage_.resize(nIn + nOut);
    for (size_t i = 0; i < inputs_.size(); ++i)  rebind(inputs_[i],  dim_storage_[i],       name_storage_[i]);
    for (size_t i = 0; i < outputs_.size(); ++i) rebind(outputs_[i], dim_storage_[nIn + i], name_storage_[nIn + i]);
    graph_name_ = gn ? gn : "";
    siface.systemContextFree(sh);

    // 2. Bring up the actual backend.
    lib_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib_) throw std::runtime_error(std::string("dlopen ") + lib_path + ": " + dlerror());
    auto bgp = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*)>(
                  dlsym(lib_, "QnnInterface_getProviders"));
    const QnnInterface_t** bprov = nullptr; uint32_t bn = 0;
    QCHECK(bgp(&bprov, &bn));
    iface_ = bprov[0]->QNN_INTERFACE_VER_NAME;

    QCHECK(iface_.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &log_));
    QCHECK(iface_.backendCreate(log_, nullptr, &backend_));
    QCHECK(iface_.contextCreateFromBinary(backend_, nullptr, nullptr,
                                           bin.data(), bin.size(), &ctx_, nullptr));
    QCHECK(iface_.graphRetrieve(ctx_, graph_name_.c_str(), &graph_));

    // 3. Allocate zero-init IO.
    in_bufs_.resize(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        size_t sz = tbytes(inputs_[i]);
        in_bufs_[i].assign(sz, 0);
        set_buf(inputs_[i], in_bufs_[i].data(), static_cast<uint32_t>(sz));
    }
    out_bufs_.resize(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
        size_t sz = tbytes(outputs_[i]);
        out_bufs_[i].assign(sz, 0);
        set_buf(outputs_[i], out_bufs_[i].data(), static_cast<uint32_t>(sz));
    }

    std::fprintf(stderr, "[qnn] loaded %s on %s graph=%s in=%zu out=%zu\n",
                 bin_path.c_str(), backend_label_.c_str(), graph_name_.c_str(),
                 inputs_.size(), outputs_.size());
}

QnnContextHolder::~QnnContextHolder() {
    if (ctx_)     iface_.contextFree(ctx_, nullptr);
    if (backend_) iface_.backendFree(backend_);
    if (log_)     iface_.logFree(log_);
    if (lib_)     dlclose(lib_);
}

void QnnContextHolder::run() {
    QCHECK(iface_.graphExecute(graph_,
        inputs_.data(),  static_cast<uint32_t>(inputs_.size()),
        outputs_.data(), static_cast<uint32_t>(outputs_.size()),
        nullptr, nullptr));
}

void QnnContextHolder::warmup(int iters) {
    if (iters <= 0) return;
    double t0 = now_us_double();
    for (int i = 0; i < iters; ++i) run();
    double t1 = now_us_double();
    std::fprintf(stderr, "[qnn] %s/%s warmup x%d in %.2f ms\n",
                 graph_name_.c_str(), backend_label_.c_str(), iters,
                 (t1 - t0) / 1000.0);
}

}  // namespace qnn_baseline
