// ROS 2 whole-network-pinning harness.
//
// One rclcpp::Node per network, one SingleThreadedExecutor per node on its own
// thread, the network's WHOLE dispatch graph pinned to ONE backend. No per-op
// placement, no tile splitting, no sharding: the only free variable is the map
// `network -> backend`, which the caller fixes in the config file. This is the
// deliberate negation of what XPU-RT's scheduler does, and it is the fixed
// baseline that scheduler is measured against.
//
// Generalised over an arbitrary network set from
// qnn_models/flow_c/sweeps/qrb5165_ros_pinsweep_*/plans/<cell>.json, which
// drive.py flattens into the line-oriented config this reads. It replaces the
// three hardcoded nodes (dronet/mlp_control/yolov8n) for sweep work; those
// still build and are untouched.
//
// The QNN bringup -- one shared backend handle per .so, contexts created from
// binary, graph tensors rebound onto owned zero-filled buffers -- is lifted
// from the runtime flowc/emit_runtime.py generates, so both sides of the
// comparison bring the hardware up the same way. In particular the shared
// backend is what makes two contexts on one .so legal in one process
// ("Context handle already exists!" otherwise).
//
// Config file (whitespace separated, one directive per line, '#' comments):
//
//   reps        <int>      measured passes
//   warm        <int>      discarded passes before each measured pass
//   timeout_ms  <double>   per-pass watchdog
//   net    <name> <backend> <lib.so> <num_instances> <period_ms> <window_ms>
//   tile   <ctx_path> <graph_name>          (repeats, applies to the last net)
//   edge   <from_net> <to_net>
//
// `period_ms` < 0 marks an aperiodic network: all of its instances are
// released together at t0, exactly as XPU-RT releases an entry with no period.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

#define CHECK(expr) do { Qnn_ErrorHandle_t _e = (expr);                        \
    if (_e != QNN_SUCCESS) {                                                   \
        std::fprintf(stderr, "QNN error 0x%llx at %s:%d - %s\n",               \
                     (unsigned long long)_e, __FILE__, __LINE__, #expr);       \
        std::exit(1);                                                          \
    } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        steady_clock::now().time_since_epoch()).count();
}

static void log_cb(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
}

static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(1); }
    auto n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

// ------------------------------------------------------------- QNN tensors
// Verbatim from the runtime flowc/emit_runtime.py generates, so the two sides
// of the comparison read tensor metadata identically.
static uint32_t tensor_rank(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
static const uint32_t* tensor_dims(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
static Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
static const char* tensor_name(const Qnn_Tensor_t& t) {
    const char* n = (t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name);
    return n ? n : "";
}
static size_t bytes_per_element(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:        return 1;
        case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_FLOAT_16:      return 2;
        case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_SFIXED_POINT_32: case QNN_DATATYPE_UFIXED_POINT_32: return 4;
        case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64:      return 8;
        default:                         return 4;
    }
}
static size_t tensor_byte_size(const Qnn_Tensor_t& t) {
    size_t n = 1; uint32_t r = tensor_rank(t); auto* d = tensor_dims(t);
    for (uint32_t i = 0; i < r; ++i) n *= d[i];
    return n * bytes_per_element(tensor_dtype(t));
}
// The binary-info tensors point into memory owned by the QnnSystem handle;
// deep-copy dims and name so they survive after we free it.
static void rebind_tensor(Qnn_Tensor_t& t, std::vector<uint32_t>& d, std::string& nm) {
    uint32_t r = tensor_rank(t); auto* dp = tensor_dims(t);
    d.assign(dp, dp + r); nm = tensor_name(t);
    if (t.version == QNN_TENSOR_VERSION_2) { t.v2.dimensions = d.data(); t.v2.name = nm.c_str(); }
    else                                   { t.v1.dimensions = d.data(); t.v1.name = nm.c_str(); }
}
static void set_tensor_buffer(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
    if (t.version == QNN_TENSOR_VERSION_2) {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data = data; t.v2.clientBuf.dataSize = bytes;
    } else {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data = data; t.v1.clientBuf.dataSize = bytes;
    }
}

// ------------------------------------------------------------- QNN handles
struct SharedBackend {
    void*                  lib     = nullptr;
    QNN_INTERFACE_VER_TYPE iface{};
    Qnn_LogHandle_t        log     = nullptr;
    Qnn_BackendHandle_t    backend = nullptr;
};
static std::unordered_map<std::string, std::shared_ptr<SharedBackend>> g_backends;

struct GraphInfo {
    Qnn_GraphHandle_t                  graph = nullptr;
    std::string                        name;
    std::vector<Qnn_Tensor_t>          inputs, outputs;
    std::vector<std::vector<uint8_t>>  inputBufs, outputBufs;
    std::vector<std::vector<uint32_t>> dimStorage;
    std::vector<std::string>           nameStorage;
};

struct SysFns {
    QnnSystemContext_CreateFn_t        create        = nullptr;
    QnnSystemContext_GetBinaryInfoFn_t getBinaryInfo = nullptr;
    QnnSystemContext_FreeFn_t          free          = nullptr;
};
static SysFns g_sys;
static int    g_warmup_execs = 2;

static SysFns load_system() {
    SysFns f{};
    void* lib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { std::fprintf(stderr, "dlopen libQnnSystem.so: %s\n", dlerror()); std::exit(1); }
    auto fn = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*)>(
        dlsym(lib, "QnnSystemInterface_getProviders"));
    const QnnSystemInterface_t** prov = nullptr; uint32_t n = 0;
    CHECK(fn(&prov, &n));
    auto& s = prov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    f.create = s.systemContextCreate;
    f.getBinaryInfo = s.systemContextGetBinaryInfo;
    f.free = s.systemContextFree;
    return f;
}

// One loaded tile: a context binary plus the one graph inside it we execute.
struct Tile {
    std::string                    ctx_file, graph_name;
    std::shared_ptr<SharedBackend> sb;
    Qnn_ContextHandle_t            ctx = nullptr;
    std::unique_ptr<GraphInfo>     gi;

    void run() {
        Qnn_ErrorHandle_t err = sb->iface.graphExecute(
            gi->graph, gi->inputs.data(), static_cast<uint32_t>(gi->inputs.size()),
            gi->outputs.data(), static_cast<uint32_t>(gi->outputs.size()),
            nullptr, nullptr);
        if (err != QNN_SUCCESS)
            std::fprintf(stderr, "[exec] %s::%s failed 0x%llx\n",
                         ctx_file.c_str(), graph_name.c_str(),
                         static_cast<unsigned long long>(err));
    }
};

static void bringup(Tile& t, const std::string& bin_path,
                    const std::string& lib_path, const std::string& label) {
    t.ctx_file = bin_path;
    auto it = g_backends.find(lib_path);
    if (it != g_backends.end()) {
        t.sb = it->second;
    } else {
        auto sb = std::make_shared<SharedBackend>();
        sb->lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!sb->lib) { std::fprintf(stderr, "dlopen %s: %s\n", lib_path.c_str(), dlerror()); std::exit(1); }
        auto getProv = reinterpret_cast<Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*)>(
            dlsym(sb->lib, "QnnInterface_getProviders"));
        const QnnInterface_t** prov = nullptr; uint32_t np = 0;
        CHECK(getProv(&prov, &np));
        sb->iface = prov[0]->QNN_INTERFACE_VER_NAME;
        CHECK(sb->iface.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &sb->log));
        CHECK(sb->iface.backendCreate(sb->log, nullptr, &sb->backend));
        g_backends[lib_path] = sb;
        t.sb = sb;
    }
    auto& iface = t.sb->iface;

    auto bin = slurp(bin_path);
    QnnSystemContext_Handle_t sysHandle = nullptr;
    CHECK(g_sys.create(&sysHandle));
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSz = 0;
    CHECK(g_sys.getBinaryInfo(sysHandle, bin.data(), bin.size(), &info, &infoSz));
    uint32_t nG = 0; const QnnSystemContext_GraphInfo_t* gs = nullptr;
    if      (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { nG = info->contextBinaryInfoV1.numGraphs; gs = info->contextBinaryInfoV1.graphs; }
    else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { nG = info->contextBinaryInfoV2.numGraphs; gs = info->contextBinaryInfoV2.graphs; }
    else                                                                { nG = info->contextBinaryInfoV3.numGraphs; gs = info->contextBinaryInfoV3.graphs; }
    if (!nG) { std::fprintf(stderr, "no graphs in %s\n", bin_path.c_str()); std::exit(1); }

    CHECK(iface.contextCreateFromBinary(t.sb->backend, nullptr, nullptr,
                                        bin.data(), bin.size(), &t.ctx, nullptr));

    std::string picked;
    for (uint32_t gi = 0; gi < nG; ++gi) {
        const auto& g = gs[gi];
        const char* nm = nullptr; uint32_t nIn = 0, nOut = 0;
        const Qnn_Tensor_t *inT = nullptr, *outT = nullptr;
        if      (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { nm = g.graphInfoV1.graphName; nIn = g.graphInfoV1.numGraphInputs; inT = g.graphInfoV1.graphInputs; nOut = g.graphInfoV1.numGraphOutputs; outT = g.graphInfoV1.graphOutputs; }
        else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { nm = g.graphInfoV2.graphName; nIn = g.graphInfoV2.numGraphInputs; inT = g.graphInfoV2.graphInputs; nOut = g.graphInfoV2.numGraphOutputs; outT = g.graphInfoV2.graphOutputs; }
        else                                                          { nm = g.graphInfoV3.graphName; nIn = g.graphInfoV3.numGraphInputs; inT = g.graphInfoV3.graphInputs; nOut = g.graphInfoV3.numGraphOutputs; outT = g.graphInfoV3.graphOutputs; }
        std::string name = nm ? nm : "";
        // The plan names the graph; a context binary can hold several.
        if (!t.graph_name.empty() && name != t.graph_name) continue;
        auto info2 = std::make_unique<GraphInfo>();
        info2->name = name;
        info2->inputs.assign(inT, inT + nIn);
        info2->outputs.assign(outT, outT + nOut);
        info2->dimStorage.resize(nIn + nOut);
        info2->nameStorage.resize(nIn + nOut);
        for (size_t i = 0; i < info2->inputs.size(); ++i)
            rebind_tensor(info2->inputs[i], info2->dimStorage[i], info2->nameStorage[i]);
        for (size_t i = 0; i < info2->outputs.size(); ++i)
            rebind_tensor(info2->outputs[i], info2->dimStorage[nIn + i], info2->nameStorage[nIn + i]);
        CHECK(iface.graphRetrieve(t.ctx, info2->name.c_str(), &info2->graph));
        info2->inputBufs.resize(info2->inputs.size());
        for (size_t i = 0; i < info2->inputs.size(); ++i) {
            size_t sz = tensor_byte_size(info2->inputs[i]);
            info2->inputBufs[i].assign(sz, 0);
            set_tensor_buffer(info2->inputs[i], info2->inputBufs[i].data(),
                              static_cast<uint32_t>(sz));
        }
        info2->outputBufs.resize(info2->outputs.size());
        for (size_t i = 0; i < info2->outputs.size(); ++i) {
            size_t sz = tensor_byte_size(info2->outputs[i]);
            info2->outputBufs[i].assign(sz, 0);
            set_tensor_buffer(info2->outputs[i], info2->outputBufs[i].data(),
                              static_cast<uint32_t>(sz));
        }
        t.gi = std::move(info2);
        picked = t.gi->name;
        break;
    }
    g_sys.free(sysHandle);
    if (!t.gi) {
        std::fprintf(stderr, "graph '%s' not found in %s\n",
                     t.graph_name.c_str(), bin_path.c_str());
        std::exit(1);
    }
    t.graph_name = picked;
    // Cold-start burn-in: the first execute on a fresh context costs several
    // times steady state, and the cost model never saw that.
    for (int w = 0; w < g_warmup_execs; ++w) t.run();
    std::fprintf(stderr, "[bringup] %-4s %s -> %s\n", label.c_str(),
                 bin_path.c_str(), t.graph_name.c_str());
}

// ------------------------------------------------------------------ plan
struct NetSpec {
    std::string name, backend, lib;
    int         num_instances = 1;
    double      period_ms = -1.0;     // < 0: aperiodic
    double      window_ms = -1.0;
    std::string upstream;             // "" if none
    std::vector<std::pair<std::string, std::string>> tiles;  // (ctx, graph)
};

struct Config {
    int    reps = 3;
    int    warm = 1;
    double timeout_ms = 120000.0;
    std::vector<NetSpec> nets;
};

static Config parse_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open config %s\n", path.c_str()); std::exit(2); }
    Config c;
    std::vector<std::pair<std::string, std::string>> edges;
    std::string line;
    while (std::getline(f, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream is(line);
        std::string kw;
        if (!(is >> kw)) continue;
        if (kw == "reps")            is >> c.reps;
        else if (kw == "warm")       is >> c.warm;
        else if (kw == "timeout_ms") is >> c.timeout_ms;
        else if (kw == "net") {
            NetSpec n;
            is >> n.name >> n.backend >> n.lib >> n.num_instances
               >> n.period_ms >> n.window_ms;
            c.nets.push_back(n);
        } else if (kw == "tile") {
            if (c.nets.empty()) { std::fprintf(stderr, "tile before net\n"); std::exit(2); }
            std::string ctx, graph;
            is >> ctx >> graph;
            c.nets.back().tiles.emplace_back(ctx, graph);
        } else if (kw == "edge") {
            std::string a, b;
            is >> a >> b;
            edges.emplace_back(a, b);
        }
    }
    for (auto& e : edges)
        for (auto& n : c.nets)
            if (n.name == e.second) n.upstream = e.first;
    return c;
}

// ------------------------------------------------------------- the harness
struct Record {
    int    rep, pass_is_warm, instance;
    double release_ms, start_ms, end_ms, dep_wait_ms;
};

//: pass-global state, reset between passes
static std::atomic<double>   g_t0{0.0};
static std::atomic<int>      g_pass_done{0};
static std::atomic<int>      g_pass_total{0};
static std::mutex            g_done_mu;
static std::condition_variable g_done_cv;
static std::atomic<bool>     g_running{false};

// Pass barrier. Executor threads are created once and park here between
// passes, so every pass's timers are created while NOTHING is spinning. That
// is not a nicety: `create_wall_timer` returns the handle the callback needs
// to cancel itself, and with a spinning executor the 1 ns kick timer could
// fire before the assignment landed -- an intermittent null dereference that
// cost 8 of the first 68 runs before it was found. Parking also releases all
// nodes within tens of microseconds of one another, which a fresh
// std::thread per pass would not.
static std::mutex              g_bar_mu;
static std::condition_variable g_bar_cv;
static int                     g_gen = 0;
static int                     g_parked = 0;
static bool                    g_quit = false;

class PinnedNode : public rclcpp::Node {
public:
    PinnedNode(const NetSpec& spec, int index)
        : rclcpp::Node("pin_" + spec.name), spec_(spec), index_(index) {
        for (auto& t : spec_.tiles) {
            Tile tile;
            tile.graph_name = t.second;
            bringup(tile, t.first, spec_.lib, spec_.backend);
            tiles_.push_back(std::move(tile));
        }
        // A network with an incoming edge is driven by its upstream's
        // completion topic -- the wiring the micro-ROS reference could not
        // express and therefore silently dropped. Its instance k additionally
        // waits for instance k of its upstream, which is the same
        // instance-to-instance rule XPU-RT applies to `edges`.
        pub_ = create_publisher<std_msgs::msg::Int32>(
            "/pin/" + spec_.name + "/done", rclcpp::QoS(64).reliable());
        if (!spec_.upstream.empty()) {
            sub_ = create_subscription<std_msgs::msg::Int32>(
                "/pin/" + spec_.upstream + "/done", rclcpp::QoS(64).reliable(),
                [this](std_msgs::msg::Int32::SharedPtr msg) { on_upstream(msg->data); });
        }
    }

    const NetSpec& spec() const { return spec_; }
    const std::vector<Record>& records() const { return recs_; }

    void begin_pass(int rep, bool warm) {
        rep_ = rep;
        warm_ = warm;
        seq_ = 0;
        up_done_ = -1;
        if (kick_) { kick_->cancel(); kick_.reset(); }
        if (timer_) { timer_->cancel(); timer_.reset(); }
    }

    // Called from the driving thread once t0 is fixed. Creating a timer on a
    // spinning node is safe: rclcpp notifies the executor through the node's
    // guard condition.
    void arm() {
        if (!spec_.upstream.empty()) return;   // subscription-driven instead
        kick_ = create_wall_timer(std::chrono::nanoseconds(1), [this]() {
            kick_->cancel();
            drain();
        });
        if (spec_.period_ms > 0 && spec_.num_instances > 1) {
            auto per = std::chrono::duration<double, std::milli>(spec_.period_ms);
            timer_ = create_wall_timer(
                std::chrono::duration_cast<std::chrono::nanoseconds>(per),
                [this]() { drain(); });
        }
    }

private:
    double release_of(int k) const {
        return g_t0.load() + (spec_.period_ms > 0 ? spec_.period_ms * k : 0.0);
    }

    void execute_one(int k, double release, double dep_wait) {
        double start = now_ms();
        for (auto& t : tiles_) t.run();
        double end = now_ms();
        recs_.push_back(Record{rep_, warm_ ? 1 : 0, k,
                               release - g_t0.load(),
                               start - g_t0.load(),
                               end - g_t0.load(), dep_wait});
        auto msg = std::make_unique<std_msgs::msg::Int32>();
        msg->data = k;
        pub_->publish(std::move(msg));
        if (++g_pass_done == g_pass_total.load()) {
            std::lock_guard<std::mutex> lk(g_done_mu);
            g_done_cv.notify_all();
        }
    }

    // L2: drain every release that is already due. rcl's wall timer skips to
    // the next period boundary after an overrun, which would idle a busy node
    // for up to a period per late instance -- a penalty that has nothing to do
    // with pinning. XPU-RT gates an entry on its release and then runs it as
    // soon as the lane frees, so draining is the matched rule.
    void drain() {
        if (!g_running.load()) return;
        while (seq_ < spec_.num_instances) {
            double r = release_of(seq_);
            if (r > now_ms() + 1e-6) break;
            execute_one(seq_, r, 0.0);
            ++seq_;
        }
    }

    void on_upstream(int k) {
        if (!g_running.load()) return;
        up_done_ = std::max(up_done_, k);
        double arrive = now_ms();
        while (seq_ < spec_.num_instances && seq_ <= up_done_) {
            double r = release_of(seq_);
            double gate = std::max(r, arrive);
            double t = now_ms();
            if (gate > t) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double, std::milli>(gate - t));
            }
            execute_one(seq_, gate, arrive - r > 0 ? arrive - r : 0.0);
            ++seq_;
        }
    }

    NetSpec                 spec_;
    int                     index_;
    std::vector<Tile>       tiles_;
    std::vector<Record>     recs_;
    int                     rep_ = 0, seq_ = 0, up_done_ = -1;
    bool                    warm_ = false;
    rclcpp::TimerBase::SharedPtr kick_, timer_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_;
};

// ------------------------------------------------------- executor-floor mode
// L3: the empty-callback floor of one SingleThreadedExecutor on this board,
// measured rather than assumed, so a declared period below it is reported per
// cell instead of silently absorbed.
static int measure_floor(double period_ms, int iters) {
    auto node = std::make_shared<rclcpp::Node>("pin_floor");
    std::vector<double> gaps;
    gaps.reserve(iters);
    double last = 0;
    int n = 0;
    rclcpp::TimerBase::SharedPtr t;
    rclcpp::executors::SingleThreadedExecutor ex;
    auto per = std::chrono::duration<double, std::milli>(period_ms);
    t = node->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(per), [&]() {
            double now = now_ms();
            if (last > 0) gaps.push_back(now - last);
            last = now;
            if (++n >= iters) ex.cancel();
        });
    ex.add_node(node);
    ex.spin();
    std::sort(gaps.begin(), gaps.end());
    if (gaps.empty()) { std::printf("[floor] no samples\n"); return 1; }
    std::printf("[floor] period=%.3f n=%zu min=%.4f p50=%.4f p95=%.4f max=%.4f ms\n",
                period_ms, gaps.size(), gaps.front(),
                gaps[gaps.size() / 2], gaps[gaps.size() * 95 / 100], gaps.back());
    return 0;
}

// ------------------------------------------------------------------- main
int main(int argc, char** argv) {
    // Line-buffer stdout: it is piped through ssh, and a crash in teardown
    // must not be able to destroy results that were already printed.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string cfg_path;
    double floor_period = -1;
    int floor_iters = 2000;
    std::vector<char*> ros_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)            cfg_path = argv[++i];
        else if (a == "--floor" && i + 1 < argc)        floor_period = std::atof(argv[++i]);
        else if (a == "--floor-iters" && i + 1 < argc)  floor_iters = std::atoi(argv[++i]);
        else if (a == "--warm-execs" && i + 1 < argc)   g_warmup_execs = std::atoi(argv[++i]);
        else ros_args.push_back(argv[i]);
    }
    int rargc = static_cast<int>(ros_args.size());
    rclcpp::init(rargc, ros_args.data());

    if (floor_period > 0) {
        int rc = measure_floor(floor_period, floor_iters);
        rclcpp::shutdown();
        return rc;
    }
    if (cfg_path.empty()) {
        std::fprintf(stderr, "usage: pin_harness --config <file> [--floor <ms>]\n");
        rclcpp::shutdown();
        return 2;
    }

    Config cfg = parse_config(cfg_path);
    g_sys = load_system();

    // One node per network. Contexts are brought up here, before any pass, so
    // no pass pays a cold context.
    std::vector<std::shared_ptr<PinnedNode>> nodes;
    int total = 0;
    for (size_t i = 0; i < cfg.nets.size(); ++i) {
        nodes.push_back(std::make_shared<PinnedNode>(cfg.nets[i], static_cast<int>(i)));
        total += cfg.nets[i].num_instances;
    }
    g_pass_total = total;

    // One SingleThreadedExecutor per node, each on its own thread. That is the
    // deployment model being measured: N independent ROS nodes, no shared
    // scheduling decision between them.
    std::vector<std::unique_ptr<rclcpp::executors::SingleThreadedExecutor>> execs;
    std::vector<std::thread> threads;
    for (auto& n : nodes) {
        execs.push_back(std::make_unique<rclcpp::executors::SingleThreadedExecutor>());
        execs.back()->add_node(n);
    }
    for (auto& e : execs) {
        auto* ep = e.get();
        threads.emplace_back([ep]() {
            int mygen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(g_bar_mu);
                    ++g_parked;
                    g_bar_cv.notify_all();
                    g_bar_cv.wait(lk, [&]() { return g_quit || g_gen != mygen; });
                    if (g_quit) return;
                    mygen = g_gen;
                    --g_parked;
                }
                ep->spin();
            }
        });
    }
    // Let discovery settle before the first pass.
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    std::printf("[main] %zu node(s), %d instance(s) total\n", nodes.size(), total);
    for (auto& n : nodes)
        std::printf("  node %-18s backend=%-4s inst=%2d period=%8.3f window=%8.3f "
                    "tiles=%zu%s\n",
                    n->spec().name.c_str(), n->spec().backend.c_str(),
                    n->spec().num_instances, n->spec().period_ms,
                    n->spec().window_ms, n->spec().tiles.size(),
                    n->spec().upstream.empty() ? ""
                        : (" <- " + n->spec().upstream).c_str());

    struct PassResult { int rep; bool warm; double makespan, np_makespan; int done; };
    std::vector<PassResult> passes;

    auto wait_parked = [&]() {
        std::unique_lock<std::mutex> lk(g_bar_mu);
        g_bar_cv.wait(lk, [&]() { return g_parked == static_cast<int>(nodes.size()); });
    };
    auto release = [&]() {
        {
            std::lock_guard<std::mutex> lk(g_bar_mu);
            ++g_gen;
        }
        g_bar_cv.notify_all();
    };

    auto run_pass = [&](int rep, bool warm) -> PassResult {
        wait_parked();                       // nothing is spinning
        for (auto& n : nodes) n->begin_pass(rep, warm);
        for (auto& n : nodes) n->arm();      // timers created off the executor
        g_pass_done = 0;
        double t0 = now_ms();
        g_t0 = t0;
        g_running = true;
        release();                           // every executor starts together
        {
            std::unique_lock<std::mutex> lk(g_done_mu);
            g_done_cv.wait_for(lk,
                std::chrono::duration<double, std::milli>(cfg.timeout_ms),
                [&]() { return g_pass_done.load() >= total; });
        }
        g_running = false;
        for (auto& e : execs) e->cancel();   // spin() returns; threads re-park
        double mk = 0, np = 0;
        bool any_np = false;
        for (auto& n : nodes) {
            for (auto& r : n->records()) {
                if (r.rep != rep || r.pass_is_warm != (warm ? 1 : 0)) continue;
                mk = std::max(mk, r.end_ms);
                if (n->spec().period_ms <= 0) { np = std::max(np, r.end_ms); any_np = true; }
            }
        }
        return PassResult{rep, warm, mk, any_np ? np : mk, g_pass_done.load()};
    };

    for (int rep = 1; rep <= cfg.reps; ++rep) {
        for (int w = 0; w < cfg.warm; ++w) run_pass(rep, true);
        passes.push_back(run_pass(rep, false));
    }

    std::printf("=== ROS_PINSWEEP_TRACE_BEGIN ===\n");
    std::printf("rep,warm,network,backend,instance,release_ms,start_ms,end_ms,dep_wait_ms,period_ms,window_ms\n");
    for (auto& n : nodes)
        for (auto& r : n->records())
            std::printf("%d,%d,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        r.rep, r.pass_is_warm, n->spec().name.c_str(),
                        n->spec().backend.c_str(), r.instance,
                        r.release_ms, r.start_ms, r.end_ms, r.dep_wait_ms,
                        n->spec().period_ms, n->spec().window_ms);
    std::printf("=== ROS_PINSWEEP_TRACE_END ===\n");

    for (auto& p : passes)
        std::printf("[summary] rep=%d executed=%d/%d makespan=%.4f np_makespan=%.4f\n",
                    p.rep, p.done, total, p.makespan, p.np_makespan);

    wait_parked();
    {
        std::lock_guard<std::mutex> lk(g_bar_mu);
        g_quit = true;
    }
    g_bar_cv.notify_all();
    for (auto& t : threads) if (t.joinable()) t.join();
    rclcpp::shutdown();
    return 0;
}
