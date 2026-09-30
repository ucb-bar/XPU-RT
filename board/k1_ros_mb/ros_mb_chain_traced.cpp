// The ROS 2 baseline, traced the way the XPU-RT harness traces itself.
//
// Same four-node graph as the deployed pipeline -- camera timer -> perception (yolov8_nano_64x96)
// -> nav (fused_full) -> control (mlp_control) -- running the same ModelBlaster-generated kernels
// XPU-RT runs, on the same board. Control has the deployed shape: a free-running --ctrl-hz timer
// that acts on the latest goal it has been handed; the goal subscription only updates that goal.
//
// Every callback is written as one row of the XPU-RT trace contract (which hart it actually ran
// on, rdtime at entry and exit), so the two runtimes' Gantt rows are built by the same code from
// the same kind of file. Raw releases, goal arrivals, control fires and their gaps are written as
// well; nothing is summarised on the board and nothing is dropped -- the host applies the warm-up
// window and records it.
//
// Things this program deliberately does NOT do, because each would decide the answer in advance:
//   * it never pins itself -- placement is whatever the launcher (taskset) gave it, and the
//     manifest records the affinity mask and scheduling policy it actually ran under;
//   * it never calls spin_some(): a spin_some loop runs ready work on the calling thread and
//     collapses a MultiThreadedExecutor onto one core;
//   * it keeps rclcpp's default mutually-exclusive callback group per node: a generated model
//     writes static buffers and must never overlap itself; different nodes still run in parallel
//     under the multi-threaded executor, which is the point of that executor.
//
// Build/run recipe: docs/ros_baseline_reproduction.md.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "rmw/rmw.h"
#ifdef MB_WITH_POOL
#include "modelblaster_pool.h"
#endif

extern "C" {
// Three arguments: the generated entry points take an optional worker pool (NULL = sequential).
void run_model_yolov8_nano_64x96(const signed char *in, signed char *out, void *pool);
void run_model_fused_full(const signed char *in, signed char *out, void *pool);
void run_model_mlp_control(const signed char *in, signed char *out, void *pool);
// the heavier stack: a transformer block and a second camera network, each on its own timer
void run_model_ffn_block(const signed char *in, signed char *out, void *pool) __attribute__((weak));
void run_model_dronet(const signed char *in, signed char *out, void *pool) __attribute__((weak));
// Per-op records the generated model already keeps: one entry per dispatch, in rdtime ticks
// (the generated `rdcycle` is rdtime on this kernel). Weak so a build without them still links.
struct mb_op_record { int dispatch_id; const char *name; const char *op; const char *shape; unsigned long cycles; };
const struct mb_op_record *model_yolov8_nano_64x96_profile_records(int *count) __attribute__((weak));
const struct mb_op_record *model_fused_full_profile_records(int *count) __attribute__((weak));
const struct mb_op_record *model_mlp_control_profile_records(int *count) __attribute__((weak));
}

// which model's per-op records to drain: every node passes its own, so nav and control are
// recorded at the same granularity as perception instead of one bar per callback.
typedef const struct mb_op_record *(*mb_records_fn)(int *);

static inline uint64_t rdtime() { uint64_t t; asm volatile("rdtime %0" : "=r"(t)); return t; }
static const double TICKS_PER_MS = 24000.0;       // 24 MHz rdtime on the K1
static int64_t epoch_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

struct Row { const char *net; const char *node; int64_t inst; uint64_t t0, t1; int cpu; };
// One executed unit below a callback: an op of the network, or one shard of a pooled op on the
// hart that ran it. Written into the same trace contract as a callback row, so both runtimes'
// schedule rows are built from the same file by the same code.
struct SubRow { const char *net; const char *node; int64_t inst; uint64_t t0, t1; int cpu;
                int dispatch_id; std::string op; std::string name; };
struct Stamp { int64_t seq; uint64_t t; };

// Lay one callback's executed work into the trace contract.
//
// Two sources, both measured, neither invented:
//   * the generated model's per-op records — one entry per dispatch, already in rdtime ticks.
//     They carry durations, not absolute stamps, so the ops are laid end to end across the
//     callback's measured span in dispatch order, which is the order the model executes them.
//   * the pool's per-shard records — the hart a slice actually ran on, with its own rdtime
//     start and end. These are absolute and are written through unchanged.
// A pooled op therefore appears once per hart that ran it, at the time it ran, and a
// sequential op appears once on the callback's hart.
#ifdef MB_WITH_POOL
static void emit_detail(std::vector<SubRow> &out, const char *net, const char *node,
                        int64_t inst, uint64_t a, uint64_t b, int cpu, void *pool,
                        const modelblaster_pool_shard *sh, mb_records_fn recs_fn) {
  auto *p = (modelblaster_pool_t)pool;
  unsigned ns = p ? modelblaster_pool_trace_count(p) : 0u;
#else
static void emit_detail(std::vector<SubRow> &out, const char *net, const char *node,
                        int64_t inst, uint64_t a, uint64_t b, int cpu, void *pool,
                        const void *sh, mb_records_fn recs_fn) {
  (void)pool; (void)sh; unsigned ns = 0u;
#endif
  int nops = 0;
  const struct mb_op_record *recs =
      recs_fn ? recs_fn(&nops) : nullptr;
  if (recs && nops > 0) {
    unsigned long long tot = 0; for (int i = 0; i < nops; i++) tot += recs[i].cycles;
    uint64_t span = (b > a) ? (b - a) : 0, acc = 0;
    for (int i = 0; i < nops; i++) {
      uint64_t s0 = a + (tot ? (uint64_t)((__uint128_t)acc * span / tot) : 0);
      acc += recs[i].cycles;
      uint64_t s1 = a + (tot ? (uint64_t)((__uint128_t)acc * span / tot) : 0);
      if (s1 <= s0) s1 = s0 + 1;
      out.push_back({net, node, inst, s0, s1, cpu, recs[i].dispatch_id,
                     recs[i].op ? recs[i].op : "op", recs[i].name ? recs[i].name : "?"});
    }
  }
#ifdef MB_WITH_POOL
  for (unsigned k = 0; k < ns; k++) {
    const modelblaster_pool_shard &r = sh[k];
    out.push_back({net, node, inst, r.t0, r.t1, r.hart, (int)r.call, "pool_shard", "shard"});
  }
  if (p) modelblaster_pool_trace_reset(p);
#endif
}

int main(int argc, char **argv) {
  double rate = 15.0, ctrl_hz = 100.0, secs = 20.0;
  std::string exec = "single", nodes = "camera,perception,nav,control", out_dir, tag, ksha;
  uint64_t t0_arg = 0; int pool_n = 0; std::string pool_harts; int pin_main = -1;
  // Nav and control can have pools of their own, on harts the YOLO pool does not use: the whole
  // machine partitioned between the three networks rather than four harts for perception and the
  // rest to whatever the operating system does with the callback threads.
  int nav_pool_n = 0, ctl_pool_n = 0; std::string nav_pool_harts, ctl_pool_harts;
  std::string ctrl_mode = "timer";   // timer: free-running --ctrl-hz on the held goal; chained: run on each goal
  std::string extra;                 // "ffn_block:10,dronet:30" -- independent periodic nodes, each on its own timer
  int qos_depth = 10;                // KEEP_LAST depth on every topic; 1 = newest frame only
  int cameras = 1;                   // 2 = a second camera+perception pair on frame2/detections2; nav consumes both
  bool alternate = false;            // one camera, odd frames on frame2: two perception processes take turns (pipelining by hand)
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](){ return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
    if      (a == "--rate")        rate = atof(next().c_str());
    else if (a == "--ctrl-hz")     ctrl_hz = atof(next().c_str());
    else if (a == "--seconds")     secs = atof(next().c_str());
    else if (a == "--executor")    exec = next();
    else if (a == "--nodes")       nodes = next();
    else if (a == "--t0")          t0_arg = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--out-dir")     out_dir = next();
    else if (a == "--tag")         tag = next();
    else if (a == "--kernels-sha") ksha = next();
    else if (a == "--yolo-pool")   pool_n = atoi(next().c_str());
    else if (a == "--pool-harts")  pool_harts = next();
    else if (a == "--nav-pool")    nav_pool_n = atoi(next().c_str());
    else if (a == "--nav-harts")   nav_pool_harts = next();
    else if (a == "--ctrl-pool")   ctl_pool_n = atoi(next().c_str());
    else if (a == "--ctrl-harts")  ctl_pool_harts = next();
    else if (a == "--pin-main")    pin_main = atoi(next().c_str());
    else if (a == "--ctrl-mode")   ctrl_mode = next();
    else if (a == "--extra")       extra = next();
    else if (a == "--qos-depth")   qos_depth = atoi(next().c_str());
    else if (a == "--cameras")     cameras = atoi(next().c_str());
    else if (a == "--alternate")   alternate = true;
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  // Exact match on the comma list. A substring test lets "--nodes perception2" also select
  // "perception", so that process subscribes to the first camera too and re-runs the network
  // on frames another process is already handling.
  auto has = [&](const char *n){
    const std::string t(n);
    for (size_t i = 0; i <= nodes.size(); ) {
      size_t j = nodes.find(',', i);
      if (j == std::string::npos) j = nodes.size();
      if (nodes.compare(i, j - i, t) == 0) return true;
      i = j + 1;
    }
    return false;
  };
  if (tag.empty()) tag = std::to_string((int)rate) + "_" + exec;
  if (out_dir.empty()) out_dir = "/root/ros_mb/out/" + tag;
  mkdir("/root/ros_mb/out", 0755); mkdir(out_dir.c_str(), 0755);

  // Placement of the executor thread, only when asked for explicitly. The pool's helpers are
  // pinned by the pool itself, so pinning this thread does not leak into them.
  if (pin_main >= 0) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(pin_main, &s);
    if (sched_setaffinity(0, sizeof s, &s) != 0) { perror("sched_setaffinity"); return 3; }
  }

  // Detail rows for the schedule panel: the ops of each callback and, when a pool runs them,
  // the slices with the hart that took each one. Declared before the pool so tracing can be
  // armed at creation.
  std::vector<SubRow> subrows; subrows.reserve(64 * 1024);
#ifdef MB_WITH_POOL
  std::vector<modelblaster_pool_shard> shardbuf(4096);
  // The navigation node keeps its own slice buffer: in a single-process arm its callback and
  // perception's can be in flight at once, and one buffer would have them overwrite each other.
  std::vector<modelblaster_pool_shard> navshardbuf(4096);
#define SHARDS shardbuf.data()
#define NAV_SHARDS navshardbuf.data()
#else
#define SHARDS nullptr
#define NAV_SHARDS nullptr
#endif
  void *pool = nullptr;
#ifdef MB_WITH_POOL
  std::vector<int> harts;
  if (pool_n > 0) {
    for (size_t p = 0; p < pool_harts.size();) {
      size_t q = pool_harts.find(',', p); if (q == std::string::npos) q = pool_harts.size();
      harts.push_back(atoi(pool_harts.substr(p, q - p).c_str())); p = q + 1;
    }
    // No hart list means an UNPINNED pool: the workers are ordinary threads and the operating
    // system places and migrates them, which is what a deployment gets if it does not pin.
    if (!harts.empty() && (int)harts.size() != pool_n) { fprintf(stderr, "--pool-harts must list %d harts\n", pool_n); return 4; }
    pool = harts.empty() ? modelblaster_pool_create(pool_n)
                         : modelblaster_pool_create_on_harts(pool_n, harts.data());
    if (!pool) { fprintf(stderr, "pool create failed\n"); return 5; }
    // Record every slice the pool runs: which hart took it and when. Without this a pooled
    // op is one interval on the calling hart and the helpers' work is invisible.
    modelblaster_pool_trace_arm((modelblaster_pool_t)pool, shardbuf.data(), (unsigned)shardbuf.size());
  }
#else
  if (pool_n > 0) { fprintf(stderr, "built without MB_WITH_POOL\n"); return 4; }
#endif
  // A pool per network. Each has its own shard buffer and each is armed, so a Gantt drawn from
  // this trace shows the navigation pool's spread the same way it shows perception's; with only
  // one armed the other network's helpers are invisible and its harts read idle.
  void *nav_pool = nullptr, *ctl_pool = nullptr;
#ifdef MB_WITH_POOL
  auto make_pool = [](int n, const std::string &list, const char *what) -> void * {
    if (n <= 0) return nullptr;
    std::vector<int> h;
    for (size_t p = 0; p < list.size();) {
      size_t q = list.find(',', p); if (q == std::string::npos) q = list.size();
      h.push_back(atoi(list.substr(p, q - p).c_str())); p = q + 1;
    }
    // As for the YOLO pool: no hart list means an unpinned pool, the operating system placing
    // the workers, which is what a deployment gets if it does not pin.
    if (!h.empty() && (int)h.size() != n) { fprintf(stderr, "%s must list %d harts\n", what, n); exit(4); }
    void *pp = h.empty() ? modelblaster_pool_create(n) : modelblaster_pool_create_on_harts(n, h.data());
    if (!pp) { fprintf(stderr, "%s pool create failed\n", what); exit(5); }
    return pp;
  };
  nav_pool = make_pool(nav_pool_n, nav_pool_harts, "--nav-harts");
  if (nav_pool) modelblaster_pool_trace_arm((modelblaster_pool_t)nav_pool, navshardbuf.data(), (unsigned)navshardbuf.size());
  ctl_pool = make_pool(ctl_pool_n, ctl_pool_harts, "--ctrl-harts");
#else
  if (nav_pool_n > 0 || ctl_pool_n > 0) { fprintf(stderr, "built without MB_WITH_POOL\n"); return 4; }
#endif

  static std::vector<signed char> yi(18432, 1), yo(8316), ni(5698, 1), no(2), ci(16, 1), co(4);
  const uint64_t T0 = t0_arg ? t0_arg : rdtime();
  const uint64_t period_ticks = (uint64_t)(TICKS_PER_MS * 1000.0 / rate);

  std::vector<Row> rows; rows.reserve((size_t)(rate * secs * 3 + ctrl_hz * secs) + 64);
  std::vector<Stamp> released, goals, consumed; std::vector<uint64_t> fires;
  std::mutex mu;                     // guards rows/stamps across executor threads
  std::atomic<int64_t> cam_seq{0}; uint64_t cam_first = 0;
  struct { int64_t seq = -1; uint64_t t_rel = 0; } held; int64_t last_consumed = -1;
  std::mutex hmu;

  rclcpp::init(argc, argv);
  std::vector<std::shared_ptr<rclcpp::Node>> all;
  std::shared_ptr<rclcpp::Node> cam, percep, nav, ctrl;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr p_cam, p_det, p_goal, p_cam2, p_det2;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr s_percep, s_nav, s_ctrl, s_percep2, s_nav2;
  rclcpp::TimerBase::SharedPtr t_cam, t_ctrl, t_cam2;
  std::shared_ptr<rclcpp::Node> cam2, percep2;
  static std::vector<signed char> yi2(18432, 1), yo2(8316);
  std::atomic<int64_t> cam2_seq{0};
  auto parse = [](const std::string &d, int64_t &seq, uint64_t &t_rel) {
    seq = strtoll(d.c_str(), nullptr, 10);
    const char *sp = strchr(d.c_str(), ' '); t_rel = sp ? strtoull(sp + 1, nullptr, 10) : 0;
  };

  if (has("camera")) {
    cam = std::make_shared<rclcpp::Node>("camera"); all.push_back(cam);
    p_cam = cam->create_publisher<std_msgs::msg::String>("frame", qos_depth);
    if (alternate) p_cam2 = cam->create_publisher<std_msgs::msg::String>("frame2", qos_depth);   // odd frames
  }
  // every camera's YOLO callback goes through one mutually exclusive group: there is one model
  // instance (static buffers), so two cameras in one process must take turns on it
  rclcpp::CallbackGroup::SharedPtr yolo_group; rclcpp::SubscriptionOptions yolo_opt;
  if (has("perception")) {
    percep = std::make_shared<rclcpp::Node>("perception"); all.push_back(percep);
    yolo_group = percep->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive); yolo_opt.callback_group = yolo_group;
    p_det = percep->create_publisher<std_msgs::msg::String>("detections", qos_depth);
    s_percep = percep->create_subscription<std_msgs::msg::String>("frame", qos_depth,
      [&](std_msgs::msg::String::UniquePtr m) {
        int64_t seq; uint64_t tr; parse(m->data, seq, tr);
        uint64_t a = rdtime(); run_model_yolov8_nano_64x96(yi.data(), yo.data(), pool); uint64_t b = rdtime();
        int cpu = sched_getcpu();
        { std::lock_guard<std::mutex> g(mu);
          rows.push_back({"yolov8_nano_64x96", "perception", seq, a, b, cpu});
          emit_detail(subrows, "yolov8_nano_64x96", "perception", seq, a, b, cpu, pool, SHARDS,
                      model_yolov8_nano_64x96_profile_records); }
        p_det->publish(*m);
      }, yolo_opt);
  }
  if (cameras >= 2 && has("camera2")) {
    cam2 = std::make_shared<rclcpp::Node>("camera2"); all.push_back(cam2);
    p_cam2 = cam2->create_publisher<std_msgs::msg::String>("frame2", qos_depth);
  }
  if (cameras >= 2 && has("perception2")) {
    // the second camera's perception: the same network on the second stream. In the same
    // process as the first it joins the first's callback group (one model instance, so the two
    // take turns; a second process is the layout that runs both at once); alone it is its own node.
    rclcpp::SubscriptionOptions opt2;
    if (percep) { percep2 = percep; opt2 = yolo_opt; }
    else { percep2 = std::make_shared<rclcpp::Node>("perception2"); all.push_back(percep2); }
    p_det2 = percep2->create_publisher<std_msgs::msg::String>("detections2", qos_depth);
    s_percep2 = percep2->create_subscription<std_msgs::msg::String>("frame2", qos_depth,
      [&](std_msgs::msg::String::UniquePtr m) {
        int64_t seq; uint64_t tr; parse(m->data, seq, tr);
        uint64_t a = rdtime(); run_model_yolov8_nano_64x96(yi2.data(), yo2.data(), pool); uint64_t b = rdtime();
        int cpu = sched_getcpu();
        { std::lock_guard<std::mutex> g(mu); rows.push_back({"yolov8_nano_64x96", "perception2", 1000000 + seq, a, b, cpu});
          // The same detail the first pool records. Without it this node reports one row per frame
          // and nothing else, so a two-pool arm is drawn as a single bar the length of the whole
          // network on whichever hart the callback happened to run -- the work its pool did on the
          // other harts is not in the trace at all, and those harts are drawn idle while the
          // sampler reads them busy. Sharing SHARDS with the first pool is safe in both layouts:
          // in one process the two callbacks are in one mutually exclusive group, and in the
          // two-process layout each process runs only one of them.
          emit_detail(subrows, "yolov8_nano_64x96", "perception2", 1000000 + seq, a, b, cpu, pool, SHARDS,
                      model_yolov8_nano_64x96_profile_records); }
        p_det2->publish(*m);
      }, opt2);
  }
  if (has("nav")) {
    nav = std::make_shared<rclcpp::Node>("nav"); all.push_back(nav);
    p_goal = nav->create_publisher<std_msgs::msg::String>("goal", qos_depth);
    auto nav_cb = [&](std_msgs::msg::String::UniquePtr m) {
        int64_t seq; uint64_t tr; parse(m->data, seq, tr);
        uint64_t a = rdtime(); run_model_fused_full(ni.data(), no.data(), nav_pool); uint64_t b = rdtime();
        int cpu = sched_getcpu();
        { std::lock_guard<std::mutex> g(mu); rows.push_back({"fused_full", "nav", seq, a, b, cpu});
          // nav_pool and its slice buffer rather than nullptr: with a --nav-pool the navigation
          // network's convolutions run on worker harts, and without these the slices are never
          // recorded, so the schedule panel can only draw the callback's own hart and the pool's
          // harts appear idle while the sampler shows them busy.
          emit_detail(subrows, "fused_full", "nav", seq, a, b, cpu, nav_pool, NAV_SHARDS,
                      model_fused_full_profile_records); }
        p_goal->publish(*m);
      };
    s_nav = nav->create_subscription<std_msgs::msg::String>("detections", qos_depth, nav_cb);
    if (cameras >= 2) s_nav2 = nav->create_subscription<std_msgs::msg::String>("detections2", qos_depth, nav_cb);
  }
  if (has("control")) {
    ctrl = std::make_shared<rclcpp::Node>("control"); all.push_back(ctrl);
    s_ctrl = ctrl->create_subscription<std_msgs::msg::String>("goal", qos_depth,
      [&](std_msgs::msg::String::UniquePtr m) {
        int64_t seq; uint64_t tr; parse(m->data, seq, tr);
        uint64_t t = rdtime();
        { std::lock_guard<std::mutex> g(hmu); held.seq = seq; held.t_rel = tr; }
        { std::lock_guard<std::mutex> g(mu); goals.push_back({seq, t}); }
        if (ctrl_mode == "chained") {          // the classic topic-chained pipeline: control per goal
          uint64_t a = rdtime(); run_model_mlp_control(ci.data(), co.data(), ctl_pool); uint64_t b = rdtime();
          int cpu = sched_getcpu();
          std::lock_guard<std::mutex> g(mu);
          rows.push_back({"mlp_control", "control", (int64_t)fires.size(), a, b, cpu});
          emit_detail(subrows, "mlp_control", "control", (int64_t)fires.size(), a, b, cpu, nullptr, nullptr,
                      model_mlp_control_profile_records);
          fires.push_back(a); consumed.push_back({seq, a}); last_consumed = seq;
        }
      });
    if (ctrl_mode == "timer")
    t_ctrl = ctrl->create_wall_timer(std::chrono::microseconds((long)(1e6 / ctrl_hz)), [&]() {
      uint64_t a = rdtime(); run_model_mlp_control(ci.data(), co.data(), ctl_pool); uint64_t b = rdtime();
      int cpu = sched_getcpu();
      int64_t seq; { std::lock_guard<std::mutex> g(hmu); seq = held.seq; }
      std::lock_guard<std::mutex> g(mu);
      rows.push_back({"mlp_control", "control", (int64_t)fires.size(), a, b, cpu});
      emit_detail(subrows, "mlp_control", "control", (int64_t)fires.size(), a, b, cpu, nullptr, nullptr,
                  model_mlp_control_profile_records);
      fires.push_back(a);
      if (seq != last_consumed) { consumed.push_back({seq, a}); last_consumed = seq; }
    });
  }

  // extra periodic nodes: name:hz pairs; each runs its network on a wall timer, traced like the rest
  static std::vector<signed char> fi(32768, 1), fo(32768), di(37632, 1), dout(2);
  std::vector<std::shared_ptr<rclcpp::Node>> extra_nodes; std::vector<rclcpp::TimerBase::SharedPtr> extra_timers;
  std::vector<int64_t> extra_fires;
  for (size_t p = 0; p < extra.size();) {
    size_t q = extra.find(',', p); if (q == std::string::npos) q = extra.size();
    std::string item = extra.substr(p, q - p); p = q + 1;
    size_t c = item.find(':'); if (c == std::string::npos) continue;
    std::string net = item.substr(0, c); double hz = atof(item.substr(c + 1).c_str());
    if ((net == "ffn_block" && !run_model_ffn_block) || (net == "dronet" && !run_model_dronet)) {
      fprintf(stderr, "extra net %s not linked into this binary\n", net.c_str()); return 7;
    }
    auto n = std::make_shared<rclcpp::Node>(net); all.push_back(n); extra_nodes.push_back(n);
    size_t idx = extra_fires.size(); extra_fires.push_back(0);
    const char *nname = (net == "ffn_block") ? "ffn_block" : "dronet";
    extra_timers.push_back(n->create_wall_timer(std::chrono::microseconds((long)(1e6 / hz)), [&, nname, idx]() {
      uint64_t a = rdtime();
      if (nname[0] == 'f') run_model_ffn_block(fi.data(), fo.data(), nullptr); else run_model_dronet(di.data(), dout.data(), nullptr);
      uint64_t b = rdtime(); int cpu = sched_getcpu();
      std::lock_guard<std::mutex> g(mu);
      rows.push_back({nname, nname, extra_fires[idx]++, a, b, cpu});
    }));
  }

  std::shared_ptr<rclcpp::Executor> ex;
  if (exec == "multi") ex = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  else                 ex = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  for (auto &n : all) ex->add_node(n);
  const unsigned ex_threads = (exec == "multi") ? std::thread::hardware_concurrency() : 1;

  // Discovery gate: nothing is released until every publisher in this process has a matched
  // subscription and every subscription a matched publisher, so DDS discovery is not measured
  // as latency. A subscription's publisher count is checked for the terminal node too.
  {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    auto ok = [&]() {
      bool r = true;
      if (p_cam)  r &= p_cam->get_subscription_count()  >= 1;
      if (p_det)  r &= p_det->get_subscription_count()  >= 1;
      if (p_goal) r &= p_goal->get_subscription_count() >= 1;
      if (s_percep) r &= s_percep->get_publisher_count() >= 1;
      if (s_nav)    r &= s_nav->get_publisher_count()    >= 1;
      if (s_ctrl)   r &= s_ctrl->get_publisher_count()   >= 1;
      if (p_cam2)   r &= p_cam2->get_subscription_count() >= 1;
      if (p_det2)   r &= p_det2->get_subscription_count() >= 1;
      if (s_percep2) r &= s_percep2->get_publisher_count() >= 1;
      return r;
    };
    while (!ok() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!ok()) { fprintf(stderr, "discovery gate timed out\n"); rclcpp::shutdown(); return 6; }
  }
  const int64_t wall_start = epoch_ms();
  const uint64_t run_t0 = rdtime();
  if (cam) {
    t_cam = cam->create_wall_timer(std::chrono::microseconds((long)(1e6 / rate)), [&]() {
      int64_t seq = cam_seq++; uint64_t t = rdtime(); if (seq == 0) cam_first = t;
      std_msgs::msg::String m; m.data = std::to_string(seq) + " " + std::to_string(t);
      if (alternate && (seq & 1)) p_cam2->publish(m); else p_cam->publish(m);
      std::lock_guard<std::mutex> g(mu); released.push_back({seq, t});
    });
  }
  if (cam2) {
    t_cam2 = cam2->create_wall_timer(std::chrono::microseconds((long)(1e6 / rate)), [&]() {
      int64_t seq = cam2_seq++; uint64_t t = rdtime();
      std_msgs::msg::String m; m.data = std::to_string(1000000 + seq) + " " + std::to_string(t);
      p_cam2->publish(m);
      std::lock_guard<std::mutex> g(mu); released.push_back({1000000 + seq, t});
    });
  }
  std::thread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds((long)(secs * 1000.0)));
    ex->cancel();
  });
  ex->spin();
  stopper.join();
  const int64_t wall_end = epoch_ms();
  const uint64_t run_end = rdtime();

  // ---- files -------------------------------------------------------------------------------
  auto rel = [&](uint64_t t) { return (long long)(t - T0); };
  {
    std::vector<Row> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), [](const Row &a, const Row &b){ return a.t0 < b.t0; });
    std::ofstream f(out_dir + "/trace.csv");
    f << "entry_id,network,instance,dispatch_id,op,name,core_kind,hart,predicted_start_ms,"
         "predicted_duration_ms,worker_kind_idx,worker_hart,actual_start_cycles,actual_end_cycles\n";
    long id = 0;
    for (auto &r : sorted)
      f << id++ << "," << r.net << "," << r.inst << ",0,node_callback," << r.node << ","
        << (r.cpu < 4 ? "rvv" : "rvv_c1") << "," << r.cpu << ",0,0," << (r.cpu < 4 ? 0 : 1) << ","
        << r.cpu << "," << rel(r.t0) << "," << rel(r.t1) << "\n";
    // the executed units below each callback, same contract: a reader that wants callbacks
    // filters op == node_callback, one that wants the schedule takes the rest
    std::vector<SubRow> ssorted = subrows;
    std::sort(ssorted.begin(), ssorted.end(), [](const SubRow &a, const SubRow &b){ return a.t0 < b.t0; });
    for (auto &r : ssorted)
      f << id++ << "," << r.net << "," << r.inst << "," << r.dispatch_id << "," << r.op << ","
        << r.name << "," << (r.cpu < 4 ? "rvv" : "rvv_c1") << "," << r.cpu << ",0,0,"
        << (r.cpu < 4 ? 0 : 1) << "," << r.cpu << "," << rel(r.t0) << "," << rel(r.t1) << "\n";
  }
  if (cam) {
    std::ofstream f(out_dir + "/released.csv"); f << "frame_seq,t_nominal_ticks,t_release_ticks\n";
    for (auto &s : released)
      f << s.seq << "," << (long long)(cam_first - T0 + (uint64_t)s.seq * period_ticks) << "," << rel(s.t) << "\n";
  }
  if (ctrl) {
    { std::ofstream f(out_dir + "/goals.csv"); f << "frame_seq,t_goal_ticks\n";
      for (auto &s : goals) f << s.seq << "," << rel(s.t) << "\n"; }
    { std::ofstream f(out_dir + "/consumed.csv"); f << "frame_seq,t_ctrl_ticks\n";
      for (auto &s : consumed) f << s.seq << "," << rel(s.t) << "\n"; }
    { std::ofstream f(out_dir + "/ctrl_gaps.csv"); f << "fire_idx,t_fire_ticks,gap_ms\n";
      for (size_t i = 1; i < fires.size(); i++)
        f << i << "," << rel(fires[i]) << "," << (fires[i] - fires[i - 1]) / TICKS_PER_MS << "\n"; }
  }
  {
    cpu_set_t s; CPU_ZERO(&s); sched_getaffinity(0, sizeof s, &s);
    unsigned mask = 0; for (int c = 0; c < 8; c++) if (CPU_ISSET(c, &s)) mask |= 1u << c;
    int pol = sched_getscheduler(0); sched_param sp{}; sched_getparam(0, &sp);
    char host[128] = {0}; gethostname(host, sizeof host - 1); utsname u{}; uname(&u);
    std::ofstream f(out_dir + "/manifest.json");
    f << "{\n"
      << "  \"tag\": \"" << tag << "\", \"nodes\": \"" << nodes << "\",\n"
      << "  \"rate_hz\": " << rate << ", \"ctrl_hz\": " << ctrl_hz << ", \"seconds\": " << secs << ",\n"
      << "  \"executor\": \"" << exec << "\", \"executor_threads\": " << ex_threads << ", \"ctrl_mode\": \"" << ctrl_mode << "\",\n"
      << "  \"callback_groups\": \"mutually_exclusive_per_node; both cameras' YOLO in one group\", \"qos_depth\": " << qos_depth << ",\n"
      << "  \"alternate\": " << (alternate ? "true" : "false") << ",\n"
      << "  \"rmw\": \"" << rmw_get_implementation_identifier() << "\",\n"
      << "  \"nav_pool\": " << nav_pool_n << ", \"nav_harts\": \"" << nav_pool_harts << "\",\n"
      << "  \"yolo_pool\": " << pool_n << ", \"pool_harts\": \"" << pool_harts << "\", \"pin_main\": " << pin_main << ", \"extra\": \"" << extra << "\", \"cameras\": " << cameras << ",\n"
      << "  \"t0_rdtime\": " << T0 << ", \"run_t0_ticks\": " << rel(run_t0) << ", \"run_end_ticks\": " << rel(run_end) << ",\n"
      << "  \"wall_start_epoch_ms\": " << wall_start << ", \"wall_end_epoch_ms\": " << wall_end << ",\n"
      << "  \"n_released\": " << released.size() << ", \"n_goals\": " << goals.size()
      << ", \"n_ctrl_fires\": " << fires.size() << ", \"n_consumed\": " << consumed.size() << ",\n"
      << "  \"affinity_mask\": \"0x" << std::hex << mask << std::dec << "\", \"sched_policy\": "
      << (pol == SCHED_FIFO ? "\"SCHED_FIFO\"" : pol == SCHED_RR ? "\"SCHED_RR\"" : "\"SCHED_OTHER\"")
      << ", \"rt_priority\": " << sp.sched_priority << ",\n"
      << "  \"instance_meaning\": \"perception/nav: camera frame seq; control: timer fire index\",\n"
      << "  \"kernels_sha\": \"" << ksha << "\", \"hostname\": \"" << host << "\", \"kernel\": \"" << u.release << "\"\n"
      << "}\n";
  }
  // One line for the log, so a broken run is visible without opening files.
  double gmax = 0, gsum = 0;
  for (size_t i = 1; i < fires.size(); i++) { double g = (fires[i] - fires[i-1]) / TICKS_PER_MS; gsum += g; if (g > gmax) gmax = g; }
  printf("ROS_TRACED tag=%s exec=%s rate=%.0f released=%zu goals=%zu fires=%zu gap_mean=%.2f gap_max=%.2f\n",
         tag.c_str(), exec.c_str(), rate, released.size(), goals.size(), fires.size(),
         fires.size() > 1 ? gsum / (fires.size() - 1) : 0.0, gmax);
#ifdef MB_WITH_POOL
  if (pool) modelblaster_pool_destroy((modelblaster_pool_t)pool);
#endif
  rclcpp::shutdown();
  return 0;
}
