// The camera->YOLO->nav->control chain as a real ROS 2 graph, running the REAL ModelBlaster
// kernels -- the same generated code XPU-RT's runtime executes. ROS replaces our runtime as
// the orchestrator; everything below the dispatch is byte-identical.
//
// This is the comparison that matters. A busy-wait stand-in measures the middleware in
// isolation but not its interaction with the real work: cache pressure from 12 MB of weights,
// the RVV kernels' own scheduling, and the executor's behaviour when a callback occupies the
// core for 30 ms rather than spinning on a clock.
//
// One node per network, as ROS deployments are written. Each node subscribes, runs its whole
// model to completion, and publishes. Latency is stamped at the camera timer and read at the
// control output, so what is reported is end-to-end camera->actuator.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

// Declared directly rather than by including all three model.h at once: each generated
// header also defines UNMANGLED aliases (model_output_t, model_state_t, ...) for
// single-model builds, so including three of them in one TU collides. The mangled entry
// points and the I/O sizes are the whole ABI we need, and they are stable.
extern "C" {
void run_model_yolov8_nano_64x96(const signed char *in, signed char *out);
void run_model_fused_full(const signed char *in, signed char *out);
void run_model_mlp_control(const signed char *in, signed char *out);
}
#define MODEL_YOLOV8_NANO_64X96_INPUT_SIZE  18432
#define MODEL_YOLOV8_NANO_64X96_OUTPUT_SIZE  8316
#define MODEL_FUSED_FULL_INPUT_SIZE          5698
#define MODEL_FUSED_FULL_OUTPUT_SIZE            2
#define MODEL_MLP_CONTROL_INPUT_SIZE           16
#define MODEL_MLP_CONTROL_OUTPUT_SIZE           4

using namespace std::chrono_literals;
static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  double rate = 20.0, secs = 20.0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--rate" && i + 1 < argc) rate = atof(argv[++i]);
    else if (a == "--seconds" && i + 1 < argc) secs = atof(argv[++i]);
  }

  static std::vector<int8_t> yin(MODEL_YOLOV8_NANO_64X96_INPUT_SIZE, 1);
  static std::vector<int8_t> yout(MODEL_YOLOV8_NANO_64X96_OUTPUT_SIZE);
  static std::vector<int8_t> nin(MODEL_FUSED_FULL_INPUT_SIZE, 1);
  static std::vector<int8_t> nout(MODEL_FUSED_FULL_OUTPUT_SIZE);
  static std::vector<int8_t> cin(MODEL_MLP_CONTROL_INPUT_SIZE, 1);
  static std::vector<int8_t> cout_(MODEL_MLP_CONTROL_OUTPUT_SIZE);

  rclcpp::init(argc, argv);
  auto cam   = std::make_shared<rclcpp::Node>("camera");
  auto percep= std::make_shared<rclcpp::Node>("perception");
  auto nav   = std::make_shared<rclcpp::Node>("nav");
  auto ctrl  = std::make_shared<rclcpp::Node>("control");

  auto p_cam = cam->create_publisher<std_msgs::msg::String>("frame", 10);
  auto p_det = percep->create_publisher<std_msgs::msg::String>("detections", 10);
  auto p_goal= nav->create_publisher<std_msgs::msg::String>("goal", 10);

  std::vector<double> lat; lat.reserve(4096);
  auto s_percep = percep->create_subscription<std_msgs::msg::String>(
      "frame", 10, [&, p_det](std_msgs::msg::String::UniquePtr m) {
        run_model_yolov8_nano_64x96(yin.data(), yout.data());
        std_msgs::msg::String o; o.data = m->data; p_det->publish(o);
      });
  auto s_nav = nav->create_subscription<std_msgs::msg::String>(
      "detections", 10, [&, p_goal](std_msgs::msg::String::UniquePtr m) {
        run_model_fused_full(nin.data(), nout.data());
        std_msgs::msg::String o; o.data = m->data; p_goal->publish(o);
      });
  auto s_ctrl = ctrl->create_subscription<std_msgs::msg::String>(
      "goal", 10, [&](std_msgs::msg::String::UniquePtr m) {
        run_model_mlp_control(cin.data(), cout_.data());
        lat.push_back(now_ms() - atof(m->data.c_str()));
      });
  auto timer = cam->create_wall_timer(
      std::chrono::microseconds((long)(1e6 / rate)), [p_cam]() {
        std_msgs::msg::String m; m.data = std::to_string(now_ms()); p_cam->publish(m);
      });

  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(cam); ex.add_node(percep); ex.add_node(nav); ex.add_node(ctrl);
  const double t_end = now_ms() + secs * 1000.0;
  while (now_ms() < t_end && rclcpp::ok()) ex.spin_some(5ms);

  if (lat.empty()) { printf("ROS_MB no messages completed the chain\n"); rclcpp::shutdown(); return 1; }
  size_t keep = lat.size() / 4;                       // drop the first quarter: cold start
  std::vector<double> s(lat.begin() + keep, lat.end());
  std::sort(s.begin(), s.end());
  printf("ROS_MB rate=%.0f n=%zu e2e_med=%.3f e2e_p95=%.3f e2e_min=%.3f\n",
         rate, s.size(), s[s.size()/2], s[(size_t)(0.95*(s.size()-1))], s.front());
  rclcpp::shutdown();
  return 0;
}
