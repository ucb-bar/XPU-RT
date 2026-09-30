// Does a ROS 2 control node actually hold its rate while perception runs?
//
// THE QUESTION THIS SETTLES. mlp_control has its own 100 Hz timer and is 0.08 ms of work, so
// on paper it is never the bottleneck. But the standard ROS deployment pattern is a
// SingleThreadedExecutor: one thread serves every callback on the node graph. While YOLO's
// ~31 ms callback is executing, no other callback on that executor can run -- including the
// control timer. So the control loop's ACHIEVED rate is not what its timer asks for; it is
// whatever the executor leaves it.
//
// This runs the real chain (real ModelBlaster kernels) with an independent 100 Hz control
// timer alongside, and reports the control timer's ACHIEVED inter-fire interval. If control
// holds 10 ms, gating it on the chain is wrong and the decoupled model is right. If it is
// starved to the chain period, the coupled model is right and it is MEASURED, not assumed.
//
// --executor multi runs the same graph on a MultiThreadedExecutor for contrast, which is the
// configuration a careful ROS engineer would reach for.
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/callback_group.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
extern "C" {
void run_model_yolov8_nano_64x96(const signed char *in, signed char *out);
void run_model_fused_full(const signed char *in, signed char *out);
void run_model_mlp_control(const signed char *in, signed char *out);
}

int main(int argc, char **argv) {
  double cam_hz = 15.0, ctrl_hz = 100.0, secs = 15.0; std::string exec = "single";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--cam-hz" && i+1 < argc) cam_hz = atof(argv[++i]);
    else if (a == "--ctrl-hz" && i+1 < argc) ctrl_hz = atof(argv[++i]);
    else if (a == "--seconds" && i+1 < argc) secs = atof(argv[++i]);
    else if (a == "--executor" && i+1 < argc) exec = argv[++i];
  }
  static std::vector<signed char> yi(18432,1), yo(8316), ni(5698,1), no(2), ci(16,1), co(4);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("chain");
  auto grp = (exec == "multi")
      ? node->create_callback_group(rclcpp::CallbackGroupType::Reentrant)
      : nullptr;
  rclcpp::SubscriptionOptions so; if (grp) so.callback_group = grp;

  auto p_frame = node->create_publisher<std_msgs::msg::String>("frame", 10);
  auto p_det   = node->create_publisher<std_msgs::msg::String>("det", 10);
  auto s_perc  = node->create_subscription<std_msgs::msg::String>("frame", 10,
      [&, p_det](std_msgs::msg::String::UniquePtr m){ run_model_yolov8_nano_64x96(yi.data(), yo.data());
                                                      run_model_fused_full(ni.data(), no.data());
                                                      p_det->publish(*m); }, so);
  auto t_cam = node->create_wall_timer(std::chrono::microseconds((long)(1e6/cam_hz)),
      [p_frame](){ std_msgs::msg::String m; m.data = std::to_string(now_ms()); p_frame->publish(m); },
      grp);

  std::vector<double> gaps; double last = -1;
  auto t_ctrl = node->create_wall_timer(std::chrono::microseconds((long)(1e6/ctrl_hz)),
      [&](){ run_model_mlp_control(ci.data(), co.data());
             double t = now_ms(); if (last > 0) gaps.push_back(t - last); last = t; }, grp);

  std::shared_ptr<rclcpp::Executor> ex;
  if (exec == "multi") ex = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  else                 ex = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  ex->add_node(node);
  // spin(), NOT spin_some(). A MultiThreadedExecutor only dispatches across its threads from
  // spin(); spin_some() runs ready work on the CALLING thread, so a spin_some loop pins the
  // whole executor to one core and would understate ROS by construction. Measured directly:
  // under spin_some only cpu5 was busy at 100% while 15 executor threads sat idle.
  std::thread stopper([&]{
    std::this_thread::sleep_for(std::chrono::milliseconds((long)(secs*1000.0)));
    ex->cancel();
  });
  ex->spin();
  stopper.join();

  if (gaps.size() < 10) { printf("CTRL exec=%s too few samples\n", exec.c_str()); rclcpp::shutdown(); return 1; }
  std::sort(gaps.begin(), gaps.end());
  double med = gaps[gaps.size()/2], p95 = gaps[(size_t)(0.95*(gaps.size()-1))], mx = gaps.back();
  printf("CTRL exec=%-6s cam=%.0fHz n=%zu  med=%.2f p60=%.2f p70=%.2f p75=%.2f p80=%.2f "
         "p85=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f  mean=%.2f\n",
         exec.c_str(), cam_hz, gaps.size(), med,
         gaps[(size_t)(0.60*(gaps.size()-1))], gaps[(size_t)(0.70*(gaps.size()-1))],
         gaps[(size_t)(0.75*(gaps.size()-1))], gaps[(size_t)(0.80*(gaps.size()-1))],
         gaps[(size_t)(0.85*(gaps.size()-1))], gaps[(size_t)(0.90*(gaps.size()-1))],
         p95, gaps[(size_t)(0.99*(gaps.size()-1))], mx,
         [&]{ double t=0; for(double g:gaps) t+=g; return t/gaps.size(); }());
  rclcpp::shutdown();
  return 0;
}
