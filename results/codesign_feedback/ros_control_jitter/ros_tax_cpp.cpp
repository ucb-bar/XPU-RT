// ROS 2 middleware tax over an N-hop chain, in C++ (rclcpp) rather than Python.
//
// The rclpy measurement of this same topology gave 17.63 ms over 3 hops. Python's per-callback
// interpreter cost is not ROS 2's middleware cost, and quoting it as such would overstate what
// a real C++ deployment pays -- which is what any serious ROS stack uses. This is the same
// chain in rclcpp, single-threaded executor, default RMW, so the two are directly comparable
// and the difference is attributable to the language runtime.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void busy_ms(double ms) {          // busy-wait, not sleep: a sleep yields the core and
  if (ms <= 0) return;                    // lets the executor hide the tax behind it
  const double end = now_ms() + ms;
  while (now_ms() < end) { }
}

int main(int argc, char **argv) {
  int hops = 3; double rate = 45.0, secs = 15.0; std::vector<double> comp;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--hops" && i + 1 < argc) hops = atoi(argv[++i]);
    else if (a == "--rate" && i + 1 < argc) rate = atof(argv[++i]);
    else if (a == "--seconds" && i + 1 < argc) secs = atof(argv[++i]);
    else if (a == "--compute" && i + 1 < argc) {
      std::string s = argv[++i], tok; size_t p = 0;
      while ((p = s.find(',')) != std::string::npos) { comp.push_back(atof(s.substr(0,p).c_str())); s.erase(0,p+1); }
      comp.push_back(atof(s.c_str()));
    }
  }
  if (comp.empty()) comp.assign(hops, 0.0);
  if ((int)comp.size() == 1) comp.assign(hops, comp[0]);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("tax_chain");
  std::vector<rclcpp::Publisher<std_msgs::msg::String>::SharedPtr> pubs;
  std::vector<rclcpp::Subscription<std_msgs::msg::String>::SharedPtr> subs;
  for (int i = 0; i <= hops; i++)
    pubs.push_back(node->create_publisher<std_msgs::msg::String>("chain_" + std::to_string(i), 10));

  std::vector<double> lat; lat.reserve(4096);
  for (int i = 0; i < hops; i++) {
    double c = comp[i];
    subs.push_back(node->create_subscription<std_msgs::msg::String>(
        "chain_" + std::to_string(i), 10,
        [pub = pubs[i + 1], c](std_msgs::msg::String::UniquePtr m) {
          busy_ms(c);
          pub->publish(*m);
        }));
  }
  subs.push_back(node->create_subscription<std_msgs::msg::String>(
      "chain_" + std::to_string(hops), 10,
      [&lat](std_msgs::msg::String::UniquePtr m) { lat.push_back(now_ms() - atof(m->data.c_str())); }));

  auto timer = node->create_wall_timer(
      std::chrono::microseconds((long)(1e6 / rate)), [pub = pubs[0]]() {
        std_msgs::msg::String m; m.data = std::to_string(now_ms()); pub->publish(m);
      });

  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  const double t_end = now_ms() + secs * 1000.0;
  while (now_ms() < t_end && rclcpp::ok()) ex.spin_some(10ms);

  double burned = 0; for (double c : comp) burned += c;
  size_t keep = lat.size() / 5;                       // drop the first fifth as warm-up
  std::vector<double> s(lat.begin() + keep, lat.end());
  std::sort(s.begin(), s.end());
  if (s.empty()) { printf("no messages completed the chain\n"); rclcpp::shutdown(); return 1; }
  double med = s[s.size()/2], p95 = s[(size_t)(0.95*(s.size()-1))];
  printf("CPP hops=%d rate=%.0f n=%zu burned=%.3f e2e_med=%.3f e2e_p95=%.3f tax_med=%.3f tax_p95=%.3f\n",
         hops, rate, s.size(), burned, med, p95, med - burned, p95 - burned);
  rclcpp::shutdown();
  return 0;
}
