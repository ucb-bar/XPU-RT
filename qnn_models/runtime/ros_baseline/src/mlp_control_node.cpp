// MLP control ROS baseline node — fires every 2 ms (the tightest deadline
// in the workload), calls QNN graphExecute, traces timing.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include "qnn_lib.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using qnn_baseline::QnnContextHolder;
using qnn_baseline::now_us_double;

class MlpControlNode : public rclcpp::Node {
public:
    MlpControlNode() : Node("mlp_control_node") {
        const std::string ctx = declare_parameter<std::string>(
            "ctx_path", "/root/qnn_runtime_ctx/ctx_mlp_control_full_seg0__Cpu.bin");
        const std::string lib = declare_parameter<std::string>(
            "backend_lib", "/root/qairt/lib/target/libQnnCpu.so");
        const int warmup = declare_parameter<int>("warmup_iters", 2);
        const double period_ms = declare_parameter<double>("period_ms", 2.0);
        const std::string trace_csv = declare_parameter<std::string>(
            "trace_csv", "/tmp/ros_baseline_mlp_control.csv");
        const int max_iters = declare_parameter<int>("max_iters", 0);
        max_iters_ = max_iters;

        ctx_ = std::make_unique<QnnContextHolder>(ctx, lib);
        ctx_->warmup(warmup);

        trace_.open(trace_csv);
        trace_ << "seq,callback_start_us,exec_start_us,exec_end_us,callback_end_us\n";

        pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("mlp_control/action", 10);

        t0_us_ = now_us_double();
        auto period = std::chrono::duration<double, std::milli>(period_ms);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&MlpControlNode::tick, this));

        RCLCPP_INFO(get_logger(),
            "mlp_control_node: ctx=%s backend=%s period=%.2fms warmup=%d",
            ctx.c_str(), ctx_->backend_label().c_str(), period_ms, warmup);
    }

    ~MlpControlNode() override { if (trace_.is_open()) trace_.close(); }

private:
    void tick() {
        if (max_iters_ > 0 && seq_ >= max_iters_) {
            timer_->cancel();
            rclcpp::shutdown();
            return;
        }
        double cb_start = now_us_double() - t0_us_;
        double exec_start = now_us_double() - t0_us_;
        ctx_->run();
        double exec_end = now_us_double() - t0_us_;
        double cb_end = now_us_double() - t0_us_;

        trace_ << seq_ << ',' << cb_start << ',' << exec_start << ','
               << exec_end << ',' << cb_end << '\n';

        std_msgs::msg::Float32MultiArray msg;
        msg.data = {static_cast<float>(seq_),
                    static_cast<float>(exec_end - exec_start)};
        pub_->publish(msg);

        ++seq_;
    }

    std::unique_ptr<QnnContextHolder>                          ctx_;
    rclcpp::TimerBase::SharedPtr                                timer_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    std::ofstream                                               trace_;
    double                                                      t0_us_ = 0;
    int                                                         seq_ = 0;
    int                                                         max_iters_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MlpControlNode>());
    rclcpp::shutdown();
    return 0;
}
