// Yolov8n ROS baseline node — chains the 2-split sub-DLCs (backbone +
// head) per the granularity finding in HETEROGENEOUS_SCHEDULING_QRB5165.md.
// Both halves run on the same backend (DSP for B1/B2 — HTA can't compose
// the head). The node holds two QnnContextHolder instances and runs them
// back-to-back in the timer callback.
//
// Periodic at e.g. 30 Hz (33 ms). At 30 Hz on DSP yolov8 takes ~33 ms per
// frame so the lane is essentially saturated — fair "every-frame
// detection" load.

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

class Yolov8nNode : public rclcpp::Node {
public:
    Yolov8nNode() : Node("yolov8n_node") {
        const std::string ctx_back = declare_parameter<std::string>(
            "backbone_ctx_path",
            "/root/qnn_runtime_ctx/ctx_yolov8n_HTA_split_seg100__Dsp.bin");
        const std::string ctx_head = declare_parameter<std::string>(
            "head_ctx_path",
            "/root/qnn_runtime_ctx/ctx_yolov8n_HTA_split_seg101__Dsp.bin");
        const std::string lib = declare_parameter<std::string>(
            "backend_lib", "/root/qairt/lib/target/libQnnDsp.so");
        const int warmup = declare_parameter<int>("warmup_iters", 2);
        const double period_ms = declare_parameter<double>("period_ms", 33.33);
        const std::string trace_csv = declare_parameter<std::string>(
            "trace_csv", "/tmp/ros_baseline_yolov8n.csv");
        const int max_iters = declare_parameter<int>("max_iters", 0);
        max_iters_ = max_iters;

        backbone_ = std::make_unique<QnnContextHolder>(ctx_back, lib);
        head_     = std::make_unique<QnnContextHolder>(ctx_head, lib);
        backbone_->warmup(warmup);
        head_->warmup(warmup);

        trace_.open(trace_csv);
        trace_ << "seq,callback_start_us,exec_start_us,backbone_end_us,exec_end_us,callback_end_us\n";

        pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("yolov8n/det", 10);

        t0_us_ = now_us_double();
        auto period = std::chrono::duration<double, std::milli>(period_ms);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&Yolov8nNode::tick, this));

        RCLCPP_INFO(get_logger(),
            "yolov8n_node: lib=%s period=%.2fms warmup=%d",
            backbone_->backend_label().c_str(), period_ms, warmup);
    }

    ~Yolov8nNode() override { if (trace_.is_open()) trace_.close(); }

private:
    void tick() {
        if (max_iters_ > 0 && seq_ >= max_iters_) {
            timer_->cancel();
            rclcpp::shutdown();
            return;
        }
        double cb_start = now_us_double() - t0_us_;
        double exec_start = now_us_double() - t0_us_;
        backbone_->run();
        double back_end = now_us_double() - t0_us_;
        head_->run();
        double exec_end = now_us_double() - t0_us_;
        double cb_end = now_us_double() - t0_us_;

        trace_ << seq_ << ',' << cb_start << ',' << exec_start << ','
               << back_end << ',' << exec_end << ',' << cb_end << '\n';

        std_msgs::msg::Float32MultiArray msg;
        msg.data = {static_cast<float>(seq_),
                    static_cast<float>(exec_end - exec_start)};
        pub_->publish(msg);

        ++seq_;
    }

    std::unique_ptr<QnnContextHolder>                          backbone_;
    std::unique_ptr<QnnContextHolder>                          head_;
    rclcpp::TimerBase::SharedPtr                                timer_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
    std::ofstream                                               trace_;
    double                                                      t0_us_ = 0;
    int                                                         seq_ = 0;
    int                                                         max_iters_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Yolov8nNode>());
    rclcpp::shutdown();
    return 0;
}
