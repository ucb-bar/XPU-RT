cmake_minimum_required(VERSION 3.8)
project(ros_qnn_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# QNN SDK headers + dlopen — the actual backend lib is loaded at runtime
# (libQnnHta.so / libQnnDsp.so / libQnnCpu.so), so we don't link it here.
set(QNN_INC "/root/qairt/include" CACHE PATH "QNN SDK include root")
include_directories(${QNN_INC} ${QNN_INC}/QNN)

# Each node is its own binary so they run in separate processes — needed
# to avoid the in-process "Context handle already exists!" warning we hit
# when two HTA contexts share one backendCreate. Cross-process FastRPC
# tolerates concurrent contexts (validated via the profile_seg POC).
foreach(node dronet_node mlp_control_node yolov8n_node)
    add_executable(${node} src/${node}.cpp src/qnn_lib.cpp)
    target_include_directories(${node} PRIVATE src)
    ament_target_dependencies(${node} rclcpp std_msgs)
    target_link_libraries(${node} dl pthread)
    install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

# The generalised pinning harness: one node per network, one
# SingleThreadedExecutor per node, all in ONE process. Multiple contexts on one
# backend .so are legal here because it keeps a single shared backend handle per
# library, which is what the flow-C generated runtime does; the three nodes
# above predate that and stay one-process-per-node.
add_executable(pin_harness src/pin_harness.cpp)
target_include_directories(pin_harness PRIVATE src)
ament_target_dependencies(pin_harness rclcpp std_msgs)
target_link_libraries(pin_harness dl pthread)
install(TARGETS pin_harness DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
