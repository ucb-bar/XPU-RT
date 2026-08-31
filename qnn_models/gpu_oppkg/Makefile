# FlowC GPU op package -- native aarch64 build (runs on the QRB5165 itself:
# g++ 9.4, aarch64-oe-linux).  QNN_SDK_ROOT must point at a QAIRT install that
# has include/QNN (on the board: /root/qairt).
QNN_SDK_ROOT ?= /root/qairt
BUILD_DIR    ?= build
LIB          := $(BUILD_DIR)/libQnnGpuOpPackageFlowC.so

SRCS := $(wildcard src/*.cpp)
OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

INCLUDES := -I$(QNN_SDK_ROOT)/include/QNN -I$(QNN_SDK_ROOT)/include/QNN/GPU -Isrc
CXXFLAGS += -std=c++11 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter $(INCLUDES) \
            -fvisibility=hidden -DQNN_API='__attribute__((visibility("default")))'
LDFLAGS  += -shared -fPIC

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
