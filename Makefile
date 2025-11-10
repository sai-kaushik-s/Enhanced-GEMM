CXX := g++
CXXFLAGS := -O3 -march=native -fopenmp -DNDEBUG -Ioptimized/cpp -g
LDFLAGS := -fopenmp -lnuma

SRC_DIR := optimized/cpp
BUILD_DIR := optimized
TARGET := $(BUILD_DIR)/gemm_opt
SRC := $(SRC_DIR)/gemm_opt.cpp

# New target and source for the second binary
BASELINE_TARGET := $(BUILD_DIR)/gemm_baseline
BASELINE_SRC := $(SRC_DIR)/gemm_baseline.cpp

# Default target
all: $(TARGET) $(BASELINE_TARGET)

# Build gemm_opt
$(TARGET): $(SRC) $(SRC_DIR)/matrix.h $(SRC_DIR)/matrix.tpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Build BASELINE_binary
$(BASELINE_TARGET): $(BASELINE_SRC)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(BASELINE_SRC) -o $(BASELINE_TARGET) $(LDFLAGS)

# Clean rule to remove both binaries
clean:
	rm -f $(TARGET) $(BASELINE_TARGET)
