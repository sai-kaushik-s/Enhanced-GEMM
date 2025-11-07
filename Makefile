CXX := g++
CXXFLAGS := -O3 -march=native -fopenmp -DNDEBUG -Ioptimized/cpp
LDFLAGS := -fopenmp -lnuma

SRC_DIR := optimized/cpp
BUILD_DIR := optimized
TARGET := $(BUILD_DIR)/gemm_opt
SRC := $(SRC_DIR)/gemm_opt.cpp

all: $(TARGET)

$(TARGET): $(SRC) $(SRC_DIR)/matrix.h $(SRC_DIR)/matrix.tpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)