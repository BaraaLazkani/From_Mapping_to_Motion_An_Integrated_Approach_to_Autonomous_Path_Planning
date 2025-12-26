# Makefile for CUDA DWA Implementation
# Compiles both GPU-accelerated (CUDA) and CPU baseline (C++) versions

# ============================================================
# Configuration
# ============================================================

# Compilers
NVCC := nvcc
CXX := g++

# Directories
SRC_DIR := src
INC_DIR := include
BIN_DIR := bin
CONFIG_DIR := configs

# CUDA compute capability (adjust for your GPU)
# Common values:
#   - 7.5: RTX 2080, RTX 2070
#   - 8.0: A100
#   - 8.6: RTX 3090, RTX 3080
#   - 8.9: RTX 4090
CUDA_ARCH := 75

# Compiler flags
NVCC_FLAGS := -arch=sm_$(CUDA_ARCH) -I$(INC_DIR) -std=c++14
CXX_FLAGS := -O3 -I$(INC_DIR) -std=c++14

# Libraries
LIBS := -lyaml-cpp

# Source files
CUDA_SRC := $(SRC_DIR)/dwa_cuda.cu
CPP_SRC := $(SRC_DIR)/dwa_cpp.cpp
CONFIG_SRC := $(SRC_DIR)/config_loader.cpp

# Output binaries
CUDA_BIN := $(BIN_DIR)/dwa_cuda
CPP_BIN := $(BIN_DIR)/dwa_cpp


# ============================================================
# Targets
# ============================================================

# Default target: build both versions
.PHONY: all
all: cuda cpp

# Build CUDA version
.PHONY: cuda
cuda: $(CUDA_BIN)

# Build C++ baseline version
.PHONY: cpp
cpp: $(CPP_BIN)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile CUDA version
$(CUDA_BIN): $(CUDA_SRC) $(CONFIG_SRC) | $(BIN_DIR)
	@echo "==================================="
	@echo "Compiling CUDA DWA..."
	@echo "==================================="
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) $(CONFIG_SRC) $(LIBS) -o $(CUDA_BIN)
	@echo "✓ CUDA binary created: $(CUDA_BIN)"

# Compile C++ baseline version
$(CPP_BIN): $(CPP_SRC) $(CONFIG_SRC) | $(BIN_DIR)
	@echo "==================================="
	@echo "Compiling C++ Baseline DWA..."
	@echo "==================================="
	$(CXX) $(CXX_FLAGS) $(CPP_SRC) $(CONFIG_SRC) $(LIBS) -o $(CPP_BIN)
	@echo "✓ C++ binary created: $(CPP_BIN)"

# Run CUDA version
.PHONY: run-cuda
run-cuda: cuda
	@echo "\n==================================="
	@echo "Running CUDA DWA..."
	@echo "==================================="
	cd .. && ./cuda_dwa/$(CUDA_BIN)

# Run C++ baseline version
.PHONY: run-cpp
run-cpp: cpp
	@echo "\n==================================="
	@echo "Running C++ Baseline DWA..."
	@echo "==================================="
	cd .. && ./cuda_dwa/$(CPP_BIN)

# Run both and compare performance
.PHONY: benchmark
benchmark: all
	@echo "\n#######################################"
	@echo "#  DWA Performance Benchmark"
	@echo "#######################################"
	@echo "\n[1/2] Running C++ Baseline..."
	@cd .. && ./cuda_dwa/$(CPP_BIN)
	@echo "\n[2/2] Running CUDA GPU..."
	@cd .. && ./cuda_dwa/$(CUDA_BIN)
	@echo "\n#######################################"
	@echo "#  Benchmark Complete"
	@echo "#######################################"
	@echo "\nCompare GPU vs CPU execution times above."
	@echo "Expected speedup: 30-50x for CUDA version."

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BIN_DIR)
	@echo "✓ Build directory removed"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@echo -n "  CUDA compiler (nvcc): "
	@which nvcc > /dev/null && echo "✓ Found" || echo "✗ Not found"
	@echo -n "  C++ compiler (g++): "
	@which g++ > /dev/null && echo "✓ Found" || echo "✗ Not found"
	@echo -n "  yaml-cpp library: "
	@pkg-config --exists yaml-cpp && echo "✓ Found" || echo "✗ Not found (install: sudo apt-get install libyaml-cpp-dev)"

# Help message
.PHONY: help
help:
	@echo "CUDA DWA Makefile"
	@echo "=================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build both CUDA and C++ versions (default)"
	@echo "  cuda         - Build only CUDA version"
	@echo "  cpp          - Build only C++ baseline version"
	@echo "  run-cuda     - Build and run CUDA version"
	@echo "  run-cpp      - Build and run C++ baseline version"
	@echo "  benchmark    - Build and run both versions for performance comparison"
	@echo "  clean        - Remove build artifacts"
	@echo "  check-deps   - Check if required dependencies are installed"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH=$(CUDA_ARCH) (adjust for your GPU in Makefile)"
	@echo ""
	@echo "Examples:"
	@echo "  make cuda           # Compile CUDA version only"
	@echo "  make run-cpp        # Compile and run C++ baseline"
	@echo "  make benchmark      # Compare GPU vs CPU performance"
	@echo "  make clean all      # Clean rebuild"
	@echo ""
	@echo "Dependencies:"
	@echo "  - CUDA Toolkit (nvcc)"
	@echo "  - g++ compiler"
	@echo "  - yaml-cpp library"
	@echo ""
	@echo "Install dependencies (Ubuntu/Debian):"
	@echo "  sudo apt-get install nvidia-cuda-toolkit libyaml-cpp-dev g++"
