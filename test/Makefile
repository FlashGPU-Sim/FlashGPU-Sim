# Copyright (c) 2009-2025, GPGPU-Sim Test Framework
# All rights reserved.

# Google Test Framework Makefile
# This makefile builds and runs unit tests for GPGPU-Sim

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread -g -O2
# Note: sm_120a enables advanced SM 12.0 features including ldmatrix/stmatrix .trans modifier
NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -arch=sm_120a

# CUDA settings
CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

# Directories
GTEST_CLONE_DIR = gtest
GTEST_DIR = gtest/googletest
SRC_DIR = ../src
BUILD_DIR = build
TEST_SRC_DIR = src
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Google Test settings
GTEST_HEADERS = $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
                $(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GTEST_SRCS_ = $(wildcard $(GTEST_DIR)/src/*.cc) $(wildcard $(GTEST_DIR)/src/*.h) $(GTEST_HEADERS)

# CUDA header dependencies - track all .cuh files in src directories
CUH_HEADERS = $(wildcard $(TEST_SRC_DIR)/*/*.cuh) \
              $(wildcard $(SRC_DIR)/*/*.cuh) \
              $(wildcard $(SRC_DIR)/*.cuh)

# Include paths
INCLUDES = -I$(GTEST_DIR)/include \
           -I$(SRC_DIR) \
           -I$(SRC_DIR)/gpgpu-sim \
           -I$(SRC_DIR)/cuda-sim \
           -I$(SRC_DIR)/accelwattch \
           -I$(CUDA_INSTALL_PATH)/include

# Test source files (recursive discovery for integration tests)
UNIT_TEST_SOURCES = $(wildcard $(TEST_SRC_DIR)/unit/*_test.cc)
INTEGRATION_TEST_SOURCES = $(shell find $(TEST_SRC_DIR)/integration -name '*_test.cc' 2>/dev/null)
TEST_SOURCES = $(UNIT_TEST_SOURCES) $(INTEGRATION_TEST_SOURCES)
TEST_OBJECTS = $(UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o) \
               $(INTEGRATION_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
TEST_TARGETS = $(TEST_SOURCES:$(TEST_SRC_DIR)/%_test.cc=$(BIN_DIR)/%)

# Create subdirectories for organized object files
UNIT_OBJ_DIR = $(OBJ_DIR)/unit
INTEGRATION_OBJ_DIR = $(OBJ_DIR)/integration
INTEGRATION_MMA_OBJ_DIR = $(OBJ_DIR)/integration/mma

# Main test runner
MAIN_TEST_TARGET = $(BIN_DIR)/run_all_tests

# GPGPU-Sim related flags (from main project)
GPGPUSIM_FLAGS = -DDEBUG=0

# GPGPU-Sim source files needed by tests (from flash subdirectory)
# Auto-discover all flash sources (uncomment when needed):
# FLASH_SOURCES = $(wildcard $(SRC_DIR)/gpgpu-sim/flash/*.cc)
# FLASH_OBJECTS = $(FLASH_SOURCES:$(SRC_DIR)/gpgpu-sim/flash/%.cc=$(OBJ_DIR)/flash/%.cu.o)

# Explicitly list only required source files
FLASH_SOURCES = $(SRC_DIR)/gpgpu-sim/flash/bulk_group.cc
FLASH_OBJECTS = $(OBJ_DIR)/flash/bulk_group.cu.o
FLASH_OBJ_DIR = $(OBJ_DIR)/flash

.PHONY: all clean test help setup-gtest run-tests

# Default target
all: setup-gtest $(MAIN_TEST_TARGET)

# Help target
help:
	@echo "GPGPU-Sim Test Framework"
	@echo "========================"
	@echo "Available targets:"
	@echo "  all         - Build all tests (default)"
	@echo "  test        - Build and run all tests"
	@echo "  clean       - Clean all build artifacts"
	@echo "  setup-gtest - Download and setup Google Test"
	@echo "  run-tests   - Run all compiled tests"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Individual test targets:"
	@for test in $(TEST_TARGETS); do \
		echo "  $$(basename $$test) - Build and run specific test"; \
	done

# Setup Google Test
setup-gtest:
	@if [ ! -d "$(GTEST_DIR)" ]; then \
		echo "Downloading Google Test..."; \
		git clone https://github.com/google/googletest.git $(GTEST_CLONE_DIR); \
		cd $(GTEST_DIR) && git checkout release-1.12.1; \
	else \
		echo "Google Test already exists."; \
	fi

# Create build directories
$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(UNIT_OBJ_DIR) $(INTEGRATION_OBJ_DIR) $(INTEGRATION_MMA_OBJ_DIR) $(FLASH_OBJ_DIR):
	mkdir -p $@

# Build Google Test library
$(OBJ_DIR)/gtest-all.o: $(GTEST_SRCS_) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o: $(GTEST_SRCS_) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest_main.cc -o $@

$(OBJ_DIR)/gtest.a: $(OBJ_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest_main.a: $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Build GPGPU-Sim flash source files needed by tests
$(OBJ_DIR)/flash/%.cu.o: $(SRC_DIR)/gpgpu-sim/flash/%.cc | $(FLASH_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Build test objects (both unit and integration tests compiled with nvcc for CUDA support)
$(OBJ_DIR)/unit/%.cu.o: $(TEST_SRC_DIR)/unit/%.cc $(CUH_HEADERS) | $(UNIT_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Integration tests with recursive subdirectory support
$(OBJ_DIR)/integration/%.cu.o: $(TEST_SRC_DIR)/integration/%.cc $(CUH_HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Build individual test executables
$(BIN_DIR)/%_test: $(OBJ_DIR)/%_test.o $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread

# Build main test runner
$(MAIN_TEST_TARGET): $(TEST_OBJECTS) $(FLASH_OBJECTS) $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread $(CUDA_LIBS)

# Run tests
test: all run-tests

run-tests:
	@echo "Running all tests..."
	@if [ -f "$(MAIN_TEST_TARGET)" ]; then \
		$(MAIN_TEST_TARGET); \
	else \
		echo "No tests found. Build tests first with 'make all'"; \
		exit 1; \
	fi

# Run individual tests
run-%: $(BIN_DIR)/%_test
	@echo "Running $< ..."
	@$<

# Clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean-all: clean
	rm -rf $(GTEST_DIR)
	@echo "Cleaned everything including Google Test"

# Debug target to print variables
debug:
	@echo "TEST_SOURCES: $(TEST_SOURCES)"
	@echo "TEST_OBJECTS: $(TEST_OBJECTS)"
	@echo "TEST_TARGETS: $(TEST_TARGETS)"
	@echo "MAIN_TEST_TARGET: $(MAIN_TEST_TARGET)"
	@echo "CUH_HEADERS: $(CUH_HEADERS)"
