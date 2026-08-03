# Copyright (c) 2009-2025, GPGPU-Sim Test Framework
# All rights reserved.

# Shared compiler, CUDA, and path settings. Domain-specific sources, targets,
# flags, and recipes live under make/.

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread -g -O2
CUDA_ARCH ?= sm_120a
HOPPER_CUDA_ARCH ?= sm_90a
# sm_120a enables advanced SM 12.0 features including ldmatrix/stmatrix .trans.
NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -arch=$(CUDA_ARCH)
HOPPER_NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -arch=$(HOPPER_CUDA_ARCH)
WGMMA_SM90A_NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a
HOPPER_EXTRA_NVCCFLAGS =

CUDA_ENV_PATH := $(if $(CUDA_HOME),$(CUDA_HOME),$(CUDA_PATH))
ifneq ($(origin CUDA_INSTALL_PATH),command line)
  ifeq ($(wildcard $(CUDA_INSTALL_PATH)/lib64/libcudart.so),)
    ifneq ($(strip $(CUDA_ENV_PATH)),)
      CUDA_INSTALL_PATH := $(CUDA_ENV_PATH)
    else
      CUDA_INSTALL_PATH ?= /usr/local/cuda
    endif
  endif
endif
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda

GTEST_CLONE_DIR = gtest
GTEST_DIR = gtest/googletest
SRC_DIR = ../src
BUILD_DIR = build
TEST_SRC_DIR = src
TEST_COMMON_DIR = common
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
HOPPER_SRC_DIR = $(TEST_SRC_DIR)/hopper
HOPPER_OBJ_DIR = $(OBJ_DIR)/hopper

GTEST_HEADERS = $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
                $(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GTEST_SRCS_ = $(wildcard $(GTEST_DIR)/src/*.cc) \
              $(wildcard $(GTEST_DIR)/src/*.h) \
              $(GTEST_HEADERS)

# Track CUDA headers shared across test domains.
CUH_HEADERS = $(shell find $(TEST_SRC_DIR) $(TEST_COMMON_DIR) $(SRC_DIR) -name '*.cuh' 2>/dev/null)

INCLUDES = -I$(GTEST_DIR)/include \
           -I. \
           -I$(SRC_DIR) \
           -I$(SRC_DIR)/gpgpu-sim \
           -I$(SRC_DIR)/cuda-sim \
           -I$(SRC_DIR)/accelwattch \
           -I$(CUDA_INSTALL_PATH)/include

GPGPUSIM_FLAGS = -DDEBUG=0

TOP_MAKEFILE := $(firstword $(MAKEFILE_LIST))
.DEFAULT_GOAL := help

$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(HOPPER_OBJ_DIR):
	mkdir -p $@

include make/gtest.mk
include make/sm120.mk
include make/sm90.mk
include make/fa3.mk
include make/fa2.mk
include make/microbench.mk
include make/registry.mk

.PHONY: clean clean-all debug

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean-all: clean
	rm -rf $(GTEST_DIR)
	@echo "Cleaned everything including Google Test"

debug:
	@echo "UNIT_TEST_SOURCES: $(UNIT_TEST_SOURCES)"
	@echo "INTEGRATION_TEST_SOURCES: $(INTEGRATION_TEST_SOURCES)"
	@echo "SM120_UNIT_TARGET: $(SM120_UNIT_TARGET)"
	@echo "SM120_INTEGRATION_TARGET: $(SM120_INTEGRATION_TARGET)"
	@echo "SM90_INSTRUCTION_SOURCES: $(SM90_INSTRUCTION_SOURCES)"
	@echo "SM90_INSTRUCTION_TARGET: $(SM90_INSTRUCTION_TARGET)"
	@echo "CUH_HEADERS: $(CUH_HEADERS)"
