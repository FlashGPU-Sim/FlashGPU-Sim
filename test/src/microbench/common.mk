# Shared settings for standalone CUDA microbenchmark Makefiles.
#
# GTest-based microbenchmarks are built by test/Makefile. The .cu
# calibration binaries under cp_async, mma, tma, and memory use this file.

CUDA_ENV_PATH := $(if $(CUDA_HOME),$(CUDA_HOME),$(CUDA_PATH))
CUDA_DEFAULT_PATH := $(if $(wildcard /usr/local/cuda-12.8/bin/nvcc),/usr/local/cuda-12.8,/usr/local/cuda)
CUDA_INSTALL_PATH ?= $(if $(strip $(CUDA_ENV_PATH)),$(CUDA_ENV_PATH),$(CUDA_DEFAULT_PATH))

MICROBENCH_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TEST_DIR := $(abspath $(MICROBENCH_ROOT)/../..)
STANDALONE_BIN_ROOT ?= $(TEST_DIR)/build/bin/microbench
STANDALONE_RUN_ROOT ?= $(TEST_DIR)/run/microbench

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC) 2>/dev/null)
CUDA_ROOT := $(if $(strip $(NVCC_BIN)),$(abspath $(dir $(NVCC_BIN))/..),$(CUDA_INSTALL_PATH))
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump
NCU ?= ncu

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)
