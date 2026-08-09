# Copyright (c) 2009-2025, GPGPU-Sim Test Framework
# All rights reserved.

# Shared compiler, CUDA, and path settings. Architecture membership is read
# from arch/*.toml; build implementation lives under make/.

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread -g -O2
BASE_NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu
SM90_NVCCFLAGS = $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm90)
WGMMA_PTX_NVCCFLAGS = \
    -gencode arch=$(ARCH_COMPUTE_TARGET_sm90),code=$(ARCH_COMPUTE_TARGET_sm90)
WGMMA_EXTRA_NVCCFLAGS = \
    -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED $(WGMMA_PTX_NVCCFLAGS)
TEST_GROUP_EXTRA_NVCCFLAGS =

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

LOCAL_GTEST_CLONE_DIR := gtest
LOCAL_GTEST_DIR := $(LOCAL_GTEST_CLONE_DIR)/googletest
GTEST_DIR ?= $(LOCAL_GTEST_DIR)
SRC_DIR = ../src
BUILD_DIR = build
TEST_SRC_DIR = src
TEST_INCLUDE_DIR = $(TEST_SRC_DIR)/include
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
SM90_OBJ_DIR = $(OBJ_DIR)/sm90

GTEST_HEADERS = $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
                $(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GTEST_SRCS_ = $(wildcard $(GTEST_DIR)/src/*.cc) \
              $(wildcard $(GTEST_DIR)/src/*.h) \
              $(GTEST_HEADERS)

# Track headers shared across test domains.
TEST_HEADERS = $(shell find $(TEST_SRC_DIR) \
                 \( -name '*.h' -o -name '*.cuh' \) 2>/dev/null) \
               $(shell find $(SRC_DIR) -name '*.cuh' 2>/dev/null)

INCLUDES = -I$(GTEST_DIR)/include \
           -I. \
           -I$(TEST_INCLUDE_DIR) \
           -I$(SRC_DIR) \
           -I$(SRC_DIR)/gpgpu-sim \
           -I$(SRC_DIR)/cuda-sim \
           -I$(SRC_DIR)/accelwattch \
           -I$(CUDA_INSTALL_PATH)/include

GPGPUSIM_FLAGS = -DDEBUG=0

TOP_MAKEFILE := $(firstword $(MAKEFILE_LIST))
.DEFAULT_GOAL := help

ARCH_MANIFEST_SCRIPT := scripts/arch_manifest.py
ARCH_MANIFESTS := $(wildcard arch/*.toml)
ARCH_MANIFEST_MK := $(BUILD_DIR)/generated/arch.mk

$(ARCH_MANIFEST_MK): $(ARCH_MANIFESTS) $(ARCH_MANIFEST_SCRIPT)
	@mkdir -p $(dir $@)
	python3 $(ARCH_MANIFEST_SCRIPT) emit-make --output $@

include $(ARCH_MANIFEST_MK)

$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(SM90_OBJ_DIR):
	mkdir -p $@

include make/gtest.mk
include make/build.mk
include make/workloads/fa.mk
include make/workloads/fa2.mk
include make/workloads/fa3.mk
include make/workloads/microbench.mk
include make/workloads/trace.mk
include make/runner.mk

.PHONY: clean clean-all debug

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean-all: clean
	rm -rf $(LOCAL_GTEST_CLONE_DIR)
	@echo "Cleaned build directory and repository-local Google Test"

debug:
	@echo "ARCHITECTURES: $(ARCHITECTURES)"
	@echo "SM90_TEST_GROUPS: $(ARCH_TEST_GROUPS_sm90)"
	@echo "SM120_TEST_GROUPS: $(ARCH_TEST_GROUPS_sm120)"
	@echo "SM90_INTEGRATION_SOURCES: $(TEST_GROUP_SOURCES_sm90_integration)"
	@echo "SM120_INTEGRATION_SOURCES: $(TEST_GROUP_SOURCES_sm120_integration)"
	@echo "TEST_HEADERS: $(TEST_HEADERS)"
