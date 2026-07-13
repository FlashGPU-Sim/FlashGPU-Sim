# Copyright (c) 2009-2025, GPGPU-Sim Test Framework
# All rights reserved.

# Google Test Framework Makefile
# This makefile builds and runs unit tests for GPGPU-Sim

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread -g -O2
CUDA_ARCH ?= sm_120a
HOPPER_CUDA_ARCH ?= sm_90a
# Note: sm_120a enables advanced SM 12.0 features including ldmatrix/stmatrix .trans modifier
NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -arch=$(CUDA_ARCH)
HOPPER_NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -arch=$(HOPPER_CUDA_ARCH)
WGMMA_SM90A_NVCCFLAGS = -std=c++17 -g -O2 --compiler-options "-pthread" -x cu -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a
HOPPER_EXTRA_NVCCFLAGS =

# CUDA settings
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

# Directories
GTEST_CLONE_DIR = gtest
GTEST_DIR = gtest/googletest
SRC_DIR = ../src
BUILD_DIR = build
TEST_SRC_DIR = src
TEST_COMMON_DIR = common
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Google Test settings
GTEST_HEADERS = $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
                $(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GTEST_SRCS_ = $(wildcard $(GTEST_DIR)/src/*.cc) $(wildcard $(GTEST_DIR)/src/*.h) $(GTEST_HEADERS)

# CUDA header dependencies - track all .cuh files in test/common, test/src, and src
CUH_HEADERS = $(shell find $(TEST_SRC_DIR) $(TEST_COMMON_DIR) $(SRC_DIR) -name '*.cuh' 2>/dev/null)

# Include paths
INCLUDES = -I$(GTEST_DIR)/include \
           -I. \
           -I$(SRC_DIR) \
           -I$(SRC_DIR)/gpgpu-sim \
           -I$(SRC_DIR)/cuda-sim \
           -I$(SRC_DIR)/accelwattch \
           -I$(CUDA_INSTALL_PATH)/include

# Test source files (recursive discovery for integration tests)
UNIT_TEST_SOURCES = $(wildcard $(TEST_SRC_DIR)/unit/*_test.cc)
INTEGRATION_TEST_SOURCES = $(shell find $(TEST_SRC_DIR)/integration -name '*_test.cc' 2>/dev/null)
HOPPER_SRC_DIR = $(TEST_SRC_DIR)/hopper
SM90_TEST_SOURCES = $(HOPPER_SRC_DIR)/named_barrier_test.cc \
                    $(sort $(wildcard $(HOPPER_SRC_DIR)/wgmma/*_test.cc))
SM90_TEST_CUH_HEADERS = $(shell find $(TEST_COMMON_DIR) $(SRC_DIR) -name '*.cuh' 2>/dev/null) \
                        $(wildcard $(HOPPER_SRC_DIR)/wgmma/*.cuh)
MICROBENCH_ALL_SOURCES = $(shell find $(TEST_SRC_DIR)/microbench -name '*_bench.cc' 2>/dev/null)
MICROBENCH_SM90_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_ALL_SOURCES))
MICROBENCH_DEFAULT_SOURCES = $(filter-out $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_ALL_SOURCES))
# run_all_tests includes unit + integration only (no microbench)
TEST_SOURCES = $(UNIT_TEST_SOURCES) $(INTEGRATION_TEST_SOURCES)
TEST_OBJECTS = $(UNIT_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o) \
               $(INTEGRATION_TEST_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
TEST_TARGETS = $(TEST_SOURCES:$(TEST_SRC_DIR)/%_test.cc=$(BIN_DIR)/%)
SM90_TEST_OBJECTS = $(SM90_TEST_SOURCES:$(HOPPER_SRC_DIR)/%.cc=$(OBJ_DIR)/hopper/%.cu.o)
HOPPER_FA2_TEST_SOURCE = $(HOPPER_SRC_DIR)/fa2/fa2_fwd_fp16_test.cc
HOPPER_FA2_VARIANTS = h32d64_full h32d64_causal h16d128_full h16d128_causal
HOPPER_FA2_GROUPS = smoke small medium large
HOPPER_FA2_SMOKE_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/smoke_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_SMALL_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/small_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_MEDIUM_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/medium_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_LARGE_OBJECTS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(OBJ_DIR)/hopper/fa2/large_$(variant)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_OBJECTS = $(HOPPER_FA2_SMOKE_OBJECTS) $(HOPPER_FA2_SMALL_OBJECTS) $(HOPPER_FA2_MEDIUM_OBJECTS) $(HOPPER_FA2_LARGE_OBJECTS)
HOPPER_FA2_BREAKDOWN_MODES = baseline skip_cp_async skip_mma skip_softmax fma_softmax only_mma only_cp_async only_softmax nothing
HOPPER_FA2_BREAKDOWN_OBJECTS = $(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(OBJ_DIR)/hopper/fa2/breakdown_$(mode)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_SCALING_MODES = baseline nothing only_cp_async only_softmax only_mma softmax_mma
HOPPER_FA2_SCALING_OBJECTS = $(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(OBJ_DIR)/hopper/fa2/scaling_$(mode)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA2_CONCURRENCY_MODES = only_cp_async only_cp_async_bhhalf qk_softmax_pv_only qk_pv_only
HOPPER_FA2_CONCURRENCY_OBJECTS = $(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(OBJ_DIR)/hopper/fa2/concurrency_$(mode)/fa2_fwd_fp16_test.cu.o)
HOPPER_FA3_TEST_SOURCES = $(wildcard $(HOPPER_SRC_DIR)/fa3/*_test.cc)
HOPPER_FA3_EXTENDED_OBJECTS = $(HOPPER_FA3_TEST_SOURCES:$(HOPPER_SRC_DIR)/fa3/%.cc=$(OBJ_DIR)/hopper/fa3_extended/%.cu.o)
HOPPER_FA3_SINGLE_TILE_OBJECT = $(OBJ_DIR)/hopper/fa3_profile/fa3_fwd_hdim128_fp16_test.cu.o
HOPPER_FA3_H1D128_PROFILE_OBJECT = $(OBJ_DIR)/hopper/fa3_profile/fa3_fwd_h1d128_profile_test.cu.o
HOPPER_FA3_MODES = \
	baseline softmax_only qk_only pv_only qk_pv_only \
	tma_only tma_only_m3_only tma_only_first_round_bh \
	tma_only_first_round_bh_m3_only baseline_noprofile \
	softmax_only_noprofile qk_pv_only_noprofile base_no_tma \
	softmax_only_no_tma qk_pv_only_no_tma base_no_tma_noprofile \
	softmax_only_no_tma_noprofile qk_pv_only_no_tma_noprofile \
	qk_pv_only_no_tma_timeline qk_pv_only_no_tma_reg_timeline \
	sync_only_no_tma sync_only_no_tma_noprofile \
	qk_pv_only_no_tma_extended qk_pv_only_no_tma_extended_noprofile \
	qk_pv_only_no_tma_extended_reg_timeline
HOPPER_FA3_MODE_OBJECT = $(OBJ_DIR)/hopper/fa3_modes/$(1)/fa3_fwd_h1d128_profile_test.cu.o
HOPPER_FA3_MODE_OBJECTS = $(foreach mode,$(HOPPER_FA3_MODES),$(call HOPPER_FA3_MODE_OBJECT,$(mode)))

# Microbench stay as separate binaries by design: faster incremental rebuilds,
# failure isolation, and benchmark-specific artifacts. run_tests.sh pre-screens
# binaries with --gtest_list_tests so users can still pass one filter.
MICROBENCH_DEFAULT_TARGETS = $(MICROBENCH_DEFAULT_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_SM90_TARGETS = $(MICROBENCH_SM90_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)

HOPPER_FA3_DIR = $(HOPPER_SRC_DIR)/fa3
HOPPER_FA3_PREPARE_SCRIPT = $(HOPPER_FA3_DIR)/prepare_flash_attention.sh
HOPPER_FA3_PATCHES = $(wildcard $(HOPPER_FA3_DIR)/patches/*.patch)
HOPPER_FA3_PREPARED_STAMP = $(HOPPER_FA3_DIR)/flash-attention/.gpgpusim-prepared
HOPPER_FA3_GENERATED_HEADERS = \
	$(HOPPER_FA3_DIR)/flash-attention/hopper/flash_profile.h \
	$(HOPPER_FA3_DIR)/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
HOPPER_FA3_COMMON_FLAGS = --ftemplate-backtrace-limit=0 -O3 \
                          --use_fast_math \
                          -lineinfo --expt-relaxed-constexpr \
                          --expt-extended-lambda \
                          -DCUTLASS_ENABLE_GDC_FOR_SM90 \
                          -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
                          -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
                          -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
                          -DNDEBUG \
                          -I$(HOPPER_FA3_DIR)/flash-attention/hopper \
                          -I$(HOPPER_FA3_DIR)/flash-attention/csrc/cutlass/include
HOPPER_FA3_EXTENDED_FLAGS = $(HOPPER_FA3_COMMON_FLAGS) -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
HOPPER_FA3_FLAGS = $(HOPPER_FA3_EXTENDED_FLAGS)
HOPPER_FA3_PROFILE_HEADERS = \
	$(HOPPER_FA3_DIR)/fa3_fwd_hdim128_fp16_case.cuh \
	$(HOPPER_FA3_DIR)/flash-attention/hopper/flash_profile.h \
	$(HOPPER_FA3_DIR)/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
ifneq ($(strip $(SM90_TEST_SOURCES)),)
SM90_TEST_TARGETS = $(SM90_TEST_TARGET)
endif

HOPPER_FA2_DIR = $(HOPPER_SRC_DIR)/fa2
HOPPER_FA2_COMMON_FLAGS = --ftemplate-backtrace-limit=0 -O3 \
                          --use_fast_math \
                          -lineinfo --expt-relaxed-constexpr \
                          --expt-extended-lambda \
                          -DNDEBUG \
                          -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
                          -I$(HOPPER_FA2_DIR) \
                          -I$(HOPPER_FA3_DIR)/flash-attention/csrc/flash_attn/src \
                          -I$(HOPPER_FA3_DIR)/flash-attention/csrc/cutlass/include
HOPPER_FA2_FLAGS = $(HOPPER_FA2_COMMON_FLAGS)

# Standalone dev tests (decoupled from simulator, separate binary)
STANDALONE_SOURCES = $(wildcard $(TEST_SRC_DIR)/standalone/*_test.cc)
STANDALONE_OBJECTS = $(STANDALONE_SOURCES:$(TEST_SRC_DIR)/%.cc=$(OBJ_DIR)/%.cu.o)
DEV_TEST_TARGET = $(BIN_DIR)/run_dev_tests

# Create subdirectories for organized object files
UNIT_OBJ_DIR = $(OBJ_DIR)/unit
INTEGRATION_OBJ_DIR = $(OBJ_DIR)/integration
INTEGRATION_MMA_OBJ_DIR = $(OBJ_DIR)/integration/mma
MICROBENCH_OBJ_DIR = $(OBJ_DIR)/microbench
STANDALONE_OBJ_DIR = $(OBJ_DIR)/standalone
HOPPER_OBJ_DIR = $(OBJ_DIR)/hopper

# Main test runner
MAIN_TEST_TARGET = $(BIN_DIR)/run_all_tests
SM90_TEST_TARGET = $(BIN_DIR)/sm90/run_sm90_tests
HOPPER_FA2_SMOKE_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_smoke_$(variant)_tests)
HOPPER_FA2_SMALL_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_small_$(variant)_tests)
HOPPER_FA2_MEDIUM_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_medium_$(variant)_tests)
HOPPER_FA2_LARGE_TARGETS = $(foreach variant,$(HOPPER_FA2_VARIANTS),$(BIN_DIR)/hopper/run_fa2_large_$(variant)_tests)
HOPPER_FA2_STANDARD_TARGETS = $(HOPPER_FA2_SMOKE_TARGETS) $(HOPPER_FA2_SMALL_TARGETS) $(HOPPER_FA2_MEDIUM_TARGETS) $(HOPPER_FA2_LARGE_TARGETS)
HOPPER_FA2_BREAKDOWN_TARGETS = $(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(BIN_DIR)/hopper/run_fa2_breakdown_$(mode)_tests)
HOPPER_FA2_SCALING_TARGETS = $(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(BIN_DIR)/hopper/run_fa2_scaling_$(mode)_tests)
HOPPER_FA2_CONCURRENCY_TARGETS = $(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(BIN_DIR)/hopper/run_fa2_concurrency_$(mode)_tests)
HOPPER_FA3_EXTENDED_TARGET = $(BIN_DIR)/hopper/run_fa3_extended_tests
HOPPER_FA3_SINGLE_TILE_TARGET = $(BIN_DIR)/hopper/run_fa3_single_tile_tests
HOPPER_FA3_H1D128_PROFILE_TARGET = $(BIN_DIR)/hopper/run_fa3_h1d128_profile_tests
HOPPER_FA3_MODE_TARGET = $(BIN_DIR)/hopper/run_fa3_$(1)_tests
HOPPER_FA3_MODE_TARGETS = $(foreach mode,$(HOPPER_FA3_MODES),$(call HOPPER_FA3_MODE_TARGET,$(mode)))

# Binary group manifest. run_tests.sh queries this list instead of duplicating
# source, variant, or mode arrays in shell.
TEST_GROUPS = \
	test \
	microbench-default \
	microbench-sm90 \
	dev \
	test-sm90 \
	fa2-smoke \
	fa2-small \
	fa2-medium \
	fa2-large \
	fa2-breakdown \
	fa2-scaling \
	fa2-concurrency \
	fa3 \
	fa3-single-tile \
	fa3-h1d128-profile \
	fa3-modes \
	$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),fa2-breakdown-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_SCALING_MODES),fa2-scaling-$(mode)) \
	$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),fa2-concurrency-$(mode)) \
	$(foreach mode,$(HOPPER_FA3_MODES),fa3-mode-$(mode))

TEST_GROUP_BINARIES_test = $(MAIN_TEST_TARGET)
TEST_GROUP_BINARIES_microbench-default = $(MICROBENCH_DEFAULT_TARGETS)
TEST_GROUP_BINARIES_microbench-sm90 = $(MICROBENCH_SM90_TARGETS)
TEST_GROUP_BINARIES_dev = $(DEV_TEST_TARGET)
TEST_GROUP_BINARIES_test-sm90 = $(SM90_TEST_TARGETS)
TEST_GROUP_BINARIES_fa2-smoke = $(HOPPER_FA2_SMOKE_TARGETS)
TEST_GROUP_BINARIES_fa2-small = $(HOPPER_FA2_SMALL_TARGETS)
TEST_GROUP_BINARIES_fa2-medium = $(HOPPER_FA2_MEDIUM_TARGETS)
TEST_GROUP_BINARIES_fa2-large = $(HOPPER_FA2_LARGE_TARGETS)
TEST_GROUP_BINARIES_fa2-breakdown = $(HOPPER_FA2_BREAKDOWN_TARGETS)
TEST_GROUP_BINARIES_fa2-scaling = $(HOPPER_FA2_SCALING_TARGETS)
TEST_GROUP_BINARIES_fa2-concurrency = $(HOPPER_FA2_CONCURRENCY_TARGETS)
TEST_GROUP_BINARIES_fa3 = $(HOPPER_FA3_EXTENDED_TARGET)
TEST_GROUP_BINARIES_fa3-single-tile = $(HOPPER_FA3_SINGLE_TILE_TARGET)
TEST_GROUP_BINARIES_fa3-h1d128-profile = $(HOPPER_FA3_H1D128_PROFILE_TARGET)
TEST_GROUP_BINARIES_fa3-modes = $(HOPPER_FA3_MODE_TARGETS)
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval TEST_GROUP_BINARIES_fa2-breakdown-$(mode) = $(BIN_DIR)/hopper/run_fa2_breakdown_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval TEST_GROUP_BINARIES_fa2-scaling-$(mode) = $(BIN_DIR)/hopper/run_fa2_scaling_$(mode)_tests))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval TEST_GROUP_BINARIES_fa2-concurrency-$(mode) = $(BIN_DIR)/hopper/run_fa2_concurrency_$(mode)_tests))
$(foreach mode,$(HOPPER_FA3_MODES),$(eval TEST_GROUP_BINARIES_fa3-mode-$(mode) = $(call HOPPER_FA3_MODE_TARGET,$(mode))))

# User-facing suite/target registry. One target is an independently buildable
# and runnable architecture domain, not an individual Make target or gtest.
TEST_SUITES = test microbench dev trace

SUITE_TARGETS_test = default sm90 fa2 fa3
SUITE_TARGETS_microbench = default sm90
SUITE_TARGETS_dev = default
SUITE_TARGETS_trace = default

SUITE_DEFAULT_TARGET_test = default
SUITE_DEFAULT_TARGET_microbench = default
SUITE_DEFAULT_TARGET_dev = default
SUITE_DEFAULT_TARGET_trace = default

SUITE_TARGET_BUILD_GROUP_test_default = test
SUITE_TARGET_BINARY_GROUP_test_default = test
SUITE_TARGET_EXECUTOR_test_default = test
SUITE_TARGET_DEFAULT_CONFIG_test_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_test_default = 12.0
SUITE_TARGET_CUDA_ARCH_test_default = sm_120a

SUITE_TARGET_BUILD_GROUP_test_sm90 = test-sm90
SUITE_TARGET_BINARY_GROUP_test_sm90 = test-sm90
SUITE_TARGET_EXECUTOR_test_sm90 = gtest-single
SUITE_TARGET_DEFAULT_CONFIG_test_sm90 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_sm90 = 9.0
SUITE_TARGET_CUDA_ARCH_test_sm90 = sm_90a

SUITE_TARGET_BUILD_GROUP_test_fa2 = group-required
SUITE_TARGET_BINARY_GROUP_test_fa2 = group-required
SUITE_TARGET_EXECUTOR_test_fa2 = group-required
SUITE_TARGET_DEFAULT_CONFIG_test_fa2 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_fa2 = 9.0
SUITE_TARGET_CUDA_ARCH_test_fa2 = sm_90a

SUITE_TARGET_BUILD_GROUP_test_fa3 = group-required
SUITE_TARGET_BINARY_GROUP_test_fa3 = group-required
SUITE_TARGET_EXECUTOR_test_fa3 = group-required
SUITE_TARGET_DEFAULT_CONFIG_test_fa3 = SM90_H100
SUITE_TARGET_REQUIRED_CC_test_fa3 = 9.0
SUITE_TARGET_CUDA_ARCH_test_fa3 = sm_90a

# FA2/FA3 group registry. Standard workload groups have no mode; analysis
# groups require one explicit compile-time mode (or the explicit value all).
TARGET_GROUPS_test_fa2 = smoke small medium large breakdown scaling concurrency
TARGET_GROUPS_test_fa3 = smoke small medium large breakdown scaling concurrency

TARGET_GROUP_MODES_test_fa2_breakdown = $(HOPPER_FA2_BREAKDOWN_MODES) all
TARGET_GROUP_MODES_test_fa2_scaling = $(HOPPER_FA2_SCALING_MODES) all
TARGET_GROUP_MODES_test_fa2_concurrency = $(HOPPER_FA2_CONCURRENCY_MODES) all
TARGET_GROUP_MODES_test_fa3_breakdown = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_test_fa3_scaling = $(HOPPER_FA3_MODES) all
TARGET_GROUP_MODES_test_fa3_concurrency = $(HOPPER_FA3_MODES) all

TARGET_GROUP_BUILD_GROUP_test_fa2_smoke = fa2-smoke
TARGET_GROUP_BINARY_GROUP_test_fa2_smoke = fa2-smoke
TARGET_GROUP_EXECUTOR_test_fa2_smoke = gtest-multi
TARGET_GROUP_FILTER_test_fa2_smoke = *
TARGET_GROUP_BUILD_GROUP_test_fa2_small = fa2-small
TARGET_GROUP_BINARY_GROUP_test_fa2_small = fa2-small
TARGET_GROUP_EXECUTOR_test_fa2_small = gtest-multi
TARGET_GROUP_FILTER_test_fa2_small = *
TARGET_GROUP_BUILD_GROUP_test_fa2_medium = fa2-medium
TARGET_GROUP_BINARY_GROUP_test_fa2_medium = fa2-medium
TARGET_GROUP_EXECUTOR_test_fa2_medium = gtest-multi
TARGET_GROUP_FILTER_test_fa2_medium = *
TARGET_GROUP_BUILD_GROUP_test_fa2_large = fa2-large
TARGET_GROUP_BINARY_GROUP_test_fa2_large = fa2-large
TARGET_GROUP_EXECUTOR_test_fa2_large = gtest-multi
TARGET_GROUP_FILTER_test_fa2_large = *

define REGISTER_FA2_MODE
TARGET_GROUP_BUILD_GROUP_test_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_BINARY_GROUP_test_fa2_$(1)_$(2) = fa2-$(1)-$(2)
TARGET_GROUP_EXECUTOR_test_fa2_$(1)_$(2) = gtest-multi
TARGET_GROUP_FILTER_test_fa2_$(1)_$(2) = *
endef
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval $(call REGISTER_FA2_MODE,breakdown,$(mode))))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval $(call REGISTER_FA2_MODE,scaling,$(mode))))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval $(call REGISTER_FA2_MODE,concurrency,$(mode))))
$(eval $(call REGISTER_FA2_MODE,breakdown,all))
$(eval $(call REGISTER_FA2_MODE,scaling,all))
$(eval $(call REGISTER_FA2_MODE,concurrency,all))
TARGET_GROUP_BUILD_GROUP_test_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BINARY_GROUP_test_fa2_breakdown_all = fa2-breakdown
TARGET_GROUP_BUILD_GROUP_test_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BINARY_GROUP_test_fa2_scaling_all = fa2-scaling
TARGET_GROUP_BUILD_GROUP_test_fa2_concurrency_all = fa2-concurrency
TARGET_GROUP_BINARY_GROUP_test_fa2_concurrency_all = fa2-concurrency

TARGET_GROUP_BUILD_GROUP_test_fa3_smoke = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_smoke = fa3
TARGET_GROUP_EXECUTOR_test_fa3_smoke = gtest-single
TARGET_GROUP_FILTER_test_fa3_smoke = Fa3PrefillFp16SmokeTest.*:Fa3PrefillFp16BackwardSmokeTest.*:Fa3FwdHdim128Fp16IntegrationTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_small = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_small = fa3
TARGET_GROUP_EXECUTOR_test_fa3_small = gtest-single
TARGET_GROUP_FILTER_test_fa3_small = Fa3PrefillFp16SmallTest.*:Fa3PrefillFp16BackwardSmallTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_medium = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_medium = fa3
TARGET_GROUP_EXECUTOR_test_fa3_medium = gtest-single
TARGET_GROUP_FILTER_test_fa3_medium = Fa3PrefillFp16MediumTest.*:Fa3PrefillFp16BackwardMediumTest.*
TARGET_GROUP_BUILD_GROUP_test_fa3_large = fa3
TARGET_GROUP_BINARY_GROUP_test_fa3_large = fa3
TARGET_GROUP_EXECUTOR_test_fa3_large = gtest-single
TARGET_GROUP_FILTER_test_fa3_large = Fa3PrefillFp16IntegrationTest.*:Fa3PrefillFp16BackwardIntegrationTest.*

FA3_GROUP_CASES_breakdown = H1D128FullB1S4096
FA3_GROUP_CASES_scaling = \
	H1D128FullB1S128 H1D128FullB1S256 H1D128FullB1S512 \
	H1D128FullB1S1024 H1D128FullB1S2048 H1D128FullB1S4096 \
	H1D128FullB1S8192
FA3_GROUP_CASES_concurrency = \
	H16D128FullB64S512 H4D128FullB64S512 H16D128FullB16S512 H1D128FullB1S512 \
	H16D128FullB32S1024 H4D128FullB32S1024 H16D128FullB8S1024 H1D128FullB1S1024 \
	H16D128FullB16S2048 H4D128FullB16S2048 H16D128FullB4S2048 H1D128FullB1S2048 \
	H16D128FullB8S4096 H4D128FullB8S4096 H16D128FullB2S4096 H1D128FullB1S4096 \
	H16D128FullB4S8192 H4D128FullB4S8192 H16D128FullB1S8192 H1D128FullB1S8192

define REGISTER_FA3_MODE
TARGET_GROUP_BUILD_GROUP_test_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_BINARY_GROUP_test_fa3_$(1)_$(2) = fa3-mode-$(2)
TARGET_GROUP_EXECUTOR_test_fa3_$(1)_$(2) = fa3-profile
TARGET_GROUP_FILTER_test_fa3_$(1)_$(2) = Fa3H1D128ProfileTest.SelectedD128FullCases
TARGET_GROUP_CASES_test_fa3_$(1)_$(2) = $(FA3_GROUP_CASES_$(1))
endef
$(foreach group,breakdown scaling concurrency,$(foreach mode,$(HOPPER_FA3_MODES),$(eval $(call REGISTER_FA3_MODE,$(group),$(mode)))))
$(foreach group,breakdown scaling concurrency,$(eval $(call REGISTER_FA3_MODE,$(group),all)))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BUILD_GROUP_test_fa3_$(group)_all = fa3-modes))
$(foreach group,breakdown scaling concurrency,$(eval TARGET_GROUP_BINARY_GROUP_test_fa3_$(group)_all = fa3-modes))

SUITE_TARGET_BUILD_GROUP_microbench_default = microbench-default
SUITE_TARGET_BINARY_GROUP_microbench_default = microbench-default
SUITE_TARGET_EXECUTOR_microbench_default = gtest-multi
SUITE_TARGET_DEFAULT_CONFIG_microbench_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_microbench_default = 12.0
SUITE_TARGET_CUDA_ARCH_microbench_default = sm_120a

SUITE_TARGET_BUILD_GROUP_microbench_sm90 = microbench-sm90
SUITE_TARGET_BINARY_GROUP_microbench_sm90 = microbench-sm90
SUITE_TARGET_EXECUTOR_microbench_sm90 = gtest-multi
SUITE_TARGET_DEFAULT_CONFIG_microbench_sm90 = SM90_H100
SUITE_TARGET_REQUIRED_CC_microbench_sm90 = 9.0
SUITE_TARGET_CUDA_ARCH_microbench_sm90 = sm_90a

SUITE_TARGET_BUILD_GROUP_dev_default = dev
SUITE_TARGET_BINARY_GROUP_dev_default = dev
SUITE_TARGET_EXECUTOR_dev_default = dev
SUITE_TARGET_DEFAULT_CONFIG_dev_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_dev_default = 12.0
SUITE_TARGET_CUDA_ARCH_dev_default = sm_120a

SUITE_TARGET_BUILD_GROUP_trace_default = trace
SUITE_TARGET_BINARY_GROUP_trace_default = none
SUITE_TARGET_EXECUTOR_trace_default = trace
SUITE_TARGET_DEFAULT_CONFIG_trace_default = SM120_RTX5090
SUITE_TARGET_REQUIRED_CC_trace_default = 12.0
SUITE_TARGET_CUDA_ARCH_trace_default = sm_120a

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

.PHONY: test test-sm90 microbench-default microbench-sm90 standalone-bench cp-async-bench mma-standalone-bench tma-standalone-bench memory-standalone-bench dev prepare-fa3-flash-attention fa2-standard fa2-smoke fa2-small fa2-medium fa2-large fa2-breakdown fa2-scaling fa2-concurrency fa3 fa3-single-tile fa3-h1d128-profile fa3-modes list-test-groups print-test-binaries list-suites list-suite-targets print-suite-default-target print-suite-target-metadata list-target-groups list-target-group-modes print-target-group-metadata clean help setup-gtest

# Machine-readable manifest used by run_tests.sh.
list-test-groups:
	@printf '%s\n' $(TEST_GROUPS)

print-test-binaries:
	@if [ -z "$(TEST_GROUP)" ]; then \
		echo "TEST_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TEST_GROUP),$(TEST_GROUPS))" ]; then \
		echo "Unknown TEST_GROUP: $(TEST_GROUP)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TEST_GROUP_BINARIES_$(TEST_GROUP)); \
	fi

list-suites:
	@printf '%s\n' $(TEST_SUITES)

list-suite-targets:
	@if [ -z "$(SUITE)" ]; then \
		echo "SUITE is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(SUITE_TARGETS_$(SUITE)); \
	fi

print-suite-default-target:
	@if [ -z "$(SUITE)" ]; then \
		echo "SUITE is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	else \
		printf '%s\n' "$(SUITE_DEFAULT_TARGET_$(SUITE))"; \
	fi

print-suite-target-metadata:
	@if [ -z "$(SUITE)" ] || [ -z "$(SUITE_TARGET)" ]; then \
		echo "SUITE and SUITE_TARGET are required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE),$(TEST_SUITES))" ]; then \
		echo "Unknown SUITE: $(SUITE)" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE_TARGET),$(SUITE_TARGETS_$(SUITE)))" ]; then \
		echo "Unknown target for $(SUITE): $(SUITE_TARGET)" >&2; \
		exit 2; \
	else \
		printf '%s|%s|%s|%s|%s|%s\n' \
			"$(SUITE_TARGET_BUILD_GROUP_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_BINARY_GROUP_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_EXECUTOR_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_DEFAULT_CONFIG_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_REQUIRED_CC_$(SUITE)_$(SUITE_TARGET))" \
			"$(SUITE_TARGET_CUDA_ARCH_$(SUITE)_$(SUITE_TARGET))"; \
	fi

list-target-groups:
	@if [ -z "$(SUITE)" ] || [ -z "$(SUITE_TARGET)" ]; then \
		echo "SUITE and SUITE_TARGET are required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(SUITE_TARGET),$(SUITE_TARGETS_$(SUITE)))" ]; then \
		echo "Unknown target for $(SUITE): $(SUITE_TARGET)" >&2; \
		exit 2; \
	elif [ -z "$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET))" ]; then \
		echo "$(SUITE)/$(SUITE_TARGET) does not define groups" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)); \
	fi

list-target-group-modes:
	@if [ -z "$(TARGET_GROUP)" ]; then \
		echo "TARGET_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TARGET_GROUP),$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)))" ]; then \
		echo "Unknown group for $(SUITE)/$(SUITE_TARGET): $(TARGET_GROUP)" >&2; \
		exit 2; \
	else \
		printf '%s\n' $(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)); \
	fi

print-target-group-metadata:
	@if [ -z "$(TARGET_GROUP)" ]; then \
		echo "TARGET_GROUP is required" >&2; \
		exit 2; \
	elif [ -z "$(filter $(TARGET_GROUP),$(TARGET_GROUPS_$(SUITE)_$(SUITE_TARGET)))" ]; then \
		echo "Unknown group for $(SUITE)/$(SUITE_TARGET): $(TARGET_GROUP)" >&2; \
		exit 2; \
	elif [ -n "$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP))" ] && [ -z "$(TARGET_MODE)" ]; then \
		echo "Group $(TARGET_GROUP) requires TARGET_MODE" >&2; \
		exit 2; \
	elif [ -z "$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP))" ] && [ -n "$(TARGET_MODE)" ]; then \
		echo "Group $(TARGET_GROUP) does not accept TARGET_MODE" >&2; \
		exit 2; \
	elif [ -n "$(TARGET_MODE)" ] && [ -z "$(filter $(TARGET_MODE),$(TARGET_GROUP_MODES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)))" ]; then \
		echo "Unknown mode for $(SUITE)/$(SUITE_TARGET)/$(TARGET_GROUP): $(TARGET_MODE)" >&2; \
		exit 2; \
	else \
		printf '%s|%s|%s|%s|%s\n' \
			"$(TARGET_GROUP_BUILD_GROUP_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_BINARY_GROUP_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_EXECUTOR_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_FILTER_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))" \
			"$(TARGET_GROUP_CASES_$(SUITE)_$(SUITE_TARGET)_$(TARGET_GROUP)$(if $(TARGET_MODE),_$(TARGET_MODE)))"; \
	fi

# Build verification tests (unit + integration, used by run_tests.sh and CI)
test: setup-gtest $(MAIN_TEST_TARGET)

microbench-default: setup-gtest $(MICROBENCH_DEFAULT_TARGETS)

microbench-sm90: setup-gtest $(MICROBENCH_SM90_TARGETS)

# Build standalone CUDA calibration microbenchmarks that are not gtest binaries.
standalone-bench: cp-async-bench mma-standalone-bench tma-standalone-bench memory-standalone-bench

cp-async-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/cp_async all ptx-bench

mma-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma sm120a sm90a saturation-sm120a saturation-sm90a

tma-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/tma all

memory-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/memory all

# Build standalone dev tests (decoupled from simulator)
dev: setup-gtest $(DEV_TEST_TARGET)

# Build SM90-only instruction tests.
test-sm90: setup-gtest $(SM90_TEST_TARGETS)

prepare-fa3-flash-attention: $(HOPPER_FA3_PREPARED_STAMP)

$(HOPPER_FA3_PREPARED_STAMP): $(HOPPER_FA3_PREPARE_SCRIPT) $(HOPPER_FA3_PATCHES)
	$(HOPPER_FA3_PREPARE_SCRIPT)
	@touch $@

$(HOPPER_FA3_GENERATED_HEADERS): $(HOPPER_FA3_PREPARED_STAMP)
	@test -f $@ || { echo "Prepared FlashAttention header is missing: $@" >&2; exit 1; }

# Build FA2 forward-only runners with extended-shape macro enabled.
fa2-standard: setup-gtest $(HOPPER_FA2_STANDARD_TARGETS)

fa2-smoke: setup-gtest $(HOPPER_FA2_SMOKE_TARGETS)

fa2-small: setup-gtest $(HOPPER_FA2_SMALL_TARGETS)

fa2-medium: setup-gtest $(HOPPER_FA2_MEDIUM_TARGETS)

fa2-large: setup-gtest $(HOPPER_FA2_LARGE_TARGETS)

fa2-breakdown: setup-gtest $(HOPPER_FA2_BREAKDOWN_TARGETS)

fa2-scaling: setup-gtest $(HOPPER_FA2_SCALING_TARGETS)

fa2-concurrency: setup-gtest $(HOPPER_FA2_CONCURRENCY_TARGETS)

define REGISTER_FA2_BUILD_TARGET
.PHONY: fa2-$(1)-$(2)
fa2-$(1)-$(2): setup-gtest $(BIN_DIR)/hopper/run_fa2_$(1)_$(2)_tests
endef
$(foreach mode,$(HOPPER_FA2_BREAKDOWN_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,breakdown,$(mode))))
$(foreach mode,$(HOPPER_FA2_SCALING_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,scaling,$(mode))))
$(foreach mode,$(HOPPER_FA2_CONCURRENCY_MODES),$(eval $(call REGISTER_FA2_BUILD_TARGET,concurrency,$(mode))))

# Build FA3-only runners. FA3 always uses extended GMMA shapes.
fa3: setup-gtest $(HOPPER_FA3_EXTENDED_TARGET)

# Build only the FA3 single-tile profiling runner.
fa3-single-tile: setup-gtest $(HOPPER_FA3_SINGLE_TILE_TARGET)

# Build only the FA3 H1/D128/full profiling runner.
fa3-h1d128-profile: setup-gtest $(HOPPER_FA3_H1D128_PROFILE_TARGET)

# Build FA3 analysis modes; groups reuse these binaries with different cases.
fa3-modes: setup-gtest $(HOPPER_FA3_MODE_TARGETS)

define REGISTER_FA3_BUILD_TARGET
.PHONY: fa3-mode-$(1)
fa3-mode-$(1): setup-gtest $(call HOPPER_FA3_MODE_TARGET,$(1))
endef
$(foreach mode,$(HOPPER_FA3_MODES),$(eval $(call REGISTER_FA3_BUILD_TARGET,$(mode))))

# Help target
help:
	@echo "Internal test build graph"
	@echo "========================="
	@echo "Use './run_tests.sh help' for the supported test CLI."
	@echo "The suite build targets in this Makefile are internal implementation details."
	@echo ""
	@echo "Standalone calibration targets:"
	@echo "  standalone-bench - Build standalone CUDA calibration microbenchmarks"
	@echo "  cp-async-bench - Build standalone cp.async calibration binaries"
	@echo "  mma-standalone-bench - Build standalone warp-MMA calibration binaries"
	@echo "  prepare-fa3-flash-attention - Prepare local flash-attention headers for FA2/FA3"
	@echo ""
	@echo "Registry inspection:"
	@echo "  list-test-groups - List groups in the binary manifest"
	@echo "  print-test-binaries TEST_GROUP=<group> - Print one group's binaries"
	@echo "  list-suites - List user-facing suites"
	@echo "  list-suite-targets SUITE=<suite> - List one suite's targets"

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
$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(UNIT_OBJ_DIR) $(INTEGRATION_OBJ_DIR) $(INTEGRATION_MMA_OBJ_DIR) $(MICROBENCH_OBJ_DIR) $(STANDALONE_OBJ_DIR) $(HOPPER_OBJ_DIR) $(FLASH_OBJ_DIR):
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

# SM90 instruction tests use their own CUDA architecture and do not depend on
# FA2/FA3 headers or a prepared FlashAttention checkout.
$(OBJ_DIR)/hopper/%.cu.o: $(HOPPER_SRC_DIR)/%.cc $(SM90_TEST_CUH_HEADERS) Makefile | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# FA2 stays in split runners so each compile unit instantiates one D/full-causal
# kernel family. It still uses HOPPER_CUDA_ARCH, which defaults to sm_90a.
HOPPER_FA2_H32D64_FULL_FLAGS = -DFA2_PREFILL_ENABLE_H32D64_FULL
HOPPER_FA2_H32D64_CAUSAL_FLAGS = -DFA2_PREFILL_ENABLE_H32D64_CAUSAL
HOPPER_FA2_H16D128_FULL_FLAGS = -DFA2_PREFILL_ENABLE_H16D128_FULL
HOPPER_FA2_H16D128_CAUSAL_FLAGS = -DFA2_PREFILL_ENABLE_H16D128_CAUSAL

$(OBJ_DIR)/hopper/fa2/smoke_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/smoke_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMOKE $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/small_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SMALL $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/medium_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_MEDIUM $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h32d64_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H32D64_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h32d64_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H32D64_CAUSAL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h16d128_full/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H16D128_FULL_FLAGS)
$(OBJ_DIR)/hopper/fa2/large_h16d128_causal/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_LARGE $(HOPPER_FA2_H16D128_CAUSAL_FLAGS)

$(HOPPER_FA2_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_BREAKDOWN_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_BREAKDOWN $(HOPPER_FA2_H16D128_FULL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/breakdown_baseline/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS)
$(OBJ_DIR)/hopper/fa2/breakdown_skip_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC
$(OBJ_DIR)/hopper/fa2/breakdown_skip_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_MMA
$(OBJ_DIR)/hopper/fa2/breakdown_skip_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_fma_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_FMA_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/breakdown_only_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA
$(OBJ_DIR)/hopper/fa2/breakdown_nothing/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_BREAKDOWN_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_SOFTMAX

$(HOPPER_FA2_BREAKDOWN_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_SCALING_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_SCALING $(HOPPER_FA2_H16D128_FULL_FLAGS) $(HOPPER_FA2_H16D128_CAUSAL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/scaling_baseline/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS)
$(OBJ_DIR)/hopper/fa2/scaling_nothing/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_only_softmax/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS
$(OBJ_DIR)/hopper/fa2/scaling_only_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/scaling_softmax_mma/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_SCALING_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC

$(HOPPER_FA2_SCALING_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA2_CONCURRENCY_FLAGS = $(HOPPER_FA2_FLAGS) -DFA2_PREFILL_GROUP_CONCURRENCY $(HOPPER_FA2_H16D128_FULL_FLAGS) -DFA2_FWD_SENS_FORCE_H100_HDIM128
$(OBJ_DIR)/hopper/fa2/concurrency_only_cp_async/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX
$(OBJ_DIR)/hopper/fa2/concurrency_only_cp_async_bhhalf/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_PREFILL_CONCURRENCY_BH_HALF -DFA2_FWD_SENS_SKIP_MMA -DFA2_FWD_SENS_SKIP_MMA_OPERANDS -DFA2_FWD_SENS_SKIP_SOFTMAX -DFA2_FWD_CTA_PROFILE
$(OBJ_DIR)/hopper/fa2/concurrency_qk_softmax_pv_only/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC
$(OBJ_DIR)/hopper/fa2/concurrency_qk_pv_only/fa2_fwd_fp16_test.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA2_CONCURRENCY_FLAGS) -DFA2_FWD_SENS_SKIP_CP_ASYNC -DFA2_FWD_SENS_SKIP_SOFTMAX

$(HOPPER_FA2_CONCURRENCY_OBJECTS): $(HOPPER_FA2_TEST_SOURCE) $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/hopper/fa3_extended/%.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_EXTENDED_FLAGS)
$(OBJ_DIR)/hopper/fa3_extended/%.cu.o: $(HOPPER_SRC_DIR)/fa3/%.cc $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# FA3 profile runner intentionally compiles a separate object with clock64
# instrumentation enabled, so the normal Hopper/collector baseline stays clean.
$(OBJ_DIR)/hopper/fa3_profile/%.cu.o: HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_FLAGS) -DFLASH_FWD_ENABLE_PROFILE_CLOCK
$(OBJ_DIR)/hopper/fa3_profile/%.cu.o: $(HOPPER_SRC_DIR)/fa3/%.cc $(CUH_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

HOPPER_FA3_MODE_PROFILE_FLAGS = $(HOPPER_FA3_EXTENDED_FLAGS) -DFLASH_FWD_ENABLE_PROFILE_CLOCK -DFLASH_FWD_PROFILE_CLOCK_NON_ATOMIC -DFLASH_FWD_PROFILE_GLOBALTIMER -DFLASH_FWD_PROFILE_TASK_GLOBALTIMER
HOPPER_FA3_MODE_NOPROFILE_FLAGS = $(HOPPER_FA3_EXTENDED_FLAGS)
HOPPER_FA3_MODE_EXTENDED_PROFILE_FLAGS = $(HOPPER_FA3_EXTENDED_FLAGS) -DFLASH_FWD_ENABLE_PROFILE_CLOCK -DFLASH_FWD_PROFILE_CLOCK_NON_ATOMIC
HOPPER_FA3_MODE_EXTENDED_NOPROFILE_FLAGS = $(HOPPER_FA3_EXTENDED_FLAGS)
$(call HOPPER_FA3_MODE_OBJECT,baseline): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS)
$(call HOPPER_FA3_MODE_OBJECT,softmax_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call HOPPER_FA3_MODE_OBJECT,qk_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call HOPPER_FA3_MODE_OBJECT,pv_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_QK_WGMMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call HOPPER_FA3_MODE_OBJECT,tma_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call HOPPER_FA3_MODE_OBJECT,tma_only_m3_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_M3_ONLY
$(call HOPPER_FA3_MODE_OBJECT,tma_only_first_round_bh): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_FIRST_BH_SECTION_ONLY
$(call HOPPER_FA3_MODE_OBJECT,tma_only_first_round_bh_m3_only): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_PROFILE_SCHED_FIRST_BH_SECTION_ONLY -DFLASH_FWD_PROFILE_SCHED_M3_ONLY
$(call HOPPER_FA3_MODE_OBJECT,baseline_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS)
$(call HOPPER_FA3_MODE_OBJECT,softmax_only_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX
$(call HOPPER_FA3_MODE_OBJECT,base_no_tma): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,softmax_only_no_tma): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,base_no_tma_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,softmax_only_no_tma_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_timeline): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_TIMELINE
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_reg_timeline): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_REG_TIMELINE -DFLASH_FWD_PROFILE_REG_TIMELINE_ONLY
$(call HOPPER_FA3_MODE_OBJECT,sync_only_no_tma): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_SENS_SYNC_ONLY
$(call HOPPER_FA3_MODE_OBJECT,sync_only_no_tma_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_QK_WGMMA -DFLASH_FWD_SENS_SKIP_PV_WGMMA -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_SENS_SYNC_ONLY
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended): HOPPER_NVCCFLAGS = $(WGMMA_SM90A_NVCCFLAGS)
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_EXTENDED_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_noprofile): HOPPER_NVCCFLAGS = $(WGMMA_SM90A_NVCCFLAGS)
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_noprofile): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_EXTENDED_NOPROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_reg_timeline): HOPPER_NVCCFLAGS = $(WGMMA_SM90A_NVCCFLAGS)
$(call HOPPER_FA3_MODE_OBJECT,qk_pv_only_no_tma_extended_reg_timeline): HOPPER_EXTRA_NVCCFLAGS = $(HOPPER_FA3_MODE_EXTENDED_PROFILE_FLAGS) -DFLASH_FWD_SENS_SKIP_SOFTMAX -DFLASH_FWD_SENS_SKIP_TMA -DFLASH_FWD_PROFILE_REG_TIMELINE -DFLASH_FWD_PROFILE_REG_TIMELINE_ONLY

$(HOPPER_FA3_MODE_OBJECTS): $(HOPPER_SRC_DIR)/fa3/fa3_fwd_h1d128_profile_test.cc $(CUH_HEADERS) $(HOPPER_FA3_PROFILE_HEADERS) Makefile $(HOPPER_FA3_PREPARED_STAMP) | $(HOPPER_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(HOPPER_NVCCFLAGS) $(HOPPER_EXTRA_NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Every WGMMA microbenchmark is forced to SM90A.
$(OBJ_DIR)/microbench/wgmma/%.cu.o: NVCCFLAGS = $(WGMMA_SM90A_NVCCFLAGS)
$(OBJ_DIR)/microbench/%.cu.o: $(TEST_SRC_DIR)/microbench/%.cc $(CUH_HEADERS) | $(MICROBENCH_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Standalone dev tests
$(OBJ_DIR)/standalone/%.cu.o: $(TEST_SRC_DIR)/standalone/%.cc $(CUH_HEADERS) | $(STANDALONE_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

# Build individual test executables
$(BIN_DIR)/%_test: $(OBJ_DIR)/%_test.o $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread

# Build main test runner
$(MAIN_TEST_TARGET): $(TEST_OBJECTS) $(FLASH_OBJECTS) $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread $(CUDA_LIBS)

# Build the SM90 instruction-test runner.
$(SM90_TEST_TARGET): $(SM90_TEST_OBJECTS) $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

# Build split FA2 forward-only runners.
$(BIN_DIR)/hopper/run_fa2_%_tests: $(OBJ_DIR)/hopper/fa2/%/fa2_fwd_fp16_test.cu.o $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

# Build FA3-only runner. Extended GMMA shapes are mandatory for FA3.
$(HOPPER_FA3_EXTENDED_TARGET): $(HOPPER_FA3_EXTENDED_OBJECTS) $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

# Build FA3 single-tile profiling runner without linking unrelated Hopper tests.
$(HOPPER_FA3_SINGLE_TILE_TARGET): $(HOPPER_FA3_SINGLE_TILE_OBJECT) $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

# Build FA3 H1/D128/full profiling runner without linking unrelated Hopper tests.
$(HOPPER_FA3_H1D128_PROFILE_TARGET): $(HOPPER_FA3_H1D128_PROFILE_OBJECT) $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

$(BIN_DIR)/hopper/run_fa3_%_tests: $(OBJ_DIR)/hopper/fa3_modes/%/fa3_fwd_h1d128_profile_test.cu.o $(OBJ_DIR)/gtest_main.a Makefile | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out Makefile,$^) -o $@ -lpthread $(CUDA_LIBS)

# Build individual microbench executables
$(BIN_DIR)/%_bench: $(OBJ_DIR)/microbench/%_bench.cu.o $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread $(CUDA_LIBS)

# Build dev test runner (standalone tests)
$(DEV_TEST_TARGET): $(STANDALONE_OBJECTS) $(OBJ_DIR)/gtest_main.a | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpthread $(CUDA_LIBS)

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
	@echo "SM90_TEST_SOURCES: $(SM90_TEST_SOURCES)"
	@echo "SM90_TEST_TARGETS: $(SM90_TEST_TARGETS)"
	@echo "MAIN_TEST_TARGET: $(MAIN_TEST_TARGET)"
	@echo "SM90_TEST_TARGET: $(SM90_TEST_TARGET)"
	@echo "CUH_HEADERS: $(CUH_HEADERS)"
