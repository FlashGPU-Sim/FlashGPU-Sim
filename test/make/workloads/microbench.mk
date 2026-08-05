# Architecture- and mechanism-scoped microbenchmarks.

MICROBENCH_MK := $(lastword $(MAKEFILE_LIST))

# Microbenchmark profiles retain their existing local source/build layouts.
TEST_GROUP_PROFILES_sm90_microbench := cp-async mma tma wgmma
TEST_GROUP_BUILD_TARGET_sm90_microbench_cp-async := microbench-sm90-cp-async
TEST_GROUP_BINARY_GROUP_sm90_microbench_cp-async := none
TEST_GROUP_EXECUTOR_sm90_microbench_cp-async := build-only
TEST_GROUP_FILTER_sm90_microbench_cp-async := *
TEST_GROUP_BUILD_TARGET_sm90_microbench_mma := microbench-sm90-mma
TEST_GROUP_BINARY_GROUP_sm90_microbench_mma := none
TEST_GROUP_EXECUTOR_sm90_microbench_mma := build-only
TEST_GROUP_FILTER_sm90_microbench_mma := *
TEST_GROUP_BUILD_TARGET_sm90_microbench_tma := microbench-sm90-tma
TEST_GROUP_BINARY_GROUP_sm90_microbench_tma := none
TEST_GROUP_EXECUTOR_sm90_microbench_tma := build-only
TEST_GROUP_FILTER_sm90_microbench_tma := *
TEST_GROUP_BUILD_TARGET_sm90_microbench_wgmma := microbench-sm90-wgmma
TEST_GROUP_BINARY_GROUP_sm90_microbench_wgmma := microbench-sm90-wgmma
TEST_GROUP_EXECUTOR_sm90_microbench_wgmma := gtest-multi
TEST_GROUP_FILTER_sm90_microbench_wgmma := *

TEST_GROUP_PROFILES_sm120_microbench := mbarrier mma memory
TEST_GROUP_BUILD_TARGET_sm120_microbench_mbarrier := microbench-sm120-mbarrier
TEST_GROUP_BINARY_GROUP_sm120_microbench_mbarrier := microbench-sm120-mbarrier
TEST_GROUP_EXECUTOR_sm120_microbench_mbarrier := gtest-multi
TEST_GROUP_FILTER_sm120_microbench_mbarrier := *
TEST_GROUP_BUILD_TARGET_sm120_microbench_mma := microbench-sm120-mma
TEST_GROUP_BINARY_GROUP_sm120_microbench_mma := microbench-sm120-mma
TEST_GROUP_EXECUTOR_sm120_microbench_mma := gtest-multi
TEST_GROUP_FILTER_sm120_microbench_mma := *
TEST_GROUP_BUILD_TARGET_sm120_microbench_memory := microbench-sm120-memory
TEST_GROUP_BINARY_GROUP_sm120_microbench_memory := none
TEST_GROUP_EXECUTOR_sm120_microbench_memory := build-only
TEST_GROUP_FILTER_sm120_microbench_memory := *

MICROBENCH_SM120_GTEST_SOURCES = $(filter %_bench.cc,$(TEST_GROUP_SOURCES_sm120_microbench))
MICROBENCH_SM90_GTEST_SOURCES = $(filter %_bench.cc,$(TEST_GROUP_SOURCES_sm90_microbench))
MICROBENCH_SM120_MBAR_SOURCE = $(filter $(TEST_SRC_DIR)/microbench/mbarrier/%,$(MICROBENCH_SM120_GTEST_SOURCES))
MICROBENCH_SM120_MMA_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/mma/%,$(MICROBENCH_SM120_GTEST_SOURCES))
MICROBENCH_SM90_WGMMA_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_SM90_GTEST_SOURCES))

MICROBENCH_SM120_MBAR_TARGETS = $(MICROBENCH_SM120_MBAR_SOURCE:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/sm120/microbench/%_bench)
MICROBENCH_SM120_MMA_TARGETS = $(MICROBENCH_SM120_MMA_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/sm120/microbench/%_bench)
MICROBENCH_SM90_WGMMA_TARGETS = $(MICROBENCH_SM90_WGMMA_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/sm90/microbench/%_bench)
MICROBENCH_SM120_GTEST_OBJECTS = $(MICROBENCH_SM120_GTEST_SOURCES:$(TEST_SRC_DIR)/microbench/%.cc=$(OBJ_DIR)/sm120/microbench/%.cu.o)
MICROBENCH_SM90_GTEST_OBJECTS = $(MICROBENCH_SM90_GTEST_SOURCES:$(TEST_SRC_DIR)/microbench/%.cc=$(OBJ_DIR)/sm90/microbench/%.cu.o)
MICROBENCH_SM120_ARCH_TAG = $(subst _,,$(ARCH_NVCC_TARGET_sm120))
MICROBENCH_SM90_ARCH_TAG = $(subst _,,$(ARCH_NVCC_TARGET_sm90))

.SECONDARY: $(MICROBENCH_SM120_GTEST_OBJECTS) $(MICROBENCH_SM90_GTEST_OBJECTS)

.PHONY: microbench-sm120-mbarrier microbench-sm120-mma \
microbench-sm120-memory microbench-sm90-cp-async microbench-sm90-mma \
microbench-sm90-tma microbench-sm90-wgmma

microbench-sm120-mbarrier: setup-gtest $(MICROBENCH_SM120_MBAR_TARGETS)

# The MMA group contains gtest timing probes plus standalone calibration binaries.
microbench-sm120-mma: setup-gtest $(MICROBENCH_SM120_MMA_TARGETS)
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma \
		ARCH=$(ARCH_NVCC_TARGET_sm120) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm120) \
		TARGET_NAME=mma_accept_queue_bench_cuda128_$(MICROBENCH_SM120_ARCH_TAG) all
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma \
		ARCH=$(ARCH_NVCC_TARGET_sm120) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm120) \
		SRC=mma_saturation_bench.cu \
		TARGET_NAME=mma_saturation_bench_cuda128_$(MICROBENCH_SM120_ARCH_TAG) all

microbench-sm120-memory:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/memory \
		ARCH=$(ARCH_NVCC_TARGET_sm120) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm120) all

microbench-sm90-cp-async:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/cp_async \
		ARCH=$(ARCH_NVCC_TARGET_sm90) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm90) all ptx-bench issue-scope

microbench-sm90-mma:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma \
		ARCH=$(ARCH_NVCC_TARGET_sm90) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm90) \
		TARGET_NAME=mma_accept_queue_bench_cuda128_$(MICROBENCH_SM90_ARCH_TAG) all
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma \
		ARCH=$(ARCH_NVCC_TARGET_sm90) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm90) \
		SRC=mma_saturation_bench.cu \
		TARGET_NAME=mma_saturation_bench_cuda128_$(MICROBENCH_SM90_ARCH_TAG) all

microbench-sm90-tma:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/tma \
		ARCH=$(ARCH_NVCC_TARGET_sm90) PTX_PROFILE=$(ARCH_COMPUTE_TARGET_sm90) all

microbench-sm90-wgmma: setup-gtest $(MICROBENCH_SM90_WGMMA_TARGETS)

$(OBJ_DIR)/sm120/microbench/%.cu.o: $(TEST_SRC_DIR)/microbench/%.cc \
$(CUH_HEADERS) $(TOP_MAKEFILE) $(MICROBENCH_MK) arch/sm120.toml
	@mkdir -p $(dir $@)
	$(NVCC) $(BASE_NVCCFLAGS) -arch=$(ARCH_NVCC_TARGET_sm120) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(OBJ_DIR)/sm90/microbench/wgmma/%.cu.o: \
$(TEST_SRC_DIR)/microbench/wgmma/%.cc $(CUH_HEADERS) \
$(TOP_MAKEFILE) $(MICROBENCH_MK) arch/sm90.toml
	@mkdir -p $(dir $@)
	$(NVCC) $(SM90_NVCCFLAGS) $(WGMMA_EXTRA_NVCCFLAGS) \
		$(INCLUDES) $(GPGPUSIM_FLAGS) \
		-c $< -o $@

$(BIN_DIR)/sm120/microbench/%_bench: \
$(OBJ_DIR)/sm120/microbench/%_bench.cu.o \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(MICROBENCH_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(MICROBENCH_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)

$(BIN_DIR)/sm90/microbench/wgmma/%_bench: \
$(OBJ_DIR)/sm90/microbench/wgmma/%_bench.cu.o \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(MICROBENCH_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(MICROBENCH_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
