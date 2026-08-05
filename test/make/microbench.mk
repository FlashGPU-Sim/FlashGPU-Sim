# Architecture- and mechanism-scoped microbenchmarks.

MICROBENCH_MK := $(lastword $(MAKEFILE_LIST))

MICROBENCH_ALL_GTEST_SOURCES = $(shell find $(TEST_SRC_DIR)/microbench -name '*_bench.cc' 2>/dev/null)
MICROBENCH_SM120_MBAR_SOURCE = $(filter $(TEST_SRC_DIR)/microbench/mbarrier/%,$(MICROBENCH_ALL_GTEST_SOURCES))
MICROBENCH_SM120_MMA_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/mma/%,$(MICROBENCH_ALL_GTEST_SOURCES))
MICROBENCH_SM90_WGMMA_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_ALL_GTEST_SOURCES))

MICROBENCH_SM120_MBAR_TARGETS = $(MICROBENCH_SM120_MBAR_SOURCE:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_SM120_MMA_TARGETS = $(MICROBENCH_SM120_MMA_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_SM90_WGMMA_TARGETS = $(MICROBENCH_SM90_WGMMA_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_OBJ_DIR = $(OBJ_DIR)/microbench

.PHONY: microbench-sm120-mbarrier microbench-sm120-mma \
microbench-sm120-memory microbench-sm90-cp-async microbench-sm90-mma \
microbench-sm90-tma microbench-sm90-wgmma

microbench-sm120-mbarrier: setup-gtest $(MICROBENCH_SM120_MBAR_TARGETS)

# The MMA group contains gtest timing probes plus standalone calibration binaries.
microbench-sm120-mma: setup-gtest $(MICROBENCH_SM120_MMA_TARGETS)
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma sm120a saturation-sm120a

microbench-sm120-memory:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/memory \
		ARCH=sm_120a PTX_PROFILE=compute_120a all

microbench-sm90-cp-async:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/cp_async \
		ARCH=sm_90a PTX_PROFILE=compute_90a all ptx-bench issue-scope

microbench-sm90-mma:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma sm90a saturation-sm90a

microbench-sm90-tma:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/tma \
		ARCH=sm_90a PTX_PROFILE=compute_90a all

microbench-sm90-wgmma: setup-gtest $(MICROBENCH_SM90_WGMMA_TARGETS)

$(MICROBENCH_OBJ_DIR):
	mkdir -p $@

# Every WGMMA microbenchmark is forced to SM90A.
$(OBJ_DIR)/microbench/wgmma/%.cu.o: NVCCFLAGS = $(WGMMA_SM90A_NVCCFLAGS)
$(OBJ_DIR)/microbench/%.cu.o: $(TEST_SRC_DIR)/microbench/%.cc \
$(CUH_HEADERS) $(TOP_MAKEFILE) $(MICROBENCH_MK) | $(MICROBENCH_OBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPGPUSIM_FLAGS) -c $< -o $@

$(BIN_DIR)/%_bench: $(OBJ_DIR)/microbench/%_bench.cu.o \
$(OBJ_DIR)/gtest_main.a $(TOP_MAKEFILE) $(MICROBENCH_MK) | $(BIN_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter-out $(TOP_MAKEFILE) $(MICROBENCH_MK),$^) \
		-o $@ -lpthread $(CUDA_LIBS)
