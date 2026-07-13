# GTest microbenchmarks and standalone CUDA calibration binaries.

MICROBENCH_MK := $(lastword $(MAKEFILE_LIST))

MICROBENCH_ALL_SOURCES = $(shell find $(TEST_SRC_DIR)/microbench -name '*_bench.cc' 2>/dev/null)
MICROBENCH_SM90_SOURCES = $(filter $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_ALL_SOURCES))
MICROBENCH_DEFAULT_SOURCES = $(filter-out $(TEST_SRC_DIR)/microbench/wgmma/%,$(MICROBENCH_ALL_SOURCES))

# Separate binaries keep incremental rebuilds and benchmark artifacts isolated.
MICROBENCH_DEFAULT_TARGETS = $(MICROBENCH_DEFAULT_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_SM90_TARGETS = $(MICROBENCH_SM90_SOURCES:$(TEST_SRC_DIR)/microbench/%_bench.cc=$(BIN_DIR)/%_bench)
MICROBENCH_OBJ_DIR = $(OBJ_DIR)/microbench

.PHONY: microbench-default microbench-sm90 standalone-bench cp-async-bench \
mma-standalone-bench tma-standalone-bench memory-standalone-bench

microbench-default: setup-gtest $(MICROBENCH_DEFAULT_TARGETS)

microbench-sm90: setup-gtest $(MICROBENCH_SM90_TARGETS)

standalone-bench: cp-async-bench mma-standalone-bench tma-standalone-bench memory-standalone-bench

cp-async-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/cp_async all ptx-bench

mma-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/mma sm120a sm90a saturation-sm120a saturation-sm90a

tma-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/tma all

memory-standalone-bench:
	$(MAKE) -C $(TEST_SRC_DIR)/microbench/memory all

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
