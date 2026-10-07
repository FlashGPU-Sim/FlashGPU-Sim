# seanzw/random tma_bw, H200 L2-resident and cold-HBM binaries.
# Built by tests/run_tests.py --profile tma-bw.
ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
include ../common.mk

BIN_DIR ?= $(STANDALONE_BIN_ROOT)/tma_bw
L2_TARGET ?= $(BIN_DIR)/bandwidth_test_h200_l2
HBM_TARGET ?= $(BIN_DIR)/bandwidth_test_h200_hbm
SRCS := bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
NVCC_FLAGS += -Xptxas=-dlcm=cg

.PHONY: all clean

all: $(L2_TARGET) $(HBM_TARGET)

$(L2_TARGET): $(SRCS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DBENCH_L2_MODE=1 -DBENCH_FORCE_CG=1 -DBENCH_TOTAL_MIB=40 -DBENCH_REPEAT=32 bandwidth_test.cu -o $@

$(HBM_TARGET): $(SRCS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -DBENCH_HBM_MODE=1 -DBENCH_FORCE_CG=1 -DBENCH_TOTAL_MIB=1024 -DBENCH_REPEAT=1 -DBENCH_REALLOCATE_EACH_CONFIG=0 -DBENCH_FLUSH_L2_MIB=256 bandwidth_test.cu -o $@

clean:
	rm -f $(L2_TARGET) $(HBM_TARGET)
