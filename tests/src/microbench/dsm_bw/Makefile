# seanzw/random dsm_bw. Built by tests/run_tests.py --profile dsm-bw.
ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
include ../common.mk

BIN_DIR ?= $(STANDALONE_BIN_ROOT)/dsm_bw
TARGET ?= $(BIN_DIR)/dsm_h200
NVCC_FLAGS += -Xptxas=-dlcm=cg

.PHONY: all clean

all: $(TARGET)

$(TARGET): benchmark.cu kernels.cuh common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) benchmark.cu -o $@

clean:
	rm -f $(TARGET)
