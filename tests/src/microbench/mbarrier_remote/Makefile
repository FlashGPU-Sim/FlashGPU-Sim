ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
include ../common.mk

HOST := ../h200_host
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/mbarrier_remote
TARGET ?= $(BIN_DIR)/mbarrier_remote_bench
NVCC_FLAGS += -I$(HOST)

.PHONY: all clean

all: $(TARGET)

$(TARGET): mbarrier_remote_bench.cu $(HOST)/suite_main.cu $(HOST)/common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) mbarrier_remote_bench.cu $(HOST)/suite_main.cu -o $@

clean:
	rm -f $(TARGET)
