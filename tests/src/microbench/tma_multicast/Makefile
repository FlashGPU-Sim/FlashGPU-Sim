ARCH ?= sm_90a
PTX_PROFILE ?= compute_90a
include ../common.mk

HOST := ../h200_host
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/tma_multicast
TARGET ?= $(BIN_DIR)/tma_multicast_bench
NVCC_FLAGS += -I$(HOST)

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cu probe_tma_multicast.cu $(HOST)/suite_main.cu $(HOST)/common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) main.cu probe_tma_multicast.cu $(HOST)/suite_main.cu -o $@

clean:
	rm -f $(TARGET)
