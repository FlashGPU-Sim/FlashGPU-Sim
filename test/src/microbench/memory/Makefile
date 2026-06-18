SRC := global_load_bw_bench.cu
TARGET_NAME := global_load_bw_bench_cuda129

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
include ../common.mk
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/memory
TARGET ?= $(BIN_DIR)/$(TARGET_NAME)
TARGET_EXE := $(notdir $(TARGET))

CONFIG_DIR ?= ../../../../configs/SM120_RTX5090
RUN_DIR ?= $(STANDALONE_RUN_ROOT)/memory/gem5_global_load_bw_smoke_cuda129

.PHONY: all clean check-ptx gem5-dir gem5-smoke run-hw

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

check-ptx: $(TARGET)
	$(CUOBJDUMP) -lptx $(TARGET)

gem5-dir: $(TARGET)
	mkdir -p $(RUN_DIR)
	rm -f $(RUN_DIR)/*.ptx $(RUN_DIR)/*.ptxas $(RUN_DIR)/_cuobjdump_list_ptx_*
	cp $(TARGET) $(RUN_DIR)/$(TARGET_EXE)
	cp $(CONFIG_DIR)/gpgpusim.config $(RUN_DIR)/
	cp $(CONFIG_DIR)/config_ampere_islip.icnt $(RUN_DIR)/

gem5-smoke: gem5-dir
	cd $(RUN_DIR) && env -u PTX_SIM_USE_PTX_FILE -u PTX_SIM_KERNELFILE ./$(TARGET_EXE) --blocks=1 --threads=256 --iters=1 --warmup=0 --tiles=1 --block-hot

run-hw: $(TARGET)
	$(TARGET) --blocks=528 --threads=256 --iters=4096 --warmup=256 --tiles=528 --repeat=1 --block-hot

clean:
	rm -f $(TARGET)
	rm -f $(RUN_DIR)/$(TARGET_EXE)
	rm -f *.ptx *.ptxas
